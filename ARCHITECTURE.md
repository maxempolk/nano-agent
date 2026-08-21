# Архитектура Nano Agent

## Обзор

Компактный local-first агент для Apple Intelligence: один agent loop,
маршрутизация между локальной AFM и Apple PCC, небольшой набор инструментов
(web search, безопасный bash, cron) и строгие ограничения выполнения
(бюджет, отмена, политика). Интерфейсы: CLI и Telegram.

## Структура проекта

```
nano-agent/
├── main.py                  # Точка входа: валидация конфигурации, сборка
│                            # политики/реестра/агента, диспетчеризация интерфейсов
├── core/
│   ├── agent.py             # Agent: единый цикл, бюджет, отмена, события
│   ├── config.py            # Типизированная конфигурация (Pydantic) + валидация
│   ├── budget.py            # RunBudget: лимиты одного пользовательского запроса
│   ├── cancellation.py      # CancellationToken: сквозная отмена выполнения
│   ├── events.py            # Типизированные progress-события + EventBus
│   ├── policy.py            # ExecutionPolicy: capabilities, workspace, approval
│   ├── llm.py               # Обёртка над OpenAI-совместимым API
│   ├── logger.py            # SessionLogger (файловые логи сессий)
│   ├── model_router.py      # Маршрутизация local/PCC (hybrid)
│   ├── prompts.py           # Профили промптов (mini, full)
│   ├── cron_runner.py       # APScheduler: отложенные задачи, отмена, stop()
│   └── tools/
│       ├── base.py          # Единый протокол: Tool, ToolResult, ToolRegistry
│       ├── web_search.py    # Evidence-driven поиск + адаптер WebSearchToolSpec
│       ├── bash.py          # BashTool: безопасное выполнение команд
│       └── cron.py          # CronManageTool: управление расписанием
├── interfaces/
│   ├── cli.py               # CLI: события, Ctrl+C-отмена
│   └── telegram.py          # Telegram: события, /cancel, worker-тред
├── tests/                   # Unit-тесты (287)
├── benchmarks/              # Оценка локальных моделей на агентных задачах
├── logs/                    # Логи сессий
└── .env                     # Переменные окружения (не в git)
```

## Единый протокол инструментов (`core/tools/base.py`)

Каждый инструмент — класс `Tool` с типизированным контрактом:

- `name`, `description` — идентификатор и описание для модели;
- `input_model` — Pydantic-схема аргументов (из неё генерируется
  OpenAI function schema);
- `capabilities` — требуемые разрешения (`core.policy.Capability`);
- `timeout`, `output_limit` — ограничения;
- `execute(args, ctx) -> ToolResult`.

`ToolResult` разделяет:

- `content` — содержимое для модели;
- `summary` — краткое описание для интерфейсов;
- `error` + `error_code` + `retryable` — структурированные ошибки;
- `warnings`, `files_created`, `structured`, `meta` — дополнения.

`ToolRegistry` — единственный источник инструментов для агента:

- аргументы валидируются Pydantic-схемой ДО выполнения; невалидный вызов
  не исполняется;
- политика проверяется ДО выполнения;
- таймаут инструмента контролируется реестром (backstop), отмена
  пробрасывается через `ToolContext`;
- модель может выполнить только инструмент, переданный в текущем
  LLM-запросе (агент отслеживает `allowed_names` на каждый запрос).

## Agent (`core/agent.py`)

Один цикл на запрос. Гарантии:

- **Бюджет**: `RunBudget` ограничивает шаги, вызовы модели, вызовы
  инструментов, суммарное время, размер tool-вывода, число ошибок подряд
  и повторы идентичного вызова. Превышение возвращает честное сообщение
  («Остановлено: … запрос не завершён»), а не правдоподобный ответ.
- **Отмена**: `CancellationToken` на каждый запуск; `agent.cancel()`
  прерывает цикл, инструменты и финализацию; история сообщений
  балансируется (незакрытые tool_calls получают ответ), чтобы контекст
  оставался валидным для API.
- **События**: `EventBus` рассылает `run_started`, `route_selected`,
  `model_started/completed`, `tool_started/completed`,
  `context_compacted`, `run_completed/failed/cancelled`. CLI и Telegram
  используют один API; события маленькие и не содержат секретов.
- **Финализация**: ответ никогда не пустой; JSON и tool-вызовы не
  выдаются за ответ; частичное исследование явно называется частичным;
  язык ответа соответствует языку вопроса.

**Контекст:** последние сообщения + summary старой истории
(`memory`), сжатие при 80% бюджета, уменьшение старых tool-результатов.
`/clear`, `/context`, `/compact`.

## Безопасный bash (`core/tools/bash.py`)

- Явная рабочая директория (`BASH_WORKSPACE`), в ней же запускается
  процесс (`cwd=workspace`).
- Запись/удаление только внутри workspace (или scratch `/tmp`,
  `/var/folders`); статический анализ путей в команде.
- Таймаут с убийством всей группы процессов (`start_new_session` +
  `killpg`); отмена тоже убивает дочерние процессы.
- Фильтрация секретов из окружения (`*TOKEN*`, `*SECRET*`,
  `TELEGRAM_BOT_TOKEN`, …).
- Запрет интерактивных программ (vim, top, ssh, «голый» python/bash …)
  и фоновых процессов (`&`, `nohup`, `disown`).
- Ограничение stdout/stderr (`BASH_OUTPUT_LIMIT`).
- Классификация команд: `read_only` / `mutating` / `destructive` /
  `interactive` / `blocked`; деструктивные команды по умолчанию
  отклоняются (`ALLOW_DESTRUCTIVE=false`) либо требуют подтверждения
  (`BASH_APPROVAL=prompt`).
- Отправка данных наружу (curl POST и т.п.) контролируется политикой;
  при включённом Telegram разрешён только `api.telegram.org`.
- Structured errors с кодами (`denied`, `timeout`, `command_failed`, …).

## Политика (`core/policy.py`)

Компактный слой, независимый от модели. Capabilities:
`filesystem.read/write`, `shell.read/write`, `network.read`,
`external.send`, `scheduler.write`, `destructive`.

Политика по умолчанию: чтение разрешено; изменения — только внутри
workspace; сетевое чтение разрешено; внешняя отправка контролируется;
деструктивные действия отклоняются или требуют подтверждения.

## Бюджет запроса (`core/budget.py`)

| Лимит | Переменная | По умолчанию |
|---|---|---|
| Шаги агента | `RUN_MAX_STEPS` | 8 |
| Вызовы модели | `RUN_MAX_MODEL_CALLS` | 12 |
| Вызовы инструментов | `RUN_MAX_TOOL_CALLS` | 12 |
| Время выполнения | `RUN_MAX_SECONDS` | 180 |
| Размер tool-вывода | `RUN_MAX_TOOL_OUTPUT_CHARS` | 2000 |
| Ошибки подряд | `RUN_MAX_CONSECUTIVE_ERRORS` | 3 |
| Повторы идентичного вызова | `RUN_MAX_IDENTICAL_CALLS` | 2 |

Web search сохраняет собственные режимные бюджеты (quick/normal/deep);
RunBudget ограничивает весь ход сверху.

## Web-исследование (`core/tools/web_search.py`)

Режимы `quick` (сниппеты, 0 LLM-вызовов), `normal` (до 2 источников,
3 вызова), `deep` (до 5 источников, один цикл верификации). Аспект-ориентированный
план, ранжирование источников, извлечение фактов локальной моделью,
верификация PCC, детерминированный fallback. Отмена пробрасывается в
`SearchBudget` через `CancellationToken`.

## Маршрутизация (`core/model_router.py`)

`hybrid` — оценка сложности (длина, код, многошаговость, ключевые слова);
`local`/`pcc` — принудительные режимы. Принудительный режим не считается
«простым запросом»: инструменты и полный системный промпт остаются
доступными (`automatic=False`).

## Конфигурация (`core/config.py`)

`load_config()` собирает ВСЕ проблемы (неверные enum, диапазоны,
отсутствующие credentials, несовместимые alias'ы моделей) и сообщает их
одним списком при старте. См. `.env.example`.

## Интерфейсы

**CLI** — события печатаются как прогресс; первый Ctrl+C отменяет
текущий запрос, второй — выход.

**Telegram** — каждый запрос выполняется в worker-треде, поэтому polling
остаётся отзывчивым: `/cancel` отменяет активный запуск, повторный запрос
во время выполнения получает busy-сообщение. Прогресс — компактный список
завершённых инструментов; полный tool-trace — в финальном сообщении.

## Cron (`core/cron_runner.py`)

Задачи хранятся в `jobs.json` (путь настраивается). Одноразовая задача
удаляется из файла ДО выполнения. Активные задачи отслеживаются;
`cancel_job(name)` отменяет выполнение; `stop()` отменяет всё и
останавливает планировщик. `max_instances=1` защищает от дублей.

## Расширение

Новый инструмент:

```python
class MyInput(BaseModel):
    param: str

class MyTool(Tool):
    name = "my_tool"
    description = "..."
    input_model = MyInput
    capabilities = frozenset({Capability.FILESYSTEM_READ})
    timeout = 10.0

    def execute(self, args: MyInput, ctx: ToolContext) -> ToolResult:
        return ToolResult(content="result", summary="готово")
```

и регистрация в `main.py`: `ToolRegistry([bash_tool, web_spec, cron_tool, MyTool()])`.

Новый интерфейс: подписаться на `agent.events`, вызывать
`agent.run_turn(text)`, при необходимости `agent.cancel()`.
