# Архитектура проекта llm-agent

## Обзор

LLM-агент с поддержкой нескольких интерфейсов (CLI, Telegram), инструментов (web search, bash, cron), и маршрутизацией между локальной Apple AFM и облачной Apple PCC.

## Структура проекта

```
llm-agent/
├── main.py                  # Точка входа: конфигурация, инициализация Agent, диспетчеризация интерфейсов
├── core/
│   ├── agent.py             # Agent class — LLM loop, вызовы инструментов, история сообщений
│   ├── config.py            # Конфигурация моделей и URL из переменных окружения
│   ├── llm.py               # Обёртка над OpenAI API для вызова LLM
│   ├── logger.py            # SessionLogger для логирования сессий в файлы
│   ├── model_router.py      # Маршрутизация между локальной и PCC моделями
│   ├── prompts.py           # Системные промпты (FULL, MINI профили)
│   ├── cron_runner.py       # APScheduler для выполнения отложенных задач
│   └── tools/
│       ├── web_search.py    # WebSearchTool — поиск в DuckDuckGo + извлечение контента через Crawl4AI
│       ├── bash.py          # execute_bash — выполнение shell-команд
│       └── cron.py          # cron_manage — управление отложенными задачами
├── interfaces/
│   ├── cli.py               # Терминальный интерфейс (stdin/stdout)
│   └── telegram.py          # Telegram long-polling интерфейс
├── tests/                   # Unit-тесты
├── benchmarks/              # Оценка качества моделей
├── logs/                    # Логи сессий
└── .env                     # Переменные окружения (не в git)
```

## Ключевые компоненты

### Agent (`core/agent.py`)

Центральный класс, интерфейсно-независимый. Принимает `user_input`, возвращает `reply`.

**Основные методы:**
- `run_turn(user_input, on_tool_call=None)` — основной цикл обработки запроса
- `_select_route(user_input)` — выбор модели через роутер
- `_compact_if_needed()` — сжатие контекста при превышении лимита
- `_finalize_research()` — финальная генерация ответа после поиска

**Управление контекстом:**
- `token_budget` — максимальный размер контекста
- `compact_trigger_ratio` — порог срабатывания сжатия (0.8 = 80%)
- `_shrink_tool_results()` — сжатие старых tool-результатов
- `memory` — сжатая история предыдущих сообщений

**Инструменты:**
- `tools` — список схем инструментов (OpenAI function calling format)
- `handlers` — словарь `{имя: функция}` для вызова инструментов
- `tool_objects` — словарь `{имя: объект инструмента}` для доступа к состоянию

### WebSearchTool (`core/tools/web_search.py`)

Многоуровневая система веб-поиска с верификацией фактов.

**Режимы поиска:**
- `quick` — быстрый поиск по сниппетам DuckDuckGo (0 LLM-вызовов)
- `normal` — поиск + извлечение контента со 2 источников (3 LLM-вызова)
- `deep` — глубокий анализ до 5 источников (8 LLM-вызовов)

**Архитектура:**
1. **Планирование** (`_plan_research`) — LLM определяет аспекты и запросы
2. **Поиск** (`_search`, `_search_many`) — DuckDuckGo через библиотеку `ddgs`
3. **Ранжирование** (`_rank_results`) — оценка релевантности результатов
4. **Извлечение** (`_scrape`) — загрузка и парсинг контента через Crawl4AI
5. **Верификация** (`_extract_normal_page`, `_synthesize_deep`) — LLM проверяет факты
6. **Финализация** (`_finalize_research` в Agent) — итоговый ответ на основе evidence

**Извлечение контента:**
- PDF: `pdftotext` (системная утилита)
- HTML: Crawl4AI (async, headless Chromium)
- Fallback: текст сниппета из поиска

**Бюджет и ограничения:**
- `SearchBudget` — контроль LLM-вызовов и таймаутов
- `MAX_RESULTS` — лимит результатов поиска (10)
- `PAGE_CONTEXT_CHARS` — лимит символов на страницу (7000)

### Model Router (`core/model_router.py`)

Маршрутизация запросов между локальной AFM и облачной PCC.

**Режимы:**
- `local` — только локальная модель (AFM Core 3)
- `pcc` — только облачная модель (Apple PCC)
- `hybrid` — автоматический выбор по сложности запроса

**Критерии выбора PCC:**
- Длина запроса (>450 символов)
- Наличие блоков кода
- Ключевые слова (реализация, анализ, рефакторинг и т.д.)
- Многошаговые инструкции

### Интерфейсы

**CLI (`interfaces/cli.py`):**
- Простой цикл `input() -> agent.run_turn() -> print()`
- Вывод tool calls и результатов

**Telegram (`interfaces/telegram.py`):**
- Long-polling через `getUpdates`
- Фильтрация по `ALLOWED_USER_ID`
- Прогресс-сообщения с обновлением
- Поддержка команд: `/clear`, `/context`, `/compact`
- Markdown -> HTML конвертация

### Cron Runner (`core/cron_runner.py`)

Планировщик отложенных задач на APScheduler.

**Типы задач:**
- `once` — одноразовые (`run_at` или `run_in`)
- `cron` — повторяющиеся (cron-выражение)

**Механизм:**
- `jobs.json` — персистентное хранение задач
- `_reload_jobs()` — синхронизация файла с планировщиком каждые 30 сек
- Результат доставляется в Telegram

## Поток данных

### Простой запрос

```
User Input -> Agent.run_turn()
           -> route_selector() -> выбор модели
           -> LLM вызов (с tools)
           -> execute_bash или прямой ответ
           -> Response
```

### Запрос с поиском

```
User Input (с "загугли" или changing fact)
         -> Agent.run_turn()
         -> _forced_web_search_query() -> определение запроса
         -> web_search.execute()
            -> _plan_research() -> LLM планирует
            -> _search() -> DuckDuckGo
            -> _rank_results() -> ранжирование
            -> _scrape() -> Crawl4AI извлекает контент
            -> _extract_normal_page() -> LLM извлекает факты
            -> _synthesize_deep() -> LLM верифицирует
         -> _finalize_research() -> итоговый ответ
         -> Response
```

### Отложенная задача

```
User: "напомни через час"
    -> cron_manage(action="add", run_in=3600)
    -> jobs.json обновлён
    -> CronRunner._reload_jobs() добавляет в APScheduler
    -> (через час) _run_job()
        -> agent_factory() -> новый Agent
        -> agent.run_turn(prompt)
        -> _send_telegram() -> результат в чат
```

## Конфигурация

### Переменные окружения (`.env`)

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `LLM_BASE_URL` | URL OpenAI-совместимого API | `http://127.0.0.1:1976/v1` |
| `LOCAL_MODEL` | Имя локальной модели | `system` |
| `PCC_MODEL` | Имя PCC модели | `pcc` |
| `TELEGRAM_BOT_TOKEN` | Токен Telegram бота | — |
| `ALLOWED_USER_ID` | Разрешённый user_id | — |
| `MODEL_MODE` | Режим маршрутизации | `hybrid` |
| `LOCAL_CONTEXT_TOKEN_BUDGET` | Лимит контекста local | `3000` |
| `PCC_CONTEXT_TOKEN_BUDGET` | Лимит контекста PCC | `12000` |
| `COMPACT_TRIGGER_RATIO` | Порог сжатия контекста | `0.8` |
| `WEB_SEARCH_FORCE_DEPTH` | Фиксированная глубина поиска | `auto` |
| `PROMPT_PROFILE` | Профиль промптов (full/mini) | — |

### CLI параметры

```bash
python main.py --cli           # терминальный интерфейс
python main.py --telegram      # Telegram бот
python main.py --model hybrid  # режим маршрутизации
python main.py --local         # только локальная модель
python main.py --server        # только PCC
python main.py --prompts mini  # профиль промптов
```

## Зависимости

### Основные
- `openai` — OpenAI-совместимый клиент
- `ddgs` — DuckDuckGo search
- `crawl4ai` — извлечение контента из HTML (headless Chromium)
- `httpx` — HTTP клиент для Telegram API
- `apscheduler` — планировщик задач
- `pydantic` — валидация structured output
- `python-dotenv` — загрузка `.env`
- `tzlocal` — локальная таймзона

### Системные
- `pdftotext` — извлечение текста из PDF (опционально)

## Логирование

`SessionLogger` создаёт файл лога на каждую сессию в `logs/`.

**Формат:**
```
══════════════════════════════════════════════════════════════
SESSION  2026-07-21 14:30:00
══════════════════════════════════════════════════════════════
[14:30:05] USER
    Запрос пользователя
────────────────────────────────────────────────────────────
[14:30:06] TOOL CALL → web_search
    {"query": "...", "depth": "auto"}
[14:30:10] TOOL RESULT
    Результат инструмента
────────────────────────────────────────────────────────────
[14:30:15] AGENT
    Ответ агента
══════════════════════════════════════════════════════════════
```

## Расширение

### Добавление нового инструмента

1. Создать файл в `core/tools/` с `SCHEMA` и `execute()`:
```python
SCHEMA = {
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "...",
        "parameters": {...}
    }
}

def execute(param1: str, param2: int = 0) -> str:
    return "result"
```

2. Добавить в `main.py`:
```python
from core.tools import my_tool
agent = _make_agent(logger, extra_tools=[web_search, cron_wrapper, my_tool])
```

### Добавление нового интерфейса

Создать модуль с функцией `run(agent)`:
```python
from core.agent import Agent

def run(agent: Agent) -> None:
    while True:
        user_input = get_input()
        reply = agent.run_turn(user_input)
        send_output(reply)
```

## Особенности реализации

### Token Economy

- Системные промпты минимальны (каждое слово стоит токенов)
- `MAX_TOOL_OUTPUT` ограничивает размер tool-результатов (2000 символов)
- `COMPRESSED_TOOL_CHARS` сжимает старые tool-результаты (400 символов)
- `compact()` сжимает историю в `memory` при превышении лимита

### Fallback механизмы

- Если LLM вернул невалидный JSON -> повторная попытка с напоминанием
- Если tool call невалиден -> сообщение об ошибке в контекст
- Если поиск не дал результатов -> `insufficient_information` flag
- Если PCC недоступен -> fallback на local модель

### Безопасность

- Секреты только из `.env`, никогда в коде
- `ALLOWED_USER_ID` фильтрует Telegram сообщения
- Деструктивные команды требуют подтверждения
- Tool output считается untrusted data
