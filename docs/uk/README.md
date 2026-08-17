# Nano Agent

[English documentation](../../README.md)

Nano Agent — локальний AI-агент з перевіркою фактів для Apple Intelligence.
Він має CLI та Telegram-інтерфейси, використовує локальну AFM Core 3 для простих
запитів і спрямовує складніші завдання до Apple Private Cloud Compute (PCC).

## Основні можливості

- Виконує bash-команди для покрокового розв'язання задач.
- Має CLI та Telegram-інтерфейси, зокрема для повідомлень, фото й документів.
- Показує виклики інструментів, аргументи та результати поруч із відповіддю.
- Зберігає контекст діалогу й підтримує `/clear`, `/context` та `/compact`.
- Виконує вебдослідження з URL-джерелами та перевіреними Pydantic-структурами.
- Маршрутизує запити між локальною моделлю та PCC за складністю завдання.

## Вимоги та встановлення

Потрібні Python 3.13+ і [uv](https://docs.astral.sh/uv/). Файл `uv.lock`
фіксує повний набір залежностей для відтворюваного встановлення.

```bash
cp .env.example .env
uv sync --all-groups
```

## Налаштування

```env
TELEGRAM_BOT_TOKEN=                 # Необов'язковий токен від @BotFather
ALLOWED_USER_ID=                    # Необов'язковий список дозволених Telegram ID
MODEL_MODE=hybrid                   # hybrid, local або pcc; auto/server — aliases
PROMPT_PROFILE=                     # Необов'язковий профіль: full або mini
LOCAL_CONTEXT_TOKEN_BUDGET=         # Типово: 3000
PCC_CONTEXT_TOKEN_BUDGET=           # Типово: 12000
COMPACT_TRIGGER_RATIO=              # Типово: 0.8
WEB_SEARCH_FORCE_DEPTH=auto         # auto, quick, normal або deep
```

## Запуск

```bash
uv run python main.py --cli                    # Гібридний режим
uv run python main.py --telegram               # Гібридний режим у Telegram
uv run python main.py --cli --local            # Лише локальна AFM Core 3
uv run python main.py --cli --server           # Лише Apple PCC
uv run python main.py --cli --model local      # Еквівалент --local
uv run python main.py --cli --model pcc        # Еквівалент --server
uv run python main.py --cli --prompts mini     # Компактний профіль промптів
```

## Перевірки якості

```bash
uv run ruff check .
uv run python -m unittest discover -s tests
```

## Маршрутизація моделей і промпти

У гібридному режимі маршрутизатор оцінює довжину запиту, наявність коду,
багатокроковість та ознаки розробки або аналізу. Локальна модель `system`
використовує профіль `mini`, а PCC — `full`; історія діалогу спільна.

`--local` вимикає PCC разом із planner і fallback, а `--server` вимикає
локальну AFM. Значення `auto` і `server` залишені як aliases для `hybrid` і
`pcc`. Профілі промптів визначені в `core/prompts.py`.

## Стиснення контексту

Агент дослівно зберігає останні повідомлення. Коли використано 80% активного
ліміту контексту, він замінює стару завершену частину семантичним резюме та
залишає десять останніх записів і поточний хід. Якщо резюмування не вдається,
використовується детерміноване скорочення. `/clear` видаляє резюме й свіжу
історію, але не змінює логи на диску.

## Вебдослідження

`web_search` підтримує режими `auto`, `quick`, `normal` і `deep`.

- **quick** повертає перевірені сніпети DuckDuckGo та URL без завантаження
  сторінок і внутрішніх LLM-викликів.
- **normal** створює короткий структурований план, обирає до двох джерел і
  отримує по одному результату з кожного. Ліміт — три LLM-виклики.
- **deep** запускається лише для явно запитаного дослідження: до п'яти запитів,
  до п'яти сторінок паралельно та фінальний синтез доказів. Ліміт — сім викликів
  і 90 секунд.

Агент надає пріоритет відомим офіційним доменам і перевіряє релевантність,
свіжість, очікувані значення та авторитетність джерела. У hybrid планування й
синтез виконує PCC, а витягування зі сторінок — AFM.

## Оцінювання локальних моделей

`benchmarks/agent_model_eval.py` перевіряє OpenAI-сумісну модель на 45 кейсах:
маршрутизація, виклики інструментів, витягування доказів, фіналізація,
відновлення після помилок і стиснення контексту.

```bash
# Apple Foundation Models
uv run python -m benchmarks.agent_model_eval --provider fm --model system

# LM Studio або інший OpenAI-сумісний сервер
uv run python -m benchmarks.agent_model_eval \
  --base-url http://127.0.0.1:1234/v1 \
  --model granite-4.0-h-tiny
```

Сирі JSONL-відповіді та зведений звіт записуються в `benchmark-results/`, який
виключено з Git. Звіт містить якість, p50/p95 затримки, використання токенів,
API-помилки й schema fallbacks.

## Структура

```text
main.py                    # Точка входу застосунку
core/agent.py              # Оркестрація агента та виклики інструментів
core/prompts.py            # Налаштовувані профілі системних промптів
interfaces/cli.py          # Інтерфейс командного рядка
interfaces/telegram.py     # Telegram-інтерфейс
benchmarks/                # Набір для оцінювання локальних моделей
```

## Провайдер моделей

Конфігурація моделей розміщена в `core/config.py`. Типовий локальний Apple
bridge — `http://127.0.0.1:1976/v1`: `system` означає AFM Core 3 на пристрої, а
`pcc` — Apple Private Cloud Compute. Для цієї конфігурації зовнішній API-ключ
моделі не потрібний.
