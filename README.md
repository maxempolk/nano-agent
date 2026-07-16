# LLM Agent

Автономный AI-агент с доступом к bash, CLI и Telegram-интерфейсом.

## Возможности

- Выполняет bash-команды для решения задач пошагово
- Отправляет сообщения, фото и документы в Telegram
- Два интерфейса: терминал и Telegram-бот

## Установка

```bash
cp .env.example .env   # заполни токены
```

Зависимости устанавливаются через `uv` или `pip`:
```bash
pip install openai python-dotenv httpx
```

## Настройка `.env`

```env
API_TOKEN=          # Groq API key — console.groq.com
TELEGRAM_BOT_TOKEN= # токен бота от @BotFather (опционально)
ALLOWED_USER_ID=    # твой Telegram user_id (@userinfobot покажет)
PROMPT_PROFILE=     # full или mini (опционально)
```

## Запуск

```bash
python main.py --cli                    # Groq + full-промпты
python main.py --telegram               # Telegram-бот через Groq
python main.py --cli --local            # Apple FM system + mini-промпты
python main.py --telegram --local       # локальная модель в Telegram
python main.py --cli --prompts mini     # явно выбрать набор промптов
```

Профиль выбирается автоматически: `mini` для `--local`, `full` для API.
Переопределить его можно через `--prompts full|mini` или переменную
`PROMPT_PROFILE`.

Наборы находятся в `core/prompts.py`. Каждый профиль содержит отдельные части
для обычного агента, Telegram, планировщика и cron-исполнителя. Чтобы добавить
новый набор, создай `PromptProfile` и зарегистрируй его в `PROFILES`.

## Структура

```
├── main.py               # точка входа
├── core/agent.py         # логика агента (LLM + tool calls)
├── core/prompts.py       # сменные наборы системных промптов
├── interfaces/cli.py     # терминальный интерфейс
├── interfaces/telegram.py# Telegram-интерфейс
```

## Модель

API- и локальные модели задаются в `core/config.py`. Локальный режим использует
Apple Foundation Model `system` через `http://127.0.0.1:1976/v1`.
