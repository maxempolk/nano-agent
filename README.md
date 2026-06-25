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
```

## Запуск

```bash
python main.py --cli        # диалог в терминале
python main.py --telegram   # Telegram-бот (long polling)
```

## Структура

```
├── main.py               # точка входа
├── core/agent.py         # логика агента (LLM + tool calls)
├── interfaces/cli.py     # терминальный интерфейс
├── interfaces/telegram.py# Telegram-интерфейс
```

## Модель

По умолчанию используется `meta-llama/llama-4-scout-17b-16e-instruct` через [Groq](https://groq.com). Модель меняется в `main.py` в переменной `MODEL`.
