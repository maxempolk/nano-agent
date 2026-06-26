# Project Rules

## Git workflow
- Remind the user to commit after every significant feature or fix. Do NOT commit automatically.
- Wait for the user to explicitly say "сделай коммит" before committing.

## Token Economy (top priority)
- Keep system prompts as short as possible. Every word costs tokens.
- Skills injected into the system prompt must be dense — no examples, no repeating what the model already knows, no filler.
- `MAX_TOOL_OUTPUT` caps tool responses to avoid wasting tokens on huge bash output.
- `CONTEXT_WINDOW` limits how many messages are sent to the API per request.

## Architecture
```
llm-agent/
├── main.py                  # entry point: config, Agent init, interface dispatch
├── core/
│   └── agent.py             # Agent class — LLM loop, tool calls, message history
├── interfaces/
│   ├── cli.py               # stdin/stdout loop
│   └── telegram.py          # Telegram long-polling, user_id filter
├── .env
└── CLAUDE.md
```

- `Agent` in `core/agent.py` is interface-agnostic: accepts `user_input`, returns `reply`.
- Each interface creates its own `Agent` instance with its own message history.
- Skills (e.g. Telegram send) are string variables injected into `SYSTEM` at startup, only when the relevant env var is set.
- All secrets come from `.env` via `python-dotenv`. Never hardcode tokens.

## Running
```bash
python main.py --cli       # terminal
python main.py --telegram  # Telegram bot (long polling)
```

## Env vars
- `API_TOKEN` — Groq API key
- `TELEGRAM_BOT_TOKEN` — Telegram bot token (optional; enables Telegram interface + skill)
- `ALLOWED_USER_ID` — only Telegram user_id allowed to interact with the bot (user_id == chat_id in private chats)
