# Project Rules

## Git workflow
- Remind the user to commit after every significant feature or fix. Do NOT commit automatically.
- Wait for the user to explicitly say "сделай коммит" before committing.

## Environment

- Python dependencies are defined in `pyproject.toml` and installed with
  `uv sync --all-groups`.
- Run the test suite with `uv run python -m unittest discover -s tests`.
- Lint with `uv run ruff check .`.

## Token Economy (top priority)
- Keep system prompts as short as possible. Every word costs tokens.
- Prompt profiles (`mini`, `full`) are dense by design — no filler.
- `max_tool_output` (run budget) caps tool responses to avoid wasting tokens
  on huge outputs.
- Context is compact-compacted before the active model's budget is exceeded.

## Architecture
```
nano-agent/
├── main.py                  # entry point: typed config, policy, registry, interface dispatch
├── core/
│   ├── agent.py             # Agent — single loop: budget, cancellation, events
│   ├── budget.py            # RunBudget per user request
│   ├── cancellation.py      # CancellationToken
│   ├── events.py            # typed progress events
│   ├── policy.py            # capability policy (workspace, external send, destructive)
│   ├── config.py            # typed configuration with startup validation
│   ├── llm.py               # OpenAI-compatible call wrapper
│   ├── model_router.py      # hybrid/local/pcc routing
│   ├── prompts.py           # prompt profiles
│   ├── cron_runner.py       # APScheduler runner with cancel/stop
│   └── tools/
│       ├── base.py          # Tool protocol: Tool, ToolResult, ToolRegistry
│       ├── bash.py          # safe workspace-pinned bash
│       ├── web_search.py    # evidence-driven search + protocol adapter
│       └── cron.py          # cron_manage tool
├── interfaces/
│   ├── cli.py               # terminal: events, Ctrl+C cancellation
│   └── telegram.py          # long-polling: events, /cancel, worker thread
├── tests/
├── benchmarks/
├── .env
└── CLAUDE.md
```

- `Agent` in `core/agent.py` is interface-agnostic: accepts `user_input`,
  returns `reply`; progress is delivered through `agent.events`.
- Tools are registered once in a `ToolRegistry`; arguments are validated
  before execution; only tools offered in a request can be executed.
- All secrets come from `.env` via `python-dotenv`. Never hardcode tokens;
  never write them into logs or events.

## Running

```bash
uv run python main.py --cli        # terminal
uv run python main.py --telegram   # Telegram bot (long polling)
```

Flags: `--local` (AFM only), `--server`/`--model pcc` (PCC only),
`--model hybrid` (default), `--prompts mini|full`.

## Env vars

See `.env.example` for the full annotated list. Highlights:
- `LLM_BASE_URL` — OpenAI-compatible local bridge URL
- `LOCAL_MODEL` / `PCC_MODEL` — local and PCC model aliases
- `MODEL_MODE` — `hybrid`, `local` or `pcc`
- `TELEGRAM_BOT_TOKEN` / `ALLOWED_USER_ID` — Telegram interface (both required together)
- `BASH_WORKSPACE`, `BASH_TIMEOUT`, `BASH_APPROVAL`, `ALLOW_DESTRUCTIVE` — safe bash
- `RUN_MAX_*` — per-request run budget
