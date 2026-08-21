# Nano Agent

[Українська версія документації](docs/uk/README.md)

Nano Agent is a local-first, evidence-driven AI agent for Apple Intelligence.
It provides CLI and Telegram interfaces, uses local AFM Core 3 for lightweight
requests, and routes more demanding work to Apple Private Cloud Compute (PCC).

## Highlights

- Single typed tool protocol: Pydantic input schemas, structured results,
  capability checks; arguments are validated before anything executes.
- Safe bash pinned to a workspace: timeouts with process-group kill, secret
  env filtering, no interactive/background processes, destructive actions
  denied or approval-gated.
- Per-request run budget (steps, model calls, tool calls, wall time, output
  size, repeated errors, identical calls) with honest stop messages.
- Cancellation end to end: Ctrl+C in CLI, `/cancel` in Telegram; subprocesses
  and crawls are interrupted, nothing keeps running in the background.
- One typed progress-event stream shared by CLI and Telegram.
- Evidence-driven web research with source URLs and validated structures.
- Routes requests between local AFM and Apple PCC based on task complexity.
- Maintains conversation context and supports `/clear`, `/context` and `/compact`.

## Requirements and Installation

The project requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).
`uv.lock` pins the complete dependency set for reproducible installs.

```bash
cp .env.example .env
uv sync --all-groups
```

## Configuration

All settings are validated at startup; problems are reported together
before the agent starts. See `.env.example` for the full annotated list.

```env
TELEGRAM_BOT_TOKEN=                 # Optional token from @BotFather
ALLOWED_USER_ID=                    # Required when TELEGRAM_BOT_TOKEN is set
MODEL_MODE=hybrid                   # hybrid, local, pcc or cloud; auto/server are aliases
PROMPT_PROFILE=                     # Optional global override: full or mini
LOCAL_CONTEXT_TOKEN_BUDGET=         # Default: 3000
PCC_CONTEXT_TOKEN_BUDGET=           # Default: 12000
CLOUD_BASE_URL=                     # Required for MODEL_MODE=cloud
CLOUD_API_KEY=                      # Required for MODEL_MODE=cloud
CLOUD_MODEL=                        # Required for MODEL_MODE=cloud
STT_MODEL=whisper-large-v3-turbo    # Voice transcription; falls back to CLOUD_* provider
COMPACT_TRIGGER_RATIO=              # Default: 0.8
WEB_SEARCH_FORCE_DEPTH=auto         # auto, quick, normal or deep

BASH_WORKSPACE=.                    # bash commands run and write only here
BASH_APPROVAL=deny                  # deny or prompt (CLI confirmation)
ALLOW_DESTRUCTIVE=false             # allow rm/kill/etc. behind approval

RUN_MAX_STEPS=8                     # per-request run budget
RUN_MAX_MODEL_CALLS=12
RUN_MAX_TOOL_CALLS=12
RUN_MAX_SECONDS=180
```

## Run

```bash
uv run python main.py --cli                    # Hybrid mode
uv run python main.py --telegram               # Hybrid mode in Telegram
uv run python main.py --cli --local            # Local AFM Core 3 only
uv run python main.py --cli --server           # Apple PCC only
uv run python main.py --cli --cloud            # Cloud provider only (CLOUD_* env)
uv run python main.py --cli --model local      # Equivalent to --local
uv run python main.py --cli --model pcc        # Equivalent to --server
uv run python main.py --cli --prompts mini     # Use the compact prompt profile
```

## Quality Checks

```bash
uv run ruff check .
uv run python -m unittest discover -s tests
```

## Model Routing and Prompts

In hybrid mode, the router evaluates request length, code, multi-step work and
development or analysis signals. The local `system` model uses the `mini`
profile, while PCC uses `full`; both share the same conversation history.

`--local` disables PCC, including the planner and fallback. `--server` disables
the local AFM model. `--cloud` sends every stage (routing, search, finalization,
compaction) to one external OpenAI-compatible model configured via `CLOUD_BASE_URL`,
`CLOUD_API_KEY` and `CLOUD_MODEL`. Legacy `auto` and `server` values remain
supported as aliases for `hybrid` and `pcc`.

Prompt profiles are defined in `core/prompts.py`. Each profile has dedicated
instructions for the primary agent, Telegram, planner and cron runner.

## Context Compaction

The agent retains recent messages verbatim. At 80% of the active context budget,
it replaces an older completed section with a semantic summary and preserves the
last ten entries and current turn. If summarization fails, it uses a deterministic
fallback. `/clear` removes both the summary and recent history; on-disk logs are
not changed.

## Web Research

`web_search` supports `auto`, `quick`, `normal` and `deep` modes.

- **quick** returns validated DuckDuckGo snippets and URLs without loading pages
  or using internal LLM calls.
- **normal** creates a short structured plan, chooses up to two sources and
  extracts one result from each source. Its total budget is three LLM calls.
- **deep** is used only for explicitly requested research. It can plan up to five
  queries, process up to five pages in parallel and combine evidence in a final
  synthesis. Its total budget is seven calls and 90 seconds.

Known official domains are prioritized. Results are checked for relevance,
freshness, expected values and source authority. A low-confidence automatic quick
result can escalate to normal research; an explicitly selected quick search never
escalates.

In hybrid mode, planning and synthesis use PCC while page extraction uses AFM.
In local and PCC-only modes, every stage uses the selected provider. Input sizes,
call budgets and deadlines prevent a single request from consuming unlimited work.

## Local Model Evaluation

`benchmarks/agent_model_eval.py` evaluates an OpenAI-compatible model on agent
work rather than generic questions. Its 45 cases cover routing, tool calling,
evidence extraction, finalization, recovery and context compaction.

```bash
# Apple Foundation Models
uv run python -m benchmarks.agent_model_eval --provider fm --model system

# LM Studio or another OpenAI-compatible server
uv run python -m benchmarks.agent_model_eval \
  --base-url http://127.0.0.1:1234/v1 \
  --model granite-4.0-h-tiny

# Run selected suites or a small pilot
uv run python -m benchmarks.agent_model_eval --model system \
  --suite tools --suite extraction --repeat 3
uv run python -m benchmarks.agent_model_eval --model system --limit-per-suite 2
```

The runner writes raw JSONL responses and an aggregated report to
`benchmark-results/`, which is excluded from Git. The report includes quality,
p50/p95 latency, token use, API errors and schema fallbacks.

## Interfaces

- **CLI**: print progress events; the first Ctrl+C cancels the active
  request, a second Ctrl+C exits.
- **Telegram**: `/clear`, `/context`, `/compact` manage context and
  `/cancel` stops the active run. Each request runs in a worker thread, so
  the bot stays responsive while working. Voice messages are transcribed to
  text (Whisper via the `STT_*` / `CLOUD_*` provider) and handled like text.

## Structure

```text
main.py                    # Entry point: config validation and wiring
core/agent.py              # Single agent loop: budget, cancellation, events
core/tools/base.py         # Tool protocol: Tool, ToolResult, ToolRegistry
core/tools/bash.py         # Safe workspace-pinned bash
core/tools/web_search.py   # Evidence-driven web research
core/tools/cron.py         # Scheduled task management
core/budget.py             # Per-request run budget
core/cancellation.py       # Cancellation token
core/events.py             # Typed progress events
core/policy.py             # Capability policy layer
core/config.py             # Typed configuration with startup validation
core/model_router.py       # AFM/PCC routing
core/prompts.py            # Prompt profiles (mini, full)
core/cron_runner.py        # APScheduler runner with cancel/stop
interfaces/cli.py          # Command-line interface
interfaces/telegram.py     # Telegram interface
benchmarks/                # Local-model evaluation suite
```

## Model Provider

Model configuration lives in `core/config.py`. The default local Apple bridge is
`http://127.0.0.1:1976/v1`: `system` targets on-device AFM Core 3 and `pcc`
targets Apple Private Cloud Compute. No external model API key is required for
this configuration.
