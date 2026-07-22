# Project Rules

## Git workflow
- Remind the user to commit after every significant feature or fix.
- Never commit automatically. Wait for an explicit "сделай коммит".
- Keep unrelated user changes and `AGENTS.md` out of a commit unless explicitly requested.

## General development rules
- Build generic mechanisms for broad task classes. Do not tune behavior for one example, country, product, query, or website.
- Topic/domain hardcodes may only be defensive hints; they must not be the primary quality mechanism.
- Prefer explicit state, schemas, budgets, and validators over prompt-only behavior.
- Never hide a failure behind a plausible answer. Missing evidence must remain visible.
- After a code change that affects the running bot, run tests and restart it.

## Token and resource economy
- Keep system prompts dense and short. Do not repeat instructions the model already knows.
- Cap tool output and context before every model request.
- Never start recursive research or silently repeat a completed search.
- Execute independent web queries and page fetches in parallel, but keep model-call concurrency bounded.
- Search-mode budgets are invariants:
  - `quick`: snippets only, 0 LLM calls;
  - `normal`: at most 2 pages and 3 LLM calls;
  - `deep`: one research cycle, at most 8 LLM calls and one replacement extraction.

## Architecture
```
llm-agent/
├── main.py
├── core/
│   ├── agent.py
│   ├── model_router.py
│   ├── prompts.py
│   └── tools/web_search.py
├── interfaces/
│   ├── cli.py
│   └── telegram.py
├── tests/
├── .env
└── AGENTS.md
```

- `Agent` is interface-agnostic: it accepts user text and returns final text.
- Each interface owns an `Agent` instance and its conversation history.
- Apple hybrid mode uses local AFM Core 3 for simple work and PCC for complex work.
- No model may execute a tool that was not offered in that exact LLM request.
- One user turn may execute `web_search` only once. A returned ghost/repeated tool call is a protocol violation and is never executed.
- All secrets come from `.env`; never hardcode or log credentials.

## Prompt profiles
- Prompt sets are replaceable profiles (`mini`, `full`), independent from model selection.
- Local compact models use the mini profile; PCC uses the full profile unless explicitly overridden.
- Prompts define policy, not topic-specific solutions.
- A tool error should cause a corrected attempt, not an identical retry. Stop after bounded failure.

## Web-search design
- `quick` and `normal` must remain fast and stable. Deep-research improvements must not expand their budgets.
- Deep research is aspect-driven:
  1. derive arbitrary aspects from the user question;
  2. define required evidence for each aspect;
  3. bind each query/result/source/fact to its aspect;
  4. assess relevance and freshness relative to that aspect;
  5. track `confirmed`, `missing`, or `rejected` with a reason;
  6. allow at most one bounded replacement;
  7. synthesize only verified evidence.
- In hybrid deep search, AFM is a candidate extractor only. Its compact output is limited to
  claim, supporting excerpt, and explicit date; it never confirms aspects or conflicts.
- PCC performs one batched verification of all AFM candidates, aspect coverage, comparison
  metadata, and conflicts before synthesis. Unverified or malformed AFM output is never
  promoted directly into the final evidence set.
- Aspects such as income, safety, camera, API compatibility, or legal exceptions are examples only and must never be fixed fields.
- Extraction must return claims with supporting page excerpts and dates. A claim without supporting evidence is not verified.
- Reject pseudo-facts that only state that information is absent or the page is irrelevant.
- An official/preferred domain suggested by a model is not automatically authoritative.
- Compare facts only when metric, units, period, geography, and definition are compatible. Different dates are not automatically conflicts.
- If required aspects remain uncovered, either use the single permitted replacement or return an explicitly partial result. Do not invent a complete conclusion.
- Structured search state is the source of truth; formatted tool text is presentation only.
- PDF pages should use a PDF text extractor before HTML fallbacks.

## Finalization
- Research results use a dedicated finalizer prompt, never the main agent prompt containing `MUST web_search`.
- Finalization order is selected model, configured fallback model, then deterministic renderer.
- A valid final answer must be non-empty, in the user's language, normal prose/Markdown, and must not be JSON, a code fence, or a function call.
- The finalizer may use only verified evidence, conflicts, sources, and coverage gaps supplied by the research result.
- A broad conclusion is allowed only when supported by sufficient aspect coverage.
- If coverage is partial, the answer must say so and identify missing aspects.

## Telegram interface
- Show a compact accumulating list of completed tools while work is running. Do not show large quotes or raw tool results in progress messages.
- Put the compact expandable tool trace only in the final response.
- Convert model Markdown to safe Telegram HTML; escape raw HTML and secrets.
- Never deliver an empty final message.
- Log successful and failed `sendMessage`/`editMessageText` operations with message IDs, never credentials.

## Context management
- `/clear` removes history and compact memory.
- `/context` reports estimated usage as `used/limit tokens`.
- `/compact` summarizes older context while retaining current work, decisions, errors, and pending tasks.
- Automatic compaction runs before the active model's context limit and must shrink old tool output first when appropriate.

## Testing requirements
- Every significant agent/search change needs regression tests before restart.
- Deep-research tests must cover unrelated task classes, not only the motivating example: country research, product comparison, software/library analysis, law/policy analysis, scientific positions, and missing/conflicting evidence.
- Assert mode budgets, one-search-per-turn, tool authorization, aspect binding, evidence rejection, conflict compatibility, finalizer fallback, non-JSON output, language, Telegram rendering, and delivery logging.
- Run the complete suite with `python -m unittest discover -s tests`.

## Running
```bash
python main.py --cli
python main.py --telegram
python main.py --telegram --local
python main.py --telegram --server
```

- No model flag means hybrid mode.
- `--local` forbids PCC usage.
- `--server` uses PCC only.
