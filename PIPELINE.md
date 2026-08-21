# Agent → Web Search Pipeline

## Overview

User sends a message. The agent decides whether to search the web. If search
happens, results go through a finalizer that writes the user-facing answer.
The pipeline has 5 stages: request guard, search trigger, search execution,
evidence formatting, finalization.

## Stage 0: Request Guard (budget, cancellation, validation)

Every turn runs under a `RunBudget` (steps, model calls, tool calls, wall
time, tool output size, consecutive errors, identical-call repeats) and a
`CancellationToken` (Ctrl+C in CLI, `/cancel` in Telegram). Tool arguments
are validated by Pydantic schemas before execution; invalid calls are never
executed. The model can only run tools that were offered in that exact LLM
request. Overruns and cancels produce honest stop messages.

## Stage 1: Search Trigger

Two paths decide whether web_search runs:

**Path A — Forced search (regex in agent.py).** Before the LLM sees the
message, regex patterns check if the query needs fresh web data. Patterns
include: changing facts (сейчас, последн, курс, погода, версия), explicit
search commands (загугли, поищи, найди, search), and deep-research requests
(подробно исследуй, сравни … источники, deep research). If matched,
web_search is executed through the registry immediately with the user's
query. The LLM does not participate in this decision. After forced search,
tools are removed from the turn so the LLM cannot call web_search again.

**Path B — LLM decides.** If no regex matched, the message goes to the LLM
with the registered tools available. The LLM chooses whether to call
web_search. A returned ghost or repeated search call is a protocol violation
and is never executed.

## Stage 2: Search Execution (web_search.py)

web_search.execute(query, depth) runs one of three modes. The search engine
has its own mode budget (`SearchBudget`); the run token is checked at every
budget checkpoint, so cancellation stops search too.

**Quick mode** (default for short factual queries):
1. DuckDuckGo search returns up to 10 results with title, URL, and snippet.
2. Ranking scores results: authority (official domains), term overlap with
   the query, presence of the expected value type, freshness, low-quality
   host penalties.
3. Top-5 ranked snippets are returned as-is. Zero LLM calls, 15s deadline.
   Quick mode never escalates on its own.

**Normal mode** (longer/analytical queries): budget 5 LLM calls, 45s deadline.
1. Planner generates 1-2 search queries and identifies the subject.
2. DuckDuckGo search for each planned query.
3. Top 2 pages are scraped with crawl4ai to get full text.
4. The local model extracts candidate facts from each page (claim + evidence
   excerpt + date).
5. Facts are filtered for relevance, freshness, and expected value match.
6. Output: "Web evidence" block with sources and extracted facts.

**Deep mode** (explicitly requested research only): budget 8 LLM calls, 100s
deadline.
1. Planner decomposes the question into aspects and generates up to 5 queries
   with evidence contracts (expected evidence, requirement, priority,
   freshness, acceptance criteria).
2. Up to 5 sources are selected across aspects, prioritizing official domains
   and aspect coverage; unreadable sources get at most two bounded
   replacements.
3. Pages are scraped with crawl4ai (2 workers).
4. The local model extracts candidate facts per bound aspect.
5. PCC verifies all candidates in one batched synthesis: aspect coverage,
   conflicts (only between compatible metric/unit/period/geography/definition),
   and a broad-conclusion flag.
6. Output: "Deep web evidence" block with sources, verified facts, conflicts,
   coverage gaps.

## Stage 3: Evidence Formatting

Each mode produces a text block:
- Quick: ranked snippets with titles and URLs.
- Normal: sources with extracted facts, or "No sufficient information
  extracted" if extraction failed.
- Deep: sources, verified facts, conflicts, coverage gaps, and a
  "Broad conclusion allowed" flag.

The structured `ResearchResult` is carried alongside the text in
`ToolResult.structured`; formatted text is presentation only.

## Stage 4: Finalization (agent.py)

The finalizer converts evidence into a user-facing answer:

1. Attempt order: configured fallback model first, then the selected model.
   Each receives a dedicated finalizer prompt + user question + evidence text.
2. Validation: response must be non-empty, not JSON/code fence, without tool
   calls, in the user's language, and — when coverage is partial — must say
   the research is partial.
3. If all attempts fail: deterministic fallback renders the structured result
   (`render_fallback`) or the raw evidence text.

An empty model reply never reaches the user: the agent falls back to the
finalizer or to an explicit message.

## Known Failure Points

1. **Finalizer language_mismatch:** A local model can ignore "answer in the
   user's language". Validation rejects it and the next model or the
   deterministic fallback answers instead.
2. **LLM skips search:** For queries not covered by forced search regex, the
   model may answer from knowledge and hallucinate.
3. **Planner latency:** Planning can consume a large part of the mode
   deadline; the budget then stops the search and reports it honestly.
4. **Empty extraction:** The extraction model can return valid JSON with an
   empty facts list. Recovery extracts facts from malformed JSON when
   possible; otherwise the source is reported as insufficient.
5. **Quick quality on transliteration:** Query terms in Russian transliteration
   may not match English terms in results, lowering quick-mode ranking.
