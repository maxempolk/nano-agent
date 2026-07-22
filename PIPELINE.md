# Agent → Web Search Pipeline

## Overview

User sends a message. The agent decides whether to search the web. If search happens, results go through a finalizer that writes the user-facing answer. The pipeline has 4 stages: search trigger, search execution, evidence formatting, finalization.

## Stage 1: Search Trigger

Two paths decide whether web_search runs:

**Path A — Forced search (regex in agent.py).** Before the LLM sees the message, regex patterns check if the query needs fresh web data. Patterns include: changing facts (сейчас, последн, курс, погода, версия), explicit search commands (загугли, поищи, найди, search), and comparison requests (сравни, сравнение). If matched, web_search is called immediately with the user's query. The LLM does not participate in this decision. After forced search, tools are removed from the turn so the LLM cannot call web_search again.

**Path B — LLM decides.** If no regex matched, the message goes to the LLM with web_search available as a tool. The LLM chooses whether to call it. Problem: the LLM sometimes skips search and answers from its own knowledge, which leads to hallucinations or empty responses for factual queries.

## Stage 2: Search Execution (web_search.py)

web_search.execute(query, depth) runs one of three modes:

**Quick mode** (default for short factual queries):
1. DuckDuckGo search returns 10 results with title, URL, and snippet.
2. Quality assessment scores the results: relevance (term overlap with query, up to 30 points), value presence (answer pattern found, +30), freshness (recent date markers, +20), authority (known official domain, +20).
3. If score >= 40 (or >= 50 for freshness-sensitive queries) and at least 1 relevant result: return snippets as-is. Zero LLM calls.
4. If quality is insufficient and depth was "auto": escalate to normal mode.

**Normal mode** (for complex queries or after quick escalation):
1. LLM planner (PCC model) generates 1-2 search queries and identifies the subject. Budget: 5 LLM calls, 45s deadline.
2. DuckDuckGo search for each planned query.
3. Top 2 pages are scraped with crawl4ai to get full text.
4. LLM extracts candidate facts from each page (claim + evidence excerpt + date).
5. Facts are filtered for relevance, freshness, and value type match.
6. Output: "Web evidence" block with sources and extracted facts.

**Deep mode** (for multi-aspect research):
1. LLM planner decomposes the question into aspects (e.g., cost of living, income, safety) and generates 1 query per aspect.
2. Up to 5 sources selected across queries, prioritizing official domains.
3. All pages scraped with crawl4ai.
4. LLM extracts facts per aspect from each page.
5. LLM synthesizes: verifies facts, detects conflicts, assesses aspect coverage.
6. Output: "Deep web evidence" block with sources, verified facts, conflicts, and coverage gaps.

## Stage 3: Evidence Formatting

Each mode produces a text block:
- Quick: raw snippets with titles and URLs.
- Normal: sources with extracted facts, or "No sufficient information extracted" if extraction failed.
- Deep: sources, verified facts, conflicts, coverage gaps, and a "Broad conclusion allowed" flag.

This text is the evidence passed to the finalizer.

## Stage 4: Finalization (agent.py)

The finalizer converts evidence into a user-facing answer:

1. Attempt 1: local model (AFM). Receives system prompt ("Lead with the direct answer, match depth to evidence") + user question + evidence text.
2. Validation: response must be non-empty, not JSON/code fence, and in the user's language. Language mismatch (model answers in English to a Russian question) causes rejection. This happens in ~28% of calls.
3. Attempt 2: PCC (fallback model). Same validation.
4. If both fail: deterministic fallback dumps raw evidence text with a prefix. The user sees unprocessed snippets, often in English.

## Known Failure Points

1. **Finalizer language_mismatch (28% of calls):** Local model ignores "answer in user's language" instruction. Causes retry delay or fallback to raw dump.
2. **LLM skips search:** For queries not covered by forced search regex, the model may answer from knowledge and hallucinate.
3. **PCC planner timeout:** Planning call can take 50+ seconds, exceeding the 45s normal mode deadline. Search is killed before scraping starts.
4. **Empty extraction:** Local model returns valid JSON with empty facts list for page content. Retry does not help because the response is structurally valid.
5. **Quick quality on transliteration:** Query terms in Russian transliteration (пайтон) don't match English terms in results (python), causing false escalation.
