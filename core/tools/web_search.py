from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import math
import re
import shutil
import subprocess
import tempfile
from threading import Lock
import time
from typing import TYPE_CHECKING, TypeVar
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from ddgs import DDGS
import httpx
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from core.llm import call_llm

if TYPE_CHECKING:
    from core.logger import SessionLogger

MAX_RESULTS = 10
MAX_RETRIES = 2
MAX_FORMATTED_RESULT_CHARS = 1950
QUICK_RESULTS = 5
NORMAL_SOURCES = 2
DEEP_SOURCES = 5
DEEP_EXTRACTION_WORKERS = 2
MAX_DEEP_FACTS = 8
MIN_USABLE_PAGE_CHARS = 200
MAX_SOURCE_REPLACEMENT_ATTEMPTS = 2
MAX_POST_EXTRACTION_REPLACEMENTS = 1
PAGE_CONTEXT_CHARS = 7_000
LOCAL_CONTEXT_LIMIT = 8_192
LLM_INPUT_TOKEN_BUDGET = 6_000
CONSERVATIVE_CHARS_PER_TOKEN = 1.5
MESSAGE_TOKEN_OVERHEAD = 128

MODE_LIMITS = {
    "quick": (0, 15.0),
    "normal": (5, 45.0),
    "deep": (8, 100.0),
}

OFFICIAL_DOMAIN_HINTS = {
    "gpt": "openai.com",
    "chatgpt": "openai.com",
    "openai": "openai.com",
    "apple": "apple.com",
    "microsoft": "microsoft.com",
    "google": "google.com",
    "anthropic": "anthropic.com",
    "claude": "anthropic.com",
    "python": "python.org",
    "github": "github.com",
}

LOW_QUALITY_RESEARCH_HOSTS = {
    "quora.com", "reddit.com", "facebook.com", "x.com", "twitter.com",
    "pinterest.com", "tiktok.com", "jotform.com", "surveymonkey.com",
    "template.net", "forms.app",
}

KNOWN_PRIMARY_DOMAINS = {
    *OFFICIAL_DOMAIN_HINTS.values(),
    "ssb.no", "regjeringen.no", "norges-bank.no",
    "europa.eu", "ec.europa.eu", "oecd.org", "worldbank.org",
    "who.int", "un.org",
}


def _is_authoritative_host(host: str) -> bool:
    host = host.lower().strip(".")
    if not host:
        return False
    if any(host == domain or host.endswith(f".{domain}") for domain in KNOWN_PRIMARY_DOMAINS):
        return True
    labels = set(host.split("."))
    return bool(labels & {"gov", "government", "gouv"})


def _is_low_quality_host(host: str) -> bool:
    return any(
        host == domain or host.endswith(f".{domain}")
        for domain in LOW_QUALITY_RESEARCH_HOSTS
    )

DEEP_QUERY = re.compile(
    r"\b(подробн\w*|глубок\w*|исслед\w*|сравни\w*|обзор\w*|"
    r"deep research|in-depth|compare|comparison|research)\b",
    re.IGNORECASE,
)
NORMAL_QUERY = re.compile(
    r"\b(проанализ\w*|проверь источники|несколько источников|"
    r"analy[sz]\w*|multiple sources|verify sources)\b",
    re.IGNORECASE,
)
LATEST_QUERY = re.compile(r"\b(последн\w*|новейш\w*|latest|newest)\b", re.IGNORECASE)
PRODUCT_VERSION = re.compile(
    r"\b[a-z][a-z0-9]*-(\d+)(?:[._](\d+))?\b",
    re.IGNORECASE,
)
CURRENT_QUERY = re.compile(
    r"\b(сейчас|сегодня|актуальн\w*|текущ\w*|current|today|live|latest|newest)\b",
    re.IGNORECASE,
)
OFFICIAL_QUERY = re.compile(
    r"\b(официальн\w*|official|госстатистик\w*|release notes?)\b",
    re.IGNORECASE,
)
NUMBER_QUERY = re.compile(r"\b(сколько|количество|число|how many|number of)\b", re.IGNORECASE)
PRICE_QUERY = re.compile(
    r"\b(курс|цен\w*|стоимость|стоит|price|rate|worth)\b",
    re.IGNORECASE,
)
WEATHER_QUERY = re.compile(r"\b(погод\w*|температур\w*|weather|temperature)\b", re.IGNORECASE)
VERSION_QUERY = re.compile(r"\b(верси\w*|version|release|модел\w*|model)\b", re.IGNORECASE)
DATE_QUERY = re.compile(r"\b(когда|дат\w*|when|date)\b", re.IGNORECASE)
NEGATIVE_EVIDENCE = re.compile(
    r"\b(insufficient information|no (?:relevant )?(?:data|information)|"
    r"does not contain (?:relevant )?(?:data|information)|"
    r"does not (?:provide|mention|include).{0,40}(?:data|information)|"
    r"(?:data|information).{0,20}(?:is|are) (?:absent|unavailable|missing)|"
    r"no comparative data|недостаточно информации|данные отсутствуют)\b",
    re.IGNORECASE,
)
SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for explicit search requests and current or changing facts. "
            "Returns source URLs and evidence. Call once per question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "depth": {
                    "type": "string",
                    "enum": ["auto", "quick", "normal", "deep"],
                    "description": (
                        "Use auto. Use deep only when the user's message explicitly requests "
                        "deep research."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


class NormalFact(BaseModel):
    claim: str
    evidence: str
    published_at: str = ""
    metric: str = ""
    unit: str = ""
    period: str = ""
    geography: str = ""
    definition: str = ""


class CandidateFact(BaseModel):
    claim: str
    evidence: str
    published_at: str = ""


class CandidateExtraction(BaseModel):
    facts: list[CandidateFact] = Field(default_factory=list)


class NormalPageEvidence(BaseModel):
    facts: list[NormalFact]
    insufficient_information: bool
    aspect_name: str = ""
    answers_aspect: bool = True
    relevance_score: int = Field(default=100, ge=0, le=100)
    rejection_reason: str = ""


class DeepFact(BaseModel):
    claim: str
    source_ids: list[int] = Field(default_factory=list)
    published_at: str = ""
    metric: str = ""
    unit: str = ""
    period: str = ""
    geography: str = ""
    definition: str = ""


class ConflictAssessment(BaseModel):
    description: str
    source_ids: list[int] = Field(default_factory=list)
    metric: str = ""
    unit: str = ""
    period: str = ""
    geography: str = ""
    definition: str = ""


class AspectReview(BaseModel):
    name: str
    status: str
    source_ids: list[int] = Field(default_factory=list)
    reason: str = ""


class DeepSynthesis(BaseModel):
    facts: list[DeepFact] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    conflict_details: list[ConflictAssessment] = Field(default_factory=list)
    aspect_reviews: list[AspectReview] = Field(default_factory=list)
    insufficient_information: bool = False


class ResearchSource(BaseModel):
    source_id: int
    title: str
    url: str
    official: bool = False
    year: int | None = None


class ResearchResult(BaseModel):
    query: str
    mode: str
    sources: list[ResearchSource] = Field(default_factory=list)
    facts: list[DeepFact] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(default_factory=list)
    aspect_statuses: dict[str, str] = Field(default_factory=dict)
    broad_conclusion_allowed: bool = True
    insufficient_information: bool = False

    def evidence_text(self) -> str:
        lines = [f"Research question: {self.query}", "Sources:"]
        for source in self.sources:
            metadata = "official" if source.official else "independent"
            if source.year:
                metadata += f", {source.year}"
            lines.append(
                f"[{source.source_id}] {source.title} ({metadata})\n{source.url}"
            )
        lines.append("Facts:")
        for fact in self.facts:
            refs = ",".join(str(source_id) for source_id in fact.source_ids)
            date = f" ({fact.published_at})" if fact.published_at else ""
            lines.append(f"- {fact.claim}{date} [{refs}]")
        if self.conflicts:
            lines.append("Conflicts:")
            lines.extend(f"- {conflict}" for conflict in self.conflicts)
        if self.coverage_gaps:
            lines.append("Missing evidence for:")
            lines.extend(f"- {aspect}" for aspect in self.coverage_gaps)
        confirmed = sum(
            status == "confirmed" for status in self.aspect_statuses.values()
        )
        total = len(self.aspect_statuses)
        if total:
            lines.append(f"Coverage: {confirmed}/{total}")
        lines.append(
            "Broad conclusion allowed: "
            + ("yes" if self.broad_conclusion_allowed else "no")
        )
        return "\n".join(lines)

    def render_fallback(self) -> str:
        if not self.sources:
            return "Поиск не вернул результатов по этому запросу."
        lines = ["По результатам поиска:"]
        for source in self.sources[:5]:
            title = source.title or source.url
            lines.append(f"• {title} — {source.url}")
        return "\n".join(lines)


StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


class StructuredOutputError(ValueError):
    def __init__(self, raw: str, cause: Exception):
        super().__init__(f"Не удалось разобрать structured output: {cause}")
        self.raw = raw


class SearchMode(str, Enum):
    QUICK = "quick"
    NORMAL = "normal"
    DEEP = "deep"


class ExpectedValue(str, Enum):
    FACT = "fact"
    NUMBER = "number"
    PRICE = "price"
    WEATHER = "weather"
    VERSION = "version"
    DATE = "date"


class EvidenceKind(str, Enum):
    FACT = "fact"
    NUMBER = "number"
    PRICE = "price"
    DATE = "date"
    CHARACTERISTIC = "characteristic"
    LIMITATION = "limitation"
    ARGUMENT = "argument"
    POSITION = "position"
    COMPARISON = "comparison"


class AspectStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    MISSING = "missing"
    REJECTED = "rejected"


class ResearchAspect(BaseModel):
    name: str
    query: str
    expected_evidence: EvidenceKind = EvidenceKind.FACT
    requirement: str = ""
    priority: int = Field(default=3, ge=1, le=5)
    requires_freshness: bool = False
    preferred_source_type: str = ""
    acceptance_criteria: str = ""


class AspectOutcome(BaseModel):
    name: str
    status: AspectStatus
    source_id: int | None = None
    failure_reason: str = ""


class PlannedQuery(BaseModel):
    query: str
    aspect: str = ""
    official_domain: str = ""
    expected_evidence: EvidenceKind = EvidenceKind.FACT
    requirement: str = ""
    priority: int = Field(default=3, ge=1, le=5)
    requires_freshness: bool = False
    preferred_source_type: str = ""
    acceptance_criteria: str = ""


class NormalPlan(BaseModel):
    queries: list[str] = Field(default_factory=list)
    subject: str = ""
    expected_value: ExpectedValue = ExpectedValue.FACT
    official_domain: str = ""


class ResearchPlan(BaseModel):
    queries: list[PlannedQuery] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    subject: str = ""
    aspects: list[str] = Field(default_factory=list)
    expected_value: ExpectedValue = ExpectedValue.FACT
    requires_freshness: bool = False
    official_domain: str = ""
    official_domains: list[str] = Field(default_factory=list)
    research_aspects: list[ResearchAspect] = Field(default_factory=list)


@dataclass(frozen=True)
class SearchIntent:
    original_query: str
    normalized_query: str
    expected_value: ExpectedValue
    requires_freshness: bool
    official_requested: bool
    official_domain: str | None
    currency: str | None = None
    preferred_domains: tuple[str, ...] = ()
    subject: str = ""

    def search_query(self) -> str:
        query = self.normalized_query
        should_restrict = bool(
            self.official_domain
            and (
                self.official_requested
                or (
                    self.requires_freshness
                    and self.expected_value in {ExpectedValue.NUMBER, ExpectedValue.VERSION}
                )
            )
        )
        if should_restrict and "site:" not in query.lower():
            return f"{query} site:{self.official_domain}"
        return query


class SearchBudgetExceeded(RuntimeError):
    pass


class SearchInputTooLarge(SearchBudgetExceeded):
    pass


@dataclass
class SearchBudget:
    mode: SearchMode
    max_llm_calls: int
    timeout_seconds: float
    started_at: float = field(default_factory=time.monotonic)
    llm_calls: int = 0
    lock: Lock = field(default_factory=Lock, repr=False)

    @classmethod
    def for_mode(cls, mode: SearchMode) -> "SearchBudget":
        max_calls, timeout = MODE_LIMITS[mode.value]
        return cls(mode, max_calls, timeout)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def check_deadline(self) -> None:
        if self.elapsed >= self.timeout_seconds:
            raise SearchBudgetExceeded(
                f"web_search timeout exceeded ({self.timeout_seconds:.0f}s)"
            )

    def consume_llm(self) -> int:
        with self.lock:
            self.check_deadline()
            if self.llm_calls >= self.max_llm_calls:
                raise SearchBudgetExceeded(
                    f"web_search LLM budget exhausted ({self.max_llm_calls})"
                )
            self.llm_calls += 1
            return self.llm_calls


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _clean_published_at(value: str) -> str:
    value = _clip(value, 40)
    if not value or re.search(
        r"insufficient|unknown|not available|no (?:date|information)|null|none",
        value,
        re.IGNORECASE,
    ):
        return ""
    return value


def _clip_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = "\n… [middle removed to fit local context] …\n"
    available = max(0, limit - len(marker))
    head = available * 2 // 5
    return text[:head] + marker + text[-(available - head):]


def _estimate_input_tokens(messages: list[dict]) -> int:
    payload = json.dumps(messages, ensure_ascii=False, default=str)
    return math.ceil(len(payload) / CONSERVATIVE_CHARS_PER_TOKEN) + MESSAGE_TOKEN_OVERHEAD


def _json_object(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("JSON object not found")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("top-level JSON must be an object")
    return value


def _flat_json_schema(model: type[BaseModel]) -> dict:
    schema = model.model_json_schema()
    definitions = schema.get("$defs", {})

    def inline(value):
        if isinstance(value, list):
            return [inline(item) for item in value]
        if not isinstance(value, dict):
            return value
        ref = value.get("$ref", "")
        prefix = "#/$defs/"
        if ref.startswith(prefix):
            resolved = definitions.get(ref[len(prefix):], {})
            return inline({**resolved, **{key: item for key, item in value.items() if key != "$ref"}})
        return {
            key: inline(item)
            for key, item in value.items()
            if key != "$defs"
        }

    return inline(schema)


def _afm_generation_schema(model: type[BaseModel]) -> dict:
    """Convert Pydantic's schema into the dialect accepted by ``fm serve``."""
    schema = _flat_json_schema(model)

    def convert(node: object, *, root: bool = False) -> object:
        if isinstance(node, list):
            return [convert(item) for item in node]
        if not isinstance(node, dict):
            return node

        converted = {
            key: convert(value)
            for key, value in node.items()
            if key not in {"default"}
            and not (key == "title" and not root and node.get("type") != "object")
        }
        if converted.get("type") == "object":
            properties = converted.get("properties", {})
            if isinstance(properties, dict):
                converted["additionalProperties"] = False
                converted["x-order"] = list(properties)
        return converted

    return convert(schema, root=True)  # type: ignore[return-value]


class WebSearchTool:
    SCHEMA = SCHEMA

    def __init__(self, client: OpenAI, model: str, model_mini: str | None = None,
                 planner_model: str | None = None,
                 deep_planner_model: str | None = None,
                 logger: SessionLogger | None = None,
                 force_depth: str | None = None):
        if force_depth not in {None, "quick", "normal", "deep"}:
            raise ValueError("force_depth должен быть quick, normal, deep или None")
        self.client = client
        self.model = model
        self.model_mini = model_mini or model
        self.planner_model = planner_model or model
        self.deep_planner_model = deep_planner_model or self.planner_model
        self.logger = logger
        self.force_depth = force_depth
        self.last_query = ""
        self.last_stats: dict = {}
        self.last_intent: SearchIntent | None = None
        self.last_plan: ResearchPlan | None = None
        self.last_result: ResearchResult | None = None
        self._budget: SearchBudget | None = None
        self._aggregate = {
            "total": 0, "quick": 0, "escalated": 0,
            "normal": 0, "deep": 0,
        }

    def _store_result(self, query: str, mode: SearchMode, results: list[dict],
                      synthesis: DeepSynthesis,
                      coverage_gaps: list[str] | None = None,
                      outcomes: list[AspectOutcome] | None = None) -> ResearchResult:
        sources = []
        for source_id, result in enumerate(results, start=1):
            host = urlparse(result.get("href", "")).hostname or ""
            sources.append(ResearchSource(
                source_id=source_id,
                title=_clip(result.get("title", ""), 160),
                url=_clip(result.get("href", ""), 240),
                official=_is_authoritative_host(host),
                year=self._source_year(result),
            ))
        statuses = {item.name: item.status.value for item in (outcomes or [])}
        required_missing = False
        if outcomes and self.last_plan:
            priorities = {item.name: item.priority for item in self.last_plan.research_aspects}
            required_missing = any(
                item.status != AspectStatus.CONFIRMED and priorities.get(item.name, 3) >= 4
                for item in outcomes
            )
        confirmed = sum(item.status == AspectStatus.CONFIRMED for item in (outcomes or []))
        coverage_ratio = confirmed / len(outcomes) if outcomes else 1.0
        broad_allowed = not required_missing and coverage_ratio >= 0.8
        self.last_result = ResearchResult(
            query=query,
            mode=mode.value,
            sources=sources,
            facts=synthesis.facts[:MAX_DEEP_FACTS],
            conflicts=synthesis.conflicts[:4],
            coverage_gaps=(coverage_gaps or [])[:6],
            aspect_statuses=statuses,
            broad_conclusion_allowed=broad_allowed,
            insufficient_information=synthesis.insufficient_information,
        )
        return self.last_result

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger.info(f"web_search | {message}")

    def _call_model(self, messages: list, stage: str, model: str | None = None,
                    response_format: dict | None = None):
        call_number = None
        input_estimate = _estimate_input_tokens(messages)
        if input_estimate > LLM_INPUT_TOKEN_BUDGET:
            raise SearchInputTooLarge(
                f"web_search input estimate {input_estimate} exceeds safe budget "
                f"{LLM_INPUT_TOKEN_BUDGET}/{LOCAL_CONTEXT_LIMIT}"
            )
        if self._budget:
            call_number = self._budget.consume_llm()
        started = time.monotonic()
        try:
            return call_llm(
                self.client,
                model or self.model_mini,
                messages,
                response_format=response_format,
            )
        except Exception as error:
            message = str(error).lower()
            if "context size" in message or "maximum allowed context" in message:
                raise SearchInputTooLarge(str(error)) from error
            raise
        finally:
            elapsed = time.monotonic() - started
            call_text = f" | call={call_number}" if call_number is not None else ""
            self._log(
                f"stage={stage}{call_text} | input_estimate={input_estimate}/"
                f"{LOCAL_CONTEXT_LIMIT} | elapsed={elapsed:.2f}s"
            )

    def _select_mode(self, query: str, depth: str = "auto") -> SearchMode:
        if self.force_depth:
            return SearchMode(self.force_depth)
        if depth not in {"auto", "quick", "normal", "deep"}:
            raise ValueError("depth должен быть auto, quick, normal или deep")
        if depth != "auto":
            return SearchMode(depth)
        if DEEP_QUERY.search(query):
            return SearchMode.DEEP
        if len(query) > 180 or NORMAL_QUERY.search(query) or query.count("\n") >= 2:
            return SearchMode.NORMAL
        return SearchMode.QUICK

    def _structured(self, prompt: str, output_type: type[StructuredModel],
                    max_attempts: int = MAX_RETRIES, *, model: str | None = None,
                    stage: str = "structured") -> StructuredModel:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": re.sub(
                    r"(?<!^)(?=[A-Z])", "_", output_type.__name__
                ).lower(),
                "strict": True,
                "schema": _afm_generation_schema(output_type),
            },
        }
        max_payload_chars = int(
            (LLM_INPUT_TOKEN_BUDGET - MESSAGE_TOKEN_OVERHEAD - 250)
            * CONSERVATIVE_CHARS_PER_TOKEN
        )
        retry_note = ""
        raw = ""
        last_error: Exception = ValueError("empty response")

        for attempt in range(max_attempts):
            prompt_limit = max(
                500,
                max_payload_chars - len(retry_note) - 100,
            )
            fitted_prompt = _clip_middle(prompt, prompt_limit)
            if fitted_prompt != prompt:
                self._log(
                    f"stage=input_trim | attempt={attempt + 1} | "
                    f"original_chars={len(prompt)} | fitted_chars={len(fitted_prompt)}"
                )
            response = self._call_model(
                [{"role": "user", "content": fitted_prompt + retry_note}],
                stage,
                model=model,
                response_format=response_format,
            )
            raw = response.choices[0].message.content or ""
            try:
                return output_type.model_validate(_json_object(raw))
            except (ValueError, json.JSONDecodeError, ValidationError) as error:
                last_error = error
                self._log(
                    f"stage=structured_invalid | attempt={attempt + 1}/{max_attempts} | "
                    f"error={_clip(str(error), 300)} | raw={_clip(raw, 500)}"
                )
                if attempt + 1 < max_attempts:
                    retry_note = (
                        "\n\nPrevious response was invalid JSON. "
                        f"Error: {_clip(str(error), 200)}. "
                        "Return valid JSON with all required fields."
                    )

        raise StructuredOutputError(raw, last_error)

    async def _scrape_crawl4ai_async(self, url: str) -> str:
        try:
            browser_config = BrowserConfig(headless=True, verbose=False)
            run_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                page_timeout=30000,
            )
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(url=url, config=run_config)
                if result.success and result.markdown:
                    return result.markdown.fit_markdown or result.markdown or ""
                return ""
        except Exception:
            return ""

    def _scrape_crawl4ai(self, url: str) -> str:
        try:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self._scrape_crawl4ai_async(url))
            finally:
                loop.close()
        except Exception:
            return ""

    async def _scrape_batch_async(self, urls: list[str]) -> dict[str, str]:
        results: dict[str, str] = {}
        try:
            browser_config = BrowserConfig(headless=True, verbose=False)
            run_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                page_timeout=30000,
            )
            async with AsyncWebCrawler(config=browser_config) as crawler:
                for url in urls:
                    try:
                        result = await crawler.arun(url=url, config=run_config)
                        if result.success and result.markdown:
                            results[url] = (
                                result.markdown.fit_markdown
                                or result.markdown or ""
                            )
                        else:
                            results[url] = ""
                    except Exception:
                        results[url] = ""
        except Exception:
            for url in urls:
                results.setdefault(url, "")
        return results

    def _scrape_batch(self, urls: list[str]) -> dict[str, str]:
        if not urls:
            return {}
        try:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self._scrape_batch_async(urls))
            finally:
                loop.close()
        except Exception:
            return {url: "" for url in urls}

    def _scrape_pdf(self, url: str) -> str:
        if not shutil.which("pdftotext"):
            return ""
        try:
            response = httpx.get(url, timeout=20, follow_redirects=True)
            response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                return ""
            with tempfile.NamedTemporaryFile(suffix=".pdf") as pdf_file, \
                    tempfile.NamedTemporaryFile(suffix=".txt") as text_file:
                pdf_file.write(response.content)
                pdf_file.flush()
                completed = subprocess.run(
                    ["pdftotext", "-layout", pdf_file.name, text_file.name],
                    capture_output=True,
                    timeout=20,
                    check=False,
                )
                if completed.returncode != 0:
                    return ""
                text_file.seek(0)
                return text_file.read().decode("utf-8", errors="replace")
        except Exception:
            return ""

    def _scrape(self, url: str) -> str:
        text = self._scrape_pdf(url) if urlparse(url).path.lower().endswith(".pdf") else ""
        if len(text) < 200:
            text = self._scrape_crawl4ai(url) or text
        return text or "Не удалось извлечь текст."

    def _search(self, query: str) -> list[dict]:
        started = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(1 + MAX_RETRIES):
            try:
                with DDGS() as ddg:
                    results = list(ddg.text(query, max_results=MAX_RESULTS))
                if self._budget:
                    self._budget.check_deadline()
                self._log(
                    f"stage=search | results={len(results)} | "
                    f"attempt={attempt + 1} | "
                    f"elapsed={time.monotonic() - started:.2f}s"
                )
                return results
            except Exception as error:
                last_error = error
                self._log(
                    f"stage=search_error | attempt={attempt + 1}/{1 + MAX_RETRIES} | "
                    f"error={_clip(str(error), 120)}"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(2 * (attempt + 1))
        self._log(f"stage=search_failed | error={_clip(str(last_error), 120)}")
        return []

    def _search_many(self, queries: list[str]) -> list[dict]:
        if len(queries) == 1:
            return self._search(queries[0])
        batches: list[list[dict] | None] = [None] * len(queries)
        with ThreadPoolExecutor(max_workers=min(3, len(queries))) as executor:
            futures = {
                executor.submit(self._search, query): index
                for index, query in enumerate(queries)
            }
            for future in as_completed(futures):
                batches[futures[future]] = future.result()

        merged: list[dict] = []
        seen: set[str] = set()
        for query_index, batch in enumerate(batches):
            for result in batch or []:
                url = result.get("href", "")
                if not url or url in seen:
                    continue
                seen.add(url)
                aspect = self._aspect_for_query_index(query_index)
                merged.append({
                    **result,
                    "_plan_query": query_index,
                    "_aspect_index": query_index,
                    "_aspect_name": aspect.name if aspect else "",
                })
        self._log(
            f"stage=search_merge | queries={len(queries)} | unique_results={len(merged)}"
        )
        return merged

    def _aspect_for_query_index(self, index: int) -> ResearchAspect | None:
        if not self.last_plan:
            return None
        if 0 <= index < len(self.last_plan.research_aspects):
            return self.last_plan.research_aspects[index]
        if 0 <= index < len(self.last_plan.queries):
            query = self.last_plan.queries[index]
            return ResearchAspect(
                name=query.aspect or query.query,
                query=query.query,
                expected_evidence=query.expected_evidence,
                requirement=query.requirement,
                priority=query.priority,
                requires_freshness=query.requires_freshness,
                preferred_source_type=query.preferred_source_type,
                acceptance_criteria=query.acceptance_criteria,
            )
        return None

    def _official_domain(self, query: str) -> str | None:
        words = set(re.findall(r"[\w-]+", query.lower()))
        domains = {
            domain
            for word, domain in OFFICIAL_DOMAIN_HINTS.items()
            if word in words
        }
        return next(iter(domains)) if len(domains) == 1 else None

    def _analyze_intent(self, query: str) -> SearchIntent:
        lowered = query.lower()
        currency = None
        if re.search(r"\b(руб\w*|rub|ruble)\b", lowered):
            currency = "RUB"
        elif re.search(r"\b(евро|eur)\b", lowered):
            currency = "EUR"
        elif re.search(r"\b(доллар\w*|usd)\b", lowered):
            currency = "USD"

        if WEATHER_QUERY.search(query):
            expected = ExpectedValue.WEATHER
        elif PRICE_QUERY.search(query):
            expected = ExpectedValue.PRICE
        elif VERSION_QUERY.search(query):
            expected = ExpectedValue.VERSION
        elif NUMBER_QUERY.search(query):
            expected = ExpectedValue.NUMBER
        elif DATE_QUERY.search(query):
            expected = ExpectedValue.DATE
        else:
            expected = ExpectedValue.FACT

        freshness = bool(
            CURRENT_QUERY.search(query) or LATEST_QUERY.search(query)
        ) or expected in {
            ExpectedValue.PRICE,
            ExpectedValue.WEATHER,
        }
        normalized = " ".join(query.split())

        official_requested = bool(OFFICIAL_QUERY.search(query))
        domain = self._official_domain(f"{query} {normalized}")
        return SearchIntent(
            original_query=query,
            normalized_query=normalized,
            expected_value=expected,
            requires_freshness=freshness,
            official_requested=official_requested,
            official_domain=domain,
            currency=currency,
        )

    def _plan_research(self, query: str, mode: SearchMode) -> tuple[ResearchPlan, SearchIntent]:
        fallback = self._analyze_intent(query)
        if mode == SearchMode.DEEP:
            prompt = (
                "Спланируй глубокое веб-исследование. Декомпозируй вопрос на аспекты "
                "и создай поисковые запросы, покрывающие каждый аспект.\n\n"
                "Структура:\n"
                "1. Помести основной объект в subject. Каждое измерение — короткий "
                "английский аспект.\n"
                "2. Создай 1-5 объектов запросов. Каждый запрос нацелен на один аспект "
                "с контрактом доказательств: expected_evidence, requirement, "
                "priority (1-5), freshness, preferred_source_type, "
                "acceptance_criteria.\n"
                "3. Отрази контракты в research_aspects в том же порядке, что и "
                "queries.\n"
                "4. Перечисли домены первичных источников в official_domains. "
                "Минимум половина запросов — на первичные источники.\n\n"
                "Правила:\n"
                "- expected_value: fact для широких вопросов; number, price, "
                "weather, version или date только когда весь вопрос запрашивает "
                "этот единственный тип значения.\n"
                "- requires_freshness: true только когда важны актуальные данные.\n"
                "- official_domain: только когда ровно один первичный домен "
                "релевантен.\n\n"
                f"Вопрос: {query}"
            )
            plan_model = self.deep_planner_model
        else:
            prompt = (
                "Сгенерируй 1-2 коротких английских поисковых запроса, которые "
                "найдут ответ.\n"
                "Помести основной объект или тему в subject.\n"
                "Если один официальный сайт — первичный источник, укажи его домен "
                "в official_domain. Иначе оставь пустым.\n\n"
                f"Вопрос: {query}"
            )
            plan_model = self.planner_model
        try:
            if mode == SearchMode.DEEP:
                plan = self._structured(
                    prompt, ResearchPlan, max_attempts=1,
                    model=plan_model, stage="plan",
                )
            else:
                normal_plan = self._structured(
                    prompt, NormalPlan, max_attempts=1,
                    model=plan_model, stage="plan",
                )
                plan = ResearchPlan(
                    search_queries=normal_plan.queries,
                    subject=normal_plan.subject,
                    expected_value=normal_plan.expected_value,
                    official_domain=normal_plan.official_domain,
                )
        except Exception as error:
            self._log(f"stage=plan_fallback | reason={_clip(str(error), 180)}")
            plan = ResearchPlan(
                queries=[PlannedQuery(
                    query=fallback.search_query(),
                    aspect=query,
                    official_domain=fallback.official_domain or "",
                )],
                search_queries=[fallback.search_query()],
                subject="",
                aspects=[query],
                expected_value=fallback.expected_value,
                requires_freshness=fallback.requires_freshness,
                official_domain=fallback.official_domain or "",
            )

        query_limit = 5 if mode == SearchMode.DEEP else 2

        def clean_domain(raw_domain: str) -> str:
            value = re.sub(
                r"^https?://", "", raw_domain.lower().strip()
            ).split("/")[0]
            return value if re.fullmatch(
                r"(?:[a-z0-9-]+\.)+[a-z]{2,}", value
            ) else ""

        raw_queries = list(plan.queries[:query_limit])
        for item in plan.search_queries:
            if len(raw_queries) >= query_limit:
                break
            if isinstance(item, str) and item.strip():
                raw_queries.append(PlannedQuery(query=item))
        for aspect in plan.research_aspects:
            if len(raw_queries) >= query_limit:
                break
            source = aspect.query or aspect.name
            if source and source.strip():
                raw_queries.append(PlannedQuery(query=source, aspect=aspect.name))
        if not raw_queries:
            raw_queries = [PlannedQuery(
                query=fallback.search_query(),
                aspect=query,
                official_domain=fallback.official_domain or "",
            )]

        planned_queries: list[PlannedQuery] = []
        for item in raw_queries:
            planned_domain = clean_domain(item.official_domain)
            search_query = _clip(item.query, 180)
            if planned_domain and "site:" not in search_query.lower():
                search_query = f"{search_query} site:{planned_domain}"
            planned_queries.append(PlannedQuery(
                query=search_query,
                aspect=_clip(item.aspect, 80),
                official_domain=planned_domain,
                expected_evidence=item.expected_evidence,
                requirement=_clip(item.requirement, 160),
                priority=item.priority,
                requires_freshness=item.requires_freshness,
                preferred_source_type=_clip(item.preferred_source_type, 80),
                acceptance_criteria=_clip(item.acceptance_criteria, 180),
            ))

        queries = [item.query for item in planned_queries]
        aspects = [
            _clip(item, 80)
            for item in plan.aspects
            if isinstance(item, str) and item.strip()
        ]
        if not aspects:
            aspects = [item.aspect for item in planned_queries if item.aspect]
        aspects = list(dict.fromkeys(aspects))[:6] or [query]
        domains: list[str] = []
        for raw_domain in [
            *[item.official_domain for item in planned_queries],
            *plan.official_domains,
            plan.official_domain,
            fallback.official_domain or "",
        ]:
            domain = clean_domain(raw_domain)
            if domain and domain not in domains:
                domains.append(domain)
        domain = domains[0] if len(domains) == 1 else ""
        research_aspects: list[ResearchAspect] = []
        for index, item in enumerate(planned_queries):
            supplied = plan.research_aspects[index] if index < len(plan.research_aspects) else None
            research_aspects.append(ResearchAspect(
                name=_clip((supplied.name if supplied else item.aspect) or item.query, 80),
                query=item.query,
                expected_evidence=supplied.expected_evidence if supplied else item.expected_evidence,
                requirement=_clip(
                    (supplied.requirement if supplied else item.requirement) or item.aspect,
                    160,
                ),
                priority=supplied.priority if supplied else item.priority,
                requires_freshness=(
                    supplied.requires_freshness if supplied else item.requires_freshness
                ),
                preferred_source_type=_clip(
                    supplied.preferred_source_type if supplied else item.preferred_source_type,
                    80,
                ),
                acceptance_criteria=_clip(
                    (supplied.acceptance_criteria if supplied else item.acceptance_criteria)
                    or "A directly supported fact answers this aspect.",
                    180,
                ),
            ))
        plan = ResearchPlan(
            queries=planned_queries,
            search_queries=queries,
            subject=_clip(plan.subject, 100),
            aspects=aspects,
            expected_value=plan.expected_value,
            requires_freshness=plan.requires_freshness,
            official_domain=domain,
            official_domains=domains,
            research_aspects=research_aspects,
        )
        intent = SearchIntent(
            original_query=query,
            normalized_query=" ".join(queries),
            expected_value=plan.expected_value,
            requires_freshness=plan.requires_freshness,
            official_requested=bool(OFFICIAL_QUERY.search(query)),
            official_domain=domain or None,
            currency=fallback.currency,
            preferred_domains=tuple(domains),
            subject=_clip(plan.subject, 100),
        )
        self._log(
            f"stage=plan | model={plan_model} | mode={mode.value} | "
            f"aspects={len(aspects)} | queries={len(queries)} | "
            f"expected={plan.expected_value.value} | "
            f"fresh={str(plan.requires_freshness).lower()} | "
            f"official={','.join(domains) or '-'}"
        )
        return plan, intent

    def _rank_results(self, intent: SearchIntent, results: list[dict]) -> list[dict]:
        preferred_domains = intent.preferred_domains or (
            (intent.official_domain,) if intent.official_domain else ()
        )
        wants_latest = intent.requires_freshness and intent.expected_value == ExpectedValue.VERSION
        query_terms = self._quality_terms(intent.normalized_query)

        def score(item: tuple[int, dict]) -> tuple[int, int]:
            index, result = item
            host = urlparse(result.get("href", "")).hostname or ""
            title = result.get("title", "").lower()
            body = result.get("body", "").lower()
            preferred = int(any(host.endswith(domain) for domain in preferred_domains))
            official = int(_is_authoritative_host(host))
            trusted = int(host.endswith(".gov") or host.endswith(".edu"))
            low_quality = int(_is_low_quality_host(host))
            title_terms = self._result_terms({"title": title, "body": "", "href": ""})
            body_terms = self._result_terms({"title": "", "body": body, "href": ""})
            overlap = len(query_terms & title_terms) * 12
            overlap += len(query_terms & body_terms) * 3
            direct_value = int(self._contains_expected_value(
                intent,
                f"{result.get('title', '')} {result.get('body', '')}",
            ))
            fresh = int(self._contains_fresh_marker(f"{title} {body}"))
            source_year = self._source_year(result)
            stale = int(bool(
                intent.requires_freshness
                and source_year
                and source_year < datetime.now().year - 3
            ))
            version_score = 0
            if wants_latest:
                versions = [
                    int(major) * 10 + int(minor or 0)
                    for major, minor in PRODUCT_VERSION.findall(f"{title} {body}")
                ]
                version_score = max(versions, default=0)
            total = (
                official * 200
                + preferred * 20
                + trusted * 40
                + direct_value * 55
                + fresh * 20
                + version_score
                + overlap
                - low_quality * 250
                - stale * 280
            )
            return total, -index

        ranked = sorted(enumerate(results), key=score, reverse=True)
        return [result for _, result in ranked]

    def _select_deep_sources(self, intent: SearchIntent, results: list[dict],
                             aspects: list[str]) -> list[dict]:
        ranked = self._rank_results(intent, results)
        non_low_quality = [
            result for result in ranked
            if not _is_low_quality_host(
                urlparse(result.get("href", "")).hostname or ""
            )
        ]
        if non_low_quality:
            ranked = non_low_quality
        aspect_terms = [self._quality_terms(aspect) for aspect in aspects]
        relevant = [
            result for result in ranked
            if any(
                terms & self._result_terms(result)
                for terms in aspect_terms
            )
        ]
        if relevant:
            ranked = relevant
        uncovered = set(range(len(aspect_terms)))
        selected: list[dict] = []
        used_hosts: dict[str, int] = {}

        query_indices = sorted({
            result.get("_plan_query")
            for result in ranked
            if isinstance(result.get("_plan_query"), int)
        })
        for query_index in query_indices:
            candidates = [
                (index, result)
                for index, result in enumerate(ranked)
                if result.get("_plan_query") == query_index
            ]
            contract = self._aspect_for_query_index(query_index)
            required_terms = self._aspect_required_terms(contract, intent.subject)
            matching = [
                item for item in candidates
                if required_terms & self._result_terms(item[1])
            ]
            if required_terms:
                candidates = matching
            if not candidates or len(selected) >= DEEP_SOURCES:
                continue
            best_index, chosen = max(
                candidates,
                key=lambda item: (
                    sum(
                        1 for aspect_index in uncovered
                        if aspect_terms[aspect_index] & self._result_terms(item[1])
                    ),
                    -item[0],
                ),
            )
            ranked.pop(best_index)
            selected.append(chosen)
            if contract:
                chosen["_aspect_name"] = contract.name
            host = urlparse(chosen.get("href", "")).hostname or ""
            used_hosts[host] = used_hosts.get(host, 0) + 1
            chosen_terms = self._result_terms(chosen)
            uncovered = {
                index for index in uncovered
                if not (aspect_terms[index] & chosen_terms)
            }

        while ranked and len(selected) < DEEP_SOURCES:
            best_index = 0
            best_score = -10_000
            for index, result in enumerate(ranked):
                result_terms = self._result_terms(result)
                new_coverage = sum(
                    1 for aspect_index in uncovered
                    if aspect_terms[aspect_index] & result_terms
                )
                host = urlparse(result.get("href", "")).hostname or ""
                query_diversity = int(
                    all(
                        result.get("_plan_query") != item.get("_plan_query")
                        for item in selected
                    )
                )
                score = new_coverage * 100 + query_diversity * 30
                score -= used_hosts.get(host, 0) * 35
                score -= index
                if score > best_score:
                    best_score = score
                    best_index = index
            chosen = ranked.pop(best_index)
            selected.append(chosen)
            host = urlparse(chosen.get("href", "")).hostname or ""
            used_hosts[host] = used_hosts.get(host, 0) + 1
            chosen_terms = self._result_terms(chosen)
            uncovered = {
                index for index in uncovered
                if not (aspect_terms[index] & chosen_terms)
            }
        return selected

    def _aspect_required_terms(self, aspect: ResearchAspect | None,
                               subject: str = "") -> set[str]:
        if not aspect:
            return set()
        contract = " ".join(filter(None, [
            aspect.name, aspect.requirement, aspect.acceptance_criteria,
        ]))
        return self._quality_terms(contract) - self._quality_terms(subject)

    def _candidate_matches_aspect(self, result: dict,
                                  aspect: ResearchAspect | None,
                                  subject: str = "") -> bool:
        required = self._aspect_required_terms(aspect, subject)
        return not required or bool(required & self._result_terms(result))

    def _rank_quick_results(self, query: str, results: list[dict]) -> list[dict]:
        intent = self._analyze_intent(query)
        return self._rank_results(intent, results)[:QUICK_RESULTS]

    def _quality_terms(self, query: str) -> set[str]:
        stop_words = {
            "the", "and", "for", "with", "from", "what", "which", "current",
            "latest", "live", "official", "statistics", "number", "price",
            "сколько", "количество", "сейчас", "актуальная", "официальная",
            "какая", "какой", "последняя", "последний", "последнюю", "версия",
            "проверь", "источникам", "источники",
            "загугли", "загугл", "поищи", "поиск", "найди", "найти", "ищи",
            "search", "google", "browse", "look", "check", "find",
        }
        return {
            word[:6]
            for word in re.findall(r"[\w-]{3,}", query.lower())
            if word not in stop_words
        }

    def _result_terms(self, result: dict) -> set[str]:
        text = " ".join([
            result.get("title", ""),
            result.get("body", ""),
            urlparse(result.get("href", "")).hostname or "",
        ]).lower()
        return {word[:6] for word in re.findall(r"[\w-]{3,}", text)}

    def _contains_expected_value(self, intent: SearchIntent, text: str) -> bool:
        if intent.expected_value == ExpectedValue.FACT:
            return True
        if intent.expected_value == ExpectedValue.VERSION:
            return bool(
                PRODUCT_VERSION.search(text)
                or re.search(r"\b(?:version|версия|v)\s*\d+(?:\.\d+)+\b", text, re.IGNORECASE)
            )
        if intent.expected_value == ExpectedValue.PRICE:
            return bool(re.search(
                r"(?:[$€£]|\b(?:usd|eur|rub|nok|btc)\b).{0,20}\d|"
                r"\d.{0,20}(?:[$€£]|\b(?:usd|eur|rub|nok|btc)\b)",
                text,
                re.IGNORECASE,
            ))
        if intent.expected_value == ExpectedValue.WEATHER:
            return bool(re.search(r"-?\d+(?:[.,]\d+)?\s*°?\s*[cf]\b", text, re.IGNORECASE))
        if intent.expected_value == ExpectedValue.DATE:
            return bool(re.search(r"\b(?:19|20)\d{2}\b|\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", text))
        month = (
            r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
            r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
            r"dec(?:ember)?|январ\w*|феврал\w*|март\w*|апрел\w*|ма[йя]|июн\w*|"
            r"июл\w*|август\w*|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*)"
        )
        without_dates = re.sub(
            rf"\b(?:{month}\s+\d{{1,2}}|\d{{1,2}}\s+{month})(?:,?\s+\d{{4}})?\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        subject_terms = self._quality_terms(intent.normalized_query)
        for match in re.finditer(r"\b\d{1,3}(?:[ ,]\d{3})*\b", without_dates):
            value = int(match.group().replace(",", "").replace(" ", ""))
            if 1900 <= value <= 2099:
                continue
            before = without_dates[max(0, match.start() - 100):match.start()]
            after = without_dates[match.end():match.end() + 80]
            before_terms = self._result_terms({"title": before, "body": "", "href": ""})
            after_terms = self._result_terms({"title": after, "body": "", "href": ""})
            leading_relation = bool(re.search(
                r"\b(has|have|had|there\s+(?:are|were)|comprises?|contains?|"
                r"totals?|насчитыва\w*|составля\w*|всего)\b[^.!?]{0,45}$",
                before,
                re.IGNORECASE,
            ))
            number_statement = bool(
                subject_terms & before_terms
                and re.search(
                    r"\b(number\s+of|total\s+number\s+of|количество|число)\b"
                    r"[^.!?]{0,60}\b(is|was|stands|составля\w*)\b[^.!?]{0,15}$",
                    before,
                    re.IGNORECASE,
                )
            )
            if (leading_relation and subject_terms & after_terms) or number_statement:
                return True
        return False

    def _contains_fresh_marker(self, text: str) -> bool:
        if CURRENT_QUERY.search(text):
            return True
        years = [int(year) for year in re.findall(r"\b20\d{2}\b", text)]
        if years and max(years) >= datetime.now().year - 1:
            return True
        return bool(re.search(
            r"\b\d+\s+(?:minutes?|hours?|days?)\s+ago\b|\bupdated\b",
            text,
            re.IGNORECASE,
        ))

    def _source_year(self, result: dict) -> int | None:
        text = f"{result.get('title', '')} {result.get('body', '')}"
        years = [int(year) for year in re.findall(r"\b20\d{2}\b", text)]
        return max(years, default=None)

    def _format_quick_results(self, query: str, results: list[dict]) -> str:
        ranked = self._rank_quick_results(query, results)
        lines = ["Quick web results (snippets only):"]
        for index, result in enumerate(ranked, start=1):
            lines.extend([
                f"[{index}] {_clip(result.get('title', ''), 180)}",
                f"URL: {_clip(result.get('href', ''), 220)}",
                f"Snippet: {_clip(result.get('body', ''), 360)}",
                "",
            ])
        formatted = "\n".join(lines).rstrip()
        if len(formatted) > MAX_FORMATTED_RESULT_CHARS:
            return formatted[:MAX_FORMATTED_RESULT_CHARS - 1] + "…"
        return formatted

    def _normalize_quick_query(self, query: str) -> str:
        return self._analyze_intent(query).search_query()

    def _normal_search_query(self, query: str) -> str:
        return self._analyze_intent(query).search_query()

    def _select_relevant_passages(self, query: str, content: str,
                                  result: dict) -> str:
        if len(content) <= PAGE_CONTEXT_CHARS:
            return content

        seed = " ".join([
            query,
            result.get("title", ""),
            result.get("body", ""),
        ]).lower()
        stop_words = {
            "the", "and", "for", "with", "что", "как", "это", "или", "какая",
            "какой", "latest", "current", "сейчас", "последняя", "последний",
        }
        terms = {
            word for word in re.findall(r"[\w.-]{3,}", seed)
            if word not in stop_words and not word.isdigit()
        }
        paragraphs = [
            part.strip()
            for part in re.split(r"\n\s*\n|\n", content)
            if len(part.strip()) >= 40
        ]
        if not paragraphs:
            return content[:PAGE_CONTEXT_CHARS]

        wants_latest = bool(LATEST_QUERY.search(query))

        def score(item: tuple[int, str]) -> tuple[int, int]:
            index, paragraph = item
            lowered = paragraph.lower()
            overlap = sum(1 for term in terms if term in lowered)
            version = max(
                (int(major) * 10 + int(minor or 0)
                 for major, minor in PRODUCT_VERSION.findall(lowered)),
                default=0,
            ) if wants_latest else 0
            recent = 2 if wants_latest and re.search(r"\b202[5-9]\b", lowered) else 0
            return overlap * 10 + version + recent, -index

        ranked = sorted(enumerate(paragraphs), key=score, reverse=True)
        selected: set[int] = set()
        total = 0
        for index, paragraph in ranked:
            extra = len(paragraph) + (2 if selected else 0)
            if selected and total + extra > PAGE_CONTEXT_CHARS:
                continue
            selected.add(index)
            total += extra
            if total >= PAGE_CONTEXT_CHARS * 0.9:
                break
        chosen = "\n\n".join(paragraphs[index] for index in sorted(selected))
        return chosen[:PAGE_CONTEXT_CHARS]

    def _extract_normal_page(self, question: str, result: dict,
                             content: str,
                             aspect: ResearchAspect | None = None) -> NormalPageEvidence:
        if aspect is None:
            aspect = next((
                item for item in (self.last_plan.research_aspects if self.last_plan else [])
                if item.name == result.get("_aspect_name")
            ), self._aspect_for_query_index(result.get("_aspect_index", -1)))
        intent = self.last_intent or self._analyze_intent(question)
        context = self._select_relevant_passages(question, content, result)
        self._log(
            f"stage=select_passages | url={_clip(result.get('href', ''), 120)} | "
            f"source_chars={len(content)} | selected_chars={len(context)}"
        )
        aspect_text = ""
        if aspect:
            aspect_text = (
                f"Aspect: {aspect.name}\nRequirement: {aspect.requirement}\n"
                f"Acceptance criterion: {aspect.acceptance_criteria}\n\n"
            )
        prompt = (
            "Извлеки до 3 фактов с этой страницы, отвечающих на вопрос.\n"
            "Для каждого факта верни:\n"
            "- claim: факт одним предложением\n"
            "- evidence: точная короткая цитата со страницы\n"
            "- published_at: дата если указана на странице, иначе пусто\n"
            "Используй только информацию с этой страницы. "
            "Верни пустой список facts если ничего релевантного нет.\n\n"
            f"{aspect_text}"
            f"Вопрос: {question}\n"
            f"Страница: {result.get('title', '')} ({result.get('href', '')})\n\n"
            f"Содержимое страницы:\n{context}"
        )
        try:
            candidates = self._structured(prompt, CandidateExtraction, max_attempts=MAX_RETRIES)
            evidence = NormalPageEvidence(
                facts=[NormalFact(**fact.model_dump()) for fact in candidates.facts[:3]],
                insufficient_information=not candidates.facts,
                answers_aspect=False,
                relevance_score=0,
                aspect_name=aspect.name if aspect else "",
            )
        except StructuredOutputError as error:
            evidence = self._recover_normal_evidence(error.raw)
        except SearchBudgetExceeded:
            return NormalPageEvidence(
                facts=[], insufficient_information=True,
                aspect_name=aspect.name if aspect else "",
            )
        facts = [
                NormalFact(
                    claim=_clip(fact.claim, 300),
                    evidence=_clip(fact.evidence, 240),
                    published_at=_clean_published_at(fact.published_at),
                    metric=_clip(fact.metric, 80),
                    unit=_clip(fact.unit, 40),
                    period=_clip(fact.period, 60),
                    geography=_clip(fact.geography, 80),
                    definition=_clip(fact.definition, 120),
                )
                for fact in evidence.facts[:3]
                if self._fact_matches_intent(intent, fact, result.get("href", ""))
            ]
        answers = bool(facts)
        return NormalPageEvidence(
            facts=facts,
            insufficient_information=not answers,
            answers_aspect=False,
            relevance_score=0,
            rejection_reason="pending PCC verification" if answers else "no candidates",
            aspect_name=aspect.name if aspect else "",
        )

    def _fact_matches_intent(self, intent: SearchIntent, fact: NormalFact,
                             source_url: str = "") -> bool:
        text = f"{fact.claim} {fact.evidence}"
        if NEGATIVE_EVIDENCE.search(text):
            return False
        if intent.expected_value == ExpectedValue.FACT:
            subject_terms = self._quality_terms(intent.normalized_query)
            fact_terms = self._result_terms({"title": fact.claim, "body": fact.evidence, "href": ""})
            if subject_terms and not (subject_terms & fact_terms):
                return False
            entity_terms = self._quality_terms(intent.subject)
            host = urlparse(source_url).hostname or ""
            official_source = _is_authoritative_host(host)
            if entity_terms and not official_source and not (entity_terms & fact_terms):
                return False
        if not self._contains_expected_value(intent, text):
            return False
        years = [
            int(year)
            for year in re.findall(r"\b20\d{2}\b", f"{fact.published_at} {text}")
        ]
        max_age = 3 if intent.expected_value == ExpectedValue.FACT else 1
        if intent.requires_freshness and years and max(years) < datetime.now().year - max_age:
            return False
        return True

    def _recover_normal_evidence(self, raw: str) -> NormalPageEvidence:
        facts: list[NormalFact] = []
        seen: set[tuple[str, str]] = set()
        try:
            value = _json_object(raw)
        except (ValueError, json.JSONDecodeError):
            value = {}

        def recover_item(item: dict) -> None:
            candidate = item
            nested = item.get("$defs", {}).get("NormalFact")
            if isinstance(nested, dict):
                candidate = nested
            claim = candidate.get("claim")
            evidence = candidate.get("evidence")
            published_at = candidate.get("published_at", "")
            if isinstance(claim, dict):
                wrapped = claim
                claim = wrapped.get("claim") or wrapped.get("title")
                evidence = wrapped.get("evidence") or evidence
                published_at = wrapped.get("published_at", published_at)
            if self._is_recoverable_fact(claim, evidence):
                key = (claim.strip(), evidence.strip())
                if key in seen:
                    return
                seen.add(key)
                facts.append(NormalFact(
                    claim=claim,
                    evidence=evidence,
                    published_at=_clean_published_at(published_at)
                    if isinstance(published_at, str) else "",
                ))

        def walk(node) -> None:
            if isinstance(node, list):
                for item in node:
                    walk(item)
                return
            if not isinstance(node, dict):
                return
            if "claim" in node or "$defs" in node:
                recover_item(node)
            for key in ("facts", "properties"):
                if key in node:
                    walk(node[key])

        walk(value)

        if not facts:
            pair = re.compile(
                r'"claim"\s*:\s*("(?:\\.|[^"\\])*")\s*,\s*'
                r'"evidence"\s*:\s*("(?:\\.|[^"\\])*")'
            )
            for claim_raw, evidence_raw in pair.findall(raw):
                try:
                    claim = json.loads(claim_raw)
                    evidence = json.loads(evidence_raw)
                    if self._is_recoverable_fact(claim, evidence):
                        key = (claim.strip(), evidence.strip())
                        if key not in seen:
                            seen.add(key)
                            facts.append(NormalFact(claim=claim, evidence=evidence))
                except json.JSONDecodeError:
                    continue

        recovered = facts[:3]
        if recovered:
            self._log(f"stage=structured_recovered | facts={len(recovered)}")
        return NormalPageEvidence(
            facts=recovered,
            insufficient_information=not recovered,
        )

    @staticmethod
    def _is_recoverable_fact(claim, evidence) -> bool:
        if not isinstance(claim, str) or not isinstance(evidence, str):
            return False
        text = f"{claim} {evidence}".strip()
        if len(claim.strip()) < 3 or len(evidence.strip()) < 3:
            return False
        return not bool(
            NEGATIVE_EVIDENCE.search(text)
            or re.search(
                r"\b(string value|field description|schema property)\b",
                text,
                re.IGNORECASE,
            )
        )

    def _format_normal_results(self, results: list[dict],
                               pages: list[NormalPageEvidence]) -> str:
        lines = ["Web evidence:"]
        for source_id, (result, page) in enumerate(zip(results, pages), start=1):
            lines.extend([
                f"[{source_id}] {_clip(result.get('title', ''), 160)}",
                f"URL: {_clip(result.get('href', ''), 220)}",
                "Official: " + (
                    "yes" if _is_authoritative_host(
                        urlparse(result.get('href', '')).hostname or ''
                    ) else "no"
                ),
            ])
            source_year = self._source_year(result)
            if source_year:
                lines.append(f"Source year: {source_year}")
            if page.facts:
                for fact in page.facts:
                    lines.append(f"- {_clip(fact.claim, 220)}")
                    lines.append(f"  Evidence: {_clip(fact.evidence, 180)}")
                    if fact.published_at:
                        lines.append(f"  Date: {_clip(fact.published_at, 40)}")
            else:
                lines.append("- No sufficient information extracted.")
            lines.append("")
        formatted = "\n".join(lines).rstrip()
        if len(formatted) > MAX_FORMATTED_RESULT_CHARS:
            return formatted[:MAX_FORMATTED_RESULT_CHARS - 1] + "…"
        return formatted

    def _fetch_pages(self, results: list[dict]) -> dict[str, str]:
        started = time.monotonic()
        urls = [result["href"] for result in results]
        scraped: dict[str, str] = {}
        crawl_urls: list[str] = []
        for url in urls:
            if urlparse(url).path.lower().endswith(".pdf"):
                text = self._scrape_pdf(url)
                if len(text) >= 200:
                    scraped[url] = text
                    continue
            crawl_urls.append(url)
        if crawl_urls:
            scraped.update(self._scrape_batch(crawl_urls))
        for url in urls:
            scraped.setdefault(url, "")
        if self._budget:
            self._budget.check_deadline()
        self._log(
            f"stage=fetch | pages={len(urls)} | elapsed={time.monotonic() - started:.2f}s"
        )
        return scraped

    @staticmethod
    def _usable_page(content: str) -> bool:
        return (
            len(content.strip()) >= MIN_USABLE_PAGE_CHARS
            and "Не удалось извлечь текст" not in content
        )

    def _replace_unreadable_sources(self, selected: list[dict], results: list[dict],
                                    scraped: dict[str, str]) -> None:
        selected_urls = {result.get("href", "") for result in selected}
        candidates = [
            result for result in self._rank_results(
                self.last_intent or self._analyze_intent(""), results
            )
            if result.get("href") not in selected_urls
        ]
        tried_urls: set[str] = set()
        for index, source in enumerate(list(selected)):
            source_url = source.get("href", "")
            if self._usable_page(scraped.get(source_url, "")):
                continue
            same_query = [
                candidate for candidate in candidates
                if candidate.get("_plan_query") == source.get("_plan_query")
            ]
            ordered = same_query + [candidate for candidate in candidates if candidate not in same_query]
            attempts = 0
            replacement = None
            replacement_text = ""
            for candidate in ordered:
                if attempts >= MAX_SOURCE_REPLACEMENT_ATTEMPTS:
                    break
                candidate_url = candidate["href"]
                if candidate_url in tried_urls:
                    continue
                tried_urls.add(candidate_url)
                attempts += 1
                candidate_text = self._scrape(candidate_url)
                candidates.remove(candidate)
                if self._usable_page(candidate_text):
                    replacement = candidate
                    replacement_text = candidate_text
                    break
            if replacement:
                selected[index] = replacement
                selected_urls.add(replacement["href"])
                scraped[replacement["href"]] = replacement_text
                self._log(
                    "stage=replace_source | "
                    f"old={_clip(source_url, 100)} | new={_clip(replacement['href'], 100)}"
                )
                continue
            snippet = "\n".join(filter(None, [
                source.get("title", ""), source.get("body", ""),
            ])).strip()
            if snippet:
                scraped[source_url] = snippet
                self._log(
                    f"stage=source_snippet_fallback | url={_clip(source_url, 120)}"
                )

    def _replace_empty_extractions(self, query: str, selected: list[dict],
                                   results: list[dict], scraped: dict[str, str],
                                   pages: list[NormalPageEvidence],
                                   aspects: list[ResearchAspect] | None = None) -> None:
        selected_urls = {result.get("href", "") for result in selected}
        candidates = [
            result for result in self._rank_results(
                self.last_intent or self._analyze_intent(query), results
            )
            if result.get("href") not in selected_urls
            and not _is_low_quality_host(
                urlparse(result.get("href", "")).hostname or ""
            )
        ]
        replacements = 0
        priorities = {
            aspect.name: aspect.priority for aspect in (aspects or [])
        }
        page_indices = sorted(
            range(len(pages)),
            key=lambda index: priorities.get(selected[index].get("_aspect_name", ""), 3),
            reverse=True,
        )
        for index in page_indices:
            page = pages[index]
            if page.facts or replacements >= MAX_POST_EXTRACTION_REPLACEMENTS:
                continue
            source = selected[index]
            same_query = [
                candidate for candidate in candidates
                if candidate.get("_plan_query") == source.get("_plan_query")
            ]
            candidate = next(iter(same_query or candidates), None)
            if not candidate:
                continue
            content = self._scrape(candidate["href"])
            if not self._usable_page(content):
                continue
            aspect = next((
                item for item in (aspects or [])
                if item.name == source.get("_aspect_name")
            ), self._aspect_for_query_index(source.get("_aspect_index", -1)))
            if aspect:
                candidate["_aspect_name"] = aspect.name
                candidate["_aspect_index"] = source.get("_aspect_index", -1)
            replacement_page = self._extract_normal_page(query, candidate, content)
            replacements += 1
            if not replacement_page.facts:
                self._log(
                    f"stage=replace_empty_failed | url={_clip(candidate['href'], 120)}"
                )
                continue
            old_url = source.get("href", "")
            selected[index] = candidate
            scraped[candidate["href"]] = content
            pages[index] = replacement_page
            candidates.remove(candidate)
            self._log(
                "stage=replace_empty | "
                f"old={_clip(old_url, 100)} | new={_clip(candidate['href'], 100)}"
            )

    def _run_normal(self, query: str, results: list[dict]) -> str:
        intent = self.last_intent or self._analyze_intent(query)
        selected = self._rank_results(intent, results)[:NORMAL_SOURCES]
        self._log(
            "stage=select_sources | "
            + " | ".join(
                f"rank={index + 1},url={_clip(result.get('href', ''), 120)}"
                for index, result in enumerate(selected)
            )
        )
        scraped = self._fetch_pages(selected)
        pages: list[NormalPageEvidence | None] = [None] * len(selected)
        with ThreadPoolExecutor(max_workers=len(selected)) as executor:
            futures = {
                executor.submit(
                    self._extract_normal_page, query, result,
                    scraped[result["href"]],
                ): index
                for index, result in enumerate(selected)
            }
            for future in as_completed(futures):
                pages[futures[future]] = future.result()
        pages = [
            page if page is not None
            else NormalPageEvidence(facts=[], insufficient_information=True)
            for page in pages
        ]
        synthesis = DeepSynthesis(
            facts=[
                DeepFact(
                    claim=fact.claim,
                    source_ids=[source_id],
                    published_at=fact.published_at,
                    metric=fact.metric,
                    unit=fact.unit,
                    period=fact.period,
                    geography=fact.geography,
                    definition=fact.definition,
                )
                for source_id, page in enumerate(pages, start=1)
                for fact in page.facts
            ][:MAX_DEEP_FACTS],
            insufficient_information=not any(page.facts for page in pages),
        )
        self._store_result(query, SearchMode.NORMAL, selected, synthesis)
        return self._format_normal_results(selected, pages)

    def _fallback_deep_synthesis(self, pages: list[NormalPageEvidence]) -> DeepSynthesis:
        facts: list[DeepFact] = []
        for source_id, page in enumerate(pages, start=1):
            for fact in page.facts:
                facts.append(DeepFact(
                    claim=fact.claim,
                    source_ids=[source_id],
                    published_at=_clean_published_at(fact.published_at),
                    metric=fact.metric,
                    unit=fact.unit,
                    period=fact.period,
                    geography=fact.geography,
                    definition=fact.definition,
                ))
                if len(facts) >= MAX_DEEP_FACTS:
                    return DeepSynthesis(facts=facts)
        return DeepSynthesis(facts=facts, insufficient_information=not facts)

    def _synthesize_deep(self, question: str,
                         pages: list[NormalPageEvidence]) -> DeepSynthesis:
        aspects = list(self.last_plan.research_aspects if self.last_plan else [])
        aspect_contracts = [aspect.model_dump(mode="json") for aspect in aspects]
        material = json.dumps([
            {
                "source_id": source_id,
                "bound_aspect": page.aspect_name,
                "facts": [fact.model_dump() for fact in page.facts],
            }
            for source_id, page in enumerate(pages, start=1)
        ], ensure_ascii=False)
        prompt = (
            f"Верифицируй факты-кандидаты для исследовательского вопроса: {question}\n\n"
            "Кандидаты получены от меньшей модели извлечения и недоверенны. "
            "Используй только информацию из кандидатов.\n\n"
            "Правила верификации:\n"
            "1. Отклоняй размытые, касательные, неподтверждённые, устаревшие "
            "или неоднозначные факты.\n"
            "2. Страница может подтверждать только свой bound_aspect.\n"
            "3. Верни один aspect_review на каждый контракт: confirmed (нужен "
            "минимум один принятый факт с валидным source_id), rejected или "
            "missing.\n"
            f"4. Верни не более {MAX_DEEP_FACTS} принятых фактов. Объединяй "
            "дубликаты, сохраняй все source_id и даты.\n"
            "5. Конфликт валиден только при совместимых метрике, единице, "
            "периоде, географии и определении. Разные даты или определения "
            "сами по себе не конфликт. Валидные расхождения — в "
            "conflict_details.\n"
            "6. Source ID начинаются с 1.\n\n"
            f"Контракты аспектов: {json.dumps(aspect_contracts, ensure_ascii=False)}\n\n"
            f"Кандидаты доказательств: {material}"
        )
        try:
            attempts = 1
            if self._budget:
                attempts = min(
                    2,
                    max(1, self._budget.max_llm_calls - self._budget.llm_calls),
                )
            synthesis = self._structured(
                prompt,
                DeepSynthesis,
                max_attempts=attempts,
                model=self.deep_planner_model,
                stage="synthesis",
            )
        except (StructuredOutputError, SearchBudgetExceeded):
            self._log("stage=synthesis_rejected | reason=verification_failed")
            return DeepSynthesis(insufficient_information=True)
        valid_facts = []
        for fact in synthesis.facts[:MAX_DEEP_FACTS]:
            source_ids = sorted({
                source_id for source_id in fact.source_ids
                if 1 <= source_id <= len(pages)
                and self._fact_supported_by_candidates(fact, pages[source_id - 1])
            })
            if fact.claim and source_ids and not NEGATIVE_EVIDENCE.search(fact.claim):
                valid_facts.append(DeepFact(
                    claim=_clip(fact.claim, 300),
                    source_ids=source_ids,
                    published_at=_clean_published_at(fact.published_at),
                    metric=_clip(fact.metric, 80),
                    unit=_clip(fact.unit, 40),
                    period=_clip(fact.period, 60),
                    geography=_clip(fact.geography, 80),
                    definition=_clip(fact.definition, 120),
                ))
        if not valid_facts and any(page.facts for page in pages):
            self._log("stage=synthesis_rejected | reason=no_verified_facts")
        valid_conflicts = [
            _clip(item.description, 240)
            for item in synthesis.conflict_details[:4]
            if self._valid_conflict(item, len(pages))
        ]
        return DeepSynthesis(
            facts=valid_facts,
            conflicts=valid_conflicts,
            aspect_reviews=synthesis.aspect_reviews,
            insufficient_information=synthesis.insufficient_information or not valid_facts,
        )

    def _fact_supported_by_candidates(self, fact: DeepFact,
                                      page: NormalPageEvidence) -> bool:
        claim_terms = self._quality_terms(fact.claim)
        for candidate in page.facts:
            candidate_terms = self._quality_terms(
                f"{candidate.claim} {candidate.evidence}"
            )
            required = 1 if len(claim_terms) <= 3 else 2
            if len(claim_terms & candidate_terms) >= required:
                return True
        return False

    @staticmethod
    def _valid_conflict(item: ConflictAssessment, source_count: int) -> bool:
        source_ids = {value for value in item.source_ids if 1 <= value <= source_count}
        if len(source_ids) < 2 or not item.metric.strip():
            return False
        if re.search(
            r"different (?:dates?|years?|periods?)|разн(?:ые|ых) (?:даты|годы|периоды)",
            item.description,
            re.IGNORECASE,
        ):
            return False
        return bool(
            item.unit.strip() and item.period.strip()
            and item.geography.strip() and item.definition.strip()
        )

    def _format_deep_results(self, results: list[dict],
                             synthesis: DeepSynthesis,
                             coverage_gaps: list[str] | None = None) -> str:
        lines = ["Deep web evidence:", "Sources:"]
        for source_id, result in enumerate(results, start=1):
            host = urlparse(result.get("href", "")).hostname or ""
            official = _is_authoritative_host(host)
            year = self._source_year(result)
            metadata = f"official={'yes' if official else 'no'}"
            if year:
                metadata += f", year={year}"
            lines.append(
                f"[{source_id}] {_clip(result.get('title', ''), 120)} | {metadata}\n"
                f"    {_clip(result.get('href', ''), 180)}"
            )
        lines.append("\nVerified facts:")
        for fact in synthesis.facts[:MAX_DEEP_FACTS]:
            citations = ",".join(str(source_id) for source_id in fact.source_ids)
            date = f" ({fact.published_at})" if fact.published_at else ""
            lines.append(f"- {_clip(fact.claim, 220)}{date} [{citations}]")
        if synthesis.conflicts:
            lines.append("\nConflicts:")
            lines.extend(f"- {_clip(conflict, 220)}" for conflict in synthesis.conflicts[:4])
        if coverage_gaps:
            lines.append("\nCoverage gaps:")
            lines.extend(f"- {_clip(aspect, 120)}" for aspect in coverage_gaps)
        if synthesis.insufficient_information and not synthesis.facts:
            lines.append("- Insufficient supported information.")
        formatted = "\n".join(lines)
        if len(formatted) > MAX_FORMATTED_RESULT_CHARS:
            return formatted[:MAX_FORMATTED_RESULT_CHARS - 1] + "…"
        return formatted

    def _coverage_gaps(self, aspects: list[str],
                       pages: list[NormalPageEvidence]) -> list[str]:
        evidence_terms = set()
        for page in pages:
            for fact in page.facts:
                evidence_terms.update(self._quality_terms(
                    f"{fact.claim} {fact.evidence}"
                ))
        return [
            aspect for aspect in aspects
            if self._quality_terms(aspect)
            and not (self._quality_terms(aspect) & evidence_terms)
        ]

    def _aspect_outcomes(self, aspects: list[ResearchAspect],
                         selected: list[dict],
                         pages: list[NormalPageEvidence]) -> list[AspectOutcome]:
        outcomes: list[AspectOutcome] = []
        for aspect in aspects:
            matches = [
                (source_id, page)
                for source_id, (source, page) in enumerate(zip(selected, pages), start=1)
                if source.get("_aspect_name") == aspect.name
            ]
            confirmed = next((
                source_id for source_id, page in matches
                if page.answers_aspect and bool(page.facts)
            ), None)
            if confirmed is not None:
                outcome = AspectOutcome(
                    name=aspect.name,
                    status=AspectStatus.CONFIRMED,
                    source_id=confirmed,
                )
            elif matches:
                reason = next((
                    page.rejection_reason for _, page in matches
                    if page.rejection_reason
                ), "selected sources did not satisfy the evidence contract")
                outcome = AspectOutcome(
                    name=aspect.name,
                    status=AspectStatus.REJECTED,
                    failure_reason=_clip(reason, 180),
                )
            else:
                outcome = AspectOutcome(
                    name=aspect.name,
                    status=AspectStatus.MISSING,
                    failure_reason="no source was selected for this aspect",
                )
            outcomes.append(outcome)
            self._log(
                f"stage=aspect | name={_clip(outcome.name, 60)} | "
                f"status={outcome.status.value} | source={outcome.source_id or '-'} | "
                f"reason={_clip(outcome.failure_reason, 100) or '-'}"
            )
        return outcomes

    def _reviewed_aspect_outcomes(self, aspects: list[ResearchAspect],
                                  pages: list[NormalPageEvidence],
                                  synthesis: DeepSynthesis) -> list[AspectOutcome]:
        accepted_sources = {
            source_id
            for fact in synthesis.facts
            for source_id in fact.source_ids
        }
        reviews = {item.name.casefold(): item for item in synthesis.aspect_reviews}
        outcomes: list[AspectOutcome] = []
        for aspect in aspects:
            review = reviews.get(aspect.name.casefold())
            bound_sources = {
                source_id
                for source_id, page in enumerate(pages, start=1)
                if page.aspect_name == aspect.name
            }
            confirmed_sources = sorted(
                set(review.source_ids if review else [])
                & bound_sources
                & accepted_sources
            )
            requested_status = (review.status.casefold() if review else "missing")
            if requested_status == AspectStatus.CONFIRMED.value and confirmed_sources:
                outcome = AspectOutcome(
                    name=aspect.name,
                    status=AspectStatus.CONFIRMED,
                    source_id=confirmed_sources[0],
                )
            else:
                has_candidates = any(
                    page.facts for page in pages if page.aspect_name == aspect.name
                )
                status = AspectStatus.REJECTED if has_candidates else AspectStatus.MISSING
                reason = review.reason if review else "PCC returned no aspect assessment"
                outcome = AspectOutcome(
                    name=aspect.name,
                    status=status,
                    failure_reason=_clip(
                        reason or "no candidate passed PCC verification", 180
                    ),
                )
            outcomes.append(outcome)
            self._log(
                f"stage=aspect | verifier=pcc | name={_clip(outcome.name, 60)} | "
                f"status={outcome.status.value} | source={outcome.source_id or '-'} | "
                f"reason={_clip(outcome.failure_reason, 100) or '-'}"
            )
        return outcomes

    def _run_deep(self, query: str, results: list[dict]) -> str:
        intent = self.last_intent or self._analyze_intent(query)
        aspect_names = self.last_plan.aspects if self.last_plan else [query]
        research_aspects = list(
            self.last_plan.research_aspects if self.last_plan else []
        )
        if not research_aspects:
            research_aspects = [
                ResearchAspect(
                    name=item.aspect or item.query,
                    query=item.query,
                    expected_evidence=item.expected_evidence,
                    requirement=item.requirement or item.aspect,
                    priority=item.priority,
                    requires_freshness=item.requires_freshness,
                    preferred_source_type=item.preferred_source_type,
                    acceptance_criteria=item.acceptance_criteria,
                )
                for item in (self.last_plan.queries if self.last_plan else [])
            ] or [ResearchAspect(name=query, query=query, requirement=query)]
        selected = self._select_deep_sources(intent, results, aspect_names)
        self._log(
            "stage=select_sources | mode=deep | "
            + " | ".join(
                f"rank={index + 1},url={_clip(result.get('href', ''), 120)}"
                for index, result in enumerate(selected)
            )
        )
        scraped = self._fetch_pages(selected)
        self._replace_unreadable_sources(selected, results, scraped)
        started = time.monotonic()
        pages: list[NormalPageEvidence | None] = [None] * len(selected)
        workers = min(DEEP_EXTRACTION_WORKERS, len(selected))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self._extract_normal_page,
                    query,
                    result,
                    scraped[result["href"]],
                ): index
                for index, result in enumerate(selected)
            }
            for future in as_completed(futures):
                pages[futures[future]] = future.result()
        extracted = [
            page if page is not None else NormalPageEvidence(
                facts=[],
                insufficient_information=True,
            )
            for page in pages
        ]
        self._replace_empty_extractions(
            query, selected, results, scraped, extracted, research_aspects
        )
        self._log(
            f"stage=extract_pages | mode=deep | pages={len(extracted)} | "
            f"workers={workers} | elapsed={time.monotonic() - started:.2f}s"
        )
        synthesis = self._synthesize_deep(query, extracted)
        outcomes = self._reviewed_aspect_outcomes(
            research_aspects, extracted, synthesis
        )
        coverage_gaps = [
            item.name for item in outcomes if item.status != AspectStatus.CONFIRMED
        ]
        if coverage_gaps:
            self._log(
                "stage=coverage_gaps | aspects="
                + ",".join(_clip(aspect, 60) for aspect in coverage_gaps)
            )
        self._store_result(
            query, SearchMode.DEEP, selected, synthesis, coverage_gaps, outcomes
        )
        return self._format_deep_results(selected, synthesis, coverage_gaps)

    def execute(self, query: str, depth: str = "auto") -> str:
        mode = self._select_mode(query, depth)
        initial_mode = mode
        intent = self._analyze_intent(query)
        budget = SearchBudget.for_mode(mode)
        self._budget = budget
        self.last_plan = None
        self.last_result = None
        self._log(
            f"start | mode={mode.value} | max_llm_calls={budget.max_llm_calls} | "
            f"deadline={budget.timeout_seconds:.0f}s | requested_depth={depth} | "
            f"forced_depth={self.force_depth or '-'} | query={_clip(query, 160)}"
        )
        if mode in {SearchMode.NORMAL, SearchMode.DEEP}:
            self.last_plan, intent = self._plan_research(query, mode)
        self.last_intent = intent
        self.last_query = intent.search_query()
        self._log(
            f"stage=intent | expected={intent.expected_value.value} | "
            f"fresh={str(intent.requires_freshness).lower()} | "
            f"official={intent.official_domain or '-'} | currency={intent.currency or '-'} | "
            f"search_query={_clip(self.last_query, 180)}"
        )

        try:
            if mode == SearchMode.QUICK:
                results = self._search(self.last_query)
                if not results:
                    self.last_result = ResearchResult(
                        query=query, mode=mode.value, insufficient_information=True
                    )
                    return "Ничего не найдено."
                ranked = self._rank_quick_results(query, results)
                synthesis = DeepSynthesis(facts=[
                    DeepFact(
                        claim=_clip(result.get("body", "") or result.get("title", ""), 300),
                        source_ids=[source_id],
                    )
                    for source_id, result in enumerate(ranked, start=1)
                    if result.get("body") or result.get("title")
                ])
                self._store_result(query, SearchMode.QUICK, ranked, synthesis)
                return self._format_quick_results(query, results)

            if mode == SearchMode.NORMAL:
                results = self._search_many(
                    self.last_plan.search_queries
                    if self.last_plan else [self.last_query]
                )
                if not results:
                    self.last_result = ResearchResult(
                        query=query, mode=mode.value, insufficient_information=True
                    )
                    return "Ничего не найдено."
                return self._run_normal(query, results)

            search_queries = (
                self.last_plan.search_queries
                if self.last_plan else [self.last_query]
            )
            results = self._search_many(search_queries)
            if not results:
                self.last_result = ResearchResult(
                    query=query, mode=mode.value, insufficient_information=True
                )
                return "Ничего не найдено."
            return self._run_deep(query, results)
        except SearchBudgetExceeded as error:
            self._log(f"stopped | reason={error}")
            return f"Поиск остановлен: {error}"
        finally:
            self.last_stats = {
                "mode": mode.value,
                "initial_mode": initial_mode.value,
                "escalated": initial_mode != mode,
                "llm_calls": budget.llm_calls,
                "elapsed": round(budget.elapsed, 3),
                "max_llm_calls": budget.max_llm_calls,
                "requested_depth": depth,
                "forced_depth": self.force_depth,
                "expected_value": intent.expected_value.value,
                "requires_freshness": intent.requires_freshness,
                "official_domain": intent.official_domain,
                "plan_aspects": list(self.last_plan.aspects) if self.last_plan else [],
                "plan_queries": list(self.last_plan.search_queries) if self.last_plan else [],
                "aspect_statuses": dict(self.last_result.aspect_statuses)
                if self.last_result else {},
                "broad_conclusion_allowed": self.last_result.broad_conclusion_allowed
                if self.last_result else None,
            }
            agg = self._aggregate
            agg["total"] += 1
            if initial_mode == SearchMode.QUICK and mode == SearchMode.QUICK:
                agg["quick"] += 1
            elif initial_mode == SearchMode.QUICK and mode != SearchMode.QUICK:
                agg["escalated"] += 1
            if mode == SearchMode.NORMAL:
                agg["normal"] += 1
            elif mode == SearchMode.DEEP:
                agg["deep"] += 1
            if agg["total"] % 10 == 0 and self.logger:
                self._log(
                    f"aggregate | total={agg['total']} | "
                    f"quick_resolved={agg['quick']} | "
                    f"quick_escalated={agg['escalated']} | "
                    f"normal={agg['normal']} | deep={agg['deep']}"
                )
            self._log(
                f"end | mode={mode.value} | llm_calls={budget.llm_calls}/"
                f"{budget.max_llm_calls} | elapsed={budget.elapsed:.2f}s"
            )
            self._budget = None


if __name__ == "__main__":
    from core.config import APPLE_BASE_URL, APPLE_LOCAL_MODEL

    client = OpenAI(base_url=APPLE_BASE_URL, api_key="apple-local")
    tool = WebSearchTool(client, APPLE_LOCAL_MODEL, model_mini=APPLE_LOCAL_MODEL)
    print(tool.execute("Что такое абоба?"))
