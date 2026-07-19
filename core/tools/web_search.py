from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import math
import re
from threading import Lock
import time
from typing import TYPE_CHECKING, TypeVar
from urllib.parse import urlparse

from ddgs import DDGS
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
import trafilatura

try:
    from newspaper import Article as NewspaperArticle
    _NEWSPAPER_OK = True
except ImportError:
    _NEWSPAPER_OK = False

from core.llm import call_llm

if TYPE_CHECKING:
    from core.logger import SessionLogger

MAX_RESULTS = 10
MAX_RETRIES = 2
MAX_FORMATTED_RESULT_CHARS = 1950
QUICK_RESULTS = 5
NORMAL_SOURCES = 2
DEEP_SOURCES = 4
DEEP_EXTRACTION_WORKERS = 2
MAX_DEEP_FACTS = 8
PAGE_CONTEXT_CHARS = 7_000
LOCAL_CONTEXT_LIMIT = 8_192
LLM_INPUT_TOKEN_BUDGET = 6_000
CONSERVATIVE_CHARS_PER_TOKEN = 1.5
MESSAGE_TOKEN_OVERHEAD = 128

MODE_LIMITS = {
    "quick": (0, 15.0),
    "normal": (2, 45.0),
    "deep": (5, 90.0),
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
    "norway": "ssb.no",
    "норвегия": "ssb.no",
    "норвегии": "ssb.no",
}

OFFICIAL_SOURCE_HINTS = {
    ("ssb.no", "number"): (
        "administrative divisions",
        "regionale-inndelingar",
    ),
}

TANGENTIAL_SOURCE_TERMS = {
    "accounts", "finance", "health", "service", "housing", "categories",
    "population by size", "municipal accounts", "kommuneregnskap",
}

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
GPT_VERSION = re.compile(r"\bgpt[-\s]?(\d+)(?:[._](\d+))?\b", re.IGNORECASE)
BTC_QUERY = re.compile(r"\b(btc|bitcoin|биткоин\w*)\b", re.IGNORECASE)
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
NORWAY_MUNICIPALITY_QUERY = re.compile(
    r"(?=.*\b(норвеги\w*|norway)\b)(?=.*\b(коммун\w*|municipalit\w*)\b)",
    re.IGNORECASE,
)
NORWAY_LIVING_STANDARD_QUERY = re.compile(
    r"(?=.*\b(норвеги\w*|norway)\b)(?=.*\b(уровень\s+жизни|living standards?|quality of life)\b)",
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


class NormalPageEvidence(BaseModel):
    facts: list[NormalFact]
    insufficient_information: bool


class DeepFact(BaseModel):
    claim: str
    source_ids: list[int] = Field(default_factory=list)
    published_at: str = ""


class DeepSynthesis(BaseModel):
    facts: list[DeepFact] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    insufficient_information: bool = False


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


@dataclass(frozen=True)
class SearchIntent:
    original_query: str
    normalized_query: str
    expected_value: ExpectedValue
    requires_freshness: bool
    official_requested: bool
    official_domain: str | None
    currency: str | None = None

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


@dataclass(frozen=True)
class QuickQuality:
    sufficient: bool
    score: int
    relevant_results: int
    value_present: bool
    fresh_present: bool
    authoritative_present: bool
    reasons: tuple[str, ...]


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


class WebSearchTool:
    SCHEMA = SCHEMA

    def __init__(self, client: OpenAI, model: str, model_mini: str | None = None,
                 logger: SessionLogger | None = None,
                 force_depth: str | None = None):
        if force_depth not in {None, "quick", "normal", "deep"}:
            raise ValueError("force_depth должен быть quick, normal, deep или None")
        self.client = client
        self.model = model
        self.model_mini = model_mini or model
        self.logger = logger
        self.force_depth = force_depth
        self.last_query = ""
        self.last_stats: dict = {}
        self.last_intent: SearchIntent | None = None
        self._budget: SearchBudget | None = None

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger.info(f"web_search | {message}")

    def _call_model(self, messages: list, stage: str):
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
            return call_llm(self.client, self.model_mini, messages)
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
                    max_attempts: int = MAX_RETRIES) -> StructuredModel:
        schema = json.dumps(_flat_json_schema(output_type), ensure_ascii=False, separators=(",", ":"))
        suffix = f"\n\nReturn ONLY one valid JSON object matching this JSON Schema:\n{schema}"
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
                max_payload_chars - len(suffix) - len(retry_note) - 100,
            )
            fitted_prompt = _clip_middle(prompt, prompt_limit)
            if fitted_prompt != prompt:
                self._log(
                    f"stage=input_trim | attempt={attempt + 1} | "
                    f"original_chars={len(prompt)} | fitted_chars={len(fitted_prompt)}"
                )
            response = self._call_model(
                [{"role": "user", "content": fitted_prompt + suffix + retry_note}],
                "structured",
            )
            raw = response.choices[0].message.content or ""
            if not raw.strip():
                self._log("stage=structured_empty | retry=false")
                raise StructuredOutputError(raw, ValueError("empty model response"))
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
                        "\n\nYour previous response was invalid. Correct the JSON and return the "
                        f"object only. Validation error: {_clip(str(error), 500)}\n"
                        f"Invalid response: {_clip(raw, 1200)}"
                    )

        raise StructuredOutputError(raw, last_error)

    def _scrape_trafilatura(self, url: str) -> str:
        try:
            downloaded = trafilatura.fetch_url(url)
            return trafilatura.extract(downloaded) or ""
        except Exception:
            return ""

    def _scrape_newspaper(self, url: str) -> str:
        if not _NEWSPAPER_OK:
            return ""
        try:
            article = NewspaperArticle(url)
            article.download()
            article.parse()
            return article.text or ""
        except Exception:
            return ""

    def _scrape(self, url: str) -> str:
        text = self._scrape_trafilatura(url)
        if len(text) < 200:
            text = self._scrape_newspaper(url) or text
        return text or "Не удалось извлечь текст."

    def _search(self, query: str) -> list[dict]:
        started = time.monotonic()
        with DDGS() as ddg:
            results = list(ddg.text(query, max_results=MAX_RESULTS))
        if self._budget:
            self._budget.check_deadline()
        self._log(
            f"stage=search | results={len(results)} | "
            f"elapsed={time.monotonic() - started:.2f}s"
        )
        return results

    def _official_domain(self, query: str) -> str | None:
        words = set(re.findall(r"[\w-]+", query.lower()))
        domains = {
            domain
            for word, domain in OFFICIAL_DOMAIN_HINTS.items()
            if word in words
        }
        return next(iter(domains)) if len(domains) == 1 else None

    def _translate_known_russian_terms(self, query: str) -> str:
        translated = query.lower()
        replacements = (
            (r"\bофициальн\w*\s+статистик\w*\b", "official statistics"),
            (r"\bсколько\b", "number of"),
            (r"\bколичество\b", "number of"),
            (r"\bкоммун\w*\b", "municipalities"),
            (r"\bнорвеги\w*\b", "Norway"),
            (r"\bсейчас\b", "current"),
            (r"\bсегодня\b", "today"),
            (r"\bактуальн\w*\b", "current"),
            (r"\bтекущ\w*\b", "current"),
        )
        changes = 0
        for pattern, replacement in replacements:
            translated, count = re.subn(pattern, replacement, translated, flags=re.IGNORECASE)
            changes += count
        if changes < 2:
            return query
        translated = re.sub(r"\s+", " ", translated)
        return translated.strip(" ?!.,")

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

        freshness = bool(CURRENT_QUERY.search(query)) or expected in {
            ExpectedValue.PRICE,
            ExpectedValue.WEATHER,
        }
        normalized = self._translate_known_russian_terms(query)

        if BTC_QUERY.search(query) and expected == ExpectedValue.PRICE:
            currency = currency or "USD"
            normalized = f"BTC {currency} live price"
            freshness = True
        elif "gpt" in lowered and LATEST_QUERY.search(query):
            normalized = "latest GPT model OpenAI"
            freshness = True
        elif NORWAY_MUNICIPALITY_QUERY.search(query):
            normalized = "current number of municipalities Norway official statistics"
            freshness = True
            expected = ExpectedValue.NUMBER
        elif NORWAY_LIVING_STANDARD_QUERY.search(query):
            normalized = "Norway standard of living quality of life latest statistics"
            freshness = True
        elif expected == ExpectedValue.WEATHER:
            location = re.search(
                r"(?:погод\w*|температур\w*)(?:\s+сейчас)?\s+в\s+(.+)",
                query,
                re.IGNORECASE,
            )
            if location:
                normalized = f"current weather {location.group(1).strip(' ?!.,')}"
                freshness = True

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

    def _rank_results(self, intent: SearchIntent, results: list[dict]) -> list[dict]:
        official_domain = intent.official_domain
        wants_latest = intent.requires_freshness and intent.expected_value == ExpectedValue.VERSION
        query_terms = self._quality_terms(intent.normalized_query)
        normalized_lower = intent.normalized_query.lower()

        def score(item: tuple[int, dict]) -> tuple[int, int]:
            index, result = item
            host = urlparse(result.get("href", "")).hostname or ""
            title = result.get("title", "").lower()
            body = result.get("body", "").lower()
            url = result.get("href", "").lower()
            official = int(bool(official_domain and host.endswith(official_domain)))
            trusted = int(host.endswith(".gov") or host.endswith(".edu"))
            title_terms = self._result_terms({"title": title, "body": "", "href": ""})
            body_terms = self._result_terms({"title": "", "body": body, "href": ""})
            overlap = len(query_terms & title_terms) * 12
            overlap += len(query_terms & body_terms) * 3
            direct_value = int(self._contains_expected_value(
                intent,
                f"{result.get('title', '')} {result.get('body', '')}",
            ))
            fresh = int(self._contains_fresh_marker(f"{title} {body}"))
            source_hint = any(
                hint in f"{title} {url}"
                for hint in OFFICIAL_SOURCE_HINTS.get(
                    (official_domain or "", intent.expected_value.value),
                    (),
                )
            )
            tangential = any(
                term in f"{title} {url}"
                and term not in normalized_lower
                for term in TANGENTIAL_SOURCE_TERMS
            )
            version_score = 0
            if wants_latest:
                versions = [
                    int(major) * 10 + int(minor or 0)
                    for major, minor in GPT_VERSION.findall(f"{title} {body}")
                ]
                version_score = max(versions, default=0)
            total = (
                official * 200
                + trusted * 40
                + int(source_hint) * 90
                + direct_value * 55
                + fresh * 20
                + version_score
                + overlap
                - int(tangential) * 60
            )
            return total, -index

        ranked = sorted(enumerate(results), key=score, reverse=True)
        return [result for _, result in ranked]

    def _rank_quick_results(self, query: str, results: list[dict]) -> list[dict]:
        intent = self._analyze_intent(query)
        return self._rank_results(intent, results)[:QUICK_RESULTS]

    def _quality_terms(self, query: str) -> set[str]:
        stop_words = {
            "the", "and", "for", "with", "from", "what", "which", "current",
            "latest", "live", "official", "statistics", "number", "price",
            "сколько", "количество", "сейчас", "актуальная", "официальная",
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
                GPT_VERSION.search(text)
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

    def _assess_quick_quality(self, intent: SearchIntent,
                              results: list[dict]) -> QuickQuality:
        ranked = self._rank_quick_results(intent.original_query, results)
        query_terms = self._quality_terms(intent.normalized_query)
        official_domain = intent.official_domain
        relevant = 0
        value_present = False
        fresh_present = False
        authoritative_present = False

        for result in ranked[:QUICK_RESULTS]:
            overlap = len(query_terms & self._result_terms(result))
            required_overlap = 1 if len(query_terms) <= 2 else 2
            if overlap >= required_overlap:
                relevant += 1
            text = f"{result.get('title', '')} {result.get('body', '')}"
            value_present = value_present or self._contains_expected_value(intent, text)
            fresh_present = fresh_present or self._contains_fresh_marker(text)
            host = urlparse(result.get("href", "")).hostname or ""
            authoritative_present = authoritative_present or bool(
                (official_domain and host.endswith(official_domain))
                or host.endswith(".gov")
                or host.endswith(".edu")
            )

        score = min(relevant, 3) * 10
        if value_present:
            score += 30
        if fresh_present:
            score += 20
        if authoritative_present:
            score += 20

        needs_value = intent.expected_value != ExpectedValue.FACT
        sufficient = relevant >= 1 and (value_present or not needs_value)
        if intent.requires_freshness:
            sufficient = sufficient and (fresh_present or authoritative_present) and score >= 60
        else:
            sufficient = sufficient and score >= 40

        reasons = []
        if relevant < 1:
            reasons.append("no_relevant_results")
        if needs_value and not value_present:
            reasons.append("expected_value_missing")
        if intent.requires_freshness and not (fresh_present or authoritative_present):
            reasons.append("freshness_unconfirmed")
        if not reasons and not sufficient:
            reasons.append("low_score")
        return QuickQuality(
            sufficient=sufficient,
            score=score,
            relevant_results=relevant,
            value_present=value_present,
            fresh_present=fresh_present,
            authoritative_present=authoritative_present,
            reasons=tuple(reasons),
        )

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
                 for major, minor in GPT_VERSION.findall(lowered)),
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
                             content: str) -> NormalPageEvidence:
        intent = self.last_intent or self._analyze_intent(question)
        context = self._select_relevant_passages(question, content, result)
        self._log(
            f"stage=select_passages | url={_clip(result.get('href', ''), 120)} | "
            f"source_chars={len(content)} | selected_chars={len(context)}"
        )
        prompt = (
            "Extract up to 3 facts that directly answer the question. Each claim must have a "
            "short supporting excerpt from the provided page. Do not use outside knowledge. "
            "Set published_at to the publication/update date supporting the fact, or an empty "
            "string when absent. Set insufficient_information=true when the page cannot answer "
            "the question. For a number question, extract only the total count of the requested "
            "subject; reject counts of categories, people, accounts, years or related objects. "
            f"Expected value type: {intent.expected_value.value}. "
            f"Current information required: {str(intent.requires_freshness).lower()}.\n\n"
            f"Question: {question}\nTitle: {result.get('title', '')}\n"
            f"URL: {result.get('href', '')}\n\nPage passages:\n{context}"
        )
        try:
            evidence = self._structured(prompt, NormalPageEvidence, max_attempts=1)
        except StructuredOutputError as error:
            evidence = self._recover_normal_evidence(error.raw)
        except SearchBudgetExceeded:
            return NormalPageEvidence(facts=[], insufficient_information=True)
        return NormalPageEvidence(
            facts=[
                NormalFact(
                    claim=_clip(fact.claim, 300),
                    evidence=_clip(fact.evidence, 240),
                    published_at=_clip(fact.published_at, 40),
                )
                for fact in evidence.facts[:3]
                if self._fact_matches_intent(intent, fact)
            ],
            insufficient_information=evidence.insufficient_information,
        )

    def _fact_matches_intent(self, intent: SearchIntent, fact: NormalFact) -> bool:
        text = f"{fact.claim} {fact.evidence}"
        if not self._contains_expected_value(intent, text):
            return False
        years = [int(year) for year in re.findall(r"\b20\d{2}\b", fact.published_at)]
        if intent.requires_freshness and years and max(years) < datetime.now().year - 1:
            return False
        return True

    def _recover_normal_evidence(self, raw: str) -> NormalPageEvidence:
        facts: list[NormalFact] = []
        try:
            value = _json_object(raw)
        except (ValueError, json.JSONDecodeError):
            value = {}

        for item in value.get("facts", []):
            if not isinstance(item, dict):
                continue
            candidate = item
            nested = item.get("$defs", {}).get("NormalFact")
            if isinstance(nested, dict):
                candidate = nested
            claim = candidate.get("claim")
            evidence = candidate.get("evidence")
            if isinstance(claim, str) and isinstance(evidence, str):
                facts.append(NormalFact(
                    claim=claim,
                    evidence=evidence,
                    published_at=candidate.get("published_at", "")
                    if isinstance(candidate.get("published_at", ""), str) else "",
                ))

        if not facts:
            pair = re.compile(
                r'"claim"\s*:\s*("(?:\\.|[^"\\])*")\s*,\s*'
                r'"evidence"\s*:\s*("(?:\\.|[^"\\])*")'
            )
            for claim_raw, evidence_raw in pair.findall(raw):
                try:
                    facts.append(NormalFact(
                        claim=json.loads(claim_raw),
                        evidence=json.loads(evidence_raw),
                    ))
                except json.JSONDecodeError:
                    continue

        recovered = facts[:3]
        if recovered:
            self._log(f"stage=structured_recovered | facts={len(recovered)}")
        return NormalPageEvidence(
            facts=recovered,
            insufficient_information=not recovered,
        )

    def _format_normal_results(self, results: list[dict],
                               pages: list[NormalPageEvidence]) -> str:
        lines = ["Web evidence:"]
        for source_id, (result, page) in enumerate(zip(results, pages), start=1):
            lines.extend([
                f"[{source_id}] {_clip(result.get('title', ''), 160)}",
                f"URL: {_clip(result.get('href', ''), 220)}",
                f"Official: {'yes' if self.last_intent and self.last_intent.official_domain and (urlparse(result.get('href', '')).hostname or '').endswith(self.last_intent.official_domain) else 'no'}",
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
        with ThreadPoolExecutor(max_workers=len(urls)) as executor:
            futures = {executor.submit(self._scrape, url): url for url in urls}
            for future in as_completed(futures):
                scraped[futures[future]] = future.result()
        if self._budget:
            self._budget.check_deadline()
        self._log(
            f"stage=fetch | pages={len(urls)} | elapsed={time.monotonic() - started:.2f}s"
        )
        return scraped

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
        pages = [
            self._extract_normal_page(query, result, scraped[result["href"]])
            for result in selected
        ]
        return self._format_normal_results(selected, pages)

    def _fallback_deep_synthesis(self, pages: list[NormalPageEvidence]) -> DeepSynthesis:
        facts: list[DeepFact] = []
        for source_id, page in enumerate(pages, start=1):
            for fact in page.facts:
                facts.append(DeepFact(
                    claim=fact.claim,
                    source_ids=[source_id],
                    published_at=fact.published_at,
                ))
                if len(facts) >= MAX_DEEP_FACTS:
                    return DeepSynthesis(facts=facts)
        return DeepSynthesis(facts=facts, insufficient_information=not facts)

    def _synthesize_deep(self, question: str,
                         pages: list[NormalPageEvidence]) -> DeepSynthesis:
        material = json.dumps([
            {
                "source_id": source_id,
                "facts": [fact.model_dump() for fact in page.facts],
            }
            for source_id, page in enumerate(pages, start=1)
        ], ensure_ascii=False)
        prompt = (
            f"Verify and combine evidence for this research question: {question}\n"
            f"Return at most {MAX_DEEP_FACTS} concise facts. Merge duplicates, preserve every "
            "supporting source_id, keep dates, and list genuine source disagreements in "
            "conflicts. Do not add outside knowledge. Source IDs are 1-based.\n\n"
            f"Evidence: {material}"
        )
        try:
            synthesis = self._structured(prompt, DeepSynthesis, max_attempts=1)
        except (StructuredOutputError, SearchBudgetExceeded):
            return self._fallback_deep_synthesis(pages)
        valid_facts = []
        for fact in synthesis.facts[:MAX_DEEP_FACTS]:
            source_ids = sorted({
                source_id for source_id in fact.source_ids
                if 1 <= source_id <= len(pages)
            })
            if fact.claim and source_ids:
                valid_facts.append(DeepFact(
                    claim=_clip(fact.claim, 300),
                    source_ids=source_ids,
                    published_at=_clip(fact.published_at, 40),
                ))
        return DeepSynthesis(
            facts=valid_facts,
            conflicts=[_clip(conflict, 240) for conflict in synthesis.conflicts[:4]],
            insufficient_information=synthesis.insufficient_information or not valid_facts,
        )

    def _format_deep_results(self, results: list[dict],
                             synthesis: DeepSynthesis) -> str:
        lines = ["Deep web evidence:", "Sources:"]
        for source_id, result in enumerate(results, start=1):
            host = urlparse(result.get("href", "")).hostname or ""
            official = bool(
                self.last_intent
                and self.last_intent.official_domain
                and host.endswith(self.last_intent.official_domain)
            )
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
        if synthesis.insufficient_information and not synthesis.facts:
            lines.append("- Insufficient supported information.")
        formatted = "\n".join(lines)
        if len(formatted) > MAX_FORMATTED_RESULT_CHARS:
            return formatted[:MAX_FORMATTED_RESULT_CHARS - 1] + "…"
        return formatted

    def _run_deep(self, query: str, results: list[dict]) -> str:
        intent = self.last_intent or self._analyze_intent(query)
        selected = self._rank_results(intent, results)[:DEEP_SOURCES]
        self._log(
            "stage=select_sources | mode=deep | "
            + " | ".join(
                f"rank={index + 1},url={_clip(result.get('href', ''), 120)}"
                for index, result in enumerate(selected)
            )
        )
        scraped = self._fetch_pages(selected)
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
        self._log(
            f"stage=extract_pages | mode=deep | pages={len(extracted)} | "
            f"workers={workers} | elapsed={time.monotonic() - started:.2f}s"
        )
        synthesis = self._synthesize_deep(query, extracted)
        return self._format_deep_results(selected, synthesis)

    def execute(self, query: str, depth: str = "auto") -> str:
        mode = self._select_mode(query, depth)
        initial_mode = mode
        intent = self._analyze_intent(query)
        budget = SearchBudget.for_mode(mode)
        self._budget = budget
        self.last_intent = intent
        self.last_query = intent.search_query()
        self._log(
            f"start | mode={mode.value} | max_llm_calls={budget.max_llm_calls} | "
            f"deadline={budget.timeout_seconds:.0f}s | requested_depth={depth} | "
            f"forced_depth={self.force_depth or '-'} | query={_clip(query, 160)}"
        )
        self._log(
            f"stage=intent | expected={intent.expected_value.value} | "
            f"fresh={str(intent.requires_freshness).lower()} | "
            f"official={intent.official_domain or '-'} | currency={intent.currency or '-'} | "
            f"search_query={_clip(self.last_query, 180)}"
        )
        quick_quality: QuickQuality | None = None

        try:
            if mode == SearchMode.QUICK:
                results = self._search(self.last_query)
                if not results:
                    return "Ничего не найдено."
                quality = self._assess_quick_quality(intent, results)
                quick_quality = quality
                self._log(
                    f"stage=quick_quality | sufficient={str(quality.sufficient).lower()} | "
                    f"score={quality.score} | relevant={quality.relevant_results} | "
                    f"value={str(quality.value_present).lower()} | "
                    f"fresh={str(quality.fresh_present).lower()} | "
                    f"authoritative={str(quality.authoritative_present).lower()} | "
                    f"reasons={','.join(quality.reasons) or '-'}"
                )
                can_escalate = depth == "auto" and self.force_depth is None
                if not quality.sufficient and can_escalate:
                    mode = SearchMode.NORMAL
                    normal_budget = SearchBudget.for_mode(mode)
                    normal_budget.started_at = budget.started_at
                    budget = normal_budget
                    self._budget = budget
                    self._log(
                        "stage=escalate | from=quick | to=normal | "
                        f"reason={','.join(quality.reasons) or 'low_score'}"
                    )
                    return self._run_normal(query, results)
                return self._format_quick_results(query, results)

            if mode == SearchMode.NORMAL:
                results = self._search(self.last_query)
                if not results:
                    return "Ничего не найдено."
                return self._run_normal(query, results)

            results = self._search(self.last_query)
            if not results:
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
                "quick_quality_score": quick_quality.score if quick_quality else None,
                "quick_quality_reasons": list(quick_quality.reasons) if quick_quality else [],
            }
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
