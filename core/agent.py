from __future__ import annotations
from openai import OpenAI, BadRequestError, RateLimitError
from typing import TYPE_CHECKING, Callable
import json
import re

from core.tools import bash
from core.llm import call_llm

if TYPE_CHECKING:
    from core.logger import SessionLogger
    from core.model_router import RouteDecision

MAX_TOOL_CALLS_PER_TURN = 20
DEFAULT_TOKEN_BUDGET = 5500
COMPACT_TRIGGER_RATIO = 0.8
CHARS_PER_TOKEN = 3
COMPRESSED_TOOL_CHARS = 400  # до скольки сжимать старые tool-результаты
MAX_SUMMARY_CHARS = 1400
DEFAULT_COMPACT_PROMPT = (
    "Суммируй транскрипт для будущих ходов. Сохрани цели, решения, факты, "
    "результаты действий, ошибки и незавершённую работу. Убери повторы. Не выдумывай."
)

EXPLICIT_WEB_SEARCH = re.compile(
    r"\b(загугл\w*|поищ\w*|ищи в сети|проверь в (?:сети|интернете)|"
    r"найди в (?:сети|интернете)|search online|search the web|browse the web|"
    r"google it|look it up)\b",
    re.IGNORECASE,
)
GENERIC_SEARCH_FOLLOWUP = re.compile(
    r"^\s*(поищи(?: в сети)?|загугли|проверь в (?:сети|интернете)|"
    r"search online|search the web|google it|look it up)[.!?\s]*$",
    re.IGNORECASE,
)
CHANGING_WEB_FACT = re.compile(
    r"(?:\b(?:последн\w*|latest|newest)\b.{0,35}\b(?:верси\w*|модел\w*|"
    r"релиз\w*|новост\w*|gpt|iphone|айфон\w*)\b|"
    r"\b(?:верси\w*|модел\w*|релиз\w*|gpt|iphone|айфон\w*)\b.{0,35}"
    r"\b(?:последн\w*|latest|newest)\b|"
    r"\b(?:кто|who)\b.{0,40}\b(?:сейча\w*|current)\b.{0,40}"
    r"\b(?:президент\w*|president|ceo)\b|"
    r"\b(?:курс|погода|weather|exchange rate|stock price)\b|"
    r"\b(?:цена|цене|стоимость|price|cost)\b.{0,30}"
    r"\b(?:битк\w*|bitcoin|btc|акци\w*|stock|iphone|айфон\w*)\b|"
    r"\b(?:битк\w*|bitcoin|btc)\b.{0,30}"
    r"\b(?:цена|цене|стоимость|курс|price|cost)\b|"
    r"\bтоп\b.{0,20}\b(?:стран\w*|country|countries)\b)",
    re.IGNORECASE,
)
DEEP_SEARCH_INTENT = re.compile(
    r"\b(подробн\w*|глубок\w*|исслед\w*|сравни\w*|обзор\w*|"
    r"deep research|in-depth|compare|comparison|research)\b",
    re.IGNORECASE,
)
FORCED_DEEP_WEB_SEARCH = re.compile(
    r"(?:\b(?:подробн\w*|глубок\w*)\b.{0,30}\b(?:исслед\w*|изуч\w*|"
    r"проанализ\w*)\b|\bисследуй\b|\bсравни\w*\b.{0,45}\bисточник\w*\b|"
    r"\bdeep research\b|\bin-depth research\b)",
    re.IGNORECASE,
)
FACTUAL_QUESTION = re.compile(
    r"^\s*(?:кто|что|какой|какая|какое|какие|сколько|где|когда|чем|чей|чья|чьё|"
    r"which|what|who|where|when|how many|how much)\b",
    re.IGNORECASE,
)


def _message_dict(message) -> dict:
    if isinstance(message, dict):
        return message
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    return {
        "role": getattr(message, "role", "unknown"),
        "content": getattr(message, "content", ""),
    }


def _estimate_tokens(messages: list, tools: list | None = None) -> int:
    payload: dict = {"messages": [_message_dict(m) for m in messages]}
    if tools:
        payload["tools"] = tools
    chars = len(json.dumps(payload, ensure_ascii=False, default=str))
    return max(1, chars // CHARS_PER_TOKEN)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n... [обрезано {len(text) - max_chars} символов] ...\n" + text[-half:]


def _forced_web_search_query(user_input: str, previous_user_input: str | None = None) -> str | None:
    if GENERIC_SEARCH_FOLLOWUP.match(user_input):
        return previous_user_input or user_input
    if (
        EXPLICIT_WEB_SEARCH.search(user_input)
        or CHANGING_WEB_FACT.search(user_input)
        or FORCED_DEEP_WEB_SEARCH.search(user_input)
    ):
        return user_input
    return None


def _forced_web_search_depth(user_input: str) -> str:
    return "deep" if FORCED_DEEP_WEB_SEARCH.search(user_input) else "auto"


def _without_tool(tools: list, name: str) -> list:
    return [
        tool for tool in tools
        if tool.get("function", {}).get("name") != name
    ]


def _tool_names(tools: list) -> set[str]:
    return {
        name
        for tool in tools
        if (name := tool.get("function", {}).get("name"))
    }


SIMPLE_SYSTEM = (
    "Ты — полезный ассистент. Отвечай кратко и точно на языке пользователя. "
    "Если не уверен — скажи об этом."
)

_FINALIZER_QUICK_SYSTEM = (
    "Напиши ответ на вопрос по предоставленным сниппетам.\n"
    "Язык ответа = язык вопроса.\n"
    "Извлеки лучший доступный ответ из сниппетов, даже если данные неполные. "
    "Не пиши «невозможно ответить», если сниппеты содержат релевантную информацию — "
    "приведи то, что есть, с оговоркой о неполноте.\n"
    "Начни с прямого ответа. Укажи источники."
)

_FINALIZER_RESEARCH_SYSTEM = (
    "Напиши ответ на вопрос по предоставленным фактам.\n"
    "Язык ответа = язык вопроса.\n"
    "1. Начни с прямого ответа.\n"
    "2. Если факты противоречат друг другу — укажи оба варианта с источниками.\n"
    "3. Если исследование частичное (Broad conclusion allowed: no) — "
    "скажи об этом и перечисли, чего не хватает.\n"
    "4. Укажи источники."
)

_FINALIZER_EXAMPLE_USER = (
    "Вопрос:\nкакая последняя версия Python?\n\n"
    "Сниппеты:\n"
    "[1] Python 3.14.6 released (python.org)\n"
    "Python 3.14.6 is the latest stable release, published June 2026."
)
_FINALIZER_EXAMPLE_ASSISTANT = (
    "Последняя стабильная версия — Python 3.14.6 (июнь 2026). "
    "Источник: python.org"
)

_FINALIZER_EXAMPLE_USER_IMPERFECT = (
    "Вопрос:\nкакой курс биткоина?\n\n"
    "Сниппеты:\n"
    "[1] Bitcoin price today (coinmarketcap.com)\n"
    "BTC to USD exchange rate is $63,928.25. Growth of 3.4% in 24 hours.\n"
    "[2] Курс Bitcoin (binance.com)\n"
    "Текущая цена биткоина: $66,847.92 за 1 BTC."
)
_FINALIZER_EXAMPLE_ASSISTANT_IMPERFECT = (
    "Курс биткоина: ~$63 900–66 800 (данные разнятся по источникам). "
    "Источники: coinmarketcap.com, binance.com"
)


def _finalization_messages(user_input: str,
                           evidence: list[tuple[str, str]],
                           quick: bool = False) -> list[dict]:
    evidence_text = "\n\n".join(
        result for _, result in evidence
    )
    system = _FINALIZER_QUICK_SYSTEM if quick else _FINALIZER_RESEARCH_SYSTEM
    evidence_label = "Сниппеты" if quick else "Факты"
    messages: list[dict] = [
        {"role": "system", "content": system},
    ]
    if re.search(r"[а-яё]", user_input, re.IGNORECASE):
        messages.extend([
            {"role": "user", "content": _FINALIZER_EXAMPLE_USER},
            {"role": "assistant", "content": _FINALIZER_EXAMPLE_ASSISTANT},
            {"role": "user", "content": _FINALIZER_EXAMPLE_USER_IMPERFECT},
            {"role": "assistant", "content": _FINALIZER_EXAMPLE_ASSISTANT_IMPERFECT},
        ])
    messages.append({
        "role": "user",
        "content": (
            f"Вопрос:\n{user_input}\n\n"
            f"{evidence_label}:\n{evidence_text or 'Нет данных.'}"
        ),
    })
    return messages


def _tool_evidence_fallback(evidence: list[tuple[str, str]],
                            structured_result=None) -> str:
    if structured_result is not None and hasattr(structured_result, "render_fallback"):
        rendered = structured_result.render_fallback()
        if rendered.strip():
            return rendered
    if not evidence:
        return "Не удалось сформировать ответ: модель вернула пустой результат."
    name, result = evidence[-1]
    result = result.strip()
    if not result:
        return f"Инструмент {name} завершился без результата."
    return (
        "Не удалось сформировать итоговый текст модели. Ниже — результат "
        f"инструмента {name}:\n\n{result}"
    )


def _validate_final_answer(content: str, user_input: str,
                           require_partial: bool = False) -> tuple[bool, str]:
    text = content.strip()
    if not text:
        return False, "empty"
    if text.startswith("```"):
        return False, "code_fence"
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, (dict, list)):
        return False, "json"
    if require_partial and not re.search(
        r"частич|недостаточ|не удалось|пробел|отсутств|partial|insufficient|missing|gap|could not",
        text,
        re.IGNORECASE,
    ):
        return False, "partial_coverage_hidden"
    if re.search(r"[а-яё]", user_input, re.IGNORECASE):
        prose = re.sub(r"https?://\S+", "", text)
        prose = re.sub(r"\b[A-Z0-9][A-Z0-9._-]*\b", "", prose)
        cyrillic_words = re.findall(r"[а-яё]{2,}", prose, re.IGNORECASE)
        latin_words = re.findall(r"[A-Za-z]{3,}", prose)
        if len(cyrillic_words) < 3 and len(latin_words) > len(cyrillic_words):
            return False, "language_mismatch"
    return True, "ok"


class Agent:
    def __init__(self, client: OpenAI, model: str, system: str,
                 compact_keep_messages: int = 10, max_tool_output: int = 2000,
                 logger: SessionLogger | None = None,
                 extra_tools: list | None = None,
                 model_fallback: str | None = None,
                 token_budget: int = DEFAULT_TOKEN_BUDGET,
                 compact_prompt: str = DEFAULT_COMPACT_PROMPT,
                 compact_trigger_ratio: float = COMPACT_TRIGGER_RATIO,
                 route_selector: Callable[[str], RouteDecision] | None = None,
                 compact_model: str | None = None):
        self.client = client
        self.model = model
        self.model_fallback = model_fallback
        self.token_budget = token_budget
        self.compact_keep_messages = compact_keep_messages
        self.max_tool_output = max_tool_output
        self.logger = logger
        self.compact_prompt = compact_prompt
        self.compact_model = compact_model or model
        self.compact_trigger_ratio = compact_trigger_ratio
        self.route_selector = route_selector
        self.base_system = system
        self.memory = ""
        self.messages: list = [{"role": "system", "content": self.base_system}]
        self.last_search_query: str | None = None
        self.last_route_name = "local" if model == "system" else model
        self.last_route_reason = "fixed model"
        self.last_route_score = 0

        self.tools = [bash.SCHEMA]
        self.handlers: dict = {"execute_bash": bash.execute}
        self.tool_objects: dict = {}
        for tool in (extra_tools or []):
            self.tools.append(tool.SCHEMA)  # type: ignore
            name = tool.SCHEMA["function"]["name"]  # type: ignore
            self.handlers[name] = tool.execute
            self.tool_objects[name] = tool

    def clear_context(self) -> None:
        self.memory = ""
        self.messages = [{"role": "system", "content": self.base_system}]
        self.last_search_query = None
        if self.logger:
            self.logger.info("Контекст очищен")

    def _select_route(self, user_input: str) -> None:
        if not self.route_selector:
            return
        decision = self.route_selector(user_input)
        route = decision.route
        self.model = route.model
        self.model_fallback = route.fallback_model
        self.token_budget = route.token_budget
        self.base_system = route.system
        self.messages[0] = {"role": "system", "content": self.base_system}
        self.last_route_name = route.name
        self.last_route_reason = decision.reason
        self.last_route_score = decision.score
        if self.logger:
            self.logger.info(
                f"route={route.name} | model={route.model} | score={decision.score} | "
                f"reason={decision.reason} | context={route.token_budget}"
            )

    def context_usage(self) -> tuple[int, int]:
        used = _estimate_tokens(self._context_messages(), self.tools)
        return used, self.token_budget

    def compact_context(self) -> tuple[int, int, bool]:
        before, _ = self.context_usage()
        compacted = self._compact_if_needed(force=True)
        after, _ = self.context_usage()
        return before, after, compacted

    def _render_transcript(self, messages: list) -> str:
        rows = []
        for message in messages:
            data = _message_dict(message)
            role = data.get("role", "unknown")
            content = data.get("content") or ""
            tool_calls = data.get("tool_calls") or []
            if tool_calls:
                calls = []
                for call in tool_calls:
                    function = call.get("function", {}) if isinstance(call, dict) else {}
                    calls.append(
                        f"{function.get('name', 'tool')}({function.get('arguments', '')})"
                    )
                content = f"{content}\nTool calls: {'; '.join(calls)}".strip()
            rows.append(f"[{role}] {content}")
        return "\n\n".join(rows)

    def _context_messages(self) -> list:
        system = self.base_system
        if self.memory:
            system += f"\n\nConversation memory:\n{self.memory}"
        return [{"role": "system", "content": system}, *self.messages[1:]]

    def _shrink_tool_results(self) -> None:
        tool_indices = [
            i for i, message in enumerate(self.messages)
            if isinstance(message, dict) and message.get("role") == "tool"
        ]
        for i in tool_indices[:-1]:
            content = self.messages[i].get("content", "") or ""
            if len(content) > COMPRESSED_TOOL_CHARS:
                self.messages[i] = dict(self.messages[i])
                self.messages[i]["content"] = _truncate(content, COMPRESSED_TOOL_CHARS)

    def _compact_if_needed(self, force: bool = False) -> bool:
        before = _estimate_tokens(self._context_messages(), self.tools)
        trigger = int(self.token_budget * self.compact_trigger_ratio)
        if not force and before < trigger:
            return False

        user_indices = [
            i for i, message in enumerate(self.messages)
            if _message_dict(message).get("role") == "user"
        ]
        if not user_indices:
            return False

        last_user_idx = user_indices[-1]
        if force:
            retain_start = last_user_idx
        else:
            lower_bound = max(1, last_user_idx - self.compact_keep_messages)
            recent_users = [i for i in user_indices if lower_bound <= i < last_user_idx]
            retain_start = recent_users[0] if recent_users else last_user_idx
        old_messages = self.messages[1:retain_start]
        if not old_messages:
            if not force and before >= self.token_budget:
                self._shrink_tool_results()
                after = _estimate_tokens(self._context_messages(), self.tools)
                return after < before
            return False

        transcript_parts = []
        if self.memory:
            transcript_parts.append(f"[memory] {self.memory}")
        transcript_parts.append(self._render_transcript(old_messages))
        transcript = "\n\n".join(transcript_parts)
        compact_messages = [
            {"role": "system", "content": self.compact_prompt},
            {"role": "user", "content": transcript},
        ]
        try:
            response = call_llm(self.client, self.compact_model, compact_messages)
            summary = (response.choices[0].message.content or "").strip()
            if not summary:
                raise ValueError("модель вернула пустое резюме")
            summary = _truncate(summary, MAX_SUMMARY_CHARS)
            self.memory = summary
            self.messages = [self.messages[0], *self.messages[retain_start:]]
        except Exception as e:
            if self.logger:
                self.logger.error(f"Compact failed: {e}")
            self.memory = _truncate(transcript, MAX_SUMMARY_CHARS)
            self.messages = [self.messages[0], *self.messages[retain_start:]]

        if _estimate_tokens(self._context_messages(), self.tools) >= self.token_budget:
            self._shrink_tool_results()

        after = _estimate_tokens(self._context_messages(), self.tools)
        if self.logger and after < before:
            self.logger.info(
                f"Контекст compact: ~{before} → ~{after} токенов, "
                f"записей={len(self.messages)}"
            )
        return True

    def _finalize_research(self, user_input: str,
                           evidence: list[tuple[str, str]]) -> str:
        web_tool = self.tool_objects.get("web_search")
        structured_result = (
            getattr(web_tool, "last_result", None)
            if any(name == "web_search" for name, _ in evidence)
            else None
        )
        is_quick = getattr(structured_result, "mode", "") == "quick"
        messages = _finalization_messages(user_input, evidence, quick=is_quick)
        require_partial = bool(
            structured_result is not None
            and not getattr(structured_result, "broad_conclusion_allowed", True)
        )
        models = list(dict.fromkeys(filter(None, [self.model_fallback, self.model])))
        for attempt, model in enumerate(models, start=1):
            if self.logger:
                self.logger.info(
                    f"finalizer | start | attempt={attempt}/{len(models)} | model={model}"
                )
            try:
                response = call_llm(self.client, model, messages)
                message = response.choices[0].message
                finish_reason = response.choices[0].finish_reason
                content = (message.content or "").strip()
                tool_calls = len(message.tool_calls) if message.tool_calls else 0
                if self.logger:
                    self.logger.info(
                        f"finalizer | end | model={model} | finish_reason={finish_reason} | "
                        f"tool_calls={tool_calls} | content_len={len(content)}"
                    )
                valid, reason = _validate_final_answer(
                    content, user_input, require_partial=require_partial
                )
                if valid and not message.tool_calls:
                    return content
                if self.logger:
                    self.logger.error(
                        f"finalizer rejected | model={model} | reason="
                        f"{'tool_calls' if message.tool_calls else reason}"
                    )
            except Exception as error:
                if self.logger:
                    self.logger.error(f"finalizer failed | model={model} | error={error}")

        if self.logger:
            self.logger.error("finalizer | deterministic_fallback")
        return _tool_evidence_fallback(evidence, structured_result)

    def run_turn(self, user_input: str, on_tool_call=None) -> str:
        self._select_route(user_input)
        previous_user_input = next(
            (
                _message_dict(message).get("content")
                for message in reversed(self.messages[1:])
                if _message_dict(message).get("role") == "user"
            ),
            None,
        )
        self.messages.append({"role": "user", "content": user_input})
        if self.logger:
            self.logger.user(user_input)

        self.last_search_query: str | None = None
        tool_calls_made = 0
        turn_tools = self.tools
        search_completed = False
        turn_evidence: list[tuple[str, str]] = []

        search_query = _forced_web_search_query(user_input, previous_user_input)
        web_handler = self.handlers.get("web_search")
        if search_query and web_handler:
            args = {
                "query": search_query,
                "depth": _forced_web_search_depth(user_input),
            }
            arguments = json.dumps(args, ensure_ascii=False)
            call_id = f"forced-web-search-{len(self.messages)}"
            if self.logger:
                self.logger.info(f"forced web_search | query={search_query}")
            try:
                result = _truncate(web_handler(**args), self.max_tool_output)
            except Exception as error:
                result = f"Ошибка вызова инструмента web_search: {error}"

            tool_obj = self.tool_objects.get("web_search")
            self.last_search_query = getattr(tool_obj, "last_query", search_query)
            if self.logger:
                self.logger.tool_call("web_search", arguments)
                self.logger.tool_result(result)
            if on_tool_call:
                on_tool_call("web_search", arguments, result)
            structured_result = getattr(tool_obj, "last_result", None)
            evidence_result = (
                structured_result.evidence_text()
                if structured_result is not None
                and hasattr(structured_result, "evidence_text")
                else result
            )
            turn_evidence.append(("web_search", evidence_result))

            self.messages.extend([
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "web_search", "arguments": arguments},
                    }],
                },
                {"role": "tool", "tool_call_id": call_id, "content": result},
            ])
            tool_calls_made = 1
            search_completed = True
            # После гарантированного поиска этому ходу больше не нужны tools:
            # компактная AFM иначе пытается повторять web_search или curl через bash.
            turn_tools = []

        simple_mode = (
            self.route_selector is not None
            and not search_completed
            and self.last_route_score == 0
        )
        if simple_mode:
            self.messages[0] = {"role": "system", "content": SIMPLE_SYSTEM}
            turn_tools = []
            if self.logger:
                self.logger.info("simple_mode | minimal prompt, no tools")

        while True:
            self._compact_if_needed()
            windowed = self._context_messages()

            if search_completed:
                reply = self._finalize_research(user_input, turn_evidence)
                self.messages.append({"role": "assistant", "content": reply})
                if self.logger:
                    self.logger.agent(reply)
                return reply

            try:
                response = call_llm(self.client, self.model, windowed, turn_tools)  # type: ignore
            except BadRequestError as e:
                if "tool_use_failed" in str(e):
                    if self.logger:
                        self.logger.error(f"tool_use_failed (tool_calls={tool_calls_made}), retry without tools")
                    response = call_llm(self.client, self.model, windowed)  # type: ignore
                else:
                    if self.logger:
                        self.logger.error(f"BadRequestError: {e}")
                    raise
            except RateLimitError as e:
                if self.model_fallback:
                    if self.logger:
                        self.logger.error(f"RateLimitError на {self.model}, переключаюсь на {self.model_fallback}: {e}")
                    try:
                        response = call_llm(self.client, self.model_fallback, windowed, turn_tools)  # type: ignore
                    except Exception as e2:
                        error_reply = f"Ошибка API (fallback): {e2}"
                        if self.logger:
                            self.logger.error(error_reply)
                        self.messages.append({"role": "assistant", "content": error_reply})
                        return error_reply
                else:
                    error_reply = f"Ошибка API: {e}"
                    if self.logger:
                        self.logger.error(error_reply)
                    self.messages.append({"role": "assistant", "content": error_reply})
                    return error_reply
            except Exception as e:
                error_reply = f"Ошибка API: {e}"
                if self.logger:
                    self.logger.error(error_reply)
                self.messages.append({"role": "assistant", "content": error_reply})
                return error_reply

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            if self.logger:
                self.logger.info(
                    f"finish_reason={finish_reason} | "
                    f"tool_calls={len(message.tool_calls) if message.tool_calls else 0} | "
                    f"content_len={len(message.content or '')}"
                )

            if message.tool_calls:
                allowed_tools = _tool_names(turn_tools)
                forbidden_calls = [
                    call for call in message.tool_calls
                    if (
                        call.function.name not in allowed_tools
                        or (call.function.name == "web_search" and search_completed)
                    )
                ]
                if forbidden_calls:
                    names = ",".join(call.function.name for call in forbidden_calls)
                    if self.logger:
                        self.logger.error(
                            "tool protocol violation | "
                            f"returned={names} | allowed={','.join(sorted(allowed_tools)) or '-'} | "
                            f"search_completed={str(search_completed).lower()}"
                        )
                    reply = self._finalize_research(user_input, turn_evidence)
                    self.messages.append({"role": "assistant", "content": reply})
                    if self.logger:
                        self.logger.error(reply)
                    return reply

                if tool_calls_made >= MAX_TOOL_CALLS_PER_TURN:
                    reply = "Достигнут лимит вызовов инструментов за один ход. Остановился."
                    if self.logger:
                        self.logger.error(reply)
                    self.messages.append({"role": "assistant", "content": reply})
                    return reply

                self.messages.append(message)  # type: ignore
                first_web_call_id = next(
                    (
                        call.id
                        for call in message.tool_calls
                        if call.function.name == "web_search"
                    ),
                    None,
                )
                if first_web_call_id:
                    # Один пакет AFM может содержать несколько поисков и curl.
                    # Даже невалидный первый вызов не должен открывать новый
                    # цикл инструментов.
                    turn_tools = []
                for call in message.tool_calls:
                    if tool_calls_made >= MAX_TOOL_CALLS_PER_TURN:
                        result = "Пропущено: достигнут лимит инструментов за один ход."
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result,
                        })
                        continue

                    if first_web_call_id and call.id != first_web_call_id:
                        result = (
                            "Пропущено: в пакете с web_search выполняется только "
                            "первый поиск; остальные инструменты заблокированы."
                        )
                        if self.logger:
                            self.logger.info(
                                f"tool skipped after batched web_search | "
                                f"name={call.function.name}"
                            )
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result,
                        })
                        continue

                    try:
                        args = json.loads(call.function.arguments or "{}")  # type: ignore
                        if not isinstance(args, dict):
                            raise ValueError("arguments must be a JSON object")
                    except (json.JSONDecodeError, ValueError) as error:
                        args = {}
                        result = (
                            f"Ошибка аргументов инструмента {call.function.name}: {error}"
                        )
                        if self.logger:
                            self.logger.info(
                                f"invalid tool arguments | name={call.function.name}"
                            )
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result,
                        })
                        tool_calls_made += 1
                        continue
                    if (
                        call.function.name == "web_search"
                        and args.get("depth") == "deep"
                        and not DEEP_SEARCH_INTENT.search(user_input)
                    ):
                        args["depth"] = "normal"
                        if self.logger:
                            self.logger.info(
                                "web_search depth downgraded deep→normal: "
                                "no explicit deep intent in user message"
                            )
                    handler = self.handlers.get(call.function.name)
                    if not handler:
                        result = f"Неизвестный инструмент: {call.function.name}"
                    else:
                        try:
                            result = _truncate(handler(**args), self.max_tool_output)
                        except TypeError as e:
                            result = f"Ошибка вызова инструмента {call.function.name}: {e}"

                    if call.function.name == "web_search":
                        search_completed = True
                        tool_obj = self.tool_objects.get("web_search")
                        self.last_search_query = getattr(tool_obj, "last_query", args.get("query"))
                        # После поиска AFM должна сформировать ответ из результата,
                        # а не повторять поиск или открывать URL через bash.
                        turn_tools = []

                    if self.logger:
                        self.logger.tool_call(
                            call.function.name,
                            json.dumps(args, ensure_ascii=False),
                        )
                        self.logger.tool_result(result)
                    if on_tool_call:
                        on_tool_call(
                            call.function.name,
                            json.dumps(args, ensure_ascii=False),
                            result,
                        )
                    evidence_result = result
                    if call.function.name == "web_search":
                        structured_result = getattr(tool_obj, "last_result", None)
                        if structured_result is not None and hasattr(
                            structured_result, "evidence_text"
                        ):
                            evidence_result = structured_result.evidence_text()
                    turn_evidence.append((call.function.name, evidence_result))

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result
                    })
                    tool_calls_made += 1
            else:
                reply = message.content or ""
                if not reply and self.logger:
                    self.logger.error(f"Пустой ответ от модели (finish_reason={finish_reason})")
                self.messages.append({"role": "assistant", "content": reply})
                if self.logger:
                    self.logger.agent(reply)
                return reply
