from __future__ import annotations
from openai import OpenAI, BadRequestError, RateLimitError
from typing import TYPE_CHECKING
import json

from core.tools import bash
from core.llm import call_llm

if TYPE_CHECKING:
    from core.logger import SessionLogger

MAX_TOOL_CALLS_PER_TURN = 20
DEFAULT_TOKEN_BUDGET = 5500
CHARS_PER_TOKEN = 4       # грубая оценка: 1 токен ≈ 4 символа
COMPRESSED_TOOL_CHARS = 400  # до скольки сжимать старые tool-результаты


def _estimate_tokens(messages: list) -> int:
    total = 0
    for m in messages:
        content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        total += len(str(content or ""))
    return total // CHARS_PER_TOKEN


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n... [обрезано {len(text) - max_chars} символов] ...\n" + text[-half:]


class Agent:
    def __init__(self, client: OpenAI, model: str, system: str,
                 context_window: int = 10, max_tool_output: int = 2000,
                 logger: SessionLogger | None = None,
                 extra_tools: list | None = None,
                 model_fallback: str | None = None,
                 token_budget: int = DEFAULT_TOKEN_BUDGET):
        self.client = client
        self.model = model
        self.model_fallback = model_fallback
        self.token_budget = token_budget
        self.context_window = context_window
        self.max_tool_output = max_tool_output
        self.logger = logger
        self.messages: list = [{"role": "system", "content": system}]

        self.tools = [bash.SCHEMA]
        self.handlers: dict = {"execute_bash": bash.execute}
        self.tool_objects: dict = {}
        for tool in (extra_tools or []):
            self.tools.append(tool.SCHEMA)  # type: ignore
            name = tool.SCHEMA["function"]["name"]  # type: ignore
            self.handlers[name] = tool.execute
            self.tool_objects[name] = tool

    def _compress_context(self, windowed: list) -> list:
        if _estimate_tokens(windowed) <= self.token_budget:
            return windowed

        # Найти индекс последнего user-сообщения — начало текущего хода
        last_user_idx = 0
        for i, msg in enumerate(windowed):
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
            if role == "user":
                last_user_idx = i

        # Сжать tool-результаты из старых ходов (до текущего)
        result = []
        saved_chars = 0
        for i, msg in enumerate(windowed):
            if (i < last_user_idx
                    and isinstance(msg, dict)
                    and msg.get("role") == "tool"):
                content = msg.get("content", "") or ""
                if len(content) > COMPRESSED_TOOL_CHARS:
                    new_msg = dict(msg)
                    new_msg["content"] = content[:COMPRESSED_TOOL_CHARS] + " ... [сжато]"
                    saved_chars += len(content) - COMPRESSED_TOOL_CHARS
                    result.append(new_msg)
                    continue
            result.append(msg)

        if self.logger and saved_chars > 0:
            self.logger.info(
                f"Контекст сжат: -{saved_chars} символов "
                f"(~{saved_chars // CHARS_PER_TOKEN} токенов), "
                f"итого ~{_estimate_tokens(result)} токенов"
            )
        return result

    def run_turn(self, user_input: str, on_tool_call=None) -> str:
        self.messages.append({"role": "user", "content": user_input})
        if self.logger:
            self.logger.user(user_input)

        self.last_search_query: str | None = None
        tool_calls_made = 0

        while True:
            history = self.messages[1:]
            windowed = [self.messages[0]] + history[-self.context_window:]
            windowed = self._compress_context(windowed)

            try:
                response = call_llm(self.client, self.model, windowed, self.tools)  # type: ignore
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
                        response = call_llm(self.client, self.model_fallback, windowed, self.tools)  # type: ignore
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
                if tool_calls_made >= MAX_TOOL_CALLS_PER_TURN:
                    reply = "Достигнут лимит вызовов инструментов за один ход. Остановился."
                    if self.logger:
                        self.logger.error(reply)
                    self.messages.append({"role": "assistant", "content": reply})
                    return reply

                self.messages.append(message)  # type: ignore
                for call in message.tool_calls:
                    args = json.loads(call.function.arguments)  # type: ignore
                    handler = self.handlers.get(call.function.name)
                    if not handler:
                        result = f"Неизвестный инструмент: {call.function.name}"
                    else:
                        try:
                            result = _truncate(handler(**args), self.max_tool_output)
                        except TypeError as e:
                            result = f"Ошибка вызова инструмента {call.function.name}: {e}"

                    if call.function.name == "web_search":
                        tool_obj = self.tool_objects.get("web_search")
                        self.last_search_query = getattr(tool_obj, "last_query", args.get("query"))

                    if self.logger:
                        self.logger.tool_call(call.function.name, call.function.arguments)
                        self.logger.tool_result(result)
                    if on_tool_call:
                        on_tool_call(call.function.name, call.function.arguments, result)

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
