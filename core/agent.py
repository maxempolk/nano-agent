from __future__ import annotations
from openai import OpenAI, BadRequestError
from typing import TYPE_CHECKING
import json

from core.tools import bash
from core.llm import call_llm

if TYPE_CHECKING:
    from core.logger import SessionLogger

MAX_TOOL_CALLS_PER_TURN = 20


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n... [обрезано {len(text) - max_chars} символов] ...\n" + text[-half:]


class Agent:
    def __init__(self, client: OpenAI, model: str, system: str,
                 context_window: int = 10, max_tool_output: int = 2000,
                 logger: SessionLogger | None = None,
                 extra_tools: list | None = None):
        self.client = client
        self.model = model
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

    def run_turn(self, user_input: str, on_tool_call=None) -> str:
        self.messages.append({"role": "user", "content": user_input})
        if self.logger:
            self.logger.user(user_input)

        self.last_search_query: str | None = None
        tool_calls_made = 0

        while True:
            history = self.messages[1:]
            windowed = [self.messages[0]] + history[-self.context_window:]

            try:
                response = call_llm(self.client, self.model, windowed, self.tools)  # type: ignore
            except BadRequestError as e:
                if "tool_use_failed" in str(e):
                    response = call_llm(self.client, self.model, windowed)  # type: ignore
                else:
                    raise
            except Exception as e:
                error_reply = f"Ошибка API: {e}"
                if self.logger:
                    self.logger.error(error_reply)
                self.messages.append({"role": "assistant", "content": error_reply})
                return error_reply

            message = response.choices[0].message

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
                self.messages.append({"role": "assistant", "content": reply})
                if self.logger:
                    self.logger.agent(reply)
                return reply
