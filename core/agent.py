from __future__ import annotations
from openai import OpenAI
from typing import TYPE_CHECKING
import json
import subprocess

if TYPE_CHECKING:
    from core.logger import SessionLogger

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Execute a bash command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"]
            }
        }
    }
]

MAX_TOOL_CALLS_PER_TURN = 20


def _execute_bash(command: str) -> str:
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return result.stdout or "Выполнено успешно (нет вывода)"
        return f"Ошибка (код {result.returncode}):\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "Ошибка: превышен таймаут 30 секунд"
    except Exception as e:
        return f"Ошибка: {str(e)}"


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n... [обрезано {len(text) - max_chars} символов] ...\n" + text[-half:]


class Agent:
    def __init__(self, client: OpenAI, model: str, system: str,
                 context_window: int = 10, max_tool_output: int = 2000,
                 logger: SessionLogger | None = None):
        self.client = client
        self.model = model
        self.context_window = context_window
        self.max_tool_output = max_tool_output
        self.logger = logger
        self.messages: list = [{"role": "system", "content": system}]

    def run_turn(self, user_input: str, on_tool_call=None) -> str:
        self.messages.append({"role": "user", "content": user_input})
        if self.logger:
            self.logger.user(user_input)

        tool_calls_made = 0

        while True:
            history = self.messages[1:]
            windowed = [self.messages[0]] + history[-self.context_window:]

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=windowed,  # type: ignore
                    tools=TOOLS,        # type: ignore
                    tool_choice="auto"
                )
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
                    if self.logger:
                        self.logger.tool_call(call.function.name, call.function.arguments)
                    result = _truncate(_execute_bash(**args), self.max_tool_output)
                    if self.logger:
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
