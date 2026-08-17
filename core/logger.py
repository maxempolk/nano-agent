import os
from datetime import datetime

DIVIDER = "─" * 60
SESSION_DIVIDER = "═" * 60


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


class SessionLogger:
    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
        self.path = os.path.join(log_dir, filename)
        self._write(
            f"{SESSION_DIVIDER}\n"
            f"SESSION  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{SESSION_DIVIDER}\n"
        )

    def _write(self, text: str) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def _ts(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def user(self, text: str) -> None:
        self._write(f"[{self._ts()}] USER\n{_indent(text)}\n{DIVIDER}")

    def tool_call(self, name: str, args: str) -> None:
        self._write(f"[{self._ts()}] TOOL CALL → {name}\n{_indent(args)}")

    def tool_result(self, result: str) -> None:
        self._write(f"[{self._ts()}] TOOL RESULT\n{_indent(result)}\n{DIVIDER}")

    def agent(self, text: str) -> None:
        self._write(f"[{self._ts()}] AGENT\n{_indent(text)}\n{SESSION_DIVIDER}")

    def error(self, text: str) -> None:
        self._write(f"[{self._ts()}] ERROR\n{_indent(text)}\n{DIVIDER}")

    def info(self, text: str) -> None:
        self._write(f"[{self._ts()}] INFO  {text}")
