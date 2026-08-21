from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass


class BudgetExceeded(RuntimeError):
    """Raised when a run-level limit is hit. The agent must surface this
    honestly instead of masking it with a plausible answer."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def _positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} должен быть больше нуля")


@dataclass(frozen=True)
class BudgetLimits:
    """Per-user-request limits for the whole agent turn."""

    max_steps: int = 8
    max_model_calls: int = 12
    max_tool_calls: int = 12
    max_wall_seconds: float = 180.0
    max_tool_output_chars: int = 2000
    max_consecutive_errors: int = 3
    max_identical_calls: int = 2

    def __post_init__(self) -> None:
        _positive(self.max_steps, "max_steps")
        _positive(self.max_model_calls, "max_model_calls")
        _positive(self.max_tool_calls, "max_tool_calls")
        _positive(self.max_wall_seconds, "max_wall_seconds")
        _positive(self.max_tool_output_chars, "max_tool_output_chars")
        _positive(self.max_consecutive_errors, "max_consecutive_errors")
        _positive(self.max_identical_calls, "max_identical_calls")


def call_signature(name: str, args: dict) -> str:
    try:
        payload = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = repr(args)
    return f"{name}:{payload}"


class RunBudget:
    """Mutable counters for one user request.

    Web-search keeps its own mode budget; RunBudget bounds the whole turn
    (steps, model calls, tool calls, wall time, tool output size, repeated
    errors and identical tool calls).
    """

    def __init__(self, limits: BudgetLimits | None = None):
        self.limits = limits or BudgetLimits()
        self.started_at = time.monotonic()
        self.steps = 0
        self.model_calls = 0
        self.tool_calls = 0
        self.consecutive_errors = 0
        self.last_error = ""
        self._signatures: dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def check_deadline(self) -> None:
        if self.elapsed >= self.limits.max_wall_seconds:
            raise BudgetExceeded(
                "time",
                f"превышено время выполнения ({self.limits.max_wall_seconds:.0f} с)",
            )

    def consume_step(self) -> None:
        self.check_deadline()
        with self._lock:
            self.steps += 1
            if self.steps > self.limits.max_steps:
                raise BudgetExceeded(
                    "steps",
                    f"превышен лимит шагов агента ({self.limits.max_steps})",
                )

    def consume_model_call(self) -> None:
        self.check_deadline()
        with self._lock:
            self.model_calls += 1
            if self.model_calls > self.limits.max_model_calls:
                raise BudgetExceeded(
                    "model_calls",
                    f"превышен лимит вызовов модели ({self.limits.max_model_calls})",
                )

    def consume_tool_call(self, signature: str) -> None:
        self.check_deadline()
        with self._lock:
            self.tool_calls += 1
            if self.tool_calls > self.limits.max_tool_calls:
                raise BudgetExceeded(
                    "tool_calls",
                    f"превышен лимит вызовов инструментов ({self.limits.max_tool_calls})",
                )
            count = self._signatures.get(signature, 0) + 1
            self._signatures[signature] = count
            if count > self.limits.max_identical_calls:
                raise BudgetExceeded(
                    "repeated_tool_call",
                    "идентичный вызов инструмента повторяется без прогресса",
                )

    def note_tool_result(self, ok: bool, error: str = "") -> None:
        with self._lock:
            if ok:
                self.consecutive_errors = 0
                self.last_error = ""
            else:
                self.consecutive_errors += 1
                self.last_error = error
                if self.consecutive_errors >= self.limits.max_consecutive_errors:
                    raise BudgetExceeded(
                        "consecutive_errors",
                        f"{self.consecutive_errors} ошибки инструментов подряд: {error}",
                    )

    def cap_output(self, text: str) -> str:
        limit = self.limits.max_tool_output_chars
        if len(text) <= limit:
            return text
        half = limit // 2
        return (
            text[:half]
            + f"\n... [обрезано {len(text) - limit} символов] ...\n"
            + text[-half:]
        )

    def snapshot(self) -> dict:
        return {
            "steps": self.steps,
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "elapsed": round(self.elapsed, 2),
        }
