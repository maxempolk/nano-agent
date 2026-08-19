from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

MAX_PREVIEW_CHARS = 160


def preview(text: str, limit: int = MAX_PREVIEW_CHARS) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


@dataclass(frozen=True)
class AgentEvent:
    kind: ClassVar[str] = "event"


@dataclass(frozen=True)
class RunStarted(AgentEvent):
    kind: ClassVar[str] = "run_started"
    input_preview: str


@dataclass(frozen=True)
class RouteSelected(AgentEvent):
    kind: ClassVar[str] = "route_selected"
    route: str
    model: str
    reason: str
    score: int
    automatic: bool


@dataclass(frozen=True)
class ModelStarted(AgentEvent):
    kind: ClassVar[str] = "model_started"
    model: str
    step: int


@dataclass(frozen=True)
class ModelCompleted(AgentEvent):
    kind: ClassVar[str] = "model_completed"
    model: str
    step: int
    finish_reason: str
    tool_calls: int
    content_chars: int
    elapsed: float


@dataclass(frozen=True)
class ToolStarted(AgentEvent):
    kind: ClassVar[str] = "tool_started"
    name: str
    args_summary: str


@dataclass(frozen=True)
class ToolCompleted(AgentEvent):
    kind: ClassVar[str] = "tool_completed"
    name: str
    summary: str
    ok: bool
    error_code: str
    elapsed: float


@dataclass(frozen=True)
class ContextCompacted(AgentEvent):
    kind: ClassVar[str] = "context_compacted"
    before_tokens: int
    after_tokens: int


@dataclass(frozen=True)
class RunCompleted(AgentEvent):
    kind: ClassVar[str] = "run_completed"
    reply_preview: str
    elapsed: float
    steps: int
    model_calls: int
    tool_calls: int


@dataclass(frozen=True)
class RunFailed(AgentEvent):
    kind: ClassVar[str] = "run_failed"
    error: str
    elapsed: float


@dataclass(frozen=True)
class RunCancelled(AgentEvent):
    kind: ClassVar[str] = "run_cancelled"
    reason: str
    elapsed: float


Listener = Callable[[AgentEvent], None]


@dataclass
class EventBus:
    """Fan-out of small typed progress events to CLI/Telegram listeners.

    Listeners never receive model internals: events carry short previews,
    summaries and statuses only. A broken listener cannot break the run.
    """

    _listeners: list[Listener] = field(default_factory=list)

    def subscribe(self, listener: Listener) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def unsubscribe(self, listener: Listener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def emit(self, event: AgentEvent) -> None:
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception as error:  # noqa: BLE001 - listener isolation
                print(f"[events] listener error ignored: {error}")
