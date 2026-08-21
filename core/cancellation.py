from __future__ import annotations

import threading


class CancelledError(RuntimeError):
    """Raised inside an agent run when its cancellation token was triggered."""


class CancellationToken:
    """Thread-safe flag that propagates cancellation across the run.

    The agent loop and every tool receive the same token and must call
    ``raise_if_cancelled`` at their checkpoints. Long blocking operations
    (model requests, crawlers, subprocesses) are interrupted as soon as
    they return control or observe the token.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason = ""
        self._lock = threading.Lock()

    def cancel(self, reason: str = "cancelled") -> None:
        with self._lock:
            if not self._reason:
                self._reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise CancelledError(self._reason or "cancelled")

    def wait(self, timeout: float) -> bool:
        """Block up to ``timeout`` seconds; True when cancelled while waiting."""
        return self._event.wait(timeout)
