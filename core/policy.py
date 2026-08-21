from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Capability(str, Enum):
    FILESYSTEM_READ = "filesystem.read"
    FILESYSTEM_WRITE = "filesystem.write"
    SHELL_READ = "shell.read"
    SHELL_WRITE = "shell.write"
    NETWORK_READ = "network.read"
    EXTERNAL_SEND = "external.send"
    SCHEDULER_WRITE = "scheduler.write"
    NOTES_WRITE = "notes.write"
    DESTRUCTIVE = "destructive"


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str
    needs_approval: bool = False


class ExecutionPolicy:
    """Compact policy layer that works independently of the model.

    Defaults: reading and searching are allowed; mutations only inside the
    workspace; network reads allowed; sending data outside is controlled;
    destructive actions require explicit allowance plus approval.
    """

    def __init__(
        self,
        workspace: str | Path,
        *,
        allow_external_send: bool = False,
        external_send_hosts: frozenset[str] | tuple[str, ...] = frozenset(),
        allow_destructive: bool = False,
        approval: Callable[[str], bool] | None = None,
    ):
        self.workspace = Path(workspace).expanduser().resolve()
        self.allow_external_send = allow_external_send
        self.external_send_hosts = frozenset(external_send_hosts)
        self.allow_destructive = allow_destructive
        self._approval = approval

    def inside_workspace(self, path: str | Path) -> bool:
        candidate = Path(str(path)).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        try:
            resolved = candidate.resolve()
            resolved.relative_to(self.workspace)
            return True
        except ValueError:
            return False

    def check_capabilities(self, capabilities: frozenset[Capability]) -> PolicyDecision:
        if Capability.DESTRUCTIVE in capabilities:
            if not self.allow_destructive:
                return PolicyDecision(False, "destructive actions отключены политикой")
            return PolicyDecision(True, "destructive разрешён конфигурацией", needs_approval=True)
        if Capability.EXTERNAL_SEND in capabilities and not self.allow_external_send:
            return PolicyDecision(False, "отправка данных во внешние сервисы отключена политикой")
        return PolicyDecision(True, "ok")

    def approve(self, description: str) -> bool:
        """Ask the interactive approval callback, when one is configured."""
        if self._approval is None:
            return False
        try:
            return bool(self._approval(description))
        except Exception as error:  # noqa: BLE001 - approval must never crash a run
            print(f"[policy] approval callback error: {error}")
            return False
