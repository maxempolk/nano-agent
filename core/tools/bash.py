from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import threading
import time
from enum import Enum
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from core.cancellation import CancelledError
from core.policy import Capability
from core.tools.base import ErrorCode, Tool, ToolContext, ToolResult

DEFAULT_TIMEOUT = 30.0
DEFAULT_OUTPUT_LIMIT = 4000
MAX_CAPTURED_BYTES = 200_000

# Scratch locations outside the workspace that mutating commands may touch.
SCRATCH_ROOTS = ("/tmp", "/private/tmp", "/var/folders")

SENSITIVE_ENV_MARKERS = (
    "TOKEN",
    "SECRET",
    "API_KEY",
    "APIKEY",
    "PASSWORD",
    "PASSWD",
    "PRIVATE_KEY",
    "CREDENTIAL",
    "AUTH",
)
SENSITIVE_ENV_KEYS = {
    "TELEGRAM_BOT_TOKEN",
    "ALLOWED_USER_ID",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GITHUB_TOKEN",
    "OPENAI_API_KEY",
}


class CommandKind(str, Enum):
    READ_ONLY = "read_only"
    MUTATING = "mutating"
    DESTRUCTIVE = "destructive"
    INTERACTIVE = "interactive"
    BLOCKED = "blocked"


class BashInput(BaseModel):
    command: str = Field(min_length=1, max_length=4000, description="bash command to run")


_BLOCKED_PATTERNS = (
    (re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"), "fork bomb"),
    (re.compile(r"\bmkfs(\.\w+)?\b"), "mkfs"),
    (re.compile(r"\bdd\b[^|;&]*\bof=/dev/"), "dd на устройство"),
    (re.compile(r">\s*/dev/(sd|nvme|disk)"), "запись на устройство"),
    (re.compile(r"\bdiskutil\s+(erase|zero|secureErase)"), "diskutil erase"),
    (re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"), "управление питанием"),
    (re.compile(r"\bsudo\b"), "sudo"),
    (re.compile(r"\brm\b[^|;&]*\s(/|~|\$HOME)(\s|$|['\"])"), "удаление корня или home"),
    (re.compile(r"--no-preserve-root"), "rm --no-preserve-root"),
    (re.compile(r"\bchmod\b[^|;&]*\s(-R\s+)?(777|666)\s+/(\s|$)"), "chmod на корень"),
    (re.compile(r"\bcrontab\s+-r\b"), "очистка crontab"),
    (re.compile(r"\bgit\s+push\b[^|;&]*\s--delete\b"), "удаление удалённой ветки"),
    (re.compile(r"\bgit\s+push\b[^|;&]*\s--mirror\b"), "git push --mirror"),
)

_DESTRUCTIVE_PATTERNS = (
    (re.compile(r"\brm\b"), "rm"),
    (re.compile(r"\brmdir\b"), "rmdir"),
    (re.compile(r"\bgit\s+push\b[^|;&]*(--force\b|-f\b|--force-with-lease)"), "git push --force"),
    (re.compile(r"\bgit\s+reset\s+--hard\b"), "git reset --hard"),
    (re.compile(r"\bgit\s+clean\b[^|;&]*-\w*f"), "git clean -f"),
    (re.compile(r"\bgit\s+branch\b[^|;&]*-D\b"), "git branch -D"),
    (re.compile(r"\bkill\b"), "kill"),
    (re.compile(r"\bpkill\b|\bkillall\b"), "pkill/killall"),
    (re.compile(r"\bchmod\b|\bchown\b"), "chmod/chown"),
    (re.compile(r"\btruncate\b"), "truncate"),
    (re.compile(r"\bDROP\s+(TABLE|DATABASE)\b", re.IGNORECASE), "SQL DROP"),
    (re.compile(r"\bbrew\s+(uninstall|remove)\b"), "brew uninstall"),
)

_INTERACTIVE_PROGRAMS = {
    "vim",
    "vi",
    "nvim",
    "nano",
    "emacs",
    "pico",
    "less",
    "more",
    "top",
    "htop",
    "btop",
    "ssh",
    "telnet",
    "ftp",
    "sftp",
    "mysql",
    "psql",
    "sqlite3",
}
_NON_INTERACTIVE_FLAGS = {
    "python": {"-c", "-m", "-V", "--version"},
    "python3": {"-c", "-m", "-V", "--version"},
    "node": {"-e", "--eval", "-v", "--version"},
    "bash": {"-c"},
    "sh": {"-c"},
    "zsh": {"-c"},
    "git": {"-h", "--help"},
}
_INTERACTIVE_GIT_SUBCOMMANDS = {"commit", "rebase", "merge", "am", "cherry-pick"}

_MUTATING_MARKERS = re.compile(
    r"(?<![2&])>>?\s*\S|"  # file redirects except 2>&1
    r"\|\s*tee\b|"
    r"\b(mv|cp|mkdir|touch|ln|install|patch|sed\s+-i|"
    r"pip3?\s+install|python3?\s+-m\s+pip\s+install|uv\s+(add|sync|pip\s+install)|"
    r"npm\s+(i|install|ci|uninstall)|yarn\s+(add|install|remove)|pnpm\s+(add|install)|"
    r"brew\s+install|apt(-get)?\s+install|"
    r"git\s+(add|commit|push|checkout|switch|merge|rebase|reset|clean|stash\s+(pop|drop)|clone|init|mv|rm)|"
    r"curl\b[^|;&]*(-o\b|--output\b|-O\b|--remote-name\b|-T\b|--upload-file\b|"
    r"-X\s*(POST|PUT|PATCH|DELETE)|\s(-d|--data\w*|--form|-F)\b)|"
    r"wget\b(?![^|;&]*-O\s*-)|rsync|tar\s+-x|unzip|gunzip|"
    r"docker\s+(run|build|pull|push|rmi)|launchctl|crontab|defaults\s+write|"
    r"npm\s+publish|cargo\s+publish)\b"
)

_BACKGROUND_PATTERN = re.compile(r"(?<!&)&(?![>&\d])")
_DAEMON_MARKERS = re.compile(r"\b(nohup|disown|setsid|caffeinate)\b")

_READ_ONLY_PROGRAMS = {
    "ls",
    "ll",
    "cat",
    "head",
    "tail",
    "grep",
    "egrep",
    "fgrep",
    "rg",
    "find",
    "fd",
    "pwd",
    "which",
    "whereis",
    "type",
    "file",
    "stat",
    "du",
    "df",
    "wc",
    "sort",
    "uniq",
    "cut",
    "diff",
    "cmp",
    "date",
    "whoami",
    "id",
    "uname",
    "hostname",
    "uptime",
    "ps",
    "env",
    "printenv",
    "echo",
    "printf",
    "true",
    "basename",
    "dirname",
    "realpath",
    "readlink",
    "jq",
    "sw_vers",
    "git",
    "curl",
    "ping",
    "dig",
    "nslookup",
    "host",
}
_GIT_READ_SUBCOMMANDS = {
    "status",
    "log",
    "diff",
    "show",
    "branch",
    "remote",
    "ls-files",
    "rev-parse",
    "config",
    "blame",
    "shortlog",
    "describe",
    "tag",
    "stash",
}

_EXTERNAL_SEND_FLAGS = re.compile(
    r"-X\s*(POST|PUT|PATCH|DELETE)|--request\s*(POST|PUT|PATCH|DELETE)|"
    r"(^|\s)(-d|--data(-\w+)?|-F|--form|-T|--upload-file)\s",
    re.IGNORECASE,
)
_URL_HOST = re.compile(r"https?://([^/\s'\"]+)")

_SEGMENT_SPLIT = re.compile(r"\|\||&&|[|;]|\$\(|`")


def filtered_env(extra_hidden: tuple[str, ...] = ()) -> dict[str, str]:
    """os.environ copy with credentials and secret-looking keys removed."""
    hidden = {key.upper() for key in SENSITIVE_ENV_KEYS} | {key.upper() for key in extra_hidden}
    cleaned = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if upper in hidden or any(marker in upper for marker in SENSITIVE_ENV_MARKERS):
            continue
        cleaned[key] = value
    return cleaned


def _segments(command: str) -> list[str]:
    return [segment.strip() for segment in _SEGMENT_SPLIT.split(command) if segment.strip()]


def _program(segment: str) -> str:
    try:
        tokens = shlex.split(segment)
    except ValueError:
        tokens = segment.split()
    if not tokens:
        return ""
    first = tokens[0]
    if "=" in first and not first.startswith("-"):
        return ""
    return Path(first).name


def _tokens(segment: str) -> list[str]:
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.split()


def _is_interactive_segment(segment: str) -> bool:
    tokens = _tokens(segment)
    if not tokens:
        return False
    program = Path(tokens[0]).name
    rest = tokens[1:]

    if program in _INTERACTIVE_PROGRAMS:
        return True
    if program == "git" and rest:
        subcommand = next((token for token in rest if not token.startswith("-")), "")
        if subcommand in _INTERACTIVE_GIT_SUBCOMMANDS:
            # git commit -m, rebase --continue etc. do not open an editor
            return not any(token.startswith("-") for token in rest)
        return False
    flags = _NON_INTERACTIVE_FLAGS.get(program)
    if program in {"python", "python3", "node", "bash", "sh", "zsh"}:
        if not rest:
            return True
        if flags and any(token in flags for token in rest):
            return False
        if program in {"bash", "sh", "zsh"}:
            return False  # script file argument
        return any(token.startswith("-") for token in rest) is False and not any(
            not token.startswith("-") and ("." in token or "/" in token) for token in rest
        )
    if program == "docker" and rest and rest[0] == "exec":
        return "-it" in rest or "-ti" in rest
    return False


def _external_send_hosts(segment: str) -> list[str]:
    """Hosts that receive uploaded data; downloads are network reads."""
    program = _program(segment)
    if program in {"nc", "ncat"}:
        return ["<raw socket>"]
    if program == "curl" and _EXTERNAL_SEND_FLAGS.search(segment):
        hosts = _URL_HOST.findall(segment)
        return hosts or ["<unknown host>"]
    return []


def classify_command(command: str) -> tuple[CommandKind, list[str]]:
    """Static classification executed before anything runs."""
    for pattern, label in _BLOCKED_PATTERNS:
        if pattern.search(command):
            return CommandKind.BLOCKED, [label]
    if _BACKGROUND_PATTERN.search(command) or _DAEMON_MARKERS.search(command):
        return CommandKind.BLOCKED, ["фоновые/демон-процессы запрещены"]
    for segment in _segments(command):
        if _is_interactive_segment(segment):
            return CommandKind.INTERACTIVE, [f"интерактивная программа: {_program(segment)}"]

    destructive = [label for pattern, label in _DESTRUCTIVE_PATTERNS if pattern.search(command)]
    if destructive:
        return CommandKind.DESTRUCTIVE, destructive

    if _MUTATING_MARKERS.search(command):
        return CommandKind.MUTATING, ["записывающая команда"]

    programs = {_program(segment) for segment in _segments(command)}
    programs.discard("")
    if programs and not programs <= _READ_ONLY_PROGRAMS:
        return CommandKind.MUTATING, [f"неизвестная программа: {', '.join(sorted(programs))}"]
    if programs == {"git"}:
        for segment in _segments(command):
            tokens = _tokens(segment)
            subcommand = next((token for token in tokens[1:] if not token.startswith("-")), "")
            if subcommand and subcommand not in _GIT_READ_SUBCOMMANDS:
                return CommandKind.MUTATING, [f"git {subcommand}"]
    return CommandKind.READ_ONLY, ["read-only"]


def extract_paths(command: str) -> list[str]:
    """Absolute and ~-prefixed path tokens referenced by the command."""
    paths: list[str] = []
    for segment in _segments(command):
        for token in _tokens(segment):
            if token.startswith(("/", "~/", "$HOME/")) or token in {"~", "$HOME"}:
                paths.append(token)
    return paths


class BashTool(Tool):
    """Safe bash execution pinned to the configured workspace.

    Guarantees: explicit cwd, workspace path confinement for mutations,
    timeout with process-group kill, filtered environment, no interactive
    or background processes, bounded output, structured errors, static
    read-only/mutating/destructive classification and approval for danger.
    """

    name: ClassVar[str] = "execute_bash"
    description: ClassVar[str] = (
        "Execute a bash command inside the agent workspace. "
        "Read-only commands may inspect any path; commands that modify files "
        "may only touch the workspace. Interactive, background, sudo and "
        "destructive commands are rejected or require approval."
    )
    input_model: ClassVar[type[BaseModel]] = BashInput
    capabilities: ClassVar[frozenset[Capability]] = frozenset(
        {Capability.SHELL_READ, Capability.SHELL_WRITE, Capability.FILESYSTEM_READ}
    )
    output_limit: ClassVar[int] = DEFAULT_OUTPUT_LIMIT

    def __init__(
        self,
        workspace: str | Path,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        output_limit: int = DEFAULT_OUTPUT_LIMIT,
    ):
        if timeout <= 0:
            raise ValueError("bash timeout должен быть больше нуля")
        self.workspace = Path(workspace).expanduser().resolve()
        self.timeout = timeout
        self.output_limit = output_limit

    def execute(self, args: BashInput, ctx: ToolContext) -> ToolResult:
        command = args.command.strip()
        kind, reasons = classify_command(command)

        if kind == CommandKind.BLOCKED:
            return ToolResult.failure(
                f"команда заблокирована политикой: {', '.join(reasons)}",
                code=ErrorCode.DENIED,
            )
        if kind == CommandKind.INTERACTIVE:
            return ToolResult.failure(
                f"интерактивные процессы запрещены ({', '.join(reasons)}). "
                "Используй неинтерактивные флаги.",
                code=ErrorCode.DENIED,
            )

        policy = ctx.policy
        paths = extract_paths(command)
        if kind in {CommandKind.MUTATING, CommandKind.DESTRUCTIVE}:
            outside = [
                path
                for path in paths
                if not self._allowed_write_target(path, policy)
            ]
            if outside:
                return ToolResult.failure(
                    "изменения разрешены только внутри workspace "
                    f"({self.workspace}); затронуты пути: {', '.join(outside[:3])}",
                    code=ErrorCode.DENIED,
                )

        if kind == CommandKind.DESTRUCTIVE:
            if policy is None or not policy.allow_destructive:
                return ToolResult.failure(
                    f"деструктивная команда отклонена политикой ({', '.join(reasons)})",
                    code=ErrorCode.DENIED,
                )
            if not policy.approve(f"Деструктивная команда в {self.workspace}: {command}"):
                return ToolResult.failure(
                    "деструктивная команда не подтверждена пользователем",
                    code=ErrorCode.DENIED,
                )

        send_hosts: list[str] = []
        for segment in _segments(command):
            send_hosts.extend(_external_send_hosts(segment))
        if send_hosts:
            if policy is None or not policy.allow_external_send:
                return ToolResult.failure(
                    "отправка данных во внешние сервисы отключена политикой",
                    code=ErrorCode.DENIED,
                )
            if policy.external_send_hosts:
                denied_hosts = [
                    host
                    for host in send_hosts
                    if not any(
                        host == allowed or host.endswith(f".{allowed}")
                        for allowed in policy.external_send_hosts
                    )
                ]
                if denied_hosts:
                    return ToolResult.failure(
                        "отправка разрешена только на "
                        f"{', '.join(sorted(policy.external_send_hosts))}, "
                        f"а не на {', '.join(sorted(set(denied_hosts)))}",
                        code=ErrorCode.DENIED,
                    )

        capabilities = self._capabilities_for(kind, send_hosts)
        if policy is not None:
            decision = policy.check_capabilities(capabilities)
            if not decision.allowed:
                return ToolResult.failure(
                    f"политика отклонила команду: {decision.reason}", code=ErrorCode.DENIED
                )

        return self._run(command, ctx, kind)

    def _allowed_write_target(self, path: str, policy) -> bool:
        expanded = path.replace("$HOME", "~")
        candidate = Path(expanded).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        try:
            resolved = str(candidate.resolve())
        except OSError:
            return False
        for root in SCRATCH_ROOTS:
            if resolved == root or resolved.startswith(root + os.sep):
                return True
        if policy is not None and policy.inside_workspace(resolved):
            return True
        return resolved == str(self.workspace) or resolved.startswith(
            str(self.workspace) + os.sep
        )

    def _capabilities_for(self, kind: CommandKind, send_hosts: list[str]) -> frozenset[Capability]:
        if kind == CommandKind.READ_ONLY:
            caps = {Capability.SHELL_READ, Capability.FILESYSTEM_READ}
            if send_hosts:
                caps.add(Capability.NETWORK_READ)
            return frozenset(caps)
        caps = {Capability.SHELL_WRITE, Capability.FILESYSTEM_WRITE}
        if send_hosts:
            caps.add(Capability.EXTERNAL_SEND)
        if kind == CommandKind.DESTRUCTIVE:
            caps.add(Capability.DESTRUCTIVE)
        return frozenset(caps)

    def _run(self, command: str, ctx: ToolContext, kind: CommandKind) -> ToolResult:
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []

        def reader(stream, sink: list[bytes]) -> None:
            total = 0
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    break
                if total < MAX_CAPTURED_BYTES:
                    sink.append(chunk)
                    total += len(chunk)

        try:
            process = subprocess.Popen(
                ["bash", "-c", command],
                cwd=self.workspace,
                env=filtered_env(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError as error:
            return ToolResult.failure(f"не удалось запустить bash: {error}")
        except OSError as error:
            return ToolResult.failure(f"ошибка запуска процесса: {error}")

        threads = [
            threading.Thread(target=reader, args=(process.stdout, stdout_chunks), daemon=True),
            threading.Thread(target=reader, args=(process.stderr, stderr_chunks), daemon=True),
        ]
        for thread in threads:
            thread.start()

        deadline = time.monotonic() + self.timeout
        cancelled = False
        while True:
            returncode = process.poll()
            if returncode is not None:
                break
            if ctx.cancel is not None and ctx.cancel.cancelled:
                cancelled = True
                break
            if time.monotonic() >= deadline:
                break
            ctx_raise = False
            if ctx.cancel is not None:
                ctx_raise = ctx.cancel.wait(0.1)
            else:
                time.sleep(0.1)
            if ctx_raise:
                cancelled = True
                break

        if cancelled or (returncode is None and time.monotonic() >= deadline):
            self._kill_process_group(process)
            process.wait(timeout=5)
            if cancelled:
                raise CancelledError("bash отменён")
            return ToolResult.failure(
                f"команда превысила таймаут {self.timeout:.0f} с и была завершена",
                code=ErrorCode.TIMEOUT,
                retryable=True,
            )

        for thread in threads:
            thread.join(timeout=2)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()

        stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace").strip()
        stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace").strip()

        if returncode == 0:
            content = stdout or "Выполнено успешно (нет вывода)"
            return ToolResult(
                content=content,
                summary=f"команда выполнена (код 0, {kind.value})",
            )

        error_text = stderr or stdout or "нет вывода"
        return ToolResult(
            content="",
            summary=f"команда завершилась с кодом {returncode}",
            error=f"код {returncode}: {error_text}",
            error_code="command_failed",
            retryable=False,
        )

    @staticmethod
    def _kill_process_group(process: subprocess.Popen) -> None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.kill()
            except OSError:
                pass


# --- legacy module API kept until the agent switches to the registry ---
SCHEMA = {
    "type": "function",
    "function": {
        "name": "execute_bash",
        "description": "Execute a bash command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}


def execute(command: str) -> str:
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return result.stdout or "Выполнено успешно (нет вывода)"
        return f"Ошибка (код {result.returncode}):\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "Ошибка: превышен таймаут 30 секунд"
    except Exception as e:
        return f"Ошибка: {str(e)}"
