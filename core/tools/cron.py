from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from core.policy import Capability
from core.tools.base import Tool, ToolContext, ToolResult

JOBS_FILE = "jobs.json"
_lock = threading.Lock()

SCHEMA = {
    "type": "function",
    "function": {
        "name": "cron_manage",
        "description": (
            "Manage scheduled tasks. "
            "action=add: create a task (requires name, prompt, and one of: schedule, run_at, run_in). "
            "action=list: show all tasks. "
            "action=remove: delete a task by name. "
            "kind=reminder: for reminders and timers — the prompt text is delivered to the user as-is, "
            "nothing is executed. kind=task (default): the agent executes the prompt at the scheduled time. "
            "Use reminder whenever the user asks to remind them of something or set a timer with a text. "
            "For recurring tasks use schedule (cron expression, e.g. '0 9 * * *'). "
            "For one-time tasks use run_at (datetime string, e.g. '2026-06-27 15:30') "
            "OR run_in (seconds from now, e.g. 10 for 'in 10 seconds', 300 for 'in 5 minutes'). "
            "Prefer run_in over run_at for relative times like 'in X seconds/minutes'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "list", "remove"]},
                "name": {"type": "string", "description": "Unique task name"},
                "schedule": {
                    "type": "string",
                    "description": "Cron expression for recurring tasks, e.g. '0 9 * * *'",
                },
                "run_at": {
                    "type": "string",
                    "description": "Absolute datetime for one-time tasks, e.g. '2026-06-27 15:30'",
                },
                "run_in": {
                    "type": "integer",
                    "description": "Seconds from now for one-time tasks, e.g. 10 for 'in 10 seconds'",
                },
                "prompt": {
                    "type": "string",
                    "description": "For task: work for the agent to execute. For reminder: the exact text to deliver. Do NOT include curl or Telegram commands — the result is delivered automatically.",
                },
                "kind": {
                    "type": "string",
                    "enum": ["task", "reminder"],
                    "description": "task (default): the agent executes the prompt. reminder: the prompt text is delivered as-is without execution.",
                },
            },
            "required": ["action"],
        },
    },
}


def _load(jobs_file: str = JOBS_FILE) -> list:
    if not os.path.exists(jobs_file):
        return []
    with open(jobs_file, encoding="utf-8") as f:
        return json.load(f)


def _save(jobs: list, jobs_file: str = JOBS_FILE) -> None:
    with open(jobs_file, "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2)


def execute(
    action: str,
    name: str = "",
    schedule: str = "",
    run_at: str = "",
    run_in: int = 0,
    prompt: str = "",
    kind: str = "task",
    jobs_file: str = JOBS_FILE,
) -> str:
    with _lock:
        jobs = _load(jobs_file)

        if action == "list":
            if not jobs:
                return "Нет активных задач."
            lines = []
            for j in jobs:
                tag = " (напоминание)" if j.get("kind") == "reminder" else ""
                if j.get("type") == "once":
                    lines.append(f"• {j['name']}{tag} [once: {j['run_at']}]: {j['prompt']}")
                else:
                    lines.append(f"• {j['name']}{tag} [cron: {j['schedule']}]: {j['prompt']}")
            return "\n".join(lines)

        if action == "add":
            if not name or not prompt:
                return "Ошибка: для add нужны name и prompt."
            if kind not in ("task", "reminder"):
                return "Ошибка: kind должен быть task или reminder."
            if any(j["name"] == name for j in jobs):
                return f"Ошибка: задача '{name}' уже существует."

            if run_in:
                run_at = (datetime.now() + timedelta(seconds=run_in)).strftime("%Y-%m-%d %H:%M:%S")

            if run_at and not schedule:
                jobs.append(
                    {
                        "name": name,
                        "type": "once",
                        "run_at": run_at,
                        "kind": kind,
                        "prompt": prompt,
                    }
                )
                _save(jobs, jobs_file)
                label = "Напоминание" if kind == "reminder" else "Одноразовая задача"
                return f"{label} '{name}' добавлено [run_at: {run_at}]."
            elif schedule:
                jobs.append(
                    {
                        "name": name,
                        "type": "cron",
                        "schedule": schedule,
                        "kind": kind,
                        "prompt": prompt,
                    }
                )
                _save(jobs, jobs_file)
                label = "Повторяющееся напоминание" if kind == "reminder" else "Повторяющаяся задача"
                return f"{label} '{name}' добавлено [{schedule}]."
            else:
                return "Ошибка: укажите schedule, run_at или run_in."

        if action == "remove":
            if not name:
                return "Ошибка: для remove нужен name."
            before = len(jobs)
            jobs = [j for j in jobs if j["name"] != name]
            if len(jobs) == before:
                return f"Задача '{name}' не найдена."
            _save(jobs, jobs_file)
            return f"Задача '{name}' удалена."

        return f"Неизвестный action: {action}"


def remove_job(name: str, jobs_file: str = JOBS_FILE) -> None:
    """Удаляет задачу из jobs.json (вызывается runner'ом после одноразовой задачи)."""
    with _lock:
        jobs = _load(jobs_file)
        jobs = [j for j in jobs if j["name"] != name]
        _save(jobs, jobs_file)


class CronInput(BaseModel):
    action: Literal["add", "list", "remove"]
    name: str = Field(default="", max_length=120)
    schedule: str = Field(default="", max_length=120)
    run_at: str = Field(default="", max_length=60)
    run_in: int = Field(default=0, ge=0, le=31_536_000)
    prompt: str = Field(default="", max_length=2000)
    kind: Literal["task", "reminder"] = "task"


class CronManageTool(Tool):
    """Scheduled-task management over the shared jobs file."""

    name: ClassVar[str] = "cron_manage"
    description: ClassVar[str] = SCHEMA["function"]["description"]
    input_model: ClassVar[type[BaseModel]] = CronInput
    capabilities: ClassVar[frozenset[Capability]] = frozenset({Capability.SCHEDULER_WRITE})
    timeout: ClassVar[float] = 5.0
    output_limit: ClassVar[int] = 2000

    def __init__(self, jobs_file: str = JOBS_FILE, on_change=None):
        self.jobs_file = jobs_file
        self.on_change = on_change

    def execute(self, args: CronInput, ctx: ToolContext) -> ToolResult:
        ctx.raise_if_cancelled()
        text = execute(
            action=args.action,
            name=args.name,
            schedule=args.schedule,
            run_at=args.run_at,
            run_in=args.run_in,
            prompt=args.prompt,
            kind=args.kind,
            jobs_file=self.jobs_file,
        )
        failed = text.startswith("Ошибка") or text.endswith("не найдена.")
        if failed:
            retryable = "уже существует" not in text and "не найдена" not in text
            return ToolResult.failure(text, code="validation", retryable=retryable)
        if args.action in {"add", "remove"} and self.on_change is not None:
            try:
                self.on_change()
            except Exception as error:  # noqa: BLE001 - scheduler reload is best-effort
                return ToolResult(
                    content=text,
                    summary=text,
                    warnings=(f"планировщик не перечитан: {error}",),
                )
        return ToolResult(content=text, summary=text)
