import json
import os
import threading
from datetime import datetime, timedelta

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
            "For recurring tasks use schedule (cron expression, e.g. '0 9 * * *'). "
            "For one-time tasks use run_at (datetime string, e.g. '2026-06-27 15:30') "
            "OR run_in (seconds from now, e.g. 10 for 'in 10 seconds', 300 for 'in 5 minutes'). "
            "Prefer run_in over run_at for relative times like 'in X seconds/minutes'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action":   {"type": "string", "enum": ["add", "list", "remove"]},
                "name":     {"type": "string", "description": "Unique task name"},
                "schedule": {"type": "string", "description": "Cron expression for recurring tasks, e.g. '0 9 * * *'"},
                "run_at":   {"type": "string", "description": "Absolute datetime for one-time tasks, e.g. '2026-06-27 15:30'"},
                "run_in":   {"type": "integer", "description": "Seconds from now for one-time tasks, e.g. 10 for 'in 10 seconds'"},
                "prompt":   {"type": "string", "description": "Task for the agent to execute. Do NOT include curl or Telegram commands — the result will be delivered automatically."}
            },
            "required": ["action"]
        }
    }
}


def _load() -> list:
    if not os.path.exists(JOBS_FILE):
        return []
    with open(JOBS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(jobs: list) -> None:
    with open(JOBS_FILE, "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2)


def execute(action: str, name: str = "", schedule: str = "",
            run_at: str = "", run_in: int = 0, prompt: str = "") -> str:
    with _lock:
        jobs = _load()

        if action == "list":
            if not jobs:
                return "Нет активных задач."
            lines = []
            for j in jobs:
                if j.get("type") == "once":
                    lines.append(f"• {j['name']} [once: {j['run_at']}]: {j['prompt']}")
                else:
                    lines.append(f"• {j['name']} [cron: {j['schedule']}]: {j['prompt']}")
            return "\n".join(lines)

        if action == "add":
            if not name or not prompt:
                return "Ошибка: для add нужны name и prompt."
            if any(j["name"] == name for j in jobs):
                return f"Ошибка: задача '{name}' уже существует."

            if run_in:
                run_at = (datetime.now() + timedelta(seconds=run_in)).strftime("%Y-%m-%d %H:%M:%S")

            if run_at and not schedule:
                jobs.append({"name": name, "type": "once", "run_at": run_at, "prompt": prompt})
                _save(jobs)
                return f"Одноразовая задача '{name}' добавлена [run_at: {run_at}]."
            elif schedule:
                jobs.append({"name": name, "type": "cron", "schedule": schedule, "prompt": prompt})
                _save(jobs)
                return f"Повторяющаяся задача '{name}' добавлена [{schedule}]."
            else:
                return "Ошибка: укажите schedule, run_at или run_in."

        if action == "remove":
            if not name:
                return "Ошибка: для remove нужен name."
            before = len(jobs)
            jobs = [j for j in jobs if j["name"] != name]
            if len(jobs) == before:
                return f"Задача '{name}' не найдена."
            _save(jobs)
            return f"Задача '{name}' удалена."

        return f"Неизвестный action: {action}"


def remove_job(name: str) -> None:
    """Удаляет задачу из jobs.json (вызывается runner'ом после одноразовой задачи)."""
    with _lock:
        jobs = _load()
        jobs = [j for j in jobs if j["name"] != name]
        _save(jobs)
