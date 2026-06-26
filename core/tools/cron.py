import json
import os
import threading

JOBS_FILE = "jobs.json"
_lock = threading.Lock()

SCHEMA = {
    "type": "function",
    "function": {
        "name": "cron_manage",
        "description": (
            "Manage scheduled cron tasks. "
            "action=add: create a new task (requires name, schedule, prompt). "
            "action=list: show all tasks. "
            "action=remove: delete a task by name. "
            "Schedule format: cron expression, e.g. '0 9 * * *' = every day at 9:00."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action":   {"type": "string", "enum": ["add", "list", "remove"]},
                "name":     {"type": "string", "description": "Unique task name"},
                "schedule": {"type": "string", "description": "Cron expression, e.g. '0 9 * * *'"},
                "prompt":   {"type": "string", "description": "Task prompt the agent will execute on schedule"}
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


def execute(action: str, name: str = "", schedule: str = "", prompt: str = "") -> str:
    with _lock:
        jobs = _load()

        if action == "list":
            if not jobs:
                return "Нет активных задач."
            return "\n".join(
                f"• {j['name']} [{j['schedule']}]: {j['prompt']}" for j in jobs
            )

        if action == "add":
            if not name or not schedule or not prompt:
                return "Ошибка: для add нужны name, schedule и prompt."
            if any(j["name"] == name for j in jobs):
                return f"Ошибка: задача '{name}' уже существует."
            jobs.append({"name": name, "schedule": schedule, "prompt": prompt})
            _save(jobs)
            return f"Задача '{name}' добавлена [{schedule}]."

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
