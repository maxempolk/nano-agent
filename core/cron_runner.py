from __future__ import annotations
import httpx
from datetime import datetime
from tzlocal import get_localzone
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from core.tools.cron import _load, _lock, remove_job


def _send_telegram(token: str, chat_id: str, text: str) -> None:
    try:
        httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10
        )
    except Exception as e:
        print(f"[cron] Ошибка отправки в Telegram: {e}")


def _run_job(job: dict, agent_factory, token: str, chat_id: str) -> None:
    print(f"[cron] Запуск задачи '{job['name']}'")
    try:
        agent = agent_factory()
        reply = agent.run_turn(job["prompt"])
        _send_telegram(token, chat_id, f"⏰ <b>{job['name']}</b>\n\n{reply}")
    except Exception as e:
        print(f"[cron] Ошибка в задаче '{job['name']}': {e}")
        _send_telegram(token, chat_id, f"⏰ <b>{job['name']}</b>\n\nОшибка: {e}")

    # Удаляем одноразовую задачу после выполнения
    if job.get("type") == "once":
        remove_job(job["name"])
        print(f"[cron] Одноразовая задача '{job['name']}' удалена после выполнения")


class CronRunner:
    def __init__(self, agent_factory, token: str, chat_id: str):
        self.agent_factory = agent_factory
        self.token = token
        self.chat_id = chat_id
        self.scheduler = BackgroundScheduler(timezone=get_localzone())

    def _reload_jobs(self) -> None:
        with _lock:
            jobs = _load()

        current_ids = {job.id for job in self.scheduler.get_jobs()}
        file_names = {j["name"] for j in jobs}

        # Удаляем задачи которых больше нет в файле
        for job_id in current_ids - file_names - {"__reload__"}:
            self.scheduler.remove_job(job_id)
            print(f"[cron] Задача '{job_id}' удалена из планировщика")

        # Добавляем новые задачи
        for job in jobs:
            if job["name"] in current_ids:
                continue

            try:
                if job.get("type") == "once":
                    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                        try:
                            run_at = datetime.strptime(job["run_at"], fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        print(f"[cron] Неверный формат run_at для '{job['name']}': {job['run_at']}")
                        continue
                    trigger = DateTrigger(run_date=run_at, timezone=get_localzone())
                    print(f"[cron] Одноразовая задача '{job['name']}' добавлена [run_at: {job['run_at']}]")
                else:
                    parts = job["schedule"].split()
                    if len(parts) != 5:
                        print(f"[cron] Неверный формат cron для '{job['name']}': {job['schedule']}")
                        continue
                    minute, hour, day, month, day_of_week = parts
                    trigger = CronTrigger(
                        minute=minute, hour=hour,
                        day=day, month=month, day_of_week=day_of_week
                    )
                    print(f"[cron] Повторяющаяся задача '{job['name']}' добавлена [{job['schedule']}]")

                self.scheduler.add_job(
                    _run_job,
                    trigger,
                    args=[job, self.agent_factory, self.token, self.chat_id],
                    id=job["name"],
                    name=job["name"],
                    misfire_grace_time=60  # запустить даже если опоздали до 60 сек
                )
            except Exception as e:
                print(f"[cron] Ошибка добавления задачи '{job['name']}': {e}")

    def start(self) -> None:
        self._reload_jobs()
        self.scheduler.start()

        self.scheduler.add_job(
            self._reload_jobs,
            "interval",
            seconds=30,
            id="__reload__"
        )
        print("[cron] Планировщик запущен")
