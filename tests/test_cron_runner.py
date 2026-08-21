import json
import os
import tempfile
from unittest import TestCase
from unittest.mock import Mock, patch

from core.cron_runner import CronRunner, _run_job
from core.tools.cron import _load


class FakeAgent:
    def __init__(self, reply="готово"):
        self.reply = reply
        self.prompts: list[str] = []
        self.cancel_reasons: list[str] = []

    def run_turn(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply

    def cancel(self, reason: str = "") -> None:
        self.cancel_reasons.append(reason)


class RunJobTests(TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.jobs_file = os.path.join(self._tmp.name, "jobs.json")
        self.addCleanup(self._tmp.cleanup)

    def _runner(self, agent: FakeAgent) -> CronRunner:
        return CronRunner(lambda: agent, "token", "42", jobs_file=self.jobs_file)

    def test_job_runs_agent_and_delivers_result(self):
        agent = FakeAgent(reply="отчёт готов")
        runner = self._runner(agent)
        job = {"name": "digest", "type": "cron", "schedule": "0 9 * * *", "prompt": "собери новости"}

        with patch("core.cron_runner._send_telegram") as send:
            _run_job(job, runner, "token", "42")

        self.assertEqual(agent.prompts, ["собери новости"])
        send.assert_called_once()
        self.assertIn("отчёт готов", send.call_args.args[2])
        self.assertIn("digest", send.call_args.args[2])

    def test_once_job_is_removed_before_execution(self):
        with open(self.jobs_file, "w", encoding="utf-8") as f:
            json.dump([{"name": "reminder", "type": "once", "run_at": "2026-01-01 10:00", "prompt": "x"}], f)
        agent = FakeAgent()
        runner = self._runner(agent)
        job = {"name": "reminder", "type": "once", "run_at": "2026-01-01 10:00", "prompt": "x"}

        with patch("core.cron_runner._send_telegram"):
            _run_job(job, runner, "token", "42")

        self.assertEqual(_load(self.jobs_file), [])

    def test_agent_error_is_delivered_not_raised(self):
        agent = FakeAgent()
        agent.run_turn = Mock(side_effect=RuntimeError("модель недоступна"))
        runner = self._runner(agent)
        job = {"name": "digest", "type": "cron", "schedule": "* * * * *", "prompt": "p"}

        with patch("core.cron_runner._send_telegram") as send:
            _run_job(job, runner, "token", "42")

        self.assertIn("Ошибка", send.call_args.args[2])

    def test_active_job_is_registered_and_cleaned_up(self):
        agent = FakeAgent()
        runner = self._runner(agent)
        observed: dict = {}

        def spy(prompt: str) -> str:
            observed["active"] = runner._active.copy()
            return "ok"

        agent.run_turn = spy
        job = {"name": "digest", "type": "cron", "schedule": "* * * * *", "prompt": "p"}

        with patch("core.cron_runner._send_telegram"):
            _run_job(job, runner, "token", "42")

        self.assertIn("digest", observed["active"])
        self.assertEqual(runner._active, {})

    def test_reminder_is_delivered_as_is_without_agent(self):
        agent = FakeAgent(reply="модель не должна запускаться")
        factory_calls: list[int] = []

        def factory():
            factory_calls.append(1)
            return agent

        runner = CronRunner(factory, "token", "42", jobs_file=self.jobs_file)
        job = {
            "name": "oven",
            "type": "cron",
            "schedule": "* * * * *",
            "kind": "reminder",
            "prompt": "выключить духовку",
        }

        with patch("core.cron_runner._send_telegram") as send:
            _run_job(job, runner, "token", "42")

        self.assertEqual(factory_calls, [])
        self.assertEqual(agent.prompts, [])
        send.assert_called_once()
        self.assertIn("выключить духовку", send.call_args.args[2])

    def test_reminder_text_is_html_escaped(self):
        runner = self._runner(FakeAgent())
        job = {
            "name": "alert",
            "type": "cron",
            "schedule": "* * * * *",
            "kind": "reminder",
            "prompt": "<b>не жирный</b> & co",
        }

        with patch("core.cron_runner._send_telegram") as send:
            _run_job(job, runner, "token", "42")

        payload = send.call_args.args[2]
        self.assertIn("&lt;b&gt;не жирный&lt;/b&gt; &amp; co", payload)
        self.assertNotIn("<b>не жирный</b>", payload)

    def test_once_reminder_is_removed_and_delivered(self):
        with open(self.jobs_file, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "name": "oven",
                        "type": "once",
                        "run_at": "2026-01-01 10:00",
                        "kind": "reminder",
                        "prompt": "выключить духовку",
                    }
                ],
                f,
            )
        runner = self._runner(FakeAgent())
        job = {
            "name": "oven",
            "type": "once",
            "run_at": "2026-01-01 10:00",
            "kind": "reminder",
            "prompt": "выключить духовку",
        }

        with patch("core.cron_runner._send_telegram") as send:
            _run_job(job, runner, "token", "42")

        self.assertEqual(_load(self.jobs_file), [])
        self.assertIn("выключить духовку", send.call_args.args[2])


class CancelJobTests(TestCase):
    def test_cancel_active_job_calls_agent_cancel(self):
        agent = FakeAgent()
        runner = CronRunner(lambda: agent, "token", "42")
        runner._register_active("digest", agent)

        self.assertTrue(runner.cancel_job("digest"))
        self.assertEqual(agent.cancel_reasons, ["отмена cron-задачи 'digest'"])

    def test_cancel_unknown_job_returns_false(self):
        runner = CronRunner(lambda: FakeAgent(), "token", "42")
        self.assertFalse(runner.cancel_job("ghost"))

    def test_stop_cancels_active_jobs_and_shuts_scheduler(self):
        agent = FakeAgent()
        runner = CronRunner(lambda: agent, "token", "42")
        runner._register_active("digest", agent)
        runner.scheduler = Mock()
        runner.scheduler.running = True

        runner.stop()

        self.assertEqual(agent.cancel_reasons, ["отмена cron-задачи 'digest'"])
        runner.scheduler.shutdown.assert_called_once_with(wait=False)


class ReloadJobsTests(TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.jobs_file = os.path.join(self._tmp.name, "jobs.json")
        self.addCleanup(self._tmp.cleanup)

    def test_reload_reads_configured_jobs_file(self):
        with open(self.jobs_file, "w", encoding="utf-8") as f:
            json.dump([{"name": "daily", "type": "cron", "schedule": "0 9 * * *", "prompt": "p"}], f)
        runner = CronRunner(lambda: FakeAgent(), "token", "42", jobs_file=self.jobs_file)

        runner._reload_jobs()

        self.assertIn("daily", {job.id for job in runner.scheduler.get_jobs()})

    def test_invalid_schedule_is_skipped_not_fatal(self):
        with open(self.jobs_file, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {"name": "broken", "type": "cron", "schedule": "not a cron", "prompt": "p"},
                    {"name": "fine", "type": "cron", "schedule": "0 9 * * *", "prompt": "p"},
                ],
                f,
            )
        runner = CronRunner(lambda: FakeAgent(), "token", "42", jobs_file=self.jobs_file)

        runner._reload_jobs()

        ids = {job.id for job in runner.scheduler.get_jobs()}
        self.assertIn("fine", ids)
        self.assertNotIn("broken", ids)

    def test_removed_jobs_disappear_from_scheduler(self):
        with open(self.jobs_file, "w", encoding="utf-8") as f:
            json.dump([{"name": "daily", "type": "cron", "schedule": "0 9 * * *", "prompt": "p"}], f)
        runner = CronRunner(lambda: FakeAgent(), "token", "42", jobs_file=self.jobs_file)
        runner._reload_jobs()

        with open(self.jobs_file, "w", encoding="utf-8") as f:
            json.dump([], f)
        runner._reload_jobs()

        self.assertNotIn("daily", {job.id for job in runner.scheduler.get_jobs()})


if __name__ == "__main__":
    import unittest

    unittest.main()
