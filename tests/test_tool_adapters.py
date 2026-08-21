import json
import os
import tempfile
from unittest import TestCase
from unittest.mock import Mock

from core.cancellation import CancellationToken, CancelledError
from core.tools.base import ErrorCode, ToolContext, ToolRegistry
from core.tools.cron import CronManageTool, _load
from core.tools.web_search import WebSearchInput, WebSearchToolSpec


class FakeSearchEngine:
    def __init__(self):
        self.last_query = "запрос"
        self.last_stats = {"mode": "quick"}
        self.last_result = object()
        self._cancel_token = None
        self.received_token = "unset"

    def execute(self, query: str, depth: str = "auto") -> str:
        self.received_token = self._cancel_token
        if self._cancel_token is not None:
            self._cancel_token.raise_if_cancelled()
        return f"результат для {query} ({depth})"


class WebSearchAdapterTests(TestCase):
    def test_adapter_returns_structured_result(self):
        engine = FakeSearchEngine()
        spec = WebSearchToolSpec(engine)
        result = spec.run({"query": "тест"}, ToolContext())

        self.assertTrue(result.ok)
        self.assertEqual(result.content, "результат для тест (auto)")
        self.assertEqual(result.meta["query"], "запрос")
        self.assertEqual(result.meta["mode"], "quick")
        self.assertIn("quick", result.summary)

    def test_adapter_passes_cancellation_token_to_engine(self):
        engine = FakeSearchEngine()
        spec = WebSearchToolSpec(engine)
        token = CancellationToken()
        spec.run({"query": "тест"}, ToolContext(cancel=token))
        self.assertIs(engine.received_token, token)
        self.assertIsNone(engine._cancel_token, "токен должен сбрасываться после вызова")

    def test_cancelled_token_aborts_search(self):
        engine = FakeSearchEngine()
        spec = WebSearchToolSpec(engine)
        token = CancellationToken()
        token.cancel()
        with self.assertRaises(CancelledError):
            spec.run({"query": "тест"}, ToolContext(cancel=token))

    def test_depth_is_validated_before_execution(self):
        engine = FakeSearchEngine()
        spec = WebSearchToolSpec(engine)
        result = spec.run({"query": "тест", "depth": "max"}, ToolContext())
        self.assertEqual(result.error_code, ErrorCode.INVALID_ARGUMENTS)

    def test_empty_query_is_rejected(self):
        engine = FakeSearchEngine()
        spec = WebSearchToolSpec(engine)
        result = spec.run({"query": ""}, ToolContext())
        self.assertEqual(result.error_code, ErrorCode.INVALID_ARGUMENTS)

    def test_registry_schema_matches_tool_name(self):
        registry = ToolRegistry([WebSearchToolSpec(FakeSearchEngine())])
        self.assertEqual(registry.schemas()[0]["function"]["name"], "web_search")
        properties = registry.schemas()[0]["function"]["parameters"]["properties"]
        self.assertIn("query", properties)
        self.assertIn("depth", properties)

    def test_input_model_bounds(self):
        self.assertEqual(WebSearchInput(query="ok").depth, "auto")


class CronToolAdapterTests(TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.jobs_file = os.path.join(self._tmp.name, "jobs.json")
        self.addCleanup(self._tmp.cleanup)

    def _tool(self, on_change=None) -> CronManageTool:
        return CronManageTool(jobs_file=self.jobs_file, on_change=on_change)

    def test_add_list_remove_cycle(self):
        tool = self._tool()
        added = tool.run(
            {"action": "add", "name": "digest", "prompt": "собери новости", "run_in": 60},
            ToolContext(),
        )
        self.assertTrue(added.ok, added.error)

        listed = tool.run({"action": "list"}, ToolContext())
        self.assertIn("digest", listed.content)

        removed = tool.run({"action": "remove", "name": "digest"}, ToolContext())
        self.assertTrue(removed.ok)
        self.assertEqual(_load(self.jobs_file), [])

    def test_missing_fields_become_structured_errors(self):
        tool = self._tool()
        result = tool.run({"action": "add", "name": "x"}, ToolContext())
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "validation")
        self.assertTrue(result.retryable)

    def test_unknown_action_fails_validation_without_execution(self):
        tool = self._tool()
        result = tool.run({"action": "destroy"}, ToolContext())
        self.assertEqual(result.error_code, ErrorCode.INVALID_ARGUMENTS)

    def test_duplicate_name_is_an_error(self):
        tool = self._tool()
        tool.run({"action": "add", "name": "a", "prompt": "p", "run_in": 5}, ToolContext())
        duplicate = tool.run(
            {"action": "add", "name": "a", "prompt": "p", "run_in": 5}, ToolContext()
        )
        self.assertFalse(duplicate.ok)
        self.assertFalse(duplicate.retryable)

    def test_on_change_fires_after_add(self):
        callback = Mock()
        tool = self._tool(on_change=callback)
        tool.run({"action": "add", "name": "a", "prompt": "p", "run_in": 5}, ToolContext())
        callback.assert_called_once()

    def test_broken_reload_is_reported_as_warning_not_failure(self):
        def broken():
            raise RuntimeError("scheduler offline")

        tool = self._tool(on_change=broken)
        result = tool.run(
            {"action": "add", "name": "a", "prompt": "p", "run_in": 5}, ToolContext()
        )
        self.assertTrue(result.ok)
        self.assertTrue(any("планировщик" in warning for warning in result.warnings))

    def test_jobs_are_persisted_to_configured_file(self):
        tool = self._tool()
        tool.run(
            {"action": "add", "name": "daily", "prompt": "p", "schedule": "0 9 * * *"},
            ToolContext(),
        )
        with open(self.jobs_file, encoding="utf-8") as f:
            stored = json.load(f)
        self.assertEqual(stored[0]["name"], "daily")
        self.assertEqual(stored[0]["type"], "cron")

    def test_add_reminder_stores_kind(self):
        tool = self._tool()
        result = tool.run(
            {
                "action": "add",
                "name": "oven",
                "prompt": "выключить духовку",
                "run_in": 20,
                "kind": "reminder",
            },
            ToolContext(),
        )
        self.assertTrue(result.ok, result.error)
        self.assertIn("Напоминание", result.content)
        with open(self.jobs_file, encoding="utf-8") as f:
            stored = json.load(f)
        self.assertEqual(stored[0]["kind"], "reminder")

    def test_default_kind_is_task(self):
        tool = self._tool()
        tool.run({"action": "add", "name": "a", "prompt": "p", "run_in": 5}, ToolContext())
        with open(self.jobs_file, encoding="utf-8") as f:
            stored = json.load(f)
        self.assertEqual(stored[0]["kind"], "task")

    def test_list_marks_reminders(self):
        tool = self._tool()
        tool.run(
            {"action": "add", "name": "oven", "prompt": "выключить духовку", "run_in": 20, "kind": "reminder"},
            ToolContext(),
        )
        listed = tool.run({"action": "list"}, ToolContext())
        self.assertIn("(напоминание)", listed.content)

    def test_invalid_kind_fails_validation_without_execution(self):
        tool = self._tool()
        result = tool.run(
            {"action": "add", "name": "a", "prompt": "p", "run_in": 5, "kind": "alarm"},
            ToolContext(),
        )
        self.assertEqual(result.error_code, ErrorCode.INVALID_ARGUMENTS)
        self.assertEqual(_load(self.jobs_file), [])


if __name__ == "__main__":
    import unittest

    unittest.main()
