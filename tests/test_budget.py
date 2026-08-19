from unittest import TestCase
from unittest.mock import patch

from core.budget import BudgetExceeded, BudgetLimits, RunBudget, call_signature


class BudgetLimitsTests(TestCase):
    def test_defaults_are_sane(self):
        limits = BudgetLimits()
        self.assertGreater(limits.max_steps, 0)
        self.assertGreater(limits.max_wall_seconds, 0)
        self.assertGreater(limits.max_tool_output_chars, 0)

    def test_rejects_non_positive_limits(self):
        with self.assertRaises(ValueError):
            BudgetLimits(max_steps=0)
        with self.assertRaises(ValueError):
            BudgetLimits(max_wall_seconds=-1)


class RunBudgetTests(TestCase):
    def test_step_limit_raises_with_kind(self):
        budget = RunBudget(BudgetLimits(max_steps=2))
        budget.consume_step()
        budget.consume_step()
        with self.assertRaises(BudgetExceeded) as caught:
            budget.consume_step()
        self.assertEqual(caught.exception.kind, "steps")

    def test_model_call_limit(self):
        budget = RunBudget(BudgetLimits(max_model_calls=1))
        budget.consume_model_call()
        with self.assertRaises(BudgetExceeded) as caught:
            budget.consume_model_call()
        self.assertEqual(caught.exception.kind, "model_calls")

    def test_tool_call_limit(self):
        budget = RunBudget(BudgetLimits(max_tool_calls=1))
        budget.consume_tool_call("a:1")
        with self.assertRaises(BudgetExceeded) as caught:
            budget.consume_tool_call("a:2")
        self.assertEqual(caught.exception.kind, "tool_calls")

    def test_identical_call_repetition_is_stopped(self):
        budget = RunBudget(BudgetLimits(max_identical_calls=2, max_tool_calls=10))
        signature = call_signature("web_search", {"query": "x"})
        budget.consume_tool_call(signature)
        budget.consume_tool_call(signature)
        with self.assertRaises(BudgetExceeded) as caught:
            budget.consume_tool_call(signature)
        self.assertEqual(caught.exception.kind, "repeated_tool_call")

    def test_changed_arguments_are_not_identical(self):
        budget = RunBudget(BudgetLimits(max_identical_calls=1, max_tool_calls=10))
        budget.consume_tool_call(call_signature("web_search", {"query": "x"}))
        budget.consume_tool_call(call_signature("web_search", {"query": "x refined"}))

    def test_wall_time_limit(self):
        budget = RunBudget(BudgetLimits(max_wall_seconds=0.01))
        with patch("core.budget.time.monotonic", return_value=budget.started_at + 5):
            with self.assertRaises(BudgetExceeded) as caught:
                budget.check_deadline()
        self.assertEqual(caught.exception.kind, "time")

    def test_consecutive_errors_stop_the_run(self):
        budget = RunBudget(BudgetLimits(max_consecutive_errors=2))
        budget.note_tool_result(ok=False, error="boom")
        with self.assertRaises(BudgetExceeded) as caught:
            budget.note_tool_result(ok=False, error="boom again")
        self.assertEqual(caught.exception.kind, "consecutive_errors")

    def test_success_resets_error_counter(self):
        budget = RunBudget(BudgetLimits(max_consecutive_errors=2))
        budget.note_tool_result(ok=False, error="boom")
        budget.note_tool_result(ok=True)
        budget.note_tool_result(ok=False, error="boom")
        self.assertEqual(budget.consecutive_errors, 1)

    def test_output_capped_with_marker(self):
        budget = RunBudget(BudgetLimits(max_tool_output_chars=100))
        capped = budget.cap_output("x" * 500)
        self.assertLessEqual(len(capped), 100 + 60)
        self.assertIn("обрезано", capped)

    def test_short_output_is_untouched(self):
        budget = RunBudget()
        self.assertEqual(budget.cap_output("short"), "short")

    def test_snapshot_reports_counters(self):
        budget = RunBudget()
        budget.consume_step()
        budget.consume_model_call()
        budget.consume_tool_call("a:1")
        snap = budget.snapshot()
        self.assertEqual(snap["steps"], 1)
        self.assertEqual(snap["model_calls"], 1)
        self.assertEqual(snap["tool_calls"], 1)


if __name__ == "__main__":
    import unittest

    unittest.main()
