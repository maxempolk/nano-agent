from unittest import TestCase
from unittest.mock import patch

from test_agent_forced_search import FakeBashTool, _completion, _tool_completion

from core.agent import Agent
from core.budget import BudgetLimits
from core.events import RunFailed
from core.tools.base import ToolRegistry


def _agent(tools, **budget_overrides) -> tuple[Agent, FakeBashTool]:
    bash = FakeBashTool()
    registry = ToolRegistry([bash] if tools else [])
    agent = Agent(None, "system", "SYSTEM", registry=registry, budget_limits=BudgetLimits(**budget_overrides))
    return agent, bash


class AgentBudgetTests(TestCase):
    def test_step_limit_stops_with_honest_message(self):
        agent, bash = _agent(True, max_steps=2)
        calls = iter(
            _tool_completion("execute_bash", f'{{"command":"cmd {i}"}}') for i in range(10)
        )

        with patch("core.agent.call_llm", side_effect=lambda *args, **kwargs: next(calls)):
            reply = agent.run_turn("сделай много шагов")

        self.assertIn("Остановлено", reply)
        self.assertIn("шагов", reply)
        self.assertIn("не завершён", reply)
        self.assertLessEqual(len(bash.calls), 3)

    def test_model_call_limit_stops_the_turn(self):
        agent, _ = _agent(True, max_model_calls=1)
        calls = iter(
            _tool_completion("execute_bash", f'{{"command":"cmd {i}"}}') for i in range(10)
        )

        with patch("core.agent.call_llm", side_effect=lambda *args, **kwargs: next(calls)):
            reply = agent.run_turn("работай")

        self.assertIn("Остановлено", reply)
        self.assertIn("вызовов модели", reply)

    def test_tool_call_limit_stops_the_turn(self):
        agent, bash = _agent(True, max_tool_calls=1, max_steps=10)
        calls = iter(
            _tool_completion("execute_bash", f'{{"command":"cmd {i}"}}') for i in range(10)
        )

        with patch("core.agent.call_llm", side_effect=lambda *args, **kwargs: next(calls)):
            reply = agent.run_turn("работай")

        self.assertIn("Остановлено", reply)
        self.assertIn("инструментов", reply)
        self.assertEqual(len(bash.calls), 1)

    def test_identical_tool_call_repetition_is_blocked(self):
        agent, bash = _agent(True, max_identical_calls=1, max_steps=10)
        identical = _tool_completion("execute_bash", '{"command":"same"}')

        with patch("core.agent.call_llm", return_value=identical):
            reply = agent.run_turn("повторяй одно и то же")

        self.assertIn("Остановлено", reply)
        self.assertEqual(bash.calls, ["same"])

    def test_consecutive_errors_stop_the_turn(self):
        class FailingBash(FakeBashTool):
            def execute(self, args, ctx):
                from core.tools.base import ToolResult

                self.calls.append(args.command)
                return ToolResult.failure("диск недоступен", code="tool_failed")

        bash = FailingBash()
        agent = Agent(
            None,
            "system",
            "SYSTEM",
            registry=ToolRegistry([bash]),
            budget_limits=BudgetLimits(max_consecutive_errors=2, max_steps=10),
        )
        calls = iter(
            _tool_completion("execute_bash", f'{{"command":"try {i}"}}') for i in range(10)
        )

        with patch("core.agent.call_llm", side_effect=lambda *args, **kwargs: next(calls)):
            reply = agent.run_turn("пробуй снова")

        self.assertIn("Остановлено", reply)
        self.assertIn("ошибки", reply)
        self.assertEqual(len(bash.calls), 2)

    def test_tool_output_is_capped_by_run_budget(self):
        class VerboseBash(FakeBashTool):
            def execute(self, args, ctx):
                from core.tools.base import ToolResult

                return ToolResult(content="x" * 5000)

        agent = Agent(
            None,
            "system",
            "SYSTEM",
            registry=ToolRegistry([VerboseBash()]),
            budget_limits=BudgetLimits(max_tool_output_chars=500),
        )

        with patch(
            "core.agent.call_llm",
            side_effect=[
                _tool_completion("execute_bash", '{"command":"verbose"}'),
                _completion("готово"),
            ],
        ):
            reply = agent.run_turn("запусти")

        self.assertEqual(reply, "готово")
        tool_message = next(
            m for m in agent.messages if isinstance(m, dict) and m.get("role") == "tool"
        )
        self.assertIn("обрезано", tool_message["content"])
        self.assertLess(len(tool_message["content"]), 700)

    def test_wall_time_limit_stops_with_honest_message(self):
        agent, _ = _agent(True, max_wall_seconds=0.000000001)

        with patch("core.agent.call_llm", return_value=_completion("не должно дойти")):
            reply = agent.run_turn("быстрый запрос")

        self.assertIn("Остановлено", reply)
        self.assertIn("время", reply)

    def test_budget_stop_emits_run_failed_event(self):
        agent, _ = _agent(True, max_steps=1)
        events: list = []
        agent.events.subscribe(events.append)
        calls = iter(
            _tool_completion("execute_bash", f'{{"command":"cmd {i}"}}') for i in range(10)
        )

        with patch("core.agent.call_llm", side_effect=lambda *args, **kwargs: next(calls)):
            agent.run_turn("много шагов")

        failed = [event for event in events if isinstance(event, RunFailed)]
        self.assertEqual(len(failed), 1)

    def test_budget_exceeded_never_masked_by_plausible_answer(self):
        agent, _ = _agent(True, max_model_calls=1)
        calls = iter(
            _tool_completion("execute_bash", f'{{"command":"cmd {i}"}}') for i in range(10)
        )

        with patch("core.agent.call_llm", side_effect=lambda *args, **kwargs: next(calls)):
            reply = agent.run_turn("работай")

        self.assertNotIn("готово", reply.lower())
        self.assertTrue(reply.startswith("⚠️"))


if __name__ == "__main__":
    import unittest

    unittest.main()
