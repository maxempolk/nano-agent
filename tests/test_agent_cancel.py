import threading
import time
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from pydantic import BaseModel
from test_agent_forced_search import _tool_completion

from core.agent import Agent
from core.events import RunCancelled
from core.tools.base import Tool, ToolContext, ToolRegistry, ToolResult


class SlowInput(BaseModel):
    seconds: float = 2.0


class SlowTool(Tool):
    name = "slow"
    description = "медленный инструмент"
    input_model = SlowInput
    timeout = 10.0

    def __init__(self):
        self.started = threading.Event()

    def execute(self, args: SlowInput, ctx: ToolContext) -> ToolResult:
        self.started.set()
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            ctx.raise_if_cancelled()
            time.sleep(0.02)
        return ToolResult(content="done")


class AgentCancelTests(TestCase):
    def _agent(self, tool: SlowTool) -> Agent:
        return Agent(None, "system", "SYSTEM", registry=ToolRegistry([tool]))

    def test_cancel_during_tool_stops_the_run(self):
        tool = SlowTool()
        agent = self._agent(tool)
        events: list = []
        agent.events.subscribe(events.append)

        def cancel_when_started():
            tool.started.wait(5)
            agent.cancel("Ctrl+C")

        canceler = threading.Thread(target=cancel_when_started)
        canceler.start()

        with patch(
            "core.agent.call_llm",
            return_value=_tool_completion("slow", '{"seconds": 5}'),
        ):
            started = time.monotonic()
            reply = agent.run_turn("запусти медленную задачу")
            elapsed = time.monotonic() - started

        canceler.join(timeout=5)
        self.assertIn("отменено", reply.lower())
        self.assertLess(elapsed, 4)
        cancelled = [event for event in events if isinstance(event, RunCancelled)]
        self.assertEqual(len(cancelled), 1)
        self.assertEqual(cancelled[0].reason, "Ctrl+C")

    def test_history_is_balanced_after_cancel(self):
        tool = SlowTool()
        agent = self._agent(tool)

        def cancel_when_started():
            tool.started.wait(5)
            agent.cancel()

        canceler = threading.Thread(target=cancel_when_started)
        canceler.start()

        with patch(
            "core.agent.call_llm",
            return_value=_tool_completion("slow", '{"seconds": 5}'),
        ):
            agent.run_turn("запусти")

        canceler.join(timeout=5)
        roles = [
            message.get("role")
            for message in agent.messages
            if isinstance(message, dict)
        ]
        # assistant with tool_calls must be followed by a tool result
        self.assertIn("tool", roles)
        last_tool = next(
            message
            for message in reversed(agent.messages)
            if isinstance(message, dict) and message.get("role") == "tool"
        )
        self.assertIn("отменён", last_tool["content"])

    def test_run_in_progress_flag(self):
        tool = SlowTool()
        agent = self._agent(tool)
        observed: list = []

        def cancel_when_started():
            tool.started.wait(5)
            observed.append(agent.run_in_progress)
            agent.cancel()

        canceler = threading.Thread(target=cancel_when_started)
        canceler.start()

        with patch(
            "core.agent.call_llm",
            return_value=_tool_completion("slow", '{"seconds": 5}'),
        ):
            agent.run_turn("запусти")

        canceler.join(timeout=5)
        self.assertEqual(observed, [True])
        self.assertFalse(agent.run_in_progress)

    def test_cancel_before_next_step_stops_loop(self):
        agent = Agent(None, "system", "SYSTEM")

        def cancel_soon():
            time.sleep(0.05)
            agent.cancel("стоп")

        canceler = threading.Thread(target=cancel_soon)
        canceler.start()

        def slow_model(*args, **kwargs):
            time.sleep(0.15)
            message = SimpleNamespace(content="ответ", tool_calls=None)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="stop")]
            )

        with patch("core.agent.call_llm", side_effect=slow_model):
            reply = agent.run_turn("долго думаю")

        canceler.join(timeout=5)
        self.assertIn("отменено", reply.lower())


if __name__ == "__main__":
    import unittest

    unittest.main()
