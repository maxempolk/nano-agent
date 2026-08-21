from unittest import TestCase
from unittest.mock import patch

from core.agent import Agent
from core.tools.base import ToolRegistry
from tests.test_agent_forced_search import (
    FakeBashTool,
    FakeWebSearchTool,
    _completion,
    _tool_completion,
)


def _work_agent(bash, web) -> Agent:
    return Agent(None, "model", "SYSTEM", registry=ToolRegistry([bash, web]), work_mode=True)


class WorkModeGuardTests(TestCase):
    def test_search_is_rejected_until_plan_is_created(self):
        bash = FakeBashTool()
        web = FakeWebSearchTool()
        agent = _work_agent(bash, web)
        scripted = [
            _tool_completion("web_search", '{"query":"рынок"}'),
            _tool_completion(
                "execute_bash", '{"command":"mkdir -p work && echo план > work/plan.md"}'
            ),
            _tool_completion("web_search", '{"query":"рынок"}'),
            _completion("готово"),
        ]

        with patch("core.agent.call_llm", side_effect=scripted):
            reply = agent.run_turn("исследуй рынок")

        self.assertEqual(reply, "готово")
        self.assertEqual(len(bash.calls), 1)
        self.assertEqual(len(web.calls), 1, "поиск до plan.md не должен выполняться")

    def test_forced_search_is_disabled_in_work_mode(self):
        bash = FakeBashTool()
        web = FakeWebSearchTool()
        agent = _work_agent(bash, web)

        with patch("core.agent.call_llm", return_value=_completion("план готов")):
            reply = agent.run_turn("поищи в сети последние новости")

        self.assertEqual(reply, "план готов")
        self.assertEqual(web.calls, [], "форс-поиск в work-режиме не запускается")

    def test_normal_mode_still_allows_search_first(self):
        bash = FakeBashTool()
        web = FakeWebSearchTool()
        agent = Agent(None, "model", "SYSTEM", registry=ToolRegistry([bash, web]))

        with patch("core.agent.call_llm", return_value=_completion("ответ")) as call:
            agent.run_turn("исследуй рынок")

        self.assertEqual(len(web.calls), 1)
        call.assert_called_once()


if __name__ == "__main__":
    import unittest

    unittest.main()
