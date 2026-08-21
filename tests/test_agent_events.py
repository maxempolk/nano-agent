from unittest import TestCase
from unittest.mock import patch

from core.agent import SIMPLE_SYSTEM, Agent
from core.model_router import AppleModelRouter, ModelRoute
from core.prompts import PROFILES, build_prompt_set
from core.tools.base import ToolRegistry
from tests.test_agent_forced_search import FakeBashTool, _completion, _tool_completion


def _kinds(events) -> list[str]:
    return [event.kind for event in events]


class AgentEventTests(TestCase):
    def test_plain_turn_emits_full_event_sequence(self):
        agent = Agent(None, "system", "SYSTEM")
        events: list = []
        agent.events.subscribe(events.append)

        with patch("core.agent.call_llm", return_value=_completion("привет")):
            reply = agent.run_turn("здравствуй")

        self.assertEqual(reply, "привет")
        self.assertEqual(
            _kinds(events),
            ["run_started", "route_selected", "model_started", "model_completed", "run_completed"],
        )
        completed = events[-1]
        self.assertEqual(completed.model_calls, 1)
        self.assertEqual(completed.tool_calls, 0)
        self.assertIn("привет", completed.reply_preview)

    def test_tool_turn_emits_tool_events_without_raw_output(self):
        bash = FakeBashTool()
        agent = Agent(None, "system", "SYSTEM", registry=ToolRegistry([bash]))
        events: list = []
        agent.events.subscribe(events.append)

        with patch(
            "core.agent.call_llm",
            side_effect=[
                _tool_completion("execute_bash", '{"command":"ls -la"}'),
                _completion("список файлов"),
            ],
        ):
            agent.run_turn("покажи файлы")

        kinds = _kinds(events)
        self.assertIn("tool_started", kinds)
        self.assertIn("tool_completed", kinds)
        started = events[kinds.index("tool_started")]
        completed = events[kinds.index("tool_completed")]
        self.assertEqual(started.name, "execute_bash")
        self.assertIn("ls -la", started.args_summary)
        self.assertTrue(completed.ok)
        self.assertEqual(completed.error_code, "")

    def test_model_error_emits_run_failed_and_honest_reply(self):
        agent = Agent(None, "system", "SYSTEM")
        events: list = []
        agent.events.subscribe(events.append)

        with patch("core.agent.call_llm", side_effect=RuntimeError("bridge offline")):
            reply = agent.run_turn("запрос")

        self.assertIn("Внутренняя ошибка агента", reply)
        self.assertIn("bridge offline", reply)
        self.assertEqual(_kinds(events)[-1], "run_failed")

    def test_final_answer_is_never_empty(self):
        agent = Agent(None, "system", "SYSTEM")

        with patch("core.agent.call_llm", return_value=_completion("")):
            reply = agent.run_turn("запрос без ответа")

        self.assertTrue(reply.strip())

    def test_compaction_emits_context_compacted(self):
        agent = Agent(
            None, "system", "BASE", compact_keep_messages=2, token_budget=500,
        )
        for i in range(6):
            agent.messages += [
                {"role": "user", "content": f"вопрос {i} " + "x" * 150},
                {"role": "assistant", "content": f"ответ {i} " + "y" * 150},
            ]
        agent.messages.append({"role": "user", "content": "текущий"})
        events: list = []
        agent.events.subscribe(events.append)

        with patch("core.agent.call_llm", return_value=_completion("сжатая память")):
            before, after, compacted = agent.compact_context()

        self.assertTrue(compacted)
        compact_events = [event for event in events if event.kind == "context_compacted"]
        self.assertEqual(len(compact_events), 1)
        self.assertEqual(compact_events[0].before_tokens, before)
        self.assertEqual(compact_events[0].after_tokens, after)


class SimpleModeAndForcedModeTests(TestCase):
    def _prompts(self):
        return build_prompt_set("mini", system_info="info")

    def test_hybrid_simple_turn_uses_minimal_prompt_and_no_tools(self):
        bash = FakeBashTool()
        local = ModelRoute("local", "system", "FULL-LOCAL", 3000, fallback_model="pcc")
        pcc = ModelRoute("pcc", "pcc", "FULL-PCC", 12000, fallback_model="system")
        router = AppleModelRouter(local, pcc, mode="hybrid")
        agent = Agent(
            None, "system", "FULL-LOCAL", registry=ToolRegistry([bash]),
            route_selector=router.select,
        )

        with patch("core.agent.call_llm", return_value=_completion("42")) as llm:
            reply = agent.run_turn("сколько будет 6*7?")

        self.assertEqual(reply, "42")
        sent_messages = llm.call_args.args[2]
        self.assertEqual(sent_messages[0]["content"], SIMPLE_SYSTEM)
        self.assertEqual(llm.call_args.args[3], [], "tools не должны передаваться")

    def test_forced_local_mode_keeps_tools_and_full_prompt(self):
        bash = FakeBashTool()
        prompts = self._prompts()
        local = ModelRoute("local", "system", prompts.agent, 3000)
        pcc = ModelRoute("pcc", "pcc", prompts.agent, 12000)
        router = AppleModelRouter(local, pcc, mode="local")
        agent = Agent(
            None, "system", prompts.agent, registry=ToolRegistry([bash]),
            route_selector=router.select,
        )

        with patch("core.agent.call_llm", return_value=_completion("готово")) as llm:
            agent.run_turn("простой вопрос")

        sent_messages = llm.call_args.args[2]
        self.assertEqual(sent_messages[0]["content"], prompts.agent)
        self.assertNotEqual(sent_messages[0]["content"], SIMPLE_SYSTEM)
        self.assertEqual(len(llm.call_args.args), 4, "tools должны передаваться")
        tool_names = {tool["function"]["name"] for tool in llm.call_args.args[3]}
        self.assertIn("execute_bash", tool_names)

    def test_forced_pcc_mode_keeps_tools(self):
        bash = FakeBashTool()
        local = ModelRoute("local", "system", "LOCAL", 3000)
        pcc = ModelRoute("pcc", "pcc", "PCC-SYSTEM", 12000)
        router = AppleModelRouter(local, pcc, mode="pcc")
        agent = Agent(
            None, "pcc", "PCC-SYSTEM", registry=ToolRegistry([bash]),
            route_selector=router.select,
        )

        with patch("core.agent.call_llm", return_value=_completion("ок")) as llm:
            agent.run_turn("простой вопрос")

        self.assertEqual(len(llm.call_args.args), 4)
        self.assertEqual(agent.model, "pcc")

    def test_profile_prompts_are_still_available(self):
        self.assertIn("mini", PROFILES)
        self.assertIn("full", PROFILES)


if __name__ == "__main__":
    import unittest

    unittest.main()
