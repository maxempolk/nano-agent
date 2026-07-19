from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from core.agent import Agent, _estimate_tokens


def _completion(text: str):
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


class AgentCompactTests(TestCase):
    def _long_agent(self) -> Agent:
        agent = Agent(
            None,
            "system",
            "BASE",
            compact_keep_messages=4,
            token_budget=700,
            compact_prompt="COMPACT",
        )
        for i in range(6):
            agent.messages += [
                {"role": "user", "content": f"old question {i} " + "x" * 180},
                {"role": "assistant", "content": f"old answer {i} " + "y" * 180},
            ]
        agent.messages.append({"role": "user", "content": "current request"})
        return agent

    def test_compacts_old_messages_into_memory(self):
        agent = self._long_agent()
        before = _estimate_tokens(agent._context_messages(), agent.tools)

        with patch("core.agent.call_llm", return_value=_completion("summary one")):
            agent._compact_if_needed()

        after = _estimate_tokens(agent._context_messages(), agent.tools)
        self.assertLess(after, before)
        self.assertEqual(agent.memory, "summary one")
        self.assertEqual(agent.messages[-1]["content"], "current request")
        context = agent._context_messages()
        self.assertEqual(sum(m["role"] == "system" for m in context), 1)
        self.assertIn("Conversation memory:\nsummary one", context[0]["content"])

    def test_repeated_compact_includes_previous_memory(self):
        agent = self._long_agent()
        agent.memory = "previous memory"

        with patch("core.agent.call_llm", return_value=_completion("new memory")) as mocked:
            agent._compact_if_needed()

        transcript = mocked.call_args.args[2][1]["content"]
        self.assertIn("[memory] previous memory", transcript)
        self.assertEqual(agent.memory, "new memory")

    def test_clear_removes_memory_and_history(self):
        agent = self._long_agent()
        agent.memory = "summary"

        agent.clear_context()

        self.assertEqual(agent.memory, "")
        self.assertEqual(agent.messages, [{"role": "system", "content": "BASE"}])

    def test_context_usage_and_forced_compact(self):
        agent = self._long_agent()
        agent.token_budget = 10_000
        used, limit = agent.context_usage()
        self.assertGreater(used, 0)
        self.assertEqual(limit, 10_000)

        with patch("core.agent.call_llm", return_value=_completion("forced memory")):
            before, after, compacted = agent.compact_context()

        self.assertTrue(compacted)
        self.assertLess(after, before)
        self.assertEqual(agent.memory, "forced memory")

    def test_falls_back_when_summarization_fails(self):
        agent = self._long_agent()

        with patch("core.agent.call_llm", side_effect=RuntimeError("offline")):
            agent._compact_if_needed()

        self.assertTrue(agent.memory)
        self.assertLessEqual(len(agent.memory), 1435)
        self.assertEqual(agent.messages[-1]["content"], "current request")


if __name__ == "__main__":
    import unittest

    unittest.main()
