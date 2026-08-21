from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from core.agent import Agent


def _completion(text: str):
    message = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice])


class AutoMemoryRecallTests(TestCase):
    def _agent(self, lookup) -> Agent:
        return Agent(None, "model", "SYSTEM", memory_lookup=lookup)

    def test_relevant_notes_are_injected_into_request(self):
        agent = self._agent(lambda q: "#1 [2026-08-21]: Меня зовут Максим")
        captured: dict = {}

        def fake_call(client, model, messages, tools):
            captured["messages"] = [dict(m) for m in messages]
            return _completion("Максим")

        with patch("core.agent.call_llm", side_effect=fake_call):
            agent.run_turn("как меня звать?")

        memory_blocks = [
            m for m in captured["messages"] if "долговременная память" in m["content"]
        ]
        self.assertEqual(len(memory_blocks), 1)
        self.assertIn("Максим", memory_blocks[0]["content"])

    def test_memory_block_does_not_accumulate_across_turns(self):
        replies = iter(["#1: имя Максим", ""])
        agent = self._agent(lambda q: next(replies))

        with patch("core.agent.call_llm", return_value=_completion("ok")):
            agent.run_turn("первый вопрос")
            agent.run_turn("второй вопрос")

        leftovers = [
            m for m in agent.messages if "долговременная память" in m.get("content", "")
        ]
        self.assertEqual(leftovers, [])

    def test_empty_lookup_injects_nothing(self):
        agent = self._agent(lambda q: "")
        captured: dict = {}

        def fake_call(client, model, messages, tools):
            captured["messages"] = [dict(m) for m in messages]
            return _completion("ok")

        with patch("core.agent.call_llm", side_effect=fake_call):
            agent.run_turn("привет")

        self.assertFalse(
            any("долговременная память" in m["content"] for m in captured["messages"])
        )

    def test_lookup_error_does_not_break_turn(self):
        def broken(q):
            raise RuntimeError("notes file gone")

        agent = self._agent(broken)
        with patch("core.agent.call_llm", return_value=_completion("жив")):
            reply = agent.run_turn("привет")

        self.assertEqual(reply, "жив")

    def test_without_lookup_nothing_is_injected(self):
        agent = Agent(None, "model", "SYSTEM")
        captured: dict = {}

        def fake_call(client, model, messages, tools):
            captured["messages"] = [dict(m) for m in messages]
            return _completion("ok")

        with patch("core.agent.call_llm", side_effect=fake_call):
            agent.run_turn("привет")

        self.assertFalse(
            any("долговременная память" in m["content"] for m in captured["messages"])
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
