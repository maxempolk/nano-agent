from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from core.agent import Agent, _forced_web_search_query


def _completion(text: str):
    message = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice])


def _tool_completion(name: str, arguments: str):
    call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    message = SimpleNamespace(content="", tool_calls=[call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice])


class FakeWebSearch:
    SCHEMA = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }

    def __init__(self):
        self.execute = Mock(return_value="[1] https://openai.com\nGPT-5.4")
        self.last_query = "latest GPT version"


class AgentForcedSearchTests(TestCase):
    def test_detects_current_version_query(self):
        self.assertEqual(
            _forced_web_search_query("какая последняя версия gpt?"),
            "какая последняя версия gpt?",
        )

    def test_generic_followup_uses_previous_user_topic(self):
        self.assertEqual(
            _forced_web_search_query("поищи в сети", "какая последняя версия gpt?"),
            "какая последняя версия gpt?",
        )

    def test_forced_search_runs_before_model_and_cannot_repeat_in_same_turn(self):
        web = FakeWebSearch()
        callback = Mock()
        agent = Agent(None, "system", "SYSTEM", extra_tools=[web])

        with patch("core.agent.call_llm", return_value=_completion("GPT-5.4")) as llm:
            reply = agent.run_turn(
                "какая последняя версия gpt?",
                on_tool_call=callback,
            )

        self.assertEqual(reply, "GPT-5.4")
        web.execute.assert_called_once_with(
            query="какая последняя версия gpt?",
            depth="auto",
        )
        callback.assert_called_once()
        self.assertEqual(agent.last_search_query, "latest GPT version")
        self.assertTrue(any(message.get("role") == "tool" for message in agent.messages))
        self.assertEqual(llm.call_args.args[3], [])

    def test_model_cannot_escalate_simple_question_to_deep_or_search_twice(self):
        web = FakeWebSearch()
        agent = Agent(None, "system", "SYSTEM", extra_tools=[web])
        first = _tool_completion(
            "web_search",
            '{"query":"Norway municipalities official statistics","depth":"deep"}',
        )

        with patch(
            "core.agent.call_llm",
            side_effect=[first, _completion("357")],
        ) as llm:
            reply = agent.run_turn("сколько коммун в Норвегии?")

        self.assertEqual(reply, "357")
        web.execute.assert_called_once_with(
            query="Norway municipalities official statistics",
            depth="normal",
        )
        second_turn_tools = llm.call_args_list[1].args[3]
        names = [tool["function"]["name"] for tool in second_turn_tools]
        self.assertNotIn("web_search", names)
        self.assertIn("execute_bash", names)

    def test_explicit_user_deep_intent_is_preserved(self):
        web = FakeWebSearch()
        agent = Agent(None, "system", "SYSTEM", extra_tools=[web])
        first = _tool_completion(
            "web_search",
            '{"query":"municipality reform research","depth":"deep"}',
        )

        with patch(
            "core.agent.call_llm",
            side_effect=[first, _completion("done")],
        ):
            agent.run_turn("подробно исследуй реформу коммун")

        web.execute.assert_called_once_with(
            query="municipality reform research",
            depth="deep",
        )
