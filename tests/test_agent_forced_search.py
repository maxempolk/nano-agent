from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from core.agent import Agent, _forced_web_search_query, _forced_web_search_depth


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

    def test_hallucinated_search_after_forced_search_is_blocked_and_recovered(self):
        web = FakeWebSearch()
        agent = Agent(None, "system", "SYSTEM", extra_tools=[web])
        ghost_call = _tool_completion(
            "web_search",
            '{"query":"repeat the whole research","depth":"deep"}',
        )

        with patch(
            "core.agent.call_llm",
            side_effect=[ghost_call, _completion("Один итоговый ответ")],
        ) as llm:
            reply = agent.run_turn("подробно исследуй уровень жизни в Норвегии")

        self.assertEqual(reply, "Один итоговый ответ")
        web.execute.assert_called_once()
        self.assertEqual(len(llm.call_args_list), 2)
        self.assertEqual(llm.call_args_list[1].args[3], [])
        recovery_messages = llm.call_args_list[1].args[2]
        self.assertFalse(any(message.get("role") == "tool" for message in recovery_messages))
        self.assertEqual([message["role"] for message in recovery_messages], ["system", "user"])
        self.assertIn("GPT-5.4", recovery_messages[1]["content"])

    def test_empty_protocol_recovery_returns_tool_evidence_instead_of_silence(self):
        web = FakeWebSearch()
        agent = Agent(None, "system", "SYSTEM", extra_tools=[web])

        with patch(
            "core.agent.call_llm",
            side_effect=[
                _tool_completion("web_search", '{"query":"repeat","depth":"deep"}'),
                _completion(""),
            ],
        ):
            reply = agent.run_turn("подробно исследуй Норвегию")

        self.assertTrue(reply.strip())
        self.assertIn("GPT-5.4", reply)
        web.execute.assert_called_once()

    def test_repeated_tool_call_during_recovery_returns_tool_evidence(self):
        web = FakeWebSearch()
        agent = Agent(None, "system", "SYSTEM", extra_tools=[web])
        ghost = _tool_completion("web_search", '{"query":"repeat","depth":"deep"}')

        with patch("core.agent.call_llm", side_effect=[ghost, ghost]):
            reply = agent.run_turn("подробно исследуй Норвегию")

        self.assertIn("GPT-5.4", reply)
        web.execute.assert_called_once()

    def test_unoffered_tool_is_never_executed(self):
        web = FakeWebSearch()
        agent = Agent(None, "system", "SYSTEM", extra_tools=[web])
        shell = Mock()
        agent.handlers["not_offered"] = shell

        with patch(
            "core.agent.call_llm",
            side_effect=[
                _tool_completion("not_offered", '{"command":"unsafe"}'),
                _completion("recovered"),
            ],
        ):
            reply = agent.run_turn("hello")

        self.assertEqual(reply, "recovered")
        shell.assert_not_called()

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
        self.assertNotIn("execute_bash", names)

    def test_explicit_user_deep_intent_is_preserved(self):
        web = FakeWebSearch()
        agent = Agent(None, "system", "SYSTEM", extra_tools=[web])

        with patch(
            "core.agent.call_llm",
            return_value=_completion("done"),
        ) as llm:
            agent.run_turn("подробно исследуй реформу коммун")

        web.execute.assert_called_once_with(
            query="подробно исследуй реформу коммун",
            depth="deep",
        )
        self.assertEqual(llm.call_args.args[3], [])

    def test_research_request_from_logs_forces_one_deep_search(self):
        request = (
            "Подробно исследуй уровень жизни в Норвегии: доходы, стоимость "
            "жизни, жильё, безопасность и удовлетворённость жизнью. Сравни "
            "данные из нескольких источников и укажи противоречия."
        )

        self.assertEqual(_forced_web_search_query(request), request)
        self.assertEqual(_forced_web_search_depth(request), "deep")

    def test_batched_web_search_executes_only_first_call_and_blocks_bash(self):
        web = FakeWebSearch()
        agent = Agent(None, "system", "SYSTEM", extra_tools=[web])
        calls = [
            SimpleNamespace(
                id="search-1",
                function=SimpleNamespace(
                    name="web_search",
                    arguments='{"query":"Norway income","depth":"auto"}',
                ),
            ),
            SimpleNamespace(
                id="search-2",
                function=SimpleNamespace(
                    name="web_search",
                    arguments='{"query":"Norway housing","depth":"auto"}',
                ),
            ),
            SimpleNamespace(
                id="bash-1",
                function=SimpleNamespace(
                    name="execute_bash",
                    arguments='{"command":"curl https://example.com"}',
                ),
            ),
        ]
        batch = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="", tool_calls=calls),
                finish_reason="tool_calls",
            )],
        )
        bash_execute = Mock()
        agent.handlers["execute_bash"] = bash_execute

        with patch(
            "core.agent.call_llm",
            side_effect=[batch, _completion("done")],
        ) as llm:
            reply = agent.run_turn("расскажи о Норвегии")

        self.assertEqual(reply, "done")
        web.execute.assert_called_once_with(query="Norway income", depth="auto")
        bash_execute.assert_not_called()
        self.assertEqual(llm.call_args_list[1].args[3], [])

    def test_invalid_first_batched_search_still_disables_tools(self):
        web = FakeWebSearch()
        agent = Agent(None, "system", "SYSTEM", extra_tools=[web])
        malformed = _tool_completion("web_search", '{"query":')

        with patch(
            "core.agent.call_llm",
            side_effect=[malformed, _completion("recovered")],
        ) as llm:
            reply = agent.run_turn("расскажи о Норвегии")

        self.assertEqual(reply, "recovered")
        web.execute.assert_not_called()
        self.assertEqual(llm.call_args_list[1].args[3], [])
