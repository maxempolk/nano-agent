from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import httpx
from openai import BadRequestError
from pydantic import BaseModel

from core.agent import (
    Agent,
    _forced_web_search_depth,
    _forced_web_search_query,
    _validate_final_answer,
)
from core.policy import Capability
from core.tools.base import Tool, ToolContext, ToolRegistry, ToolResult


def _completion(text: str):
    message = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice])


def _tool_completion(name: str, arguments: str):
    call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    message = SimpleNamespace(role="assistant", content="", tool_calls=[call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice])


class FakeSearchInput(BaseModel):
    query: str
    depth: str = "auto"


class FakeBashInput(BaseModel):
    command: str


class FakeResearchResult:
    def evidence_text(self):
        return "Facts:\n- Подтверждённый факт [1]"

    def render_fallback(self):
        return "По результатам поиска: подтверждённый факт [1]."


class FakeWebSearchTool(Tool):
    name = "web_search"
    description = "search"
    input_model = FakeSearchInput
    capabilities = frozenset({Capability.NETWORK_READ})
    timeout = 5.0

    def __init__(self, structured=None):
        self.calls: list[dict] = []
        self.last_query = "latest GPT version"
        self.structured = structured

    def execute(self, args: FakeSearchInput, ctx: ToolContext) -> ToolResult:
        self.calls.append({"query": args.query, "depth": args.depth})
        return ToolResult(
            content="[1] https://openai.com\nGPT-5.4",
            summary="поиск завершён",
            structured=self.structured,
            meta={"query": self.last_query},
        )


class FakeBashTool(Tool):
    name = "execute_bash"
    description = "bash"
    input_model = FakeBashInput
    capabilities = frozenset({Capability.SHELL_READ})
    timeout = 5.0

    def __init__(self):
        self.calls: list[str] = []

    def execute(self, args: FakeBashInput, ctx: ToolContext) -> ToolResult:
        self.calls.append(args.command)
        return ToolResult(content="ok")


def _agent(web: FakeWebSearchTool, model: str = "system", fallback: str | None = None,
           extra_tools: list[Tool] | None = None) -> Agent:
    tools: list[Tool] = [web]
    if extra_tools:
        tools.extend(extra_tools)
    return Agent(None, model, "SYSTEM", registry=ToolRegistry(tools), model_fallback=fallback)


class AgentForcedSearchTests(TestCase):
    def test_final_answer_validator_rejects_json_and_wrong_language(self):
        self.assertEqual(
            _validate_final_answer('```json\n{"answer":"x"}\n```', "ответь по-русски")[1],
            "code_fence",
        )
        self.assertEqual(
            _validate_final_answer("English answer only", "ответь по-русски")[1],
            "language_mismatch",
        )
        self.assertTrue(
            _validate_final_answer("Краткий вывод: данные подтверждены.", "ответь по-русски")[0]
        )

    def test_final_answer_cannot_hide_partial_coverage(self):
        valid, reason = _validate_final_answer(
            "Это лучший вариант по всем критериям.",
            "сравни варианты",
            require_partial=True,
        )
        self.assertFalse(valid)
        self.assertEqual(reason, "partial_coverage_hidden")
        self.assertTrue(
            _validate_final_answer(
                "Исследование частичное: не удалось проверить совместимость.",
                "сравни варианты",
                require_partial=True,
            )[0]
        )

    def test_json_from_pcc_is_rewritten_by_local_finalizer(self):
        web = FakeWebSearchTool(structured=FakeResearchResult())
        agent = _agent(web, model="pcc", fallback="system")

        with patch(
            "core.agent.call_llm",
            side_effect=[
                _completion('```json\n{"income":{"details":"fact"}}\n```'),
                _completion("Краткий вывод: уровень жизни высокий, но данные неполны."),
            ],
        ) as llm:
            reply = agent.run_turn("подробно исследуй уровень жизни")

        self.assertTrue(reply.startswith("Краткий вывод"))
        self.assertEqual(len(llm.call_args_list), 2)

    def test_invalid_answers_from_both_models_use_structured_renderer(self):
        web = FakeWebSearchTool(structured=FakeResearchResult())
        agent = _agent(web, model="pcc", fallback="system")

        with patch(
            "core.agent.call_llm",
            side_effect=[
                _completion('{"answer":"x"}'),
                _completion("English only"),
            ],
        ):
            reply = agent.run_turn("подробно исследуй уровень жизни")

        self.assertEqual(reply, "По результатам поиска: подтверждённый факт [1].")

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

    def test_local_memory_reference_does_not_force_web_search(self):
        for phrase in (
            "поищи в памяти",
            "поищи в заметках",
            "проверь в notes",
            "загляни в мою память",
            "поищи заметки про отпуск",
        ):
            self.assertIsNone(_forced_web_search_query(phrase), phrase)

    def test_web_search_phrases_still_force_search(self):
        for phrase in (
            "поищи в сети",
            "поищи последние новости",
            "загугли это",
        ):
            self.assertIsNotNone(_forced_web_search_query(phrase), phrase)

    def test_forced_search_runs_before_model_and_cannot_repeat_in_same_turn(self):
        web = FakeWebSearchTool()
        agent = _agent(web)
        completed_events = []
        agent.events.subscribe(lambda event: completed_events.append(event))

        with patch("core.agent.call_llm", return_value=_completion("GPT-5.4")) as llm:
            reply = agent.run_turn("какая последняя версия gpt?")

        self.assertEqual(reply, "GPT-5.4")
        self.assertEqual(web.calls, [{"query": "какая последняя версия gpt?", "depth": "auto"}])
        tool_completed = [e for e in completed_events if e.kind == "tool_completed"]
        self.assertEqual(len(tool_completed), 1)
        self.assertTrue(tool_completed[0].ok)
        self.assertEqual(agent.last_search_query, "latest GPT version")
        self.assertTrue(any(message.get("role") == "tool" for message in agent.messages))
        self.assertEqual(len(llm.call_args.args), 3)
        self.assertTrue(llm.call_args.args[2][0]["content"].startswith("Напиши ответ на вопрос"))

    def test_hallucinated_search_after_forced_search_is_blocked_and_recovered(self):
        web = FakeWebSearchTool()
        agent = _agent(web, model="pcc", fallback="system")
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
        self.assertEqual(len(web.calls), 1)
        self.assertEqual(len(llm.call_args_list), 2)
        self.assertEqual(len(llm.call_args_list[1].args), 3)
        recovery_messages = llm.call_args_list[1].args[2]
        self.assertFalse(any(message.get("role") == "tool" for message in recovery_messages))
        self.assertEqual(
            [message["role"] for message in recovery_messages],
            ["system", "user", "assistant", "user", "assistant", "user"],
        )
        self.assertIn("GPT-5.4", recovery_messages[-1]["content"])

    def test_empty_protocol_recovery_returns_tool_evidence_instead_of_silence(self):
        web = FakeWebSearchTool()
        agent = _agent(web)

        with patch("core.agent.call_llm", return_value=_completion("")):
            reply = agent.run_turn("подробно исследуй Норвегию")

        self.assertTrue(reply.strip())
        self.assertIn("GPT-5.4", reply)
        self.assertEqual(len(web.calls), 1)

    def test_repeated_tool_call_during_recovery_returns_tool_evidence(self):
        web = FakeWebSearchTool()
        agent = _agent(web, model="pcc", fallback="system")
        ghost = _tool_completion("web_search", '{"query":"repeat","depth":"deep"}')

        with patch("core.agent.call_llm", side_effect=[ghost, ghost]):
            reply = agent.run_turn("подробно исследуй Норвегию")

        self.assertIn("GPT-5.4", reply)
        self.assertEqual(len(web.calls), 1)

    def test_unoffered_tool_is_never_executed(self):
        web = FakeWebSearchTool()
        agent = _agent(web)
        self.assertFalse(agent.registry.has("not_offered"))

        with patch(
            "core.agent.call_llm",
            side_effect=[
                _tool_completion("not_offered", '{"command":"unsafe"}'),
                _completion("recovered"),
            ],
        ) as call:
            reply = agent.run_turn("hello")

        # Нарушение протокола без доказательств завершается честным отказом,
        # а не выдуманным ответом финализатора.
        self.assertIn("не дал проверенных данных", reply)
        self.assertEqual(call.call_count, 1)

    def test_model_cannot_escalate_simple_question_to_deep_or_search_twice(self):
        web = FakeWebSearchTool()
        agent = _agent(web, fallback="system")

        with patch(
            "core.agent.call_llm",
            side_effect=[
                _tool_completion("web_search", '{"query":"коммуны Норвегии","depth":"auto"}'),
                _tool_completion("web_search", '{"query":"repeat","depth":"deep"}'),
                _completion("В Норвегии 357 коммун."),
            ],
        ):
            agent.run_turn("расскажи про коммуны Норвегии")

        self.assertEqual(len(web.calls), 1)
        self.assertEqual(web.calls[0]["depth"], "auto")

    def test_explicit_user_deep_intent_is_preserved(self):
        web = FakeWebSearchTool()
        agent = _agent(web)

        with patch(
            "core.agent.call_llm",
            return_value=_completion("Готово"),
        ) as llm:
            agent.run_turn("подробно исследуй реформу коммун")

        self.assertEqual(
            web.calls, [{"query": "подробно исследуй реформу коммун", "depth": "deep"}]
        )
        self.assertEqual(len(llm.call_args.args), 3)

    def test_research_request_from_logs_forces_one_deep_search(self):
        request = (
            "Подробно исследуй уровень жизни в Норвегии: доходы, стоимость "
            "жизни, жильё, безопасность и удовлетворённость жизнью. Сравни "
            "данные из нескольких источников и укажи противоречия."
        )

        self.assertEqual(_forced_web_search_query(request), request)
        self.assertEqual(_forced_web_search_depth(request), "deep")

    def test_batched_web_search_executes_only_first_call_and_blocks_bash(self):
        web = FakeWebSearchTool()
        bash = FakeBashTool()
        agent = _agent(web, extra_tools=[bash])
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
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(role="assistant", content="", tool_calls=calls),
                    finish_reason="tool_calls",
                )
            ],
        )

        with patch(
            "core.agent.call_llm",
            side_effect=[batch, _completion("Готово")],
        ) as llm:
            reply = agent.run_turn("расскажи о Норвегии")

        self.assertEqual(reply, "Готово")
        self.assertEqual(web.calls, [{"query": "Norway income", "depth": "auto"}])
        self.assertEqual(bash.calls, [])
        self.assertEqual(len(llm.call_args_list[1].args), 3)

    def test_invalid_first_batched_search_still_disables_tools(self):
        web = FakeWebSearchTool()
        agent = _agent(web)
        malformed = _tool_completion("web_search", '{"query":')

        with patch(
            "core.agent.call_llm",
            side_effect=[malformed, _completion("recovered")],
        ) as llm:
            reply = agent.run_turn("расскажи о Норвегии")

        self.assertEqual(reply, "recovered")
        self.assertEqual(web.calls, [])
        self.assertEqual(llm.call_args_list[1].args[3], [])

    def test_invalid_arguments_are_reported_to_model_without_execution(self):
        web = FakeWebSearchTool()
        agent = _agent(web)
        bad_args = _tool_completion("web_search", '{"query":123}')

        with patch(
            "core.agent.call_llm",
            side_effect=[bad_args, _completion("исправился")],
        ):
            reply = agent.run_turn("расскажи про версии Python")

        self.assertEqual(reply, "исправился")
        self.assertEqual(web.calls, [])
        tool_message = next(
            m for m in agent.messages if isinstance(m, dict) and m.get("role") == "tool"
        )
        self.assertIn("невалидные аргументы", tool_message["content"])


class FailingWebSearchTool(FakeWebSearchTool):
    def execute(self, args, ctx) -> ToolResult:
        self.calls.append({"query": args.query, "depth": args.depth})
        return ToolResult.failure("схема отклонена провайдером", code="tool_failed")


class FinalizerBarrierTests(TestCase):
    def test_failed_search_yields_honest_refusal_without_llm_finalizer(self):
        web = FailingWebSearchTool()
        agent = _agent(web)

        with patch("core.agent.call_llm") as call:
            reply = agent.run_turn("поищи последние новости про Python")

        call.assert_not_called()
        self.assertIn("не дал проверенных данных", reply)
        self.assertIn("по памяти не буду", reply)
        self.assertEqual(len(web.calls), 1)


class ParseFailureRetryTests(TestCase):
    def test_output_parse_failure_is_retried_once(self):
        web = FakeWebSearchTool()
        agent = _agent(web)
        parse_error = BadRequestError(
            "Parsing failed. The model generated output that could not be parsed. "
            "code: output_parse_failed",
            response=httpx.Response(400, request=httpx.Request("POST", "http://t")),
            body=None,
        )

        with patch(
            "core.agent.call_llm", side_effect=[parse_error, _completion("жив")]
        ) as call:
            reply = agent.run_turn("привет")

        self.assertEqual(reply, "жив")
        self.assertEqual(call.call_count, 2)


if __name__ == "__main__":
    import unittest

    unittest.main()
