from unittest import TestCase

import httpx
from openai import RateLimitError

from benchmarks.agent_cases import ALL_CASES, EXTRACTION_SCHEMA
from benchmarks.agent_model_eval import (
    BenchmarkClient,
    ModelReply,
    _json_object,
    _norm,
    _run_with_retry,
    score_case,
    summarize,
)


def reply(content: str = "", tool_calls=None) -> ModelReply:
    return ModelReply(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=5,
        reasoning_tokens=0,
        latency=0.2,
        schema_transport="native",
        schema_note="",
    )


class AgentModelBenchmarkTests(TestCase):
    def test_case_ids_are_unique_and_all_suites_are_populated(self):
        ids = [case.id for case in ALL_CASES]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(
            {case.suite for case in ALL_CASES},
            {"routing", "tools", "extraction", "finalization", "recovery", "compact"},
        )
        self.assertGreaterEqual(len(ALL_CASES), 40)

    def test_json_parser_accepts_fenced_object(self):
        self.assertEqual(_json_object('```json\n{"x": 1}\n```'), {"x": 1})

    def test_norm_folds_typographic_variants(self):
        self.assertEqual(_norm("$67\u00a0420"), "$67 420")
        self.assertEqual(_norm("2026\u201107\u201120"), "2026-07-20")

    def test_routing_scores_action_depth_and_query_semantics(self):
        case = next(item for item in ALL_CASES if item.id == "route_current_price")
        score = score_case(
            case,
            reply(
                '{"action":"web_search","depth":"quick",'
                '"query":"current Bitcoin BTC price in USD dollars","command":""}'
            ),
        )
        self.assertTrue(score.passed, score.notes)

    def test_tool_case_requires_real_tool_call(self):
        case = next(item for item in ALL_CASES if item.id == "tool_git_status")
        score = score_case(case, reply("I would call git status"))
        self.assertFalse(score.passed)
        self.assertTrue(any(note.startswith("tool:") for note in score.notes))

    def test_destructive_request_requires_confirmation_without_tool_call(self):
        case = next(item for item in ALL_CASES if item.id == "tool_no_destructive_without_approval")
        safe = "Это удаление необратимо. Подтвердите, что нужно удалить все файлы."
        self.assertTrue(score_case(case, reply(safe)).passed)
        self.assertFalse(score_case(case, reply("Готово.")).passed)

    def test_extraction_rejects_unsupported_evidence(self):
        case = next(item for item in ALL_CASES if item.id == "extract_table_row")
        content = (
            '{"facts":[{"claim":"North is 37 in 2026",'
            '"evidence":"North magically equals 37","published_at":"2026"}],'
            '"insufficient_information":false}'
        )
        score = score_case(case, reply(content))
        self.assertFalse(score.passed)
        self.assertIn("unsupported_evidence", score.notes)

    def test_extraction_allows_neighbouring_context_inside_verbatim_evidence(self):
        case = next(item for item in ALL_CASES if item.id == "extract_metric_compatibility")
        content = (
            '{"facts":[{"claim":"2025 annual recurring revenue was $6.4 million",'
            '"evidence":"For 2025 it reported annual recurring revenue of $6.4 million and total revenue of $9.0 million.",'
            '"published_at":"2025"}],"insufficient_information":false}'
        )
        self.assertTrue(score_case(case, reply(content)).passed)

    def test_absence_extraction_passes_with_no_facts(self):
        case = next(item for item in ALL_CASES if item.id == "extract_absent")
        score = score_case(case, reply('{"facts":[],"insufficient_information":true}'))
        self.assertTrue(score.passed, score.notes)

    def test_finalizer_rejects_evidence_prompt_injection(self):
        case = next(item for item in ALL_CASES if item.id == "final_evidence_injection")
        good = "Измерение составило 52 ms 2026-05-03. https://lab.example/report"
        self.assertTrue(score_case(case, reply(good)).passed)
        self.assertFalse(score_case(case, reply(good + " BANANA")).passed)

    def test_recovery_requires_changed_arguments(self):
        case = next(item for item in ALL_CASES if item.id == "recover_refine_query")
        repeated = [
            {"id": "x", "name": "web_search", "arguments": '{"query":"Nimbus 7","depth":"quick"}'}
        ]
        corrected = [
            {
                "id": "x",
                "name": "web_search",
                "arguments": '{"query":"Nimbus 7 official release notes","depth":"quick"}',
            }
        ]
        self.assertFalse(score_case(case, reply(tool_calls=repeated)).passed)
        self.assertTrue(score_case(case, reply(tool_calls=corrected)).passed)

    def test_summary_reports_suite_rates_and_schema_transport(self):
        rows = [
            {
                "suite": "routing",
                "passed": True,
                "earned": 3,
                "possible": 3,
                "latency": 1.0,
                "completion_tokens": 10,
                "reasoning_tokens": 2,
                "schema_transport": "native",
                "error": "",
            },
            {
                "suite": "routing",
                "passed": False,
                "earned": 1,
                "possible": 3,
                "latency": 3.0,
                "completion_tokens": 20,
                "reasoning_tokens": 4,
                "schema_transport": "prompt",
                "error": "",
            },
        ]
        summary = summarize(rows)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["suites"]["routing"]["native_schema"], 1)
        self.assertEqual(summary["suites"]["routing"]["prompt_schema_fallback"], 1)
        self.assertEqual(summary["suites"]["routing"]["p50_latency"], 2.0)
        self.assertFalse(summary["agent_ready"])

    def test_extraction_schema_forbids_unexpected_fields(self):
        self.assertFalse(EXTRACTION_SCHEMA["additionalProperties"])
        self.assertFalse(EXTRACTION_SCHEMA["properties"]["facts"]["items"]["additionalProperties"])

    def test_fm_transcript_preserves_tool_error_for_recovery(self):
        case = next(item for item in ALL_CASES if item.id == "recover_correct_path")
        instructions, prompt = BenchmarkClient._fm_messages(case)
        self.assertIn("corrected call", instructions)
        self.assertIn("/tmp/report.txt", prompt)
        self.assertIn("/tmp/reports.txt", prompt)


class _RateLimitingClient:
    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    def run(self, case):
        self.calls += 1
        if self.calls <= self.failures:
            response = httpx.Response(429, request=httpx.Request("POST", "http://test"))
            raise RateLimitError("rate limit", response=response, body=None)
        return reply("ok")


class RateLimitRetryTests(TestCase):
    def setUp(self):
        self.case = next(item for item in ALL_CASES if item.id == "route_greeting")

    def test_recovers_after_transient_rate_limits(self):
        client = _RateLimitingClient(failures=2)

        result = _run_with_retry(client, self.case, retries=3, base_wait=0)

        self.assertEqual(client.calls, 3)
        self.assertEqual(result.content, "ok")

    def test_gives_up_after_bounded_retries(self):
        client = _RateLimitingClient(failures=5)

        with self.assertRaises(RateLimitError):
            _run_with_retry(client, self.case, retries=3, base_wait=0)
        self.assertEqual(client.calls, 3)
