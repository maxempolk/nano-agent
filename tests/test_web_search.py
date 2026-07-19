from types import SimpleNamespace
import time
from unittest import TestCase
from unittest.mock import MagicMock, patch

from core.tools.web_search import (
    DeepFact,
    DeepSynthesis,
    ExpectedValue,
    MAX_FORMATTED_RESULT_CHARS,
    LLM_INPUT_TOKEN_BUDGET,
    NormalFact,
    NormalPageEvidence,
    PAGE_CONTEXT_CHARS,
    SearchBudget,
    SearchBudgetExceeded,
    SearchMode,
    WebSearchTool,
    _estimate_input_tokens,
    _flat_json_schema,
    _json_object,
)


def _completion(text: str):
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


class WebSearchStructuredTests(TestCase):
    def setUp(self):
        self.tool = WebSearchTool(None, "system")

    def test_extracts_json_from_markdown_fence(self):
        value = _json_object('```json\n{"facts": []}\n```')

        self.assertEqual(value, {"facts": []})

    def test_normal_schema_inlines_pydantic_definitions(self):
        schema = _flat_json_schema(NormalPageEvidence)

        encoded = str(schema)
        self.assertNotIn("$defs", encoded)
        self.assertNotIn("$ref", encoded)
        self.assertIn("claim", encoded)

    def test_retries_invalid_structured_output(self):
        valid = '{"facts":[],"insufficient_information":true}'
        with patch(
            "core.tools.web_search.call_llm",
            side_effect=[_completion("not json"), _completion(valid)],
        ) as mocked:
            result = self.tool._structured("extract", NormalPageEvidence)

        self.assertEqual(result.facts, [])
        self.assertEqual(mocked.call_count, 2)
        retry_prompt = mocked.call_args_list[1].args[2][0]["content"]
        self.assertIn("previous response was invalid", retry_prompt)

    def test_empty_structured_response_is_not_retried(self):
        with patch(
            "core.tools.web_search.call_llm",
            return_value=_completion(""),
        ) as mocked:
            with self.assertRaises(ValueError):
                self.tool._structured("extract", NormalPageEvidence)

        mocked.assert_called_once()

    def test_structured_prompt_is_trimmed_to_safe_local_input_budget(self):
        valid = '{"facts":[],"insufficient_information":true}'
        with patch(
            "core.tools.web_search.call_llm",
            return_value=_completion(valid),
        ) as mocked:
            self.tool._structured(
                "start\n" + "x" * 30_000 + "\nend",
                NormalPageEvidence,
            )

        messages = mocked.call_args.args[2]
        self.assertLessEqual(
            _estimate_input_tokens(messages),
            LLM_INPUT_TOKEN_BUDGET,
        )
        content = messages[0]["content"]
        self.assertIn("start", content)
        self.assertIn("end", content)

    def test_formatted_result_stays_inside_agent_tool_limit(self):
        results = [{
            "title": "Source " + "t" * 300,
            "href": "https://example.com/" + "u" * 400,
            "body": "2026",
        }]
        synthesis = DeepSynthesis(facts=[
            DeepFact(claim="fact " + "x" * 500, source_ids=[1])
            for _ in range(8)
        ])

        result = self.tool._format_deep_results(results, synthesis)

        self.assertLessEqual(len(result), MAX_FORMATTED_RESULT_CHARS)
        self.assertIn("[1]", result)

    def test_execute_processes_each_selected_page_separately(self):
        results = [
            {"title": f"Source {i}", "href": f"https://example.com/{i}", "body": "body"}
            for i in range(3)
        ]
        page = NormalPageEvidence(
            facts=[NormalFact(claim="fact", evidence="evidence")],
            insufficient_information=False,
        )

        with patch.object(self.tool, "_search", return_value=results), \
             patch.object(self.tool, "_scrape", return_value="page text"), \
             patch.object(self.tool, "_extract_normal_page", return_value=page) as extract:
            result = self.tool.execute("question", depth="normal")

        self.assertEqual(extract.call_count, 2)
        self.assertIn("fact", result)

    def test_auto_mode_uses_quick_for_simple_question(self):
        self.assertEqual(
            self.tool._select_mode("Какая последняя версия GPT?"),
            SearchMode.QUICK,
        )
        self.assertEqual(
            self.tool._select_mode("Подробно исследуй и сравни несколько моделей"),
            SearchMode.DEEP,
        )
        self.assertEqual(
            self.tool._select_mode("Привет", depth="normal"),
            SearchMode.NORMAL,
        )

    def test_forced_quick_overrides_explicit_deep_from_model(self):
        tool = WebSearchTool(None, "system", force_depth="quick")

        mode = tool._select_mode("Подробно исследуй тему", depth="deep")

        self.assertEqual(mode, SearchMode.QUICK)

    def test_quick_path_uses_snippets_without_llm_or_scraping(self):
        results = [
            {
                "title": "GPT models",
                "href": "https://example.com/gpt",
                "body": "Latest GPT-5.6 model information",
            }
        ]
        self.tool.logger = MagicMock()

        with patch.object(self.tool, "_search", return_value=results), \
             patch.object(self.tool, "_scrape") as scrape, \
             patch("core.tools.web_search.call_llm") as llm:
            result = self.tool.execute("Какая последняя версия GPT?")

        llm.assert_not_called()
        scrape.assert_not_called()
        self.assertIn("snippets only", result)
        self.assertEqual(self.tool.last_stats["mode"], "quick")
        self.assertEqual(self.tool.last_stats["llm_calls"], 0)
        logged = "\n".join(call.args[0] for call in self.tool.logger.info.call_args_list)
        self.assertIn("mode=quick", logged)
        self.assertIn("llm_calls=0/0", logged)

    def test_low_quality_numeric_quick_escalates_internally_to_normal(self):
        results = [{
            "title": "Administrative divisions - Statistics Norway",
            "href": "https://www.ssb.no/en/regions",
            "body": "Updated in 2026. Information about Norway municipalities.",
        }]
        evidence = NormalPageEvidence(
            facts=[NormalFact(claim="Norway has 357 municipalities", evidence="357")],
            insufficient_information=False,
        )

        with patch.object(self.tool, "_search", return_value=results), \
             patch.object(self.tool, "_scrape", return_value="Norway has 357 municipalities"), \
             patch.object(self.tool, "_extract_normal_page", return_value=evidence) as extract:
            result = self.tool.execute("сколько коммун в Норвегии?", depth="auto")

        extract.assert_called_once()
        self.assertIn("357", result)
        self.assertEqual(self.tool.last_stats["initial_mode"], "quick")
        self.assertEqual(self.tool.last_stats["mode"], "normal")
        self.assertTrue(self.tool.last_stats["escalated"])
        self.assertIn(
            "expected_value_missing",
            self.tool.last_stats["quick_quality_reasons"],
        )

    def test_date_number_is_not_mistaken_for_requested_count(self):
        intent = self.tool._analyze_intent("сколько коммун в Норвегии?")
        text = (
            "Jan 20, 2026. Statistics about administrative divisions. "
            "A municipality is an administrative level in Norway."
        )

        self.assertFalse(self.tool._contains_expected_value(intent, text))

    def test_number_near_subject_satisfies_requested_count(self):
        intent = self.tool._analyze_intent("сколько коммун в Норвегии?")

        self.assertTrue(self.tool._contains_expected_value(
            intent,
            "Norway currently has 357 municipalities.",
        ))

    def test_related_category_count_is_not_mistaken_for_subject_count(self):
        intent = self.tool._analyze_intent("сколько коммун в Норвегии?")

        self.assertFalse(self.tool._contains_expected_value(
            intent,
            "Statistics Norway classified municipalities into 17 categories.",
        ))

    def test_explicit_quick_never_escalates(self):
        results = [{
            "title": "Administrative divisions - Statistics Norway",
            "href": "https://www.ssb.no/en/regions",
            "body": "Updated in 2026. Information about Norway municipalities.",
        }]

        with patch.object(self.tool, "_search", return_value=results), \
             patch.object(self.tool, "_scrape") as scrape:
            result = self.tool.execute("сколько коммун в Норвегии?", depth="quick")

        scrape.assert_not_called()
        self.assertIn("snippets only", result)
        self.assertEqual(self.tool.last_stats["mode"], "quick")
        self.assertFalse(self.tool.last_stats["escalated"])

    def test_quick_path_prioritizes_known_official_domain(self):
        results = [
            {"title": "GPT news", "href": "https://news.example/gpt", "body": "GPT"},
            {"title": "Models", "href": "https://openai.com/models", "body": "Official"},
        ]

        ranked = self.tool._rank_quick_results("latest GPT model", results)

        self.assertEqual(ranked[0]["href"], "https://openai.com/models")

    def test_latest_query_prioritizes_newer_official_gpt_version(self):
        results = [
            {"title": "GPT-5", "href": "https://openai.com/gpt-5", "body": "GPT-5"},
            {
                "title": "GPT-5.3 and GPT-5.4",
                "href": "https://help.openai.com/models",
                "body": "GPT-5.4 release information",
            },
        ]

        ranked = self.tool._rank_quick_results("latest GPT version", results)

        self.assertEqual(ranked[0]["href"], "https://help.openai.com/models")

    def test_normal_ranking_prefers_official_administrative_divisions_page(self):
        results = [
            {
                "title": "Municipal accounts - SSB",
                "href": "https://www.ssb.no/en/statistikk/kommuneregnskap",
                "body": "Municipalities are classified into 17 categories.",
            },
            {
                "title": "Administrative divisions - SSB",
                "href": "https://www.ssb.no/en/statistikk/regionale-inndelingar",
                "body": "Current administrative divisions and municipalities in Norway, 2026.",
            },
            {
                "title": "Municipal health care service - SSB",
                "href": "https://www.ssb.no/en/statistikk/health-service",
                "body": "Health services provided by municipalities.",
            },
        ]
        intent = self.tool._analyze_intent("сколько коммун в Норвегии?")

        ranked = self.tool._rank_results(intent, results)

        self.assertIn("regionale-inndelingar", ranked[0]["href"])

    def test_normal_fact_filter_rejects_related_count_and_stale_fact(self):
        intent = self.tool._analyze_intent("сколько коммун в Норвегии?")

        self.assertFalse(self.tool._fact_matches_intent(intent, NormalFact(
            claim="Municipalities are classified into 17 categories",
            evidence="Statistics Norway classified municipalities into 17 categories.",
            published_at="2026-01-01",
        )))
        self.assertFalse(self.tool._fact_matches_intent(intent, NormalFact(
            claim="Norway has 356 municipalities",
            evidence="Norway has 356 municipalities.",
            published_at="2020-01-01",
        )))
        self.assertTrue(self.tool._fact_matches_intent(intent, NormalFact(
            claim="Norway has 357 municipalities",
            evidence="Norway currently has 357 municipalities.",
            published_at="2026-01-01",
        )))

    def test_source_year_uses_most_recent_year_in_result_metadata(self):
        result = {
            "title": "Administrative divisions 2024",
            "body": "Updated January 2026",
        }

        self.assertEqual(self.tool._source_year(result), 2026)

    def test_quick_query_normalizes_live_btc_currency_without_llm(self):
        self.assertEqual(
            self.tool._normalize_quick_query("скажи какой сейчас курс btc?"),
            "BTC USD live price",
        )
        self.assertEqual(
            self.tool._normalize_quick_query("курс биткоина к рублю"),
            "BTC RUB live price",
        )

    def test_intent_normalizes_current_norway_municipality_count(self):
        intent = self.tool._analyze_intent("сколько коммун в Норвегии?")

        self.assertEqual(intent.expected_value, ExpectedValue.NUMBER)
        self.assertTrue(intent.requires_freshness)
        self.assertEqual(intent.official_domain, "ssb.no")
        self.assertEqual(
            intent.search_query(),
            "current number of municipalities Norway official statistics site:ssb.no",
        )

    def test_intent_extracts_weather_location_without_llm(self):
        intent = self.tool._analyze_intent("какая температура в Stonglandseidet?")

        self.assertEqual(intent.expected_value, ExpectedValue.WEATHER)
        self.assertTrue(intent.requires_freshness)
        self.assertEqual(intent.search_query(), "current weather Stonglandseidet")

    def test_intent_normalizes_norway_living_standard_research(self):
        intent = self.tool._analyze_intent("подробно исследуй уровень жизни в Норвегии")

        self.assertTrue(intent.requires_freshness)
        self.assertEqual(
            intent.search_query(),
            "Norway standard of living quality of life latest statistics",
        )

    def test_multiple_known_vendors_do_not_force_one_official_domain(self):
        intent = self.tool._analyze_intent("сравни Apple и Google")

        self.assertIsNone(intent.official_domain)
        self.assertNotIn("site:", intent.search_query())

    def test_normal_latest_gpt_query_uses_official_domain(self):
        self.assertEqual(
            self.tool._normal_search_query("проверь последнюю версию GPT по источникам"),
            "latest GPT model OpenAI site:openai.com",
        )

    def test_normal_comparison_is_not_restricted_to_one_vendor(self):
        self.assertEqual(
            self.tool._normal_search_query("сравни Apple и Google"),
            "сравни Apple и Google",
        )

    def test_relevant_passages_can_select_fact_from_end_of_long_page(self):
        filler = "Unrelated archive paragraph about cooking and gardening. " * 80
        target = "July 2026 release: GPT-5.4 is the current production model."
        content = "\n\n".join([filler for _ in range(8)] + [target])
        result = {
            "title": "GPT model releases",
            "body": "Latest GPT-5.4 release information",
            "href": "https://openai.com/models",
        }

        selected = self.tool._select_relevant_passages(
            "latest GPT model",
            content,
            result,
        )

        self.assertIn(target, selected)
        self.assertLessEqual(len(selected), PAGE_CONTEXT_CHARS)

    def test_normal_execute_searches_with_deterministic_normalized_query(self):
        results = [
            {
                "title": "GPT-5.6",
                "href": "https://openai.com/gpt-5-6",
                "body": "Latest release",
            }
        ]
        evidence = NormalPageEvidence(
            facts=[NormalFact(claim="GPT-5.6", evidence="release")],
            insufficient_information=False,
        )

        with patch.object(self.tool, "_search", return_value=results) as search, \
             patch.object(self.tool, "_scrape", return_value="short page"), \
             patch.object(self.tool, "_structured", return_value=evidence):
            self.tool.execute("проверь последнюю версию GPT", depth="normal")

        search.assert_called_once_with("latest GPT model OpenAI site:openai.com")

    def test_deep_uses_four_page_extractions_and_one_synthesis(self):
        results = [
            {
                "title": f"Research {index}",
                "href": f"https://example.com/research-{index}",
                "body": "2026 research evidence",
            }
            for index in range(5)
        ]
        page = NormalPageEvidence(
            facts=[NormalFact(claim="fact", evidence="evidence")],
            insufficient_information=False,
        )
        synthesis = DeepSynthesis(
            facts=[DeepFact(claim="verified", source_ids=[1, 2])],
        )

        with patch.object(self.tool, "_search", return_value=results) as search, \
             patch.object(self.tool, "_scrape", return_value="page"), \
             patch.object(self.tool, "_extract_normal_page", return_value=page) as extract, \
             patch.object(self.tool, "_synthesize_deep", return_value=synthesis) as synthesize:
            result = self.tool.execute("подробно исследуй тему", depth="deep")

        search.assert_called_once_with("подробно исследуй тему")
        self.assertEqual(extract.call_count, 4)
        synthesize.assert_called_once()
        self.assertIn("verified", result)
        self.assertEqual(self.tool.last_stats["max_llm_calls"], 5)

    def test_deep_parallel_extraction_preserves_ranked_source_order(self):
        results = [
            {
                "title": f"Research {index}",
                "href": f"https://example.com/research-{index}",
                "body": "2026 research evidence",
            }
            for index in range(4)
        ]
        captured_pages = []

        def extract(_question, result, _page):
            index = int(result["title"].rsplit(" ", 1)[1])
            time.sleep((3 - index) * 0.005)
            return NormalPageEvidence(
                facts=[NormalFact(claim=str(index), evidence="evidence")],
                insufficient_information=False,
            )

        def synthesize(_question, pages):
            captured_pages.extend(pages)
            return DeepSynthesis()

        with patch.object(self.tool, "_search", return_value=results), \
             patch.object(self.tool, "_scrape", return_value="page"), \
             patch.object(self.tool, "_extract_normal_page", side_effect=extract), \
             patch.object(self.tool, "_synthesize_deep", side_effect=synthesize):
            self.tool.execute("подробно исследуй тему", depth="deep")

        self.assertEqual(
            [page.facts[0].claim for page in captured_pages],
            ["0", "1", "2", "3"],
        )

    def test_normal_path_makes_one_structured_extraction_per_page(self):
        results = [
            {"title": f"Source {i}", "href": f"https://example.com/{i}", "body": "body"}
            for i in range(2)
        ]
        evidence = NormalPageEvidence(
            facts=[NormalFact(claim="claim", evidence="quote")],
            insufficient_information=False,
        )

        with patch.object(self.tool, "_search", return_value=results), \
             patch.object(self.tool, "_scrape", return_value="short page"), \
             patch.object(self.tool, "_structured", return_value=evidence) as structured:
            self.tool.execute("verify this", depth="normal")

        self.assertEqual(structured.call_count, 2)
        self.assertLessEqual(self.tool.last_stats["llm_calls"], 2)

    def test_normal_recovers_facts_from_afm_defs_shape(self):
        raw = (
            '{"facts":[{"ref":"1","$defs":{"NormalFact":'
            '{"claim":"GPT-5.6 is current","evidence":"GPT-5.6 release"}}}],'
            '"insufficient_information":false}'
        )

        recovered = self.tool._recover_normal_evidence(raw)

        self.assertFalse(recovered.insufficient_information)
        self.assertEqual(recovered.facts[0].claim, "GPT-5.6 is current")

    def test_normal_recovers_complete_pairs_from_truncated_json(self):
        raw = (
            '{"facts":[{"claim":"first","evidence":"quote one"},'
            '{"claim":"second","evidence":"quote two"'
        )

        recovered = self.tool._recover_normal_evidence(raw)

        self.assertEqual(
            [fact.claim for fact in recovered.facts],
            ["first", "second"],
        )

    def test_search_budget_blocks_extra_llm_calls(self):
        budget = SearchBudget(SearchMode.NORMAL, max_llm_calls=1, timeout_seconds=60)

        self.assertEqual(budget.consume_llm(), 1)
        with self.assertRaises(SearchBudgetExceeded):
            budget.consume_llm()

    def test_search_budget_detects_deadline(self):
        budget = SearchBudget(
            SearchMode.QUICK,
            max_llm_calls=0,
            timeout_seconds=1,
            started_at=time.monotonic() - 2,
        )

        with self.assertRaises(SearchBudgetExceeded):
            budget.check_deadline()
