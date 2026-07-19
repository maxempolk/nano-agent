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
    SearchIntent,
    SearchMode,
    ResearchPlan,
    WebSearchTool,
    _estimate_input_tokens,
    _flat_json_schema,
    _json_object,
)


def _completion(text: str):
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _planned(query: str, *, search_queries=None, aspects=None,
             expected=ExpectedValue.FACT, fresh=False, domain=None):
    queries = search_queries or [query]
    plan = ResearchPlan(
        search_queries=queries,
        subject="",
        aspects=aspects or [query],
        expected_value=expected,
        requires_freshness=fresh,
        official_domain=domain or "",
        official_domains=[domain] if domain else [],
    )
    intent = SearchIntent(
        original_query=query,
        normalized_query=" ".join(queries),
        expected_value=expected,
        requires_freshness=fresh,
        official_requested=False,
        official_domain=domain,
        preferred_domains=(domain,) if domain else (),
    )
    return plan, intent


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
             patch.object(self.tool, "_plan_research", return_value=_planned("question")), \
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
             patch.object(
                 self.tool,
                 "_plan_research",
                 return_value=_planned(
                     "сколько коммун в Норвегии?",
                     search_queries=["current Norway municipality count"],
                     expected=ExpectedValue.NUMBER,
                     fresh=True,
                     domain="ssb.no",
                 ),
             ), \
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
            "Норвегия насчитывает 357 коммун.",
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
        _, intent = _planned(
            "сколько коммун в Норвегии?",
            search_queries=["current number of Norway municipalities administrative divisions"],
            expected=ExpectedValue.NUMBER,
            fresh=True,
            domain="ssb.no",
        )

        ranked = self.tool._rank_results(intent, results)

        self.assertIn("regionale-inndelingar", ranked[0]["href"])

    def test_normal_fact_filter_rejects_related_count_and_stale_fact(self):
        _, intent = _planned(
            "сколько коммун в Норвегии?",
            search_queries=["current Norway municipality count"],
            expected=ExpectedValue.NUMBER,
            fresh=True,
            domain="ssb.no",
        )

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

    def test_quick_query_stays_deterministic_without_subject_hardcodes(self):
        self.assertEqual(
            self.tool._normalize_quick_query("скажи какой сейчас курс btc?"),
            "скажи какой сейчас курс btc?",
        )
        self.assertEqual(
            self.tool._normalize_quick_query("курс биткоина к рублю"),
            "курс биткоина к рублю",
        )

    def test_basic_intent_classifies_without_subject_specific_rewrite(self):
        intent = self.tool._analyze_intent("сколько коммун в Норвегии?")

        self.assertEqual(intent.expected_value, ExpectedValue.NUMBER)
        self.assertFalse(intent.requires_freshness)
        self.assertIsNone(intent.official_domain)
        self.assertEqual(intent.search_query(), "сколько коммун в Норвегии?")

    def test_intent_extracts_weather_location_without_llm(self):
        intent = self.tool._analyze_intent("какая температура в Stonglandseidet?")

        self.assertEqual(intent.expected_value, ExpectedValue.WEATHER)
        self.assertTrue(intent.requires_freshness)
        self.assertEqual(intent.search_query(), "какая температура в Stonglandseidet?")

    def test_structured_planner_preserves_multiple_research_aspects(self):
        raw = (
            '{"queries":[{"query":"Norway income and cost of living statistics",'
            '"aspect":"income and cost of living","official_domain":"ssb.no"},'
            '{"query":"Norway housing safety life satisfaction statistics",'
            '"aspect":"housing safety and life satisfaction","official_domain":""}],'
            '"aspects":["income","cost of living","housing","safety",'
            '"life satisfaction"],"expected_value":"fact",'
            '"subject":"Norway",'
            '"requires_freshness":true,"official_domain":"",'
            '"official_domains":["ssb.no"]}'
        )
        self.tool._budget = SearchBudget.for_mode(SearchMode.DEEP)

        with patch("core.tools.web_search.call_llm", return_value=_completion(raw)) as llm:
            plan, intent = self.tool._plan_research(
                "подробно исследуй уровень жизни в Норвегии: доходы, стоимость "
                "жизни, жильё, безопасность и удовлетворённость",
                SearchMode.DEEP,
            )

        self.assertEqual(len(plan.aspects), 5)
        self.assertEqual(len(plan.search_queries), 2)
        self.assertEqual(intent.expected_value, ExpectedValue.FACT)
        self.assertEqual(intent.subject, "Norway")
        self.assertEqual(intent.official_domain, "ssb.no")
        self.assertTrue(plan.search_queries[0].endswith("site:ssb.no"))
        self.assertEqual(plan.official_domains, ["ssb.no"])
        self.assertEqual(llm.call_args.args[1], "system")

    def test_planner_cannot_label_commercial_domain_as_official(self):
        intent = SearchIntent(
            original_query="Norway cost of living",
            normalized_query="Norway cost of living",
            expected_value=ExpectedValue.FACT,
            requires_freshness=True,
            official_requested=False,
            official_domain="numbeo.com",
            preferred_domains=("numbeo.com",),
        )
        results = [{
            "title": "Cost of Living",
            "href": "https://www.numbeo.com/cost-of-living/country_result.jsp?country=Norway",
            "body": "Current prices in Norway",
        }]

        rendered = self.tool._format_normal_results(
            results,
            [NormalPageEvidence(facts=[], insufficient_information=True)],
        )

        self.assertIn("Official: no", rendered)

    def test_hybrid_uses_pcc_for_plan_and_local_for_page_extraction(self):
        tool = WebSearchTool(
            None,
            "system",
            model_mini="system",
            planner_model="pcc",
        )
        plan_raw = (
            '{"queries":[{"query":"Norway income statistics","aspect":"income",'
            '"official_domain":"ssb.no"}],'
            '"aspects":["income"],"expected_value":"fact",'
            '"subject":"Norway",'
            '"requires_freshness":true,"official_domain":"",'
            '"official_domains":["ssb.no"]}'
        )
        evidence_raw = '{"facts":[],"insufficient_information":true}'
        tool._budget = SearchBudget(SearchMode.DEEP, 2, 60)

        with patch(
            "core.tools.web_search.call_llm",
            side_effect=[_completion(plan_raw), _completion(evidence_raw)],
        ) as llm:
            tool._plan_research("исследуй доходы Норвегии", SearchMode.DEEP)
            tool._structured("extract page", NormalPageEvidence, max_attempts=1)

        self.assertEqual(llm.call_args_list[0].args[1], "pcc")
        self.assertEqual(llm.call_args_list[1].args[1], "system")

    def test_empty_deep_synthesis_falls_back_to_extracted_facts(self):
        pages = [NormalPageEvidence(
            facts=[NormalFact(claim="Supported fact", evidence="source text")],
            insufficient_information=False,
        )]

        with patch.object(self.tool, "_structured", return_value=DeepSynthesis()):
            synthesis = self.tool._synthesize_deep("question", pages)

        self.assertEqual(synthesis.facts[0].claim, "Supported fact")
        self.assertEqual(synthesis.facts[0].source_ids, [1])

    def test_broad_intent_rejects_tangential_and_stale_facts(self):
        intent = SearchIntent(
            original_query="research Norway living standards",
            normalized_query="income housing safety life satisfaction",
            expected_value=ExpectedValue.FACT,
            requires_freshness=True,
            official_requested=False,
            official_domain=None,
        )

        self.assertFalse(self.tool._fact_matches_intent(intent, NormalFact(
            claim="Five hospital projects receive new loans",
            evidence="The budget authorises hospital construction in 2026.",
        )))
        self.assertFalse(self.tool._fact_matches_intent(intent, NormalFact(
            claim="Norway ranked among the happiest countries in 2019",
            evidence="The 2019 report measured life satisfaction.",
        )))
        self.assertTrue(self.tool._fact_matches_intent(intent, NormalFact(
            claim="Household income increased in 2026",
            evidence="Income statistics were updated in 2026.",
        )))
        self.assertTrue(self.tool._fact_matches_intent(intent, NormalFact(
            claim="Housing costs averaged 132,263 NOK in 2023",
            evidence="Housing statistics reported the 2023 average.",
        )))

    def test_external_fact_must_match_planned_subject(self):
        intent = SearchIntent(
            original_query="Norway quality of life",
            normalized_query="quality of life satisfaction",
            expected_value=ExpectedValue.FACT,
            requires_freshness=True,
            official_requested=False,
            official_domain="ssb.no",
            preferred_domains=("ssb.no",),
            subject="Norway",
        )

        self.assertFalse(self.tool._fact_matches_intent(
            intent,
            NormalFact(
                claim="Sweden ranks first for quality of life in 2026",
                evidence="Sweden leads the quality of life ranking.",
            ),
            "https://example.org/ranking",
        ))
        self.assertTrue(self.tool._fact_matches_intent(
            intent,
            NormalFact(
                claim="Life satisfaction averaged 7.0 in 2025",
                evidence="The official survey reports life satisfaction of 7.0.",
            ),
            "https://www.ssb.no/survey",
        ))

    def test_research_ranking_demotes_clearly_stale_official_page(self):
        intent = SearchIntent(
            original_query="income research",
            normalized_query="current household income statistics",
            expected_value=ExpectedValue.FACT,
            requires_freshness=True,
            official_requested=False,
            official_domain="ssb.no",
            preferred_domains=("ssb.no",),
        )
        results = [
            {
                "title": "Household income 2010",
                "href": "https://www.ssb.no/archive/income",
                "body": "Income statistics from 2010",
            },
            {
                "title": "Current household income statistics 2025",
                "href": "https://example.org/current-income",
                "body": "Updated income evidence for 2025",
            },
        ]

        ranked = self.tool._rank_results(intent, results)

        self.assertEqual(ranked[0]["href"], "https://example.org/current-income")

    def test_deep_source_selection_covers_distinct_aspects(self):
        intent = SearchIntent(
            original_query="research",
            normalized_query="income housing safety satisfaction",
            expected_value=ExpectedValue.FACT,
            requires_freshness=True,
            official_requested=False,
            official_domain=None,
        )
        results = [
            {"title": "Income statistics", "href": "https://a.test/income", "body": "income"},
            {"title": "Housing statistics", "href": "https://b.test/housing", "body": "housing"},
            {"title": "Safety statistics", "href": "https://c.test/safety", "body": "safety"},
            {"title": "Life satisfaction", "href": "https://d.test/life", "body": "satisfaction"},
            {"title": "Income opinion", "href": "https://e.test/income", "body": "income"},
        ]

        selected = self.tool._select_deep_sources(
            intent,
            results,
            ["income", "housing", "safety", "satisfaction"],
        )

        selected_text = " ".join(item["title"].lower() for item in selected)
        for aspect in ("income", "housing", "safety", "satisfaction"):
            self.assertIn(aspect, selected_text)

    def test_deep_replaces_unreadable_source_before_extraction(self):
        intent = SearchIntent(
            original_query="Norway income",
            normalized_query="Norway income",
            expected_value=ExpectedValue.FACT,
            requires_freshness=True,
            official_requested=False,
            official_domain=None,
        )
        broken = {
            "title": "Broken income page",
            "href": "https://broken.test/income",
            "body": "income statistics",
            "_plan_query": 0,
        }
        replacement = {
            "title": "Readable income page",
            "href": "https://readable.test/income",
            "body": "income statistics 2026",
            "_plan_query": 0,
        }
        selected = [broken]
        scraped = {broken["href"]: "Не удалось извлечь текст."}
        self.tool.last_intent = intent

        with patch.object(self.tool, "_scrape", return_value="x" * 300):
            self.tool._replace_unreadable_sources(
                selected, [broken, replacement], scraped
            )

        self.assertEqual(selected[0], replacement)
        self.assertEqual(scraped[replacement["href"]], "x" * 300)

    def test_deep_reports_requested_aspects_missing_from_evidence(self):
        pages = [NormalPageEvidence(
            facts=[NormalFact(
                claim="Household income increased in 2026",
                evidence="Income statistics rose.",
            )],
            insufficient_information=False,
        )]

        gaps = self.tool._coverage_gaps(["income", "public safety"], pages)
        rendered = self.tool._format_deep_results([], DeepSynthesis(), gaps)

        self.assertEqual(gaps, ["public safety"])
        self.assertIn("Coverage gaps", rendered)
        self.assertIn("public safety", rendered)

    def test_deep_selection_takes_one_source_from_each_planned_query(self):
        intent = SearchIntent(
            original_query="research",
            normalized_query="income housing safety satisfaction",
            expected_value=ExpectedValue.FACT,
            requires_freshness=True,
            official_requested=False,
            official_domain=None,
        )
        results = [
            {"title": "Income", "href": "https://a.test", "body": "income", "_plan_query": 0},
            {"title": "Housing", "href": "https://b.test", "body": "housing", "_plan_query": 1},
            {"title": "Safety", "href": "https://c.test", "body": "safety", "_plan_query": 2},
            {"title": "Satisfaction one", "href": "https://d.test", "body": "satisfaction", "_plan_query": 3},
            {"title": "Satisfaction two", "href": "https://e.test", "body": "satisfaction", "_plan_query": 3},
        ]

        selected = self.tool._select_deep_sources(
            intent,
            results,
            ["income", "housing", "safety", "satisfaction"],
        )

        self.assertEqual(
            {item["_plan_query"] for item in selected},
            {0, 1, 2, 3},
        )

    def test_multiple_known_vendors_do_not_force_one_official_domain(self):
        intent = self.tool._analyze_intent("сравни Apple и Google")

        self.assertIsNone(intent.official_domain)
        self.assertNotIn("site:", intent.search_query())

    def test_normal_latest_gpt_query_uses_official_domain(self):
        self.assertEqual(
            self.tool._normal_search_query("проверь последнюю версию GPT по источникам"),
            "проверь последнюю версию GPT по источникам site:openai.com",
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

    def test_normal_execute_searches_with_planned_query(self):
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
             patch.object(
                 self.tool,
                 "_plan_research",
                 return_value=_planned(
                     "проверь последнюю версию GPT",
                     search_queries=["latest GPT model OpenAI"],
                     expected=ExpectedValue.VERSION,
                     fresh=True,
                     domain="openai.com",
                 ),
             ), \
             patch.object(self.tool, "_scrape", return_value="short page"), \
             patch.object(self.tool, "_extract_normal_page", return_value=evidence):
            self.tool.execute("проверь последнюю версию GPT", depth="normal")

        search.assert_called_once_with("latest GPT model OpenAI")

    def test_deep_uses_five_page_extractions_and_one_synthesis(self):
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
             patch.object(
                 self.tool,
                 "_plan_research",
                 return_value=_planned("подробно исследуй тему"),
             ), \
             patch.object(self.tool, "_scrape", return_value="page"), \
             patch.object(self.tool, "_extract_normal_page", return_value=page) as extract, \
             patch.object(self.tool, "_synthesize_deep", return_value=synthesis) as synthesize:
            result = self.tool.execute("подробно исследуй тему", depth="deep")

        search.assert_called_once_with("подробно исследуй тему")
        self.assertEqual(extract.call_count, 5)
        synthesize.assert_called_once()
        self.assertIn("verified", result)
        self.assertEqual(self.tool.last_stats["max_llm_calls"], 7)

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
             patch.object(
                 self.tool,
                 "_plan_research",
                 return_value=_planned("подробно исследуй тему"),
             ), \
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
             patch.object(
                 self.tool,
                 "_plan_research",
                 return_value=_planned("verify this"),
             ), \
             patch.object(self.tool, "_scrape", return_value="short page"), \
             patch.object(self.tool, "_structured", return_value=evidence) as structured:
            self.tool.execute("verify this", depth="normal")

        self.assertEqual(structured.call_count, 2)
        self.assertLessEqual(self.tool.last_stats["llm_calls"], 3)

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

    def test_normal_recovery_rejects_insufficient_information_as_a_fact(self):
        raw = (
            '{"facts":[{"claim":"Insufficient information",'
            '"evidence":"No relevant data was found"}]}'
        )

        recovered = self.tool._recover_normal_evidence(raw)

        self.assertEqual(recovered.facts, [])
        self.assertTrue(recovered.insufficient_information)

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
