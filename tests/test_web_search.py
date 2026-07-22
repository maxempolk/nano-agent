from types import SimpleNamespace
import time
from unittest import TestCase
from unittest.mock import MagicMock, patch

from core.tools.web_search import (
    AspectStatus,
    AspectReview,
    CandidateExtraction,
    ConflictAssessment,
    DeepFact,
    DeepSynthesis,
    ExpectedValue,
    MAX_FORMATTED_RESULT_CHARS,
    MAX_RETRIES,
    LLM_INPUT_TOKEN_BUDGET,
    NormalFact,
    NormalPageEvidence,
    PAGE_CONTEXT_CHARS,
    PlannedQuery,
    ResearchResult,
    ResearchSource,
    SearchBudget,
    SearchBudgetExceeded,
    SearchIntent,
    SearchMode,
    ResearchPlan,
    ResearchAspect,
    WebSearchTool,
    _afm_generation_schema,
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

    def test_afm_schema_uses_generation_schema_dialect(self):
        schema = _afm_generation_schema(NormalPageEvidence)

        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["x-order"], list(schema["properties"]))
        fact = schema["properties"]["facts"]["items"]
        self.assertFalse(fact["additionalProperties"])
        self.assertEqual(fact["x-order"], list(fact["properties"]))
        self.assertNotIn("title", fact["properties"]["claim"])
        self.assertNotIn("default", fact["properties"]["published_at"])

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

    def test_structured_output_uses_native_schema_not_prompt_instructions(self):
        valid = '{"facts":[],"insufficient_information":true}'
        with patch(
            "core.tools.web_search.call_llm",
            return_value=_completion(valid),
        ) as mocked:
            self.tool._structured("extract supported facts", NormalPageEvidence)

        prompt = mocked.call_args.args[2][0]["content"]
        response_format = mocked.call_args.kwargs["response_format"]
        self.assertNotIn("JSON", prompt.upper())
        self.assertNotIn("schema", prompt.lower())
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(
            response_format["json_schema"]["name"],
            "normal_page_evidence",
        )
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertIn(
            "facts",
            response_format["json_schema"]["schema"]["properties"],
        )

    def test_structured_retry_does_not_request_json_in_prompt(self):
        valid = '{"facts":[],"insufficient_information":true}'
        with patch(
            "core.tools.web_search.call_llm",
            side_effect=[_completion("invalid"), _completion(valid)],
        ) as mocked:
            self.tool._structured("extract", NormalPageEvidence)

        retry_prompt = mocked.call_args_list[1].args[2][0]["content"]
        self.assertNotIn("JSON", retry_prompt.upper())
        self.assertNotIn("Invalid response:", retry_prompt)

    def test_empty_structured_response_exhausts_retries_then_raises(self):
        with patch(
            "core.tools.web_search.call_llm",
            return_value=_completion(""),
        ) as mocked:
            with self.assertRaises(ValueError):
                self.tool._structured("extract", NormalPageEvidence)

        self.assertEqual(mocked.call_count, MAX_RETRIES)

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

    def test_structured_research_result_has_readable_russian_fallback(self):
        result = ResearchResult(
            query="исследуй уровень жизни",
            mode="deep",
            sources=[ResearchSource(
                source_id=1,
                title="Official statistics",
                url="https://example.test/statistics",
                official=True,
            )],
            facts=[DeepFact(claim="Подтверждённый факт", source_ids=[1])],
            coverage_gaps=["безопасность"],
        )

        rendered = result.render_fallback()

        self.assertIn("Подтверждённый факт [1]", rendered)
        self.assertIn("Не удалось надёжно проверить", rendered)
        self.assertIn("https://example.test/statistics", rendered)

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
            "no_relevant_results",
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

    def test_empty_pcc_verification_does_not_promote_afm_candidates(self):
        pages = [NormalPageEvidence(
            facts=[NormalFact(claim="Supported fact", evidence="source text")],
            insufficient_information=False,
        )]

        with patch.object(self.tool, "_structured", return_value=DeepSynthesis()):
            synthesis = self.tool._synthesize_deep("question", pages)

        self.assertEqual(synthesis.facts, [])
        self.assertTrue(synthesis.insufficient_information)

    def test_afm_candidate_schema_is_intentionally_minimal(self):
        schema = _flat_json_schema(CandidateExtraction)
        encoded = str(schema)

        self.assertIn("claim", encoded)
        self.assertIn("evidence", encoded)
        self.assertIn("published_at", encoded)
        self.assertNotIn("relevance_score", encoded)
        self.assertNotIn("acceptance_criteria", encoded)

    def test_page_extraction_produces_unverified_candidates_only(self):
        candidates = CandidateExtraction(facts=[{
            "claim": "The measured value was 10",
            "evidence": "The measured value was 10",
            "published_at": "2026",
        }])
        result = {
            "title": "Measurement",
            "href": "https://example.test/value",
            "body": "",
        }
        with patch.object(
            self.tool, "_structured", return_value=candidates
        ) as structured:
            page = self.tool._extract_normal_page(
                "What was the measured value?",
                result,
                "The measured value was 10 in 2026.",
            )

        self.assertIs(structured.call_args.args[1], CandidateExtraction)
        self.assertEqual(page.facts[0].claim, "The measured value was 10")
        self.assertFalse(page.answers_aspect)
        self.assertEqual(page.rejection_reason, "pending PCC verification")

    def test_pcc_review_is_required_before_aspect_confirmation(self):
        aspect = ResearchAspect(
            name="income",
            query="income",
            requirement="median household income",
        )
        pages = [NormalPageEvidence(
            facts=[NormalFact(
                claim="Median income was 10",
                evidence="Median income was 10",
            )],
            insufficient_information=False,
            aspect_name="income",
        )]
        unreviewed = DeepSynthesis(facts=[DeepFact(
            claim="Median income was 10",
            source_ids=[1],
        )])
        reviewed = unreviewed.model_copy(update={
            "aspect_reviews": [AspectReview(
                name="income",
                status="confirmed",
                source_ids=[1],
            )]
        })

        self.assertEqual(
            self.tool._reviewed_aspect_outcomes([aspect], pages, unreviewed)[0].status,
            AspectStatus.REJECTED,
        )
        self.assertEqual(
            self.tool._reviewed_aspect_outcomes([aspect], pages, reviewed)[0].status,
            AspectStatus.CONFIRMED,
        )

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

    def test_fact_filter_rejects_absence_statements(self):
        intent = SearchIntent(
            original_query="Norway income",
            normalized_query="Norway income",
            expected_value=ExpectedValue.FACT,
            requires_freshness=True,
            official_requested=False,
            official_domain=None,
        )

        self.assertFalse(self.tool._fact_matches_intent(
            intent,
            NormalFact(
                claim="Information about incomes is absent.",
                evidence="The page does not provide relevant information.",
            ),
        ))

    def test_pdf_scraper_uses_pdftotext_path_before_html_extractors(self):
        with patch.object(self.tool, "_scrape_pdf", return_value="x" * 300) as pdf, \
             patch.object(self.tool, "_scrape_crawl4ai") as html:
            content = self.tool._scrape("https://example.test/report.pdf")

        self.assertEqual(content, "x" * 300)
        pdf.assert_called_once()
        html.assert_not_called()

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

    def test_deep_selection_excludes_template_site_when_research_source_exists(self):
        intent = SearchIntent(
            original_query="Norway life satisfaction",
            normalized_query="Norway life satisfaction",
            expected_value=ExpectedValue.FACT,
            requires_freshness=True,
            official_requested=False,
            official_domain=None,
            subject="Norway",
        )
        self.tool.last_plan = ResearchPlan(queries=[PlannedQuery(
            query="Norway life satisfaction",
            aspect="life satisfaction",
        )])
        results = [
            {
                "title": "Life Satisfaction Survey Template",
                "href": "https://www.jotform.com/form-templates/life-satisfaction",
                "body": "life satisfaction survey form",
                "_plan_query": 0,
            },
            {
                "title": "Life satisfaction in Norway 2025",
                "href": "https://www.ssb.no/en/life-satisfaction",
                "body": "official life satisfaction statistics",
                "_plan_query": 0,
            },
        ]

        selected = self.tool._select_deep_sources(
            intent, results, ["life satisfaction"]
        )

        self.assertEqual(selected[0]["href"], "https://www.ssb.no/en/life-satisfaction")
        self.assertFalse(any("jotform.com" in item["href"] for item in selected))

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

    def test_deep_replaces_source_that_extracts_no_facts(self):
        broken = {
            "title": "Broken safety page",
            "href": "https://broken.test/safety",
            "body": "safety",
            "_plan_query": 0,
        }
        replacement = {
            "title": "Crime and safety statistics",
            "href": "https://www.ssb.no/en/safety",
            "body": "crime safety statistics",
            "_plan_query": 0,
        }
        selected = [broken]
        pages = [NormalPageEvidence(facts=[], insufficient_information=True)]
        replacement_page = NormalPageEvidence(
            facts=[NormalFact(
                claim="Crime declined",
                evidence="Official statistics",
            )],
            insufficient_information=False,
        )

        with patch.object(self.tool, "_scrape", return_value="x" * 300), \
             patch.object(
                 self.tool, "_extract_normal_page", return_value=replacement_page
             ):
            self.tool._replace_empty_extractions(
                "Norway safety", selected, [broken, replacement], {}, pages
            )

        self.assertEqual(selected[0], replacement)
        self.assertEqual(pages[0], replacement_page)

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
        self.assertEqual(self.tool.last_stats["max_llm_calls"], 8)
        self.assertIsNotNone(self.tool.last_result)
        self.assertEqual(self.tool.last_result.mode, "deep")
        self.assertEqual(self.tool.last_result.facts[0].claim, "verified")

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

    def test_normal_recovers_nested_claim_and_properties_shape(self):
        raw = (
            '{"properties":[{"facts":[{"claim":{'
            '"title":"Work conditions are generally safe in Norway",'
            '"evidence":"Most people work under good and safe conditions.",'
            '"published_at":"19/12/2024"},"type":"NormalFact"}]}]}'
        )

        recovered = self.tool._recover_normal_evidence(raw)

        self.assertEqual(len(recovered.facts), 1)
        self.assertEqual(
            recovered.facts[0].claim,
            "Work conditions are generally safe in Norway",
        )
        self.assertEqual(recovered.facts[0].published_at, "19/12/2024")

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

    def test_universal_aspect_contracts_cover_unrelated_task_classes(self):
        tasks = [
            ("country", "income and safety", "official statistics"),
            ("phone", "camera and battery", "laboratory measurements"),
            ("software", "API compatibility", "official documentation"),
            ("law", "exceptions and effective date", "primary legislation"),
            ("science", "competing positions", "peer reviewed studies"),
        ]
        for name, requirement, source_type in tasks:
            with self.subTest(task=name):
                aspect = ResearchAspect(
                    name=name,
                    query=f"test {name}",
                    requirement=requirement,
                    preferred_source_type=source_type,
                    acceptance_criteria=f"Evidence directly covers {requirement}",
                )
                self.assertEqual(aspect.requirement, requirement)
                self.assertNotEqual(aspect.acceptance_criteria, "")

    def test_aspect_outcomes_require_bound_relevant_evidence(self):
        aspects = [
            ResearchAspect(name="camera", query="camera", requirement="camera quality"),
            ResearchAspect(name="battery", query="battery", requirement="battery endurance"),
        ]
        selected = [
            {"_aspect_name": "camera"},
            {"_aspect_name": "battery"},
        ]
        pages = [
            NormalPageEvidence(
                facts=[NormalFact(claim="Camera resolves detail", evidence="Measured chart")],
                insufficient_information=False,
                answers_aspect=True,
                relevance_score=90,
            ),
            NormalPageEvidence(
                facts=[],
                insufficient_information=True,
                answers_aspect=False,
                relevance_score=20,
                rejection_reason="Only charging accessories are discussed",
            ),
        ]

        outcomes = self.tool._aspect_outcomes(aspects, selected, pages)

        self.assertEqual(outcomes[0].status, AspectStatus.CONFIRMED)
        self.assertEqual(outcomes[1].status, AspectStatus.REJECTED)
        self.assertIn("accessories", outcomes[1].failure_reason)

    def test_different_dates_are_not_a_conflict(self):
        conflict = ConflictAssessment(
            description="Sources report different years",
            source_ids=[1, 2],
            metric="population",
            unit="people",
            period="2024 versus 2025",
            geography="same country",
            definition="resident population",
        )
        self.assertFalse(self.tool._valid_conflict(conflict, 2))

    def test_conflict_requires_compatible_comparison_metadata(self):
        valid = ConflictAssessment(
            description="Two sources give incompatible values",
            source_ids=[1, 2],
            metric="battery endurance",
            unit="hours",
            period="same test run",
            geography="same market",
            definition="continuous web browsing",
        )
        missing_definition = valid.model_copy(update={"definition": ""})

        self.assertTrue(self.tool._valid_conflict(valid, 2))
        self.assertFalse(self.tool._valid_conflict(missing_definition, 2))

    def test_authoritative_fresh_source_sufficient_without_value_pattern(self):
        results = [{
            "title": "Python Releases for Windows",
            "href": "https://www.python.org/downloads/windows/",
            "body": "Latest Python 3 Release - Python 3.14.6 - June 10, 2026",
        }]
        intent = SearchIntent(
            original_query="Какая последняя версия Python?",
            normalized_query="Какая последняя версия Python?",
            expected_value=ExpectedValue.VERSION,
            requires_freshness=True,
            official_requested=False,
            official_domain="python.org",
            preferred_domains=("python.org",),
        )

        quality = self.tool._assess_quick_quality(intent, results)

        self.assertTrue(quality.sufficient)
        self.assertTrue(quality.authoritative_present)
        self.assertTrue(quality.fresh_present)

    def test_empty_structured_response_retries_through_general_path(self):
        valid = '{"facts":[],"insufficient_information":true}'
        with patch(
            "core.tools.web_search.call_llm",
            side_effect=[_completion(""), _completion(valid)],
        ) as mocked:
            result = self.tool._structured("extract", NormalPageEvidence)

        self.assertEqual(result.facts, [])
        self.assertEqual(mocked.call_count, 2)

    def test_extraction_uses_max_retries(self):
        candidates = CandidateExtraction(facts=[{
            "claim": "fact",
            "evidence": "evidence",
            "published_at": "",
        }])
        result = {
            "title": "Test",
            "href": "https://example.test/page",
            "body": "",
        }
        with patch.object(
            self.tool, "_structured", return_value=candidates
        ) as structured:
            self.tool._extract_normal_page("question", result, "page text")

        self.assertEqual(structured.call_args.kwargs.get("max_attempts"), MAX_RETRIES)

    def test_planner_normalization_uses_research_aspects_when_queries_empty(self):
        raw = (
            '{"queries":[],"search_queries":[],"subject":"Norway and Sweden",'
            '"aspects":["cost of living","incomes","housing","safety"],'
            '"expected_value":"fact","requires_freshness":true,'
            '"official_domain":"","official_domains":[],'
            '"research_aspects":['
            '{"name":"cost of living","query":"Norway Sweden cost of living comparison"},'
            '{"name":"incomes","query":"Norway Sweden average income statistics"},'
            '{"name":"housing","query":"Norway Sweden housing prices comparison"},'
            '{"name":"safety","query":"Norway Sweden crime safety statistics"}]}'
        )
        self.tool._budget = SearchBudget.for_mode(SearchMode.DEEP)

        with patch("core.tools.web_search.call_llm", return_value=_completion(raw)):
            plan, intent = self.tool._plan_research(
                "исследуй стоимость жизни в Норвегии и Швеции",
                SearchMode.DEEP,
            )

        self.assertEqual(len(plan.search_queries), 4)
        self.assertIn("cost of living", plan.search_queries[0])
        self.assertIn("safety", plan.search_queries[3])

    def test_replacement_skips_duplicate_urls(self):
        broken = {
            "title": "Broken page",
            "href": "https://broken.test/page",
            "body": "content",
            "_plan_query": 0,
        }
        duplicate_a = {
            "title": "Duplicate A",
            "href": "https://same.test/page",
            "body": "content",
            "_plan_query": 0,
        }
        duplicate_b = {
            "title": "Duplicate B",
            "href": "https://same.test/page",
            "body": "content",
            "_plan_query": 1,
        }
        selected = [broken]
        scraped = {broken["href"]: "Не удалось извлечь текст."}
        self.tool.last_intent = SearchIntent(
            original_query="test",
            normalized_query="test",
            expected_value=ExpectedValue.FACT,
            requires_freshness=False,
            official_requested=False,
            official_domain=None,
        )

        with patch.object(self.tool, "_scrape", return_value="") as scrape:
            self.tool._replace_unreadable_sources(
                selected, [broken, duplicate_a, duplicate_b], scraped
            )

        self.assertEqual(scrape.call_count, 1)
