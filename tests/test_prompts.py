from unittest import TestCase

from core.prompts import build_prompt_set


class PromptProfileTests(TestCase):
    def test_profiles_require_corrected_retry_after_tool_error(self):
        for name in ("full", "mini"):
            prompt = build_prompt_set(name, system_info="SYS").agent.lower()
            self.assertIn("tool", prompt)
            self.assertIn("error", prompt)
            self.assertIn("correct", prompt)
            self.assertIn("3", prompt)

    def test_profiles_make_search_mandatory_for_current_and_explicit_requests(self):
        for name in ("full", "mini"):
            prompt = build_prompt_set(name, system_info="SYS").agent.lower()
            self.assertIn("must call web_search", prompt)
            self.assertIn("latest/current/today", prompt)
            self.assertIn("previous user topic", prompt)
            self.assertIn("before one web_search call", prompt)
            self.assertIn("official sources", prompt)
            self.assertIn("never invent checked sources", prompt)
