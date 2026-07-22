from unittest import TestCase

from core.prompts import build_prompt_set


class PromptProfileTests(TestCase):
    def test_profiles_require_corrected_retry_after_tool_error(self):
        for name in ("full", "mini"):
            prompt = build_prompt_set(name, system_info="SYS").agent.lower()
            self.assertIn("инструмент", prompt)
            self.assertIn("ошибк", prompt)
            self.assertIn("исправь", prompt)
            self.assertIn("3", prompt)

    def test_profiles_make_search_mandatory_for_current_and_explicit_requests(self):
        for name in ("full", "mini"):
            prompt = build_prompt_set(name, system_info="SYS").agent.lower()
            self.assertIn("обязательно вызывай web_search", prompt)
            self.assertIn("последние/текущие/сегодняшние", prompt)
            self.assertIn("предыдущей теме", prompt)
            self.assertIn("до одного вызова web_search", prompt)
            self.assertIn("официальные источники", prompt)
            self.assertIn("никогда не выдумывай проверенные источники", prompt)
