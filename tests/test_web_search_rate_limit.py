from unittest import TestCase
from unittest.mock import Mock, patch

from openai import OpenAI

from core.tools.web_search import WebSearchTool


def _engine() -> WebSearchTool:
    client = OpenAI(base_url="http://test.local/v1", api_key="x")
    return WebSearchTool(client, "model")


class CallModelRateLimitTests(TestCase):
    def test_rate_limit_is_retried_with_backoff(self):
        engine = _engine()
        ok_response = Mock()

        with (
            patch(
                "core.tools.web_search.call_llm",
                side_effect=[RuntimeError("Rate limit reached for model"), ok_response],
            ) as call,
            patch("core.tools.web_search.time.sleep") as sleep,
        ):
            result = engine._call_model([{"role": "user", "content": "q"}], "plan")

        self.assertIs(result, ok_response)
        self.assertEqual(call.call_count, 2)
        sleep.assert_called_once_with(5.0)

    def test_rate_limit_stops_after_bounded_attempts(self):
        engine = _engine()

        with (
            patch(
                "core.tools.web_search.call_llm",
                side_effect=RuntimeError("Rate limit reached"),
            ) as call,
            patch("core.tools.web_search.time.sleep"),
        ):
            with self.assertRaises(RuntimeError):
                engine._call_model([{"role": "user", "content": "q"}], "plan")

        self.assertEqual(call.call_count, 3)

    def test_other_errors_are_not_retried(self):
        engine = _engine()

        with (
            patch(
                "core.tools.web_search.call_llm",
                side_effect=RuntimeError("boom"),
            ) as call,
            patch("core.tools.web_search.time.sleep"),
        ):
            with self.assertRaises(RuntimeError):
                engine._call_model([{"role": "user", "content": "q"}], "plan")

        self.assertEqual(call.call_count, 1)


if __name__ == "__main__":
    import unittest

    unittest.main()
