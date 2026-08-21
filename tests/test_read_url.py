from unittest import TestCase

from core.tools.base import ErrorCode, ToolContext, ToolRegistry
from core.tools.read_url import ReadUrlTool


class FakeEngine:
    def __init__(self, page: str = "содержимое страницы"):
        self.page = page
        self.urls: list[str] = []

    def read_page(self, url: str) -> str:
        self.urls.append(url)
        return self.page


class ReadUrlToolTests(TestCase):
    def _tool(self, engine: FakeEngine) -> ReadUrlTool:
        return ReadUrlTool(engine)

    def test_reads_page_content(self):
        engine = FakeEngine(page="Привет со страницы")
        result = self._tool(engine).run({"url": "https://example.com/page"}, ToolContext())

        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.content, "Привет со страницы")
        self.assertEqual(engine.urls, ["https://example.com/page"])
        self.assertEqual(result.meta["url"], "https://example.com/page")
        self.assertIn("прочитал", result.summary)

    def test_relative_url_is_rejected(self):
        engine = FakeEngine()
        result = self._tool(engine).run({"url": "/local/path"}, ToolContext())

        self.assertEqual(result.error_code, ErrorCode.INVALID_ARGUMENTS)
        self.assertEqual(engine.urls, [])

    def test_ftp_scheme_is_rejected(self):
        engine = FakeEngine()
        result = self._tool(engine).run({"url": "ftp://example.com/file"}, ToolContext())

        self.assertEqual(result.error_code, ErrorCode.INVALID_ARGUMENTS)
        self.assertEqual(engine.urls, [])

    def test_extraction_failure_is_not_retryable(self):
        engine = FakeEngine(page="Не удалось извлечь текст.")
        result = self._tool(engine).run({"url": "https://example.com/gated"}, ToolContext())

        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ErrorCode.TOOL_FAILED)
        self.assertFalse(result.retryable)

    def test_extraction_failure_asks_user_before_fallback_search(self):
        engine = FakeEngine(page="Не удалось извлечь текст.")
        result = self._tool(engine).run({"url": "https://example.com/gated"}, ToolContext())

        self.assertIn("спроси", result.error)
        self.assertIn("не начинай поиск сам", result.error)

    def test_long_content_is_capped(self):
        engine = FakeEngine(page="а" * 20_000)
        result = self._tool(engine).run({"url": "https://example.com/long"}, ToolContext())

        self.assertTrue(result.ok)
        self.assertLessEqual(len(result.content), 6000 + 100)
        self.assertIn("обрезан", result.content)

    def test_registry_schema_exposes_url(self):
        registry = ToolRegistry([self._tool(FakeEngine())])
        schema = registry.schemas()[0]["function"]

        self.assertEqual(schema["name"], "read_url")
        self.assertIn("url", schema["parameters"]["properties"])
