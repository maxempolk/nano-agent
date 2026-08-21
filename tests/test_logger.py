import tempfile
from unittest import TestCase

from core.logger import SessionLogger


class SessionLoggerRedactionTests(TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.logger = SessionLogger(self._tmp.name)

    def _log_text(self) -> str:
        with open(self.logger.path, encoding="utf-8") as f:
            return f.read()

    def test_registered_secret_never_reaches_the_log(self):
        self.logger.add_secret("super-secret-token")

        self.logger.user("напомни через curl с super-secret-token")
        self.logger.tool_call("execute_bash", '{"command":"curl super-secret-token"}')
        self.logger.tool_result("ответ содержит super-secret-token")
        self.logger.agent("готово")
        self.logger.info("secret=super-secret-token")
        self.logger.error("упало с super-secret-token")

        content = self._log_text()
        self.assertNotIn("super-secret-token", content)
        self.assertIn("[скрыто]", content)

    def test_empty_secret_is_ignored(self):
        self.logger.add_secret("")
        self.logger.add_secret("   ")
        self.logger.info("обычный текст")
        self.assertIn("обычный текст", self._log_text())

    def test_multiple_secrets_are_redacted(self):
        self.logger.add_secret("token-a")
        self.logger.add_secret("user-42")
        self.logger.info("token-a и user-42 в одной строке")
        content = self._log_text()
        self.assertNotIn("token-a", content)
        self.assertNotIn("user-42", content)


if __name__ == "__main__":
    import unittest

    unittest.main()
