from unittest import TestCase
from unittest.mock import Mock, patch

from interfaces.telegram import (
    _command_name,
    _context_command_reply,
    _deliver_final,
    _format_tool_trace,
    _messages_with_tool_trace,
    _markdown_to_telegram_html,
    _progress_message,
    _telegram_post,
)


class TelegramContextCommandTests(TestCase):
    def test_markdown_is_rendered_as_safe_telegram_html(self):
        rendered = _markdown_to_telegram_html(
            "### Вывод\n\n**Важно:** данные [SSB](https://ssb.no?a=1&b=2).\n"
            "- Значение `118 307 NOK`\n<script>"
        )

        self.assertIn("<b>Вывод</b>", rendered)
        self.assertIn("<b>Важно:</b>", rendered)
        self.assertIn('<a href="https://ssb.no?a=1&amp;b=2">SSB</a>', rendered)
        self.assertIn("• Значение <code>118 307 NOK</code>", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertNotIn("###", rendered)
        self.assertNotIn("**", rendered)

    def test_fenced_code_becomes_telegram_pre_block(self):
        rendered = _markdown_to_telegram_html(
            "```python\nprint(\"<ok>\")\n```"
        )

        self.assertEqual(
            rendered,
            "<pre><code>print(&quot;&lt;ok&gt;&quot;)</code></pre>",
        )

    def test_command_with_bot_name(self):
        self.assertEqual(_command_name("/context@my_bot"), "/context")
        self.assertEqual(_command_name("/compact now"), "/compact")

    def test_context(self):
        agent = Mock()
        agent.context_usage.return_value = (2400, 6000)

        reply = _context_command_reply(agent, "/context")

        self.assertEqual(reply, "2400/6000 tokens")

    def test_compact(self):
        agent = Mock(token_budget=6000)
        agent.compact_context.return_value = (2400, 900, True)

        reply = _context_command_reply(agent, "/compact")

        self.assertEqual(reply, "Контекст сжат: 2400/6000 → 900/6000 tokens")

    def test_compact_with_nothing_to_summarize(self):
        agent = Mock(token_budget=6000)
        agent.compact_context.return_value = (500, 500, False)

        reply = _context_command_reply(agent, "/compact")

        self.assertEqual(reply, "Сжимать пока нечего. 500/6000 tokens")

    def test_clear(self):
        agent = Mock()

        reply = _context_command_reply(agent, "/clear")

        agent.clear_context.assert_called_once_with()
        self.assertEqual(reply, "Контекст очищен. Начинаем новый диалог.")

    def test_progress_accumulates_compact_tool_actions_without_quote_or_results(self):
        message = _progress_message([
            (
                "web_search",
                '{"query":"latest GPT","token":"secret-token"}',
                "Found GPT-5.6 using secret-token",
            ),
            (
                "execute_bash",
                '{"command":"git status"}',
                "clean",
            ),
        ], secret="secret-token")

        self.assertIn("Продолжаю работу", message)
        self.assertIn("web_search", message)
        self.assertIn("latest GPT", message)
        self.assertIn("execute_bash", message)
        self.assertIn("git status", message)
        self.assertNotIn("Found GPT-5.6", message)
        self.assertNotIn("<blockquote", message)
        self.assertNotIn("secret-token", message)

    def test_final_tool_trace_is_compact_expandable_quote(self):
        trace = _format_tool_trace([
            ("web_search", '{"query":"latest GPT"}', "x" * 1000),
        ])

        self.assertIn("<blockquote expandable>", trace)
        self.assertIn("web_search", trace)
        self.assertLess(len(trace), 1000)

    def test_successful_telegram_delivery_is_logged(self):
        response = Mock()
        response.json.return_value = {
            "ok": True,
            "result": {"message_id": 321},
        }
        logger = Mock()

        with patch("interfaces.telegram.httpx.post", return_value=response):
            payload = _telegram_post("base", "sendMessage", {"text": "hi"}, logger)

        self.assertTrue(payload["ok"])
        logger.info.assert_called_once_with(
            "Telegram sendMessage ok | message_id=321"
        )

    def test_final_replaces_progress_message_then_sends_overflow(self):
        with patch("interfaces.telegram._edit_message", return_value=True) as edit, \
             patch("interfaces.telegram._send_messages") as send:
            _deliver_final("base", 123, 456, ["answer", "trace"])

        edit.assert_called_once_with("base", 123, 456, "answer", None)
        send.assert_called_once_with("base", 123, ["trace"], None)

    def test_empty_agent_reply_is_never_delivered_silently(self):
        messages = _messages_with_tool_trace("", [])

        self.assertEqual(len(messages), 1)
        self.assertTrue(messages[0].strip())

    def test_empty_final_replaces_progress_with_explicit_error(self):
        with patch("interfaces.telegram._edit_message", return_value=True) as edit:
            _deliver_final("base", 123, 456, [])

        self.assertIn("Не удалось сформировать ответ", edit.call_args.args[3])
