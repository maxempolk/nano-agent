from unittest import TestCase
from unittest.mock import Mock, patch

from interfaces.telegram import (
    _command_name,
    _context_command_reply,
    _deliver_final,
    _messages_with_tool_trace,
    _progress_message,
)


class TelegramContextCommandTests(TestCase):
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

    def test_progress_shows_tool_arguments_result_and_hides_secret(self):
        message = _progress_message([
            (
                "web_search",
                '{"query":"latest GPT","token":"secret-token"}',
                "Found GPT-5.6 using secret-token",
            )
        ], secret="secret-token")

        self.assertIn("Продолжаю работу", message)
        self.assertIn("web_search", message)
        self.assertIn("latest GPT", message)
        self.assertIn("Found GPT-5.6", message)
        self.assertNotIn("secret-token", message)

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
