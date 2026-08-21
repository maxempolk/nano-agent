import threading
import time
from unittest import TestCase
from unittest.mock import Mock, patch

from core.agent import Agent
from core.tools.base import ToolRegistry
from interfaces.telegram import (
    _cancel_command_reply,
    _handle_update,
    _process_message,
    _transcribe_voice,
)
from tests.test_agent_forced_search import FakeBashTool, _completion, _tool_completion


def _update(text: str, user_id: str = "42", chat_id: int = 42, update_id: int = 1) -> dict:
    return {
        "update_id": update_id,
        "message": {"from": {"id": user_id}, "chat": {"id": chat_id}, "text": text},
    }


def _voice_update(user_id: str = "42", chat_id: int = 42, update_id: int = 1) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "from": {"id": user_id},
            "chat": {"id": chat_id},
            "voice": {"file_id": "abc123", "duration": 3, "file_size": 100},
        },
    }


def _wait_for_unlock(lock: threading.Lock) -> None:
    deadline = time.monotonic() + 2
    while lock.locked() and time.monotonic() < deadline:
        time.sleep(0.01)


class CancelCommandTests(TestCase):
    def test_cancel_without_active_run(self):
        agent = Agent(None, "system", "SYSTEM")
        self.assertEqual(_cancel_command_reply(agent), "Сейчас нет активного запроса.")

    def test_cancel_with_active_run_calls_agent_cancel(self):
        agent = Mock()
        agent.run_in_progress = True
        reply = _cancel_command_reply(agent)
        agent.cancel.assert_called_once_with("команда /cancel")
        self.assertIn("Отменяю", reply)


class HandleUpdateTests(TestCase):
    def test_foreign_user_is_ignored(self):
        agent = Mock()
        lock = threading.Lock()
        with patch("interfaces.telegram._send_messages") as send:
            _handle_update(agent, _update("привет", user_id="999"), "base", "tok", "42", lock)
        send.assert_not_called()
        agent.run_turn.assert_not_called()

    def test_cancel_command_sends_reply_without_running_agent(self):
        agent = Agent(None, "system", "SYSTEM")
        lock = threading.Lock()
        with patch("interfaces.telegram._send_messages") as send:
            _handle_update(agent, _update("/cancel"), "base", "tok", "42", lock)
        send.assert_called_once()
        self.assertIn("нет активного запроса", send.call_args.args[2][0])

    def test_busy_guard_rejects_second_run(self):
        agent = Agent(None, "system", "SYSTEM")
        lock = threading.Lock()
        lock.acquire()
        with patch("interfaces.telegram._send_messages") as send:
            _handle_update(agent, _update("ещё один запрос"), "base", "tok", "42", lock)
        send.assert_called_once()
        self.assertIn("обрабатываю предыдущий", send.call_args.args[2][0])

    def test_run_message_spawns_worker_and_releases_lock(self):
        agent = Agent(None, "system", "SYSTEM")
        lock = threading.Lock()
        started = threading.Event()

        def fake_process(*args, **kwargs):
            started.set()

        with patch("interfaces.telegram._process_message", side_effect=fake_process):
            _handle_update(agent, _update("сделай что-нибудь"), "base", "tok", "42", lock)

        self.assertTrue(started.wait(2))
        deadline = time.monotonic() + 2
        while lock.locked() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(lock.locked())

    def test_context_commands_still_work(self):
        agent = Agent(None, "system", "SYSTEM")
        lock = threading.Lock()
        with patch("interfaces.telegram._send_messages") as send:
            _handle_update(agent, _update("/context"), "base", "tok", "42", lock)
        self.assertIn("tokens", send.call_args.args[2][0])


class ProcessMessageTests(TestCase):
    def test_final_reply_is_delivered_with_model_badge(self):
        agent = Agent(None, "system", "SYSTEM")

        with (
            patch("core.agent.call_llm", return_value=_completion("Готово")),
            patch("interfaces.telegram._send_message", return_value=7) as send,
            patch("interfaces.telegram._deliver_final") as deliver,
        ):
            _process_message(agent, "base", 42, "запрос", "secret-token", None)

        send.assert_called_once()
        delivered_messages = deliver.call_args.args[3]
        self.assertIn("Готово", delivered_messages[0])

    def test_tool_progress_events_update_status_message(self):
        bash = FakeBashTool()
        agent = Agent(None, "system", "SYSTEM", registry=ToolRegistry([bash]))
        edits: list[str] = []

        def fake_edit(base, chat_id, message_id, text, logger=None):
            edits.append(text)
            return True

        with (
            patch(
                "core.agent.call_llm",
                side_effect=[
                    _tool_completion("execute_bash", '{"command":"git status"}'),
                    _completion("чисто"),
                ],
            ),
            patch("interfaces.telegram._send_message", return_value=5),
            patch("interfaces.telegram._edit_message", side_effect=fake_edit),
            patch("interfaces.telegram._deliver_final") as deliver,
        ):
            _process_message(agent, "base", 42, "проверь git", "secret-token", None)

        self.assertTrue(edits, "прогресс не редактировался")
        self.assertIn("execute_bash", edits[-1])
        self.assertNotIn("чисто", edits[-1], "результат не должен попасть в прогресс")
        final_text = deliver.call_args.args[3][0]
        self.assertIn("чисто", final_text)

    def test_agent_error_is_delivered_not_raised(self):
        agent = Agent(None, "system", "SYSTEM")

        with (
            patch("core.agent.call_llm", side_effect=RuntimeError("мост лёг")),
            patch("interfaces.telegram._send_message", return_value=None),
            patch("interfaces.telegram._deliver_final") as deliver,
        ):
            _process_message(agent, "base", 42, "запрос", "tok", None)

        delivered = deliver.call_args.args[3]
        self.assertTrue(any("Внутренняя ошибка агента" in message for message in delivered))

    def test_secret_is_hidden_in_tool_trace(self):
        bash = FakeBashTool()
        agent = Agent(None, "system", "SYSTEM", registry=ToolRegistry([bash]))
        secret = "super-secret-token"

        with (
            patch(
                "core.agent.call_llm",
                side_effect=[
                    _tool_completion("execute_bash", f'{{"command":"curl {secret}"}}'),
                    _completion("отправил"),
                ],
            ),
            patch("interfaces.telegram._send_message", return_value=None),
            patch("interfaces.telegram._deliver_final") as deliver,
        ):
            _process_message(agent, "base", 42, "отправь", secret, None)

        for message in deliver.call_args.args[3]:
            self.assertNotIn(secret, message)


class VoiceMessageTests(TestCase):
    def _mock_agent(self) -> Mock:
        agent = Mock()
        agent.run_turn.return_value = "готово"
        agent.model = "model"
        agent.last_route_name = "cloud"
        agent.last_search_query = ""
        return agent

    def test_voice_is_transcribed_and_run_by_agent(self):
        agent = self._mock_agent()
        lock = threading.Lock()

        with (
            patch(
                "interfaces.telegram._transcribe_voice", return_value="выключить духовку"
            ) as transcribe,
            patch("interfaces.telegram._send_message", return_value=1),
            patch("interfaces.telegram._edit_message", return_value=True),
            patch("interfaces.telegram._deliver_final"),
        ):
            _handle_update(
                agent,
                _voice_update(),
                "base",
                "tok",
                "42",
                lock,
                stt_client=object(),
                stt_model="whisper-large-v3-turbo",
            )

        _wait_for_unlock(lock)
        transcribe.assert_called_once()
        self.assertIn("abc123", transcribe.call_args.args)
        agent.run_turn.assert_called_once_with("выключить духовку")

    def test_voice_without_stt_is_rejected_without_agent_run(self):
        agent = self._mock_agent()
        lock = threading.Lock()

        with patch("interfaces.telegram._send_messages") as send:
            _handle_update(agent, _voice_update(), "base", "tok", "42", lock)

        send.assert_called_once()
        self.assertIn("не понимаю", send.call_args.args[2][0])
        agent.run_turn.assert_not_called()

    def test_voice_recognized_as_empty_is_reported(self):
        agent = self._mock_agent()
        lock = threading.Lock()

        with (
            patch("interfaces.telegram._transcribe_voice", return_value="   "),
            patch("interfaces.telegram._send_message", return_value=1),
            patch("interfaces.telegram._deliver_final") as deliver,
        ):
            _handle_update(
                agent, _voice_update(), "base", "tok", "42", lock, stt_client=object()
            )

        _wait_for_unlock(lock)
        agent.run_turn.assert_not_called()
        self.assertTrue(
            any("Не удалось распознать" in message for message in deliver.call_args.args[3])
        )

    def test_transcription_error_delivers_message_not_exception(self):
        agent = self._mock_agent()

        with (
            patch("interfaces.telegram._transcribe_voice", side_effect=RuntimeError("STT лёг")),
            patch("interfaces.telegram._send_message", return_value=3),
            patch("interfaces.telegram._deliver_final") as deliver,
        ):
            _process_message(
                agent,
                "base",
                42,
                "",
                "tok",
                None,
                voice={"file_id": "x"},
                stt_client=object(),
                stt_model="w",
            )

        agent.run_turn.assert_not_called()
        self.assertTrue(
            any("Не удалось распознать" in message for message in deliver.call_args.args[3])
        )


class TranscribeVoiceTests(TestCase):
    def test_downloads_file_and_returns_transcription(self):
        stt = Mock()
        stt.audio.transcriptions.create.return_value = "  привет  "

        with (
            patch(
                "interfaces.telegram._telegram_post",
                return_value={"ok": True, "result": {"file_path": "voice/file_1.oga"}},
            ),
            patch("interfaces.telegram.httpx.get") as get,
        ):
            get.return_value.content = b"audio"
            get.return_value.raise_for_status = Mock()
            result = _transcribe_voice("base", "tok", "fid", stt, "whisper-large-v3-turbo")

        self.assertEqual(result, "привет")
        get.assert_called_once()
        self.assertIn("api.telegram.org/file/bot", get.call_args.args[0])
        kwargs = stt.audio.transcriptions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "whisper-large-v3-turbo")
        self.assertEqual(kwargs["file"][0], "voice.ogg")

    def test_missing_file_path_raises(self):
        with patch("interfaces.telegram._telegram_post", return_value=None):
            with self.assertRaises(RuntimeError):
                _transcribe_voice("base", "tok", "fid", Mock(), "w")


if __name__ == "__main__":
    import unittest

    unittest.main()
