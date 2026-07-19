from __future__ import annotations
from html import escape
import time
import httpx
from urllib.parse import quote_plus
from core.agent import Agent
from core.logger import SessionLogger

MAX_TRACE_ITEM_CHARS = 600
MAX_TRACE_CHARS = 2800
MAX_COMBINED_MESSAGE_CHARS = 3800
MAX_PROGRESS_ARGUMENT_CHARS = 350
MAX_PROGRESS_RESULT_CHARS = 500
MAX_PROGRESS_TOOLS = 6


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [ещё {len(text) - limit} символов]"


def _format_tool_trace(tool_calls: list[tuple[str, str, str]], secret: str = "") -> str:
    if not tool_calls:
        return ""

    parts = ["🛠 <b>Вызванные инструменты</b>"]
    for index, (name, arguments, result) in enumerate(tool_calls):
        arguments = arguments.replace(secret, "[скрыто]") if secret else arguments
        result = result.replace(secret, "[скрыто]") if secret else result
        arguments = escape(_shorten(arguments, MAX_TRACE_ITEM_CHARS))
        result = escape(_shorten(result, MAX_TRACE_ITEM_CHARS))
        item = (
            f"\n<b>{escape(name)}</b>\n"
            f"Аргументы: <code>{arguments}</code>\n"
            f"Результат: <code>{result}</code>"
        )
        if len("\n".join(parts)) + len(item) > MAX_TRACE_CHARS:
            omitted = len(tool_calls) - index
            parts.append(f"\n<i>Ещё вызовов скрыто: {omitted}</i>")
            break
        parts.append(item)

    body = "\n".join(parts)
    return f"\n\n<blockquote expandable>{body}</blockquote>"


def _messages_with_tool_trace(reply: str, tool_calls: list[tuple[str, str, str]],
                              secret: str = "") -> list[str]:
    trace = _format_tool_trace(tool_calls, secret=secret)
    if trace and len(reply) + len(trace) > MAX_COMBINED_MESSAGE_CHARS:
        return [reply, trace.strip()]
    return [reply + trace]


def _progress_message(tool_calls: list[tuple[str, str, str]], secret: str = "") -> str:
    if not tool_calls:
        return "🧠 <b>Думаю…</b>\n<i>Начал обрабатывать запрос.</i>"

    visible = tool_calls[-MAX_PROGRESS_TOOLS:]
    completed = "\n".join(
        f"✓ <code>{escape(name)}</code>"
        for name, _, _ in visible
    )
    omitted = len(tool_calls) - len(visible)
    if omitted:
        completed = f"… ещё {omitted}\n" + completed

    name, arguments, result = tool_calls[-1]
    if secret:
        arguments = arguments.replace(secret, "[скрыто]")
        result = result.replace(secret, "[скрыто]")
    arguments = escape(_shorten(arguments, MAX_PROGRESS_ARGUMENT_CHARS))
    result = escape(_shorten(result.strip() or "Нет вывода", MAX_PROGRESS_RESULT_CHARS))
    return (
        "🧠 <b>Продолжаю работу…</b>\n"
        f"{completed}\n\n"
        f"<b>Последний инструмент:</b> <code>{escape(name)}</code>\n"
        f"<blockquote><b>Аргументы</b>\n<code>{arguments}</code>\n\n"
        f"<b>Ответ инструмента</b>\n{result}</blockquote>\n"
        "<i>Результат получен, формирую ответ.</i>"
    )


def _model_badge(agent: Agent) -> str:
    if agent.last_route_name == "pcc":
        return "\n\n<i>☁️ Apple PCC</i>"
    return "\n\n<i>🍎 AFM Core 3 · local</i>"


def _command_name(text: str) -> str:
    return text.split(maxsplit=1)[0].split("@", 1)[0].lower()


def _context_command_reply(agent: Agent, command: str) -> str | None:
    if command == "/clear":
        agent.clear_context()
        return "Контекст очищен. Начинаем новый диалог."
    if command == "/context":
        used, limit = agent.context_usage()
        return f"{used}/{limit} tokens"
    if command == "/compact":
        before, after, compacted = agent.compact_context()
        if compacted:
            return f"Контекст сжат: {before}/{agent.token_budget} → {after}/{agent.token_budget} tokens"
        return f"Сжимать пока нечего. {after}/{agent.token_budget} tokens"
    return None


def _telegram_post(base: str, method: str, data: dict,
                   logger: SessionLogger | None = None) -> dict | None:
    try:
        response = httpx.post(f"{base}/{method}", data=data, timeout=10)
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            raise RuntimeError(payload.get("description", "Telegram API error"))
        return payload
    except Exception as error:
        err = f"Telegram {method}: {error}"
        print(f"[telegram] {err}")
        if logger:
            logger.error(err)
        return None


def _send_message(base: str, chat_id: int, message: str,
                  logger: SessionLogger | None = None) -> int | None:
    payload = _telegram_post(base, "sendMessage", {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
    }, logger)
    if not payload:
        return None
    message_id = payload.get("result", {}).get("message_id")
    return message_id if isinstance(message_id, int) else None


def _edit_message(base: str, chat_id: int, message_id: int, message: str,
                  logger: SessionLogger | None = None) -> bool:
    return _telegram_post(base, "editMessageText", {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": message,
        "parse_mode": "HTML",
    }, logger) is not None


def _send_messages(base: str, chat_id: int, messages: list[str],
                   logger: SessionLogger | None = None) -> None:
    for message in messages:
        if _send_message(base, chat_id, message, logger) is None:
            break


def _deliver_final(base: str, chat_id: int, status_message_id: int | None,
                   messages: list[str], logger: SessionLogger | None = None) -> None:
    if status_message_id is not None and messages:
        if _edit_message(base, chat_id, status_message_id, messages[0], logger):
            _send_messages(base, chat_id, messages[1:], logger)
            return
    _send_messages(base, chat_id, messages, logger)


def run(agent: Agent, token: str, allowed_user_id: str, logger: SessionLogger | None = None) -> None:
    base = f"https://api.telegram.org/bot{token}"

    # Пропускаем накопленные сообщения — обрабатываем только новые
    try:
        resp = httpx.get(f"{base}/getUpdates", params={"offset": -1}, timeout=10)
        updates = resp.json().get("result", [])
        offset = updates[-1]["update_id"] + 1 if updates else 0
    except Exception as e:
        err = f"Не удалось инициализировать polling: {e}"
        print(f"[telegram] {err}")
        if logger:
            logger.error(err)
        offset = 0

    print(f"Telegram бот запущен. Слушаю сообщения от user_id={allowed_user_id}\n")

    while True:
        try:
            resp = httpx.get(
                f"{base}/getUpdates",
                params={"timeout": 30, "offset": offset},
                timeout=35
            )
            updates = resp.json().get("result", [])
        except Exception as e:
            err = f"Ошибка polling: {e}"
            print(f"[telegram] {err}")
            if logger:
                logger.error(err)
            time.sleep(5)
            continue

        for update in updates:
            offset = update["update_id"] + 1

            tg_message = update.get("message")
            if not tg_message:
                continue

            user_id = str(tg_message.get("from", {}).get("id", ""))
            if user_id != allowed_user_id:
                continue

            text = tg_message.get("text", "").strip()
            if not text:
                continue

            chat_id = tg_message["chat"]["id"]
            print(f"[telegram] {user_id}: {text}")

            command = _command_name(text)
            if command in {"/clear", "/context", "/compact"}:
                if logger:
                    logger.user(text)
                try:
                    reply = _context_command_reply(agent, command) or "Неизвестная команда"
                except Exception as e:
                    reply = f"Ошибка команды {command}: {e}"
                    if logger:
                        logger.error(reply)
                if logger:
                    logger.agent(reply)
                _send_messages(base, chat_id, [reply], logger)
                continue

            try:
                tool_calls: list[tuple[str, str, str]] = []
                status_message_id = _send_message(
                    base,
                    chat_id,
                    _progress_message(tool_calls),
                    logger,
                )

                def on_tool_call(name: str, arguments: str, result: str) -> None:
                    tool_calls.append((name, arguments, result))
                    if status_message_id is not None:
                        _edit_message(
                            base,
                            chat_id,
                            status_message_id,
                            _progress_message(tool_calls, secret=token),
                            logger,
                        )

                reply = agent.run_turn(text, on_tool_call=on_tool_call)
                reply += _model_badge(agent)
                if agent.last_search_query:
                    url = f"https://duckduckgo.com/?q={quote_plus(agent.last_search_query)}"
                    query = escape(agent.last_search_query)
                    reply += f'\n\n<i>🔍 <a href="{url}">{query}</a></i>'
                outgoing = _messages_with_tool_trace(reply, tool_calls, secret=token)
            except Exception as e:
                reply = f"Внутренняя ошибка агента: {e}"
                outgoing = [reply]
                if logger:
                    logger.error(reply)

            _deliver_final(base, chat_id, status_message_id, outgoing, logger)
