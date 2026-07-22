from __future__ import annotations
from html import escape
import json
import re
import time
import httpx
from urllib.parse import quote_plus
from core.agent import Agent
from core.logger import SessionLogger

MAX_TRACE_ARGUMENT_CHARS = 180
MAX_TRACE_RESULT_CHARS = 280
MAX_TRACE_CHARS = 1800
MAX_COMBINED_MESSAGE_CHARS = 3800
MAX_PROGRESS_ACTION_CHARS = 140
MAX_PROGRESS_TOOLS = 20


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [ещё {len(text) - limit} символов]"


def _inline_markdown_to_html(text: str) -> str:
    placeholders: dict[str, str] = {}

    def stash(value: str) -> str:
        token = f"\x00{len(placeholders)}\x00"
        placeholders[token] = value
        return token

    text = re.sub(
        r"`([^`\n]+)`",
        lambda match: stash(f"<code>{escape(match.group(1))}</code>"),
        text,
    )
    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
        lambda match: stash(
            f'<a href="{escape(match.group(2), quote=True)}">'
            f"{escape(match.group(1))}</a>"
        ),
        text,
    )
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", text)
    for token, value in placeholders.items():
        text = text.replace(token, value)
    return text


def _markdown_to_telegram_html(markdown: str) -> str:
    lines = markdown.splitlines()
    output: list[str] = []
    code_lines: list[str] | None = None

    for line in lines:
        if line.strip().startswith("```"):
            if code_lines is None:
                code_lines = []
            else:
                output.append(
                    f"<pre><code>{escape(chr(10).join(code_lines))}</code></pre>"
                )
                code_lines = None
            continue
        if code_lines is not None:
            code_lines.append(line)
            continue

        heading = re.match(r"^#{1,6}\s+(.+)$", line)
        if heading:
            output.append(f"<b>{_inline_markdown_to_html(heading.group(1))}</b>")
            continue
        bullet = re.match(r"^\s*[-*+]\s+(.+)$", line)
        if bullet:
            output.append(f"• {_inline_markdown_to_html(bullet.group(1))}")
            continue
        quote = re.match(r"^\s*>\s?(.*)$", line)
        if quote:
            output.append(
                f"<blockquote>{_inline_markdown_to_html(quote.group(1))}</blockquote>"
            )
            continue
        output.append(_inline_markdown_to_html(line))

    if code_lines is not None:
        output.append(f"<pre><code>{escape(chr(10).join(code_lines))}</code></pre>")
    return "\n".join(output)


def _format_tool_trace(tool_calls: list[tuple[str, str, str]], secret: str = "") -> str:
    if not tool_calls:
        return ""

    parts = ["🛠 <b>Вызванные инструменты</b>"]
    for index, (name, arguments, result) in enumerate(tool_calls):
        arguments = arguments.replace(secret, "[скрыто]") if secret else arguments
        result = result.replace(secret, "[скрыто]") if secret else result
        arguments = escape(_shorten(arguments, MAX_TRACE_ARGUMENT_CHARS))
        result = escape(_shorten(result, MAX_TRACE_RESULT_CHARS))
        item = (
            f"\n<b>{index + 1}. {escape(name)}</b>\n"
            f"<code>{arguments}</code>\n"
            f"<i>Результат:</i> <code>{result}</code>"
        )
        if len("\n".join(parts)) + len(item) > MAX_TRACE_CHARS:
            omitted = len(tool_calls) - index
            parts.append(f"\n<i>Ещё вызовов скрыто: {omitted}</i>")
            break
        parts.append(item)

    body = "\n".join(parts)
    return f"\n\n<blockquote expandable>{body}</blockquote>"


def _tool_action(name: str, arguments: str) -> str:
    try:
        values = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        values = {}
    if not isinstance(values, dict):
        values = {}

    if name == "web_search":
        query = str(values.get("query") or "поиск в интернете")
        depth = values.get("depth")
        suffix = f" · {depth}" if depth else ""
        return f"ищет «{query}»{suffix}"
    if name == "execute_bash":
        return f"выполняет команду {values.get('command') or ''}".strip()
    if name == "cron":
        return f"работает с расписанием: {values.get('action') or 'операция'}"

    for key in ("query", "action", "command", "url", "path"):
        if values.get(key):
            return f"{key}: {values[key]}"
    return "выполняет операцию"


def _messages_with_tool_trace(reply: str, tool_calls: list[tuple[str, str, str]],
                              secret: str = "") -> list[str]:
    if not reply.strip():
        reply = "Модель не сформировала текстовый ответ. Результаты инструментов приведены ниже."
    trace = _format_tool_trace(tool_calls, secret=secret)
    if trace and len(reply) + len(trace) > MAX_COMBINED_MESSAGE_CHARS:
        return [reply, trace.strip()]
    return [reply + trace]


def _progress_message(tool_calls: list[tuple[str, str, str]], secret: str = "") -> str:
    if not tool_calls:
        return "🧠 <b>Думаю…</b>\n<i>Начал обрабатывать запрос.</i>"

    visible = tool_calls[-MAX_PROGRESS_TOOLS:]
    completed_rows = []
    for index, (name, arguments, _) in enumerate(visible, start=1):
        if secret:
            arguments = arguments.replace(secret, "[скрыто]")
        action = escape(_shorten(
            _tool_action(name, arguments), MAX_PROGRESS_ACTION_CHARS
        )).replace("\n", " ")
        completed_rows.append(
            f"{index}. ✓ <code>{escape(name)}</code> — <i>{action}</i>"
        )
    completed = "\n".join(completed_rows)
    omitted = len(tool_calls) - len(visible)
    if omitted:
        completed = f"… ещё {omitted}\n" + completed

    return (
        "🧠 <b>Продолжаю работу…</b>\n"
        f"{completed}\n\n"
        "<i>Инструменты завершены, формирую ответ.</i>"
    )


def _model_badge(agent: Agent) -> str:
    model = agent.model or "unknown"
    if agent.last_route_name == "pcc":
        return f"\n\n<i>☁️ {model}</i>"
    return f"\n\n<i>🍎 {model}</i>"


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
        if logger:
            message_id = payload.get("result", {}).get("message_id", "-")
            logger.info(f"Telegram {method} ok | message_id={message_id}")
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
    messages = [message for message in messages if message.strip()]
    if not messages:
        messages = ["Не удалось сформировать ответ. Попробуйте повторить запрос."]
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

                reply = _markdown_to_telegram_html(
                    agent.run_turn(text, on_tool_call=on_tool_call)
                )
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
