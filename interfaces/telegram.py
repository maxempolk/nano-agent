from __future__ import annotations
import time
import httpx
from core.agent import Agent
from core.logger import SessionLogger


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

            try:
                reply = agent.run_turn(text)
            except Exception as e:
                reply = f"Внутренняя ошибка агента: {e}"
                if logger:
                    logger.error(reply)

            try:
                httpx.post(f"{base}/sendMessage", data={
                    "chat_id": chat_id,
                    "text": reply,
                    "parse_mode": "HTML"
                }, timeout=10)
            except Exception as e:
                err = f"Не удалось отправить сообщение: {e}"
                print(f"[telegram] {err}")
                if logger:
                    logger.error(err)
