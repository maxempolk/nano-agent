import time
import httpx
from core.agent import Agent


def run(agent: Agent, token: str, allowed_user_id: str) -> None:
    base = f"https://api.telegram.org/bot{token}"
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
            print(f"[telegram] Ошибка polling: {e}")
            time.sleep(5)
            continue

        for update in updates:
            offset = update["update_id"] + 1

            msg = update.get("message")
            if not msg:
                continue

            user_id = str(msg.get("from", {}).get("id", ""))
            if user_id != allowed_user_id:
                continue

            text = msg.get("text", "").strip()
            if not text:
                continue

            chat_id = msg["chat"]["id"]
            print(f"[telegram] {user_id}: {text}")

            reply = agent.run_turn(text)

            httpx.post(f"{base}/sendMessage", data={
                "chat_id": chat_id,
                "text": reply,
                "parse_mode": "HTML"
            })
