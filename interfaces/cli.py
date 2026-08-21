from __future__ import annotations

import signal

from core.agent import Agent
from core.events import AgentEvent, RunCancelled, RunFailed, ToolCompleted, ToolStarted


def _progress_printer(event: AgentEvent) -> None:
    if isinstance(event, ToolStarted):
        print(f"  [инструмент] {event.name} {event.args_summary}")
    elif isinstance(event, ToolCompleted):
        status = "ok" if event.ok else f"ошибка: {event.error_code}"
        print(f"  [результат] {event.name}: {event.summary} ({status}, {event.elapsed}s)")
    elif isinstance(event, RunCancelled):
        print(f"  [отменено] {event.reason}")
    elif isinstance(event, RunFailed):
        print(f"  [ошибка] {event.error}")


def run(agent: Agent) -> None:
    print("Агент запущен. Ctrl+C во время запроса — отмена, Ctrl+C при вводе — выход.\n")
    agent.events.subscribe(_progress_printer)
    interrupts = {"count": 0}

    def _sigint_handler(_signum, _frame):
        interrupts["count"] += 1
        if interrupts["count"] == 1 and agent.run_in_progress:
            agent.cancel("Ctrl+C")
            print("\n  [отмена запрошена — жду остановки…]")
            return
        raise KeyboardInterrupt

    while True:
        try:
            user_input = input("Вы: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nВыход.")
            break

        if not user_input:
            continue

        interrupts["count"] = 0
        previous_handler = signal.signal(signal.SIGINT, _sigint_handler)
        try:
            reply = agent.run_turn(user_input)
        except KeyboardInterrupt:
            print("\nПринудительное прерывание. Выход.")
            break
        finally:
            signal.signal(signal.SIGINT, previous_handler)

        print(f"[{agent.last_route_name} · {agent.model}]")
        print(f"Агент: {reply}\n")
