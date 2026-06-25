from core.agent import Agent


def run(agent: Agent) -> None:
    print("Агент запущен. Для выхода нажмите Ctrl+C.\n")

    while True:
        try:
            user_input = input("Вы: ").strip()
        except KeyboardInterrupt:
            print("\nВыход.")
            break

        if not user_input:
            continue

        def on_tool_call(name: str, args: str, result: str) -> None:
            print(f"  [инструмент] {name}({args})")
            print(f"  [результат] {result.strip()}\n")

        reply = agent.run_turn(user_input, on_tool_call=on_tool_call)
        print(f"Агент: {reply}\n")
