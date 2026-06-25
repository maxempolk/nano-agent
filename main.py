from openai import OpenAI
import json
import subprocess

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": (
                "Выполнить команду в bash. "
                "Используй для работы с файлами, запуска скриптов, "
                "установки пакетов и других системных операций."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Команда для выполнения в bash, например: ls -la или pip install requests"
                    }
                },
                "required": ["command"]
            }
        }
    }
]


def execute_bash(command: str) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout or "Выполнено успешно (нет вывода)"
        else:
            return f"Ошибка (код {result.returncode}):\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "Ошибка: превышен таймаут 30 секунд"
    except Exception as e:
        return f"Ошибка: {str(e)}"


messages = [
    {"role": "system", "content": "Ты полезный ассистент с доступом к bash."}
]

print("Агент запущен. Для выхода нажмите Ctrl+C.\n")

while True:
    try:
        user_input = input("Вы: ").strip()
    except KeyboardInterrupt:
        print("\nВыход.")
        break

    if not user_input:
        continue

    messages.append({"role": "user", "content": user_input})

    # Цикл обработки вызовов инструментов
    while True:
        response = client.chat.completions.create(
            model="qwen3.5-4b",
            messages=messages,  # type: ignore
            tools=tools,        # type: ignore
            tool_choice="auto"
        )

        message = response.choices[0].message

        if message.tool_calls:
            messages.append(message)  # type: ignore

            for call in message.tool_calls:
                print(f"  [инструмент] {call.function.name}({call.function.arguments})")
                args = json.loads(call.function.arguments)  # type: ignore
                result = execute_bash(**args)
                print(f"  [результат] {result.strip()}\n")

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result
                })
        else:
            # Финальный текстовый ответ
            reply = message.content or ""
            messages.append({"role": "assistant", "content": reply})
            print(f"Агент: {reply}\n")
            break
