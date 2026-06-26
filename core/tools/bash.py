import subprocess

SCHEMA = {
    "type": "function",
    "function": {
        "name": "execute_bash",
        "description": "Execute a bash command.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"}
            },
            "required": ["command"]
        }
    }
}


def execute(command: str) -> str:
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return result.stdout or "Выполнено успешно (нет вывода)"
        return f"Ошибка (код {result.returncode}):\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "Ошибка: превышен таймаут 30 секунд"
    except Exception as e:
        return f"Ошибка: {str(e)}"
