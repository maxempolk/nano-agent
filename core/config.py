MODEL = "qwen/qwen3-32b" # Основная модель — лучший tool calling, thinking mode, 60 RPM
MODEL_FALLBACK = "openai/gpt-oss-120b" # Фоллбек — быстрее, дешевле по input, встроенный reasoning
MODEL_MINI = "openai/gpt-oss-20b"        # 1000 t/s, качество o3-mini уровня
MODEL_MINI_FALLBACK = "llama-3.1-8b-instant"  # 560 t/s, production фоллбек

# Apple Foundation Models через локальный OpenAI-совместимый сервер
LOCAL_BASE_URL = "http://127.0.0.1:1976/v1"
LOCAL_MODEL = "system"
LOCAL_MODEL_FALLBACK = "system"
LOCAL_MODEL_MINI = "system"
LOCAL_MODEL_MINI_FALLBACK = "system"
