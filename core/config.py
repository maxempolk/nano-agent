MODEL = "qwen/qwen3-32b"           # основная: лучший tool calling, 60 RPM
MODEL_FALLBACK = "llama-3.3-70b-versatile"  # фоллбек если qwen недоступен
MODEL_MINI = "llama-3.1-8b-instant"         # роутер/нормализация
MODEL_MINI_FALLBACK = "llama-3.1-8b-instant" # тот же, просто резервный эндпоинт