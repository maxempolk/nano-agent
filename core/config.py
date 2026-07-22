import os


APPLE_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:1976/v1")

# Defaults target Apple FM; environment overrides can temporarily use another
# OpenAI-compatible local server without changing application code.
APPLE_LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "system")
APPLE_PCC_MODEL = os.environ.get("PCC_MODEL", "pcc")

DEFAULT_LOCAL_CONTEXT_TOKEN_BUDGET = 3000
DEFAULT_PCC_CONTEXT_TOKEN_BUDGET = 12000
