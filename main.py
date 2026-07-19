import argparse
from datetime import datetime
import os
import platform
import sys

from dotenv import load_dotenv
from openai import OpenAI

from core.agent import Agent
from core.config import (
    APPLE_BASE_URL,
    APPLE_LOCAL_MODEL,
    APPLE_PCC_MODEL,
    DEFAULT_LOCAL_CONTEXT_TOKEN_BUDGET,
    DEFAULT_PCC_CONTEXT_TOKEN_BUDGET,
)
from core.cron_runner import CronRunner
from core.logger import SessionLogger
from core.model_router import AppleModelRouter, ModelRoute, resolve_model_mode
from core.prompts import PROFILES, build_prompt_set
from core.tools import cron as cron_tool
from core.tools.web_search import WebSearchTool

load_dotenv()

parser = argparse.ArgumentParser(description="LLM Agent")
interface = parser.add_mutually_exclusive_group()
interface.add_argument("--cli", action="store_true", help="терминальный интерфейс")
interface.add_argument("--telegram", action="store_true", help="Telegram-интерфейс")
model_group = parser.add_mutually_exclusive_group()
model_group.add_argument(
    "--model", choices=("hybrid", "auto", "local", "pcc", "server"),
    help="маршрутизация Apple-моделей (по умолчанию: hybrid)",
)
model_group.add_argument(
    "--local", action="store_true",
    help="только локальная AFM Core 3, без PCC",
)
model_group.add_argument(
    "--server", action="store_true",
    help="только Apple PCC, без локальной модели",
)
parser.add_argument(
    "--prompts",
    choices=PROFILES,
    help="принудительно использовать один профиль промптов для обеих моделей",
)
args = parser.parse_args()

mode = "telegram" if args.telegram else "cli"
try:
    model_mode = resolve_model_mode(
        cli_model=args.model,
        local=args.local,
        server=args.server,
        env_mode=os.environ.get("MODEL_MODE"),
    )
except ValueError:
    print("Ошибка: MODEL_MODE должен быть hybrid, local или pcc")
    sys.exit(1)

prompt_override = args.prompts or os.environ.get("PROMPT_PROFILE")
if prompt_override and prompt_override not in PROFILES:
    available = ", ".join(PROFILES)
    print(f"Ошибка: неизвестный PROMPT_PROFILE '{prompt_override}'. Доступны: {available}")
    sys.exit(1)


def _int_env(primary: str, fallback: str, default: int) -> int:
    raw = os.environ.get(primary, os.environ.get(fallback, str(default)))
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{primary} должен быть больше нуля")
    return value


try:
    LOCAL_TOKEN_BUDGET = _int_env(
        "LOCAL_CONTEXT_TOKEN_BUDGET", "CONTEXT_TOKEN_BUDGET",
        DEFAULT_LOCAL_CONTEXT_TOKEN_BUDGET,
    )
    PCC_TOKEN_BUDGET = _int_env(
        "PCC_CONTEXT_TOKEN_BUDGET", "CONTEXT_TOKEN_BUDGET",
        DEFAULT_PCC_CONTEXT_TOKEN_BUDGET,
    )
    COMPACT_RATIO = float(os.environ.get("COMPACT_TRIGGER_RATIO", "0.8"))
except ValueError as e:
    print(f"Ошибка конфигурации контекста: {e}")
    sys.exit(1)
if not 0.5 <= COMPACT_RATIO < 1:
    print("Ошибка: COMPACT_TRIGGER_RATIO должен быть от 0.5 до 1")
    sys.exit(1)

WEB_SEARCH_FORCE_DEPTH = os.environ.get("WEB_SEARCH_FORCE_DEPTH", "auto").lower()
if WEB_SEARCH_FORCE_DEPTH not in {"auto", "quick", "normal", "deep"}:
    print("Ошибка: WEB_SEARCH_FORCE_DEPTH должен быть auto, quick, normal или deep")
    sys.exit(1)

if mode == "telegram" and not os.environ.get("TELEGRAM_BOT_TOKEN"):
    print("Ошибка: TELEGRAM_BOT_TOKEN не задан в .env")
    sys.exit(1)


def _system_info() -> str:
    return (
        f"OS: {platform.system()} {platform.release()} | "
        f"Python: {platform.python_version()} | "
        f"CWD: {os.getcwd()} | "
        f"Shell: {os.environ.get('SHELL', 'unknown')} | "
        f"DateTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (local)"
    )


TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
ALLOWED_USER_ID = os.environ.get("ALLOWED_USER_ID", "")
local_profile = prompt_override or "mini"
pcc_profile = prompt_override or "full"
local_prompts = build_prompt_set(
    local_profile,
    system_info=_system_info(),
    telegram_token=TELEGRAM_BOT_TOKEN,
    allowed_user_id=ALLOWED_USER_ID,
)
pcc_prompts = build_prompt_set(
    pcc_profile,
    system_info=_system_info(),
    telegram_token=TELEGRAM_BOT_TOKEN,
    allowed_user_id=ALLOWED_USER_ID,
)

COMPACT_KEEP_MESSAGES = 10
MAX_TOOL_OUTPUT = 2000

# Apple bridge не проверяет ключ, но OpenAI SDK требует непустое значение.
client = OpenAI(base_url=APPLE_BASE_URL, api_key="apple-local")
logger = SessionLogger()
logger.info(
    f"mode={mode} | apple={model_mode} | local={APPLE_LOCAL_MODEL}/{local_profile} | "
    f"pcc={APPLE_PCC_MODEL}/{pcc_profile} | "
    f"context={LOCAL_TOKEN_BUDGET}/{PCC_TOKEN_BUDGET} | "
    f"web_search={WEB_SEARCH_FORCE_DEPTH}"
)

# Hybrid: PCC планирует/синтезирует исследование, AFM извлекает страницы.
# Строгие local/server режимы не пересекают выбранную границу.
search_worker_model = APPLE_PCC_MODEL if model_mode == "pcc" else APPLE_LOCAL_MODEL
search_planner_model = APPLE_LOCAL_MODEL if model_mode == "local" else APPLE_PCC_MODEL
web_search = WebSearchTool(
    client,
    search_worker_model,
    model_mini=search_worker_model,
    planner_model=search_planner_model,
    logger=logger,
    force_depth=None if WEB_SEARCH_FORCE_DEPTH == "auto" else WEB_SEARCH_FORCE_DEPTH,
)


class CronToolWrapper:
    SCHEMA = cron_tool.SCHEMA

    def __init__(self):
        self._runner = None

    def execute(self, **kwargs):
        result = cron_tool.execute(**kwargs)
        if kwargs.get("action") == "add" and self._runner:
            self._runner._reload_jobs()
        return result


cron_wrapper = CronToolWrapper()


def _router(*, cron: bool = False) -> AppleModelRouter:
    local_system = local_prompts.cron_agent if cron else local_prompts.agent
    pcc_system = pcc_prompts.cron_agent if cron else pcc_prompts.agent
    local = ModelRoute(
        "local", APPLE_LOCAL_MODEL, local_system, LOCAL_TOKEN_BUDGET,
        fallback_model=APPLE_PCC_MODEL if model_mode == "hybrid" else None,
    )
    pcc = ModelRoute(
        "pcc", APPLE_PCC_MODEL, pcc_system, PCC_TOKEN_BUDGET,
        fallback_model=APPLE_LOCAL_MODEL if model_mode == "hybrid" else None,
    )
    return AppleModelRouter(local, pcc, mode=model_mode)


def _make_agent(agent_logger, *, cron: bool = False, extra_tools=None) -> Agent:
    router = _router(cron=cron)
    initial = router.pcc if model_mode == "pcc" else router.local
    return Agent(
        client,
        initial.model,
        initial.system,
        COMPACT_KEEP_MESSAGES,
        MAX_TOOL_OUTPUT,
        logger=agent_logger,
        extra_tools=extra_tools,
        model_fallback=initial.fallback_model,
        token_budget=initial.token_budget,
        compact_prompt=local_prompts.compact,
        compact_trigger_ratio=COMPACT_RATIO,
        route_selector=router.select,
        compact_model=APPLE_PCC_MODEL if model_mode == "pcc" else APPLE_LOCAL_MODEL,
    )


# Крон-агент не получает cron_manage, чтобы задачи не создавали задачи рекурсивно.
def cron_agent_factory():
    cron_logger = SessionLogger()
    cron_logger.info(f"mode=cron | apple={model_mode}")
    return _make_agent(cron_logger, cron=True, extra_tools=[web_search])


agent = _make_agent(logger, extra_tools=[web_search, cron_wrapper])

if TELEGRAM_BOT_TOKEN and ALLOWED_USER_ID:
    cron_runner = CronRunner(cron_agent_factory, TELEGRAM_BOT_TOKEN, ALLOWED_USER_ID)
    cron_runner.start()
    cron_wrapper._runner = cron_runner

if mode == "telegram":
    from interfaces.telegram import run
    run(agent, TELEGRAM_BOT_TOKEN, ALLOWED_USER_ID, logger=logger)
else:
    from interfaces.cli import run
    run(agent)
