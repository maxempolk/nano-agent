import argparse
import os
import sys
import platform
from dotenv import load_dotenv
from openai import OpenAI

from core.agent import Agent
from core.logger import SessionLogger
from core.tools.web_search import WebSearchTool
from core.tools import cron as cron_tool
from core.cron_runner import CronRunner
from core.prompts import PROFILES, build_prompt_set
from core.config import (
    MODEL, MODEL_MINI, MODEL_FALLBACK,
    LOCAL_BASE_URL, LOCAL_MODEL, LOCAL_MODEL_MINI, LOCAL_MODEL_FALLBACK,
)

load_dotenv()

parser = argparse.ArgumentParser(description="LLM Agent")
interface = parser.add_mutually_exclusive_group()
interface.add_argument("--cli", action="store_true", help="терминальный интерфейс")
interface.add_argument("--telegram", action="store_true", help="Telegram-интерфейс")
parser.add_argument("--local", action="store_true", help="локальная Apple Foundation Model")
parser.add_argument(
    "--prompts",
    choices=PROFILES,
    help="профиль промптов (по умолчанию: mini для --local, full для API)",
)
args = parser.parse_args()

mode = "telegram" if args.telegram else "cli"
use_local = args.local
prompt_profile = args.prompts or os.environ.get("PROMPT_PROFILE")
if not prompt_profile:
    prompt_profile = "mini" if use_local else "full"
if prompt_profile not in PROFILES:
    available = ", ".join(PROFILES)
    print(f"Ошибка: неизвестный PROMPT_PROFILE '{prompt_profile}'. Доступны: {available}")
    sys.exit(1)

if use_local:
    BASE_URL = LOCAL_BASE_URL
    API_KEY = "ollama"
    TOKEN_BUDGET = 3000
    MODEL = LOCAL_MODEL
    MODEL_MINI = LOCAL_MODEL_MINI
    MODEL_FALLBACK = LOCAL_MODEL_FALLBACK
else:
    BASE_URL = "https://api.groq.com/openai/v1"
    TOKEN_BUDGET = 5500
    if not os.environ.get("API_TOKEN"):
        print("Ошибка: API_TOKEN не задан в .env")
        sys.exit(1)
    API_KEY = os.environ["API_TOKEN"]

if mode == "telegram" and not os.environ.get("TELEGRAM_BOT_TOKEN"):
    print("Ошибка: TELEGRAM_BOT_TOKEN не задан в .env")
    sys.exit(1)


def _system_info() -> str:
    from datetime import datetime
    return (
        f"OS: {platform.system()} {platform.release()} | "
        f"Python: {platform.python_version()} | "
        f"CWD: {os.getcwd()} | "
        f"User: {os.environ.get('USER', 'unknown')} | "
        f"Shell: {os.environ.get('SHELL', 'unknown')} | "
        f"DateTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (local)"
    )

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
ALLOWED_USER_ID = os.environ.get("ALLOWED_USER_ID", "")

prompts = build_prompt_set(
    prompt_profile,
    system_info=_system_info(),
    telegram_token=TELEGRAM_BOT_TOKEN,
    allowed_user_id=ALLOWED_USER_ID,
)

CONTEXT_WINDOW = 10
MAX_TOOL_OUTPUT = 2000

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

logger = SessionLogger()
logger.info(
    f"mode={mode} | {'local' if use_local else 'api'} | "
    f"model={MODEL} | prompts={prompt_profile}"
)

web_search = WebSearchTool(client, MODEL, model_mini=MODEL_MINI)


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


# Фабрика для крон-задач — без cron_manage, чтобы задачи не могли создавать задачи рекурсивно
def cron_agent_factory():
    cron_logger = SessionLogger()
    cron_logger.info(f"mode=cron | model={MODEL} | prompts={prompt_profile}")
    return Agent(client, MODEL, prompts.cron_agent, CONTEXT_WINDOW, MAX_TOOL_OUTPUT,
                 logger=cron_logger, extra_tools=[web_search], model_fallback=MODEL_FALLBACK,
                 token_budget=TOKEN_BUDGET)

agent = Agent(client, MODEL, prompts.agent, CONTEXT_WINDOW, MAX_TOOL_OUTPUT,
              logger=logger, extra_tools=[web_search, cron_wrapper], model_fallback=MODEL_FALLBACK,
              token_budget=TOKEN_BUDGET)

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
