import argparse
import os
import platform
import sys
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from core.agent import Agent
from core.config import AppConfig, BashApproval, ConfigError, load_config
from core.cron_runner import CronRunner
from core.logger import SessionLogger
from core.model_router import AppleModelRouter, ModelRoute
from core.policy import ExecutionPolicy
from core.prompts import build_prompt_set
from core.tools.base import ToolRegistry
from core.tools.bash import BashTool
from core.tools.cron import CronManageTool
from core.tools.web_search import WebSearchTool, WebSearchToolSpec

parser = argparse.ArgumentParser(description="LLM Agent")
interface = parser.add_mutually_exclusive_group()
interface.add_argument("--cli", action="store_true", help="терминальный интерфейс")
interface.add_argument("--telegram", action="store_true", help="Telegram-интерфейс")
model_group = parser.add_mutually_exclusive_group()
model_group.add_argument(
    "--model",
    choices=("hybrid", "auto", "local", "pcc", "server"),
    help="маршрутизация Apple-моделей (по умолчанию: hybrid)",
)
model_group.add_argument(
    "--local",
    action="store_true",
    help="только локальная AFM Core 3, без PCC",
)
model_group.add_argument(
    "--server",
    action="store_true",
    help="только Apple PCC, без локальной модели",
)
parser.add_argument(
    "--prompts",
    choices=("mini", "full"),
    help="принудительно использовать один профиль промптов для обеих моделей",
)
args = parser.parse_args()

mode = "telegram" if args.telegram else "cli"

# CLI flags override env before validation so errors are reported once.
if args.model:
    os.environ["MODEL_MODE"] = {"auto": "hybrid", "server": "pcc"}.get(args.model, args.model)
if args.local:
    os.environ["MODEL_MODE"] = "local"
if args.server:
    os.environ["MODEL_MODE"] = "pcc"
if args.prompts:
    os.environ["PROMPT_PROFILE"] = args.prompts

try:
    config: AppConfig = load_config(telegram_required=(mode == "telegram"))
except ConfigError as error:
    print(error)
    sys.exit(1)


def _system_info() -> str:
    return (
        f"OS: {platform.system()} {platform.release()} | "
        f"Python: {platform.python_version()} | "
        f"CWD: {os.getcwd()} | "
        f"Shell: {os.environ.get('SHELL', 'unknown')} | "
        f"DateTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (local)"
    )


def _cli_approval(description: str) -> bool:
    print(f"\n⚠️ Требуется подтверждение.\n{description}")
    try:
        answer = input("Выполнить? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in {"y", "yes", "да"}


# Apple bridge не проверяет ключ, но OpenAI SDK требует непустое значение.
client = OpenAI(
    base_url=config.llm_base_url, api_key="apple-local", timeout=config.llm_timeout
)
logger = SessionLogger(config.log_dir)
logger.info(
    f"mode={mode} | apple={config.model_mode} | "
    f"local={config.local_model}/{config.local_prompt_profile.value} | "
    f"pcc={config.pcc_model}/{config.pcc_prompt_profile.value} | "
    f"context={config.local_context_token_budget}/{config.pcc_context_token_budget} | "
    f"web_search={config.web_search_force_depth.value} | "
    f"workspace={config.bash_workspace}"
)

approval = (
    _cli_approval
    if mode == "cli" and config.bash_approval == BashApproval.PROMPT
    else None
)
policy = ExecutionPolicy(
    config.bash_workspace,
    allow_external_send=bool(config.telegram_bot_token),
    external_send_hosts=("api.telegram.org",) if config.telegram_bot_token else (),
    allow_destructive=config.allow_destructive,
    approval=approval,
)

local_prompts = build_prompt_set(
    config.local_prompt_profile.value,
    system_info=_system_info(),
    telegram_token=config.telegram_bot_token,
    allowed_user_id=config.allowed_user_id,
)
pcc_prompts = build_prompt_set(
    config.pcc_prompt_profile.value,
    system_info=_system_info(),
    telegram_token=config.telegram_bot_token,
    allowed_user_id=config.allowed_user_id,
)

# Hybrid: AFM планирует normal и извлекает страницы, PCC планирует/синтезирует deep.
# Строгие local/server режимы не пересекают выбранную границу.
search_worker_model = (
    config.pcc_model if config.model_mode == "pcc" else config.local_model
)
search_planner_model = (
    config.pcc_model if config.model_mode == "pcc" else config.local_model
)
search_deep_planner = (
    config.local_model if config.model_mode == "local" else config.pcc_model
)
web_engine = WebSearchTool(
    client,
    search_worker_model,
    model_mini=search_worker_model,
    planner_model=search_planner_model,
    deep_planner_model=search_deep_planner,
    logger=logger,
    force_depth=(
        None
        if config.web_search_force_depth.value == "auto"
        else config.web_search_force_depth.value
    ),
)

bash_tool = BashTool(
    config.bash_workspace,
    timeout=config.bash_timeout,
    output_limit=config.bash_output_limit,
)
cron_tool = CronManageTool(jobs_file=config.jobs_file)
registry = ToolRegistry([bash_tool, WebSearchToolSpec(web_engine), cron_tool])


def _router(*, cron: bool = False) -> AppleModelRouter:
    local_system = local_prompts.cron_agent if cron else local_prompts.agent
    pcc_system = pcc_prompts.cron_agent if cron else pcc_prompts.agent
    local = ModelRoute(
        "local",
        config.local_model,
        local_system,
        config.local_context_token_budget,
        fallback_model=config.pcc_model if config.model_mode == "hybrid" else None,
    )
    pcc = ModelRoute(
        "pcc",
        config.pcc_model,
        pcc_system,
        config.pcc_context_token_budget,
        fallback_model=config.local_model if config.model_mode == "hybrid" else None,
    )
    return AppleModelRouter(local, pcc, mode=config.model_mode)


def _make_agent(agent_logger, *, cron: bool = False) -> Agent:
    router = _router(cron=cron)
    initial = router.pcc if config.model_mode == "pcc" else router.local
    # Крон-агент не получает cron_manage, чтобы задачи не создавали задачи рекурсивно.
    tools = registry.without("cron_manage") if cron else registry
    return Agent(
        client,
        initial.model,
        initial.system,
        registry=tools,
        compact_keep_messages=config.compact_keep_messages,
        max_tool_output=config.max_tool_output,
        logger=agent_logger,
        model_fallback=initial.fallback_model,
        token_budget=initial.token_budget,
        compact_prompt=local_prompts.compact,
        compact_trigger_ratio=config.compact_trigger_ratio,
        route_selector=router.select,
        compact_model=(
            config.pcc_model if config.model_mode == "pcc" else config.local_model
        ),
        budget_limits=config.budget,
        policy=policy,
    )


def cron_agent_factory():
    cron_logger = SessionLogger(config.log_dir)
    cron_logger.info(f"mode=cron | apple={config.model_mode}")
    return _make_agent(cron_logger, cron=True)


agent = _make_agent(logger)

cron_runner: CronRunner | None = None
if config.telegram_bot_token and config.allowed_user_id:
    cron_runner = CronRunner(
        cron_agent_factory,
        config.telegram_bot_token,
        config.allowed_user_id,
        jobs_file=config.jobs_file,
    )
    cron_runner.start()
    cron_tool.on_change = cron_runner._reload_jobs

try:
    if mode == "telegram":
        from interfaces.telegram import run

        run(agent, config.telegram_bot_token, config.allowed_user_id, logger=logger)
    else:
        from interfaces.cli import run

        run(agent)
except KeyboardInterrupt:
    print("\nОстановка…")
finally:
    if cron_runner is not None:
        cron_runner.stop()
