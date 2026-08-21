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
from core.tools.notes import NotesTool
from core.tools.notes import execute as notes_execute
from core.tools.read_url import ReadUrlTool
from core.tools.web_search import WebSearchTool, WebSearchToolSpec

parser = argparse.ArgumentParser(description="LLM Agent")
interface = parser.add_mutually_exclusive_group()
interface.add_argument("--cli", action="store_true", help="терминальный интерфейс")
interface.add_argument("--telegram", action="store_true", help="Telegram-интерфейс")
model_group = parser.add_mutually_exclusive_group()
model_group.add_argument(
    "--model",
    choices=("hybrid", "auto", "local", "pcc", "server", "cloud"),
    help="маршрутизация моделей (по умолчанию: hybrid)",
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
model_group.add_argument(
    "--cloud",
    action="store_true",
    help="только облачная модель (CLOUD_BASE_URL/CLOUD_API_KEY/CLOUD_MODEL)",
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
if args.cloud:
    os.environ["MODEL_MODE"] = "cloud"
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
cloud_client = None
if config.model_mode == "cloud":
    cloud_client = OpenAI(
        base_url=config.cloud_base_url,
        api_key=config.cloud_api_key,
        timeout=config.llm_timeout,
    )
llm_client = cloud_client if config.model_mode == "cloud" else client

stt_client = None
if config.stt_api_key and config.stt_base_url:
    stt_client = OpenAI(
        base_url=config.stt_base_url,
        api_key=config.stt_api_key,
        timeout=config.llm_timeout,
    )

logger = SessionLogger(config.log_dir)
logger.add_secret(config.telegram_bot_token)
logger.add_secret(config.allowed_user_id)
logger.add_secret(config.cloud_api_key)
logger.add_secret(config.stt_api_key)
logger.info(
    f"mode={mode} | model_mode={config.model_mode} | "
    f"local={config.local_model}/{config.local_prompt_profile.value} | "
    f"pcc={config.pcc_model}/{config.pcc_prompt_profile.value} | "
    f"cloud={config.cloud_model or '-'}/{config.cloud_prompt_profile.value} | "
    f"context={config.local_context_token_budget}/{config.pcc_context_token_budget}/"
    f"{config.cloud_context_token_budget} | "
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
    work_dir=config.work_dir,
)
pcc_prompts = build_prompt_set(
    config.pcc_prompt_profile.value,
    system_info=_system_info(),
    telegram_token=config.telegram_bot_token,
    allowed_user_id=config.allowed_user_id,
    work_dir=config.work_dir,
)
cloud_prompts = build_prompt_set(
    config.cloud_prompt_profile.value,
    system_info=_system_info(),
    telegram_token=config.telegram_bot_token,
    allowed_user_id=config.allowed_user_id,
    work_dir=config.work_dir,
)

# Hybrid: AFM планирует normal и извлекает страницы, PCC планирует/синтезирует deep.
# Строгие local/server/cloud режимы не пересекают выбранную границу.
if config.model_mode == "cloud":
    search_worker_model = config.cloud_model
    search_planner_model = config.cloud_model
    search_deep_planner = config.cloud_model
else:
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
    llm_client,
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
read_url_tool = ReadUrlTool(web_engine)
notes_tool = NotesTool(notes_file=config.notes_file)
registry = ToolRegistry(
    [bash_tool, WebSearchToolSpec(web_engine), read_url_tool, notes_tool, cron_tool]
)


def _router(*, cron: bool = False, work: bool = False) -> AppleModelRouter:
    def system_for(prompt_set) -> str:
        if cron:
            return prompt_set.cron_agent
        if work:
            return prompt_set.work_agent or prompt_set.agent
        return prompt_set.agent

    local_system = system_for(local_prompts)
    pcc_system = system_for(pcc_prompts)
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
    cloud = None
    if config.model_mode == "cloud":
        cloud_system = system_for(cloud_prompts)
        cloud = ModelRoute(
            "cloud",
            config.cloud_model,
            cloud_system,
            config.cloud_context_token_budget,
        )
    return AppleModelRouter(local, pcc, mode=config.model_mode, cloud=cloud)


def _memory_lookup(query: str) -> str:
    """Авто-вспоминание: заметки по словам запроса, без участия модели."""
    result = notes_execute(action="search", query=query, notes_file=config.notes_file)
    if not result or result.startswith("Ничего не найдено") or result.startswith("Ошибка"):
        return ""
    lines = result.splitlines()
    return "\n".join(lines[-3:])


def _make_agent(agent_logger, *, cron: bool = False, work: bool = False) -> Agent:
    router = _router(cron=cron, work=work)
    if config.model_mode == "cloud":
        initial = router.cloud
    elif config.model_mode == "pcc":
        initial = router.pcc
    else:
        initial = router.local
    # Крон-агент не получает cron_manage, чтобы задачи не создавали задачи рекурсивно.
    tools = registry.without("cron_manage") if cron else registry
    return Agent(
        llm_client,
        initial.model,
        initial.system,
        registry=tools,
        compact_keep_messages=config.compact_keep_messages,
        max_tool_output=config.max_tool_output,
        logger=agent_logger,
        model_fallback=initial.fallback_model,
        token_budget=initial.token_budget,
        compact_prompt=(
            cloud_prompts.compact if config.model_mode == "cloud" else local_prompts.compact
        ),
        compact_trigger_ratio=config.compact_trigger_ratio,
        route_selector=router.select,
        compact_model=(
            config.cloud_model
            if config.model_mode == "cloud"
            else config.pcc_model
            if config.model_mode == "pcc"
            else config.local_model
        ),
        budget_limits=config.work_budget if work else config.budget,
        policy=policy,
        memory_lookup=_memory_lookup,
        work_mode=work,
    )


def cron_agent_factory():
    cron_logger = SessionLogger(config.log_dir)
    cron_logger.add_secret(config.telegram_bot_token)
    cron_logger.add_secret(config.allowed_user_id)
    cron_logger.info(f"mode=cron | apple={config.model_mode}")
    return _make_agent(cron_logger, cron=True)


def work_agent_factory():
    """Свежий агент на каждую work-задачу: свой контекст и увеличенный бюджет."""
    work_logger = SessionLogger(config.log_dir)
    work_logger.add_secret(config.telegram_bot_token)
    work_logger.add_secret(config.allowed_user_id)
    work_logger.add_secret(config.cloud_api_key)
    work_logger.info(f"mode=work | model_mode={config.model_mode}")
    return _make_agent(work_logger, work=True)


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

        run(
            agent,
            config.telegram_bot_token,
            config.allowed_user_id,
            logger=logger,
            stt_client=stt_client,
            stt_model=config.stt_model,
            work_agent_factory=work_agent_factory,
            work_dir=config.work_dir,
        )
    else:
        from interfaces.cli import run

        run(agent)
except KeyboardInterrupt:
    print("\nОстановка…")
finally:
    if cron_runner is not None:
        cron_runner.stop()
