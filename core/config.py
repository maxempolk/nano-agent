from __future__ import annotations

import os
from collections.abc import Mapping
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from core.budget import BudgetLimits
from core.model_router import resolve_model_mode
from core.prompts import PROFILES

DEFAULT_LOCAL_CONTEXT_TOKEN_BUDGET = 3000
DEFAULT_PCC_CONTEXT_TOKEN_BUDGET = 12000


class ConfigError(RuntimeError):
    """Startup configuration failure with every detected problem."""

    def __init__(self, problems: list[str]):
        super().__init__("Проблемы конфигурации:\n- " + "\n- ".join(problems))
        self.problems = problems


class PromptProfileName(str, Enum):
    MINI = "mini"
    FULL = "full"


class ForceDepth(str, Enum):
    AUTO = "auto"
    QUICK = "quick"
    NORMAL = "normal"
    DEEP = "deep"


class BashApproval(str, Enum):
    DENY = "deny"
    PROMPT = "prompt"


class AppConfig(BaseModel):
    # Apple bridge / models
    llm_base_url: str = "http://127.0.0.1:1976/v1"
    local_model: str = "system"
    pcc_model: str = "pcc"
    model_mode: str = "hybrid"

    # Cloud OpenAI-compatible provider (model_mode=cloud)
    cloud_base_url: str = ""
    cloud_api_key: str = ""
    cloud_model: str = ""
    cloud_prompt_profile: PromptProfileName = PromptProfileName.FULL
    cloud_context_token_budget: int = Field(default=12000, ge=500, le=200_000)

    # Speech-to-text for Telegram voice messages (falls back to CLOUD_* provider)
    stt_base_url: str = ""
    stt_api_key: str = ""
    stt_model: str = "whisper-large-v3-turbo"

    # Prompt profiles
    local_prompt_profile: PromptProfileName = PromptProfileName.MINI
    pcc_prompt_profile: PromptProfileName = PromptProfileName.FULL

    # Context budgets
    local_context_token_budget: int = Field(default=3000, ge=500, le=200_000)
    pcc_context_token_budget: int = Field(default=12000, ge=500, le=200_000)
    compact_trigger_ratio: float = Field(default=0.8, ge=0.5, lt=1.0)
    compact_keep_messages: int = Field(default=10, ge=1, le=100)
    max_tool_output: int = Field(default=2000, ge=200, le=100_000)

    # Search
    web_search_force_depth: ForceDepth = ForceDepth.AUTO

    # Telegram
    telegram_bot_token: str = ""
    allowed_user_id: str = ""

    # Bash workspace
    bash_workspace: Path = Path(".")
    bash_timeout: float = Field(default=30.0, gt=0, le=600)
    bash_output_limit: int = Field(default=4000, ge=200, le=200_000)
    bash_approval: BashApproval = BashApproval.DENY
    allow_destructive: bool = False

    # LLM client
    llm_timeout: float = Field(default=120.0, gt=0, le=1800)

    # Scheduler / state
    jobs_file: str = "jobs.json"
    notes_file: str = "notes.json"
    log_dir: str = "logs"

    # Per-request run budget
    budget: BudgetLimits = Field(default_factory=BudgetLimits)


def _env_str(env: Mapping[str, str], key: str, default: str = "") -> str:
    value = env.get(key)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(env: Mapping[str, str], key: str, fallback_key: str, default: int) -> int | str:
    raw = env.get(key) or env.get(fallback_key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return f"{key} должен быть целым числом, получено '{raw}'"


def _env_float(env: Mapping[str, str], key: str, default: float) -> float | str:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return f"{key} должен быть числом, получено '{raw}'"


def _env_bool(env: Mapping[str, str], key: str, default: bool = False) -> bool | str:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return f"{key} должен быть true/false, получено '{raw}'"


def load_config(
    env: Mapping[str, str] | None = None, *, telegram_required: bool = False
) -> AppConfig:
    """Build and validate AppConfig from environment variables.

    Every problem is collected and reported together; nothing half-valid
    reaches runtime.
    """
    if env is None:
        env = dict(os.environ)
    problems: list[str] = []
    values: dict = {}

    values["llm_base_url"] = _env_str(env, "LLM_BASE_URL", "http://127.0.0.1:1976/v1")
    values["local_model"] = _env_str(env, "LOCAL_MODEL", "system")
    values["pcc_model"] = _env_str(env, "PCC_MODEL", "pcc")
    values["cloud_base_url"] = _env_str(env, "CLOUD_BASE_URL")
    values["cloud_api_key"] = _env_str(env, "CLOUD_API_KEY")
    values["cloud_model"] = _env_str(env, "CLOUD_MODEL")
    values["stt_base_url"] = _env_str(env, "STT_BASE_URL") or values["cloud_base_url"]
    values["stt_api_key"] = _env_str(env, "STT_API_KEY") or values["cloud_api_key"]
    values["stt_model"] = _env_str(env, "STT_MODEL", "whisper-large-v3-turbo")

    try:
        values["model_mode"] = resolve_model_mode(env_mode=env.get("MODEL_MODE") or None)
    except ValueError:
        problems.append(
            f"MODEL_MODE должен быть hybrid, local, pcc или cloud, получено '{env.get('MODEL_MODE')}'"
        )
        values["model_mode"] = "hybrid"

    profile_override = _env_str(env, "PROMPT_PROFILE")
    for key, env_name, default in (
        ("local_prompt_profile", "LOCAL_PROMPT_PROFILE", profile_override or "mini"),
        ("pcc_prompt_profile", "PCC_PROMPT_PROFILE", profile_override or "full"),
        ("cloud_prompt_profile", "CLOUD_PROMPT_PROFILE", profile_override or "full"),
    ):
        profile = _env_str(env, env_name, default)
        if profile in PROFILES:
            values[key] = profile
        else:
            problems.append(
                f"Неизвестный профиль промптов '{profile}' ({env_name or 'PROMPT_PROFILE'}). "
                f"Доступны: {', '.join(PROFILES)}"
            )
            values[key] = default if default in PROFILES else "mini"

    for key, env_name, fallback_env, default in (
        (
            "local_context_token_budget",
            "LOCAL_CONTEXT_TOKEN_BUDGET",
            "CONTEXT_TOKEN_BUDGET",
            DEFAULT_LOCAL_CONTEXT_TOKEN_BUDGET,
        ),
        (
            "pcc_context_token_budget",
            "PCC_CONTEXT_TOKEN_BUDGET",
            "CONTEXT_TOKEN_BUDGET",
            DEFAULT_PCC_CONTEXT_TOKEN_BUDGET,
        ),
        (
            "cloud_context_token_budget",
            "CLOUD_CONTEXT_TOKEN_BUDGET",
            "CONTEXT_TOKEN_BUDGET",
            DEFAULT_PCC_CONTEXT_TOKEN_BUDGET,
        ),
        ("compact_keep_messages", "COMPACT_KEEP_MESSAGES", "", 10),
        ("max_tool_output", "MAX_TOOL_OUTPUT", "", 2000),
    ):
        parsed = _env_int(env, env_name, fallback_env, default)
        if isinstance(parsed, str):
            problems.append(parsed)
        else:
            values[key] = parsed

    ratio = _env_float(env, "COMPACT_TRIGGER_RATIO", 0.8)
    if isinstance(ratio, str):
        problems.append(ratio)
    elif not 0.5 <= ratio < 1:
        problems.append(f"COMPACT_TRIGGER_RATIO должен быть от 0.5 до 1, получено {ratio}")
    else:
        values["compact_trigger_ratio"] = ratio

    force_depth = _env_str(env, "WEB_SEARCH_FORCE_DEPTH", "auto").lower()
    if force_depth in {item.value for item in ForceDepth}:
        values["web_search_force_depth"] = force_depth
    else:
        problems.append(
            f"WEB_SEARCH_FORCE_DEPTH должен быть auto, quick, normal или deep, "
            f"получено '{force_depth}'"
        )

    values["telegram_bot_token"] = _env_str(env, "TELEGRAM_BOT_TOKEN")
    values["allowed_user_id"] = _env_str(env, "ALLOWED_USER_ID")
    if telegram_required and not values["telegram_bot_token"]:
        problems.append("TELEGRAM_BOT_TOKEN не задан в .env")
    if values["telegram_bot_token"] and not values["allowed_user_id"]:
        problems.append("ALLOWED_USER_ID обязателен, когда задан TELEGRAM_BOT_TOKEN")

    values["bash_workspace"] = Path(_env_str(env, "BASH_WORKSPACE", ".")).expanduser()

    timeout = _env_float(env, "BASH_TIMEOUT", 30.0)
    if isinstance(timeout, str):
        problems.append(timeout)
    else:
        values["bash_timeout"] = timeout

    output_limit = _env_int(env, "BASH_OUTPUT_LIMIT", "", 4000)
    if isinstance(output_limit, str):
        problems.append(output_limit)
    else:
        values["bash_output_limit"] = output_limit

    approval = _env_str(env, "BASH_APPROVAL", "deny").lower()
    if approval in {item.value for item in BashApproval}:
        values["bash_approval"] = approval
    else:
        problems.append(f"BASH_APPROVAL должен быть deny или prompt, получено '{approval}'")

    allow_destructive = _env_bool(env, "ALLOW_DESTRUCTIVE", False)
    if isinstance(allow_destructive, str):
        problems.append(allow_destructive)
    else:
        values["allow_destructive"] = allow_destructive

    llm_timeout = _env_float(env, "LLM_TIMEOUT", 120.0)
    if isinstance(llm_timeout, str):
        problems.append(llm_timeout)
    else:
        values["llm_timeout"] = llm_timeout

    values["jobs_file"] = _env_str(env, "JOBS_FILE", "jobs.json")
    values["notes_file"] = _env_str(env, "NOTES_FILE", "notes.json")
    values["log_dir"] = _env_str(env, "LOG_DIR", "logs")

    budget_fields: dict = {}
    for key, env_name, default, converter in (
        ("max_steps", "RUN_MAX_STEPS", 8, int),
        ("max_model_calls", "RUN_MAX_MODEL_CALLS", 12, int),
        ("max_tool_calls", "RUN_MAX_TOOL_CALLS", 12, int),
        ("max_wall_seconds", "RUN_MAX_SECONDS", 180.0, float),
        ("max_tool_output_chars", "RUN_MAX_TOOL_OUTPUT_CHARS", 2000, int),
        ("max_consecutive_errors", "RUN_MAX_CONSECUTIVE_ERRORS", 3, int),
        ("max_identical_calls", "RUN_MAX_IDENTICAL_CALLS", 2, int),
    ):
        raw = env.get(env_name)
        if raw is None or raw.strip() == "":
            budget_fields[key] = default
            continue
        try:
            budget_fields[key] = converter(raw)
        except ValueError:
            problems.append(f"{env_name} должен быть числом, получено '{raw}'")

    try:
        budget = BudgetLimits(**budget_fields)
    except ValueError as error:
        problems.append(f"бюджеты выполнения: {error}")
        budget = BudgetLimits()

    if values["model_mode"] == "hybrid" and values["local_model"] == values["pcc_model"]:
        problems.append(
            "LOCAL_MODEL и PCC_MODEL должны различаться в hybrid режиме, "
            "иначе маршрутизация и fallback не имеют смысла"
        )

    if values["model_mode"] == "cloud":
        for key, env_name in (
            ("cloud_base_url", "CLOUD_BASE_URL"),
            ("cloud_api_key", "CLOUD_API_KEY"),
            ("cloud_model", "CLOUD_MODEL"),
        ):
            if not values[key]:
                problems.append(f"{env_name} обязателен в режиме cloud")

    if problems:
        raise ConfigError(problems)

    values["budget"] = budget
    try:
        return AppConfig(**values)
    except ValidationError as error:
        raise ConfigError(
            [
                f"{'.'.join(str(loc) for loc in item['loc']) or 'config'}: {item['msg']}"
                for item in error.errors()
            ]
        ) from error
