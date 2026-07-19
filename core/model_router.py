from __future__ import annotations

from dataclasses import dataclass, replace
import re


def resolve_model_mode(*, cli_model: str | None = None, local: bool = False,
                       server: bool = False, env_mode: str | None = None) -> str:
    requested = "local" if local else "server" if server else (
        cli_model or env_mode or "hybrid"
    )
    mode = {"auto": "hybrid", "server": "pcc"}.get(requested, requested)
    if mode not in {"hybrid", "local", "pcc"}:
        raise ValueError(f"Неизвестный режим модели: {requested}")
    return mode


@dataclass(frozen=True)
class ModelRoute:
    name: str
    model: str
    system: str
    token_budget: int
    fallback_model: str | None = None


@dataclass(frozen=True)
class RouteDecision:
    route: ModelRoute
    reason: str
    score: int


class AppleModelRouter:
    """Routes simple turns on-device and complex work to Apple PCC."""

    _COMPLEX = re.compile(
        r"\b(реализ\w*|разработ\w*|рефактор\w*|отлад\w*|дебаг\w*|"
        r"проанализ\w*|исслед\w*|архитектур\w*|оптимиз\w*|миграц\w*|"
        r"исправ\w*|логи?|тест\w*|код\w*|команд\w*|функц\w*|фич\w*|"
        r"проект\w*|файл\w*|приложен\w*|сравни\w*|спроектир\w*|"
        r"implement\w*|develop\w*|refactor\w*|debug\w*|analy[sz]\w*|"
        r"research\w*|architect\w*|optimi[sz]\w*|migrat\w*|fix\w*|"
        r"logs?|tests?|code|design\w*)\b",
        re.IGNORECASE,
    )
    _MULTISTEP = re.compile(
        r"(^|\n)\s*(?:\d+[.)]|[-*])\s+|\b(сначала|затем|после этого|"
        r"несколько шагов|step by step|first.+then)\b",
        re.IGNORECASE,
    )
    _FOLLOW_UP = re.compile(
        r"^\s*(да|ок(?:ей)?|хорошо|продолжай|делай|сделай|попробуй|"
        r"исправь это|continue|go ahead|do it|try again)[.!\s]*$",
        re.IGNORECASE,
    )

    def __init__(self, local: ModelRoute, pcc: ModelRoute, mode: str = "hybrid"):
        mode = resolve_model_mode(cli_model=mode)
        self.local = replace(local, fallback_model=None) if mode == "local" else local
        self.pcc = replace(pcc, fallback_model=None) if mode == "pcc" else pcc
        self.mode = mode
        self._last_auto_route = self.local

    def select(self, user_input: str) -> RouteDecision:
        if self.mode == "local":
            return RouteDecision(self.local, "forced local mode", 0)
        if self.mode == "pcc":
            return RouteDecision(self.pcc, "forced PCC mode", 0)

        score = 0
        reasons: list[str] = []
        length = len(user_input)
        if length >= 1200:
            score += 3
            reasons.append("large request")
        elif length >= 450:
            score += 2
            reasons.append("long request")
        if "```" in user_input:
            score += 2
            reasons.append("code block")
        if self._COMPLEX.search(user_input):
            score += 2
            reasons.append("complex task")
        if self._MULTISTEP.search(user_input):
            score += 2
            reasons.append("multiple steps")

        if score >= 2:
            decision = RouteDecision(self.pcc, ", ".join(reasons), score)
        elif self._last_auto_route == self.pcc and self._FOLLOW_UP.match(user_input):
            decision = RouteDecision(self.pcc, "follow-up to complex task", score)
        else:
            decision = RouteDecision(self.local, "simple request", score)
        self._last_auto_route = decision.route
        return decision
