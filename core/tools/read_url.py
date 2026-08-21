from __future__ import annotations

from typing import ClassVar
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from core.policy import Capability
from core.tools.base import ErrorCode, Tool, ToolContext, ToolResult

EXTRACT_FAILED = "Не удалось извлечь текст."


class ReadUrlInput(BaseModel):
    url: str = Field(min_length=8, max_length=500)


class ReadUrlTool(Tool):
    """Читает конкретную страницу или PDF через скрейпер поискового движка."""

    name: ClassVar[str] = "read_url"
    description: ClassVar[str] = (
        "Read a specific web page or PDF by its URL and return the text content. "
        "Use it when the user gives a ready link and asks to read, summarize or analyze it. "
        "For searching the web without a ready link use web_search instead."
    )
    input_model: ClassVar[type[BaseModel]] = ReadUrlInput
    capabilities: ClassVar[frozenset[Capability]] = frozenset({Capability.NETWORK_READ})
    timeout: ClassVar[float] = 45.0
    output_limit: ClassVar[int] = 6000

    def __init__(self, engine):
        self.engine = engine

    def execute(self, args: ReadUrlInput, ctx: ToolContext) -> ToolResult:
        ctx.raise_if_cancelled()
        url = args.url.strip()
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return ToolResult.failure(
                f"URL должен быть абсолютным http(s), получено: '{url}'",
                code=ErrorCode.INVALID_ARGUMENTS,
            )

        text = self.engine.read_page(url)
        if text == EXTRACT_FAILED:
            return ToolResult.failure(
                "Не удалось извлечь текст страницы: она недоступна, закрыта "
                "защитой от ботов, требует входа или не содержит текста. "
                "Сообщи пользователю причину и спроси, искать ли информацию "
                "веб-поиском вместо чтения страницы; не начинай поиск сам.",
                code=ErrorCode.TOOL_FAILED,
                retryable=False,
            )
        return ToolResult(
            content=text,
            summary=f"прочитал страницу ({len(text)} симв.)",
            meta={"url": url},
        )
