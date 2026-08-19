from __future__ import annotations

import json
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from core.cancellation import CancellationToken, CancelledError
from core.policy import Capability, ExecutionPolicy


class ErrorCode:
    INVALID_ARGUMENTS = "invalid_arguments"
    DENIED = "denied"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    TOOL_FAILED = "tool_failed"
    UNKNOWN_TOOL = "unknown_tool"
    BUDGET = "budget_exceeded"


class ToolError(Exception):
    """Structured tool failure with a stable code and retryability flag."""

    def __init__(
        self,
        message: str,
        *,
        code: str = ErrorCode.TOOL_FAILED,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class ToolResult:
    """Structured outcome of one tool execution.

    ``content`` goes to the model, ``summary`` to the user-facing progress;
    errors carry a stable code and retryability instead of free text.
    """

    content: str
    summary: str = ""
    error: str | None = None
    error_code: str | None = None
    retryable: bool = False
    warnings: tuple[str, ...] = ()
    files_created: tuple[str, ...] = ()
    structured: Any = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    def model_text(self) -> str:
        if self.ok:
            return self.content
        text = f"Ошибка инструмента [{self.error_code}]: {self.error}"
        if self.retryable:
            text += " Исправь аргументы или подход и попробуй снова."
        else:
            text += " Повтор идентичного вызова бесполезен."
        for warning in self.warnings:
            text += f"\nПредупреждение: {warning}"
        return text

    @staticmethod
    def failure(
        message: str,
        *,
        code: str = ErrorCode.TOOL_FAILED,
        retryable: bool = False,
        warnings: tuple[str, ...] = (),
    ) -> ToolResult:
        return ToolResult(
            content="",
            error=message,
            error_code=code,
            retryable=retryable,
            warnings=warnings,
        )


@dataclass(frozen=True)
class ToolContext:
    """Run-scoped state handed to every tool: cancellation, policy, logger."""

    cancel: CancellationToken | None = None
    policy: ExecutionPolicy | None = None
    logger: Any = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def raise_if_cancelled(self) -> None:
        if self.cancel is not None:
            self.cancel.raise_if_cancelled()


def flat_json_schema(model: type[BaseModel]) -> dict:
    """Inline all $defs so the schema is a single self-contained object."""
    schema = model.model_json_schema()
    definitions = schema.get("$defs", {})

    def inline(value):
        if isinstance(value, list):
            return [inline(item) for item in value]
        if not isinstance(value, dict):
            return value
        ref = value.get("$ref", "")
        prefix = "#/$defs/"
        if ref.startswith(prefix):
            resolved = definitions.get(ref[len(prefix) :], {})
            return inline(
                {**resolved, **{key: item for key, item in value.items() if key != "$ref"}}
            )
        return {key: inline(value) for key, value in value.items() if key != "$defs"}

    return inline(schema)


class Tool:
    """Single typed contract for every agent tool.

    Subclasses declare name, description, a Pydantic ``input_model``,
    required ``capabilities``, ``timeout`` and ``output_limit``, and
    implement ``execute`` returning a structured ``ToolResult``.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    input_model: ClassVar[type[BaseModel]]
    capabilities: ClassVar[frozenset[Capability]] = frozenset()
    timeout: ClassVar[float] = 30.0
    output_limit: ClassVar[int] = 4000

    def openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": flat_json_schema(self.input_model),
            },
        }

    def parse_args(self, raw: str | dict | None) -> BaseModel:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw or "{}")
            except json.JSONDecodeError as error:
                raise ToolError(
                    f"аргументы не являются JSON-объектом: {error}",
                    code=ErrorCode.INVALID_ARGUMENTS,
                ) from error
        if not isinstance(raw, dict):
            raise ToolError(
                "аргументы должны быть JSON-объектом", code=ErrorCode.INVALID_ARGUMENTS
            )
        try:
            return self.input_model.model_validate(raw)
        except ValidationError as error:
            details = "; ".join(
                f"{'.'.join(str(loc) for loc in item['loc']) or 'input'}: {item['msg']}"
                for item in error.errors()[:4]
            )
            raise ToolError(
                f"невалидные аргументы: {details}", code=ErrorCode.INVALID_ARGUMENTS
            ) from error

    def execute(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        raise NotImplementedError

    def run(self, raw_args: str | dict | None, ctx: ToolContext) -> ToolResult:
        ctx.raise_if_cancelled()
        try:
            args = self.parse_args(raw_args)
            result = self.execute(args, ctx)
        except CancelledError:
            raise
        except ToolError as error:
            return ToolResult.failure(str(error), code=error.code, retryable=error.retryable)
        except Exception as error:  # noqa: BLE001 - tool must not crash the loop
            return ToolResult.failure(f"непредвиденная ошибка: {error}")
        if not isinstance(result, ToolResult):
            raise TypeError(f"{self.name}.execute должен вернуть ToolResult")
        return self._cap_output(result)

    def _cap_output(self, result: ToolResult) -> ToolResult:
        if len(result.content) <= self.output_limit:
            return result
        half = self.output_limit // 2
        clipped = (
            result.content[:half]
            + f"\n... [вывод обрезан на {len(result.content) - self.output_limit} символов] ...\n"
            + result.content[-half:]
        )
        return ToolResult(
            content=clipped,
            summary=result.summary,
            error=result.error,
            error_code=result.error_code,
            retryable=result.retryable,
            warnings=result.warnings,
            files_created=result.files_created,
            structured=result.structured,
            meta=result.meta,
        )


class ToolRegistry:
    """The only place the agent gets tools from.

    ``execute`` validates arguments and policy BEFORE running the tool and
    enforces the tool timeout via the cancellation token.
    """

    def __init__(self, tools: Iterable[Tool]):
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if not tool.name:
                raise ValueError("инструмент без имени нельзя зарегистрировать")
            if tool.name in self._tools:
                raise ValueError(f"дубликат инструмента: {tool.name}")
            self._tools[tool.name] = tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        return [tool.openai_schema() for tool in self._tools.values()]

    def without(self, *excluded: str) -> ToolRegistry:
        return ToolRegistry(
            tool for name, tool in self._tools.items() if name not in set(excluded)
        )

    def execute(self, name: str, raw_args: str | dict | None, ctx: ToolContext) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.failure(
                f"инструмент '{name}' не зарегистрирован", code=ErrorCode.UNKNOWN_TOOL
            )

        if ctx.policy is not None:
            decision = ctx.policy.check_capabilities(tool.capabilities)
            if not decision.allowed:
                return ToolResult.failure(
                    f"политика запретила '{name}': {decision.reason}", code=ErrorCode.DENIED
                )

        holder: dict[str, ToolResult | BaseException] = {}

        def worker() -> None:
            try:
                holder["result"] = tool.run(raw_args, ctx)
            except BaseException as error:  # noqa: BLE001 - re-raised below
                holder["error"] = error

        thread = threading.Thread(target=worker, name=f"tool-{name}", daemon=True)
        thread.start()
        thread.join(tool.timeout)
        if thread.is_alive():
            # A hung tool must not poison the whole run token: tools enforce
            # their own deadlines (subprocess kill, page timeout, budget).
            if ctx.cancel is not None and ctx.cancel.cancelled:
                return ToolResult.failure(
                    f"инструмент '{name}' остановлен отменой", code=ErrorCode.CANCELLED
                )
            return ToolResult.failure(
                f"инструмент '{name}' превысил таймаут {tool.timeout:.0f} с",
                code=ErrorCode.TIMEOUT,
                retryable=True,
            )
        if "error" in holder:
            error = holder["error"]
            if isinstance(error, CancelledError):
                raise error
            raise RuntimeError(f"инструмент '{name}' завершился ошибкой: {error}") from error
        return holder["result"]  # type: ignore[return-value]
