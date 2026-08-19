import time
from unittest import TestCase

from pydantic import BaseModel

from core.cancellation import CancellationToken, CancelledError
from core.policy import Capability, ExecutionPolicy
from core.tools.base import ErrorCode, Tool, ToolContext, ToolRegistry, ToolResult


class EchoInput(BaseModel):
    text: str
    count: int = 1


class EchoTool(Tool):
    name = "echo"
    description = "Эхо для тестов"
    input_model = EchoInput
    capabilities = frozenset({Capability.SHELL_READ})
    timeout = 2.0
    output_limit = 100

    def __init__(self):
        self.calls = 0

    def execute(self, args: EchoInput, ctx: ToolContext) -> ToolResult:
        self.calls += 1
        return ToolResult(content=args.text * args.count, summary="эхо")


class SlowTool(Tool):
    name = "slow"
    description = "Медленный инструмент"
    input_model = EchoInput
    timeout = 0.2

    def execute(self, args: EchoInput, ctx: ToolContext) -> ToolResult:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            ctx.raise_if_cancelled()
            time.sleep(0.02)
        return ToolResult(content="done")


class SendTool(Tool):
    name = "sender"
    description = "Внешняя отправка"
    input_model = EchoInput
    capabilities = frozenset({Capability.EXTERNAL_SEND})

    def execute(self, args: EchoInput, ctx: ToolContext) -> ToolResult:
        return ToolResult(content="sent")


class ToolProtocolTests(TestCase):
    def _registry(self) -> tuple[ToolRegistry, EchoTool]:
        echo = EchoTool()
        return ToolRegistry([echo]), echo

    def test_schema_is_derived_from_pydantic_model(self):
        registry, _ = self._registry()
        schema = registry.schemas()[0]
        self.assertEqual(schema["function"]["name"], "echo")
        properties = schema["function"]["parameters"]["properties"]
        self.assertIn("text", properties)
        self.assertIn("count", properties)
        self.assertEqual(schema["function"]["parameters"]["required"], ["text"])

    def test_duplicate_names_are_rejected(self):
        with self.assertRaises(ValueError):
            ToolRegistry([EchoTool(), EchoTool()])

    def test_unknown_tool_is_not_executed(self):
        registry, echo = self._registry()
        result = registry.execute("ghost", "{}", ToolContext())
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ErrorCode.UNKNOWN_TOOL)
        self.assertEqual(echo.calls, 0)

    def test_invalid_arguments_are_rejected_before_execution(self):
        registry, echo = self._registry()
        result = registry.execute("echo", '{"count": "не число"}', ToolContext())
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ErrorCode.INVALID_ARGUMENTS)
        self.assertEqual(echo.calls, 0)
        self.assertIn("count", result.error)

    def test_malformed_json_is_rejected_before_execution(self):
        registry, echo = self._registry()
        result = registry.execute("echo", '{"text":', ToolContext())
        self.assertEqual(result.error_code, ErrorCode.INVALID_ARGUMENTS)
        self.assertEqual(echo.calls, 0)

    def test_valid_call_executes_with_defaults(self):
        registry, echo = self._registry()
        result = registry.execute("echo", {"text": "аб"}, ToolContext())
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "аб")
        self.assertEqual(echo.calls, 1)

    def test_policy_denies_capability_before_execution(self):
        policy = ExecutionPolicy("/tmp", allow_external_send=False)
        registry = ToolRegistry([SendTool()])
        result = registry.execute("sender", {"text": "x"}, ToolContext(policy=policy))
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ErrorCode.DENIED)

    def test_output_is_capped_to_tool_limit(self):
        registry, _ = self._registry()
        result = registry.execute("echo", {"text": "x", "count": 500}, ToolContext())
        self.assertTrue(result.ok)
        self.assertIn("обрезан", result.content)
        self.assertLess(len(result.content), 200)

    def test_timeout_backstop_returns_timeout_error(self):
        registry = ToolRegistry([SlowTool()])
        started = time.monotonic()
        result = registry.execute("slow", {"text": "x"}, ToolContext())
        elapsed = time.monotonic() - started
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ErrorCode.TIMEOUT)
        self.assertTrue(result.retryable)
        self.assertLess(elapsed, 1.5)

    def test_cancelled_run_interrupts_tool(self):
        token = CancellationToken()
        registry = ToolRegistry([SlowTool()])
        token.cancel("стоп")
        with self.assertRaises(CancelledError):
            registry.execute("slow", {"text": "x"}, ToolContext(cancel=token))

    def test_without_returns_registry_without_tool(self):
        registry = ToolRegistry([EchoTool(), SendTool()])
        reduced = registry.without("sender")
        self.assertEqual(reduced.names(), ["echo"])

    def test_model_text_marks_retryability(self):
        retryable = ToolResult.failure("сеть недоступна", code="timeout", retryable=True)
        final = ToolResult.failure("команда заблокирована", code="denied")
        self.assertIn("попробуй снова", retryable.model_text())
        self.assertIn("бесполезен", final.model_text())

    def test_ok_result_model_text_is_content(self):
        result = ToolResult(content="готово")
        self.assertEqual(result.model_text(), "готово")


if __name__ == "__main__":
    import unittest

    unittest.main()
