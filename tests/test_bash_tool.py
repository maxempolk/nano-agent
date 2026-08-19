import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from core.cancellation import CancellationToken, CancelledError
from core.policy import ExecutionPolicy
from core.tools.base import ErrorCode, ToolContext
from core.tools.bash import (
    BashTool,
    CommandKind,
    classify_command,
    extract_paths,
    filtered_env,
)


def _ctx(workspace: Path, **policy_kwargs) -> ToolContext:
    policy = ExecutionPolicy(workspace, **policy_kwargs)
    return ToolContext(policy=policy)


class ClassifyCommandTests(TestCase):
    def test_read_only_commands(self):
        for command in [
            "ls -la",
            "cat file.txt",
            "git status",
            "git log --oneline",
            "grep -r pattern .",
            "echo hello",
            "pwd",
            "curl -s https://example.com",
        ]:
            kind, _ = classify_command(command)
            self.assertEqual(kind, CommandKind.READ_ONLY, command)

    def test_mutating_commands(self):
        for command in [
            "touch a.txt",
            "mv a.txt b.txt",
            "echo hi > out.txt",
            "git commit -m 'msg'",
            "pip install requests",
            "mkdir -p build",
            "sed -i s/a/b/ file.txt",
            "curl -o file.zip https://example.com/file.zip",
            "some_unknown_binary --flag",
        ]:
            kind, _ = classify_command(command)
            self.assertEqual(kind, CommandKind.MUTATING, command)

    def test_destructive_commands(self):
        for command in [
            "rm -rf build",
            "rm file.txt",
            "git push --force origin main",
            "git reset --hard HEAD~1",
            "git clean -fd",
            "kill -9 1234",
            "chmod 600 file",
        ]:
            kind, _ = classify_command(command)
            self.assertEqual(kind, CommandKind.DESTRUCTIVE, command)

    def test_blocked_commands(self):
        for command in [
            "sudo ls",
            "mkfs.ext4 /dev/disk2",
            "dd if=/dev/zero of=/dev/disk0",
            "shutdown -h now",
            "sleep 100 &",
            "nohup server &",
            "rm -rf /",
            "rm -rf ~",
        ]:
            kind, reasons = classify_command(command)
            self.assertEqual(kind, CommandKind.BLOCKED, command)
            self.assertTrue(reasons)

    def test_ampersand_inside_redirect_is_not_background(self):
        kind, _ = classify_command("ls > out.txt 2>&1")
        self.assertEqual(kind, CommandKind.MUTATING)
        kind, _ = classify_command("cat a && cat b")
        self.assertEqual(kind, CommandKind.READ_ONLY)

    def test_interactive_commands(self):
        for command in [
            "vim file.txt",
            "top",
            "ssh user@host",
            "python3",
            "node",
            "bash",
            "git commit",
        ]:
            kind, _ = classify_command(command)
            self.assertEqual(kind, CommandKind.INTERACTIVE, command)

    def test_non_interactive_variants_are_allowed(self):
        self.assertEqual(classify_command("python3 -c 'print(1)'")[0], CommandKind.MUTATING)
        self.assertEqual(classify_command("bash script.sh")[0], CommandKind.MUTATING)
        self.assertEqual(classify_command("git commit -m 'x'")[0], CommandKind.MUTATING)
        self.assertEqual(classify_command("python3 script.py")[0], CommandKind.MUTATING)

    def test_external_send_detection(self):
        kind, _ = classify_command("curl -X POST https://api.example.com -d 'x=1'")
        self.assertEqual(kind, CommandKind.MUTATING)

    def test_extract_paths_finds_absolute_and_home_paths(self):
        paths = extract_paths("cp ~/a.txt /etc/passwd && cat relative.txt")
        self.assertIn("~/a.txt", paths)
        self.assertIn("/etc/passwd", paths)
        self.assertNotIn("relative.txt", paths)


class FilteredEnvTests(TestCase):
    def test_secret_keys_are_removed(self):
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "secret",
                "MY_API_KEY": "secret",
                "DATABASE_PASSWORD": "secret",
                "PATH": "/usr/bin",
                "HOME": "/Users/x",
            },
        ):
            env = filtered_env()
        self.assertNotIn("TELEGRAM_BOT_TOKEN", env)
        self.assertNotIn("MY_API_KEY", env)
        self.assertNotIn("DATABASE_PASSWORD", env)
        self.assertIn("PATH", env)
        self.assertIn("HOME", env)


class BashToolTests(TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="nano-bash-ws-")
        self.workspace = Path(self._tmp.name).resolve()
        self.tool = BashTool(self.workspace, timeout=10)
        self.addCleanup(self._tmp.cleanup)

    def _run(self, command: str, ctx: ToolContext | None = None):
        return self.tool.run({"command": command}, ctx or _ctx(self.workspace))

    def test_successful_read_only_command(self):
        result = self._run("echo hello")
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "hello")

    def test_command_runs_inside_workspace(self):
        result = self._run("pwd")
        self.assertEqual(result.content, str(self.workspace))

    def test_relative_mutation_stays_in_workspace(self):
        result = self._run("touch marker.txt")
        self.assertTrue(result.ok, result.error)
        self.assertTrue((self.workspace / "marker.txt").exists())

    def test_write_outside_workspace_is_denied(self):
        result = self._run("touch /etc/nano_agent_test_file")
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ErrorCode.DENIED)
        self.assertFalse(Path("/etc/nano_agent_test_file").exists())

    def test_read_outside_workspace_is_allowed(self):
        result = self._run("ls /etc")
        self.assertTrue(result.ok)

    def test_scratch_paths_are_allowed_for_writes(self):
        result = self._run("touch /tmp/nano_agent_scratch_test && rm /tmp/nano_agent_scratch_test")
        # rm makes it destructive: allow_destructive off → denied, so test write only
        self.assertEqual(result.error_code, ErrorCode.DENIED)
        result = self._run("touch /tmp/nano_agent_scratch_test")
        self.assertTrue(result.ok, result.error)
        Path("/tmp/nano_agent_scratch_test").unlink(missing_ok=True)

    def test_failed_command_returns_structured_error(self):
        result = self._run("ls /nonexistent_dir_xyz")
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "command_failed")
        self.assertIn("код", result.error)

    def test_sudo_is_blocked(self):
        result = self._run("sudo ls")
        self.assertEqual(result.error_code, ErrorCode.DENIED)
        self.assertIn("заблокирована", result.error)

    def test_interactive_is_denied(self):
        result = self._run("vim notes.txt")
        self.assertEqual(result.error_code, ErrorCode.DENIED)
        self.assertIn("интерактив", result.error)

    def test_background_processes_are_blocked(self):
        result = self._run("sleep 5 &")
        self.assertEqual(result.error_code, ErrorCode.DENIED)

    def test_destructive_requires_policy_and_approval(self):
        (self.workspace / "victim.txt").write_text("x")
        denied = self._run("rm victim.txt")
        self.assertEqual(denied.error_code, ErrorCode.DENIED)
        self.assertTrue((self.workspace / "victim.txt").exists())

        approved_ctx = ToolContext(
            policy=ExecutionPolicy(self.workspace, allow_destructive=True, approval=lambda _: True)
        )
        approved = self._run("rm victim.txt", approved_ctx)
        self.assertTrue(approved.ok, approved.error)
        self.assertFalse((self.workspace / "victim.txt").exists())

    def test_destructive_without_approval_is_denied(self):
        (self.workspace / "victim.txt").write_text("x")
        ctx = ToolContext(
            policy=ExecutionPolicy(self.workspace, allow_destructive=True, approval=lambda _: False)
        )
        result = self._run("rm victim.txt", ctx)
        self.assertEqual(result.error_code, ErrorCode.DENIED)
        self.assertTrue((self.workspace / "victim.txt").exists())

    def test_destructive_outside_workspace_is_never_approved(self):
        ctx = ToolContext(
            policy=ExecutionPolicy(self.workspace, allow_destructive=True, approval=lambda _: True)
        )
        result = self._run("rm /etc/nano_agent_test_file", ctx)
        self.assertEqual(result.error_code, ErrorCode.DENIED)

    def test_external_send_is_denied_by_default(self):
        result = self._run("curl -X POST https://example.com -d 'x=1'")
        self.assertEqual(result.error_code, ErrorCode.DENIED)
        self.assertIn("внешн", result.error)

    def test_external_send_is_host_restricted(self):
        ctx = _ctx(
            self.workspace,
            allow_external_send=True,
            external_send_hosts=("api.telegram.org",),
        )
        result = self._run("curl -X POST https://example.com -d 'x=1'", ctx)
        self.assertEqual(result.error_code, ErrorCode.DENIED)

    def test_env_secrets_are_hidden_from_commands(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "super-secret-value"}):
            result = self._run("env | grep -c super-secret-value || true")
        self.assertTrue(result.ok)
        self.assertEqual(result.content.strip(), "0")

    def test_timeout_kills_the_command(self):
        tool = BashTool(self.workspace, timeout=0.5)
        started = time.monotonic()
        result = tool.run({"command": "sleep 5"}, _ctx(self.workspace))
        elapsed = time.monotonic() - started
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ErrorCode.TIMEOUT)
        self.assertLess(elapsed, 3)

    def test_cancellation_kills_subprocess(self):
        token = CancellationToken()
        ctx = ToolContext(cancel=token, policy=ExecutionPolicy(self.workspace))
        tool = BashTool(self.workspace, timeout=30)
        marker = self.workspace / "started.txt"

        def target():
            tool.run(
                {"command": f"touch {marker}; sleep 30"},
                ctx,
            )

        thread = threading.Thread(target=target)
        thread.start()
        for _ in range(50):
            if marker.exists():
                break
            time.sleep(0.05)
        self.assertTrue(marker.exists(), "команда не стартовала")

        token.cancel("отмена пользователем")
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive(), "выполнение не остановилось после отмены")

        probe = subprocess.run(
            ["pgrep", "-f", "sleep 30"], capture_output=True, text=True
        )
        self.assertNotEqual(probe.returncode, 0, "дочерний sleep пережил отмену")

    def test_cancellation_raises_cancelled_error(self):
        token = CancellationToken()
        token.cancel()
        ctx = ToolContext(cancel=token, policy=ExecutionPolicy(self.workspace))
        with self.assertRaises(CancelledError):
            self.tool.run({"command": "sleep 2"}, ctx)

    def test_empty_command_fails_validation_without_execution(self):
        result = self.tool.run({"command": ""}, _ctx(self.workspace))
        self.assertEqual(result.error_code, ErrorCode.INVALID_ARGUMENTS)

    def test_missing_command_fails_validation(self):
        result = self.tool.run({}, _ctx(self.workspace))
        self.assertEqual(result.error_code, ErrorCode.INVALID_ARGUMENTS)

    def test_output_is_capped(self):
        tool = BashTool(self.workspace, timeout=10, output_limit=300)
        result = tool.run({"command": "yes nano | head -5000"}, _ctx(self.workspace))
        self.assertTrue(result.ok)
        self.assertIn("обрезан", result.content)

    def test_openai_schema_exposes_command(self):
        schema = self.tool.openai_schema()
        self.assertEqual(schema["function"]["name"], "execute_bash")
        self.assertIn("command", schema["function"]["parameters"]["properties"])


if __name__ == "__main__":
    import unittest

    unittest.main()
