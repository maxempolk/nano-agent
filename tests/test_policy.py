import tempfile
from pathlib import Path
from unittest import TestCase

from core.policy import Capability, ExecutionPolicy


class PolicyTests(TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)

    def test_read_capabilities_are_allowed_by_default(self):
        policy = ExecutionPolicy(self.workspace)
        decision = policy.check_capabilities(
            frozenset({Capability.FILESYSTEM_READ, Capability.SHELL_READ, Capability.NETWORK_READ})
        )
        self.assertTrue(decision.allowed)

    def test_external_send_is_controlled(self):
        policy = ExecutionPolicy(self.workspace)
        denied = policy.check_capabilities(frozenset({Capability.EXTERNAL_SEND}))
        self.assertFalse(denied.allowed)

        allowed = ExecutionPolicy(self.workspace, allow_external_send=True).check_capabilities(
            frozenset({Capability.EXTERNAL_SEND})
        )
        self.assertTrue(allowed.allowed)

    def test_destructive_is_rejected_by_default(self):
        policy = ExecutionPolicy(self.workspace)
        decision = policy.check_capabilities(frozenset({Capability.DESTRUCTIVE}))
        self.assertFalse(decision.allowed)

    def test_destructive_requires_approval_when_enabled(self):
        policy = ExecutionPolicy(self.workspace, allow_destructive=True)
        decision = policy.check_capabilities(frozenset({Capability.DESTRUCTIVE}))
        self.assertTrue(decision.allowed)
        self.assertTrue(decision.needs_approval)

    def test_workspace_membership(self):
        policy = ExecutionPolicy(self.workspace)
        inside = self.workspace / "sub" / "file.txt"
        self.assertTrue(policy.inside_workspace(inside))
        self.assertTrue(policy.inside_workspace("relative/path.txt"))
        self.assertFalse(policy.inside_workspace("/etc/passwd"))
        self.assertFalse(policy.inside_workspace(Path.home()))

    def test_workspace_membership_rejects_parent_traversal(self):
        policy = ExecutionPolicy(self.workspace)
        self.assertFalse(policy.inside_workspace(self.workspace / ".." / "other"))

    def test_approval_without_callback_is_denied(self):
        policy = ExecutionPolicy(self.workspace)
        self.assertFalse(policy.approve("rm -rf x"))

    def test_approval_callback_is_used(self):
        policy = ExecutionPolicy(self.workspace, approval=lambda description: "safe" in description)
        self.assertTrue(policy.approve("safe operation"))
        self.assertFalse(policy.approve("risky operation"))

    def test_broken_approval_callback_denies(self):
        def broken(_description):
            raise RuntimeError("no")

        policy = ExecutionPolicy(self.workspace, approval=broken)
        self.assertFalse(policy.approve("anything"))


if __name__ == "__main__":
    import unittest

    unittest.main()
