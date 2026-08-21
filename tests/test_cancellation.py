import threading
from unittest import TestCase

from core.cancellation import CancellationToken, CancelledError


class CancellationTokenTests(TestCase):
    def test_starts_not_cancelled(self):
        token = CancellationToken()
        self.assertFalse(token.cancelled)
        token.raise_if_cancelled()

    def test_cancel_sets_flag_and_reason(self):
        token = CancellationToken()
        token.cancel("user request")
        self.assertTrue(token.cancelled)
        self.assertEqual(token.reason, "user request")
        with self.assertRaises(CancelledError):
            token.raise_if_cancelled()

    def test_first_reason_wins(self):
        token = CancellationToken()
        token.cancel("first")
        token.cancel("second")
        self.assertEqual(token.reason, "first")

    def test_cancel_is_visible_from_other_thread(self):
        token = CancellationToken()
        seen = {}

        def worker():
            token.wait(5)
            seen["cancelled"] = token.cancelled

        thread = threading.Thread(target=worker)
        thread.start()
        token.cancel()
        thread.join(timeout=5)
        self.assertTrue(seen["cancelled"])

    def test_wait_returns_false_when_not_cancelled(self):
        token = CancellationToken()
        self.assertFalse(token.wait(0.01))


if __name__ == "__main__":
    import unittest

    unittest.main()
