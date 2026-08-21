from unittest import TestCase

from core.events import (
    EventBus,
    ModelCompleted,
    RouteSelected,
    RunCancelled,
    RunCompleted,
    RunStarted,
    ToolCompleted,
    ToolStarted,
    preview,
)


class EventTests(TestCase):
    def test_events_are_small_and_typed(self):
        started = RunStarted(input_preview="привет")
        route = RouteSelected(route="local", model="system", reason="simple", score=0, automatic=True)
        completed = ModelCompleted(
            model="system", step=1, finish_reason="stop", tool_calls=0, content_chars=12, elapsed=0.4
        )
        self.assertEqual(started.kind, "run_started")
        self.assertEqual(route.kind, "route_selected")
        self.assertEqual(completed.kind, "model_completed")

    def test_preview_truncates_long_text(self):
        text = "abc " * 200
        clipped = preview(text, limit=50)
        self.assertLessEqual(len(clipped), 50)
        self.assertTrue(clipped.endswith("…"))

    def test_preview_collapses_whitespace(self):
        self.assertEqual(preview("a\n\n  b\tc"), "a b c")

    def test_bus_delivers_events_in_order(self):
        bus = EventBus()
        received = []
        bus.subscribe(received.append)

        bus.emit(ToolStarted(name="web_search", args_summary="query=x"))
        bus.emit(
            ToolCompleted(name="web_search", summary="found", ok=True, error_code="", elapsed=0.2)
        )

        self.assertEqual([event.kind for event in received], ["tool_started", "tool_completed"])

    def test_broken_listener_does_not_break_other_listeners(self):
        bus = EventBus()
        received = []

        def broken(_event):
            raise RuntimeError("boom")

        bus.subscribe(broken)
        bus.subscribe(received.append)

        bus.emit(RunStarted(input_preview="x"))
        self.assertEqual(len(received), 1)

    def test_unsubscribe_stops_delivery(self):
        bus = EventBus()
        received = []
        bus.subscribe(received.append)
        bus.unsubscribe(received.append)

        bus.emit(RunCancelled(reason="user", elapsed=0.1))
        self.assertEqual(received, [])

    def test_run_completed_event_carries_stats(self):
        event = RunCompleted(reply_preview="ответ", elapsed=1.2, steps=2, model_calls=3, tool_calls=1)
        self.assertEqual(event.kind, "run_completed")
        self.assertEqual(event.model_calls, 3)


if __name__ == "__main__":
    import unittest

    unittest.main()
