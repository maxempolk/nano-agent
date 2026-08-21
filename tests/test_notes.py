import json
import os
import tempfile
from unittest import TestCase

from core.tools.base import ErrorCode, ToolContext, ToolRegistry
from core.tools.notes import NotesTool, _load


class NotesToolTests(TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.notes_file = os.path.join(self._tmp.name, "notes.json")
        self.addCleanup(self._tmp.cleanup)

    def _tool(self) -> NotesTool:
        return NotesTool(notes_file=self.notes_file)

    def test_add_list_remove_cycle(self):
        tool = self._tool()
        added = tool.run({"action": "add", "text": "машина на ТО 15.09"}, ToolContext())
        self.assertTrue(added.ok, added.error)
        self.assertIn("#1", added.content)

        listed = tool.run({"action": "list"}, ToolContext())
        self.assertIn("машина на ТО 15.09", listed.content)

        removed = tool.run({"action": "remove", "note_id": 1}, ToolContext())
        self.assertTrue(removed.ok)
        self.assertEqual(_load(self.notes_file), [])

    def test_add_assigns_incrementing_ids(self):
        tool = self._tool()
        tool.run({"action": "add", "text": "первая"}, ToolContext())
        tool.run({"action": "add", "text": "вторая"}, ToolContext())

        stored = _load(self.notes_file)
        self.assertEqual([note["id"] for note in stored], [1, 2])

    def test_search_is_case_insensitive(self):
        tool = self._tool()
        tool.run({"action": "add", "text": "Поляк М.В — список поступления"}, ToolContext())
        tool.run({"action": "add", "text": "купить хлеб"}, ToolContext())

        found = tool.run({"action": "search", "query": "поляк"}, ToolContext())
        self.assertIn("Поляк М.В", found.content)
        self.assertNotIn("хлеб", found.content)

    def test_search_matches_any_query_word(self):
        tool = self._tool()
        tool.run(
            {"action": "add", "text": "Меня зовут Максим, но называй меня хозяин"},
            ToolContext(),
        )

        found = tool.run({"action": "search", "query": "как меня зовут"}, ToolContext())
        self.assertIn("Максим", found.content)

    def test_search_ignores_single_letter_words(self):
        tool = self._tool()
        tool.run({"action": "add", "text": "заметка"}, ToolContext())

        result = tool.run({"action": "search", "query": "а б в"}, ToolContext())
        self.assertIn("Ничего не найдено", result.content)

    def test_search_empty_result_is_honest(self):
        tool = self._tool()
        result = tool.run({"action": "search", "query": "нет такого"}, ToolContext())
        self.assertTrue(result.ok)
        self.assertIn("Ничего не найдено", result.content)

    def test_empty_list_is_honest(self):
        tool = self._tool()
        result = tool.run({"action": "list"}, ToolContext())
        self.assertTrue(result.ok)
        self.assertIn("Заметок пока нет", result.content)

    def test_missing_text_is_validation_error(self):
        tool = self._tool()
        result = tool.run({"action": "add"}, ToolContext())
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "validation")
        self.assertTrue(result.retryable)

    def test_remove_unknown_id_is_not_retryable(self):
        tool = self._tool()
        result = tool.run({"action": "remove", "note_id": 99}, ToolContext())
        self.assertFalse(result.ok)
        self.assertFalse(result.retryable)

    def test_unknown_action_fails_validation_without_execution(self):
        tool = self._tool()
        result = tool.run({"action": "wipe"}, ToolContext())
        self.assertEqual(result.error_code, ErrorCode.INVALID_ARGUMENTS)
        self.assertEqual(_load(self.notes_file), [])

    def test_notes_are_persisted_to_configured_file(self):
        tool = self._tool()
        tool.run({"action": "add", "text": "факт"}, ToolContext())
        with open(self.notes_file, encoding="utf-8") as f:
            stored = json.load(f)
        self.assertEqual(stored[0]["text"], "факт")
        self.assertIn("created_at", stored[0])

    def test_registry_schema_exposes_actions(self):
        registry = ToolRegistry([self._tool()])
        schema = registry.schemas()[0]["function"]
        self.assertEqual(schema["name"], "notes")
        self.assertIn("action", schema["parameters"]["properties"])


if __name__ == "__main__":
    import unittest

    unittest.main()
