from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from core.policy import Capability
from core.tools.base import Tool, ToolContext, ToolResult

NOTES_FILE = "notes.json"
MAX_LIST_NOTES = 20
MAX_SEARCH_RESULTS = 20
_lock = threading.Lock()


def _load(notes_file: str = NOTES_FILE) -> list:
    if not os.path.exists(notes_file):
        return []
    with open(notes_file, encoding="utf-8") as f:
        return json.load(f)


def _save(notes: list, notes_file: str = NOTES_FILE) -> None:
    with open(notes_file, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)


def _next_id(notes: list) -> int:
    return max((note["id"] for note in notes), default=0) + 1


def _render(notes: list) -> str:
    return "\n".join(f"#{note['id']} [{note['created_at']}]: {note['text']}" for note in notes)


def _search_terms(query: str) -> list[str]:
    return [word for word in query.casefold().split() if len(word) >= 2]


def execute(
    action: str,
    text: str = "",
    query: str = "",
    note_id: int = 0,
    notes_file: str = NOTES_FILE,
) -> str:
    with _lock:
        notes = _load(notes_file)

        if action == "add":
            text = text.strip()
            if not text:
                return "Ошибка: для add нужен text."
            note = {
                "id": _next_id(notes),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "text": text,
            }
            notes.append(note)
            _save(notes, notes_file)
            return f"Заметка #{note['id']} сохранена."

        if action == "search":
            query = query.strip()
            if not query:
                return "Ошибка: для search нужен query."
            terms = _search_terms(query)
            # Поиск по любому слову запроса: заметки короткие, важнее найти,
            # чем требовать точного совпадения всей фразы.
            found = [
                note
                for note in notes
                if terms and any(term in note["text"].casefold() for term in terms)
            ]
            if not found:
                return f"Ничего не найдено по запросу '{query}'."
            return _render(found[-MAX_SEARCH_RESULTS:])

        if action == "list":
            if not notes:
                return "Заметок пока нет."
            return _render(notes[-MAX_LIST_NOTES:])

        if action == "remove":
            before = len(notes)
            notes = [note for note in notes if note["id"] != note_id]
            if len(notes) == before:
                return f"Заметка #{note_id} не найдена."
            _save(notes, notes_file)
            return f"Заметка #{note_id} удалена."

        return f"Неизвестный action: {action}"


class NotesInput(BaseModel):
    action: Literal["add", "search", "list", "remove"]
    text: str = Field(default="", max_length=2000)
    query: str = Field(default="", max_length=200)
    note_id: int = Field(default=0, ge=0)


class NotesTool(Tool):
    """Persistent notes in a shared JSON file — the agent's long-term memory."""

    name: ClassVar[str] = "notes"
    description: ClassVar[str] = (
        "Persistent memory for facts that must survive across conversations. "
        "action=add: save a note (requires text). "
        "action=search: find notes containing any word of query (case-insensitive); "
        "extract the topic keyword from the question instead of searching the question verbatim. "
        "action=list: show recent notes. "
        "action=remove: delete a note by note_id. "
        "Use add when the user asks to remember something for the future, "
        "and search before answering about previously saved facts."
    )
    input_model: ClassVar[type[BaseModel]] = NotesInput
    capabilities: ClassVar[frozenset[Capability]] = frozenset({Capability.NOTES_WRITE})
    timeout: ClassVar[float] = 5.0
    output_limit: ClassVar[int] = 3000

    def __init__(self, notes_file: str = NOTES_FILE):
        self.notes_file = notes_file

    def execute(self, args: NotesInput, ctx: ToolContext) -> ToolResult:
        ctx.raise_if_cancelled()
        text = execute(
            action=args.action,
            text=args.text,
            query=args.query,
            note_id=args.note_id,
            notes_file=self.notes_file,
        )
        failed = text.startswith("Ошибка") or text.endswith("не найдена.")
        if failed:
            retryable = "не найдена" not in text
            return ToolResult.failure(text, code="validation", retryable=retryable)
        return ToolResult(content=text, summary=text)
