import json
import os
from collections import deque
from datetime import datetime


class UserContextStore:
    """
    Two separate structures serve two separate purposes:

    context_window  — a rolling deque of the last 5 turns (query + answer).
                      Passed into the retriever/generator on every query so
                      recent conversation is always available as context.
                      When the 6th turn arrives the oldest is popped out.

    history         — append-only log of every turn, capped at 50 entries to
                      keep the JSON file small. Used only for persona building.

    Persona lifecycle
    -----------------
    Every 5th query: analyse the last 5 queries for topics, level, and
    question style, then MERGE the new analysis into the existing persona
    (replace changed fields, keep unchanged ones). No accumulation bloat.
    """

    WINDOW_SIZE   = 5   # rolling context window size
    PERSONA_EVERY = 5   # rebuild persona after every N queries
    HISTORY_CAP   = 10  # max turns kept in history JSON

    def __init__(
        self,
        storage_path: str = "project/context_store.json",
        persona_path: str = "project/user_persona.txt",
    ):
        self.storage_path = storage_path
        self.persona_path = persona_path
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)

        self.context_window: deque = deque(maxlen=self.WINDOW_SIZE)
        self.history: list         = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not os.path.exists(self.storage_path):
            return
        with open(self.storage_path, "r") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            self.history = data.get("history", [])
            # Restore context window from the tail of history
            for entry in self.history[-self.WINDOW_SIZE:]:
                self.context_window.append(entry)
        else:
            # Legacy plain-list format
            self.history = data
            for entry in self.history[-self.WINDOW_SIZE:]:
                self.context_window.append(entry)

    def _persist(self) -> None:
        with open(self.storage_path, "w") as fh:
            json.dump({"history": self.history}, fh, indent=4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_query(self, query: str, answer: str) -> bool:
        """
        Record a turn and return True when a persona rebuild is due.

        The context window is updated automatically (oldest entry popped
        when it exceeds WINDOW_SIZE). History is capped at HISTORY_CAP.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query":     query,
            "answer":    answer,
        }

        # Rolling context window -- deque handles the pop automatically
        self.context_window.append(entry)

        # Append-only history, capped
        self.history.append(entry)
        if len(self.history) > self.HISTORY_CAP:
            self.history = self.history[-self.HISTORY_CAP:]

        self._persist()

        # Trigger persona rebuild on every PERSONA_EVERY-th query
        return len(self.history) % self.PERSONA_EVERY == 0

    def get_context_window(self) -> list:
        """
        Return the last up-to-5 turns as a plain list, oldest first.
        Passed into the generator so it can use recent Q&A as context.
        """
        return list(self.context_window)

    def get_persona_window(self) -> list:
        """
        The last PERSONA_EVERY turns from history, used for persona analysis.
        Always exactly PERSONA_EVERY entries once enough history exists.
        """
        return self.history[-self.PERSONA_EVERY:]

    def get_persona(self) -> str:
        if os.path.exists(self.persona_path):
            with open(self.persona_path, "r") as fh:
                return fh.read().strip()
        return "No persona yet."

    def has_persona(self) -> bool:
        return os.path.exists(self.persona_path) and self.get_persona() != "No persona yet."

    def update_persona_file(self, new_summary: str) -> None:
        with open(self.persona_path, "w") as fh:
            fh.write(new_summary)

    def window_progress(self) -> str:
        """e.g. '3/5' — how many turns are in the current persona cycle."""
        position = len(self.history) % self.PERSONA_EVERY
        if position == 0 and self.history:
            position = self.PERSONA_EVERY
        return f"{position}/{self.PERSONA_EVERY}"
