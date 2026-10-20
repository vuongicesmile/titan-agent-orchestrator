"""SQLite repository for local development and deterministic tests."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from titan_agent_orchestrator.models import (
    Conversation,
    Message,
    Proposal,
    Run,
    RunEvent,
    RunStatus,
)


def _now() -> datetime:
    return datetime.now(UTC)


class SQLiteRepository:
    def __init__(self, database_url: str) -> None:
        if not database_url.startswith("sqlite:///"):
            raise ValueError("The initial repository supports sqlite:/// URLs only")
        self.path = Path(database_url.removeprefix("sqlite:///"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connection() as database:
            database.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, org_id TEXT NOT NULL, user_id TEXT NOT NULL,
                    title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, role TEXT NOT NULL,
                    content TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, org_id TEXT NOT NULL,
                    user_id TEXT NOT NULL, status TEXT NOT NULL, proposal TEXT, result TEXT,
                    error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS run_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, kind TEXT NOT NULL,
                    message TEXT NOT NULL, data TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conversations_scope ON conversations(org_id, user_id, updated_at);
                CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_runs_conversation ON runs(conversation_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_events_run ON run_events(run_id, id);
                """
            )

    @staticmethod
    def _conversation(row: sqlite3.Row) -> Conversation:
        return Conversation(**dict(row))

    @staticmethod
    def _message(row: sqlite3.Row) -> Message:
        return Message(**dict(row))

    @staticmethod
    def _run(row: sqlite3.Row) -> Run:
        doc = dict(row)
        doc["proposal"] = json.loads(doc["proposal"]) if doc["proposal"] else None
        doc["result"] = json.loads(doc["result"]) if doc["result"] else None
        return Run(**doc)

    @staticmethod
    def _event(row: sqlite3.Row) -> RunEvent:
        doc = dict(row)
        doc["data"] = json.loads(doc["data"])
        return RunEvent(**doc)

    def create_conversation(self, org_id: str, user_id: str, title: str) -> Conversation:
        now, identity = _now(), uuid4().hex
        with self._connection() as database:
            database.execute(
                "INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?)",
                (identity, org_id, user_id, title, now.isoformat(), now.isoformat()),
            )
        return self.get_conversation(identity, org_id, user_id)

    def get_conversation(self, identity: str, org_id: str, user_id: str) -> Conversation:
        with self._connection() as database:
            row = database.execute(
                "SELECT * FROM conversations WHERE id=? AND org_id=? AND user_id=?",
                (identity, org_id, user_id),
            ).fetchone()
        if row is None:
            raise KeyError("Conversation was not found")
        return self._conversation(row)

    def add_message(self, conversation_id: str, role: str, content: str) -> Message:
        now, identity = _now(), uuid4().hex
        with self._connection() as database:
            database.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
                (identity, conversation_id, role, content, now.isoformat()),
            )
            database.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?",
                (now.isoformat(), conversation_id),
            )
        return Message(id=identity, conversation_id=conversation_id, role=role, content=content, created_at=now)

    def list_messages(self, conversation_id: str) -> list[Message]:
        with self._connection() as database:
            rows = database.execute(
                "SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at, id",
                (conversation_id,),
            ).fetchall()
        return [self._message(row) for row in rows]

    def create_run(self, conversation_id: str, org_id: str, user_id: str) -> Run:
        now, identity = _now(), uuid4().hex
        with self._connection() as database:
            database.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?)",
                (
                    identity,
                    conversation_id,
                    org_id,
                    user_id,
                    "queued",
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
        return self.get_run(identity, org_id, user_id)

    def get_run(self, identity: str, org_id: str, user_id: str) -> Run:
        with self._connection() as database:
            row = database.execute(
                "SELECT * FROM runs WHERE id=? AND org_id=? AND user_id=?",
                (identity, org_id, user_id),
            ).fetchone()
        if row is None:
            raise KeyError("Run was not found")
        return self._run(row)

    def update_run(
        self,
        identity: str,
        status: RunStatus,
        *,
        proposal: Proposal | None = None,
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        now = _now().isoformat()
        with self._connection() as database:
            database.execute(
                "UPDATE runs SET status=?, proposal=?, result=?, error=?, updated_at=? WHERE id=?",
                (
                    status,
                    json.dumps(proposal.model_dump(mode="json")) if proposal else None,
                    json.dumps(result) if result is not None else None,
                    error,
                    now,
                    identity,
                ),
            )

    def add_event(self, run_id: str, kind: str, message: str, data: dict | None = None) -> RunEvent:
        now = _now()
        with self._connection() as database:
            cursor = database.execute(
                "INSERT INTO run_events (run_id, kind, message, data, created_at) VALUES (?, ?, ?, ?, ?)",
                (run_id, kind, message, json.dumps(data or {}), now.isoformat()),
            )
        return RunEvent(
            id=cursor.lastrowid,
            run_id=run_id,
            kind=kind,
            message=message,
            data=data or {},
            created_at=now,
        )

    def list_events(self, run_id: str, after_id: int = 0) -> list[RunEvent]:
        with self._connection() as database:
            rows = database.execute(
                "SELECT * FROM run_events WHERE run_id=? AND id>? ORDER BY id", (run_id, after_id)
            ).fetchall()
        return [self._event(row) for row in rows]
