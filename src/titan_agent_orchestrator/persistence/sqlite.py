"""SQLite repository cho phát triển local và test xác định."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from titan_agent_orchestrator.models import (
    Conversation,
    Message,
    Plan,
    Proposal,
    Run,
    RunEvent,
    RunStatus,
)


def _now() -> datetime:
    return datetime.now(UTC)


_UNSET = object()


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
                    user_id TEXT NOT NULL, status TEXT NOT NULL, proposal TEXT, plan TEXT, result TEXT,
                    error TEXT, response_type TEXT NOT NULL DEFAULT 'run',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
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
            # The repository intentionally has no migration dependency. Keep
            # existing local databases usable when the PlannerAgent fields are
            # introduced after a previous checkout was run.
            columns = {row[1] for row in database.execute("PRAGMA table_info(runs)").fetchall()}
            if "plan" not in columns:
                database.execute("ALTER TABLE runs ADD COLUMN plan TEXT")
            if "response_type" not in columns:
                database.execute("ALTER TABLE runs ADD COLUMN response_type TEXT NOT NULL DEFAULT 'run'")

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
        doc["plan"] = json.loads(doc["plan"]) if doc.get("plan") else None
        doc["result"] = json.loads(doc["result"]) if doc["result"] else None
        doc["response_type"] = doc.get("response_type") or "run"
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

    def list_conversations(self, org_id: str, user_id: str, limit: int = 20) -> list[Conversation]:
        """Đọc lịch sử hội thoại gần nhất trong đúng phạm vi tenant và người dùng."""
        with self._connection() as database:
            rows = database.execute(
                "SELECT * FROM conversations WHERE org_id=? AND user_id=? "
                "ORDER BY updated_at DESC, id DESC LIMIT ?",
                (org_id, user_id, limit),
            ).fetchall()
        return [self._conversation(row) for row in rows]

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
                "INSERT INTO runs (id, conversation_id, org_id, user_id, status, proposal, plan, result, "
                "error, response_type, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, 'run', ?, ?)",
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

    def list_runs(self, conversation_id: str, org_id: str, user_id: str) -> list[Run]:
        """Đọc run theo thứ tự tạo để khôi phục trạng thái tool của hội thoại."""
        with self._connection() as database:
            rows = database.execute(
                "SELECT * FROM runs WHERE conversation_id=? AND org_id=? AND user_id=? "
                "ORDER BY created_at, id",
                (conversation_id, org_id, user_id),
            ).fetchall()
        return [self._run(row) for row in rows]

    def update_run(
        self,
        identity: str,
        status: RunStatus,
        *,
        proposal: Proposal | None | object = _UNSET,
        plan: Plan | None | object = _UNSET,
        result: dict | None | object = _UNSET,
        error: str | None | object = _UNSET,
        response_type: str | object = _UNSET,
    ) -> None:
        now = _now().isoformat()
        with self._connection() as database:
            row = database.execute(
                "SELECT proposal, plan, result, error, response_type FROM runs WHERE id=?", (identity,)
            ).fetchone()
            if row is None:
                raise KeyError("Run was not found")
            current = dict(row)
            selected_proposal = current["proposal"] if proposal is _UNSET else (
                json.dumps(proposal.model_dump(mode="json")) if proposal is not None else None
            )
            selected_plan = current["plan"] if plan is _UNSET else (
                json.dumps(plan.model_dump(mode="json")) if plan is not None else None
            )
            selected_result = current["result"] if result is _UNSET else (
                json.dumps(result) if result is not None else None
            )
            selected_error = current["error"] if error is _UNSET else error
            selected_response_type = current["response_type"] if response_type is _UNSET else response_type
            database.execute(
                "UPDATE runs SET status=?, proposal=?, plan=?, result=?, error=?, response_type=?, "
                "updated_at=? WHERE id=?",
                (
                    status,
                    selected_proposal,
                    selected_plan,
                    selected_result,
                    selected_error,
                    selected_response_type,
                    now,
                    identity,
                ),
            )

    def claim_plan_decision(
        self,
        identity: str,
        org_id: str,
        user_id: str,
        plan_id: str,
        plan_version: int,
        target_status: str,
        response_type: str = "run",
    ) -> bool:
        """Atomically claim a pending plan so repeated clicks execute once."""
        with self._connection() as database:
            cursor = database.execute(
                "UPDATE runs SET status=?, plan=json_set(plan, '$.status', ?), response_type=?, updated_at=? "
                "WHERE id=? AND org_id=? AND user_id=? AND status='waiting_confirmation' "
                "AND json_extract(plan, '$.id')=? AND CAST(json_extract(plan, '$.version') AS INTEGER)=? "
                "AND json_extract(plan, '$.status')='pending_approval'",
                (
                    "running" if target_status == "approved" else "cancelled",
                    target_status,
                    response_type,
                    _now().isoformat(),
                    identity,
                    org_id,
                    user_id,
                    plan_id,
                    plan_version,
                ),
            )
            return cursor.rowcount == 1

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
