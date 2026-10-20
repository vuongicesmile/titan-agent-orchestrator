from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from titan_agent_orchestrator import auth
from titan_agent_orchestrator.main import create_app
from titan_agent_orchestrator.models import Capability, CapabilityExecution, Principal, Proposal
from titan_agent_orchestrator.persistence.sqlite import SQLiteRepository
from titan_agent_orchestrator.settings import Settings


class FakeGateway:
    def catalog(self, principal: Principal) -> list[Capability]:
        return [
            Capability(
                name="data.answer_question",
                description="Answer an approved metric question.",
                input_schema={"type": "object"},
            )
        ]

    def execute(self, principal, run_id, idempotency_key, capability, arguments) -> CapabilityExecution:
        return CapabilityExecution(
            capability=capability,
            arguments=arguments,
            result={"answer": "Đã nhận câu hỏi", "run_id": run_id},
        )


def signed_context(secret: str) -> str:
    doc = {
        "org_id": "demo",
        "user_id": "user-1",
        "permissions": ["agent:chat", "agent:execute"],
        "expires_at": (datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
    }
    encoded = base64.urlsafe_b64encode(json.dumps(doc).encode()).decode().rstrip("=")
    signature = hmac.new(secret.encode(), encoded.encode(), hashlib.sha256).hexdigest()
    return encoded + "." + signature


def test_data_question_creates_durable_run(monkeypatch, tmp_path):
    secret = "test-context-secret"
    monkeypatch.setattr(auth, "settings", Settings(context_signing_secret=secret))
    repository = SQLiteRepository(f"sqlite:///{tmp_path / 'agent.db'}")
    client = TestClient(create_app(repository, FakeGateway()))
    headers = {"X-Titan-Context": signed_context(secret)}

    conversation = client.post("/v1/conversations", json={"title": "FM"}, headers=headers)
    assert conversation.status_code == 201
    conversation_id = conversation.json()["id"]

    response = client.post(
        f"/v1/conversations/{conversation_id}/messages",
        json={"content": "Kiểm tra xu hướng dữ liệu hôm nay"},
        headers=headers,
    )
    assert response.status_code == 202
    run = response.json()
    assert run["status"] == "completed"
    assert run["result"]["answer"] == "Đã nhận câu hỏi"

    events = client.get(f"/v1/runs/{run['id']}/events", headers=headers).json()
    assert [event["kind"] for event in events] == ["run_started", "tool_started", "tool_finished"]


def test_rejected_proposal_never_executes(monkeypatch, tmp_path):
    secret = "test-context-secret"
    monkeypatch.setattr(auth, "settings", Settings(context_signing_secret=secret))
    repository = SQLiteRepository(f"sqlite:///{tmp_path / 'agent.db'}")
    client = TestClient(create_app(repository, FakeGateway()))
    headers = {"X-Titan-Context": signed_context(secret)}
    conversation_id = client.post("/v1/conversations", json={}, headers=headers).json()["id"]
    run = repository.create_run(conversation_id, "demo", "user-1")
    proposal = Proposal(
        capability="fm.create_request",
        arguments={"title": "Inspect AHU"},
        summary="Create an FM request.",
        proposal_hash="a" * 64,
    )
    repository.update_run(run.id, "waiting_confirmation", proposal=proposal)

    response = client.post(
        f"/v1/runs/{run.id}/decision",
        json={"decision": "reject", "proposal_hash": proposal.proposal_hash},
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
