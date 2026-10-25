from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from titan_agent_orchestrator import auth
from titan_agent_orchestrator.main import create_app
from titan_agent_orchestrator.models import Capability, CapabilityExecution, Principal
from titan_agent_orchestrator.persistence.sqlite import SQLiteRepository
from titan_agent_orchestrator.settings import Settings


class Gateway:
    def catalog(self, principal: Principal) -> list[Capability]:
        return [
            Capability(name="data.answer_question", description="Analyze data", input_schema={"required": ["question"]}),
            Capability(
                name="reports.create_draft",
                description="Create dashboard draft",
                input_schema={"required": ["prompt"]},
                write=True,
                requires_confirmation=True,
            ),
        ]

    def execute(self, principal, run_id, idempotency_key, capability, arguments) -> CapabilityExecution:
        return CapabilityExecution(
            capability=capability,
            arguments=arguments,
            result={"answer": "ok"} if capability == "data.answer_question" else {"session_id": "draft-1"},
        )


def context(secret: str, permissions: list[str]) -> str:
    payload = {
        "org_id": "demo",
        "user_id": "user-1",
        "permissions": permissions,
        "expires_at": (datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return encoded + "." + hmac.new(secret.encode(), encoded.encode(), hashlib.sha256).hexdigest()


def test_plan_approval_contract_and_duplicate_click(monkeypatch, tmp_path):
    secret = "planner-api-secret"
    monkeypatch.setattr(auth, "settings", Settings(context_signing_secret=secret))
    client = TestClient(create_app(SQLiteRepository(f"sqlite:///{tmp_path / 'planner.db'}"), Gateway()))
    headers = {"X-Titan-Context": context(secret, ["agent:chat", "agent:execute"])}
    conversation = client.post("/v1/conversations", json={}, headers=headers).json()
    run = client.post(
        f"/v1/conversations/{conversation['id']}/messages",
        json={"content": "Analyze consumption and create a dashboard"},
        headers=headers,
    ).json()

    assert run["type"] == "approval_required"
    assert run["plan"]["status"] == "pending_approval"
    response = client.post(
        f"/v1/runs/{run['id']}/plan/decision",
        json={"decision": "approve", "plan_id": run["plan"]["id"], "plan_version": run["plan"]["version"]},
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    duplicate = client.post(
        f"/v1/runs/{run['id']}/plan/decision",
        json={"decision": "approve", "plan_id": run["plan"]["id"], "plan_version": run["plan"]["version"]},
        headers=headers,
    )
    assert duplicate.status_code == 200


def test_plan_approval_requires_execute_permission(monkeypatch, tmp_path):
    secret = "planner-api-secret"
    monkeypatch.setattr(auth, "settings", Settings(context_signing_secret=secret))
    repository = SQLiteRepository(f"sqlite:///{tmp_path / 'planner.db'}")
    client = TestClient(create_app(repository, Gateway()))
    headers = {"X-Titan-Context": context(secret, ["agent:chat", "agent:execute"])}
    conversation = client.post("/v1/conversations", json={}, headers=headers).json()
    run = client.post(
        f"/v1/conversations/{conversation['id']}/messages",
        json={"content": "Analyze consumption and create a dashboard"},
        headers=headers,
    ).json()
    limited = {"X-Titan-Context": context(secret, ["agent:chat"])}
    response = client.post(
        f"/v1/runs/{run['id']}/plan/decision",
        json={"decision": "approve", "plan_id": run["plan"]["id"], "plan_version": run["plan"]["version"]},
        headers=limited,
    )
    assert response.status_code == 403
