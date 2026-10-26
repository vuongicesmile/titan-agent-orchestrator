from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient

from titan_agent_orchestrator import auth
from titan_agent_orchestrator.main import create_app
from titan_agent_orchestrator.models import Capability, CapabilityExecution, Principal, Proposal
from titan_agent_orchestrator.orchestration.engine import _capability_failure_message, _humanize_result
from titan_agent_orchestrator.orchestration.planner import IntentPlanner
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
            result={"answer": "Question received", "run_id": run_id},
        )


class FMGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def catalog(self, principal: Principal) -> list[Capability]:
        return [
            Capability(
                name="fm.list_workflows",
                description="List active FM workflows.",
                input_schema={"type": "object"},
            ),
            Capability(
                name="fm.create_request",
                description="Create an FM request.",
                input_schema={
                    "type": "object",
                    "required": ["workflow_definition_id", "request_type", "title"],
                },
                write=True,
                requires_confirmation=True,
            ),
        ]

    def execute(self, principal, run_id, idempotency_key, capability, arguments) -> CapabilityExecution:
        self.calls.append((capability, arguments))
        if capability == "fm.list_workflows":
            result = {"result": {"structuredContent": {"result": [{"id": "workflow-1", "name": "FM default"}]}}}
        else:
            result = {"result": {"structuredContent": {"result": {"request_code": "REQ-TEST", "status": "draft"}}}}
        return CapabilityExecution(capability=capability, arguments=arguments, result=result)


class DashboardGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def catalog(self, principal: Principal) -> list[Capability]:
        return [
            Capability(
                name="reports.create_draft",
                description="Create a dashboard draft.",
                input_schema={"type": "object", "required": ["prompt"]},
                write=True,
                requires_confirmation=True,
            ),
            Capability(
                name="reports.publish_draft",
                description="Publish a dashboard draft.",
                input_schema={"type": "object", "required": ["session_id"]},
                write=True,
                requires_confirmation=True,
            ),
        ]

    def execute(self, principal, run_id, idempotency_key, capability, arguments) -> CapabilityExecution:
        self.calls.append((capability, arguments))
        if capability == "reports.create_draft":
            result = {
                "session_id": "dashboard-session-1",
                "title": "Energy by building",
                "report": "energy_by_building",
                "reply": "I prepared the requested dashboard draft.",
                "preview": {"energy": {"rows": []}},
            }
        else:
            result = {"job_id": "job-1", "report": "energy_by_building", "dashboard_url": "http://superset/dashboard"}
        return CapabilityExecution(capability=capability, arguments=arguments, result=result)


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
    assert run["result"]["answer"] == "Question received"

    events = client.get(f"/v1/runs/{run['id']}/events", headers=headers).json()
    assert [event["kind"] for event in events] == ["run_started", "progress", "progress", "tool_started", "tool_finished"]


def test_config_exposes_router_order_without_credentials(monkeypatch, tmp_path):
    secret = "test-context-secret"
    monkeypatch.setattr(auth, "settings", Settings(context_signing_secret=secret))
    app = create_app(SQLiteRepository(f"sqlite:///{tmp_path / 'agent.db'}"), FakeGateway())
    app.state.engine.planner = IntentPlanner(Settings(groq_api_key="private-groq", zai_api_key="private-zai"))
    try:
        response = TestClient(app).get("/v1/config", headers={"X-Titan-Context": signed_context(secret)})
    finally:
        app.state.engine.planner.llm.close()
    assert response.status_code == 200
    assert response.json() == [
        {
            "provider": "groq",
            "model": "openai/gpt-oss-120b",
            "base_url": "https://api.groq.com/openai/v1",
            "priority": "primary",
        },
        {
            "provider": "zai",
            "model": "glm-4.7",
            "base_url": "https://api.z.ai/api/paas/v4",
            "priority": "fallback",
        },
    ]
    assert "private-groq" not in response.text
    assert "private-zai" not in response.text


def test_normal_chat_uses_llm_when_no_capability_is_selected(monkeypatch, tmp_path):
    secret = "test-context-secret"
    monkeypatch.setattr(auth, "settings", Settings(context_signing_secret=secret))
    repository = SQLiteRepository(f"sqlite:///{tmp_path / 'agent.db'}")
    app = create_app(repository, FakeGateway())
    app.state.engine.planner = SimpleNamespace(
        propose=lambda message, catalog: None,
        answer_normal_chat=lambda history: "Hello! How can I help with Titan Data?",
    )
    client = TestClient(app)
    headers = {"X-Titan-Context": signed_context(secret)}
    conversation_id = client.post("/v1/conversations", json={}, headers=headers).json()["id"]

    response = client.post(
        f"/v1/conversations/{conversation_id}/messages",
        json={"content": "Giúp tôi lập checklist kiểm tra thiết bị trước khi bảo trì."},
        headers=headers,
    )

    assert response.status_code == 202
    run = response.json()
    assert run["status"] == "completed"
    assert run["result"] == {"message": "Hello! How can I help with Titan Data?"}
    events = client.get(f"/v1/runs/{run['id']}/events", headers=headers).json()
    assert [event["kind"] for event in events] == ["run_started", "progress", "progress", "progress", "progress"]
    assert events[-2]["message"] == "Generating a conversation response."


def test_profiling_overview_is_humanized_without_raw_tool_payload():
    response = _humanize_result(
        "fm.get_data_profiling_overview",
        {
            "result": {
                "structuredContent": {
                    "result": {
                        "layer": "bronze",
                        "totals": {
                            "assets": 2,
                            "profiled_rows": 209,
                            "quality_score": 81.5,
                            "statuses": {"pass": 1, "warn": 0, "fail": 1},
                        },
                        "assets": [
                            {
                                "connection_id": "bms_fake",
                                "table": "cov_events",
                                "result": {
                                    "status": "fail",
                                    "violations": [{"message": "3 negative readings require review."}],
                                },
                            }
                        ],
                    }
                }
            }
        },
    )

    assert "Đánh giá dữ liệu BRONZE" in response
    assert "bms_fake/cov_events" in response
    assert "3 negative readings require review." in response


def test_conversation_history_is_scoped_and_ordered(monkeypatch, tmp_path):
    secret = "test-context-secret"
    monkeypatch.setattr(auth, "settings", Settings(context_signing_secret=secret))
    repository = SQLiteRepository(f"sqlite:///{tmp_path / 'agent.db'}")
    client = TestClient(create_app(repository, FakeGateway()))
    headers = {"X-Titan-Context": signed_context(secret)}

    first = client.post("/v1/conversations", json={"title": "First FM question"}, headers=headers).json()
    second = client.post("/v1/conversations", json={"title": "Second FM question"}, headers=headers).json()
    client.post(
        f"/v1/conversations/{first['id']}/messages",
        json={"content": "Kiểm tra xu hướng dữ liệu hôm nay"},
        headers=headers,
    )

    history = client.get("/v1/conversations", headers=headers)
    assert history.status_code == 200
    assert [item["id"] for item in history.json()] == [first["id"], second["id"]]

    detail = client.get(f"/v1/conversations/{first['id']}", headers=headers)
    assert detail.status_code == 200
    assert [message["role"] for message in detail.json()["messages"]] == ["user", "assistant"]
    assert len(detail.json()["runs"]) == 1


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


def test_natural_language_fm_request_resolves_workflow_then_requires_confirmation(monkeypatch, tmp_path):
    secret = "test-context-secret"
    monkeypatch.setattr(auth, "settings", Settings(context_signing_secret=secret))
    gateway = FMGateway()
    client = TestClient(create_app(SQLiteRepository(f"sqlite:///{tmp_path / 'agent.db'}"), gateway))
    headers = {"X-Titan-Context": signed_context(secret)}
    conversation_id = client.post("/v1/conversations", json={}, headers=headers).json()["id"]

    created = client.post(
        f"/v1/conversations/{conversation_id}/messages",
        json={"content": "Tạo yêu cầu FM kiểm tra AHU-01, ưu tiên cao"},
        headers=headers,
    )
    assert created.status_code == 202
    proposal_run = created.json()
    assert proposal_run["status"] == "waiting_confirmation"
    assert proposal_run["proposal"]["capability"] == "fm.create_request"
    assert proposal_run["proposal"]["arguments"]["workflow_definition_id"] == "workflow-1"
    assert proposal_run["proposal"]["arguments"]["priority"] == "high"
    assert gateway.calls == [("fm.list_workflows", {})]

    completed = client.post(
        f"/v1/runs/{proposal_run['id']}/decision",
        json={"decision": "confirm", "proposal_hash": proposal_run["proposal"]["proposal_hash"]},
        headers=headers,
    )
    assert completed.status_code == 200
    assert completed.json()["status"] == "completed"
    assert completed.json()["result"]["message"] == "Đã tạo yêu cầu FM dạng nháp REQ-TEST."


def test_dashboard_draft_then_publish_uses_the_same_conversation_draft(monkeypatch, tmp_path):
    secret = "test-context-secret"
    monkeypatch.setattr(auth, "settings", Settings(context_signing_secret=secret))
    gateway = DashboardGateway()
    client = TestClient(create_app(SQLiteRepository(f"sqlite:///{tmp_path / 'agent.db'}"), gateway))
    headers = {"X-Titan-Context": signed_context(secret)}
    conversation_id = client.post("/v1/conversations", json={}, headers=headers).json()["id"]

    proposed_draft = client.post(
        f"/v1/conversations/{conversation_id}/messages",
        json={"content": "Tạo dashboard điện năng theo tòa nhà"},
        headers=headers,
    ).json()
    assert proposed_draft["status"] == "waiting_confirmation"
    assert proposed_draft["proposal"]["capability"] == "reports.create_draft"

    created_draft = client.post(
        f"/v1/runs/{proposed_draft['id']}/decision",
        json={"decision": "confirm", "proposal_hash": proposed_draft["proposal"]["proposal_hash"]},
        headers=headers,
    ).json()
    assert created_draft["status"] == "completed"
    assert created_draft["result"]["session_id"] == "dashboard-session-1"
    assert gateway.calls == [("reports.create_draft", {"prompt": "Tạo dashboard điện năng theo tòa nhà"})]

    proposed_publish = client.post(
        f"/v1/conversations/{conversation_id}/messages",
        json={"content": "Xuất bản dashboard"},
        headers=headers,
    ).json()
    assert proposed_publish["status"] == "waiting_confirmation"
    assert proposed_publish["proposal"]["capability"] == "reports.publish_draft"
    assert proposed_publish["proposal"]["arguments"] == {"session_id": "dashboard-session-1"}

    published = client.post(
        f"/v1/runs/{proposed_publish['id']}/decision",
        json={"decision": "confirm", "proposal_hash": proposed_publish["proposal"]["proposal_hash"]},
        headers=headers,
    ).json()
    assert published["status"] == "completed"
    assert published["result"]["message"] == "Đã gửi job publish dashboard **energy_by_building**."
    assert gateway.calls[-1] == ("reports.publish_draft", {"session_id": "dashboard-session-1"})


def test_capability_http_failure_does_not_claim_the_gateway_is_unreachable():
    request = httpx.Request(
        "POST", "http://titan-api:8000/api/internal/agent/capabilities/reports.create_draft/execute"
    )
    error = httpx.HTTPStatusError("bad gateway", request=request, response=httpx.Response(502, request=request))

    message = _capability_failure_message("reports.create_draft", error)

    assert "received reports.create_draft" in message
    assert "HTTP 502" in message
    assert "gateway did not respond" not in message
