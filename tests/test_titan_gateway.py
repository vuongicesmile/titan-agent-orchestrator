import httpx

from titan_agent_orchestrator.adapters.titan import TitanCapabilityGateway
from titan_agent_orchestrator.models import Principal


def test_gateway_uses_control_plane_service_auth_and_preserves_delegated_scope():
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["X-Titan-Agent-Gateway-Token"] == "test-service-token"
        assert request.headers["X-Titan-Org-Id"] == "demo"
        assert request.headers["X-Titan-User-Id"] == "user-1"
        assert request.headers["X-Titan-Permissions"] == "agent:chat,dataset:query"
        assert "Authorization" not in request.headers
        if request.method == "GET":
            return httpx.Response(200, json=[{
                "name": "data.answer_question", "description": "Hỏi dữ liệu", "input_schema": {"type": "object"},
            }])
        assert request.headers["Idempotency-Key"] == "run-1-read"
        return httpx.Response(200, json={"answer": "OK"})

    principal = Principal(org_id="demo", user_id="user-1", permissions={"agent:chat", "dataset:query"})
    with httpx.Client(base_url="http://titan-api:8000", transport=httpx.MockTransport(handle)) as client:
        gateway = TitanCapabilityGateway("http://titan-api:8000", "test-service-token", client=client)
        assert gateway.catalog(principal)[0].name == "data.answer_question"
        execution = gateway.execute(principal, "run-1", "run-1-read", "data.answer_question", {"question": "Test"})
        assert execution.result == {"answer": "OK"}
    assert [request.url.path for request in requests] == [
        "/api/internal/agent/capabilities",
        "/api/internal/agent/capabilities/data.answer_question/execute",
    ]
