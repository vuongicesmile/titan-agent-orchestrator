from __future__ import annotations

import json

import httpx

from titan_agent_orchestrator.models import Capability
from titan_agent_orchestrator.orchestration.planner import IntentPlanner
from titan_agent_orchestrator.settings import Settings


def test_llm_planner_selects_catalog_capability_and_derives_safe_arguments():
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"capability":"data.answer_question"}'}}]},
        )

    planner = IntentPlanner(
        Settings(
            llm_base_url="https://llm.example/v1",
            llm_api_key="test-key",
            llm_model="test-model",
        ),
        httpx.Client(transport=httpx.MockTransport(handle)),
    )
    proposal = planner.propose(
        "Điện năng hôm nay biến động thế nào?",
        [
            Capability(
                name="data.answer_question",
                description="Answer governed data questions.",
                input_schema={"type": "object", "required": ["question"]},
            )
        ],
    )

    assert proposal is not None
    assert proposal.capability == "data.answer_question"
    assert proposal.arguments == {"question": "Điện năng hôm nay biến động thế nào?"}
    payload = json.loads(requests[0].content)
    assert payload["model"] == "test-model"
    assert payload["messages"][0]["content"].startswith("You are Titan's capability router.")
    assert "test-key" not in requests[0].content.decode()


def test_llm_planner_rejects_required_tool_arguments_invented_by_model():
    planner = IntentPlanner(
        Settings(llm_base_url="https://llm.example/v1", llm_api_key="test-key", llm_model="test-model"),
        httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={"choices": [{"message": {"content": '{"capability":"fm.create_request"}'}}]},
                )
            )
        ),
    )
    proposal = planner.propose(
        "Tạo yêu cầu bảo trì AHU",
        [
            Capability(
                name="fm.create_request",
                description="Create an FM request.",
                input_schema={"type": "object", "required": ["title"]},
                write=True,
                requires_confirmation=True,
            )
        ],
    )

    assert proposal is None
