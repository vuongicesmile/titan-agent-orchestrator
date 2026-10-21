from __future__ import annotations

import json

import httpx

from titan_agent_orchestrator.models import Capability, Message
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


def test_planner_uses_backup_and_keeps_original_question():
    hosts = []

    def handle(request):
        hosts.append(request.url.host)
        if request.url.host == "primary.example":
            return httpx.Response(503, text="unavailable")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '```json\n{"capability":"data.answer_question"}\n```',
                        }
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        planner = IntentPlanner(
            Settings(
                groq_base_url="https://primary.example/v1",
                groq_api_key="primary-key",
                zai_base_url="https://backup.example/v4",
                zai_api_key="backup-key",
            ),
            client,
        )
        question = "Điện năng hôm nay biến động thế nào?"
        proposal = planner.propose(
            question,
            [
                Capability(
                    name="data.answer_question",
                    description="Answer governed data questions.",
                    input_schema={"type": "object", "required": ["question"]},
                )
            ],
        )
    assert hosts == ["primary.example", "backup.example"]
    assert proposal is not None
    assert proposal.arguments == {"question": question}
    assert not proposal.requires_confirmation


def test_planner_recovers_with_rules_after_both_providers_fail():
    hosts = []

    def handle(request):
        hosts.append(request.url.host)
        return httpx.Response(503, text="unavailable")

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        planner = IntentPlanner(
            Settings(
                groq_api_key="primary-key",
                zai_api_key="backup-key",
            ),
            client,
        )
        proposal = planner.propose(
            "Kiểm tra xu hướng dữ liệu",
            [
                Capability(
                    name="data.answer_question",
                    description="Answer governed data questions.",
                    input_schema={"type": "object", "required": ["question"]},
                )
            ],
        )
    assert hosts == ["api.groq.com", "api.z.ai"]
    assert proposal is not None
    assert proposal.capability == "data.answer_question"


def test_backup_cannot_select_capability_outside_catalog():
    def handle(request):
        if request.url.host == "api.groq.com":
            return httpx.Response(503, text="unavailable")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"capability":"fm.create_request"}',
                        }
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        planner = IntentPlanner(
            Settings(
                groq_api_key="primary-key",
                zai_api_key="backup-key",
            ),
            client,
        )
        assert planner.propose("Tạo yêu cầu bảo trì", []) is None


def test_llm_planner_answers_normal_chat_instead_of_selecting_a_capability():
    def handle(request):
        payload = json.loads(request.content)
        assert payload["messages"][0]["content"].startswith("You are Titan AI")
        assert "MCP means Model Context" in payload["messages"][0]["content"]
        assert payload["temperature"] == 0.2
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Chào bạn, tôi có thể hỗ trợ gì?"}}]},
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        planner = IntentPlanner(
            Settings(llm_base_url="https://llm.example/v1", llm_api_key="test-key", llm_model="test-model"),
            client,
        )
        answer = planner.answer_normal_chat([
            Message(id="m-1", conversation_id="c-1", role="user", content="hi", created_at="2026-01-01T00:00:00Z")
        ])

    assert answer == "Chào bạn, tôi có thể hỗ trợ gì?"
