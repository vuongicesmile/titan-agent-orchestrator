from __future__ import annotations

import httpx

from titan_agent_orchestrator.models import Capability, CapabilityExecution, Principal
from titan_agent_orchestrator.orchestration.engine import OrchestrationEngine
from titan_agent_orchestrator.orchestration.planner import PlannerAgent, _validated_plan
from titan_agent_orchestrator.persistence.sqlite import SQLiteRepository
from titan_agent_orchestrator.settings import Settings


def catalog() -> list[Capability]:
    return [
        Capability(
            name="data.answer_question",
            description="Analyze governed tenant data.",
            input_schema={"type": "object", "required": ["question"]},
        ),
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


class Gateway:
    def __init__(self, fail_once: bool = False) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.fail_once = fail_once

    def catalog(self, principal: Principal) -> list[Capability]:
        return catalog()

    def execute(self, principal, run_id, idempotency_key, capability, arguments) -> CapabilityExecution:
        self.calls.append((capability, arguments))
        if self.fail_once and capability == "data.answer_question":
            self.fail_once = False
            raise httpx.ReadTimeout("temporary", request=httpx.Request("POST", "http://gateway"))
        if capability == "data.answer_question":
            result = {"answer": "September analysis"}
        elif capability == "reports.create_draft":
            result = {"session_id": "draft-1", "title": "Power by building"}
        else:
            result = {"dashboard_url": "https://superset.local/dashboards/1"}
        return CapabilityExecution(capability=capability, arguments=arguments, result=result)


def principal(**kwargs) -> Principal:
    return Principal(org_id="demo", user_id=kwargs.get("user_id", "user-1"), permissions=frozenset(kwargs.get("permissions", {"agent:chat", "agent:execute"})))


def test_planner_generates_valid_dependency_aware_plan_without_hardcoded_execution():
    planner = PlannerAgent(Settings())
    plan = planner.plan("Analyze power consumption and detect anomalies, then create a dashboard", catalog())
    planner.llm.close()

    assert plan is not None
    assert plan.status == "pending_approval"
    assert [task.capability for task in plan.tasks] == [
        "data.answer_question",
        "reports.create_draft",
        "reports.publish_draft",
    ]
    assert plan.tasks[1].depends_on == [plan.tasks[0].id]
    assert plan.tasks[2].depends_on == [plan.tasks[1].id]
    assert plan.tasks[2].requires_approval
    assert plan.tasks[2].arguments == {"session_id": "$task_draft.session_id"}


def test_plan_validation_rejects_unknown_capability_and_cycle():
    draft = {
        "title": "Bad plan",
        "description": "",
        "intent": "test",
        "tasks": [
            {"id": "a", "agent": "A", "capability": "data.answer_question", "description": "a", "arguments": {"question": "x"}, "depends_on": ["b"]},
            {"id": "b", "agent": "B", "capability": "not.in.catalog", "description": "b", "arguments": {}, "depends_on": ["a"]},
        ],
    }
    try:
        _validated_plan("x", draft, catalog())
    except ValueError as error:
        assert "catalog" in str(error)
    else:
        raise AssertionError("invalid plan was accepted")


def test_engine_executes_ready_tasks_then_waits_and_approval_is_idempotent(tmp_path):
    gateway = Gateway()
    repository = SQLiteRepository(f"sqlite:///{tmp_path / 'planner.db'}")
    engine = OrchestrationEngine(repository, gateway)
    conversation = repository.create_conversation("demo", "user-1", "Planner")
    repository.add_message(conversation.id, "user", "Analyze power consumption and detect anomalies, then create a dashboard")
    run = engine.start(conversation.id, "Analyze power consumption and detect anomalies, then create a dashboard", principal())

    assert run.status == "waiting_confirmation"
    assert run.response_type == "approval_required"
    assert run.plan is not None and run.plan.status == "pending_approval"
    assert gateway.calls == [("data.answer_question", {"question": "Analyze power consumption and detect anomalies, then create a dashboard"})]

    approved = engine.decide_plan(run, "approve", run.plan.id, run.plan.version, principal())
    assert approved.status == "completed"
    assert [name for name, _ in gateway.calls] == [
        "data.answer_question",
        "reports.create_draft",
        "reports.publish_draft",
    ]
    duplicate = engine.decide_plan(run, "approve", run.plan.id, run.plan.version, principal())
    assert duplicate.status == "completed"
    assert len(gateway.calls) == 3


def test_discard_never_executes_protected_tasks(tmp_path):
    gateway = Gateway()
    repository = SQLiteRepository(f"sqlite:///{tmp_path / 'discard.db'}")
    engine = OrchestrationEngine(repository, gateway)
    conversation = repository.create_conversation("demo", "user-1", "Planner")
    repository.add_message(conversation.id, "user", "Analyze power consumption and detect anomalies, then create a dashboard")
    run = engine.start(conversation.id, "Analyze power consumption and detect anomalies, then create a dashboard", principal())
    discarded = engine.decide_plan(run, "discard", run.plan.id, run.plan.version, principal())
    assert discarded.status == "cancelled"
    assert discarded.plan is not None and discarded.plan.status == "discarded"
    assert gateway.calls == [("data.answer_question", {"question": "Analyze power consumption and detect anomalies, then create a dashboard"})]


def test_transient_read_failure_retries_but_successful_result_is_preserved(tmp_path):
    gateway = Gateway(fail_once=True)
    repository = SQLiteRepository(f"sqlite:///{tmp_path / 'retry.db'}")
    engine = OrchestrationEngine(repository, gateway)
    conversation = repository.create_conversation("demo", "user-1", "Planner")
    repository.add_message(conversation.id, "user", "Analyze power consumption and detect anomalies, then create a dashboard")
    run = engine.start(conversation.id, "Analyze power consumption and detect anomalies, then create a dashboard", principal())
    assert run.status == "waiting_confirmation"
    assert [name for name, _ in gateway.calls].count("data.answer_question") == 2
