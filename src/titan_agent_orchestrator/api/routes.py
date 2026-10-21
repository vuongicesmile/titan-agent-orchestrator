"""API cho conversation và run."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from titan_agent_orchestrator.auth import PrincipalDep
from titan_agent_orchestrator.llm.models import ProviderStatus
from titan_agent_orchestrator.models import (
    Conversation,
    CreateConversationInput,
    DecisionInput,
    MessageInput,
    Run,
    RunEvent,
)
from titan_agent_orchestrator.orchestration.engine import OrchestrationEngine

router = APIRouter(prefix="/v1", tags=["agent-orchestration"])


def engine(request: Request) -> OrchestrationEngine:
    return request.app.state.engine


EngineDep = Annotated[OrchestrationEngine, Depends(engine)]


def _conversation_or_404(conversation_id: str, principal: PrincipalDep, service: EngineDep) -> Conversation:
    try:
        return service.repository.get_conversation(conversation_id, principal.org_id, principal.user_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation was not found") from None


def _run_or_404(run_id: str, principal: PrincipalDep, service: EngineDep) -> Run:
    try:
        return service.repository.get_run(run_id, principal.org_id, principal.user_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run was not found") from None


@router.get("/config", response_model=list[ProviderStatus])
def get_config(principal: PrincipalDep, service: EngineDep) -> list[ProviderStatus]:
    """Công bố router hiện hành cho Control Plane, không kèm credential."""
    return service.planner.llm.public_config()


@router.post("/conversations", status_code=201)
def create_conversation(body: CreateConversationInput, principal: PrincipalDep, service: EngineDep) -> Conversation:
    return service.repository.create_conversation(principal.org_id, principal.user_id, body.title.strip())


@router.get("/conversations", response_model=list[Conversation])
def list_conversations(principal: PrincipalDep, service: EngineDep, limit: int = 20) -> list[Conversation]:
    """Liệt kê hội thoại gần nhất; nội dung chỉ được trả qua endpoint chi tiết."""
    return service.repository.list_conversations(principal.org_id, principal.user_id, min(max(limit, 1), 50))


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, principal: PrincipalDep, service: EngineDep) -> dict:
    conversation = _conversation_or_404(conversation_id, principal, service)
    messages = service.repository.list_messages(conversation_id)
    runs = service.repository.list_runs(conversation_id, principal.org_id, principal.user_id)
    return {"conversation": conversation, "messages": messages, "runs": runs}


@router.post("/conversations/{conversation_id}/messages", status_code=202)
def send_message(
    conversation_id: str,
    body: MessageInput,
    principal: PrincipalDep,
    service: EngineDep,
) -> Run:
    _conversation_or_404(conversation_id, principal, service)
    service.repository.add_message(conversation_id, "user", body.content.strip())
    return service.start(conversation_id, body.content.strip(), principal)


@router.get("/runs/{run_id}")
def get_run(run_id: str, principal: PrincipalDep, service: EngineDep) -> Run:
    return _run_or_404(run_id, principal, service)


@router.get("/runs/{run_id}/events")
def get_events(run_id: str, principal: PrincipalDep, service: EngineDep, after_id: int = 0) -> list[RunEvent]:
    _run_or_404(run_id, principal, service)
    return service.repository.list_events(run_id, max(after_id, 0))


@router.post("/runs/{run_id}/decision")
def decide_run(
    run_id: str,
    body: DecisionInput,
    principal: PrincipalDep,
    service: EngineDep,
) -> Run:
    run = _run_or_404(run_id, principal, service)
    try:
        return service.decide(run, body.decision, body.proposal_hash, principal)
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from None
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
