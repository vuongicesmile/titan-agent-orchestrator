"""Model request và response công khai."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RunStatus = Literal["queued", "running", "waiting_confirmation", "completed", "failed", "cancelled"]
EventKind = Literal[
    "run_started",
    "progress",
    "proposal",
    "plan_generated",
    "approval_required",
    "task_started",
    "task_finished",
    "replanned",
    "tool_started",
    "tool_finished",
    "failed",
    "cancelled",
]
PlanStatus = Literal["draft", "pending_approval", "approved", "running", "completed", "failed", "discarded"]
PlanTaskStatus = Literal["pending", "running", "waiting_approval", "completed", "failed", "skipped"]


class Principal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    org_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,30}$")
    user_id: str = Field(min_length=1, max_length=200)
    permissions: frozenset[str] = Field(default_factory=frozenset)


class SignedContext(Principal):
    expires_at: datetime


class CreateConversationInput(BaseModel):
    title: str = Field(default="", max_length=160)


class Conversation(BaseModel):
    id: str
    org_id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class MessageInput(BaseModel):
    content: str = Field(min_length=1, max_length=4000)
    background: bool = False


class Message(BaseModel):
    id: str
    conversation_id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class Proposal(BaseModel):
    capability: str
    arguments: dict[str, Any]
    summary: str
    proposal_hash: str
    requires_confirmation: bool = True


class PlanTask(BaseModel):
    """One delegated capability invocation in a PlannerAgent plan."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    agent: str = Field(min_length=1, max_length=120)
    capability: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,100}$")
    description: str = Field(min_length=1, max_length=1000)
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list, max_length=32)
    requires_approval: bool = False
    write: bool = False
    status: PlanTaskStatus = "pending"
    attempts: int = Field(default=0, ge=0)
    result: dict[str, Any] | None = None
    error: str | None = None


class Plan(BaseModel):
    """Durable, user-reviewable execution plan owned by a run."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-f0-9]{16,64}$")
    version: int = Field(default=1, ge=1)
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    intent: str = Field(min_length=1, max_length=200)
    status: PlanStatus = "draft"
    tasks: list[PlanTask] = Field(min_length=1, max_length=32)
    plan_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    @property
    def requires_approval(self) -> bool:
        return any(task.requires_approval for task in self.tasks)


class PlanDecisionInput(BaseModel):
    """Approval binds to server-persisted plan identity and version only."""

    decision: Literal["approve", "discard"]
    plan_id: str = Field(pattern=r"^[a-f0-9]{16,64}$")
    plan_version: int = Field(ge=1)
    idempotency_key: str = Field(default="", min_length=0, max_length=200)


class RunEvent(BaseModel):
    id: int
    run_id: str
    kind: EventKind
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class Run(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    conversation_id: str
    org_id: str
    user_id: str
    status: RunStatus
    proposal: Proposal | None = None
    plan: Plan | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    response_type: Literal["run", "approval_required", "result"] = Field(default="run", alias="type")
    created_at: datetime
    updated_at: datetime


class DecisionInput(BaseModel):
    decision: Literal["confirm", "reject"]
    proposal_hash: str = Field(min_length=64, max_length=64)


class Capability(BaseModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,100}$")
    description: str
    input_schema: dict[str, Any]
    # Newer Control Plane catalogs may publish the owning agent. Older
    # catalogs omit it; PlannerAgent derives a display label from the namespace.
    agent: str | None = Field(default=None, min_length=1, max_length=120)
    write: bool = False
    requires_confirmation: bool = False
    timeout_seconds: int = Field(default=20, ge=1, le=120)


class CapabilityExecution(BaseModel):
    capability: str
    arguments: dict[str, Any]
    result: dict[str, Any]
