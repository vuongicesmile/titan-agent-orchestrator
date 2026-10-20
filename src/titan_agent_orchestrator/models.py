"""Public request and response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RunStatus = Literal["queued", "running", "waiting_confirmation", "completed", "failed", "cancelled"]
EventKind = Literal["run_started", "progress", "proposal", "tool_started", "tool_finished", "failed", "cancelled"]


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


class RunEvent(BaseModel):
    id: int
    run_id: str
    kind: EventKind
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class Run(BaseModel):
    id: str
    conversation_id: str
    org_id: str
    user_id: str
    status: RunStatus
    proposal: Proposal | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class DecisionInput(BaseModel):
    decision: Literal["confirm", "reject"]
    proposal_hash: str = Field(min_length=64, max_length=64)


class Capability(BaseModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,100}$")
    description: str
    input_schema: dict[str, Any]
    write: bool = False
    requires_confirmation: bool = False
    timeout_seconds: int = Field(default=20, ge=1, le=120)


class CapabilityExecution(BaseModel):
    capability: str
    arguments: dict[str, Any]
    result: dict[str, Any]
