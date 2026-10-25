"""Các trạng thái chuyển đổi hợp lệ của run."""

from titan_agent_orchestrator.models import PlanStatus, RunStatus

_ALLOWED: dict[RunStatus, frozenset[RunStatus]] = {
    "queued": frozenset({"running", "cancelled"}),
    "running": frozenset({"waiting_confirmation", "completed", "failed"}),
    "waiting_confirmation": frozenset({"running", "cancelled"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}


def transition(current: RunStatus, target: RunStatus) -> RunStatus:
    if target not in _ALLOWED[current]:
        raise ValueError(f"Run cannot transition from {current} to {target}")
    return target


_PLAN_ALLOWED: dict[PlanStatus, frozenset[PlanStatus]] = {
    "draft": frozenset({"pending_approval", "approved", "failed", "discarded"}),
    "pending_approval": frozenset({"approved", "discarded", "failed"}),
    "approved": frozenset({"running", "failed", "discarded"}),
    "running": frozenset({"completed", "failed"}),
    "completed": frozenset(),
    "failed": frozenset({"draft", "pending_approval", "approved"}),
    "discarded": frozenset(),
}


def plan_transition(current: PlanStatus, target: PlanStatus) -> PlanStatus:
    if target not in _PLAN_ALLOWED[current]:
        raise ValueError(f"Plan cannot transition from {current} to {target}")
    return target
