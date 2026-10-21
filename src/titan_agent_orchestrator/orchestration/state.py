"""Các trạng thái chuyển đổi hợp lệ của run."""

from titan_agent_orchestrator.models import RunStatus

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
