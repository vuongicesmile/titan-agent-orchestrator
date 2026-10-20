"""Ports for capability catalog and execution."""

from typing import Protocol

from titan_agent_orchestrator.models import Capability, CapabilityExecution, Principal


class CapabilityGateway(Protocol):
    def catalog(self, principal: Principal) -> list[Capability]: ...

    def execute(
        self,
        principal: Principal,
        run_id: str,
        idempotency_key: str,
        capability: str,
        arguments: dict,
    ) -> CapabilityExecution: ...
