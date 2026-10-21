"""Điều phối bền vững cho một agent run."""

from __future__ import annotations

import httpx

from titan_agent_orchestrator.capabilities.contracts import CapabilityGateway
from titan_agent_orchestrator.models import Principal, Run
from titan_agent_orchestrator.orchestration.planner import IntentPlanner
from titan_agent_orchestrator.orchestration.state import transition
from titan_agent_orchestrator.persistence.sqlite import SQLiteRepository


class OrchestrationEngine:
    def __init__(self, repository: SQLiteRepository, gateway: CapabilityGateway) -> None:
        self.repository = repository
        self.gateway = gateway
        self.planner = IntentPlanner()

    def start(self, conversation_id: str, message: str, principal: Principal) -> Run:
        run = self.repository.create_run(conversation_id, principal.org_id, principal.user_id)
        self.repository.add_event(run.id, "run_started", "Request processing started.")
        self.repository.update_run(run.id, transition(run.status, "running"))
        try:
            proposal = self.planner.propose(message, self.gateway.catalog(principal))
        except httpx.HTTPError:
            return self._fail(run, "Unable to load the capability catalog from Titan Control Plane.")
        if proposal is None:
            if any(term in message.casefold() for term in ("tạo yêu cầu", "tạo request", "bảo trì")):
                response = (
                    "To create an FM request, provide the request type, title, and "
                    "supporting evidence or related equipment."
                )
            else:
                response = "More context is required to select an appropriate capability."
            self.repository.add_message(conversation_id, "assistant", response)
            self.repository.update_run(run.id, "completed", result={"message": response})
            self.repository.add_event(run.id, "progress", response)
            return self.repository.get_run(run.id, principal.org_id, principal.user_id)
        if proposal.requires_confirmation:
            self.repository.update_run(run.id, transition("running", "waiting_confirmation"), proposal=proposal)
            self.repository.add_event(
                run.id, "proposal", proposal.summary, {"proposal": proposal.model_dump(mode="json")}
            )
            self.repository.add_message(conversation_id, "assistant", proposal.summary)
            return self.repository.get_run(run.id, principal.org_id, principal.user_id)
        return self._execute(run, proposal, principal)

    def decide(self, run: Run, decision: str, proposal_hash: str, principal: Principal) -> Run:
        if run.status != "waiting_confirmation" or run.proposal is None:
            raise ValueError("Run is not waiting for a decision")
        if run.proposal.proposal_hash != proposal_hash:
            raise ValueError("Decision does not match the current proposal")
        if decision == "reject":
            self.repository.update_run(run.id, transition(run.status, "cancelled"), proposal=run.proposal)
            self.repository.add_event(run.id, "cancelled", "The user rejected the proposal.")
            self.repository.add_message(run.conversation_id, "assistant", "Proposal cancelled; no tool was called.")
            return self.repository.get_run(run.id, principal.org_id, principal.user_id)
        if "agent:execute" not in principal.permissions:
            raise PermissionError("Missing agent:execute permission")
        self.repository.update_run(run.id, transition(run.status, "running"), proposal=run.proposal)
        self.repository.add_event(run.id, "progress", "Proposal confirmed; executing capability.")
        return self._execute(
            self.repository.get_run(run.id, principal.org_id, principal.user_id),
            run.proposal,
            principal,
        )

    def _execute(self, run: Run, proposal, principal: Principal) -> Run:
        self.repository.add_event(run.id, "tool_started", f"Calling {proposal.capability}.")
        try:
            execution = self.gateway.execute(
                principal,
                run.id,
                f"agent-run:{run.id}:{proposal.proposal_hash}",
                proposal.capability,
                proposal.arguments,
            )
        except httpx.HTTPError:
            return self._fail(
                run,
                "Capability gateway did not respond; write tools will not be retried automatically.",
                proposal,
            )
        result = execution.result
        self.repository.update_run(run.id, "completed", proposal=proposal, result=result)
        self.repository.add_event(
            run.id, "tool_finished", "Capability completed.", {"capability": proposal.capability}
        )
        self.repository.add_message(
            run.conversation_id,
            "assistant",
            "Capability completed and the result was stored in the run.",
        )
        return self.repository.get_run(run.id, principal.org_id, principal.user_id)

    def _fail(self, run: Run, message: str, proposal=None) -> Run:
        self.repository.update_run(run.id, "failed", proposal=proposal, error=message)
        self.repository.add_event(run.id, "failed", message)
        self.repository.add_message(run.conversation_id, "assistant", message)
        return self.repository.get_run(run.id, run.org_id, run.user_id)
