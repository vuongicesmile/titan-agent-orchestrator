"""Durable orchestration of one agent run."""

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
        self.repository.add_event(run.id, "run_started", "Đã bắt đầu xử lý yêu cầu.")
        self.repository.update_run(run.id, transition(run.status, "running"))
        try:
            proposal = self.planner.propose(message, self.gateway.catalog(principal))
        except httpx.HTTPError:
            return self._fail(run, "Không đọc được capability catalog từ Titan Control Plane.")
        if proposal is None:
            if any(term in message.casefold() for term in ("tạo yêu cầu", "tạo request", "bảo trì")):
                response = (
                    "Để tạo yêu cầu FM, hãy cung cấp loại yêu cầu, tiêu đề và bằng chứng hoặc thiết bị liên quan."
                )
            else:
                response = "Mình cần thêm ngữ cảnh để chọn capability phù hợp."
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
            self.repository.add_event(run.id, "cancelled", "Người dùng đã từ chối đề xuất.")
            self.repository.add_message(run.conversation_id, "assistant", "Đã hủy đề xuất; chưa có tool nào được gọi.")
            return self.repository.get_run(run.id, principal.org_id, principal.user_id)
        if "agent:execute" not in principal.permissions:
            raise PermissionError("Missing agent:execute permission")
        self.repository.update_run(run.id, transition(run.status, "running"), proposal=run.proposal)
        self.repository.add_event(run.id, "progress", "Đã xác nhận; đang thực thi capability.")
        return self._execute(
            self.repository.get_run(run.id, principal.org_id, principal.user_id),
            run.proposal,
            principal,
        )

    def _execute(self, run: Run, proposal, principal: Principal) -> Run:
        self.repository.add_event(run.id, "tool_started", f"Đang gọi {proposal.capability}.")
        try:
            execution = self.gateway.execute(
                principal,
                run.id,
                f"agent-run:{run.id}:{proposal.proposal_hash}",
                proposal.capability,
                proposal.arguments,
            )
        except httpx.HTTPError:
            return self._fail(run, "Capability gateway không phản hồi; tool ghi sẽ không tự retry.", proposal)
        result = execution.result
        self.repository.update_run(run.id, "completed", proposal=proposal, result=result)
        self.repository.add_event(
            run.id, "tool_finished", "Capability đã hoàn tất.", {"capability": proposal.capability}
        )
        self.repository.add_message(run.conversation_id, "assistant", "Đã hoàn tất capability và lưu kết quả vào run.")
        return self.repository.get_run(run.id, principal.org_id, principal.user_id)

    def _fail(self, run: Run, message: str, proposal=None) -> Run:
        self.repository.update_run(run.id, "failed", proposal=proposal, error=message)
        self.repository.add_event(run.id, "failed", message)
        self.repository.add_message(run.conversation_id, "assistant", message)
        return self.repository.get_run(run.id, run.org_id, run.user_id)
