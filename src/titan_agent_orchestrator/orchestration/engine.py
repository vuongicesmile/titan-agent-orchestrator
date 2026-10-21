"""Điều phối bền vững cho một agent run."""

from __future__ import annotations

import httpx

from titan_agent_orchestrator.capabilities.contracts import CapabilityGateway
from titan_agent_orchestrator.models import Principal, Run
from titan_agent_orchestrator.orchestration.planner import IntentPlanner, _hash
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
            if any(term in message.casefold() for term in ("tạo yêu cầu", "tạo request", "create request")):
                response = (
                    "To create an FM request, provide the request type, title, and "
                    "supporting evidence or related equipment."
                )
            else:
                response = self.planner.answer_normal_chat(self.repository.list_messages(conversation_id))
            self.repository.add_message(conversation_id, "assistant", response)
            self.repository.update_run(run.id, "completed", result={"message": response})
            self.repository.add_event(run.id, "progress", "General conversation response generated.")
            return self.repository.get_run(run.id, principal.org_id, principal.user_id)
        if proposal.capability == "fm.create_request":
            prepared = self._prepare_request(proposal, principal, run)
            if prepared is None:
                response = (
                    "Chưa có workflow FM khả dụng để tạo yêu cầu. "
                    "Hãy tạo workflow đầu tiên trong mục Yêu cầu FM, rồi thử lại."
                )
                self.repository.add_message(conversation_id, "assistant", response)
                self.repository.update_run(run.id, "completed", result={"message": response})
                self.repository.add_event(run.id, "progress", "No active FM workflow was available for the request.")
                return self.repository.get_run(run.id, principal.org_id, principal.user_id)
            proposal = prepared
        if proposal.capability == "reports.publish_draft":
            prepared = self._prepare_report_publish(proposal, principal, run)
            if prepared is None:
                response = "Chưa có bản nháp dashboard trong hội thoại này để publish. Hãy tạo và kiểm tra bản nháp trước."
                self.repository.add_message(conversation_id, "assistant", response)
                self.repository.update_run(run.id, "completed", result={"message": response})
                self.repository.add_event(run.id, "progress", "No dashboard draft was available for publishing.")
                return self.repository.get_run(run.id, principal.org_id, principal.user_id)
            proposal = prepared
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
        except httpx.HTTPStatusError as exc:
            return self._fail(run, _capability_failure_message(proposal.capability, exc), proposal)
        except httpx.HTTPError:
            return self._fail(
                run,
                "Capability gateway did not respond; write tools will not be retried automatically.",
                proposal,
            )
        result = {**execution.result, "message": _humanize_result(proposal.capability, execution.result)}
        self.repository.update_run(run.id, "completed", proposal=proposal, result=result)
        self.repository.add_event(
            run.id, "tool_finished", "Capability completed.", {"capability": proposal.capability}
        )
        self.repository.add_message(
            run.conversation_id,
            "assistant",
            result["message"],
        )
        return self.repository.get_run(run.id, principal.org_id, principal.user_id)

    def _prepare_request(self, proposal, principal: Principal, run: Run):
        """Chọn workflow active trước khi tạo proposal write để người dùng kiểm tra."""
        if proposal.arguments.get("workflow_definition_id") != "__resolve_default__":
            return proposal
        self.repository.add_event(run.id, "progress", "Resolving the active FM workflow.")
        try:
            result = self.gateway.execute(
                principal,
                run.id,
                f"agent-run:{run.id}:resolve-workflow",
                "fm.list_workflows",
                {},
            ).result
        except httpx.HTTPError:
            return None
        workflows = _items_from_result(result)
        if not workflows:
            return None
        workflow_id = workflows[0].get("id") if isinstance(workflows[0], dict) else None
        if not isinstance(workflow_id, str) or not workflow_id:
            return None
        arguments = {**proposal.arguments, "workflow_definition_id": workflow_id}
        return proposal.model_copy(
            update={
                "arguments": arguments,
                "proposal_hash": _hash(proposal.capability, arguments),
                "summary": "Đã chuẩn bị yêu cầu FM dạng nháp. Hãy xác nhận để tạo yêu cầu.",
            }
        )

    def _prepare_report_publish(self, proposal, principal: Principal, run: Run):
        """Gắn publish với bản nháp mới nhất cùng hội thoại, không nhận session tùy ý từ LLM."""
        if proposal.arguments.get("session_id") != "__resolve_latest__":
            return proposal
        for previous in reversed(self.repository.list_runs(run.conversation_id, principal.org_id, principal.user_id)):
            if previous.id == run.id or previous.status != "completed" or previous.proposal is None:
                continue
            if previous.proposal.capability != "reports.create_draft" or not previous.result:
                continue
            draft = _first_result(previous.result)
            session_id = draft.get("session_id") if isinstance(draft, dict) else None
            if not isinstance(session_id, str) or not session_id:
                continue
            arguments = {"session_id": session_id}
            return proposal.model_copy(
                update={
                    "arguments": arguments,
                    "proposal_hash": _hash(proposal.capability, arguments),
                    "summary": "Đã chọn bản nháp dashboard gần nhất. Hãy xác nhận để chạy job publish.",
                }
            )
        return None

    def _fail(self, run: Run, message: str, proposal=None) -> Run:
        self.repository.update_run(run.id, "failed", proposal=proposal, error=message)
        self.repository.add_event(run.id, "failed", message)
        self.repository.add_message(run.conversation_id, "assistant", message)
        return self.repository.get_run(run.id, run.org_id, run.user_id)


def _items_from_result(result: dict) -> list[dict]:
    """MCP trả structuredContent lồng trong envelope của Control Plane."""
    value: object = result
    if isinstance(value, dict):
        value = value.get("result", value)
    if isinstance(value, dict):
        value = value.get("structuredContent", value)
    if isinstance(value, dict) and "result" in value:
        value = value["result"]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict) and isinstance(value.get("items"), list):
        return [item for item in value["items"] if isinstance(item, dict)]
    return []


def _humanize_result(capability: str, result: dict) -> str:
    """Chuyển kết quả tool thành câu trả lời người dùng đọc được; raw chỉ nằm trong trace."""
    if capability == "data.answer_question":
        answer = result.get("answer")
        return answer.strip() if isinstance(answer, str) and answer.strip() else "Không tìm thấy câu trả lời từ dữ liệu được cấp quyền."

    items = _items_from_result(result)
    if capability == "fm.list_requests":
        if not items:
            return "Không tìm thấy yêu cầu FM nào trong phạm vi bạn có quyền xem."
        return _format_requests("Tìm thấy", items)
    if capability == "fm.get_overdue_requests":
        if not items:
            return "Không có yêu cầu FM quá hạn trong phạm vi bạn có quyền xem."
        return _format_requests("Tìm thấy yêu cầu FM quá hạn", items)
    if capability == "fm.list_workflows":
        if not items:
            return "Chưa có workflow FM đang hoạt động."
        names = [str(item.get("name") or item.get("code") or "Workflow") for item in items[:5]]
        suffix = "" if len(items) <= 5 else f" và {len(items) - 5} workflow khác"
        return f"Có {len(items)} workflow FM đang hoạt động: " + ", ".join(names) + suffix + "."
    if capability == "fm.get_pending_approvals":
        if not items:
            return "Không có bước phê duyệt FM nào đang chờ bạn xử lý."
        return f"Có {len(items)} bước phê duyệt FM đang chờ bạn xử lý."
    if capability == "fm.create_request":
        request = _first_result(result)
        code = request.get("request_code") if isinstance(request, dict) else None
        return f"Đã tạo yêu cầu FM dạng nháp{f' {code}' if isinstance(code, str) else ''}."
    if capability == "reports.list":
        if not items:
            return "Chưa có dashboard nào trong phạm vi tenant này. Bạn có thể mô tả dashboard cần tạo."
        names = [str(item.get("title") or item.get("name") or "Dashboard") for item in items[:5]]
        suffix = "" if len(items) <= 5 else f" và {len(items) - 5} dashboard khác"
        return f"Có {len(items)} dashboard: " + ", ".join(names) + suffix + "."
    if capability == "reports.create_draft":
        reply = result.get("reply")
        title = result.get("title")
        suffix = f" Bản nháp: **{title}**." if isinstance(title, str) and title else ""
        if isinstance(reply, str) and reply.strip():
            return reply.strip() + suffix
        return "Đã tạo bản nháp dashboard để bạn kiểm tra." + suffix
    if capability == "reports.publish_draft":
        report = result.get("report")
        return f"Đã gửi job publish dashboard{f' **{report}**' if isinstance(report, str) else ''}."
    return "Đã hoàn tất thao tác FM."


def _first_result(result: dict) -> dict | None:
    value: object = result
    if isinstance(value, dict):
        value = value.get("result", value)
    if isinstance(value, dict):
        value = value.get("structuredContent", value)
    if isinstance(value, dict) and isinstance(value.get("result"), dict):
        return value["result"]
    return value if isinstance(value, dict) else None


def _capability_failure_message(capability: str, error: httpx.HTTPStatusError) -> str:
    """Phân biệt Control Plane từ chối tool với lỗi mất kết nối đến gateway."""
    status = error.response.status_code
    if status >= 500:
        return (
            f"Control Plane received {capability} but could not complete it (HTTP {status}). "
            "The action was not retried automatically; verify the result before trying again."
        )
    return f"Control Plane rejected {capability} (HTTP {status}). Review permissions or the request and try again."


def _format_requests(prefix: str, items: list[dict]) -> str:
    lines = []
    for item in items[:5]:
        code = item.get("request_code") or "Yêu cầu FM"
        title = item.get("title") or "Không có tiêu đề"
        status = item.get("status")
        due_at = item.get("due_at")
        detail = f"- **{code}** — {title}"
        if status:
            detail += f" · trạng thái: {status}"
        if due_at:
            detail += f" · hạn: {due_at}"
        lines.append(detail)
    extra = f"\nCòn {len(items) - 5} yêu cầu khác." if len(items) > 5 else ""
    return f"{prefix} {len(items)} yêu cầu:\n" + "\n".join(lines) + extra
