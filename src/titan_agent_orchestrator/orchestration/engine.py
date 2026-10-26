"""Điều phối bền vững cho một agent run."""

from __future__ import annotations

from typing import Any

import httpx

from titan_agent_orchestrator.capabilities.contracts import CapabilityGateway
from titan_agent_orchestrator.models import Plan, PlanTask, Principal, Run
from titan_agent_orchestrator.orchestration.planner import IntentPlanner, _hash
from titan_agent_orchestrator.orchestration.state import transition
from titan_agent_orchestrator.persistence.sqlite import SQLiteRepository


class OrchestrationEngine:
    def __init__(self, repository: SQLiteRepository, gateway: CapabilityGateway) -> None:
        self.repository = repository
        self.gateway = gateway
        self.planner = IntentPlanner()
        self.max_task_retries = getattr(getattr(self.planner, "settings", None), "planner_max_task_retries", 2)

    def start(self, conversation_id: str, message: str, principal: Principal) -> Run:
        return self.execute_prepared(self.prepare(conversation_id, principal), message, principal)

    def prepare(self, conversation_id: str, principal: Principal) -> Run:
        """Lưu receipt trước khi chạy để UI có thể theo dõi đúng run ngay lập tức."""
        run = self.repository.create_run(conversation_id, principal.org_id, principal.user_id)
        self.repository.add_event(run.id, "run_started", "Request processing started.")
        return run

    def execute_prepared(self, run: Run, message: str, principal: Principal) -> Run:
        """Dùng chung luồng thực thi cho HTTP đồng bộ và tác vụ nền trong process."""
        conversation_id = run.conversation_id
        self.repository.update_run(run.id, transition(run.status, "running"))
        self.repository.add_event(run.id, "progress", "Loading available capabilities.")
        try:
            catalog = self.gateway.catalog(principal)
        except httpx.HTTPError:
            return self._fail(run, "Unable to load the capability catalog from Titan Control Plane.")
        self.repository.add_event(run.id, "progress", "Selecting a capability or execution plan.")
        from titan_agent_orchestrator.orchestration.dashboard import dashboard_route, question_with_dashboard_context

        previous_dashboard = None
        previous_capability = None
        for previous in reversed(self.repository.list_runs(conversation_id, principal.org_id, principal.user_id)):
            if previous.status == "completed" and previous.proposal and previous.proposal.capability in {"fm.get_bms_dashboard", "fm.get_energy_comparison"}:
                previous_dashboard = _first_result(previous.result or {})
                previous_capability = previous.proposal.capability
                break
        route = dashboard_route(message, previous_capability, previous_dashboard)
        dashboard_capability = next((item for item in catalog if route and item.name == route[0]), None)
        dashboard_filters = route[1] if route and dashboard_capability else None
        if dashboard_filters is not None:
            proposal = self.planner._proposal(dashboard_capability, dashboard_filters)
            if proposal.requires_confirmation:
                self.repository.update_run(run.id, "waiting_confirmation", proposal=proposal)
                self.repository.add_event(run.id, "proposal", proposal.summary, {"proposal": proposal.model_dump(mode="json")})
                self.repository.add_message(conversation_id, "assistant", proposal.summary)
                return self.repository.get_run(run.id, principal.org_id, principal.user_id)
            return self._execute(run, proposal, principal)
        plan = self.planner.plan(message, catalog) if hasattr(self.planner, "plan") else None
        if plan is not None:
            self.repository.update_run(run.id, "running", plan=plan)
            self.repository.add_event(
                run.id,
                "plan_generated",
                "PlannerAgent generated a dependency-aware plan.",
                {"plan": plan.model_dump(mode="json")},
            )
            return self._execute_plan(run, plan, principal, approval_granted=plan.status == "approved")
        proposal = self.planner.propose(message, catalog)
        if proposal is None:
            if any(term in message.casefold() for term in ("tạo yêu cầu", "tạo request", "create request")):
                response = (
                    "To create an FM request, provide the request type, title, and "
                    "supporting evidence or related equipment."
                )
            else:
                self.repository.add_event(run.id, "progress", "Generating a conversation response.")
                response = self.planner.answer_normal_chat(self.repository.list_messages(conversation_id))
            self.repository.add_message(conversation_id, "assistant", response)
            self.repository.update_run(run.id, "completed", result={"message": response})
            self.repository.add_event(run.id, "progress", "General conversation response generated.")
            return self.repository.get_run(run.id, principal.org_id, principal.user_id)
        if proposal.capability == "data.answer_question":
            arguments = dict(proposal.arguments)
            question = arguments.get('question')
            if isinstance(question, str):
                contextual = question_with_dashboard_context(question, previous_dashboard)
                if contextual != question:
                    arguments['question'] = contextual
                    proposal = proposal.model_copy(update={
                        'arguments': arguments,
                        'proposal_hash': _hash(proposal.capability, arguments),
                    })
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
                response = (
                    "Chưa có bản nháp dashboard trong hội thoại này để publish. Hãy tạo và kiểm tra bản nháp trước."
                )
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

    def decide_plan(self, run: Run, decision: str, plan_id: str, plan_version: int, principal: Principal) -> Run:
        """Approve/discard a persisted plan with an atomic one-click claim."""
        plan = run.plan
        if plan is None:
            raise ValueError("Run is not waiting for a plan decision")
        if plan.id != plan_id or plan.version != plan_version:
            raise ValueError("Decision does not match the current plan")
        if run.status != "waiting_confirmation" or plan.status != "pending_approval":
            # Repeated clicks after the atomic claim are safe and idempotent.
            if decision == "approve" and plan.status in {"approved", "running", "completed"}:
                return run
            if decision == "discard" and plan.status == "discarded":
                return run
            raise ValueError("Run is not waiting for a plan decision")
        if decision == "approve" and "agent:execute" not in principal.permissions:
            raise PermissionError("Missing agent:execute permission")
        claimed = self.repository.claim_plan_decision(
            run.id,
            principal.org_id,
            principal.user_id,
            plan_id,
            plan_version,
            "approved" if decision == "approve" else "discarded",
            "run" if decision == "approve" else "result",
        )
        if not claimed:
            # A repeated click is idempotent: return the already claimed result.
            current = self.repository.get_run(run.id, principal.org_id, principal.user_id)
            if current.plan and current.plan.status in {"approved", "running", "completed", "failed", "discarded"}:
                return current
            raise ValueError("Plan decision was already claimed")
        current = self.repository.get_run(run.id, principal.org_id, principal.user_id)
        if decision == "discard":
            self.repository.add_event(run.id, "cancelled", "The user discarded the PlannerAgent plan.")
            self.repository.add_message(
                run.conversation_id,
                "assistant",
                "Plan discarded; no protected action was called.",
            )
            return self.repository.get_run(run.id, principal.org_id, principal.user_id)
        self.repository.add_event(run.id, "progress", "Plan approved; resuming delegated tasks.")
        approved_plan = current.plan.model_copy(update={"status": "approved"}) if current.plan else None
        if approved_plan is None:
            raise ValueError("Approved plan is missing")
        return self._execute_plan(current, approved_plan, principal, approval_granted=True)

    def replan(self, run: Run, principal: Principal) -> Run:
        """Generate a bounded replacement plan after failure, preserving successes."""
        if run.plan is None or run.status != "failed":
            raise ValueError("Only a failed PlannerAgent run can be replanned")
        history = self.repository.list_messages(run.conversation_id)
        request = next((item.content for item in reversed(history) if item.role == "user"), "")
        if not request:
            raise ValueError("The original request is unavailable")
        catalog = self.gateway.catalog(principal)
        new_plan = self.planner.plan(request, catalog) if hasattr(self.planner, "plan") else None
        if new_plan is None:
            raise ValueError("PlannerAgent could not produce a replacement plan")
        completed = {
            task.id: task
            for task in run.plan.tasks
            if task.status == "completed" and task.result is not None
        }
        tasks = [
            task.model_copy(update={"status": "completed", "result": completed[task.id].result})
            if task.id in completed and task.capability == completed[task.id].capability
            else task
            for task in new_plan.tasks
        ]
        new_plan = new_plan.model_copy(
            update={"id": run.plan.id, "version": run.plan.version + 1, "tasks": tasks}
        )
        self.repository.update_run(run.id, "running", plan=new_plan, error=None, response_type="run")
        self.repository.add_event(
            run.id,
            "replanned",
            "PlannerAgent generated a replacement plan.",
            {"plan": new_plan.model_dump(mode="json")},
        )
        return self._execute_plan(
            self.repository.get_run(run.id, principal.org_id, principal.user_id),
            new_plan,
            principal,
            approval_granted=new_plan.status == "approved",
        )

    def _execute_plan(self, run: Run, plan: Plan, principal: Principal, *, approval_granted: bool) -> Run:
        tasks = [task.model_copy(deep=True) for task in plan.tasks]
        plan = plan.model_copy(update={"status": "running"})
        self.repository.update_run(run.id, "running", plan=plan, response_type="run")
        results = {task.id: task.result for task in tasks if task.status == "completed" and task.result is not None}
        max_retries = self.max_task_retries
        while True:
            progressed = False
            for index, task in enumerate(tasks):
                if task.status == "completed":
                    continue
                dependencies = [
                    next((item for item in tasks if item.id == dependency), None)
                    for dependency in task.depends_on
                ]
                if any(item is None or item.status in {"failed", "skipped"} for item in dependencies):
                    task = task.model_copy(update={"status": "skipped", "error": "A dependency failed."})
                    tasks[index] = task
                    continue
                if any(item is None or item.status != "completed" for item in dependencies):
                    continue
                if task.requires_approval and not approval_granted:
                    task = task.model_copy(update={"status": "waiting_approval"})
                    tasks[index] = task
                    pending = plan.model_copy(update={"status": "pending_approval", "tasks": tasks})
                    self.repository.update_run(
                        run.id,
                        "waiting_confirmation",
                        plan=pending,
                        response_type="approval_required",
                    )
                    self.repository.add_event(
                        run.id,
                        "approval_required",
                        "Review the plan before protected actions run.",
                        {"plan": pending.model_dump(mode="json")},
                    )
                    self.repository.add_message(
                        run.conversation_id,
                        "assistant",
                        f"Please review plan: {pending.title}",
                    )
                    return self.repository.get_run(run.id, principal.org_id, principal.user_id)
                task = task.model_copy(update={"status": "running", "attempts": task.attempts + 1, "error": None})
                tasks[index] = task
                self.repository.update_run(run.id, "running", plan=plan.model_copy(update={"tasks": tasks}))
                self.repository.add_event(
                    run.id,
                    "task_started",
                    f"Delegating {task.id} to {task.agent}.",
                    {"task": task.model_dump(mode="json")},
                )
                try:
                    arguments = _resolve_task_arguments(task.arguments, results)
                    execution = self.gateway.execute(
                        principal,
                        run.id,
                        f"agent-plan:{run.id}:{plan.id}:{plan.version}:{task.id}",
                        task.capability,
                        arguments,
                    )
                except (httpx.HTTPError, ValueError) as exc:
                    transient = isinstance(exc, httpx.HTTPError) and not isinstance(exc, httpx.HTTPStatusError)
                    if isinstance(exc, httpx.HTTPStatusError):
                        transient = exc.response.status_code >= 500
                    if transient and not task.write and task.attempts <= max_retries:
                        self.repository.add_event(run.id, "progress", f"Retrying {task.id} after a transient failure.")
                        task = task.model_copy(update={"status": "pending"})
                        tasks[index] = task
                        continue
                    failed = task.model_copy(update={"status": "failed", "error": _plan_error(task.capability, exc)})
                    tasks[index] = failed
                    failed_plan = plan.model_copy(update={"status": "failed", "tasks": tasks})
                    return self._fail_plan(run, failed_plan, failed.error or "Delegated task failed.")
                task_result = {**execution.result, "message": _humanize_result(task.capability, execution.result)}
                task = task.model_copy(update={"status": "completed", "result": task_result})
                tasks[index] = task
                results[task.id] = task_result
                plan = plan.model_copy(update={"tasks": tasks})
                self.repository.update_run(run.id, "running", plan=plan)
                self.repository.add_event(run.id, "task_finished", f"{task.id} completed.", {"task_id": task.id})
                progressed = True
            if all(task.status == "completed" for task in tasks):
                final_plan = plan.model_copy(update={"status": "completed", "tasks": tasks})
                aggregate = {
                    "plan_id": final_plan.id,
                    "tasks": {task.id: task.result for task in tasks},
                    "message": "Plan completed successfully.",
                }
                self.repository.update_run(
                    run.id,
                    "completed",
                    plan=final_plan,
                    result=aggregate,
                    response_type="result",
                )
                self.repository.add_message(run.conversation_id, "assistant", aggregate["message"])
                return self.repository.get_run(run.id, principal.org_id, principal.user_id)
            if not progressed and not any(task.status == "pending" for task in tasks):
                failed_plan = plan.model_copy(update={"status": "failed", "tasks": tasks})
                return self._fail_plan(run, failed_plan, "Plan dependencies could not be satisfied.")

    def _fail_plan(self, run: Run, plan: Plan, message: str) -> Run:
        self.repository.update_run(run.id, "failed", plan=plan, error=message, response_type="result")
        self.repository.add_event(run.id, "failed", message, {"plan": plan.model_dump(mode="json")})
        self.repository.add_message(run.conversation_id, "assistant", message)
        return self.repository.get_run(run.id, run.org_id, run.user_id)

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
        self.repository.add_event(run.id, "tool_finished", "Capability completed.", {"capability": proposal.capability})
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
    if capability in {"fm.get_bms_dashboard", "fm.get_energy_comparison"}:
        dashboard = _first_result(result) or {}
        return dashboard.get("message", "Đã cập nhật bộ lọc dashboard BMS.")
    if capability == "data.answer_question":
        answer = result.get("answer")
        return (
            answer.strip()
            if isinstance(answer, str) and answer.strip()
            else "Không tìm thấy câu trả lời từ dữ liệu được cấp quyền."
        )

    items = _items_from_result(result)
    if capability == "fm.get_data_profiling_overview":
        return _format_profiling_overview(_first_result(result))
    if capability == "fm.list_data_profiling_assets":
        return _format_profiling_assets(items)
    if capability == "fm.run_data_profiling":
        profile = _first_result(result)
        if not profile:
            return "Đã chạy đánh giá dữ liệu. Mở trang Đánh giá dữ liệu để xem kết quả chi tiết."
        profile_result = profile.get("result") if isinstance(profile.get("result"), dict) else profile
        connection_id = profile.get("connection_id") or profile_result.get("connection_id")
        table = profile.get("table") or profile_result.get("table")
        score = profile_result.get("quality_score")
        status = profile_result.get("status")
        target = "/".join(str(value) for value in (connection_id, table) if value)
        details = []
        if target:
            details.append(f"**{target}**")
        if isinstance(score, (int, float)):
            details.append(f"điểm chất lượng **{score}/100**")
        if isinstance(status, str):
            details.append(f"trạng thái **{status.upper()}**")
        return (
            "Đã lưu kết quả đánh giá"
            + (": " + " · ".join(details) if details else ".")
            + ". Mở trang Đánh giá dữ liệu để xem chi tiết."
        )
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


def _format_profiling_overview(overview: dict | None) -> str:
    """Tóm tắt an toàn từ aggregate profile; raw sample không được đưa vào chat."""
    if not overview:
        return "Không nhận được kết quả tổng hợp profiling. Bạn có thể thử lại từ trang Đánh giá dữ liệu."
    totals = overview.get("totals") if isinstance(overview.get("totals"), dict) else {}
    assets = overview.get("assets") if isinstance(overview.get("assets"), list) else []
    layer = str(overview.get("layer") or "bronze").upper()
    asset_count = totals.get("assets") if isinstance(totals.get("assets"), int) else len(assets)
    if not asset_count:
        return (
            f"Chưa có kết quả profiling cho lớp **{layer}** trong phạm vi đã chọn. "
            "Bạn có thể mở trang Đánh giá dữ liệu để chọn raw asset và chạy đánh giá."
        )
    statuses = totals.get("statuses") if isinstance(totals.get("statuses"), dict) else {}
    score = totals.get("quality_score")
    profiled_rows = totals.get("profiled_rows")
    first_line = f"**Đánh giá dữ liệu {layer}:** {asset_count} asset đã có kết quả"
    if isinstance(score, (int, float)):
        first_line += f", điểm chất lượng trung bình **{score}/100**"
    if isinstance(profiled_rows, int):
        first_line += f" trên **{profiled_rows:,}** dòng"
    first_line += "."
    status_line = " · ".join(
        f"{str(status).upper()}: {count}"
        for status, count in (
            ("pass", statuses.get("pass", 0)),
            ("warn", statuses.get("warn", 0)),
            ("fail", statuses.get("fail", 0)),
        )
    )
    findings: list[str] = []
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        profile = asset.get("result") if isinstance(asset.get("result"), dict) else {}
        if profile.get("status") not in {"warn", "fail"}:
            continue
        issues = profile.get("violations") if isinstance(profile.get("violations"), list) else []
        issue = next((item for item in issues if isinstance(item, dict) and isinstance(item.get("message"), str)), None)
        target = "/".join(str(value) for value in (asset.get("connection_id"), asset.get("table")) if value)
        level = str(profile.get("status", "warn")).upper()
        if target:
            finding = f"- **{target}** · {level}"
            if issue:
                finding += f" — {issue['message']}"
            findings.append(finding)
        if len(findings) == 3:
            break
    response = [first_line, status_line]
    if findings:
        response.extend(["", "Ưu tiên rà soát:", *findings])
    else:
        response.extend(["", "Không có asset WARN hoặc FAIL trong kết quả hiện tại."])
    response.extend(["", "Mở trang **Đánh giá dữ liệu** để xem dimension, schema và từng lượt profiling."])
    return "\n".join(response)


def _format_profiling_assets(items: list[dict]) -> str:
    if not items:
        return "Chưa tìm thấy raw asset BMS hoặc SPO có thể đánh giá trong kết nối FM MCP hiện tại."
    tables = [
        f"**{asset.get('connection_id', 'source')}**: "
        + ", ".join(
            str(table.get("table"))
            for table in asset.get("tables", [])[:5]
            if isinstance(table, dict) and table.get("table")
        )
        for asset in items
        if isinstance(asset, dict)
    ]
    tables = [item for item in tables if not item.endswith(": ")]
    if not tables:
        return "Đã tìm thấy raw asset, nhưng chưa có bảng nào sẵn sàng để đánh giá."
    return (
        "Nguồn dữ liệu có thể đánh giá:\n"
        + "\n".join(f"- {item}" for item in tables[:8])
        + "\n\nMở trang **Đánh giá dữ liệu** để chọn bảng và chạy profiling."
    )


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


def _resolve_task_arguments(arguments: dict[str, Any], results: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    """Resolve only explicit `$task_id.field` references from completed tasks."""
    def resolve(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("$"):
            path = value[1:].split(".")
            result = results.get(path[0])
            if result is None:
                raise ValueError(f"Task result reference {value} is unavailable")
            for part in path[1:]:
                if not isinstance(result, dict) or part not in result:
                    raise ValueError(f"Task result reference {value} is unavailable")
                result = result[part]
            return result
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        return value

    return resolve(arguments)


def _plan_error(capability: str, error: Exception) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        return _capability_failure_message(capability, error)
    if isinstance(error, httpx.HTTPError):
        return f"Capability {capability} did not respond; completed task results were preserved."
    return f"Capability {capability} could not be executed: {error}"


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
