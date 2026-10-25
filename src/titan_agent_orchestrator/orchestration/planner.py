"""Chọn capability an toàn có hỗ trợ của LLM.

LLM chỉ có thể chọn capability trong catalog. Input tool được tạo cục bộ và
tool ghi vẫn phải qua bước proposal/xác nhận.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field

from titan_agent_orchestrator.llm.exceptions import LLMError
from titan_agent_orchestrator.llm.models import LLMRequest, RouterDecision, parse_json_object
from titan_agent_orchestrator.llm.router import ModelRouter
from titan_agent_orchestrator.models import Capability, Message, Plan, PlanTask, Proposal
from titan_agent_orchestrator.prompts.chat import GENERAL_CHAT_SYSTEM_PROMPT
from titan_agent_orchestrator.prompts.planner import CAPABILITY_ROUTER_SYSTEM_PROMPT, PLANNER_AGENT_SYSTEM_PROMPT
from titan_agent_orchestrator.settings import Settings, settings


class PlanDraftTask(BaseModel):
    """Strict shape accepted from an LLM before local capability validation."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    agent: str = Field(min_length=1, max_length=120)
    capability: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,100}$")
    description: str = Field(min_length=1, max_length=1000)
    arguments: dict = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list, max_length=32)
    requires_approval: bool = False


class PlanDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    intent: str = Field(min_length=1, max_length=200)
    tasks: list[PlanDraftTask] = Field(min_length=2, max_length=32)


def _hash(capability: str, arguments: dict) -> str:
    canonical = json.dumps({"capability": capability, "arguments": arguments}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class PlannerAgent:
    """Independent planner that delegates work through the existing gateway.

    The agent owns intent/plan decisions only. It never calls MCP, a database,
    or a specialized agent directly; every task remains a Control Plane
    capability invocation.
    """

    def __init__(self, runtime_settings: Settings | None = None, client: httpx.Client | None = None) -> None:
        self.settings = runtime_settings or settings
        self.llm = ModelRouter(self.settings, client)

    def _llm_enabled(self) -> bool:
        return self.llm.enabled

    def plan(self, message: str, catalog: list[Capability]) -> Plan | None:
        """Build a validated multi-task plan when the request needs one.

        A single-step request intentionally falls through to the mature legacy
        proposal path, preserving existing API behavior. LLM output is only a
        candidate; capability, dependency, argument, and approval validation
        happens locally before a plan can reach the engine.
        """
        if not catalog or not _looks_like_multi_step(message):
            return None
        draft: dict | None = None
        if self._llm_enabled():
            draft = self._plan_with_llm(message, catalog)
        if draft is None:
            draft = _fallback_plan(message, catalog)
        if draft is None:
            return None
        try:
            return _validated_plan(message, draft, catalog)
        except ValueError:
            return None

    def _plan_with_llm(self, message: str, catalog: list[Capability]) -> dict | None:
        safe_catalog = [
            {
                "name": capability.name,
                "agent": capability.agent or _agent_label(capability.name),
                "description": capability.description,
                "input_schema": capability.input_schema,
                "write": capability.write,
                "requires_approval": capability.requires_confirmation or capability.write,
            }
            for capability in catalog
        ]
        try:
            response = self.llm.complete(
                LLMRequest(
                    messages=[
                        {"role": "system", "content": PLANNER_AGENT_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {"request": message, "capabilities": safe_catalog},
                                ensure_ascii=False,
                            ),
                        },
                    ],
                    temperature=0,
                    max_tokens=1600,
                ),
                response_model=PlanDraft,
            )
            return parse_json_object(response.content)
        except (LLMError, ValueError, TypeError):
            return None

    def propose(self, message: str, catalog: list[Capability]) -> Proposal | None:
        if not catalog:
            return None
        # Profiling có các tool đọc và ghi gần tên nhau. Chọn theo rule đã kiểm
        # duyệt trước khi gọi LLM để một từ như "demo" không thể đổi câu hỏi
        # review thành mutation tạo fixture.
        if _is_profiling_request(message.casefold()):
            return self._propose_with_rules(message, catalog)
        if self._llm_enabled():
            proposed = self._propose_with_llm(message, catalog)
            if proposed is not None:
                return proposed
        return self._propose_with_rules(message, catalog)

    def answer_normal_chat(self, history: list[Message]) -> str:
        """Trả lời hội thoại không có capability mà không khẳng định dữ liệu chưa truy vấn."""
        if self._llm_enabled():
            messages = [{"role": "system", "content": GENERAL_CHAT_SYSTEM_PROMPT}]
            messages.extend({"role": message.role, "content": message.content} for message in history[-12:])
            try:
                response = self.llm.complete(LLMRequest(messages=messages, temperature=0.2, max_tokens=500))
                if response.content.strip():
                    return response.content.strip()
            except LLMError:
                pass
        return _normal_chat_fallback(history[-1].content if history else "")

    def _propose_with_rules(self, message: str, catalog: list[Capability]) -> Proposal | None:
        content = message.casefold()
        wanted = None
        arguments: dict = {}
        profiling_request = _is_profiling_request(content)
        profiling_target = _profiling_target(message)
        if (
            profiling_request
            and profiling_target
            and any(term in content for term in ("chạy", "chay", "run", "đánh giá lại", "danh gia lai", "profile"))
        ):
            wanted = "fm.run_data_profiling"
            arguments = profiling_target
        elif (
            profiling_request
            and any(
                term in content
                for term in ("nguồn", "nguon", "asset", "bảng nào", "bang nao", "table nào", "table nao")
            )
            and any(
                term in content
                for term in ("có thể", "co the", "có sẵn", "co san", "danh sách", "danh sach", "liệt kê", "liet ke")
            )
        ):
            wanted = "fm.list_data_profiling_assets"
        elif profiling_request and not any(
            term in content
            for term in (
                "tạo dashboard",
                "tao dashboard",
                "tạo báo cáo",
                "tao bao cao",
                "create dashboard",
                "create report",
            )
        ):
            wanted = "fm.get_data_profiling_overview"
            arguments = _profiling_overview_arguments(content)
        elif any(
            term in content
            for term in ("publish dashboard", "xuất bản dashboard", "xuất bản báo cáo", "publish report")
        ):
            wanted = "reports.publish_draft"
            arguments = {"session_id": "__resolve_latest__"}
        elif any(
            term in content
            for term in (
                "tạo dashboard",
                "tao dashboard",
                "tạo báo cáo",
                "tao bao cao",
                "create dashboard",
                "create report",
            )
        ):
            wanted = "reports.create_draft"
            arguments = {"prompt": message}
        elif any(
            term in content
            for term in (
                "danh sách dashboard",
                "danh sach dashboard",
                "dashboard hiện có",
                "bao cao hien co",
                "list dashboards",
                "list reports",
            )
        ):
            wanted = "reports.list"
        elif any(term in content for term in ("tạo yêu cầu", "tao yeu cau", "tạo request", "create request")):
            wanted = "fm.create_request"
            arguments = _request_arguments(message)
        elif any(
            term in content for term in ("phê duyệt chờ", "phe duyet cho", "chờ duyệt", "cho duyet", "pending approval")
        ):
            wanted = "fm.get_pending_approvals"
        elif any(term in content for term in ("quá hạn", "qua han")):
            wanted = "fm.get_overdue_requests"
        elif any(term in content for term in ("yêu cầu", "yeu cau", "request")) and any(
            term in content for term in ("mở", "mo", "danh sách", "danh sach", "gần đây", "gan day")
        ):
            wanted = "fm.list_requests"
        elif any(term in content for term in ("bao nhiêu", "xu hướng", "xu huong", "dữ liệu", "du lieu")):
            wanted = "data.answer_question"
            arguments = {"question": message}
        capability = next((item for item in catalog if item.name == wanted), None)
        if capability is None:
            return None
        return self._proposal(capability, arguments)

    def _propose_with_llm(self, message: str, catalog: list[Capability]) -> Proposal | None:
        safe_catalog = [
            {
                "name": capability.name,
                "description": capability.description,
                "required_arguments": capability.input_schema.get("required", []),
                "write": capability.write,
            }
            for capability in catalog
        ]
        try:
            response = self.llm.complete(
                LLMRequest(
                    messages=[
                        {"role": "system", "content": CAPABILITY_ROUTER_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {"request": message, "catalog": safe_catalog},
                                ensure_ascii=False,
                            ),
                        },
                    ]
                ),
                response_model=RouterDecision,
            )
            selected = RouterDecision.model_validate(parse_json_object(response.content)).capability
        except (LLMError, ValueError):
            return None
        if not isinstance(selected, str):
            return None
        capability = next((item for item in catalog if item.name == selected), None)
        if capability is None:
            return None
        arguments = _safe_arguments(capability, message)
        if arguments is None:
            return None
        return self._proposal(capability, arguments)

    @staticmethod
    def _proposal(capability: Capability, arguments: dict) -> Proposal:
        requires_confirmation = capability.requires_confirmation or capability.write
        if capability.name == "reports.create_draft":
            summary = "Đã chuẩn bị tạo bản nháp dashboard. Hãy xác nhận để Titan AI thiết kế và xem trước dữ liệu."
        elif capability.name == "reports.publish_draft":
            summary = "Đã chuẩn bị publish bản nháp dashboard gần nhất. Hãy xác nhận để chạy job publish."
        elif capability.name == "fm.run_data_profiling":
            summary = (
                "Đã chuẩn bị chạy đánh giá dữ liệu cho raw asset đã chỉ rõ. Hãy xác nhận để lưu kết quả profiling."
            )
        elif requires_confirmation:
            summary = "FM action prepared; confirmation is required before execution."
        else:
            summary = "Read-only data capability selected."
        return Proposal(
            capability=capability.name,
            arguments=arguments,
            summary=summary,
            proposal_hash=_hash(capability.name, arguments),
            requires_confirmation=requires_confirmation,
        )


# Backwards-compatible import used by existing callers and tests.
IntentPlanner = PlannerAgent


def _looks_like_multi_step(message: str) -> bool:
    content = message.casefold()
    conjunction = any(token in content for token in (" and ", " rồi ", " sau đó ", " then ", ";"))
    multi_intent = any(
        token in content
        for token in (
            "detect anomal",
            "phát hiện bất thường",
            "phat hien bat thuong",
            "tạo dashboard",
            "tao dashboard",
            "create dashboard",
            "create a dashboard",
        )
    )
    return conjunction and multi_intent or ("analyze" in content and "dashboard" in content)


def _agent_label(capability: str) -> str:
    """Display-only ownership label for legacy catalogs without `agent`."""
    namespace = capability.split(".", 1)[0]
    return {"data": "QAAgent", "reports": "ReportAgent", "fm": "FMAgent"}.get(
        namespace, f"{namespace.title()}Agent"
    )


def _fallback_plan(message: str, catalog: list[Capability]) -> dict | None:
    """Conservative local fallback when no LLM is configured or available."""
    by_name = {capability.name: capability for capability in catalog}
    tasks: list[dict] = []
    analysis = next(
        (by_name[name] for name in ("data.answer_question", "fm.get_data_profiling_overview") if name in by_name),
        None,
    )
    draft = next(
        (by_name[name] for name in ("reports.create_draft", "reports.create_report") if name in by_name), None
    )
    publish = next(
        (by_name[name] for name in ("reports.publish_draft", "reports.publish") if name in by_name), None
    )
    if analysis:
        tasks.append(
            {
                "id": "task_analysis",
                "agent": analysis.agent or _agent_label(analysis.name),
                "capability": analysis.name,
                "description": "Analyze the requested data and identify relevant anomalies.",
                "arguments": _safe_arguments(analysis, message) or {},
                "depends_on": [],
                "requires_approval": False,
            }
        )
    if draft:
        tasks.append(
            {
                "id": "task_draft",
                "agent": draft.agent or _agent_label(draft.name),
                "capability": draft.name,
                "description": "Generate a dashboard specification and preview.",
                "arguments": _safe_arguments(draft, message) or {},
                "depends_on": [tasks[-1]["id"]] if tasks else [],
                "requires_approval": draft.requires_confirmation or draft.write,
            }
        )
    if publish:
        tasks.append(
            {
                "id": "task_publish",
                "agent": publish.agent or _agent_label(publish.name),
                "capability": publish.name,
                "description": "Publish the approved dashboard to the tenant workspace.",
                "arguments": {"session_id": "$task_draft.session_id"},
                "depends_on": [tasks[-1]["id"]] if tasks else [],
                "requires_approval": True,
            }
        )
    if len(tasks) < 2:
        return None
    return {
        "title": "Planned Titan task",
        "description": "A dependency-aware plan generated from the available capabilities.",
        "intent": "multi_step_task",
        "tasks": tasks,
    }


def _validated_plan(message: str, draft: dict, catalog: list[Capability]) -> Plan:
    candidate = PlanDraft.model_validate(draft)
    capabilities = {capability.name: capability for capability in catalog}
    task_ids = {task.id for task in candidate.tasks}
    if len(task_ids) != len(candidate.tasks):
        raise ValueError("Plan task ids must be unique")
    normalized: list[PlanTask] = []
    for task in candidate.tasks:
        capability = capabilities.get(task.capability)
        if capability is None:
            raise ValueError("Plan selected a capability outside the filtered catalog")
        if task.id in task.depends_on or any(dependency not in task_ids for dependency in task.depends_on):
            raise ValueError("Plan contains an invalid dependency")
        arguments = _validated_arguments(capability, task.arguments, message)
        if arguments is None:
            raise ValueError(f"Arguments are not valid for {capability.name}")
        normalized.append(
            PlanTask(
                id=task.id,
                agent=capability.agent or _agent_label(capability.name),
                capability=capability.name,
                description=task.description,
                arguments=arguments,
                depends_on=list(dict.fromkeys(task.depends_on)),
                requires_approval=bool(task.requires_approval or capability.requires_confirmation or capability.write),
                write=capability.write,
            )
        )
    _assert_acyclic(normalized)
    canonical = {
        "title": candidate.title,
        "description": candidate.description,
        "intent": candidate.intent,
        "tasks": [task.model_dump(exclude={"status", "attempts", "result", "error"}) for task in normalized],
    }
    plan_hash = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return Plan(
        id=uuid4().hex,
        title=candidate.title,
        description=candidate.description,
        intent=candidate.intent,
        status="pending_approval" if any(task.requires_approval for task in normalized) else "approved",
        tasks=normalized,
        plan_hash=plan_hash,
    )


def _validated_arguments(capability: Capability, arguments: dict, message: str) -> dict | None:
    # Known adapters derive sensitive/side-effecting arguments locally. LLM
    # output may only provide safe references for dependent tasks.
    safe = _safe_arguments(capability, message)
    selected = safe if safe is not None else dict(arguments)
    if capability.name == "reports.publish_draft" and arguments.get("session_id"):
        selected = {"session_id": arguments["session_id"]}
    schema = capability.input_schema or {}
    required = schema.get("required", [])
    if any(key not in selected for key in required):
        return None
    properties = schema.get("properties")
    if isinstance(properties, dict) and any(key not in properties for key in selected):
        # A dependent result reference is still an argument, but its key must
        # be declared by the capability schema.
        return None
    if isinstance(properties, dict) and not _matches_schema(selected, properties):
        return None
    if not _safe_json(selected):
        return None
    return selected


def _matches_schema(arguments: dict, properties: dict) -> bool:
    """Validate the small JSON argument surface before delegation.

    The Control Plane remains authoritative, but rejecting obvious type/key
    mismatches here prevents an LLM from smuggling arbitrary tool input into a
    task. Explicit task-result references are resolved later and therefore are
    accepted for any declared property type.
    """
    for key, value in arguments.items():
        if isinstance(value, str) and value.startswith("$"):
            continue
        expected = properties.get(key, {}).get("type") if isinstance(properties.get(key), dict) else None
        if expected == "string" and not isinstance(value, str):
            return False
        if expected == "boolean" and not isinstance(value, bool):
            return False
        if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            return False
        if expected == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            return False
        if expected == "array" and not isinstance(value, list):
            return False
        if expected == "object" and not isinstance(value, dict):
            return False
    return True


def _safe_json(value: object, depth: int = 0) -> bool:
    if depth > 5:
        return False
    if value is None or isinstance(value, (str, int, float, bool)):
        return not isinstance(value, str) or len(value) <= 4000
    if isinstance(value, list):
        return len(value) <= 64 and all(_safe_json(item, depth + 1) for item in value)
    if isinstance(value, dict):
        return len(value) <= 64 and all(
            isinstance(key, str) and len(key) <= 120 and _safe_json(item, depth + 1)
            for key, item in value.items()
        )
    return False


def _assert_acyclic(tasks: Iterable[PlanTask]) -> None:
    graph = {task.id: set(task.depends_on) for task in tasks}
    resolved: set[str] = set()
    while graph:
        ready = {task_id for task_id, dependencies in graph.items() if dependencies <= resolved}
        if not ready:
            raise ValueError("Plan dependencies must be acyclic")
        resolved.update(ready)
        for task_id in ready:
            graph.pop(task_id)


def _safe_arguments(capability: Capability, message: str) -> dict | None:
    """Không đưa output của LLM vào input tool; chỉ mở rộng sau khi đã kiểm duyệt capability."""
    if capability.name == "data.answer_question":
        return {"question": message}
    if capability.name == "fm.create_request":
        return _request_arguments(message)
    if capability.name == "reports.create_draft":
        return {"prompt": message}
    if capability.name == "reports.publish_draft":
        return {"session_id": "__resolve_latest__"}
    if capability.name == "fm.get_data_profiling_overview":
        return _profiling_overview_arguments(message.casefold())
    if capability.name == "fm.run_data_profiling":
        return _profiling_target(message)
    if capability.input_schema.get("required"):
        return None
    return {}


def _is_profiling_request(content: str) -> bool:
    """Nhận diện chủ đề chất lượng raw data, không dùng output LLM làm tool input."""
    terms = (
        "data profiling",
        "profiling",
        "profile dữ liệu",
        "profile du lieu",
        "đánh giá dữ liệu",
        "danh gia du lieu",
        "chất lượng dữ liệu",
        "chat luong du lieu",
        "chất lượng raw",
        "chat luong raw",
        "raw data quality",
        "data quality",
        "dữ liệu bronze",
        "du lieu bronze",
        "bms quality",
        "spo quality",
        "raw asset",
        "bảng raw",
        "bang raw",
        "asset bms",
        "asset spo",
    )
    return any(term in content for term in terms)


def _profiling_overview_arguments(content: str) -> dict:
    """Chỉ trích xuất filter layer/thời gian từ mẫu xác định trước."""
    layer = "gold" if "gold" in content else "silver" if "silver" in content else "bronze"
    arguments: dict[str, object] = {"layer": layer}
    month = re.search(r"(?:tháng|thang|month)\s*(1[0-2]|0?[1-9])\b", content)
    year = re.search(r"(?:năm|nam|year)\s*(20\d{2}|21\d{2}|22\d{2})\b", content)
    if month:
        arguments["month"] = int(month.group(1))
    if year:
        arguments["year"] = int(year.group(1))
    if any(term in content for term in ("demo", "mô phỏng", "mo phong", "fixture")):
        arguments["include_demo"] = True
    return arguments


def _profiling_target(message: str) -> dict | None:
    """Chỉ nhận target theo dạng connection_id/table; không suy diễn từ LLM."""
    match = re.search(r"\b([a-z][a-z0-9_]{1,80})\s*/\s*([a-z][a-z0-9_.]{1,120})\b", message.casefold())
    if not match:
        return None
    return {"connection_id": match.group(1), "table": match.group(2)}


def _request_arguments(message: str) -> dict:
    """Trích xuất trường an toàn, có thể kiểm tra được từ câu tạo yêu cầu.

    Không lấy argument tool từ output của LLM. Workflow được engine chọn từ
    workflow đang active của tenant trước khi hiện hộp xác nhận.
    """
    content = message.casefold()
    priority = "medium"
    if any(term in content for term in ("khẩn", "urgent", "critical")):
        priority = "critical"
    elif any(term in content for term in ("ưu tiên cao", "cao", "high")):
        priority = "high"
    elif any(term in content for term in ("ưu tiên thấp", "thấp", "low")):
        priority = "low"
    title = message.strip()
    for prefix in (
        "tạo yêu cầu fm",
        "tạo yêu cầu",
        "tao yeu cau fm",
        "tao yeu cau",
        "create fm request",
        "create request",
    ):
        if title.casefold().startswith(prefix):
            title = title[len(prefix) :].lstrip(" :,-") or message.strip()
            break
    return {
        "workflow_definition_id": "__resolve_default__",
        "request_type": "facility_request",
        "title": title[:255],
        "description": message.strip(),
        "priority": priority,
        "context": {},
    }


def _normal_chat_fallback(message: str) -> str:
    """Câu trả lời an toàn khi provider tạm thời không phản hồi."""
    content = message.casefold()
    if "checklist" in content and any(term in content for term in ("bảo trì", "bao tri", "thiết bị", "thiet bi")):
        return (
            "Bạn có thể dùng checklist kiểm tra trước bảo trì: \n"
            "1. Xác nhận thiết bị, vị trí và phạm vi công việc.\n"
            "2. Kiểm tra điều kiện an toàn, cô lập năng lượng khi cần.\n"
            "3. Ghi nhận hiện trạng, cảnh báo và bằng chứng ảnh.\n"
            "4. Kiểm tra vật tư, dụng cụ và hướng dẫn kỹ thuật.\n"
            "5. Sau bảo trì, chạy thử, ghi kết quả và cập nhật yêu cầu FM."
        )
    if any(term in content for term in ("xin chào", "chào", "hello", "hi")):
        return "Xin chào! Tôi có thể hỗ trợ trao đổi thông thường hoặc hướng dẫn các luồng dữ liệu và FM của Titan."
    return (
        "Tôi có thể hỗ trợ trao đổi thông thường hoặc hướng dẫn cách truy vấn dữ liệu và xử lý yêu cầu FM trong Titan."
    )
