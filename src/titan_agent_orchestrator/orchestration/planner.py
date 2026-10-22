"""Chọn capability an toàn có hỗ trợ của LLM.

LLM chỉ có thể chọn capability trong catalog. Input tool được tạo cục bộ và
tool ghi vẫn phải qua bước proposal/xác nhận.
"""

from __future__ import annotations

import hashlib
import json
import re

import httpx

from titan_agent_orchestrator.llm.exceptions import LLMError
from titan_agent_orchestrator.llm.models import LLMRequest, RouterDecision, parse_json_object
from titan_agent_orchestrator.llm.router import ModelRouter
from titan_agent_orchestrator.models import Capability, Message, Proposal
from titan_agent_orchestrator.prompts.chat import GENERAL_CHAT_SYSTEM_PROMPT
from titan_agent_orchestrator.prompts.planner import CAPABILITY_ROUTER_SYSTEM_PROMPT
from titan_agent_orchestrator.settings import Settings, settings


def _hash(capability: str, arguments: dict) -> str:
    canonical = json.dumps({"capability": capability, "arguments": arguments}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class IntentPlanner:
    """Chọn capability trong catalog mà không cho LLM thực thi trực tiếp."""

    def __init__(self, runtime_settings: Settings | None = None, client: httpx.Client | None = None) -> None:
        self.settings = runtime_settings or settings
        self.llm = ModelRouter(self.settings, client)

    def _llm_enabled(self) -> bool:
        return self.llm.enabled

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
