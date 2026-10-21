"""Chọn capability an toàn có hỗ trợ của LLM.

LLM chỉ có thể chọn capability trong catalog. Input tool được tạo cục bộ và
tool ghi vẫn phải qua bước proposal/xác nhận.
"""

from __future__ import annotations

import hashlib
import json

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
        if self._llm_enabled():
            proposed = self._propose_with_llm(message, catalog)
            if proposed is not None:
                return proposed
        return self._propose_with_rules(message, catalog)

    def answer_normal_chat(self, history: list[Message]) -> str:
        """Trả lời hội thoại không có capability mà không khẳng định dữ liệu chưa truy vấn."""
        if self._llm_enabled():
            messages = [{"role": "system", "content": GENERAL_CHAT_SYSTEM_PROMPT}]
            messages.extend(
                {"role": message.role, "content": message.content}
                for message in history[-12:]
            )
            try:
                response = self.llm.complete(
                    LLMRequest(messages=messages, temperature=0.2, max_tokens=500)
                )
                if response.content.strip():
                    return response.content.strip()
            except LLMError:
                pass
        return _normal_chat_fallback(history[-1].content if history else "")

    def _propose_with_rules(self, message: str, catalog: list[Capability]) -> Proposal | None:
        content = message.casefold()
        wanted = None
        arguments: dict = {}
        if any(term in content for term in ("quá hạn", "qua han")):
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
        return Proposal(
            capability=capability.name,
            arguments=arguments,
            summary=(
                "FM action prepared; confirmation is required before execution."
                if requires_confirmation
                else "Read-only data capability selected."
            ),
            proposal_hash=_hash(capability.name, arguments),
            requires_confirmation=requires_confirmation,
        )


def _safe_arguments(capability: Capability, message: str) -> dict | None:
    """Không đưa output của LLM vào input tool; chỉ mở rộng sau khi đã kiểm duyệt capability."""
    if capability.name == "data.answer_question":
        return {"question": message}
    if capability.input_schema.get("required"):
        return None
    return {}


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
        "Tôi có thể hỗ trợ trao đổi thông thường hoặc hướng dẫn cách truy vấn dữ liệu "
        "và xử lý yêu cầu FM trong Titan."
    )
