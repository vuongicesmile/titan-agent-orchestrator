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
        if self._llm_enabled():
            proposed = self._propose_with_llm(message, catalog)
            if proposed is not None:
                return proposed
        return self._propose_with_rules(message, catalog)

    def answer_normal_chat(self, history: list[Message]) -> str | None:
        """Trả lời hội thoại không có capability mà không khẳng định dữ liệu chưa truy vấn."""
        if not self._llm_enabled():
            return None
        messages = [{"role": "system", "content": GENERAL_CHAT_SYSTEM_PROMPT}]
        messages.extend(
            {"role": message.role, "content": message.content}
            for message in history[-12:]
        )
        try:
            response = self.llm.complete(
                LLMRequest(messages=messages, temperature=0.2, max_tokens=500)
            )
        except LLMError:
            return None
        return response.content.strip() or None

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
