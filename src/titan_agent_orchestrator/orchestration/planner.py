"""Chọn capability an toàn có hỗ trợ của LLM.

LLM chỉ có thể chọn capability trong catalog. Input tool được tạo cục bộ và
tool ghi vẫn phải qua bước proposal/xác nhận.
"""

from __future__ import annotations

import hashlib
import json

import httpx

from titan_agent_orchestrator.models import Capability, Proposal
from titan_agent_orchestrator.prompts.planner import CAPABILITY_ROUTER_SYSTEM_PROMPT
from titan_agent_orchestrator.settings import Settings, settings


def _hash(capability: str, arguments: dict) -> str:
    canonical = json.dumps({"capability": capability, "arguments": arguments}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class IntentPlanner:
    """Chọn capability trong catalog mà không cho LLM thực thi trực tiếp."""

    def __init__(self, runtime_settings: Settings | None = None, client: httpx.Client | None = None) -> None:
        self.settings = runtime_settings or settings
        self.client = client or httpx.Client(timeout=self.settings.llm_timeout_s, trust_env=False)

    def _llm_enabled(self) -> bool:
        return bool(self.settings.llm_base_url and self.settings.llm_api_key and self.settings.llm_model)

    def propose(self, message: str, catalog: list[Capability]) -> Proposal | None:
        if self._llm_enabled():
            proposed = self._propose_with_llm(message, catalog)
            if proposed is not None:
                return proposed
        return self._propose_with_rules(message, catalog)

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
            response = self.client.post(
                f"{self.settings.llm_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.llm_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.settings.llm_model,
                    "messages": [
                        {"role": "system", "content": CAPABILITY_ROUTER_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {"request": message, "catalog": safe_catalog}, ensure_ascii=False
                            ),
                        },
                    ],
                    "temperature": 0,
                    "max_tokens": 180,
                },
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"].get("content") or ""
            selected = _json_object(content).get("capability")
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
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


def _json_object(content: str) -> dict:
    """Nhận JSON object kể cả khi endpoint tương thích bọc nội dung trong markdown fence."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("LLM planner response must be an object")
    return value
