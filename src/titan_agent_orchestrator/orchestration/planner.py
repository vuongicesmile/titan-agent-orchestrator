"""Safe initial planner; production will delegate semantic planning to Titan's LLM gateway."""

from __future__ import annotations

import hashlib
import json

from titan_agent_orchestrator.models import Capability, Proposal


def _hash(capability: str, arguments: dict) -> str:
    canonical = json.dumps({"capability": capability, "arguments": arguments}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class IntentPlanner:
    """Routes only known, explicit intents. It never invents tool arguments from prose."""

    def propose(self, message: str, catalog: list[Capability]) -> Proposal | None:
        content = message.casefold()
        wanted = (
            "data.answer_question"
            if any(term in content for term in ("bao nhiêu", "xu hướng", "kiểm tra", "dữ liệu"))
            else None
        )
        capability = next((item for item in catalog if item.name == wanted), None)
        if capability is None:
            return None
        arguments = {"question": message}
        requires_confirmation = capability.requires_confirmation or capability.write
        return Proposal(
            capability=capability.name,
            arguments=arguments,
            summary=(
                "Đã chuẩn bị hành động FM; cần xác nhận trước khi thực thi."
                if requires_confirmation
                else "Đã chọn capability đọc dữ liệu."
            ),
            proposal_hash=_hash(capability.name, arguments),
            requires_confirmation=requires_confirmation,
        )
