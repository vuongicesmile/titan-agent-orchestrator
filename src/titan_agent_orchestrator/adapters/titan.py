"""HTTP adapter for the capability gateway hosted by Titan Control Plane."""

from __future__ import annotations

import httpx

from titan_agent_orchestrator.models import Capability, CapabilityExecution, Principal


class TitanCapabilityGateway:
    def __init__(self, base_url: str, service_token: str, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.service_token = service_token
        self.client = client or httpx.Client(base_url=self.base_url, timeout=25, trust_env=False)

    def _headers(self, principal: Principal) -> dict[str, str]:
        headers = {
            "X-Titan-Org-Id": principal.org_id,
            "X-Titan-User-Id": principal.user_id,
            "X-Titan-Permissions": ",".join(sorted(principal.permissions)),
        }
        if self.service_token:
            headers["Authorization"] = "Bearer " + self.service_token
        return headers

    def catalog(self, principal: Principal) -> list[Capability]:
        response = self.client.get("/api/internal/agent/capabilities", headers=self._headers(principal))
        response.raise_for_status()
        return [Capability.model_validate(item) for item in response.json()]

    def execute(
        self,
        principal: Principal,
        run_id: str,
        idempotency_key: str,
        capability: str,
        arguments: dict,
    ) -> CapabilityExecution:
        response = self.client.post(
            f"/api/internal/agent/capabilities/{capability}/execute",
            headers={**self._headers(principal), "Idempotency-Key": idempotency_key},
            json={"run_id": run_id, "arguments": arguments},
        )
        response.raise_for_status()
        return CapabilityExecution(capability=capability, arguments=arguments, result=response.json())
