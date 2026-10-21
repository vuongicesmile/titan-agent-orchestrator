"""Cấu hình runtime của Orchestrator."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("ORCHESTRATOR_DATABASE_URL", "sqlite:///./data/orchestrator.db")
    titan_control_plane_url: str = os.getenv("TITAN_CONTROL_PLANE_URL", "http://127.0.0.1:8000")
    titan_capability_gateway_token: str = os.getenv("TITAN_CAPABILITY_GATEWAY_TOKEN", "")
    context_signing_secret: str = os.getenv("TITAN_CONTEXT_SIGNING_SECRET", "")
    allow_dev_context: bool = os.getenv("ORCHESTRATOR_ALLOW_DEV_CONTEXT", "false").lower() == "true"
    # Endpoint tương thích OpenAI chỉ dùng để lập kế hoạch intent. Key chỉ ở
    # Orchestrator và không bao giờ được chuyển đến Control Plane.
    llm_base_url: str = os.getenv("LLM_BASE_URL", "").rstrip("/")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "")
    llm_timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "30"))


settings = Settings()
