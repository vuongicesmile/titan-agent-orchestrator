"""Cấu hình runtime của Orchestrator."""

import os
from dataclasses import dataclass, field


def _env(name: str, default: str = "", *, secret: bool = False):
    """Đọc env khi tạo Settings để có thể nạp lại cấu hình và kiểm thử độc lập."""
    return field(default_factory=lambda: os.getenv(name, default), repr=not secret)


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("ORCHESTRATOR_DATABASE_URL", "sqlite:///./data/orchestrator.db")
    titan_control_plane_url: str = os.getenv("TITAN_CONTROL_PLANE_URL", "http://127.0.0.1:8000")
    titan_capability_gateway_token: str = os.getenv("TITAN_CAPABILITY_GATEWAY_TOKEN", "")
    context_signing_secret: str = os.getenv("TITAN_CONTEXT_SIGNING_SECRET", "")
    allow_dev_context: bool = os.getenv("ORCHESTRATOR_ALLOW_DEV_CONTEXT", "false").lower() == "true"
    # Endpoint tương thích OpenAI chỉ dùng để lập kế hoạch intent. Key chỉ ở
    # Orchestrator và không bao giờ được chuyển đến Control Plane.
    llm_base_url: str = _env("LLM_BASE_URL")
    llm_api_key: str = _env("LLM_API_KEY", secret=True)
    llm_model: str = _env("LLM_MODEL")
    llm_timeout_s: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT_S", "30")))
    llm_provider: str = _env("LLM_PROVIDER", "groq")
    llm_fallback_provider: str = _env("LLM_FALLBACK_PROVIDER", "zai")
    groq_api_key: str = _env("GROQ_API_KEY", secret=True)
    groq_base_url: str = _env("GROQ_BASE_URL")
    groq_model: str = _env("GROQ_MODEL")
    zai_api_key: str = _env("ZAI_API_KEY", secret=True)
    zai_base_url: str = _env("ZAI_BASE_URL")
    zai_model: str = _env("ZAI_MODEL")
    gemini_api_key: str = _env("GEMINI_API_KEY", secret=True)
    gemini_base_url: str = _env("GEMINI_BASE_URL")
    gemini_model: str = _env("GEMINI_MODEL")
    planner_max_task_retries: int = field(default_factory=lambda: max(0, min(int(os.getenv("PLANNER_MAX_TASK_RETRIES", "2")), 5)))


settings = Settings()
