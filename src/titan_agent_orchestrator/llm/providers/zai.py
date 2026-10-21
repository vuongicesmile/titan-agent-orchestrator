"""Cấu hình z.ai dự phòng; credential độc lập với Groq."""

from titan_agent_orchestrator.llm.models import ProviderConfig

DEFAULT_BASE_URL = "https://api.z.ai/api/paas/v4"
DEFAULT_MODEL = "glm-4.7"


def build_config(api_key: str, base_url: str = "", model: str = "", timeout_s: float = 30) -> ProviderConfig:
    selected_model = model or DEFAULT_MODEL
    return ProviderConfig(
        provider="zai",
        base_url=base_url or DEFAULT_BASE_URL,
        api_key=api_key,
        model=selected_model,
        timeout_s=timeout_s,
        extra_body={"thinking": {"type": "disabled"}} if selected_model.startswith("glm-4.7") else {},
    )
