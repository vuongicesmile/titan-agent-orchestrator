"""Cấu hình Groq mặc định, dùng endpoint tương thích chat completions."""

from titan_agent_orchestrator.llm.models import ProviderConfig

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"


def build_config(api_key: str, base_url: str = "", model: str = "", timeout_s: float = 30) -> ProviderConfig:
    selected_model = model or DEFAULT_MODEL
    return ProviderConfig(
        provider="groq",
        base_url=base_url or DEFAULT_BASE_URL,
        api_key=api_key,
        model=selected_model,
        timeout_s=timeout_s,
        token_parameter="max_completion_tokens",
        extra_body={"reasoning_effort": "low"} if selected_model.startswith("openai/gpt-oss-") else {},
    )
