"""Cấu hình Gemini qua lớp tương thích chat completions của Google."""

from titan_agent_orchestrator.llm.models import ProviderConfig

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_MODEL = "gemini-2.5-flash"


def build_config(api_key: str, base_url: str = "", model: str = "", timeout_s: float = 30) -> ProviderConfig:
    return ProviderConfig(
        provider="gemini",
        base_url=base_url or DEFAULT_BASE_URL,
        api_key=api_key,
        model=model or DEFAULT_MODEL,
        timeout_s=timeout_s,
    )
