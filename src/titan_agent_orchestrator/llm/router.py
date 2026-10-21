"""Định tuyến planner theo provider và dự phòng một lần khi LLM thất bại."""

from __future__ import annotations

import logging

import httpx
from pydantic import BaseModel

from titan_agent_orchestrator.settings import Settings

from .client import LLMClient
from .exceptions import LLMError, LLMUnavailableError
from .models import LLMRequest, LLMResponse, ProviderConfig, parse_json_object
from .providers import PROVIDERS

logger = logging.getLogger(__name__)


class ModelRouter:
    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.providers: list[ProviderConfig] = []
        names = [settings.llm_provider, settings.llm_fallback_provider]
        for index, name in enumerate(names):
            if not name or name in names[:index]:
                continue
            if name not in PROVIDERS:
                raise ValueError(f"Unsupported LLM provider: {name}")
            # Env cũ chỉ áp dụng cho provider chính; không chuyển key sang provider dự phòng.
            legacy = index == 0
            api_key = getattr(settings, f"{name}_api_key") or (settings.llm_api_key if legacy else "")
            if not api_key:
                continue
            self.providers.append(
                PROVIDERS[name](
                    api_key=api_key,
                    base_url=getattr(settings, f"{name}_base_url") or (settings.llm_base_url if legacy else ""),
                    model=getattr(settings, f"{name}_model") or (settings.llm_model if legacy else ""),
                    timeout_s=settings.llm_timeout_s,
                )
            )
        self.client = LLMClient(client)

    @property
    def enabled(self) -> bool:
        return bool(self.providers)

    def complete(self, request: LLMRequest, response_model: type[BaseModel] | None = None) -> LLMResponse:
        for config in self.providers:
            try:
                response = self.client.complete(config, request)
                if response_model is not None:
                    response_model.model_validate(parse_json_object(response.content))
                logger.info("llm.completion_succeeded", extra={"provider": config.provider, "model": config.model})
                return response
            except (LLMError, ValueError, TypeError) as exc:
                # Không log exception body vì provider có thể echo prompt hoặc key.
                logger.warning(
                    "llm.provider_failed",
                    extra={
                        "provider": config.provider,
                        "model": config.model,
                        "error_type": type(exc).__name__,
                    },
                )
        raise LLMUnavailableError("No configured LLM provider produced a valid response")

    def close(self) -> None:
        self.client.close()
