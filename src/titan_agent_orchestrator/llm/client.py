"""Transport dùng chung và chuẩn hóa response từ các endpoint tương thích."""

from __future__ import annotations

import httpx
from pydantic import BaseModel, Field

from .exceptions import InvalidResponseError, ProviderError
from .models import LLMRequest, LLMResponse, ProviderConfig, TokenUsage


class _Message(BaseModel):
    content: str = Field(min_length=1)


class _Choice(BaseModel):
    message: _Message
    finish_reason: str | None = None


class _Usage(BaseModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class _Completion(BaseModel):
    model: str | None = None
    choices: list[_Choice] = Field(min_length=1)
    usage: _Usage = Field(default_factory=_Usage)


class LLMClient:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self._owns_client = client is None
        self.client = client or httpx.Client(trust_env=False, follow_redirects=False)

    def complete(self, config: ProviderConfig, request: LLMRequest) -> LLMResponse:
        payload = {
            **config.extra_body,
            "model": config.model,
            "messages": [message.model_dump() for message in request.messages],
            "temperature": request.temperature,
            config.token_parameter: request.max_tokens,
            "stream": False,
        }
        try:
            response = self.client.post(
                f"{config.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {config.api_key.get_secret_value()}"},
                json=payload,
                timeout=config.timeout_s,
                follow_redirects=False,
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            raise ProviderError(f"{config.provider} request timed out") from None
        except httpx.HTTPStatusError as exc:
            raise ProviderError(f"{config.provider} returned HTTP {exc.response.status_code}") from None
        except httpx.HTTPError:
            raise ProviderError(f"{config.provider} connection failed") from None
        try:
            completion = _Completion.model_validate(response.json())
            choice = completion.choices[0]
            if not choice.message.content.strip() or choice.finish_reason not in {None, "stop"}:
                raise ValueError("Incomplete LLM response")
            return LLMResponse(
                provider=config.provider,
                model=completion.model or config.model,
                content=choice.message.content,
                finish_reason=choice.finish_reason,
                usage=TokenUsage(
                    input_tokens=completion.usage.prompt_tokens,
                    output_tokens=completion.usage.completion_tokens,
                    total_tokens=(
                        completion.usage.total_tokens
                        if completion.usage.total_tokens is not None
                        else completion.usage.prompt_tokens + completion.usage.completion_tokens
                    ),
                ),
            )
        except (ValueError, TypeError):
            raise InvalidResponseError(f"{config.provider} returned an invalid completion") from None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()
