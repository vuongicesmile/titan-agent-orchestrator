"""Hợp đồng cấu hình, request và response độc lập với provider."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

ProviderName = Literal["groq", "zai", "gemini"]


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: ProviderName
    base_url: str
    api_key: SecretStr = Field(exclude=True, repr=False)
    model: str = Field(min_length=1)
    timeout_s: float = Field(gt=0, le=300)
    token_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @field_validator("base_url")
    @classmethod
    def normalize_url(cls, value: str) -> str:
        from urllib.parse import urlsplit

        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Provider base URL must be an HTTP(S) endpoint")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Provider base URL must not contain credentials, query, or fragment")
        return value.rstrip("/")


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    messages: list[LLMMessage] = Field(min_length=1)
    temperature: float = Field(default=0, ge=0, le=2)
    max_tokens: int = Field(default=1024, ge=1)


class TokenUsage(BaseModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class LLMResponse(BaseModel):
    provider: ProviderName
    model: str
    content: str = Field(min_length=1)
    finish_reason: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)


class RouterDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    capability: str | None = Field(pattern=r"^[a-z][a-z0-9_.-]{2,100}$")


def parse_json_object(content: str) -> dict:
    """Chấp nhận JSON thuần hoặc JSON bọc trong markdown fence."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("LLM structured response must be an object")
    return value
