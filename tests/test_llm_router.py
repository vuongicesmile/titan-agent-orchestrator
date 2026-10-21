from __future__ import annotations

import json
from dataclasses import fields

import httpx
import pytest
from pydantic import ValidationError

from titan_agent_orchestrator.llm.exceptions import LLMUnavailableError
from titan_agent_orchestrator.llm.models import LLMRequest, RouterDecision
from titan_agent_orchestrator.llm.router import ModelRouter
from titan_agent_orchestrator.settings import Settings

REQUEST = LLMRequest(messages=[{"role": "user", "content": "Choose a capability."}])
CONTENT = '{"capability":"data.answer_question"}'


@pytest.fixture(autouse=True)
def clear_llm_env(monkeypatch):
    for item in fields(Settings):
        if item.name.startswith(("llm_", "groq_", "zai_", "gemini_")):
            monkeypatch.delenv(item.name.upper(), raising=False)


def completion(content=CONTENT, **extra):
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}], **extra}


def test_default_groq_success_does_not_call_backup_and_normalizes_usage():
    seen = []

    def handle(request):
        seen.append(request)
        assert request.url == "https://api.groq.com/openai/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer groq-test-key"
        body = json.loads(request.content)
        assert body["model"] == "openai/gpt-oss-120b"
        assert body["reasoning_effort"] == "low"
        assert body["max_completion_tokens"] == 1024
        assert "max_tokens" not in body
        return httpx.Response(
            200,
            json=completion(
                model="actual-model",
                usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            ),
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        router = ModelRouter(Settings(groq_api_key="groq-test-key", zai_api_key="zai-test-key"), client)
        result = router.complete(REQUEST, RouterDecision)
    assert len(seen) == 1
    assert result.provider == "groq"
    assert result.model == "actual-model"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 4
    assert result.usage.total_tokens == 14
    assert "groq-test-key" not in repr(router.providers)
    assert "groq-test-key" not in json.dumps(router.providers[0].model_dump())


@pytest.mark.parametrize(
    "failure", ["timeout", "connection", "302", "429", "500", "401", "json", "schema", "empty", "length"]
)
def test_failed_groq_uses_zai_once_with_independent_key(failure, caplog):
    seen = []

    def handle(request):
        seen.append(request)
        if request.url.host == "api.groq.com":
            if failure == "timeout":
                raise httpx.ReadTimeout("groq-test-key must not be logged", request=request)
            if failure == "connection":
                raise httpx.ConnectError("secret response", request=request)
            if failure.isdigit():
                return httpx.Response(int(failure), text="groq-test-key must not be logged")
            if failure == "json":
                return httpx.Response(200, text="Internal Server Error")
            if failure == "schema":
                return httpx.Response(200, json=completion('{"capability":4}'))
            if failure == "length":
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {"message": {"content": CONTENT}, "finish_reason": "length"},
                        ]
                    },
                )
            return httpx.Response(200, json=completion(""))
        assert request.url == "https://api.z.ai/api/paas/v4/chat/completions"
        assert request.headers["Authorization"] == "Bearer zai-test-key"
        body = json.loads(request.content)
        assert body["model"] == "glm-4.7"
        assert body["max_tokens"] == 1024
        assert body["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in body
        return httpx.Response(200, json=completion())

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        router = ModelRouter(Settings(groq_api_key="groq-test-key", zai_api_key="zai-test-key"), client)
        assert router.complete(REQUEST, RouterDecision).provider == "zai"
    assert len(seen) == 2
    assert "groq-test-key" not in caplog.text
    assert "zai-test-key" not in caplog.text


def test_legacy_configuration_is_used_only_for_primary():
    router = ModelRouter(
        Settings(
            llm_base_url="https://custom.example/v1/",
            llm_model="custom-model",
            llm_api_key="legacy-key",
            zai_api_key="backup-key",
        )
    )
    try:
        assert router.providers[0].base_url == "https://custom.example/v1"
        assert router.providers[0].model == "custom-model"
        assert router.providers[1].base_url == "https://api.z.ai/api/paas/v4"
        assert router.providers[1].model == "glm-4.7"
        assert router.providers[1].api_key.get_secret_value() == "backup-key"
    finally:
        router.close()


def test_provider_specific_settings_override_legacy_and_env_is_read_at_creation(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "specific-key")
    monkeypatch.setenv("GROQ_MODEL", "specific-model")
    router = ModelRouter(Settings(llm_api_key="legacy-key", llm_model="legacy-model"))
    try:
        assert router.providers[0].api_key.get_secret_value() == "specific-key"
        assert router.providers[0].model == "specific-model"
    finally:
        router.close()


@pytest.mark.parametrize("extra", [{}, {"llm_fallback_provider": ""}])
def test_unconfigured_or_disabled_backup_is_not_called(extra):
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(503, text="unavailable")

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        router = ModelRouter(Settings(llm_api_key="legacy-key", **extra), client)
        with pytest.raises(LLMUnavailableError):
            router.complete(REQUEST)
    assert len(seen) == 1


def test_both_providers_fail_without_retry_or_credentials_in_error():
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(502, text="secret-key")

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        router = ModelRouter(Settings(groq_api_key="secret-key", zai_api_key="other-key"), client)
        with pytest.raises(LLMUnavailableError, match="No configured LLM provider") as error:
            router.complete(REQUEST)
    assert len(seen) == 2
    assert "secret-key" not in str(error.value)


def test_valid_null_decision_does_not_invoke_backup():
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(200, json=completion('{"capability":null}'))

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        router = ModelRouter(Settings(groq_api_key="key", zai_api_key="backup"), client)
        assert router.complete(REQUEST, RouterDecision).provider == "groq"
    assert len(seen) == 1


def test_gemini_can_be_selected_and_backup_can_be_disabled():
    def handle(request):
        assert request.url == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        assert request.headers["Authorization"] == "Bearer gemini-key"
        assert json.loads(request.content)["model"] == "gemini-2.5-flash"
        return httpx.Response(200, json=completion())

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        router = ModelRouter(
            Settings(
                llm_provider="gemini",
                llm_fallback_provider="",
                gemini_api_key="gemini-key",
            ),
            client,
        )
        assert router.complete(REQUEST, RouterDecision).provider == "gemini"


def test_duplicate_provider_is_attempted_once():
    router = ModelRouter(Settings(llm_fallback_provider="groq", groq_api_key="key"))
    try:
        assert len(router.providers) == 1
    finally:
        router.close()


def test_missing_primary_key_allows_configured_backup():
    router = ModelRouter(Settings(zai_api_key="backup-key"))
    try:
        assert router.enabled
        assert [config.provider for config in router.providers] == ["zai"]
    finally:
        router.close()


def test_no_keys_disables_llm():
    router = ModelRouter(Settings())
    try:
        assert not router.enabled
    finally:
        router.close()


def test_unknown_provider_fails_configuration():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        ModelRouter(Settings(llm_provider="typo"))


def test_router_decision_rejects_model_supplied_tool_arguments():
    with pytest.raises(ValidationError):
        RouterDecision.model_validate({"capability": "fm.create_request", "arguments": {"title": "invented"}})
