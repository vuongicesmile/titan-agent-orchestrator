"""Lỗi LLM an toàn để ghi log, không chứa response body hoặc credential."""


class LLMError(Exception):
    pass


class ProviderError(LLMError):
    pass


class InvalidResponseError(LLMError):
    pass


class LLMUnavailableError(LLMError):
    pass
