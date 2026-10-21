"""Registry cấu hình các provider được hỗ trợ."""

from . import gemini, groq, zai

PROVIDERS = {"groq": groq.build_config, "zai": zai.build_config, "gemini": gemini.build_config}
