from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Student TODO: define the provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Map aliases like `anthorpic` -> `anthropic`."""
    v = value.strip().lower()
    mapping = {
        "openai": "openai",
        "gemini": "gemini",
        "google": "gemini",
        "anthropic": "anthropic",
        "anthorpic": "anthropic",
        "ollama": "ollama",
        "openrouter": "openrouter",
        "custom": "custom"
    }
    return mapping.get(v, v)


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider."""
    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url
        )
    elif provider == "custom":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            openai_api_key=config.api_key,
            openai_api_base=config.base_url
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url
        )
    elif provider == "ollama":
        if config.api_key:
            from langchain_openai import ChatOpenAI
            base = config.base_url or "https://ollama.com/v1"
            if not base.endswith("/v1") and not base.endswith("/v1/"):
                base = base.rstrip("/") + "/v1"
            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                openai_api_key=config.api_key,
                openai_api_base=base
            )
        else:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=config.model_name,
                temperature=config.temperature,
                base_url=config.base_url or "http://localhost:11434"
            )
    elif provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter
        return ChatOpenRouter(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
