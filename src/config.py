from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    """Shared configuration for the lab."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a LabConfig."""
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # Load .env file
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    # Create state and profiles directories
    state_dir = root / "state"
    state_dir.mkdir(exist_ok=True)
    (state_dir / "profiles").mkdir(exist_ok=True)

    # Resolve LLM Model Provider configuration
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    model_name = os.getenv("LLM_MODEL", "llama3.1")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    
    api_key = None
    base_url = None
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        base_url = os.getenv("ANTHROPIC_BASE_URL")
    elif provider == "ollama":
        api_key = os.getenv("OLLAMA_API_KEY")
        base_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    elif provider == "custom":
        api_key = os.getenv("CUSTOM_API_KEY")
        base_url = os.getenv("CUSTOM_BASE_URL")

    model_config = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )

    # Resolve Judge Model Provider configuration
    judge_provider = os.getenv("JUDGE_PROVIDER", provider).lower()
    judge_model_name = os.getenv("JUDGE_MODEL", model_name)
    judge_temperature = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))

    judge_api_key = None
    judge_base_url = None
    if judge_provider == "openai":
        judge_api_key = os.getenv("OPENAI_API_KEY")
        judge_base_url = os.getenv("OPENAI_BASE_URL")
    elif judge_provider == "gemini":
        judge_api_key = os.getenv("GEMINI_API_KEY")
    elif judge_provider == "anthropic":
        judge_api_key = os.getenv("ANTHROPIC_API_KEY")
        judge_base_url = os.getenv("JUDGE_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL")
    elif judge_provider == "ollama":
        judge_api_key = os.getenv("OLLAMA_API_KEY")
        judge_base_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
    elif judge_provider == "openrouter":
        judge_api_key = os.getenv("OPENROUTER_API_KEY")
        judge_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    elif judge_provider == "custom":
        judge_api_key = os.getenv("CUSTOM_API_KEY")
        judge_base_url = os.getenv("CUSTOM_BASE_URL")

    judge_config = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=judge_temperature,
        api_key=judge_api_key,
        base_url=judge_base_url,
    )

    # Read compact memory settings
    compact_threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "1000"))
    compact_keep = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=model_config,
        judge_model=judge_config,
    )
