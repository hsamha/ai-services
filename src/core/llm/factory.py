from collections.abc import Callable
from functools import lru_cache

from fastapi import HTTPException, status

from src.context import get_context
from src.core.llm import openai
from src.core.llm.base import LLM, PROVIDER_BY_MODEL, LLMProvider
from src.settings import get_settings

_BUILDERS: dict[LLMProvider, Callable[[str, str], LLM]] = {
    LLMProvider.OPENAI: openai.build,
}


def current_model() -> str:
    return get_context().llm_model or get_settings().llm_model


def get_llm() -> LLM:
    return _build(get_context().provider_key, current_model())


@lru_cache(maxsize=32)
def _build(api_key: str, model: str) -> LLM:
    """One model client per key and model, kept for reuse."""
    provider = PROVIDER_BY_MODEL.get(model)
    builder = _BUILDERS.get(provider) if provider else None

    if builder is None:
        available = ", ".join(sorted(PROVIDER_BY_MODEL))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown model {model!r}. Available: {available}.",
        )

    return builder(api_key, model)
