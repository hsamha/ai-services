from functools import lru_cache

from fastapi import HTTPException, status
from langchain_openai import ChatOpenAI

from src.context import get_context
from src.core.llm.base import LLM, PROVIDER_BY_MODEL, LLMProvider
from src.settings import get_settings


def get_llm() -> LLM:
    """The model for the request being handled, or the configured one."""
    context = get_context()
    model = context.llm_model or get_settings().llm_model
    return _build(context.provider_key, model)


@lru_cache(maxsize=32)
def _build(api_key: str, model: str) -> LLM:
    """One model client per key and model, kept for reuse."""
    provider = PROVIDER_BY_MODEL.get(model)

    if provider == LLMProvider.OPENAI:
        return ChatOpenAI(model=model, api_key=api_key)

    available = ", ".join(sorted(PROVIDER_BY_MODEL))
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unknown model {model!r}. Available: {available}.",
    )
