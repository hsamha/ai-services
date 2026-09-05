from functools import lru_cache

from fastapi import HTTPException, status
from langchain_openai import OpenAIEmbeddings

from src.context import get_context
from src.core.embeddings.base import PROVIDER_BY_MODEL, Embedder, EmbeddingProvider
from src.settings import get_settings


def get_embedder() -> Embedder:
    context = get_context()
    model = context.embedding_model or get_settings().embedding_model
    return _build(context.provider_key, model)


@lru_cache(maxsize=32)
def _build(api_key: str, model: str) -> Embedder:
    provider = PROVIDER_BY_MODEL.get(model)

    if provider == EmbeddingProvider.OPENAI:
        return OpenAIEmbeddings(model=model, api_key=api_key)

    available = ", ".join(sorted(PROVIDER_BY_MODEL))
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unknown embedding model {model!r}. Available: {available}.",
    )
