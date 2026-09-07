from collections.abc import Callable
from functools import lru_cache

from fastapi import HTTPException, status

from src.core.embeddings import openai
from src.core.embeddings.base import Embedder
from src.core.embeddings.constants import PROVIDER_BY_MODEL
from src.core.embeddings.enums import EmbeddingProvider
from src.settings import get_settings


_BUILDERS: dict[EmbeddingProvider, Callable[[str, str], Embedder]] = {
    EmbeddingProvider.OPENAI: openai.build,
}


def get_embedding_model_name() -> str:
    return get_settings().embedding_model


def get_embedding_model() -> Embedder:
    settings = get_settings()
    if not settings.embedding_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No EMBEDDING_API_KEY configured.",
        )

    return _build(settings.embedding_api_key, settings.embedding_model)


@lru_cache(maxsize=32)
def _build(api_key: str, model: str) -> Embedder:
    provider = PROVIDER_BY_MODEL.get(model)
    builder = _BUILDERS.get(provider) if provider else None

    if builder is None:
        available = ", ".join(sorted(PROVIDER_BY_MODEL))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown embedding model {model!r}. Available: {available}.",
        )

    return builder(api_key, model)
