from src.core.embeddings.enums import EmbeddingProvider


PROVIDER_BY_MODEL: dict[str, EmbeddingProvider] = {
    "text-embedding-3-small": EmbeddingProvider.OPENAI,
    "text-embedding-3-large": EmbeddingProvider.OPENAI,
    "text-embedding-ada-002": EmbeddingProvider.OPENAI,
}


DIMENSIONS_BY_MODEL: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def dimensions_for(model: str) -> int | None:
    return DIMENSIONS_BY_MODEL.get(model)
