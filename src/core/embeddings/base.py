from enum import StrEnum

from langchain_core.embeddings import Embeddings

Embedder = Embeddings

class EmbeddingProvider(StrEnum):
    OPENAI = "openai"

PROVIDER_BY_MODEL: dict[str, EmbeddingProvider] = {
    "text-embedding-3-small": EmbeddingProvider.OPENAI,
    "text-embedding-3-large": EmbeddingProvider.OPENAI,
    "text-embedding-ada-002": EmbeddingProvider.OPENAI,
}
