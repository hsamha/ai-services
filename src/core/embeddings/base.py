from enum import StrEnum
from typing import Protocol


class Embedder(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    async def embed_text(self, text: str) -> list[float]:
        ...


class EmbeddingProvider(StrEnum):
    OPENAI = "openai"


PROVIDER_BY_MODEL: dict[str, EmbeddingProvider] = {
    "text-embedding-3-small": EmbeddingProvider.OPENAI,
    "text-embedding-3-large": EmbeddingProvider.OPENAI,
    "text-embedding-ada-002": EmbeddingProvider.OPENAI,
}
