from typing import Protocol

from src.core.embeddings.base import Embedder
from src.core.types import Chunk, Metadata, SearchHit


class VectorStore(Protocol):
    """What every vector database implementation must provide."""

    async def create_collection(self, collection: str, recreate: bool = False) -> bool:
        """Create a collection. Returns False if it already existed and was kept."""
        ...

    async def delete_collection(self, collection: str) -> bool:
        """Remove a collection and everything in it."""
        ...

    async def collection_exists(self, collection: str) -> bool:
        """Whether the collection is present."""
        ...

    async def list_collections(self) -> list[str]:
        """Every collection name currently stored."""
        ...

    async def add_chunks(self, collection: str, chunks: list[Chunk]) -> list[str]:
        """Store chunks, embedding them first. Returns the stored ids."""
        ...

    async def search(
        self,
        collection: str,
        query: str,
        *,
        limit: int | None = None,
        filters: Metadata | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchHit]:
        """Find the chunks closest to the query, optionally narrowed by metadata."""
        ...

    async def delete_chunks(self, collection: str, ids: list[str]) -> bool:
        """Remove specific chunks by id."""
        ...

    async def close(self) -> None:
        """Release the connections."""
        ...


class VectorStoreBackend(Protocol):
    async def connect(self) -> None:
        """Open the connection. Called once, when the service starts."""
        ...

    async def disconnect(self) -> None:
        """Close it again. Called once, when the service stops."""
        ...

    async def ensure_collection(self, name: str, dimensions: int) -> bool:
        """Create a collection if it is not there"""
        ...

    def build(self, embedder: Embedder) -> VectorStore:
        """A store on the open connection, embedding with the caller's model."""
        ...

