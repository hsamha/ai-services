from uuid import uuid4

from fastapi import HTTPException, status
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from src.core.embeddings.base import Embedder
from src.core.types import Chunk, Metadata, SearchHit
from src.settings import get_settings

# The chunk's text and its metadata sit side by side in the payload, so a filter
# names a field as "metadata.<field>".
_TEXT_KEY = "text"
_METADATA_KEY = "metadata"


class QdrantStore:
    def __init__(
        self,
        client: AsyncQdrantClient,
        embedder: Embedder,
        distance: models.Distance = models.Distance.COSINE,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._distance = distance

    # ------------------------------------------------------------------ collections

    async def create_collection(self, collection: str, *, recreate: bool = False) -> bool:
        """Create a collection. Returns False if it already existed and was kept."""
        if await self._client.collection_exists(collection):
            if not recreate:
                return False
            await self.delete_collection(collection)

        probe = await self._embedder.embed_text("dimension probe")
        await self._client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=len(probe), distance=self._distance),
        )
        return True

    async def delete_collection(self, collection: str) -> bool:
        """Remove a collection and everything in it."""
        return await self._client.delete_collection(collection_name=collection)

    async def collection_exists(self, collection: str) -> bool:
        """Whether the collection is present."""
        return await self._client.collection_exists(collection)

    async def list_collections(self) -> list[str]:
        """Every collection name currently stored."""
        response = await self._client.get_collections()
        return [found.name for found in response.collections]

    async def count(self, collection: str) -> int:
        """How many chunks the collection holds."""
        await self._require(collection)
        result = await self._client.count(collection_name=collection, exact=True)
        return result.count

    # ----------------------------------------------------------------------- chunks

    async def add_chunks(self, collection: str, chunks: list[Chunk]) -> list[str]:
        """Store chunks, embedding them first. Returns the stored ids."""
        if not chunks:
            return []

        await self._require(collection)
        vectors = await self._embedder.embed_texts([chunk.text for chunk in chunks])

        # Qdrant only accepts a UUID or an unsigned integer as a point id.
        ids = [chunk.id or str(uuid4()) for chunk in chunks]
        points = [
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={_TEXT_KEY: chunk.text, _METADATA_KEY: dict(chunk.metadata)},
            )
            for point_id, chunk, vector in zip(ids, chunks, vectors, strict=True)
        ]

        await self._client.upsert(collection_name=collection, points=points)
        return ids

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
        await self._require(collection)
        vector = await self._embedder.embed_text(query)

        found = await self._client.query_points(
            collection_name=collection,
            query=vector,
            limit=limit if limit is not None else get_settings().top_k,
            query_filter=build_filter(filters),
            score_threshold=score_threshold,
            with_payload=True,
        )

        return [
            SearchHit(
                text=str(point.payload.get(_TEXT_KEY, "")),
                score=point.score,
                metadata=point.payload.get(_METADATA_KEY, {}),
            )
            for point in found.points
            if point.payload is not None
        ]

    async def delete_chunks(self, collection: str, ids: list[str]) -> bool:
        """Remove specific chunks by id."""
        if not ids:
            return True

        await self._require(collection)
        result = await self._client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=list(ids)),
        )
        return result.status == models.UpdateStatus.COMPLETED

    # ---------------------------------------------------------------------- internals

    async def _require(self, collection: str) -> None:
        """Refuse to work on a collection that was never created."""
        if not await self._client.collection_exists(collection):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection {collection!r} does not exist. Create it first.",
            )


def build_filter(filters: Metadata | None) -> models.Filter | None:
    """Turn a plain field/value mapping into a Qdrant filter.

    Every field must match, compared exactly against the chunk's metadata.
    """
    if not filters:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(
                key=f"{_METADATA_KEY}.{field}",
                match=models.MatchValue(value=value),
            )
            for field, value in filters.items()
        ]
    )


# --------------------------------------------------------------------------- backend

_client: AsyncQdrantClient | None = None


async def connect() -> None:
    """Open the connection. Called once, when the service starts."""
    global _client
    if _client is None:
        _client = AsyncQdrantClient(url=get_settings().qdrant_url)


async def disconnect() -> None:
    """Close it again. Called once, when the service stops."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None


def build(embedder: Embedder) -> QdrantStore:
    """A store on the open connection, embedding with the caller's model."""
    if _client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Not connected to Qdrant.",
        )
    return QdrantStore(client=_client, embedder=embedder)


async def ensure_collection(name: str, dimensions: int) -> bool:
    """Create a collection if it is not there. Returns whether this call made it.

    For the collections the service creates at startup. No caller has arrived
    yet, so there is no key to embed a probe string with -- the vector size is
    given rather than measured. An existing collection is left as it is.
    """
    if _client is None:
        raise RuntimeError("Not connected to Qdrant.")

    if await _client.collection_exists(name):
        return False

    await _client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=dimensions, distance=models.Distance.COSINE),
    )
    return True
