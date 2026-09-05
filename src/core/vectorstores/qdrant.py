"""Qdrant storage.

Fulfils the vector store contract against Qdrant, keeping each chunk's text and
its metadata alongside the vector.

Two connections are used on purpose, both owned by the service and shared.
Collection management runs on the async client, which is natively async.
Everything that goes through LangChain uses the sync client, because LangChain's current Qdrant integration is sync only -- its
async methods are thread-pool wrappers over the same code. Per the async rules
in CLAUDE.md, those calls are isolated behind ``anyio.to_thread.run_sync`` here
so nothing blocks the event loop.
"""

from functools import partial

from anyio import to_thread
from fastapi import HTTPException, status
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models

from src.core.types import Chunk, Metadata, SearchHit
from src.settings import get_settings

# LangChain writes the chunk's metadata under this payload key, so filters must
# address fields as "metadata.<field>".
_METADATA_KEY = "metadata"

# LangChain's Qdrant store reads and writes the unnamed (default) vector, so a
# collection created here has to declare its vector under the empty name.
_VECTOR_NAME = ""


class QdrantStore:
    """Vector storage backed by Qdrant, with embedding handled by LangChain."""

    def __init__(
        self,
        async_client: AsyncQdrantClient,
        sync_client: QdrantClient,
        embeddings: Embeddings,
        distance: models.Distance = models.Distance.COSINE,
    ) -> None:
        self._async_client = async_client
        self._sync_client = sync_client
        self._embeddings = embeddings
        self._distance = distance
        self._stores: dict[str, QdrantVectorStore] = {}
        self._vector_size: int | None = None

    # ------------------------------------------------------------------ collections

    async def create_collection(self, collection: str, *, recreate: bool = False) -> bool:
        """Create a collection. Returns False if it already existed and was kept."""
        if await self._async_client.collection_exists(collection):
            if not recreate:
                return False
            await self.delete_collection(collection)

        size = await self._dimension()
        await self._async_client.create_collection(
            collection_name=collection,
            vectors_config={
                _VECTOR_NAME: models.VectorParams(size=size, distance=self._distance)
            },
        )
        return True

    async def delete_collection(self, collection: str) -> bool:
        """Remove a collection and everything in it."""
        self._stores.pop(collection, None)
        return await self._async_client.delete_collection(collection_name=collection)

    async def collection_exists(self, collection: str) -> bool:
        """Whether the collection is present."""
        return await self._async_client.collection_exists(collection)

    async def list_collections(self) -> list[str]:
        """Every collection name currently stored."""
        response = await self._async_client.get_collections()
        return [c.name for c in response.collections]

    async def count(self, collection: str) -> int:
        """How many chunks the collection holds."""
        result = await self._async_client.count(collection_name=collection, exact=True)
        return result.count

    # ----------------------------------------------------------------------- chunks

    async def add_chunks(self, collection: str, chunks: list[Chunk]) -> list[str]:
        """Store chunks, embedding them first. Returns the stored ids."""
        if not chunks:
            return []

        store = await self._store_for(collection)
        documents = [Document(page_content=c.text, metadata=dict(c.metadata)) for c in chunks]
        ids = [c.id for c in chunks] if all(c.id for c in chunks) else None

        # LangChain's add_documents is synchronous -- run it off the event loop.
        added = await to_thread.run_sync(partial(store.add_documents, documents, ids=ids))
        return [str(i) for i in added]

    async def search(
        self,
        collection: str,
        query: str,
        *,
        limit: int | None = None,
        filters: Metadata | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchHit]:
        """Find the chunks closest to the query, optionally narrowed by metadata.

        Returns the configured number of chunks unless the caller asks for another.
        """
        store = await self._store_for(collection)
        search = partial(
            store.similarity_search_with_score,
            query,
            k=limit if limit is not None else get_settings().top_k,
            filter=build_filter(filters),
            score_threshold=score_threshold,
        )
        # Synchronous in LangChain -- run it off the event loop.
        results = await to_thread.run_sync(search)
        return [
            SearchHit(text=doc.page_content, score=score, metadata=doc.metadata)
            for doc, score in results
        ]

    async def delete_chunks(self, collection: str, ids: list[str]) -> bool:
        """Remove specific chunks by id."""
        if not ids:
            return True
        store = await self._store_for(collection)
        deleted = await to_thread.run_sync(partial(store.delete, ids=ids))
        return bool(deleted)

    # ---------------------------------------------------------------------- internals

    async def _dimension(self) -> int:
        """Vector width of the embedding model, asked once and remembered."""
        if self._vector_size is None:
            probe = await self._embeddings.aembed_query("dimension probe")
            self._vector_size = len(probe)
        return self._vector_size

    async def _store_for(self, collection: str) -> QdrantVectorStore:
        """The LangChain store bound to one collection, built once per collection."""
        cached = self._stores.get(collection)
        if cached is not None:
            return cached

        if not await self._async_client.collection_exists(collection):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection {collection!r} does not exist. Create it first.",
            )

        # Constructing the store validates the collection over the wire.
        store = await to_thread.run_sync(
            partial(
                QdrantVectorStore,
                client=self._sync_client,
                collection_name=collection,
                embedding=self._embeddings,
                vector_name=_VECTOR_NAME,
            )
        )
        self._stores[collection] = store
        return store


def build_filter(filters: Metadata | None) -> models.Filter | None:
    """Turn a plain field/value mapping into a Qdrant filter.

    Every field must match. Values are compared exactly, against the metadata
    stored with each chunk.
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

_async_client: AsyncQdrantClient | None = None
_sync_client: QdrantClient | None = None


async def connect() -> None:
    """Open the connection. Called once, when the service starts."""
    global _async_client, _sync_client
    if _async_client is None:
        url = get_settings().qdrant_url
        _async_client = AsyncQdrantClient(url=url)
        _sync_client = QdrantClient(url=url)


async def disconnect() -> None:
    """Close it again. Called once, when the service stops."""
    global _async_client, _sync_client
    if _async_client is not None:
        await _async_client.close()
        _async_client = None
    if _sync_client is not None:
        # Closing the sync client talks to the network, so keep it off the loop.
        await to_thread.run_sync(_sync_client.close)
        _sync_client = None


def build(embedder: Embeddings) -> QdrantStore:
    """A store on the open connection, embedding with the caller's model."""
    if _async_client is None or _sync_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Not connected to Qdrant.",
        )
    return QdrantStore(
        async_client=_async_client,
        sync_client=_sync_client,
        embeddings=embedder,
    )
