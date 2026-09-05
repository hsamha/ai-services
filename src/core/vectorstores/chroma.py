from functools import partial

import chromadb
from anyio import to_thread
from chromadb.api import ClientAPI
from fastapi import HTTPException, status
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.core.types import Chunk, Metadata, SearchHit
from src.core.vectorstores.base import ChromaMode
from src.settings import get_settings


class ChromaStore:
    # Chroma has no async client for the in-memory and on-disk modes, so every
    # call here goes through a thread rather than blocking the event loop.
    def __init__(self, client: ClientAPI, embeddings: Embeddings) -> None:
        self._client = client
        self._embeddings = embeddings
        self._stores: dict[str, Chroma] = {}

    # ------------------------------------------------------------------ collections

    async def create_collection(self, collection: str, *, recreate: bool = False) -> bool:
        """Create a collection. Returns False if it already existed and was kept."""
        if await self.collection_exists(collection):
            if not recreate:
                return False
            await self.delete_collection(collection)

        await to_thread.run_sync(partial(self._client.create_collection, name=collection))
        return True

    async def delete_collection(self, collection: str) -> bool:
        """Remove a collection and everything in it."""
        self._stores.pop(collection, None)
        await to_thread.run_sync(partial(self._client.delete_collection, name=collection))
        return True

    async def collection_exists(self, collection: str) -> bool:
        """Whether the collection is present."""
        return collection in await self.list_collections()

    async def list_collections(self) -> list[str]:
        """Every collection name currently stored."""
        found = await to_thread.run_sync(self._client.list_collections)
        return [str(name) for name in found]

    async def count(self, collection: str) -> int:
        """How many chunks the collection holds."""
        found = await to_thread.run_sync(
            partial(self._client.get_collection, collection, embedding_function=None)
        )
        return await to_thread.run_sync(found.count)

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
        """Find the chunks closest to the query, optionally narrowed by metadata."""
        store = await self._store_for(collection)
        search = partial(
            store.similarity_search_with_score,
            query,
            k=limit if limit is not None else get_settings().top_k,
            filter=build_filter(filters),
        )
        # Synchronous in LangChain -- run it off the event loop.
        results = await to_thread.run_sync(search)

        hits = [
            SearchHit(text=doc.page_content, score=score, metadata=doc.metadata)
            for doc, score in results
        ]
        if score_threshold is None:
            return hits

        # Chroma has no threshold of its own, and its score is a distance, so
        # anything further away than the threshold is dropped here.
        return [hit for hit in hits if hit.score <= score_threshold]

    async def delete_chunks(self, collection: str, ids: list[str]) -> bool:
        """Remove specific chunks by id."""
        if not ids:
            return True
        store = await self._store_for(collection)
        await to_thread.run_sync(partial(store.delete, ids=ids))
        return True

    # ---------------------------------------------------------------------- internals

    async def _store_for(self, collection: str) -> Chroma:
        """The LangChain store bound to one collection, built once per collection."""
        cached = self._stores.get(collection)
        if cached is not None:
            return cached

        if not await self.collection_exists(collection):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection {collection!r} does not exist. Create it first.",
            )

        store = await to_thread.run_sync(
            partial(
                Chroma,
                client=self._client,
                collection_name=collection,
                embedding_function=self._embeddings,
                create_collection_if_not_exists=False,
            )
        )
        self._stores[collection] = store
        return store


def build_filter(filters: Metadata | None) -> dict[str, object] | None:
    """Turn a plain field/value mapping into a Chroma where clause.

    Every field must match. Chroma takes a single condition as it is, but needs
    more than one wrapped in an explicit "and".
    """
    if not filters:
        return None
    if len(filters) == 1:
        field, value = next(iter(filters.items()))
        return {field: value}
    return {"$and": [{field: value} for field, value in filters.items()]}


# --------------------------------------------------------------------------- backend

_client: ClientAPI | None = None


def _open(mode: ChromaMode) -> ClientAPI:
    """The client for the configured mode."""
    settings = get_settings()

    if mode == ChromaMode.MEMORY:
        return chromadb.EphemeralClient()

    if mode == ChromaMode.PERSISTENT:
        return chromadb.PersistentClient(path=settings.chroma_path)

    return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)


async def connect() -> None:
    """Open the connection. Called once, when the service starts."""
    global _client
    if _client is None:
        # Every constructor is synchronous, and the server one opens a socket.
        _client = await to_thread.run_sync(partial(_open, get_settings().chroma_mode))


async def disconnect() -> None:
    """Close it again. Called once, when the service stops."""
    global _client
    _client = None


def build(embedder: Embeddings) -> ChromaStore:
    """A store on the open connection, embedding with the caller's model."""
    if _client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Not connected to Chroma.",
        )
    return ChromaStore(client=_client, embeddings=embedder)
