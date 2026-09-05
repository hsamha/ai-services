from functools import partial
from uuid import uuid4

import chromadb
from anyio import to_thread
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from fastapi import HTTPException, status

from src.core.embeddings.base import Embedder
from src.core.types import Chunk, Metadata, SearchHit
from src.core.vectorstores.enums import ChromaMode
from src.settings import get_settings


class ChromaStore:
    # Chroma's client is synchronous, so every call to it goes through a thread
    # and the event loop is never blocked. Metadata is kept as plain top level
    # fields, so a filter names a field directly, and scores are distances --
    # smaller is closer.
    def __init__(self, client: ClientAPI, embedder: Embedder) -> None:
        self._client = client
        self._embedder = embedder

    # ------------------------------------------------------------------ collections

    async def create_collection(self, collection: str, *, recreate: bool = False) -> bool:
        """Create a collection. Returns False if it already existed and was kept."""
        if await self.collection_exists(collection):
            if not recreate:
                return False
            await self.delete_collection(collection)

        # We bring our own vectors, so Chroma never needs an embedding model.
        await to_thread.run_sync(
            partial(self._client.create_collection, name=collection, embedding_function=None)
        )
        return True

    async def delete_collection(self, collection: str) -> bool:
        """Remove a collection and everything in it."""
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
        found = await self._open(collection)
        return await to_thread.run_sync(found.count)

    # ----------------------------------------------------------------------- chunks

    async def add_chunks(self, collection: str, chunks: list[Chunk]) -> list[str]:
        """Store chunks, embedding them first. Returns the stored ids."""
        if not chunks:
            return []

        found = await self._open(collection)
        vectors = await self._embedder.embed_texts([chunk.text for chunk in chunks])
        ids = [chunk.id or str(uuid4()) for chunk in chunks]

        await to_thread.run_sync(
            partial(
                found.add,
                ids=ids,
                embeddings=vectors,
                documents=[chunk.text for chunk in chunks],
                # Chroma rejects an empty mapping, so a chunk without metadata
                # gets nothing rather than {}.
                metadatas=[dict(chunk.metadata) or None for chunk in chunks],
            )
        )
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
        found = await self._open(collection)
        vector = await self._embedder.embed_text(query)

        answer = await to_thread.run_sync(
            partial(
                found.query,
                query_embeddings=[vector],
                n_results=limit if limit is not None else get_settings().top_k,
                where=build_filter(filters),
                include=["documents", "metadatas", "distances"],
            )
        )

        # One query in, so one list of results out.
        documents = (answer.get("documents") or [[]])[0]
        metadatas = (answer.get("metadatas") or [[]])[0]
        distances = (answer.get("distances") or [[]])[0]

        hits = [
            SearchHit(text=str(text), score=float(distance), metadata=dict(metadata or {}))
            for text, metadata, distance in zip(documents, metadatas, distances, strict=True)
        ]
        if score_threshold is None:
            return hits

        return [hit for hit in hits if hit.score <= score_threshold]

    async def delete_chunks(self, collection: str, ids: list[str]) -> bool:
        """Remove specific chunks by id."""
        if not ids:
            return True

        found = await self._open(collection)
        await to_thread.run_sync(partial(found.delete, ids=ids))
        return True

    # ---------------------------------------------------------------------- internals

    async def _open(self, collection: str) -> Collection:
        """The collection, or a clear refusal if it was never created."""
        if not await self.collection_exists(collection):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection {collection!r} does not exist. Create it first.",
            )
        return await to_thread.run_sync(
            partial(self._client.get_collection, collection, embedding_function=None)
        )


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


def _open_client(mode: ChromaMode) -> ClientAPI:
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
        # Building a client is blocking work, so it happens off the event loop.
        _client = await to_thread.run_sync(partial(_open_client, get_settings().chroma_mode))


async def disconnect() -> None:
    """Close it again. Called once, when the service stops."""
    global _client
    _client = None


def build(embedder: Embedder) -> ChromaStore:
    """A store on the open connection, embedding with the caller's model."""
    if _client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Not connected to Chroma.",
        )
    return ChromaStore(client=_client, embedder=embedder)
