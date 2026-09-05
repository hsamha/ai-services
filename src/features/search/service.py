from src.core.vectorstores.registry import get_store
from src.features.search.schemas import (
    AddChunksRequest,
    AddChunksResponse,
    CollectionCount,
    CollectionCreated,
    CollectionsResponse,
    CreateCollectionRequest,
    DeleteChunksRequest,
    DeleteResponse,
    SearchRequest,
    SearchResponse,
)


async def create_collection(body: CreateCollectionRequest) -> CollectionCreated:
    """Create a collection, or report that it was already there."""
    created = await get_store().create_collection(body.name, recreate=body.recreate)
    return CollectionCreated(collection=body.name, created=created)


async def list_collections() -> CollectionsResponse:
    """Every collection currently stored."""
    return CollectionsResponse(collections=await get_store().list_collections())


async def delete_collection(collection: str) -> DeleteResponse:
    """Remove a collection and everything in it."""
    return DeleteResponse(deleted=await get_store().delete_collection(collection))


async def count_chunks(collection: str) -> CollectionCount:
    """How many chunks the collection holds."""
    count = await get_store().count(collection)
    return CollectionCount(collection=collection, count=count)


async def add_chunks(collection: str, body: AddChunksRequest) -> AddChunksResponse:
    """Store chunks, embedding them on the way in."""
    ids = await get_store().add_chunks(collection, body.chunks)
    return AddChunksResponse(ids=ids)


async def search(collection: str, body: SearchRequest) -> SearchResponse:
    """Find the chunks closest to the query, optionally narrowed by metadata."""
    hits = await get_store().search(
        collection,
        body.query,
        limit=body.limit,
        filters=body.filters,
        score_threshold=body.score_threshold,
    )
    return SearchResponse(hits=hits)


async def delete_chunks(collection: str, body: DeleteChunksRequest) -> DeleteResponse:
    """Remove specific chunks by id."""
    return DeleteResponse(deleted=await get_store().delete_chunks(collection, body.ids))
