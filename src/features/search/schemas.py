"""Question answering request and response shapes.

What callers send and receive: collections to manage, chunks to store, and a
query with the passages it matched.
"""

from pydantic import BaseModel, Field

from src.core.types import Chunk, Metadata, SearchHit


class CreateCollectionRequest(BaseModel):
    """Ask for a collection to exist."""

    name: str
    # Wipe an existing collection instead of keeping it.
    recreate: bool = False


class CollectionCreated(BaseModel):
    """Whether the collection had to be made."""

    collection: str
    created: bool


class CollectionsResponse(BaseModel):
    """Every collection the store holds."""

    collections: list[str]


class CollectionCount(BaseModel):
    """How many chunks a collection holds."""

    collection: str
    count: int


class AddChunksRequest(BaseModel):
    """Text to store, already split."""

    chunks: list[Chunk]


class AddChunksResponse(BaseModel):
    """The ids the chunks were stored under."""

    ids: list[str]


class SearchRequest(BaseModel):
    """A search, optionally narrowed to chunks whose metadata matches."""

    query: str
    limit: int | None = None
    filters: Metadata | None = None
    score_threshold: float | None = None


class SearchResponse(BaseModel):
    """What the search matched, closest first."""

    hits: list[SearchHit] = Field(default_factory=list)


class DeleteChunksRequest(BaseModel):
    """Which chunks to remove."""

    ids: list[str]


class DeleteResponse(BaseModel):
    """Whether the removal went through."""

    deleted: bool
