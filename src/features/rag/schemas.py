
from datetime import UTC, datetime
from typing import Self
from uuid import uuid4

from pydantic import BaseModel, Field

from src.core.types import Chunk, Metadata, SearchHit
from src.core.tools.enums import FileType
from src.features.rag.constants import ChatRole


def _new_id() -> str:
    """A primary key. Random, and a UUID because that is what a store accepts."""
    return str(uuid4())


def _now() -> str:
    return datetime.now(UTC).isoformat()


class DocumentMetadata(BaseModel):
    """What is known about a document, beside its text."""

    id: str = Field(default_factory=_new_id)
    document_id: str
    title: str
    source_type: FileType = FileType.TEXT
    # A fingerprint of the text, so the same content is never stored twice.
    content_hash: str = ""
    char_count: int
    token_count: int
    chunk_count: int = 0
    created_at: str = Field(default_factory=_now)

    def to_metadata(self) -> Metadata:
        """Flatten to what the store keeps beside the text."""
        return self.model_dump(mode="json")

    @classmethod
    def from_metadata(cls, metadata: Metadata) -> Self:
        """Read back what `to_metadata` wrote."""
        return cls.model_validate(metadata)


class ChunkMetadata(BaseModel):
    """What is known about one piece of a document."""

    id: str = Field(default_factory=_new_id)
    document_id: str
    # Position in the document, counting from zero.
    index: int
    created_at: str = Field(default_factory=_now)

    def to_metadata(self) -> Metadata:
        """Flatten to what the store keeps beside the text."""
        return self.model_dump(mode="json")

    @classmethod
    def from_metadata(cls, metadata: Metadata) -> Self:
        """Read back what `to_metadata` wrote."""
        return cls.model_validate(metadata)


class DocumentRecord(BaseModel):
    """A whole document, as the documents collection holds it."""

    text: str
    metadata: DocumentMetadata

    def to_chunk(self) -> Chunk:
        """The point to store, under the record's own primary key."""
        return Chunk(
            id=self.metadata.id,
            text=self.text,
            metadata=self.metadata.to_metadata(),
        )

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> Self:
        """Rebuild a record from a stored point."""
        return cls(text=chunk.text, metadata=DocumentMetadata.from_metadata(chunk.metadata))


class ChunkRecord(BaseModel):
    """One piece of a document, as the chunks collection holds it."""

    text: str
    metadata: ChunkMetadata

    def to_chunk(self) -> Chunk:
        """The point to store, under the chunk's own id."""
        return Chunk(
            id=self.metadata.id,
            text=self.text,
            metadata=self.metadata.to_metadata(),
        )

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> Self:
        """Rebuild a piece from a stored point."""
        return cls(text=chunk.text, metadata=ChunkMetadata.from_metadata(chunk.metadata))


# --------------------------------------------------------------------- the API


class IngestTextRequest(BaseModel):
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)


class IngestResponse(BaseModel):
    id: str
    document_id: str
    title: str
    source_type: FileType
    content_hash: str
    char_count: int
    token_count: int
    chunk_count: int


class DocumentResponse(BaseModel):
    metadata: DocumentMetadata
    text: str | None = None
    message: str | None = None


class ChunkResponse(BaseModel):
    id: str
    document_id: str
    index: int
    text: str

    @classmethod
    def from_record(cls, record: "ChunkRecord") -> Self:
        return cls(
            id=record.metadata.id,
            document_id=record.metadata.document_id,
            index=record.metadata.index,
            text=record.text,
        )


class ChunksResponse(BaseModel):
    document_id: str
    chunks: list[ChunkResponse] = Field(default_factory=list)


class SearchRequest(BaseModel):
    """A question, optionally narrowed to one document."""

    query: str = Field(min_length=1)
    document_id: str | None = None
    score_threshold: float | None = None


class SearchResponse(BaseModel):
    """The chunks the query matched, closest first.

    Hits carry the store's own `SearchHit`; a chunk's id, document and index
    are in its metadata, as `ChunkMetadata` wrote them.
    """

    query: str
    hits: list[SearchHit] = Field(default_factory=list)


# -------------------------------------------------------------- what tools answer


class CurrentDateTime(BaseModel):
    """The clock, as the datetime tool reports it."""

    iso: str
    timezone: str
    # Spelled out, so a model does not have to parse the ISO string to read it.
    readable: str
    weekday: str
    utc_offset: str


class Translation(BaseModel):
    """One piece of text, put into another language."""

    text: str
    target_language: str
    source_language: str | None = None


class WebSearchResult(BaseModel):
    """What the web was asked, and what came back."""

    query: str
    answer: str


# ------------------------------------------------------------------- the agent


class HistoryMessage(BaseModel):
    role: ChatRole
    content: str = Field(min_length=1)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[HistoryMessage] = Field(default_factory=list)


class AskResponse(BaseModel):
    question: str
    answer: str
    model: str
