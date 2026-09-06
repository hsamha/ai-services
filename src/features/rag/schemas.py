"""The knowledge base contracts.

A document is stored twice. Its full text goes into the documents collection as
one record, so it can be shown back or split again later. Its pieces go into the
chunks collection, one point each, and those are what a question is matched
against.

"""

from datetime import UTC, datetime
from typing import Self

from pydantic import BaseModel, Field

from src.core.types import Chunk, Metadata
from src.features.rag.enums import SourceType


def _now() -> str:
    return datetime.now(UTC).isoformat()


class DocumentMetadata(BaseModel):
    """What is known about a document, beside its text."""

    document_id: str
    title: str
    source_type: SourceType = SourceType.TEXT
    char_count: int
    # How many chunks it was split into. Zero until the split has happened.
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

    chunk_id: str
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
        """The point to store. Its id is the document id, so re-ingesting the
        same document replaces the record instead of duplicating it."""
        return Chunk(
            id=self.metadata.document_id,
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
            id=self.metadata.chunk_id,
            text=self.text,
            metadata=self.metadata.to_metadata(),
        )

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> Self:
        """Rebuild a piece from a stored point."""
        return cls(text=chunk.text, metadata=ChunkMetadata.from_metadata(chunk.metadata))
