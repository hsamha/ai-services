from fastapi import HTTPException, status

from src.core.types import Chunk
from src.core.vectorstores.registry import get_store
from src.features.rag.constants import CHUNKS_COLLECTION, DOCUMENTS_COLLECTION
from src.features.rag.schemas import (
    ChunkRecord,
    ChunkResponse,
    ChunksResponse,
    DocumentRecord,
    DocumentResponse,
)
from src.settings import get_settings

TOO_LARGE = "Document too large - try to decompose question and use knowledge tool"


async def get_document(document_id: str) -> DocumentResponse:
    document = await _document(document_id)

    if document.metadata.token_count > get_settings().max_document_tokens:
        return DocumentResponse(metadata=document.metadata, message=TOO_LARGE)

    return DocumentResponse(metadata=document.metadata, text=document.text)


async def get_document_chunks(
    document_id: str,
    limit: int | None = None,
    offset: int = 0,
) -> ChunksResponse:
    await _document(document_id)
    records = await _chunks_of(document_id)
    wanted = records[offset : offset + limit] if limit is not None else records[offset:]

    return ChunksResponse(
        document_id=document_id,
        chunks=[ChunkResponse.from_record(record) for record in wanted],
    )


async def expand_chunk(chunk_id: str, window: int = 1) -> ChunksResponse:
    chunk = await _chunk(chunk_id)
    records = await _chunks_of(chunk.metadata.document_id)

    first = chunk.metadata.index - window
    last = chunk.metadata.index + window
    neighbours = [record for record in records if first <= record.metadata.index <= last]

    return ChunksResponse(
        document_id=chunk.metadata.document_id,
        chunks=[ChunkResponse.from_record(record) for record in neighbours],
    )


async def _document(document_id: str) -> DocumentRecord:
    """The document record, or a 404."""
    found: list[Chunk] = await get_store().get_records(
        DOCUMENTS_COLLECTION, filters={"document_id": document_id}, limit=1
    )
    if not found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No document stored under {document_id!r}.",
        )
    return DocumentRecord.from_chunk(found[0])


async def _chunk(chunk_id: str) -> ChunkRecord:
    """One chunk by its own id, or a 404."""
    found: list[Chunk] = await get_store().get_records(
        CHUNKS_COLLECTION, filters={"id": chunk_id}, limit=1
    )
    if not found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No chunk stored under {chunk_id!r}.",
        )
    return ChunkRecord.from_chunk(found[0])


async def _chunks_of(document_id: str) -> list[ChunkRecord]:
    """Every chunk of a document, ordered by position."""
    found: list[Chunk] = await get_store().get_records(
        CHUNKS_COLLECTION, filters={"document_id": document_id}
    )
    records = [ChunkRecord.from_chunk(point) for point in found]
    # A store returns points in whatever order suits it, so ordering is ours.
    return sorted(records, key=lambda record: record.metadata.index)
