from fastapi import HTTPException, status

from src.core.llm.factory import get_text_llm_name
from src.core.types import Chunk, Metadata, SearchHit
from src.core.vectorstores.registry import get_store
from src.features.rag.constants import CHUNKS_COLLECTION, DOCUMENTS_COLLECTION
from src.features.rag.schemas import (
    AskRequest,
    AskResponse,
    ChunkRecord,
    ChunkResponse,
    ChunksResponse,
    DeleteDocumentResponse,
    DocumentRecord,
    DocumentResponse,
    DocumentsResponse,
    SearchRequest,
    SearchResponse,
)
from src.settings import get_settings

TOO_LARGE_MSG = "Document too large - try to decompose question and use knowledge tool"


async def get_document(document_id: str) -> DocumentResponse:
    document = await _document(document_id)

    if document.metadata.token_count > get_settings().max_document_tokens:
        return DocumentResponse(metadata=document.metadata, message=TOO_LARGE_MSG)

    return DocumentResponse(metadata=document.metadata, text=document.text)


async def get_document_chunks(document_id: str) -> ChunksResponse:
    await _document(document_id)
    records = await _chunks_of(document_id)

    return ChunksResponse(
        document_id=document_id,
        chunks=[ChunkResponse.from_record(record) for record in records],
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


async def search(body: SearchRequest) -> SearchResponse:
    """The chunks closest to the query, closest first."""
    filters: Metadata | None = {"document_id": body.document_id} if body.document_id else None
    hits: list[SearchHit] = await get_store().search(
        CHUNKS_COLLECTION,
        body.query,
        limit=get_settings().top_k,
        filters=filters,
        score_threshold=body.score_threshold,
    )

    return SearchResponse(query=body.query, hits=hits)


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



async def list_documents() -> DocumentsResponse:
    """Every document the store holds, newest first."""
    found: list[Chunk] = await get_store().get_records(DOCUMENTS_COLLECTION)
    documents = [DocumentRecord.from_chunk(point).metadata for point in found]
    # A store returns points in whatever order suits it, so ordering is ours.
    return DocumentsResponse(
        documents=sorted(documents, key=lambda metadata: metadata.created_at, reverse=True)
    )


async def delete_document(document_id: str) -> DeleteDocumentResponse:
    """Remove a document and every chunk of it, or a 404 if it was never stored."""
    from src.features.rag import ingest

    # Raises when there is nothing there, so a delete never quietly succeeds.
    await _document(document_id)
    await ingest.remove_records(document_id)

    return DeleteDocumentResponse(document_id=document_id, deleted=True)


async def ask(body: AskRequest) -> AskResponse:
    """Put a question to the agent, with the whole run behind the reply."""
    from src.features.rag import agent

    settled = await agent.answer(body.question, body.history)

    return AskResponse(
        question=body.question,
        answer=settled.text,
        model=get_text_llm_name(),
        transcript=settled.transcript,
    )
