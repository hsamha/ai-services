from fastapi import UploadFile

from src.core.tools import splitter, tokens
from src.core.tools.enums import FileType
from src.core.tools.loaders.registry import get_loader
from src.core.types import Chunk
from src.core.vectorstores.registry import get_store
from src.features.rag.collections import ensure_collections
from src.features.rag.constants import CHUNKS_COLLECTION, DOCUMENTS_COLLECTION
from src.features.rag.schemas import (
    ChunkMetadata,
    ChunkRecord,
    DocumentMetadata,
    DocumentRecord,
    IngestResponse,
)


async def ingest_text(document_id: str, title: str, text: str) -> IngestResponse:
    """Store text sent inline."""
    return await _store(document_id, title, text, FileType.TEXT)


async def ingest_file(
    document_id: str,
    title: str | None,
    source_type: FileType,
    upload: UploadFile,
) -> IngestResponse:
    """Store an uploaded file, reading its text with the loader for its kind."""
    text = await get_loader(source_type).load(await upload.read())
    return await _store(document_id, title or upload.filename or document_id, text, source_type)


async def _store(
    document_id: str,
    title: str,
    text: str,
    source_type: FileType,
) -> IngestResponse:
    """Split the text and write both collections."""
    # An ingest works even if the seeding at startup could not reach the store.
    await ensure_collections()

    pieces = await splitter.split(text)
    document = DocumentRecord(
        text=text,
        metadata=DocumentMetadata(
            document_id=document_id,
            title=title,
            source_type=source_type,
            char_count=len(text),
            token_count=await tokens.count_tokens(text),
            chunk_count=len(pieces),
        ),
    )
    chunks = [
        ChunkRecord(
            text=piece,
            metadata=ChunkMetadata(document_id=document_id, index=index),
        )
        for index, piece in enumerate(pieces)
    ]

    await _remove(document_id)

    store = get_store()
    await store.add_records(DOCUMENTS_COLLECTION, [document.to_chunk()])
    if chunks:
        await store.add_chunks(CHUNKS_COLLECTION, [chunk.to_chunk() for chunk in chunks])

    return IngestResponse(
        id=document.metadata.id,
        document_id=document_id,
        title=title,
        source_type=source_type,
        char_count=document.metadata.char_count,
        token_count=document.metadata.token_count,
        chunk_count=len(chunks),
    )


async def _remove(document_id: str) -> None:
    """Clear out whatever is already stored under this document id."""
    store = get_store()
    for collection in (DOCUMENTS_COLLECTION, CHUNKS_COLLECTION):
        found: list[Chunk] = await store.get_records(
            collection, filters={"document_id": document_id}
        )
        ids = [point.id for point in found if point.id is not None]
        if ids:
            await store.delete_chunks(collection, ids)
