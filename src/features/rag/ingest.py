from hashlib import sha256

from fastapi import HTTPException, UploadFile, status

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
from src.settings import get_settings


def _signature(raw: bytes) -> str:
    """The fingerprint of what was sent, taken over the bytes themselves."""
    return sha256(raw).hexdigest()


def _reject_empty(detail: str) -> None:
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


async def ingest_text(document_id: str, title: str, text: str) -> IngestResponse:
    """Store text sent inline."""
    if not text.strip():
        _reject_empty("The text is empty.")
    return await _store(document_id, title, text, FileType.TEXT, _signature(text.encode()))


async def ingest_file(
    document_id: str,
    title: str | None,
    source_type: FileType,
    upload: UploadFile,
) -> IngestResponse:
    """Store an uploaded file, reading its text with the loader for its kind."""
    raw = await upload.read()
    if not raw:
        _reject_empty("The uploaded file is empty.")

    text = await get_loader(source_type).load(raw)
    # A file can carry bytes and still hold no readable text — a scanned PDF, say.
    if not text.strip():
        _reject_empty("No text could be read from the uploaded file.")

    return await _store(
        document_id,
        title or upload.filename or document_id,
        text,
        source_type,
        _signature(raw),
    )


async def _store(
    document_id: str,
    title: str,
    text: str,
    source_type: FileType,
    content_hash: str,
) -> IngestResponse:
    """Split the text and write both collections."""
    # An ingest works even if the seeding at startup could not reach the store.
    await ensure_collections()

    held = await _held_with_hash(content_hash)
    if held is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"This content is already stored as {held.document_id!r} "
                f"({held.title!r})."
            ),
        )

    token_count = await tokens.count_tokens(text)
    pieces = await _pieces(text, token_count)
    document = DocumentRecord(
        text=text,
        metadata=DocumentMetadata(
            document_id=document_id,
            title=title,
            source_type=source_type,
            content_hash=content_hash,
            char_count=len(text),
            token_count=token_count,
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

    return _response(document.metadata)


def _response(metadata: DocumentMetadata) -> IngestResponse:
    """The receipt for a document that was written."""
    return IngestResponse(
        id=metadata.id,
        document_id=metadata.document_id,
        title=metadata.title,
        source_type=metadata.source_type,
        content_hash=metadata.content_hash,
        char_count=metadata.char_count,
        token_count=metadata.token_count,
        chunk_count=metadata.chunk_count,
    )


async def _held_with_hash(content_hash: str) -> DocumentMetadata | None:
    """The document already stored under this signature, if there is one."""
    found: list[Chunk] = await get_store().get_records(
        DOCUMENTS_COLLECTION, filters={"content_hash": content_hash}, limit=1
    )
    if not found:
        return None
    return DocumentMetadata.from_metadata(found[0].metadata)


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


async def _pieces(text: str, token_count: int) -> list[str]:
    """Split the text, unless it is small enough to stand as a single chunk."""
    if not text.strip():
        return []
    if token_count <= get_settings().single_chunk_max_tokens:
        return [text]
    return await splitter.split(text)
