from typing import Annotated

from fastapi import APIRouter, File, Form, Query, UploadFile, status

from src.features.rag import ingest, service
from src.core.tools.enums import FileType
from src.features.rag.schemas import (
    ChunksResponse,
    DocumentResponse,
    IngestResponse,
    IngestTextRequest,
)

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/documents", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_text(body: IngestTextRequest) -> IngestResponse:
    """Store text sent inline, under the caller's own document id."""
    return await ingest.ingest_text(body.document_id, body.title, body.text)


@router.post(
    "/documents/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    document_id: Annotated[str, Form(description="The caller's own id for this document.")],
    source_type: Annotated[FileType, Form(description="What kind of file this is.")],
    file: Annotated[UploadFile, File()],
    title: Annotated[str | None, Form()] = None,
) -> IngestResponse:
    """Store an uploaded file. Its text is read out first."""
    return await ingest.ingest_file(document_id, title, source_type, file)


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str) -> DocumentResponse:
    """The whole document, unless it is too long to be read in one piece."""
    return await service.get_document(document_id)


@router.get("/documents/{document_id}/chunks", response_model=ChunksResponse)
async def get_document_chunks(
    document_id: str,
    limit: Annotated[int | None, Query(ge=1)] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ChunksResponse:
    """Every piece of the document, in the order it was split."""
    return await service.get_document_chunks(document_id, limit, offset)


@router.get("/chunks/{chunk_id}/expand", response_model=ChunksResponse)
async def expand_chunk(
    chunk_id: str,
    window: Annotated[int, Query(ge=0, le=10, description="How many either side.")] = 1,
) -> ChunksResponse:
    """The chunk, with the ones either side of it for context."""
    return await service.expand_chunk(chunk_id, window)
