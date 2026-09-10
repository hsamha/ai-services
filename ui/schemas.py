"""The service's contract, as the apps use it.

Imported straight from the service, not copied: the apps call its code in
process, so its own models are the ones that go in and come out. One place
for the pages to import them from.
"""

from src.core.tools.enums import FileType
from src.core.types import SearchHit
from src.features.llm.schemas import ModelInfo, ModelsResponse
from src.features.rag.constants import ChatRole
from src.features.rag.schemas import (
    AskRequest,
    AskResponse,
    ChunkResponse,
    ChunksResponse,
    DeleteDocumentResponse,
    DocumentMetadata,
    DocumentResponse,
    DocumentsResponse,
    HistoryMessage,
    IngestResponse,
    IngestTextRequest,
    SearchRequest,
    SearchResponse,
)

__all__ = [
    "AskRequest",
    "AskResponse",
    "ChatRole",
    "ChunkResponse",
    "ChunksResponse",
    "DeleteDocumentResponse",
    "DocumentMetadata",
    "DocumentResponse",
    "DocumentsResponse",
    "FileType",
    "HistoryMessage",
    "IngestResponse",
    "IngestTextRequest",
    "ModelInfo",
    "ModelsResponse",
    "SearchHit",
    "SearchRequest",
    "SearchResponse",
]
