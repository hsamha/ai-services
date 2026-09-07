"""The service's contract, as this app needs it.

Deliberately a copy, not an import: the apps stand on their own and talk to the
service over HTTP only, so nothing here reaches into the service's code. What
is mirrored is only what the two apps actually read or send. If a route's shape
changes, this file is the one place to follow it.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

# What the service sends with each request, and what it calls them.
SERVICE_KEY_HEADER = "X-API-Key"
PROVIDER_KEY_HEADER = "X-AI-Provider-Key"
LLM_MODEL_HEADER = "X-LLM-Model"

MetadataValue = str | int | float | bool
Metadata = dict[str, MetadataValue]


class FileType(StrEnum):
    """What kind of document a file holds. Decides which loader reads it."""

    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"

    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"

    CSV = "csv"
    JSON = "json"

    UNKNOWN = "unknown"


class ChatRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class SearchHit(BaseModel):
    text: str
    score: float
    metadata: Metadata = Field(default_factory=dict)


# ------------------------------------------------------------------- requests


class IngestTextRequest(BaseModel):
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    document_id: str | None = None
    score_threshold: float | None = None


class HistoryMessage(BaseModel):
    role: ChatRole
    content: str = Field(min_length=1)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[HistoryMessage] = Field(default_factory=list)


# ------------------------------------------------------------------ responses


class DocumentMetadata(BaseModel):
    """What the service knows about a document, beside its text."""

    id: str
    document_id: str
    title: str
    source_type: FileType = FileType.TEXT
    content_hash: str = ""
    char_count: int
    token_count: int
    chunk_count: int = 0
    created_at: str


class IngestResponse(BaseModel):
    id: str
    document_id: str
    title: str
    source_type: FileType
    content_hash: str = ""
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


class ChunksResponse(BaseModel):
    document_id: str
    chunks: list[ChunkResponse] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit] = Field(default_factory=list)


class AskResponse(BaseModel):
    question: str
    answer: str
    model: str


class ModelInfo(BaseModel):
    """A model the service will accept in the `X-LLM-Model` header."""

    name: str
    provider: str


class ModelsResponse(BaseModel):
    """Every model that can be asked for, and the one used when none is."""

    default: str
    models: list[ModelInfo] = Field(default_factory=list)
