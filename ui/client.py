"""The one way the apps talk to the service.

In process: each call goes straight to the feature's service functions, the same
ones the HTTP routes call, so nothing has to be running but the Streamlit app
(and the vector store it reads). The routes are untouched and still work on
their own.

What the HTTP layer would have done around a call is done here instead: the
request context the middleware fills in is set for the life of the call, and a
refusal -- an `HTTPException` from deep inside -- comes back as an `APIError`,
exactly as the pages already expect.
"""

import logging
from collections.abc import Awaitable, Callable
from io import BytesIO
from typing import Self, TypeVar

from fastapi import HTTPException, UploadFile, status
from pydantic import BaseModel, Field, ValidationError

from src.context import RequestContext, reset_context, set_context
from src.features.llm import service as llm_service
from src.features.rag import ingest
from src.features.rag import service as rag_service
from src.features.rag.collections import ensure_collections
from ui.schemas import (
    AskRequest,
    AskResponse,
    ChatRole,
    ChunksResponse,
    DeleteDocumentResponse,
    DocumentResponse,
    DocumentsResponse,
    FileType,
    HistoryMessage,
    IngestResponse,
    ModelsResponse,
    SearchRequest,
    SearchResponse,
)
from ui.settings import UISettings, get_ui_settings

T = TypeVar("T")

logger = logging.getLogger("ui")


class APIError(Exception):
    """What the service said when it refused a call."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"{status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class UploadFilePayload(BaseModel):
    """A file picked in the browser, ready to be stored."""

    filename: str
    content: bytes
    content_type: str = "application/octet-stream"


class RAGClient(BaseModel):
    """Typed calls into the rag feature's services."""

    provider_key: str
    llm_model: str = ""

    @classmethod
    def from_settings(cls, settings: UISettings | None = None) -> Self:
        resolved = settings or get_ui_settings()
        return cls(provider_key=resolved.provider_key, llm_model=resolved.llm_model)

    async def _call(self, work: Callable[[], Awaitable[T]]) -> T:
        """One call, inside this caller's context. Raises `APIError` on refusal.

        `work` is a thunk rather than a coroutine so that building the request
        models happens inside the `try` too -- a value the schema rejects is a
        422 here, as it would be over HTTP, not a crash of the page.
        """
        token = set_context(
            RequestContext(
                # No service key in process: there is no one to authenticate.
                api_key="",
                provider_key=self.provider_key,
                llm_model=self.llm_model or None,
            )
        )
        try:
            return await work()
        except HTTPException as error:
            raise APIError(error.status_code, str(error.detail)) from error
        except ValidationError as error:
            raise APIError(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error)) from error
        except Exception as error:
            # The same last resort as the service's error middleware.
            logger.exception("In-process call failed")
            raise APIError(
                status.HTTP_500_INTERNAL_SERVER_ERROR, f"{type(error).__name__}: {error}"
            ) from error
        finally:
            reset_context(token)

    # ------------------------------------------------------------- uploading

    async def ingest_text(self, document_id: str, title: str, text: str) -> IngestResponse:
        """Store text typed straight into the page."""
        return await self._call(lambda: ingest.ingest_text(document_id, title, text))

    async def upload_document(
        self,
        document_id: str,
        source_type: FileType,
        upload: UploadFilePayload,
        title: str | None = None,
    ) -> IngestResponse:
        """Store a picked file, handed over as the upload a route would receive."""
        file = UploadFile(file=BytesIO(upload.content), filename=upload.filename)
        return await self._call(lambda: ingest.ingest_file(document_id, title, source_type, file))

    # -------------------------------------------------------------- reading

    async def list_documents(self) -> DocumentsResponse:
        """Every document in the store, newest first."""
        return await self._call(rag_service.list_documents)

    async def get_document(self, document_id: str) -> DocumentResponse:
        return await self._call(lambda: rag_service.get_document(document_id))

    async def get_document_chunks(self, document_id: str) -> ChunksResponse:
        return await self._call(lambda: rag_service.get_document_chunks(document_id))

    async def delete_document(self, document_id: str) -> DeleteDocumentResponse:
        """Remove a document and every chunk of it."""
        return await self._call(lambda: rag_service.delete_document(document_id))

    async def expand_chunk(self, chunk_id: str, window: int = 1) -> ChunksResponse:
        return await self._call(lambda: rag_service.expand_chunk(chunk_id, window))

    # ------------------------------------------------------------- asking

    async def search(
        self,
        query: str,
        document_id: str | None = None,
        score_threshold: float | None = None,
    ) -> SearchResponse:
        return await self._call(
            lambda: rag_service.search(
                SearchRequest(
                    query=query, document_id=document_id, score_threshold=score_threshold
                )
            )
        )

    async def ask(self, question: str, history: list[HistoryMessage] | None = None) -> AskResponse:
        # Asking spends the caller's own provider key, so it cannot go without one.
        if not self.provider_key:
            raise APIError(status.HTTP_401_UNAUTHORIZED, "An AI provider key is needed to ask.")
        return await self._call(
            lambda: rag_service.ask(AskRequest(question=question, history=history or []))
        )

    async def list_models(self) -> ModelsResponse:
        """Every chat model the service accepts, and the one it defaults to."""
        return await self._call(llm_service.list_models)

    # --------------------------------------------------------------- health

    async def health(self) -> bool:
        """Whether the vector store answers and holds its collections. Never raises."""
        try:
            await self._call(ensure_collections)
        except APIError:
            return False
        return True


class ChatTurn(BaseModel):
    """One line of the conversation, as the chat page keeps it."""

    role: ChatRole
    content: str
    model: str = ""
    # The whole run as JSON. Empty on anything the user said.
    transcript: str = ""

    def to_history(self) -> HistoryMessage:
        return HistoryMessage(role=self.role, content=self.content)


class ChatState(BaseModel):
    """The whole conversation held between reruns."""

    turns: list[ChatTurn] = Field(default_factory=list)

    def history(self) -> list[HistoryMessage]:
        return [turn.to_history() for turn in self.turns]
