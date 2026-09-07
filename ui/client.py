"""The one way the apps talk to the service.

Streamlit runs a script top to bottom on every interaction, so each call here is
opened and closed on its own: `asyncio.run(...)` from the page, one client per
call. The service's own conventions hold -- every call is async, over
`httpx.AsyncClient`.
"""

from typing import Any, Self

import httpx
from pydantic import BaseModel, Field

from ui.schemas import (
    LLM_MODEL_HEADER,
    PROVIDER_KEY_HEADER,
    SERVICE_KEY_HEADER,
    AskRequest,
    AskResponse,
    ChatRole,
    ChunksResponse,
    DocumentResponse,
    FileType,
    HistoryMessage,
    IngestResponse,
    IngestTextRequest,
    ModelsResponse,
    SearchRequest,
    SearchResponse,
)
from ui.settings import UISettings, get_ui_settings

API_PREFIX = "/api/v1"


class APIError(Exception):
    """What the service said when it refused a call."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"{status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class UploadFilePayload(BaseModel):
    """A file picked in the browser, ready to be posted."""

    filename: str
    content: bytes
    content_type: str = "application/octet-stream"


class RAGClient(BaseModel):
    """Typed calls against the rag feature's routes."""

    base_url: str
    api_key: str
    provider_key: str
    llm_model: str = ""
    timeout: float = 120.0

    @classmethod
    def from_settings(cls, settings: UISettings | None = None) -> Self:
        resolved = settings or get_ui_settings()
        return cls(
            base_url=resolved.api_base_url,
            api_key=resolved.api_key,
            provider_key=resolved.provider_key,
            llm_model=resolved.llm_model,
            timeout=resolved.request_timeout,
        )

    def _headers(self) -> dict[str, str]:
        headers = {
            SERVICE_KEY_HEADER: self.api_key,
            PROVIDER_KEY_HEADER: self.provider_key,
        }
        if self.llm_model:
            headers[LLM_MODEL_HEADER] = self.llm_model
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, str | int] | None = None,
        data: dict[str, str] | None = None,
        files: dict[str, tuple[str, bytes, str]] | None = None,
    ) -> dict[str, Any]:
        """One call, with the caller's headers on it. Raises `APIError` on refusal."""
        url = f"{self.base_url.rstrip('/')}{API_PREFIX}{path}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.request(
                method,
                url,
                headers=self._headers(),
                json=json,
                params=params,
                data=data,
                files=files,
            )
        if response.is_error:
            raise APIError(response.status_code, _detail(response))
        return response.json()

    # ------------------------------------------------------------- uploading

    async def ingest_text(self, document_id: str, title: str, text: str) -> IngestResponse:
        """Store text typed straight into the page."""
        body = IngestTextRequest(document_id=document_id, title=title, text=text)
        payload = await self._request("POST", "/rag/documents", json=body.model_dump(mode="json"))
        return IngestResponse.model_validate(payload)

    async def upload_document(
        self,
        document_id: str,
        source_type: FileType,
        upload: UploadFilePayload,
        title: str | None = None,
    ) -> IngestResponse:
        """Store a picked file, as multipart."""
        form: dict[str, str] = {"document_id": document_id, "source_type": source_type.value}
        if title:
            form["title"] = title
        payload = await self._request(
            "POST",
            "/rag/documents/upload",
            data=form,
            files={"file": (upload.filename, upload.content, upload.content_type)},
        )
        return IngestResponse.model_validate(payload)

    # -------------------------------------------------------------- reading

    async def get_document(self, document_id: str) -> DocumentResponse:
        payload = await self._request("GET", f"/rag/documents/{document_id}")
        return DocumentResponse.model_validate(payload)

    async def get_document_chunks(self, document_id: str) -> ChunksResponse:
        payload = await self._request("GET", f"/rag/documents/{document_id}/chunks")
        return ChunksResponse.model_validate(payload)

    async def expand_chunk(self, chunk_id: str, window: int = 1) -> ChunksResponse:
        payload = await self._request(
            "GET", f"/rag/chunks/{chunk_id}/expand", params={"window": window}
        )
        return ChunksResponse.model_validate(payload)

    # ------------------------------------------------------------- asking

    async def search(
        self,
        query: str,
        document_id: str | None = None,
        score_threshold: float | None = None,
    ) -> SearchResponse:
        body = SearchRequest(
            query=query, document_id=document_id, score_threshold=score_threshold
        )
        payload = await self._request("POST", "/rag/search", json=body.model_dump(mode="json"))
        return SearchResponse.model_validate(payload)

    async def ask(self, question: str, history: list[HistoryMessage] | None = None) -> AskResponse:
        body = AskRequest(question=question, history=history or [])
        payload = await self._request("POST", "/rag/ask", json=body.model_dump(mode="json"))
        return AskResponse.model_validate(payload)

    async def list_models(self) -> ModelsResponse:
        """Every chat model the service accepts, and the one it defaults to."""
        payload = await self._request("GET", "/llm/models")
        return ModelsResponse.model_validate(payload)

    # --------------------------------------------------------------- health

    async def health(self) -> bool:
        """Whether the service answers at all. Never raises."""
        url = f"{self.base_url.rstrip('/')}/health"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
        except httpx.HTTPError:
            return False
        return response.status_code == httpx.codes.OK


class ChatTurn(BaseModel):
    """One line of the conversation, as the chat page keeps it."""

    role: ChatRole
    content: str
    model: str = ""

    def to_history(self) -> HistoryMessage:
        return HistoryMessage(role=self.role, content=self.content)


class ChatState(BaseModel):
    """The whole conversation held between reruns."""

    turns: list[ChatTurn] = Field(default_factory=list)

    def history(self) -> list[HistoryMessage]:
        return [turn.to_history() for turn in self.turns]


def _detail(response: httpx.Response) -> str:
    """The service's own message, or the raw body when it sent something else."""
    try:
        payload = response.json()
    except ValueError:
        return response.text or response.reason_phrase
    if isinstance(payload, dict) and "detail" in payload:
        return str(payload["detail"])
    return response.text
