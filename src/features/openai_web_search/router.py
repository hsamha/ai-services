from fastapi import APIRouter

from src.features.openai_web_search import service
from src.features.openai_web_search.schemas import WebSearchRequest, WebSearchResponse

router = APIRouter(prefix="/openai-web-search", tags=["openai-web-search"])


@router.post("/search", response_model=WebSearchResponse)
async def search(body: WebSearchRequest) -> WebSearchResponse:
    return await service.search(body.text)
