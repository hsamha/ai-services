from fastapi import APIRouter

from src.features.llm import service
from src.features.llm.schemas import AskRequest, AskResponse, ModelsResponse

router = APIRouter(prefix="/llm", tags=["llm"])


@router.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    """Every chat model this service accepts, for the `X-LLM-Model` header."""
    return await service.list_models()


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest) -> AskResponse:
    return await service.ask(body.prompt)
