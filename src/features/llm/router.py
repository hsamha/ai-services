from fastapi import APIRouter

from src.features.llm import service
from src.features.llm.schemas import AskRequest, AskResponse

router = APIRouter(prefix="/llm", tags=["llm"])


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest) -> AskResponse:
    return await service.ask(body.prompt)
