from fastapi import APIRouter

from src.context import get_context
from src.core.llm.factory import get_llm
from src.features.llm.schemas import AskRequest, AskResponse
from src.features.llm.service import ask
from src.settings import get_settings

router = APIRouter(prefix="/llm", tags=["llm"])


@router.post("/ask", response_model=AskResponse)
async def ask_model(body: AskRequest) -> AskResponse:
    """Answer a prompt with the model the caller named, or the configured one."""
    model = get_context().llm_model or get_settings().llm_model
    answer = await ask(get_llm(), body.prompt)
    return AskResponse(answer=answer, model=model)
