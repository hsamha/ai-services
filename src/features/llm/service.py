from src.core.llm.factory import get_text_llm_name, get_text_llm
from src.features.llm.schemas import AskResponse


async def ask(prompt: str) -> AskResponse:
    """Put the prompt to the model and return its answer."""
    answer = await get_text_llm().ask(prompt)
    return AskResponse(answer=answer, model=get_text_llm_name())
