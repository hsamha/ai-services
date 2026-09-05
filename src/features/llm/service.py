from src.core.llm.factory import current_model, get_llm
from src.features.llm.schemas import AskResponse


async def ask(prompt: str) -> AskResponse:
    """Put the prompt to the model and return its answer."""
    answer = await get_llm().ask(prompt)
    return AskResponse(answer=answer, model=current_model())
