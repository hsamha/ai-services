from src.core.llm.base import LLM
from src.core.llm.factory import current_model, get_llm
from src.features.llm.schemas import AskResponse


async def ask(prompt: str) -> AskResponse:
    """Put the prompt to the model and return its answer."""
    answer = await _answer(get_llm(), prompt)
    return AskResponse(answer=answer, model=current_model())


async def _answer(llm: LLM, prompt: str) -> str:
    """The model's reply as one string."""
    content = (await llm.ainvoke(prompt)).content

    # A model may answer in parts rather than one string, so join what is text.
    if isinstance(content, str):
        return content
    return "".join(part for part in content if isinstance(part, str))
