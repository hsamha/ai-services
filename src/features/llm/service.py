from src.core.llm.constants import PROVIDER_BY_MODEL
from src.core.llm.factory import get_text_llm, get_text_llm_name
from src.features.llm.schemas import AskResponse, ModelInfo, ModelsResponse
from src.settings import get_settings


async def ask(prompt: str) -> AskResponse:
    """Put the prompt to the model and return its answer."""
    answer = await get_text_llm().ask(prompt)
    return AskResponse(answer=answer, model=get_text_llm_name())


async def list_models() -> ModelsResponse:
    """The models a caller may name, and the one they get by default.

    Read straight from what the factory will accept, so the two cannot disagree.
    """
    models = [
        ModelInfo(name=name, provider=provider)
        for name, provider in sorted(PROVIDER_BY_MODEL.items())
    ]
    return ModelsResponse(default=get_settings().llm_model, models=models)
