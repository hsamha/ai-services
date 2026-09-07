from pydantic import BaseModel, Field

from src.core.llm.enums import LLMProvider


class AskRequest(BaseModel):
    prompt: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str
    model: str


class ModelInfo(BaseModel):
    """A model this service will accept in the `X-LLM-Model` header."""

    name: str
    provider: LLMProvider


class ModelsResponse(BaseModel):
    """Every model that can be asked for, and the one used when none is."""

    default: str
    models: list[ModelInfo] = Field(default_factory=list)
