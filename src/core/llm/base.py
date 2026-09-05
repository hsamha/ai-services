from enum import StrEnum
from typing import Protocol


class LLM(Protocol):

    async def ask(self, prompt: str) -> str:
        ...


class LLMProvider(StrEnum):
    OPENAI = "openai"


PROVIDER_BY_MODEL: dict[str, LLMProvider] = {
    "gpt-4o-mini": LLMProvider.OPENAI,
    "gpt-4o": LLMProvider.OPENAI,
    "gpt-4.1": LLMProvider.OPENAI,
    "gpt-4.1-mini": LLMProvider.OPENAI,
    "gpt-5": LLMProvider.OPENAI,
    "gpt-5-mini": LLMProvider.OPENAI,
}
