from typing import Protocol

from langchain_core.language_models import BaseChatModel


class LLM(Protocol):
    async def ask(self, prompt: str) -> str:
        ...

    def chat_model(self) -> BaseChatModel:
        """The underlying chat model, for callers that need to bind tools to it."""
        ...
