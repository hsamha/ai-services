from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq

from src.core.llm.base import LLM


class GroqLLM:

    def __init__(self, api_key: str, model: str) -> None:
        self._chat = ChatGroq(model=model, api_key=api_key)

    async def ask(self, prompt: str) -> str:
        content = (await self._chat.ainvoke(prompt)).content

        # A model may answer in parts rather than one string, so join what is text.
        if isinstance(content, str):
            return content
        return "".join(part for part in content if isinstance(part, str))

    def chat_model(self) -> BaseChatModel:
        """The client itself. An agent binds its tools to this."""
        return self._chat


def build(api_key: str, model: str) -> LLM:
    return GroqLLM(api_key=api_key, model=model)
