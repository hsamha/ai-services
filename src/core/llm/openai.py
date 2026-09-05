from langchain_openai import ChatOpenAI

from src.core.llm.base import LLM


class OpenAILLM:

    def __init__(self, api_key: str, model: str) -> None:
        self._chat = ChatOpenAI(model=model, api_key=api_key)

    async def ask(self, prompt: str) -> str:
        content = (await self._chat.ainvoke(prompt)).content

        # A model may answer in parts rather than one string, so join what is text.
        if isinstance(content, str):
            return content
        return "".join(part for part in content if isinstance(part, str))


def build(api_key: str, model: str) -> LLM:
    return OpenAILLM(api_key=api_key, model=model)
