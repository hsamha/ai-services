from langchain_openai import OpenAIEmbeddings

from src.core.embeddings.base import Embedder


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str) -> None:
        self._embeddings = OpenAIEmbeddings(model=model, api_key=api_key)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await self._embeddings.aembed_documents(texts)

    async def embed_text(self, text: str) -> list[float]:
        return await self._embeddings.aembed_query(text)


def build(api_key: str, model: str) -> Embedder:
    return OpenAIEmbedder(api_key=api_key, model=model)
