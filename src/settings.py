from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.core.vectorstores.enums import VectorStoreProvider


class Settings(BaseSettings):
    """Values read from the environment, or from a .env file beside the code."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    vector_store: VectorStoreProvider = VectorStoreProvider.QDRANT

    qdrant_url: str = "http://localhost:6333"

    chunk_size: int = 300
    chunk_overlap: int = 50

    top_k: int = 5

    max_document_tokens: int = 5000

    max_history_messages: int = 5

    max_agent_steps: int = 6


@lru_cache
def get_settings() -> Settings:
    return Settings()
