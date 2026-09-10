from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.core.vectorstores.enums import QdrantEnvironment, VectorStoreProvider


class Settings(BaseSettings):
    """Values read from the environment, or from a .env file beside the code."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    embedding_api_key: str = ""
    vector_store: VectorStoreProvider = VectorStoreProvider.QDRANT

    qdrant_environment: QdrantEnvironment = QdrantEnvironment.LOCAL

    qdrant_url: str = "http://localhost:6333"

    qdrant_cloud_url: str = ""
    qdrant_api_key: str = ""

    chunk_size: int = 300
    chunk_overlap: int = 50

    # A document at or under this size is kept whole instead of being split.
    single_chunk_max_tokens: int = 1000

    top_k: int = 5

    max_document_tokens: int = 5000

    max_history_messages: int = 5

    max_agent_steps: int = 6

    timezone: str = "Asia/Amman"

    # Where the constitution was split to, relative to the project root.
    materials_dir: str = "data/materials"
    sections_dir: str = "data/sections"


    tool_search_knowledge_base: bool = True
    tool_read_document: bool = True
    tool_list_document_chunks: bool = True
    tool_expand_chunk: bool = True
    tool_current_datetime: bool = True
    tool_translate: bool = True
    tool_openai_web_search: bool = True
    tool_get_material: bool = True
    tool_get_section: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
