from functools import lru_cache

from anyio import to_thread
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.settings import get_settings


@lru_cache
def _splitter() -> RecursiveCharacterTextSplitter:
    """Built once. The sizes come from configuration and do not change."""
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )


async def split(text: str) -> list[str]:
    """Split text into overlapping pieces, in document order."""
    if not text.strip():
        return []

    # Splitting is CPU work, so it stays off the event loop.
    return await to_thread.run_sync(_splitter().split_text, text)
