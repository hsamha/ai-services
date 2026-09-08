from functools import lru_cache

from anyio import to_thread
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.tools.tokens import MODEL
from src.settings import get_settings


@lru_cache
def _splitter() -> RecursiveCharacterTextSplitter:

    settings = get_settings()
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=MODEL,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def _split(text: str) -> list[str]:
    """Build the splitter and run it. Both load the encoding, so both go off the loop."""
    return _splitter().split_text(text)


async def split(text: str) -> list[str]:
    """Split text into overlapping pieces, in document order."""
    if not text.strip():
        return []

    # Splitting is CPU work, so it stays off the event loop.
    return await to_thread.run_sync(_split, text)
