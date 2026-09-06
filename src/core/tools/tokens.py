from functools import lru_cache

import tiktoken
from anyio import to_thread

MODEL = "gpt-4o"


@lru_cache
def _encoding() -> tiktoken.Encoding:
    return tiktoken.encoding_for_model(MODEL)


def _count(text: str) -> int:
    return len(_encoding().encode(text))


async def count_tokens(text: str) -> int:
    """How many tokens this text holds."""
    # Counting is CPU work, so it stays off the event loop.
    return await to_thread.run_sync(_count, text)
