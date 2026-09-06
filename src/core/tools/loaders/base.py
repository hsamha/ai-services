from collections.abc import Callable
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

from anyio import to_thread
from langchain_core.document_loaders import BaseLoader

# What a loader module gives the registry: a LangChain loader for a path.
BuildLoader = Callable[[str], BaseLoader]


class Loader(Protocol):
    """Turns an uploaded file's bytes into its text."""

    async def load(self, data: bytes) -> str:
        """The document's text, however this kind of file has to be read."""
        ...


def _read(build: BuildLoader, data: bytes, suffix: str) -> str:
    """Write the bytes where a loader can reach them, read, then throw away."""
    with TemporaryDirectory() as folder:
        path = Path(folder) / f"upload{suffix}"
        path.write_bytes(data)
        documents = build(str(path)).load()

    # A loader returns a document per page or per section. They are one text.
    return "\n\n".join(document.page_content for document in documents).strip()


async def load_with(build: BuildLoader, data: bytes, suffix: str) -> str:
    """Run a LangChain loader over bytes.

    LangChain loaders read from a path and an upload arrives as bytes, so the
    bytes go to a temporary file. Writing and parsing are both blocking, so the
    whole of it happens off the event loop.
    """
    return await to_thread.run_sync(partial(_read, build, data, suffix))
