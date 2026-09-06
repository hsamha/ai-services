from fastapi import HTTPException, status
from langchain_community.document_loaders import TextLoader

from src.core.tools.loaders.base import Loader, load_with


def _build(path: str) -> TextLoader:
    return TextLoader(path, encoding="utf-8")


class PlainTextLoader:
    """A file that is already text."""

    async def load(self, data: bytes) -> str:
        try:
            return await load_with(_build, data, ".txt")
        except RuntimeError as exc:
            # What the loader raises when the bytes are not text in this encoding.
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is not valid UTF-8 text.",
            ) from exc


def build() -> Loader:
    return PlainTextLoader()
