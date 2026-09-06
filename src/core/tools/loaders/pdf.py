from langchain_community.document_loaders import PyPDFLoader

from src.core.tools.loaders.base import Loader, load_with


class PdfLoader:
    """A PDF, read page by page."""

    async def load(self, data: bytes) -> str:
        return await load_with(PyPDFLoader, data, ".pdf")


def build() -> Loader:
    return PdfLoader()
