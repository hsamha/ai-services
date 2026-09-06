from collections.abc import Callable
from functools import lru_cache

from fastapi import HTTPException, status

from src.core.tools.enums import FileType
from src.core.tools.loaders import pdf, text
from src.core.tools.loaders.base import Loader

_BUILDERS: dict[FileType, Callable[[], Loader]] = {
    FileType.TEXT: text.build,
    FileType.PDF: pdf.build,
}


def readable() -> list[str]:
    """The kinds of file that can be read today."""
    return sorted(_BUILDERS)


@lru_cache
def get_loader(file_type: FileType) -> Loader:
    """The loader for this kind of file, or a clear refusal."""
    builder = _BUILDERS.get(file_type)
    if builder is None:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Cannot read {file_type} files yet. Readable: {', '.join(readable())}.",
        )
    return builder()
