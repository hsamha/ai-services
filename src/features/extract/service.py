from fastapi import HTTPException, UploadFile, status

from src.core.tools import tokens
from src.core.tools.enums import FileType
from src.core.tools.loaders.registry import get_loader
from src.features.extract.schemas import ExtractResponse

_FALLBACK_NAME = "upload.pdf"


async def extract_pdf(upload: UploadFile) -> ExtractResponse:
    """The text of the uploaded PDF, page after page."""
    raw = await upload.read()
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    text = await get_loader(FileType.PDF).load(raw)
    # A PDF can carry bytes and still hold no readable text — a scanned one, say.
    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No text could be read from this PDF. It may be scanned images.",
        )

    return ExtractResponse(
        filename=upload.filename or _FALLBACK_NAME,
        text=text,
        char_count=len(text),
        token_count=await tokens.count_tokens(text),
    )
