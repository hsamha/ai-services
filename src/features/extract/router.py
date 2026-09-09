from typing import Annotated

from fastapi import APIRouter, File, UploadFile, status

from src.features.extract import service
from src.features.extract.schemas import ExtractResponse

router = APIRouter(prefix="/extract", tags=["extract"])


@router.post(
    "/pdf",
    response_model=ExtractResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "The uploaded file is empty."},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "The PDF holds no text."},
    },
)
async def extract_pdf(file: Annotated[UploadFile, File()]) -> ExtractResponse:
    """The text of the uploaded PDF."""
    return await service.extract_pdf(file)
