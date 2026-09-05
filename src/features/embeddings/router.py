from fastapi import APIRouter

from src.features.embeddings import service
from src.features.embeddings.schemas import EmbedRequest, EmbedResponse

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post("", response_model=EmbedResponse)
async def embed(body: EmbedRequest) -> EmbedResponse:
    return await service.embed(body.texts)
