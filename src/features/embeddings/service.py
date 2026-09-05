from src.core.embeddings.factory import current_model, get_embedder
from src.features.embeddings.schemas import EmbedResponse


async def embed(texts: list[str]) -> EmbedResponse:
    vectors = await get_embedder().embed_texts(texts)
    return EmbedResponse(
        vectors=vectors,
        model=current_model(),
        dimensions=len(vectors[0]) if vectors else 0,
    )
