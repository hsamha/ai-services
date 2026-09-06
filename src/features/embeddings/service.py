from src.core.embeddings.factory import get_embedding_model_name, get_embedding_model
from src.features.embeddings.schemas import EmbedResponse


async def embed(texts: list[str]) -> EmbedResponse:
    vectors = await get_embedding_model().embed_texts(texts)
    return EmbedResponse(
        vectors=vectors,
        model=get_embedding_model_name(),
        dimensions=len(vectors[0]) if vectors else 0,
    )
