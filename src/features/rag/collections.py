"""The two collections the knowledge base lives in.

`rag_documents` holds one point per document -- the full text, kept whole so it
can be shown back or split again. `rag_chunks` holds one point per piece, and
those are what a question is matched against.

Both are created when the service starts, so the first upload has somewhere to
go. A collection has to be told its vector size, and no caller has arrived yet
whose key we could embed a probe string with -- so the size is read from the
embedding model instead. That model is fixed at startup, which is what makes
this possible: the collections are sized for it, and every document that ever
lands in them is embedded with it.

Nothing here ever wipes stored data: a collection that is already there is left
exactly as it is.
"""

import logging

from src.core.embeddings.constants import dimensions_for
from src.core.embeddings.factory import get_embedding_model_name
from src.core.vectorstores.registry import ensure_collection
from src.features.rag.constants import CHUNKS_COLLECTION, DOCUMENTS_COLLECTION

logger = logging.getLogger(__name__)

# Every collection the knowledge base needs before it can take an upload.
RAG_COLLECTIONS: tuple[str, ...] = (DOCUMENTS_COLLECTION, CHUNKS_COLLECTION)


async def ensure_collections() -> dict[str, bool]:
    model = get_embedding_model_name()
    dimensions = dimensions_for(model)
    if dimensions is None:
        raise RuntimeError(
            f"Vector size for embedding model {model!r} is unknown, so its "
            "collections cannot be created. Add it to DIMENSIONS_BY_MODEL."
        )

    created: dict[str, bool] = {}
    for name in RAG_COLLECTIONS:
        created[name] = await ensure_collection(name, dimensions)
        logger.info(
            "Collection %r %s.",
            name,
            f"created for {model}, {dimensions} dimensions" if created[name] else "already there",
        )

    return created


async def seed_collections() -> None:
    try:
        await ensure_collections()
    except Exception:
        logger.exception("Could not create the knowledge base collections.")
