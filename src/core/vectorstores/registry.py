from src.core.embeddings.factory import get_embedding_model
from src.core.vectorstores import qdrant
from src.core.vectorstores.base import VectorStore, VectorStoreBackend
from src.core.vectorstores.enums import VectorStoreProvider
from src.settings import get_settings


_BACKENDS: dict[VectorStoreProvider, VectorStoreBackend] = {
    VectorStoreProvider.QDRANT: qdrant,
}


def _backend() -> VectorStoreBackend:
    """The backend this deployment is configured to use."""
    configured = get_settings().vector_store
    backend = _BACKENDS.get(configured)
    if backend is None:
        available = ", ".join(_BACKENDS)
        raise RuntimeError(f"Unknown vector store {configured!r}. Available: {available}.")
    return backend


async def open_connections() -> None:
    """Connect to the configured database. Called once, when the service starts."""
    await _backend().connect()


async def close_connections() -> None:
    """Disconnect. Called once, when the service stops."""
    await _backend().disconnect()


async def ensure_collection(name: str, dimensions: int) -> bool:
    return await _backend().ensure_collection(name, dimensions)


def get_store() -> VectorStore:
    """A store for the current caller.

    Not cached: the embedder inside may be on the caller's own key, so a store
    kept from one call would embed the next caller's text on someone else's.
    Building one is cheap -- the connection is shared, and embedders are cached
    per key in their factory.
    """
    return _backend().build(get_embedding_model())
