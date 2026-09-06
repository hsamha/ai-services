from functools import lru_cache

from src.context import get_context
from src.core.embeddings.factory import get_embedding_model
from src.core.vectorstores import chroma, qdrant
from src.core.vectorstores.base import VectorStore, VectorStoreBackend
from src.core.vectorstores.enums import VectorStoreProvider
from src.settings import get_settings


_BACKENDS: dict[VectorStoreProvider, VectorStoreBackend] = {
    VectorStoreProvider.QDRANT: qdrant,
    VectorStoreProvider.CHROMA: chroma,
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
    _store_for_caller.cache_clear()


async def ensure_collection(name: str, dimensions: int) -> bool:
    return await _backend().ensure_collection(name, dimensions)


def get_store() -> VectorStore:
    """The store for the request being handled."""
    context = get_context()
    return _store_for_caller(context.provider_key)


@lru_cache(maxsize=32)
def _store_for_caller(api_key: str) -> VectorStore:
    """One store per caller, so its collection lookups stay warm."""
    return _backend().build(get_embedding_model())
