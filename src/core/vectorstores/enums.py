from enum import StrEnum


class VectorStoreProvider(StrEnum):
    """The vector databases this service can be pointed at."""

    QDRANT = "qdrant"


class QdrantEnvironment(StrEnum):
    LOCAL = "local"
    CLOUD = "cloud"
