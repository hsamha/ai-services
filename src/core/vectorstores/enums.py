from enum import StrEnum


class VectorStoreProvider(StrEnum):
    QDRANT = "qdrant"
    CHROMA = "chroma"


class ChromaMode(StrEnum):
    """Where a Chroma database keeps its data."""

    # In the API process. Nothing to run, nothing kept: emptied on restart.
    MEMORY = "memory"
    # A folder on disk beside the service. Survives a restart.
    PERSISTENT = "persistent"
    # A Chroma server of its own, reached over HTTP.
    SERVER = "server"
