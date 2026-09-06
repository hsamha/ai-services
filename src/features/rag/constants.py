from enum import StrEnum


DOCUMENTS_COLLECTION = "rag_documents"

CHUNKS_COLLECTION = "rag_chunks"


class ChatRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
