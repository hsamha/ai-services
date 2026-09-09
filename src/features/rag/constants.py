from enum import StrEnum


DOCUMENTS_COLLECTION = "rag_documents"

CHUNKS_COLLECTION = "rag_chunks"

class ChatRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class AnswerLanguage(StrEnum):
    """The languages an answer is written in. There are only these two."""

    ARABIC = "Arabic"
    ENGLISH = "English"


DEFAULT_LANGUAGE = AnswerLanguage.ARABIC
