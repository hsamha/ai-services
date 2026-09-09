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

# What a caller reads when a run does not finish. Fixed text, and in both
# languages: whatever failed may be the very thing that decides which language
# to answer in, so this cannot depend on it. The reason goes to the log, where
# it belongs -- never to the reader, who can do nothing with a stack trace.
TROUBLE_ANSWER = (
    "عذرًا، حدثت مشكلة أثناء معالجة سؤالك. حاول مرة أخرى.\n\n"
    "Sorry, there was a problem handling your question. Please try again."
)

# The one failure worth telling apart: the agent worked, and kept working, and
# never settled. Trying again as asked would only run it round again.
TOO_MANY_STEPS_ANSWER = (
    "سؤالك احتاج خطوات بحث أكثر من اللازم. جرّب سؤالًا أضيق.\n\n"
    "That question took too many search steps to settle. Try a narrower one."
)
