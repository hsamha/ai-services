import logging

from src.core.llm.factory import get_text_llm
from src.features.rag.constants import DEFAULT_LANGUAGE, AnswerLanguage
from src.features.rag.prompts import DETECT_LANGUAGE_PROMPT


logger = logging.getLogger(__name__)


async def detect(question: str) -> AnswerLanguage:
    """The language to answer `question` in.

    Falls back to the default whenever the reply is not one of the two, which
    covers the model refusing, explaining itself, naming a third language, or
    being handed a question too mixed to call.
    """
    reply = await get_text_llm().ask(
        DETECT_LANGUAGE_PROMPT.format(question=question, default_language=DEFAULT_LANGUAGE)
    )

    name = reply.strip().strip(".\"'").strip().casefold()

    for candidate in AnswerLanguage:
        if name == candidate.casefold():
            return candidate

    logger.warning("Language detection returned %r; falling back to %s", reply, DEFAULT_LANGUAGE)

    return DEFAULT_LANGUAGE
