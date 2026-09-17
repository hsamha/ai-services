import logging
import re
import time

from fastapi import HTTPException, status
from openai import AsyncOpenAI, OpenAIError

from src.context import get_context
from src.core.llm.constants import PROVIDER_BY_MODEL
from src.core.llm.enums import LLMProvider
from src.core.llm.factory import get_text_llm_name
from src.features.openai_web_search.schemas import WebSearchResponse
from src.settings import get_settings


logger = logging.getLogger(__name__)


def is_available() -> bool:
    """Whether web search is switched on and the current model can run it.

    OpenAI's hosted search only runs on a model OpenAI serves.
    """
    return (
        get_settings().tool_openai_web_search
        and PROVIDER_BY_MODEL.get(get_text_llm_name()) is LLMProvider.OPENAI
    )


def ensure_available() -> None:
    """Refuse the call up front when web search cannot run."""
    if not is_available():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Web search is not available: it is switched off, "
            "or the current model is not served by OpenAI.",
        )


async def search(text: str) -> WebSearchResponse:
    """Search the web for the text, through OpenAI's hosted search, and return the result."""
    ensure_available()

    model = get_text_llm_name()
    client = AsyncOpenAI(api_key=get_context().provider_key)
    logger.info("  websearch -> %s, %d chars of input", model, len(text))
    started = time.perf_counter()

    try:
        response = await client.responses.create(
            model=model,
            tools=[{"type": "web_search"}],
            input=text,
        )
    except OpenAIError as error:
        logger.warning(
            "  websearch <- FAILED after %.1fs: %s",
            time.perf_counter() - started,
            " ".join(str(error).split())[:120],
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"The web search failed: {error}",
        ) from error

    result = _strip_tracking(response.output_text).strip()
    # How many searches the model actually ran: zero means it answered from
    # memory, which is worth seeing.
    searches = sum(1 for item in response.output if item.type == "web_search_call")
    logger.info(
        "  websearch <- %d searches, %d chars, %.1fs",
        searches,
        len(result),
        time.perf_counter() - started,
    )

    return WebSearchResponse(text=text, result=result, model=model)


# OpenAI tags every link it cites with `utm_source=openai`. Asking the model not
# to is not enough, so it is taken out here.
_UTM_OPENAI = re.compile(r"([?&])utm_source=openai(?![\w.-])(&?)")


def _drop_utm(match: re.Match[str]) -> str:
    # More parameters follow: keep the `?` or `&` before it for them.
    # It was the last one: drop the separator along with it.
    return match.group(1) if match.group(2) else ""


def _strip_tracking(text: str) -> str:
    """The text with OpenAI's tracking parameter taken out of every link."""
    return _UTM_OPENAI.sub(_drop_utm, text)
