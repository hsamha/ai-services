import logging
from collections.abc import Awaitable
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import HTTPException, status
from langchain_core.tools import BaseTool, tool
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel

from src.context import get_context
from src.core.llm.constants import PROVIDER_BY_MODEL
from src.core.llm.enums import LLMProvider
from src.core.llm.factory import get_text_llm, get_text_llm_name
from src.features.rag import corpus, service
from src.features.rag.prompts import TRANSLATE_PROMPT
from src.features.rag.schemas import (
    CurrentDateTime,
    SearchRequest,
    ToolFailure,
    Translation,
    WebSearchResult,
)
from src.settings import get_settings


logger = logging.getLogger(__name__)


def _as_json(payload: BaseModel) -> str:
    """What the model is handed back. Compact, and no Python repr in sight."""
    return payload.model_dump_json(exclude_none=True)


def _is_empty(payload: BaseModel) -> bool:
    echoed = {"query", "number", "document_id"}

    carried = [
        value
        for name, value in payload
        if name not in echoed and isinstance(value, (str, list, dict, type(None)))
    ]

    return bool(carried) and all(
        not value.strip() if isinstance(value, str) else not value for value in carried
    )


async def _result(call: Awaitable[BaseModel]) -> str:
    try:
        payload = await call
    except HTTPException as error:
        logger.info("Tool failed: %s", error.detail)
        return _as_json(ToolFailure(error=str(error.detail)))

    if _is_empty(payload):
        logger.info("Tool found nothing: %s", type(payload).__name__)
        return _as_json(
            ToolFailure(
                error="Nothing found. The knowledge base does not hold this -- "
                "say so, rather than answering from anywhere else."
            )
        )

    return _as_json(payload)


@tool(parse_docstring=True)
async def search_knowledge_base(
    query: str,
    document_id: str | None = None,
    score_threshold: float | None = None,
) -> str:
    """Search the knowledge base for the passages that answer a question.

    This is the way in, and the only way to find anything. Ask it a question in
    plain words -- it matches on meaning, not on keywords -- and it gives back
    the closest passages, each with its own chunk id, the document it came from
    and a similarity score. Use those ids to read further with the other tools.

    Search more than once when the first passages are thin, or when a question
    has several parts: one search per part finds more than one long query does.
    No hits, or only weak ones, means the knowledge base does not cover it.

    Args:
        query: The question, in full. A whole question matches better than a
            keyword, so ask for what you want to know rather than naming a term.
            Search in the language the documents are written in.
        document_id: Search only inside this one document. Leave it out to
            search everything stored.
        score_threshold: Drop anything matching more weakly than this. Leave it
            out unless the results are coming back too loose.
    """
    return await _result(
        service.search(
            SearchRequest(query=query, document_id=document_id, score_threshold=score_threshold)
        )
    )


@tool(parse_docstring=True)
async def read_document(document_id: str) -> str:
    """Read a whole document from the knowledge base.

    Use this when a document is short enough to take in at once and you want
    all of it rather than the passages that matched. A document too long to
    return comes back as metadata plus a note -- when that happens, search it
    instead, or list its chunks and read them in pieces.

    Args:
        document_id: The id the document was stored under, as a search hit or
            a chunk reports it. Never invent one.
    """
    return await _result(service.get_document(document_id))


@tool(parse_docstring=True)
async def list_document_chunks(document_id: str) -> str:
    """List every piece of a document, in the order it was split.

    Use this to work through a long document from the start, or to see how it
    is laid out before choosing what to read.

    Args:
        document_id: The id the document was stored under. Never invent one.
    """
    return await _result(service.get_document_chunks(document_id))


@tool(parse_docstring=True)
async def expand_chunk(chunk_id: str, window: int = 1) -> str:
    """Read the passages either side of one that matched.

    A search hit is cut to a fixed size, so an answer can run off its edge.
    This gives back the same passage with its neighbours, in document order,
    so a sentence that starts in one and finishes in the next reads whole.

    Args:
        chunk_id: The id of the chunk to widen around, as a search hit reports it.
        window: How many neighbours to take on each side. One is usually enough;
            raise it when the answer is still cut off.
    """
    return await _result(service.expand_chunk(chunk_id, window))


@tool(parse_docstring=True)
async def get_material(number: int) -> str:
    """Read one numbered article (مادة) of the constitution, in full.

    Use this whenever a question names an article by its number, or when a
    passage you have read cites one -- it gives back that article whole, so
    nothing is cut off the way a search hit can be. Ask for the articles a
    citation names rather than guessing what they say.

    Args:
        number: The article number as the text writes it, so 96 for
            "المادة (96)". A number no article carries comes back as an error
            naming the range that exists.
    """
    return await _result(corpus.get_material(number))


@tool(parse_docstring=True)
async def get_section(number: int) -> str:
    """Read one numbered chapter (فصل) of the constitution, in full.

    A chapter holds every article on one subject -- rights, the executive, the
    judiciary -- so use this when a question is about a whole topic rather than
    a single article, or to see which articles a chapter covers before reading
    them one by one. Chapters are long; prefer an article when you know which.

    Args:
        number: The chapter number, counting from one in the order the
            constitution sets them out. A number no chapter carries comes back
            as an error naming the range that exists.
    """
    return await _result(corpus.get_section(number))


@tool(parse_docstring=True)
async def current_datetime() -> str:
    """The date and time right now.

    Use this whenever a question turns on when "now" is -- how old something
    is, whether a date has passed, what "last year" or "next month" refers to.
    You have no clock of your own, so never work a date out from memory.
    """
    return await _result(_now())


async def _now() -> CurrentDateTime:
    """The clock, read in the timezone the service is configured for."""
    name = get_settings().timezone

    try:
        zone = ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown timezone {name!r}. Use an IANA name, such as 'Asia/Amman'.",
        ) from error

    now = datetime.now(zone)

    return CurrentDateTime(
        iso=now.isoformat(),
        timezone=name,
        readable=now.strftime("%d %B %Y, %H:%M"),
        weekday=now.strftime("%A"),
        utc_offset=now.strftime("%z"),
    )


@tool(parse_docstring=True)
async def translate(text: str, target_language: str, source_language: str | None = None) -> str:
    """Put text into another language.

    Use this to translate a passage a search returned, or to read a question
    asked in one language against documents written in another. Translating a
    question before searching often finds passages the original misses.

    Args:
        text: The text to translate, as it stands. Pass the whole passage
            rather than a summary of it.
        target_language: The language to translate into, named in plain words,
            such as "English" or "Arabic".
        source_language: The language the text is in, if you know it. Leave it
            out to let the model work it out.
    """
    return await _result(_translate(text, target_language, source_language))


async def _translate(
    text: str, target_language: str, source_language: str | None
) -> Translation:
    """The text in the language asked for, as the model renders it."""
    try:
        translated = await get_text_llm().ask(
            TRANSLATE_PROMPT.format(target_language=target_language, text=text)
        )
    except OpenAIError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"The translation failed: {error}",
        ) from error

    return Translation(
        text=translated.strip(),
        target_language=target_language,
        source_language=source_language,
    )


@tool(parse_docstring=True)
async def openai_web_search(query: str) -> str:
    """Search the public web for something the knowledge base does not hold.

    Use this only once the knowledge base has come up short, and say in your
    answer which parts came from the web rather than from the documents. It
    reaches the open internet, so it knows nothing about the private documents
    and must never be used to look for them.

    Args:
        query: What to look up, in full and in plain words.
    """
    return await _result(_web_search(query))


async def _web_search(query: str) -> WebSearchResult:
    """What the open web says, through OpenAI's own hosted search tool."""
    client = AsyncOpenAI(api_key=get_context().provider_key)

    try:
        response = await client.responses.create(
            model=get_text_llm_name(),
            tools=[{"type": "web_search"}],
            input=query,
        )
    except OpenAIError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"The web search failed: {error}",
        ) from error

    return WebSearchResult(query=query, answer=response.output_text.strip())


def _serves_openai(model: str) -> bool:
    """Whether this model is one OpenAI serves, and so can run a hosted tool."""
    return PROVIDER_BY_MODEL.get(model) is LLMProvider.OPENAI


def get_tools() -> list[BaseTool]:

    settings = get_settings()

    wanted: list[tuple[bool, BaseTool]] = [
        (settings.tool_search_knowledge_base, search_knowledge_base),
        (settings.tool_read_document, read_document),
        (settings.tool_list_document_chunks, list_document_chunks),
        (settings.tool_expand_chunk, expand_chunk),
        (settings.tool_get_material, get_material),
        (settings.tool_get_section, get_section),
        (settings.tool_current_datetime, current_datetime),
        (settings.tool_translate, translate),
        (
            settings.tool_openai_web_search and _serves_openai(get_text_llm_name()),
            openai_web_search,
        ),
    ]

    return [tool_ for enabled, tool_ in wanted if enabled]
