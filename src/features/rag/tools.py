from collections.abc import Awaitable
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import HTTPException, status
from langchain_core.tools import BaseTool, tool
from openai import OpenAIError
from pydantic import BaseModel

from src.core.llm.factory import get_text_llm
from src.features.rag import corpus, service
from src.features.rag.prompts import TRANSLATE_PROMPT
from src.features.rag.schemas import (
    CurrentDateTime,
    SearchRequest,
    ToolFailure,
    Translation,
)
from src.settings import get_settings


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
        return _as_json(ToolFailure(error=str(error.detail)))

    if _is_empty(payload):
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
    """Read one numbered article (مادة) of the Jordanian Constitution, in full.

    Only the Jordanian Constitution (الدستور الأردني) -- never use it for an
    article of any other law. Use it when a question names a constitution
    article by number, or a passage cites one; it returns the article whole.

    Args:
        number: The article number as the text writes it, so 96 for
            "المادة (96)". A number no article carries comes back as an error
            naming the range that exists.
    """
    return await _result(corpus.get_material(number))


@tool(parse_docstring=True)
async def get_section(number: int) -> str:
    """Read one numbered chapter (فصل) of the Jordanian Constitution, in full.

    Only the Jordanian Constitution (الدستور الأردني) -- never use it for any
    other law. A chapter holds every article on one subject (rights, the
    executive, the judiciary); use it for a whole topic. Chapters are long, so
    prefer get_material when you know the article.

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
    ]

    return [tool_ for enabled, tool_ in wanted if enabled]
