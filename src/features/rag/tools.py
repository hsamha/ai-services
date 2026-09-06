"""The knowledge base, as tools a model can call.

Each tool is a thin wrapper over `service`: the service holds the retrieval
logic, and the docstring here is what the model reads to decide when to reach
for it. Every one is async, so the agent calls them without blocking the loop
while a store is being read.

They are bound with `@tool(parse_docstring=True)`, which turns the `Args:`
block into the argument schema the model is shown -- so the wording of a
docstring is part of the contract, not a comment.

A tool answers in JSON. The typed response models stay the service's business;
what crosses into the conversation is text, because that is what a model reads
and what a tool message can carry unchanged.

A tool reads the store through `get_store()`, which resolves the caller from
the request context. So a tool call only works inside a request, exactly like
the handlers do.
"""

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel

from src.features.rag import service
from src.features.rag.schemas import SearchRequest


def _as_json(payload: BaseModel) -> str:
    """What the model is handed back. Compact, and no Python repr in sight."""
    return payload.model_dump_json(exclude_none=True)


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
    return _as_json(
        await service.search(
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
    return _as_json(await service.get_document(document_id))


@tool(parse_docstring=True)
async def list_document_chunks(document_id: str) -> str:
    """List every piece of a document, in the order it was split.

    Use this to work through a long document from the start, or to see how it
    is laid out before choosing what to read.

    Args:
        document_id: The id the document was stored under. Never invent one.
    """
    return _as_json(await service.get_document_chunks(document_id))


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
    return _as_json(await service.expand_chunk(chunk_id, window))


# What the agent is given. Search comes first: it is the tool that finds the ids
# every other one takes.
RAG_TOOLS: list[BaseTool] = [
    search_knowledge_base,
    read_document,
    list_document_chunks,
    expand_chunk,
]
