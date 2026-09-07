import logging
from functools import lru_cache

from fastapi import HTTPException, status
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from src.context import get_context
from src.core.llm.factory import get_text_llm, get_text_llm_name
from src.features.rag.constants import ChatRole
from src.features.rag.prompts import SYSTEM_PROMPT
from src.features.rag.schemas import HistoryMessage
from src.features.rag.tools import RAG_TOOLS
from src.settings import get_settings


logger = logging.getLogger(__name__)


def get_agent() -> CompiledStateGraph:
    """The agent for the request being handled."""
    return _build(get_context().provider_key, get_text_llm_name())


@lru_cache(maxsize=32)
def _build(api_key: str, model: str) -> CompiledStateGraph:
    """One agent per caller and model. Building it compiles a graph, so it is kept.

    Neither argument is read: the agent is built from the request context, the
    same context `get_agent` took these from. They are here to be the cache key
    -- an agent holds the key and model it was built with, so without them every
    caller after the first would be handed an agent spending someone else's key.
    """
    return create_agent(
        model=get_text_llm().chat_model(),
        tools=RAG_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        name="rag_agent",
    )


async def answer(question: str, history: list[HistoryMessage]) -> str:
    """Put the question to the agent and return what it settled on."""
    try:
        result = await get_agent().ainvoke(
            {"messages": _conversation(question, history)},
            config={"recursion_limit": get_settings().max_agent_steps * 2},
        )
    except GraphRecursionError as error:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The question took too many retrieval steps to settle. Try a narrower one.",
        ) from error

    _log_tool_calls(result["messages"])

    return _text(result["messages"][-1])


def _conversation(question: str, history: list[HistoryMessage]) -> list[BaseMessage]:
    """The earlier turns the agent is shown, with the new question last."""
    limit = get_settings().max_history_messages
    # The turns nearest the question are the ones that explain it, so a long
    # history is cut from the front.
    recent = history[-limit:] if limit > 0 else []

    messages: list[BaseMessage] = [
        HumanMessage(content=turn.content)
        if turn.role is ChatRole.USER
        else AIMessage(content=turn.content)
        for turn in recent
    ]
    messages.append(HumanMessage(content=question))

    return messages


def _text(message: BaseMessage) -> str:
    """The answer as one string. A model may reply in parts rather than one."""
    content = message.content
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            parts.append(str(part.get("text", "")))

    return "".join(parts)


def _log_tool_calls(messages: list[BaseMessage]) -> None:
    """Write down which tools the agent reached for, and what each gave back.

    A call and its result are two separate messages, so both are logged as they
    are met -- in the order the agent worked -- rather than paired up.
    """
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                logger.info("Agent calling %s(%s).", call["name"], call["args"])
        elif isinstance(message, ToolMessage):
            logger.info(
                "Tool %s returned %d characters.", message.name, len(_text(message))
            )
