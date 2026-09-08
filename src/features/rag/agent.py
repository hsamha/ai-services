import json
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
from src.features.rag.schemas import (
    AgentAnswer,
    HistoryMessage,
    ToolCall,
)
from src.features.rag.tools import get_tools
from src.settings import get_settings


logger = logging.getLogger(__name__)

_RULE = "─" * 22

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

    The model is part of the key for a second reason: which tools an agent is
    given depends on it, since a hosted tool only runs on its own provider.
    """
    return create_agent(
        model=get_text_llm().chat_model(),
        tools=get_tools(),
        system_prompt=SYSTEM_PROMPT,
        name="rag_agent",
    )


async def answer(question: str, history: list[HistoryMessage]) -> AgentAnswer:
    """Put the question to the agent, with the passages it leaned on."""
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

    messages: list[BaseMessage] = result["messages"]
    _log_tool_calls(messages)

    return AgentAnswer(text=_text(messages[-1]), tool_calls=_tool_calls(messages))


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
    logger.info("%s TOOL CALLS %s", _RULE, _RULE)

    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                logger.info("  ->  %s(%s)", call["name"], call["args"])
        elif isinstance(message, ToolMessage):
            logger.info(
                "  <-  %s returned %d characters", message.name, len(_text(message))
            )

    logger.info("%s", _RULE * 3)


def _tool_calls(messages: list[BaseMessage]) -> list[ToolCall]:
    """Every tool the agent reached for, in the order it worked.

    A call and its result are two separate messages, tied together by the id
    the model gave the call -- so the results are indexed first, then each call
    is matched to its own. A call still waiting on its result is not reported.
    """
    results: dict[str, ToolMessage] = {
        message.tool_call_id: message
        for message in messages
        if isinstance(message, ToolMessage) and message.tool_call_id
    }

    calls: list[ToolCall] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            result = results.get(call["id"] or "")
            if result is None:
                continue
            output, truncated = _clip(_text(result))
            calls.append(
                ToolCall(
                    name=call["name"],
                    arguments=json.dumps(call["args"], ensure_ascii=False, default=str),
                    output=output,
                    truncated=truncated,
                )
            )

    return calls


def _clip(output: str) -> tuple[str, bool]:
    """Keep a tool's answer readable. A whole document is more than a reader wants."""
    limit = 4000
    if limit <= 0 or len(output) <= limit:
        return output, False
    return output[:limit], True
