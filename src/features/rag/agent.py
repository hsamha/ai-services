import json
import logging
from functools import lru_cache

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from src.context import get_context
from src.core.llm.factory import get_text_llm, get_text_llm_name
from src.features.rag import language
from src.features.rag.constants import (
    TOO_MANY_STEPS_ANSWER,
    TROUBLE_ANSWER,
    AnswerLanguage,
    ChatRole,
)
from src.features.rag.prompts import SYSTEM_PROMPT
from src.features.rag.schemas import AgentAnswer, HistoryMessage
from src.features.rag.tools import get_tools
from src.settings import get_settings


logger = logging.getLogger(__name__)

_RULE = "─" * 22

def get_agent(answer_language: AnswerLanguage) -> CompiledStateGraph:
    """The agent for the request being handled, set to answer in one language."""
    return _build(get_context().provider_key, get_text_llm_name(), answer_language)


@lru_cache(maxsize=32)
def _build(api_key: str, model: str, answer_language: AnswerLanguage) -> CompiledStateGraph:
    """One agent per caller, model and language. Building it compiles a graph, so it is kept.

    The first two arguments are not read: the agent is built from the request
    context, the same context `get_agent` took these from. They are here to be
    the cache key -- an agent holds the key and model it was built with, so
    without them every caller after the first would be handed an agent spending
    someone else's key.

    The model is part of the key for a second reason: which tools an agent is
    given depends on it, since a hosted tool only runs on its own provider.

    The language is baked into the system prompt rather than asked for in the
    conversation, so it reads as a standing rule instead of one more thing the
    model was told once and can drift away from.
    """
    return create_agent(
        model=get_text_llm().chat_model(),
        tools=get_tools(),
        system_prompt=SYSTEM_PROMPT.format(answer_language=answer_language),
        name="rag_agent",
    )


async def answer(question: str, history: list[HistoryMessage]) -> AgentAnswer:
    """Put the question to the agent, with the passages it leaned on.

    A run that cannot finish still answers. Whatever went wrong -- the provider
    refusing, a step too many, a bug of our own -- the caller is a person typing
    in a chat box, and they get one plain line saying so rather than a status
    code and an empty screen. The exception itself is logged in full.
    """
    try:
        return await _run(question, history)
    except Exception:
        logger.exception("The run failed. Answering with the standing message.")
        return AgentAnswer(text=TROUBLE_ANSWER, transcript="[]")


async def _run(question: str, history: list[HistoryMessage]) -> AgentAnswer:
    """The question put to the agent, for real."""
    answer_language = await language.detect(question)
    logger.info("Answering in %s", answer_language)

    try:
        result = await get_agent(answer_language).ainvoke(
            {"messages": _conversation(question, history)},
            config={"recursion_limit": get_settings().max_agent_steps * 2},
        )
    except GraphRecursionError:
        # Not a failure of ours to hide: the agent kept searching and never
        # settled, and saying which is the difference between "try again" and
        # "ask something narrower".
        logger.warning("The question took too many steps to settle.")
        return AgentAnswer(text=TOO_MANY_STEPS_ANSWER, transcript="[]")

    messages: list[BaseMessage] = result["messages"]
    _log_tool_calls(messages)

    text = _text(messages[-1]).strip()
    if not text:
        # The run finished and said nothing. Nothing to show, so say so rather
        # than hand back a blank bubble.
        logger.warning("The run finished with an empty answer.")
        return AgentAnswer(text=TROUBLE_ANSWER, transcript=_transcript(messages))

    return AgentAnswer(text=text, transcript=_transcript(messages))


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


def _transcript(messages: list[BaseMessage]) -> str:
    """The whole run as JSON -- every message, as its own provider shaped it.

    Dumped rather than picked over: a message carries the tool calls it made,
    the output it got back and whatever the provider hung off it, and any of
    that can be the thing you need when an answer comes out wrong.
    """
    return json.dumps(
        [message.model_dump() for message in messages],
        ensure_ascii=False,
        # A message can hold a datetime or a provider object json does not know.
        default=str,
    )
