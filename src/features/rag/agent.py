import json
import logging
import time
from collections.abc import Awaitable, Callable
from functools import lru_cache

from fastapi import HTTPException
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    wrap_model_call,
    wrap_tool_call,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from src.context import get_context
from src.core.embeddings.factory import get_embedding_model_name
from src.core.llm.factory import get_text_llm, get_text_llm_name
from src.features.openai_web_search import service as openai_web_search
from src.features.rag import language
from src.features.rag.constants import (
    TOO_MANY_STEPS_ANSWER,
    TROUBLE_ANSWER,
    AnswerLanguage,
    AnswerStatus,
    ChatRole,
)
from src.features.rag.prompts import OUT_OF_STEPS_PROMPT, SYSTEM_PROMPT, WEB_ANSWER_PROMPT
from src.features.rag.schemas import AgentAnswer, AgentReply, HistoryMessage
from src.features.rag.tools import get_tools
from src.settings import get_settings


logger = logging.getLogger(__name__)

_DELIMITER = "═" * 80


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

    The language is baked into the system prompt rather than asked for in the
    conversation, so it reads as a standing rule instead of one more thing the
    model was told once and can drift away from.
    """
    return create_agent(
        model=get_text_llm().chat_model(),
        tools=get_tools(),
        system_prompt=SYSTEM_PROMPT.format(answer_language=answer_language),
        # Passed as a bare schema so LangChain picks the provider's native
        # structured output where it has one, and a forced tool call where not.
        response_format=AgentReply,
        middleware=[_answer_when_out_of_steps, _log_tool_call],
        name="rag_agent",
    )


@wrap_model_call
async def _answer_when_out_of_steps(
    request: ModelRequest,
    handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
) -> ModelResponse:

    rounds = sum(
        1 for message in request.messages if isinstance(message, AIMessage) and message.tool_calls
    )
    if rounds < get_settings().max_agent_steps - 1:
        return await handler(request)

    logger.warning("  limit     %d tool rounds used, answering from what was found", rounds)

    return await handler(
        request.override(
            messages=[*request.messages, HumanMessage(content=OUT_OF_STEPS_PROMPT)],
            tools=[],
        )
    )


@wrap_tool_call
async def _log_tool_call(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
) -> ToolMessage | Command:
    call = request.tool_call
    args = ", ".join(
        f"{name}={_short(json.dumps(value, ensure_ascii=False), 60)}"
        for name, value in call["args"].items()
    )
    logger.info("  tool      -> %s(%s)", call["name"], args)

    result = await handler(request)

    if isinstance(result, ToolMessage):
        logger.info("  tool      <- %s", _outcome(_text(result)))

    return result


def _outcome(output: str) -> str:
    """What a tool gave back, in a few words: a count, a failure, or its size."""
    try:
        payload = json.loads(output)
    except ValueError:
        return f"{len(output)} chars"

    if not isinstance(payload, dict):
        return f"{len(output)} chars"
    if "error" in payload:
        error = str(payload["error"])
        if error.startswith("Nothing found"):
            return "nothing found"
        return f"FAILED: {_short(error, 80)}"
    for name in ("hits", "chunks"):
        if isinstance(payload.get(name), list):
            return f"{len(payload[name])} {name}"
    return f"{len(output)} chars"


def _short(text: str, limit: int) -> str:
    """One line, cut to a length a log can carry."""
    line = " ".join(text.split())
    return line if len(line) <= limit else line[: limit - 1] + "…"


async def answer(
    question: str, history: list[HistoryMessage], web_search: bool
) -> AgentAnswer:
    """Put the question to the agent, with the passages it leaned on.

    A run that cannot finish still answers. Whatever went wrong -- the provider
    refusing, a step too many, a bug of our own -- the caller is a person typing
    in a chat box, and they get one plain line saying so rather than a status
    code and an empty screen. The exception itself is logged in full.
    """
    logger.info(
        "── QUESTION %r (history: %d, web search: %s)",
        _short(question, 80),
        len(history),
        "on" if web_search else "off",
    )
    started = time.perf_counter()

    try:
        settled = await _run(question, history, web_search)
    except Exception:
        logger.exception("  error     the run failed, answering with the standing message")
        settled = AgentAnswer(text=TROUBLE_ANSWER, transcript="[]", status=AnswerStatus.NOT_FOUND)

    logger.info(
        "── DONE in %.1fs: %s (%d chars)",
        time.perf_counter() - started,
        settled.status,
        len(settled.text),
    )
    logger.info(_DELIMITER)
    return settled


async def _run(
    question: str, history: list[HistoryMessage], web_search: bool
) -> AgentAnswer:
    """The question put to the agent, for real."""
    answer_language = await language.detect(question)
    logger.info("  language  %s", answer_language)
    logger.info(
        "  model     %s (embedding: %s)", get_text_llm_name(), get_embedding_model_name()
    )

    try:
        result = await get_agent(answer_language).ainvoke(
            {"messages": _conversation(question, history)},
            config={"recursion_limit": get_settings().max_agent_steps * 2},
        )
    except GraphRecursionError:
        # Not a failure of ours to hide: the agent kept searching and never
        # settled, and saying which is the difference between "try again" and
        # "ask something narrower".
        logger.warning("  error     too many steps, the agent never settled")
        return AgentAnswer(
            text=TOO_MANY_STEPS_ANSWER, transcript="[]", status=AnswerStatus.NOT_FOUND
        )

    messages: list[BaseMessage] = result["messages"]
    transcript = _transcript(messages)

    reply = result.get("structured_response")
    if not isinstance(reply, AgentReply) or not reply.response.strip():
        # The run finished without a usable reply. Nothing to show, so say so
        # rather than hand back a blank bubble.
        logger.warning("  error     the agent gave no usable reply")
        return AgentAnswer(
            text=TROUBLE_ANSWER, transcript=transcript, status=AnswerStatus.NOT_FOUND
        )

    logger.info("  rag       %s", reply.status)

    if web_search and reply.status is AnswerStatus.NOT_FOUND:
        return await _answer_from_web(question, history, answer_language, messages)

    return AgentAnswer(text=reply.response.strip(), transcript=transcript, status=reply.status)


async def _answer_from_web(
    question: str,
    history: list[HistoryMessage],
    answer_language: AnswerLanguage,
    messages: list[BaseMessage],
) -> AgentAnswer:

    limit = get_settings().max_history_messages
    recent = history[-limit:] if limit > 0 else []
    text = WEB_ANSWER_PROMPT.format(
        answer_language=answer_language,
        history="\n".join(f"{turn.role.value}: {turn.content}" for turn in recent),
        question=question,
    )

    logger.info("  fallback  knowledge base had nothing, trying the web")

    try:
        found = await openai_web_search.search(text)
    except HTTPException:
        # The web search logged why; the run just ends with the standing message.
        return AgentAnswer(
            text=TROUBLE_ANSWER, transcript=_transcript(messages), status=AnswerStatus.NOT_FOUND
        )

    # Kept in the transcript as the step it was, so a run reads end to end.
    web_step = AIMessage(content=found.result, name="openai_web_search")
    transcript = _transcript([*messages, web_step])

    if not found.result:
        logger.warning("  fallback  the web came back empty")
        return AgentAnswer(
            text=TROUBLE_ANSWER, transcript=transcript, status=AnswerStatus.NOT_FOUND
        )

    logger.info("  fallback  answered from the web")
    return AgentAnswer(text=found.result, transcript=transcript, status=AnswerStatus.ANSWERED)


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
