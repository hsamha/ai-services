"""The chatbot: ask the agent questions over whatever is in the RAG store.

Run with:  streamlit run ui/chat_app.py
"""

import json

import streamlit as st

from ui.schemas import ChatRole, ToolCall
from ui.client import APIError, ChatState, ChatTurn, RAGClient
from ui.common import connection_sidebar, run

st.set_page_config(page_title="Ask your documents", page_icon="📚", layout="centered")

CHAT_KEY = "chat_state"
WORKING_KEY = "show_working"

AVATARS: dict[ChatRole, str] = {ChatRole.USER: "🧑", ChatRole.ASSISTANT: "🤖"}


def chat_state() -> ChatState:
    """The conversation, kept across reruns."""
    if CHAT_KEY not in st.session_state:
        st.session_state[CHAT_KEY] = ChatState()
    return st.session_state[CHAT_KEY]


def showing_working() -> bool:
    """Whether the sidebar toggle is on. Read on every rerun, so it survives one."""
    return bool(st.session_state.get(WORKING_KEY, False))


def render_tool_calls(calls: list[ToolCall]) -> None:
    """Every tool the agent ran for this turn, in the order it ran them."""
    if not calls:
        return

    with st.expander(f"Tool calls ({len(calls)})"):
        for position, call in enumerate(calls, start=1):
            st.markdown(f"**{position}. `{call.name}`**")
            st.json(_arguments(call))
            st.code(call.output, language="json")
            if call.truncated:
                st.caption("Output cut short — raise MAX_TOOL_OUTPUT_CHARS to see all of it.")


def _arguments(call: ToolCall) -> dict[str, object] | str:
    """The arguments as a model would have written them, or the raw string."""
    try:
        return json.loads(call.arguments)
    except ValueError:
        return call.arguments


def render_turn(turn: ChatTurn) -> None:
    with st.chat_message(turn.role.value, avatar=AVATARS[turn.role]):
        st.markdown(turn.content)
        if turn.model:
            st.caption(turn.model)
        if showing_working():
            render_tool_calls(turn.tool_calls)


def render_history(state: ChatState) -> None:
    for turn in state.turns:
        render_turn(turn)


def answer(client: RAGClient, state: ChatState, question: str) -> None:
    """Send the question with what has been said so far, and show the reply.

    The history posted is what came before the question -- the service decides
    for itself how far back it reads.
    """
    history = state.history()
    state.turns.append(ChatTurn(role=ChatRole.USER, content=question))

    with st.chat_message(ChatRole.USER.value, avatar=AVATARS[ChatRole.USER]):
        st.markdown(question)

    with st.chat_message(ChatRole.ASSISTANT.value, avatar=AVATARS[ChatRole.ASSISTANT]):
        with st.spinner("Looking through the documents…"):
            try:
                response = run(client.ask(question, history))
            except APIError as error:
                st.error(error.detail)
                # Drop the question again, so a retry is not sent twice.
                state.turns.pop()
                return
        st.markdown(response.answer)
        st.caption(response.model)
        if showing_working():
            render_tool_calls(response.tool_calls)

    state.turns.append(
        ChatTurn(
            role=ChatRole.ASSISTANT,
            content=response.answer,
            model=response.model,
            tool_calls=response.tool_calls,
        )
    )


def main() -> None:
    st.title("📚 Ask your documents")
    st.caption("Answers come from the documents in the store.")

    client = connection_sidebar(with_model=True)
    state = chat_state()

    with st.sidebar:
        st.subheader("Conversation")
        st.metric("Messages", len(state.turns))
        if st.button("Clear chat", width="stretch"):
            st.session_state[CHAT_KEY] = ChatState()
            st.rerun()
        st.toggle("Show the agent's working", value=False, key=WORKING_KEY)

    render_history(state)

    question = st.chat_input("Ask something about your documents", max_chars=255)
    if question and question.strip():
        answer(client, state, question.strip())


main()
