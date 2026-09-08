"""The chatbot: ask the agent questions over whatever is in the RAG store.

Run with:  streamlit run ui/chat_app.py
"""

import json

import streamlit as st

from ui.schemas import ChatRole
from ui.client import APIError, ChatState, ChatTurn, RAGClient
from ui.common import connection_sidebar, run

st.set_page_config(page_title="Ask your documents", page_icon="📚", layout="centered")

CHAT_KEY = "chat_state"
TRANSCRIPT_KEY = "show_transcript"

AVATARS: dict[ChatRole, str] = {ChatRole.USER: "🧑", ChatRole.ASSISTANT: "🤖"}


def chat_state() -> ChatState:
    """The conversation, kept across reruns."""
    if CHAT_KEY not in st.session_state:
        st.session_state[CHAT_KEY] = ChatState()
    return st.session_state[CHAT_KEY]


def showing_transcript() -> bool:
    """Whether the sidebar toggle is on. Read on every rerun, so it survives one."""
    return bool(st.session_state.get(TRANSCRIPT_KEY, False))


def render_transcript(transcript: str) -> None:
    """The whole run, as the service reported it.

    Shown as it came, not summarised: every message, its tool calls, what each
    one returned, and whatever the provider hung off them.
    """
    if not transcript:
        return

    try:
        messages = json.loads(transcript)
    except ValueError:
        st.caption("The transcript could not be read as JSON.")
        st.code(transcript)
        return

    with st.expander(f"Transcript ({len(messages)} messages)"):
        st.json(messages, expanded=2)


def render_turn(turn: ChatTurn) -> None:
    with st.chat_message(turn.role.value, avatar=AVATARS[turn.role]):
        st.markdown(turn.content)
        if turn.model:
            st.caption(turn.model)
        if showing_transcript():
            render_transcript(turn.transcript)


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
        if showing_transcript():
            render_transcript(response.transcript)

    state.turns.append(
        ChatTurn(
            role=ChatRole.ASSISTANT,
            content=response.answer,
            model=response.model,
            transcript=response.transcript,
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
        st.toggle("Show the transcript", value=False, key=TRANSCRIPT_KEY)

    render_history(state)

    question = st.chat_input("Ask something about your documents", max_chars=255)
    if question and question.strip():
        answer(client, state, question.strip())


main()
