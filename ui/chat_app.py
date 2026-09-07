"""The chatbot: ask the agent questions over whatever is in the RAG store.

Run with:  streamlit run ui/chat_app.py
"""

import streamlit as st

from ui.schemas import ChatRole
from ui.client import APIError, ChatState, ChatTurn, RAGClient
from ui.common import connection_sidebar, run

st.set_page_config(page_title="Ask your documents", page_icon="📚", layout="centered")

CHAT_KEY = "chat_state"

AVATARS: dict[ChatRole, str] = {ChatRole.USER: "🧑", ChatRole.ASSISTANT: "🤖"}


def chat_state() -> ChatState:
    """The conversation, kept across reruns."""
    if CHAT_KEY not in st.session_state:
        st.session_state[CHAT_KEY] = ChatState()
    return st.session_state[CHAT_KEY]


def render_history(state: ChatState) -> None:
    for turn in state.turns:
        with st.chat_message(turn.role.value, avatar=AVATARS[turn.role]):
            st.markdown(turn.content)
            if turn.model:
                st.caption(turn.model)


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

    state.turns.append(
        ChatTurn(role=ChatRole.ASSISTANT, content=response.answer, model=response.model)
    )


def sources_expander(client: RAGClient, question: str) -> None:
    """What the retriever finds for the last question, shown on demand."""
    with st.expander("What the retriever found"):
        try:
            results = run(client.search(question))
        except APIError as error:
            st.error(error.detail)
            return
        if not results.hits:
            st.info("Nothing matched.")
            return
        for position, hit in enumerate(results.hits, start=1):
            source = hit.metadata.get("document_id", "unknown")
            st.markdown(f"**{position}. {source}** · score {hit.score:.3f}")
            st.write(hit.text)


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
        show_sources = st.toggle("Show retrieved chunks", value=False)

    render_history(state)

    question = st.chat_input("Ask something about your documents")
    if question and question.strip():
        answer(client, state, question.strip())

    last_user = next(
        (turn for turn in reversed(state.turns) if turn.role is ChatRole.USER), None
    )
    if show_sources and last_user is not None:
        sources_expander(client, last_user.content)


main()
