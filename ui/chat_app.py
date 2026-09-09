"""The chatbot: ask the agent questions over whatever is in the RAG store.

Run with:  streamlit run ui/chat_app.py
"""

import json

import streamlit as st

from ui.schemas import ChatRole
from ui.client import APIError, ChatState, ChatTurn, RAGClient
from ui.common import connection_sidebar, run

st.set_page_config(page_title="Jordanian Constitution Assistant", page_icon="📚", layout="centered")

CHAT_KEY = "chat_state"
TRANSCRIPT_KEY = "show_transcript"
BUSY_KEY = "answering"
PENDING_KEY = "pending_question"

DISCLAIMER = "AI can make mistakes — double-check important answers."

TITLE = "📚 Jordanian Constitution Assistant"

# Sits under the title, saying in one line what the assistant is for. The
# `{link}` is filled with the source link, so the name of the text doubles as
# the way to reach the official wording.
SUBTITLE = "Ask and get answers about the {link}"

LINK_TEXT = "Jordanian Constitution"

# Where the text came from, so an answer can be checked against the official
# wording rather than taken on trust.
SOURCE_URL = (
    "https://representatives.jo/Ar/Pages/"
    "%D8%A7%D9%84%D8%AF%D8%B3%D8%AA%D9%88%D8%B1"
)

# Streamlit's own title wraps on a narrow screen, so the heading is sized in
# viewport units instead: it shrinks with the window and stays on one line.
# The clamp keeps it readable on a phone and stops it outgrowing the column
# on a wide monitor, where vw would otherwise run past the centred container.
STYLE = """
<style>
/* Streamlit styles headings as `[data-testid="stMarkdownContainer"] h1`, which
   outranks a bare class, so this has to match at least as tightly and force
   the size. Element + class, plus !important, holds against a version bump. */
h1.app-title,
[data-testid="stMarkdownContainer"] h1.app-title {
    font-size: clamp(0.85rem, 3.2vw, 1.6rem) !important;
    font-weight: 700 !important;
    line-height: 1.25 !important;
    white-space: nowrap !important;
    margin: 0 0 0.75rem 0 !important;
    padding: 0 !important;
}
.app-subtitle,
[data-testid="stMarkdownContainer"] .app-subtitle {
    font-size: clamp(0.75rem, 2vw, 0.95rem) !important;
    opacity: 0.7;
    margin: 0 0 1.5rem 0 !important;
}
/* The source link sits inside the sentence, so it is underlined rather than
   recoloured -- clearly a link, without breaking the line up. */
.app-subtitle a {
    color: inherit;
    text-decoration: underline;
    text-underline-offset: 0.15em;
}
.app-subtitle a:hover { opacity: 1; }
/* The chat input is docked to the bottom of the window, so the disclaimer is
   pinned under it rather than written after it -- anything written after it in
   the script would render above it instead. */
.app-disclaimer {
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0.4rem;
    text-align: center;
    font-size: 0.75rem;
    opacity: 0.6;
    z-index: 1000;
    /* Never swallow a click meant for the input above it. */
    pointer-events: none;
}
/* Make room for it, so the input does not sit flush on top of the text. */
[data-testid="stBottomBlockContainer"] { padding-bottom: 2.5rem; }
</style>
"""

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
    st.markdown(STYLE, unsafe_allow_html=True)
    st.markdown(f'<h1 class="app-title">{TITLE}</h1>', unsafe_allow_html=True)
    link = f'<a href="{SOURCE_URL}" target="_blank" rel="noopener">{LINK_TEXT}</a>'
    st.markdown(
        f'<p class="app-subtitle">{SUBTITLE.format(link=link)}</p>',
        unsafe_allow_html=True,
    )

    client = connection_sidebar(with_model=True)
    state = chat_state()

    # True while an answer is in flight: the input is locked until it lands.
    busy = bool(st.session_state.get(BUSY_KEY, False))

    with st.sidebar:
        st.subheader("Conversation")
        st.metric("Messages", len(state.turns))
        if st.button("Clear chat", width="stretch", disabled=busy):
            st.session_state[CHAT_KEY] = ChatState()
            st.rerun()
        st.toggle("Show the transcript", value=False, key=TRANSCRIPT_KEY)

    render_history(state)

    question = st.chat_input(
        "Thinking…" if busy else "Ask something about Jordanian constitution",
        max_chars=255,
        disabled=busy,
    )

    st.markdown(f'<div class="app-disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)

    # Park the question and rerun, so the input comes back disabled before the
    # call goes out -- a widget already on screen cannot be locked mid-script.
    if question and question.strip():
        st.session_state[PENDING_KEY] = question.strip()
        st.session_state[BUSY_KEY] = True
        st.rerun()

    pending = st.session_state.pop(PENDING_KEY, None)
    if pending:
        try:
            answer(client, state, pending)
        finally:
            st.session_state[BUSY_KEY] = False
            st.rerun()


main()
