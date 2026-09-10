"""Pieces both apps use: the settings sidebar, and the bridge to async calls."""

from pathlib import Path

import streamlit as st

from ui.client import APIError, RAGClient
from ui.runtime import run
from ui.schemas import FileType, ModelsResponse
from ui.settings import get_ui_settings

__all__ = [
    "ACCEPTED_EXTENSIONS",
    "EXTENSION_TYPES",
    "connection_sidebar",
    "file_type_for",
    "model_picker",
    "run",
]


# What the picker offers, and what the service is told the file is.
EXTENSION_TYPES: dict[str, FileType] = {
    ".txt": FileType.TEXT,
    ".md": FileType.MARKDOWN,
    ".markdown": FileType.MARKDOWN,
    ".html": FileType.HTML,
    ".htm": FileType.HTML,
    ".pdf": FileType.PDF,
    ".docx": FileType.DOCX,
    ".pptx": FileType.PPTX,
    ".xlsx": FileType.XLSX,
    ".csv": FileType.CSV,
    ".json": FileType.JSON,
}

ACCEPTED_EXTENSIONS: list[str] = [suffix.lstrip(".") for suffix in EXTENSION_TYPES]


def file_type_for(filename: str) -> FileType:
    """Guess from the extension. `UNKNOWN` when the name says nothing useful."""
    return EXTENSION_TYPES.get(Path(filename).suffix.lower(), FileType.UNKNOWN)


@st.cache_data(show_spinner=False, ttl=300)
def fetch_models() -> ModelsResponse | None:
    """What the service will accept, or `None` when it cannot be asked.

    Cached, so the list is read once rather than on every rerun. Failure is not
    an error here: the picker falls back to a plain text box.
    """
    try:
        return run(RAGClient(provider_key="").list_models())
    except APIError:
        return None


def model_picker() -> str:
    """Choose a chat model from what the service offers.

    Empty means the service's own default, which is the first option.
    """
    available = fetch_models()

    if available is None:
        return st.text_input(
            "Chat model",
            value=get_ui_settings().llm_model,
            placeholder="empty for the service default",
            help="The models could not be listed. Type one, or leave empty.",
        )

    default_label = f"{available.default} (service default)"
    labels = [default_label, *[model.name for model in available.models]]

    configured = get_ui_settings().llm_model
    index = labels.index(configured) if configured in labels else 0

    chosen = st.selectbox("Chat model", options=labels, index=index)
    # The default is sent as no model at all, so the service decides.
    return "" if chosen == default_label else chosen


def connection_sidebar(with_model: bool = False) -> RAGClient:
    """Show and collect the caller's settings, and hand back a ready client.

    The values start from the environment, so a configured deployment needs no
    typing; anything entered here overrides them for this session only.
    `with_model` adds the provider key and chat-model picker, for the apps that
    ask the LLM -- storing and searching embed with the service's own key.
    """
    settings = get_ui_settings()

    with st.sidebar:
        st.subheader("Settings")

        provider_key = ""
        llm_model = ""
        if with_model:
            provider_key = st.text_input(
                "AI provider key",
                value=settings.provider_key,
                type="password",
                help="Your own model provider key. Used per question, never stored.",
            )
            llm_model = model_picker()

        client = RAGClient(provider_key=provider_key, llm_model=llm_model)

        if st.button("Check store", width="stretch"):
            if run(client.health()):
                st.success("The vector store is up.")
            else:
                st.error("The vector store did not answer. Is Qdrant running?")

        if with_model and not provider_key:
            st.warning("An AI provider key is needed before the assistant will answer.")

    return client
