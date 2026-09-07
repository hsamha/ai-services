"""Pieces both apps use: the connection sidebar, and the bridge to async calls."""

import asyncio
from collections.abc import Awaitable
from pathlib import Path
from typing import TypeVar

import httpx
import streamlit as st

from ui.client import APIError, RAGClient
from ui.schemas import FileType, ModelsResponse
from ui.settings import get_ui_settings


T = TypeVar("T")

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


def run(coro: Awaitable[T]) -> T:
    """Drive one async call from Streamlit's synchronous script run.

    Streamlit's own thread has no running loop, so a fresh one per call is both
    correct and cheap -- the client opens and closes its connection inside it.
    """
    return asyncio.run(coro)


def file_type_for(filename: str) -> FileType:
    """Guess from the extension. `UNKNOWN` when the name says nothing useful."""
    return EXTENSION_TYPES.get(Path(filename).suffix.lower(), FileType.UNKNOWN)


@st.cache_data(show_spinner=False, ttl=300)
def fetch_models(base_url: str, api_key: str, provider_key: str) -> ModelsResponse | None:
    """What the service will accept, or `None` when it cannot be asked.

    Cached against the connection, so the list is fetched once rather than on
    every rerun. Failure is not an error here: the picker falls back to a plain
    text box, which is all the header ever needed.
    """
    probe = RAGClient(base_url=base_url, api_key=api_key, provider_key=provider_key)
    try:
        return run(probe.list_models())
    except (APIError, httpx.HTTPError):
        return None


def model_picker(base_url: str, api_key: str, provider_key: str) -> str:
    """Choose a chat model from what the service offers.

    Empty means the service's own default, which is the first option.
    """
    available = fetch_models(base_url, api_key, provider_key)

    if available is None:
        return st.text_input(
            "Chat model",
            value=get_ui_settings().llm_model,
            placeholder="empty for the service default",
            help="The service could not be asked which models it has. Type one, or leave empty.",
        )

    default_label = f"{available.default} (service default)"
    labels = [default_label, *[model.name for model in available.models]]

    configured = get_ui_settings().llm_model
    index = labels.index(configured) if configured in labels else 0

    chosen = st.selectbox("Chat model", options=labels, index=index)
    # The default is sent as no header at all, so the service decides.
    return "" if chosen == default_label else chosen


def connection_sidebar(with_model: bool = False) -> RAGClient:
    """Show and collect the connection settings, and hand back a ready client.

    The values start from the environment, so a configured deployment needs no
    typing; anything entered here overrides them for this session only.
    `with_model` adds the chat-model picker, for the apps that ask the LLM.
    """
    settings = get_ui_settings()

    with st.sidebar:
        st.subheader("Connection")
        base_url = st.text_input("Service URL", value=settings.api_base_url)
        api_key = st.text_input("Service key", value=settings.api_key, type="password")
        provider_key = st.text_input(
            "AI provider key",
            value=settings.provider_key,
            type="password",
            help="Your own model provider key. Sent per request, never stored by the service.",
        )

        llm_model = model_picker(base_url, api_key, provider_key) if with_model else ""

        client = RAGClient(
            base_url=base_url,
            api_key=api_key,
            provider_key=provider_key,
            llm_model=llm_model,
            timeout=settings.request_timeout,
        )

        if st.button("Check service", width="stretch"):
            if run(client.health()):
                st.success("Service is up.")
            else:
                st.error(f"No answer from {base_url}.")

        if not api_key or not provider_key:
            st.warning("Both keys are needed before the service will answer.")

    return client
