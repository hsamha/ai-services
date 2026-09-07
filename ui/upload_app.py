"""The upload app: put documents into the RAG store, and look at what went in.

Run with:  streamlit run ui/upload_app.py
"""

from uuid import uuid4

import streamlit as st

from ui.schemas import FileType
from ui.client import APIError, IngestResponse, RAGClient, UploadFilePayload
from ui.common import ACCEPTED_EXTENSIONS, connection_sidebar, file_type_for, run

st.set_page_config(page_title="Knowledge uploads", page_icon="📄", layout="wide")

INGESTED_KEY = "ingested_documents"

# What the service answers when the same content is already in the store.
ALREADY_STORED = 409


def ingested() -> list[IngestResponse]:
    """What this session has put in, newest first."""
    if INGESTED_KEY not in st.session_state:
        st.session_state[INGESTED_KEY] = []
    return st.session_state[INGESTED_KEY]


def remember(response: IngestResponse) -> None:
    ingested().insert(0, response)


def show_receipt(response: IngestResponse) -> None:
    """What the service made of the document."""
    st.success(f"Stored **{response.title}** as `{response.document_id}`.")
    chars, tokens, chunks = st.columns(3)
    chars.metric("Characters", f"{response.char_count:,}")
    tokens.metric("Tokens", f"{response.token_count:,}")
    chunks.metric("Chunks", f"{response.chunk_count:,}")


def upload_files_tab(client: RAGClient) -> None:
    """Pick one or more files and send each one on its own."""
    uploads = st.file_uploader(
        "Files",
        type=ACCEPTED_EXTENSIONS,
        accept_multiple_files=True,
        help="Each file is stored as its own document.",
    )
    if not uploads:
        return

    prefix = st.text_input(
        "Document id prefix",
        value="doc",
        help="Each file gets this, then its own name.",
    )
    override = st.selectbox(
        "File type",
        options=["from the file name", *[kind.value for kind in FileType]],
        help="Only set this when the extension is misleading.",
    )

    if not st.button("Upload", type="primary"):
        return

    progress = st.progress(0.0, text="Uploading…")
    for position, upload in enumerate(uploads, start=1):
        stem = upload.name.rsplit(".", 1)[0]
        document_id = f"{prefix}-{stem}" if prefix else stem
        source_type = (
            file_type_for(upload.name)
            if override == "from the file name"
            else FileType(override)
        )
        payload = UploadFilePayload(
            filename=upload.name,
            content=upload.getvalue(),
            content_type=upload.type or "application/octet-stream",
        )
        try:
            response = run(
                client.upload_document(document_id, source_type, payload, title=upload.name)
            )
        except APIError as error:
            # A refused duplicate is the expected answer, not a failure.
            if error.status_code == ALREADY_STORED:
                st.warning(f"{upload.name}: {error.detail}")
            else:
                st.error(f"{upload.name}: {error.detail}")
        else:
            remember(response)
            show_receipt(response)
        progress.progress(position / len(uploads), text=f"{position} of {len(uploads)}")

    progress.empty()


def paste_text_tab(client: RAGClient) -> None:
    """Type or paste text straight in, without a file."""
    document_id = st.text_input("Document id", value=f"note-{uuid4().hex[:8]}")
    title = st.text_input("Title", value="Pasted note")
    text = st.text_area("Text", height=280, placeholder="Paste the text to store…")

    if not st.button("Store text", type="primary", disabled=not text.strip()):
        return

    try:
        response = run(client.ingest_text(document_id, title, text))
    except APIError as error:
        # A refused duplicate is the expected answer, not a failure.
        if error.status_code == ALREADY_STORED:
            st.warning(error.detail)
        else:
            st.error(error.detail)
        return
    remember(response)
    show_receipt(response)


def inspect_tab(client: RAGClient) -> None:
    """Read back a stored document, whole or in pieces."""
    known = [item.document_id for item in ingested()]
    document_id = st.text_input(
        "Document id",
        value=known[0] if known else "",
        help="Anything already in the store, not only this session's uploads.",
    )
    if not document_id:
        return

    whole, pieces = st.columns(2)

    if whole.button("Show document", width="stretch"):
        try:
            document = run(client.get_document(document_id))
        except APIError as error:
            st.error(error.detail)
        else:
            st.json(document.metadata.model_dump(mode="json"))
            if document.message:
                st.info(document.message)
            if document.text:
                st.text_area("Text", value=document.text, height=360, disabled=True)

    if pieces.button("Show chunks", width="stretch"):
        try:
            chunks = run(client.get_document_chunks(document_id))
        except APIError as error:
            st.error(error.detail)
        else:
            st.caption(f"{len(chunks.chunks)} chunks")
            for chunk in chunks.chunks:
                with st.expander(f"#{chunk.index} · {chunk.id}"):
                    st.write(chunk.text)


def search_tab(client: RAGClient) -> None:
    """Check that a document can actually be found, before asking the chatbot."""
    query = st.text_input("Query")
    document_id = st.text_input("Narrow to one document id", value="")
    threshold = st.slider("Score threshold", 0.0, 1.0, 0.0, 0.01)

    if not st.button("Search", type="primary", disabled=not query.strip()):
        return

    try:
        results = run(
            client.search(
                query,
                document_id=document_id or None,
                score_threshold=threshold or None,
            )
        )
    except APIError as error:
        st.error(error.detail)
        return

    if not results.hits:
        st.info("Nothing matched.")
        return

    for position, hit in enumerate(results.hits, start=1):
        source = hit.metadata.get("document_id", "unknown")
        with st.expander(f"{position}. {source} · score {hit.score:.3f}"):
            st.write(hit.text)
            st.json(hit.metadata)


def main() -> None:
    st.title("📄 Knowledge uploads")
    st.caption("Put documents into the store, then check what the retriever sees.")

    client = connection_sidebar()

    with st.sidebar:
        st.subheader("This session")
        st.metric("Documents stored", len(ingested()))

    files, text, inspect, search = st.tabs(["Files", "Paste text", "Inspect", "Search"])
    with files:
        upload_files_tab(client)
    with text:
        paste_text_tab(client)
    with inspect:
        inspect_tab(client)
    with search:
        search_tab(client)


main()
