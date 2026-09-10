# Streamlit apps

Two small front ends for the RAG service. They call the service's code **in
process** -- the same service functions the HTTP routes call -- so there is no
API to start first. The routes in `src/` are untouched and still run on their
own with uvicorn; the apps simply do not go through them.

| App | Port | What it is for |
|---|---|---|
| `upload_app.py` | 8501 | Put documents in: upload files, paste text, inspect what was stored, and test-search it. |
| `chat_app.py` | 8502 | Ask the agent questions over everything in the store. |

- `client.py` is the one way the pages reach the service. It sets the request
  context the middleware would have set, and turns an `HTTPException` into an
  `APIError`.
- `runtime.py` holds the single event loop every call runs on, so the service's
  long-lived async clients (Qdrant, OpenAI) stay usable across reruns.
- `schemas.py` re-exports the service's own models.

## What still has to be there

- **Qdrant**. With Docker, this folder's `docker compose up` starts its own.
  Without Docker, whatever the root `.env` points at.
- **The root `.env`**, for `EMBEDDING_API_KEY` and the rest of the service's
  settings. `API_KEY_HASH` is not used -- nothing is authenticated in process.

## Running without Docker

```bash
pip install -r ui/requirements.txt

streamlit run ui/chat_app.py   --server.port 8502
streamlit run ui/upload_app.py --server.port 8501
```

Run them from the repository root: that is where the service reads `.env` and
`data/` from, and where `ui` and `src` import as packages. If `src` does not
import, prefix the command with `PYTHONPATH=.`.

## Running with Docker

A stack of its own -- the two apps and their own Qdrant -- fully independent
of the API's stack in the root `docker-compose.yml`. From `ui/`:

```bash
docker compose up
```

- upload app → [localhost:8501](http://localhost:8501)
- chatbot → [localhost:8502](http://localhost:8502)

This Qdrant keeps its data in `ui/qdrant_storage/`, apart from the API's, and
publishes no ports, so both stacks can run at the same time. Documents stored
through the API are not visible here, and the other way round.

The image is built from `ui/Dockerfile` with the repository root as context,
since it needs `src/` too. `src/`, `ui/` and `data/` are mounted, so Python
changes need no rebuild. After a change to a requirements file, run
`docker compose up --build`.

## Configuration

Everything is read from the root `.env` -- the same file the API uses, so there
is no separate one for the apps. Your own model provider key is typed into the
chat app's sidebar, and `LLM_MODEL` is the model its picker starts on.

The rest -- `EMBEDDING_API_KEY`, embedding model, chunking, tools -- is the
service's configuration. `QDRANT_ENVIRONMENT` decides which Qdrant is used:
`cloud` goes to your cluster, `local` to the stack's own Qdrant container.
