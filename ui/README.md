# Streamlit apps

Two small front ends for the RAG service. They are **standalone**: they talk to
it over HTTP and nothing else, with their own dependencies, their own
`Dockerfile` and their own compose file. Nothing in `src/` imports them, and
they import nothing from `src/`.

| App | Port | What it is for |
|---|---|---|
| `upload_app.py` | 8501 | Put documents in: upload files, paste text, inspect what was stored, and test-search it. |
| `chat_app.py` | 8502 | Ask the agent questions over everything in the store. |

`schemas.py` holds the part of the service's contract the apps use. It is a
deliberate copy rather than an import -- that is what keeps the two sides
separate. If a route's shape changes, follow it there.

## Running with Docker

Its own stack, brought up on its own:

```bash
docker compose -f ui/docker-compose.yml up -d --build
```

- upload app → [localhost:8501](http://localhost:8501)
- chatbot → [localhost:8502](http://localhost:8502)

The service is expected to be running already. By default the apps look for it
on the host at `http://localhost:8000`, which is where the service's own compose
file publishes it -- so the two stacks sit side by side without sharing a
network. Point them elsewhere with `API_BASE_URL`:

```bash
API_BASE_URL=http://my-service:8000 docker compose -f ui/docker-compose.yml up -d
```

| Command | |
|---|---|
| `docker compose -f ui/docker-compose.yml up -d --build` | start, after a change to `requirements.txt` or the `Dockerfile` |
| `docker compose -f ui/docker-compose.yml logs -f chat` | follow one app's logs |
| `docker compose -f ui/docker-compose.yml down` | stop and remove |

Python changes need no rebuild -- the folder is mounted, so Streamlit reruns on
save (hit **R** in the app if it does not).

## Running without Docker

```bash
pip install -r ui/requirements.txt

streamlit run ui/upload_app.py --server.port 8501
streamlit run ui/chat_app.py   --server.port 8502
```

Run them from the repository root, so `ui` imports as a package. Each blocks, so
give each its own terminal.

## Configuration

Read from the environment, or from a `ui/.env` of your own (`cp ui/.env.example
ui/.env`). The service's `.env` is untouched and unread. Every value can also be
overridden in the app's sidebar for the session.

| Variable | Meaning |
|---|---|
| `API_BASE_URL` | Where the service is. Defaults to `http://localhost:8000`. |
| `API_KEY` | Sent as `X-API-Key`. Must match the service's `API_KEY_HASH`. |
| `PROVIDER_KEY` | Sent as `X-AI-Provider-Key`. Your own model provider key. |
| `LLM_MODEL` | Sent as `X-LLM-Model`. Empty means the service's default. |
| `REQUEST_TIMEOUT` | Seconds to wait on a call. Defaults to 120. |

Neither app stores a key. They are held for the life of the browser session and
sent with each request, as the service expects.
