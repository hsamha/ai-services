# AI Services

## Hard rules for Claude

- **Never stage, commit or push.** No `git add`, `git mv`, `git rm`, `git commit`, `git push`, or
  anything else that touches the index or the remote. Change files in the working tree only; the
  user does all git operations.
- **Never run the project.** No `docker compose up`, `uvicorn`, `streamlit run`, or any other
  command that starts the service, the apps or their databases. Say what to run and let the user
  run it.

## Non-negotiable conventions

### 1. Everything is async

Every function that touches I/O is `async def` — route handlers, service methods, every protocol
method in `core/*/base.py`, and every provider implementation.

- Use the async clients: `AsyncOpenAI`, `AsyncQdrantClient`.
- Never block the event loop: no `time.sleep`, no sync HTTP, no blocking file reads.
  Unavoidable CPU-bound or sync work goes through `anyio.to_thread.run_sync`.
- No sync twin of an async function. There is one version, and it is async.

### 2. Async libraries only

A new dependency must have a real async API.

| Use | Not |
|---|---|
| `httpx.AsyncClient` | `requests` |
| `aiofiles` | bare `open()` for I/O in a handler |
| `AsyncOpenAI` | `OpenAI` |
| `AsyncQdrantClient` | `QdrantClient` |
| `asyncpg`, `redis.asyncio` | `psycopg2`, `redis` |

If only a sync library exists, isolate it behind `anyio.to_thread.run_sync` and note why in a
comment.

### 3. Every method has explicit input and output types

- Annotate every parameter and every return, including `-> None`.
- No bare `dict`, `list`, or `Any`. Use the domain models in `src/core/types.py` (`Document`,
  `Chunk`, `SearchHit`) and the feature schemas in `schemas.py`, with real generics:
  `list[Chunk]`, `dict[str, str]`.
- The `Protocol` in each `base.py` is the contract. A provider class must match its signatures
  exactly — same names, same types, same async-ness.
