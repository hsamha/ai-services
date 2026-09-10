# Deploying the chatbot to Hugging Face Spaces

One container, one process: Streamlit running `ui/chat_app.py`, which calls the
service's code in process. No FastAPI and no Qdrant container -- the vectors are
in Qdrant Cloud. The upload app is not deployed.

## 1. Create the Space

On huggingface.co: **New Space** → SDK **Docker** → template **Blank**.
Choose **Private** if the demo is not for everyone yet.

## 2. Put the files in the Space repository

Clone the Space, then copy into its root:

| From this repo | To the Space root |
|---|---|
| `deploy/huggingface/Dockerfile` | `Dockerfile` |
| `deploy/huggingface/README.md` | `README.md` (replace the generated one) |
| `src/` | `src/` |
| `ui/` | `ui/` |
| `data/` | `data/` |
| `requirements.txt` | `requirements.txt` |

**Do not copy** `.env`, `server/`, `qdrant_storage/`, `scripts/` or `.git/`.
A Space repository is public unless the Space is private.

## 3. Set the secrets

Space → **Settings** → **Variables and secrets** → **New secret**:

| Secret | Value |
|---|---|
| `QDRANT_CLOUD_URL` | The cluster endpoint, with `:6333` on the end. |
| `QDRANT_API_KEY` | The cluster's API key. |

Leave `EMBEDDING_API_KEY` **unset**: each visitor's own key embeds their
searches, so nobody's usage is billed to you.

Optional, as a **variable** (not secret): `LLM_MODEL` for the default chat
model. It must be an OpenAI one; `gpt-4o-mini` if unset.

Already set in the Dockerfile, no need to add: `QDRANT_ENVIRONMENT=cloud`,
`OPENAI_MODELS_ONLY=true`.

The transcript toggle stays in the sidebar, off by default, so a visitor can
switch it on to see which passages an answer came from. To remove it, add the
variable `SHOW_TRANSCRIPT=false`.

## 4. Push

Commit and push the Space repository. The Space builds the image and starts it;
the build log is under **Logs**.

## Before the demo

- **The knowledge base must already be in Qdrant Cloud.** Nothing here uploads.
  Seed it from your own machine with the API or the upload app, pointed at the
  same cluster.
- **Free Qdrant Cloud clusters are suspended when unused**, and deleted after a
  longer idle time. Check the cluster before showing the demo.
- **Spaces sleep when idle.** The first visit after a sleep waits for the
  container to start again.
