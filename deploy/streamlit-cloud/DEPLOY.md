# Deploying the chatbot to Streamlit Community Cloud

Free, no Docker. Community Cloud installs `ui/requirements.txt` and runs
`ui/streamlit_app.py`, which starts the chat app. The app calls the service's
code in process -- no FastAPI -- and reads its vectors from Qdrant Cloud. The
upload app is not deployed.

## 1. Before pushing to GitHub

Community Cloud deploys from a GitHub repository, so this repo has to be on
GitHub. Before the first push:

- **Remove the service key** from `scripts/seed_rag.py` (the `API_KEY=...` line
  in its docstring), and rotate that key -- it is already in your git history.
- Check that `.env`, `server/` and `qdrant_storage/` are **not** committed.
  All three are in `.gitignore`; confirm with `git status` before pushing.
- A **private** repository works too; Community Cloud asks for access to it.

## 2. Create the app

1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. **Create app** → **Deploy a public app from GitHub**.
3. Fill in:

   | Field | Value |
   |---|---|
   | Repository | `<your-github-user>/ai-services` |
   | Branch | `main` |
   | Main file path | `ui/streamlit_app.py` |
   | App URL | pick a subdomain, e.g. `jordanian-constitution` |

4. Open **Advanced settings**:
   - **Python version: 3.13.** The service's code needs it.
   - **Secrets:** paste the block below, with your own values.

```toml
QDRANT_ENVIRONMENT = "cloud"
QDRANT_CLOUD_URL = "https://<cluster>.cloud.qdrant.io:6333"
QDRANT_API_KEY = "<your-qdrant-api-key>"

# Only OpenAI chat models: the visitor's key embeds their searches too, and
# the embedding models are OpenAI's.
OPENAI_MODELS_ONLY = "true"

# Optional. Must be an OpenAI model; gpt-4o-mini if left out.
# LLM_MODEL = "gpt-4o-mini"
```

Keep every value at the top level, not under a `[section]`: top-level secrets
are also set as environment variables, which is where the app reads them.

Leave `EMBEDDING_API_KEY` **out**: each visitor's own key embeds their
searches, so nobody's usage is billed to you.

5. Click **Deploy**.

## 3. Watch it start

The first deploy installs the requirements and takes a few minutes. Progress
and errors are in the log panel (**Manage app**, bottom right of the app).

## 4. Test it

1. In the sidebar, paste an OpenAI key and click **Check store** -- it should
   say the vector store is up.
2. Ask something, e.g. `ما هو الدستور الأردني؟`

## Updating

Push to the branch. The app picks up the change and restarts. A changed secret
is applied from **Manage app → Settings → Secrets**, and the app restarts.

## If something goes wrong

| Symptom | Likely cause |
|---|---|
| `ModuleNotFoundError: ui` or `src` | Main file path is not `ui/streamlit_app.py` |
| Install fails on a nested requirement | The installer did not follow `-r ../requirements.txt` in `ui/requirements.txt`. Replace that line with the root file's contents. |
| Syntax or typing errors at import | Python version is not 3.13 -- it can only be changed by deleting and redeploying the app |
| **Check store** fails | `QDRANT_CLOUD_URL` missing `:6333`, a wrong key, a secret nested under a `[section]`, or the free cluster suspended |
| "No EMBEDDING_API_KEY configured, and no provider key sent" | No key entered in the sidebar |
| "Unknown model" | `LLM_MODEL` misspelled, or not an OpenAI model |
| App is asleep | Community Cloud sleeps apps with no visitors; open it and click to wake it |
| Out of memory | The app went over Community Cloud's resource limit; check the logs |

## Before the demo

- **The knowledge base must already be in Qdrant Cloud.** Nothing here uploads.
- **Free Qdrant Cloud clusters are suspended when unused**, and deleted after a
  longer idle time. Check the cluster first.
