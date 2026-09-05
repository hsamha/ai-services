# Docker Guide

This project has **two containers**, defined in `docker-compose.yml`:

- **`qdrant`** — the vector database. Uses a prebuilt image, never rebuilt.
- **`api`** — our FastAPI app. Built from our `Dockerfile`, so **it must be rebuilt whenever our
  code or dependencies change**.

Run every command from the project root.

---

## Daily use

```bash
docker compose up -d          # start both, in the background
docker compose ps             # what's running
docker compose logs -f api    # follow the API logs (Ctrl+C just stops watching)
docker compose stop           # stop, keep the containers
docker compose start          # start them again
docker compose down           # stop AND delete the containers (data survives)
```

`up -d` is safe to run repeatedly — it only touches what changed.

---

## "I changed something. What do I run?"

| What you changed | Command |
|---|---|
| `requirements.txt` | `docker compose up -d --build api` |
| `Dockerfile` | `docker compose up -d --build api` |
| Python code in `src/` | `docker compose up -d --build api` |
| `.env` | `docker compose up -d api` (no rebuild — it's read at start) |
| `docker-compose.yml` | `docker compose up -d` |

**The one rule:** the image is a frozen snapshot taken at build time. Anything copied *into* the
image (`src/`, installed packages) needs `--build` to take effect. Anything read at *runtime*
(`.env`, ports) just needs a restart.

If a rebuild seems to ignore your change, force a clean one:

```bash
docker compose build --no-cache api
docker compose up -d api
```

### Faster loop while developing

Rebuilding for every code edit is slow. Instead, run Qdrant in Docker and the API on your machine:

```bash
docker compose up -d qdrant
pip install -r requirements.txt
uvicorn src.main:app --reload
```

`--reload` picks up code changes instantly. Use the full Docker setup to verify before shipping.

---


---

## Installing a new package

**Always through `requirements.txt`.** Two steps:

```bash
# 1. add the package name on its own line in requirements.txt
echo "tiktoken" >> requirements.txt

# 2. rebuild the api image so it gets installed
docker compose up -d --build api
```

Check it landed:

```bash
docker compose exec api pip show tiktoken
```

> **Do not** run `docker compose exec api pip install <pkg>`. It works for about five minutes, then
> disappears on the next rebuild — and your teammates never get it. Containers are disposable;
> `requirements.txt` is the source of truth.

If you're using the local dev loop (uvicorn on your machine), install it there too:

```bash
pip install -r requirements.txt
```

## Data and volumes

Qdrant's data lives in the **`./qdrant_storage/` folder in this project** (a "bind mount" — a real
folder on your disk mapped into the container). That's why your collections survive
`docker compose down`.

**Wipe all vector data and start fresh:**

```bash
docker compose down
rm -rf qdrant_storage
docker compose up -d
```

Qdrant recreates the folder empty on start. Do this when embeddings look wrong, you switched
embedding models, or you just want a clean slate.

> Because it's a plain folder, `docker volume ls` / `docker volume rm` will **not** show or delete
> it. Deleting the folder is the way.

---

## When things go wrong

```bash
docker compose logs api          # read the error — usually it's right here
docker compose logs qdrant
docker compose restart api       # quick restart
docker compose exec api sh       # open a shell inside the running container
```

**Port already in use** (`8000` or `6333`) — something else is using it. Find and stop it, or change
the left-hand number in `docker-compose.yml` (`"8001:8000"`).

**API can't reach Qdrant** — inside Docker the address is `http://qdrant:6333` (the service name),
not `localhost`. Compose sets this for you; `localhost` only works when you run uvicorn on your
machine.

**Full reset**, containers and images (data folder untouched):

```bash
docker compose down --rmi local
docker compose up -d --build
```

---

## Cheat sheet

| Goal | Command |
|---|---|
| Start | `docker compose up -d` |
| Start after changing code/deps | `docker compose up -d --build` |
| Stop | `docker compose stop` |
| Stop and remove containers | `docker compose down` |
| Logs | `docker compose logs -f api` |
| Wipe vector data | `docker compose down && rm -rf qdrant_storage` |
| Shell inside the API | `docker compose exec api sh` |
