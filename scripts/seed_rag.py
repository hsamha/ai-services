"""Upload the constitution to a running instance of this service.

The materials and sections under `data/` are demo seed data. This walks them
and posts each one to `POST /api/v1/rag/documents`, so the knowledge base is
filled the same way any other caller would -- over the API, not by reaching
into the store.

Standalone on purpose. It imports nothing from `src` and needs nothing
installed: the standard library only, so plain `python3 seed_rag.py` works
without a virtualenv. That is also why it is synchronous rather than async
like the service -- it is a one-off script with no event loop to block.

The service must already be running, and the plaintext service key must be
given (the service keeps only its hash, in API_KEY_HASH):

    API_KEY=your-key python3 scripts/seed_rag.py

Or put a line reading `API_KEY=your-key` in `.env` and just run:

    python3 scripts/seed_rag.py
    python3 scripts/seed_rag.py --only materials --base-url http://localhost:8000

Re-running is safe: content already stored comes back as a 409 and is counted
as skipped rather than treated as a failure.

API_KEY=09bdf163-fa5a-43e4-af0b-76a2c18a765c python3 seed_rag.py

"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# The plaintext service key. The service stores only its hash in API_KEY_HASH,
# so it cannot hand the key back -- it has to be given here.
API_KEY_VAR = "API_KEY"

# The middleware demands this header before the handler runs. When the service
# has its own EMBEDDING_API_KEY, ingest embeds with that and any non-empty value
# gets an upload through. When it does not, ingest embeds with this header's
# key instead -- so this placeholder fails, and a real OpenAI key must be sent.
PROVIDER_KEY_PLACEHOLDER = "not-used-for-ingest"

DEFAULT_BASE_URL = "http://localhost:8000"

# The routers are all mounted under this, in src/api.py.
API_PREFIX = "/api/v1"

# The project root, one parent up from this script's own directory.
ROOT = Path(__file__).resolve().parents[1]

MATERIALS_DIR = ROOT / "data" / "materials"
SECTIONS_DIR = ROOT / "data" / "sections"

# One upload is an embed call, so give it more room than a default timeout.
REQUEST_TIMEOUT = 60

Kind = Literal["materials", "sections", "all"]


@dataclass(frozen=True)
class SeedDocument:
    """One file, ready to be posted."""

    document_id: str
    title: str
    text: str


@dataclass
class Totals:
    """What the run did, counted as it goes."""

    uploaded: int = 0
    skipped: int = 0
    failed: int = 0


class AuthFailed(Exception):
    """The service rejected our key. Every upload after this would too."""


def read_api_key() -> str:
    """The service key, from the environment if exported, else from `.env`."""
    if os.environ.get(API_KEY_VAR):
        return os.environ[API_KEY_VAR]

    path = ROOT / ".env"
    if not path.is_file():
        return ""

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == API_KEY_VAR:
            return value.strip().strip("\"'")

    return ""


def _numbered_files(directory: Path, prefix: str) -> list[tuple[int, Path]]:
    """Every `<prefix>_<n>.txt` in this directory, in numeric order."""
    if not directory.is_dir():
        raise SystemExit(f"No such directory: {directory}. Split the source text first.")

    found: list[tuple[int, Path]] = []

    for path in directory.glob(f"{prefix}_*.txt"):
        stem = path.stem.removeprefix(f"{prefix}_")
        if stem.isdigit():
            found.append((int(stem), path))

    return sorted(found)


def collect_materials() -> list[SeedDocument]:
    return [
        SeedDocument(
            document_id=f"material_{number}",
            title=f"المادة ({number})",
            text=path.read_text(encoding="utf-8").strip(),
        )
        for number, path in _numbered_files(MATERIALS_DIR, "material")
    ]


def collect_sections() -> list[SeedDocument]:
    documents: list[SeedDocument] = []

    for number, path in _numbered_files(SECTIONS_DIR, "section"):
        text = path.read_text(encoding="utf-8").strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        documents.append(
            SeedDocument(
                document_id=f"section_{number}",
                # The heading is the first line, the subject it covers the second.
                title=" - ".join(lines[:2]),
                text=text,
            )
        )

    return documents


def collect(kind: Kind) -> list[SeedDocument]:
    documents: list[SeedDocument] = []

    if kind in ("materials", "all"):
        documents.extend(collect_materials())

    if kind in ("sections", "all"):
        documents.extend(collect_sections())

    return documents


def post(base_url: str, api_key: str, document: SeedDocument) -> tuple[int, str]:
    """Post one document. Returns the status and the body, errors included."""
    body = json.dumps(
        {
            "document_id": document.document_id,
            "title": document.title,
            "text": document.text,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{API_PREFIX}/rag/documents",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key,
            "X-AI-Provider-Key": PROVIDER_KEY_PLACEHOLDER,
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        # A 4xx or 5xx arrives as an exception, but it is still an answer.
        return error.code, error.read().decode("utf-8", "replace")


def upload(base_url: str, api_key: str, document: SeedDocument, totals: Totals) -> None:
    """Post one document, and say on the terminal how it went."""
    try:
        status, body = post(base_url, api_key, document)
    except urllib.error.URLError as error:
        totals.failed += 1
        print(f"  failed  {document.document_id}: {error.reason}")
        return

    if status in (401, 403):
        # Worth stopping for: the key is wrong, not this one document.
        raise AuthFailed(f"{status} {body[:200]}")

    if status == 409:
        totals.skipped += 1
        print(f"  already {document.document_id}")
        return

    if status >= 400:
        totals.failed += 1
        print(f"  failed  {document.document_id}: {status} {body[:200]}")
        return

    totals.uploaded += 1
    print(f"  stored  {document.document_id} ({json.loads(body).get('chunk_count')} chunks)")


def seed(base_url: str, api_key: str, kind: Kind) -> Totals:
    documents = collect(kind)
    totals = Totals()

    print(f"Uploading {len(documents)} documents to {base_url}")

    # One at a time: each upload embeds, and a burst only trips rate limits.
    for document in documents:
        upload(base_url, api_key, document, totals)

    return totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SEED_BASE_URL", DEFAULT_BASE_URL),
        help=f"Where the service is listening. Default {DEFAULT_BASE_URL}.",
    )
    parser.add_argument(
        "--only",
        dest="kind",
        choices=("materials", "sections", "all"),
        default="all",
        help="Upload just one of the two sets. Default all.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = read_api_key()

    if not api_key:
        print(
            f"No {API_KEY_VAR} found. Run it as {API_KEY_VAR}=your-key python3 "
            f"{Path(__file__).name},\nor add a line reading {API_KEY_VAR}=your-key "
            f"to .env. It must be the plaintext\nkey whose sha256 is the "
            f"API_KEY_HASH already in .env.",
            file=sys.stderr,
        )
        return 2

    try:
        totals = seed(args.base_url, api_key, args.kind)
    except AuthFailed as error:
        print(f"\nThe service rejected {API_KEY_VAR}: {error}", file=sys.stderr)
        return 2

    print(
        f"\nDone. {totals.uploaded} uploaded, "
        f"{totals.skipped} already stored, {totals.failed} failed."
    )

    return 1 if totals.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
