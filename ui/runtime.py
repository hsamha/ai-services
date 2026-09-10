"""The one event loop the apps run the service's code on.

Streamlit runs a page script synchronously, on a thread with no loop of its own.
The obvious bridge, `asyncio.run(...)` per call, does not hold here: the service
keeps long-lived async clients -- the Qdrant connection, the cached OpenAI and
agent clients -- and each is bound to the loop it was first used on. A fresh loop
per call would strand them after the first one.

So there is exactly one loop per Streamlit process, running forever on a daemon
thread. Every call from every page and session is handed to it, and the page
waits for the result. The connections are opened on it once, the same way the
service's own lifespan opens them.
"""

import asyncio
import logging
import threading
from collections.abc import Coroutine
from typing import TypeVar

from src.core.vectorstores.registry import open_connections
from src.features.rag.collections import seed_collections

T = TypeVar("T")

_loop: asyncio.AbstractEventLoop | None = None
_lock = threading.Lock()


async def _open() -> None:
    """What the service does at startup: connect, and make sure the collections exist."""
    await open_connections()
    await seed_collections()


def _get_loop() -> asyncio.AbstractEventLoop:
    """The shared loop, started and connected on first use."""
    global _loop
    with _lock:
        if _loop is None:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, name="service-loop", daemon=True).start()
            asyncio.run_coroutine_threadsafe(_open(), loop).result()
            _loop = loop
    return _loop


def run(coro: Coroutine[object, object, T]) -> T:
    """Run one coroutine on the shared loop, and block the page until it settles."""
    return asyncio.run_coroutine_threadsafe(coro, _get_loop()).result()
