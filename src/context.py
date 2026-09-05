"""What the current request carries.

Who is calling, whose model usage the request pays for, and which models and
store they asked for. The context middleware fills this in as each request arrives; anything
deeper -- a factory, a service, a handler -- reads it from here, so a key 
is read in one place and never travels through the rest of the code.
"""

from contextvars import ContextVar, Token
from dataclasses import dataclass, field


@dataclass(frozen=True, repr=False)
class RequestContext:
    api_key: str = field(repr=False)
    provider_key: str = field(repr=False)
    llm_model: str | None
    embedding_model: str | None

    def __repr__(self) -> str:
        """Describe the context without ever printing either key."""
        return (
            f"RequestContext(api_key='***', provider_key='***', "
            f"llm_model={self.llm_model!r}, "
            f"embedding_model={self.embedding_model!r})"
        )


# Each request runs in its own copy of this, so callers never see each other's.
_current: ContextVar[RequestContext | None] = ContextVar("request_context", default=None)


def set_context(context: RequestContext) -> Token[RequestContext | None]:
    return _current.set(context)


def reset_context(token: Token[RequestContext | None]) -> None:
    _current.reset(token)


def get_context() -> RequestContext:
    context = _current.get()
    if context is None:
        raise RuntimeError("No request context.")

    return context
