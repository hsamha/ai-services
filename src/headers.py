"""
The headers a caller may send, declared for the docs.
"""

from typing import Annotated

from fastapi import Header, Security
from fastapi.security import APIKeyHeader

from src.middleware.auth import SERVICE_KEY_HEADER
from src.middleware.context import LLM_MODEL_HEADER, PROVIDER_KEY_HEADER

# Each needs its own scheme_name, or they collapse into a single Authorize box.
service_key_scheme = APIKeyHeader(
    name=SERVICE_KEY_HEADER,
    scheme_name="Service key",
    description="Your key for this service.",
    auto_error=False,
)

provider_key_scheme = APIKeyHeader(
    name=PROVIDER_KEY_HEADER,
    scheme_name="AI provider key",
    description="Your own model provider key. Used for this request only, never stored.",
    auto_error=False,
)


async def request_headers(
    service_key: Annotated[str | None, Security(service_key_scheme)] = None,
    provider_key: Annotated[str | None, Security(provider_key_scheme)] = None,
    llm_model: Annotated[
        str | None,
        Header(
            alias=LLM_MODEL_HEADER,
            description="Chat model, e.g. gpt-4o-mini. Empty for the configured default.",
        ),
    ] = None,
) -> None:
    """Declare the headers so the docs offer them. Reading them happens earlier."""
    return
