"""Values the two Streamlit apps read from the environment, or from the root `.env`.

Only what a caller brings, and how the apps present themselves. The service's own
configuration -- the embedding key, where Qdrant is -- is read by the service's
settings from the same file.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# The repository root's, found from this file so it is the same wherever the
# apps are run from.
ENV_FILE = Path(__file__).parents[1] / ".env"


class UISettings(BaseSettings):
    """The caller's own values, used as the sidebar's starting point."""

    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    provider_key: str = ""
    llm_model: str = ""

    # Offer the toggle that shows each run's tool calls. Off for a public demo,
    # where it would show every visitor the agent's internals.
    show_transcript: bool = True

    # Offer only OpenAI chat models. For a deployment with no EMBEDDING_API_KEY:
    # the visitor's key embeds too, and embeddings are OpenAI's, so any other
    # provider's key could chat but never search.
    openai_models_only: bool = False


@lru_cache
def get_ui_settings() -> UISettings:
    return UISettings()
