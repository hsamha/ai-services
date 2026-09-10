"""Values the two Streamlit apps read from the environment, or from the root `.env`.

Only what a caller brings. The service's own configuration -- the embedding key,
where Qdrant is -- is read by the service's settings from the same file.
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


@lru_cache
def get_ui_settings() -> UISettings:
    return UISettings()
