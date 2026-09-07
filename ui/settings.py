"""Values the two Streamlit apps read from the environment, or from `ui/.env`."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Beside this file, so the apps keep their own settings wherever they are run
# from -- the repository root, or `/app/ui` in their image.
ENV_FILE = Path(__file__).parent / ".env"


class UISettings(BaseSettings):
    """Where the service is, and the keys to reach it with."""

    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    api_base_url: str = "http://localhost:8000"
    api_key: str = ""
    provider_key: str = ""
    llm_model: str = ""

    request_timeout: float = 120.0


@lru_cache
def get_ui_settings() -> UISettings:
    return UISettings()
