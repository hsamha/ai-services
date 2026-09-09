
from pydantic import BaseModel


class ExtractResponse(BaseModel):
    """The text of an uploaded PDF."""

    filename: str
    text: str
    char_count: int
    token_count: int
