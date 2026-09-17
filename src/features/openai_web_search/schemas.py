from pydantic import BaseModel, Field


class WebSearchRequest(BaseModel):
    text: str = Field(min_length=1)


class WebSearchResponse(BaseModel):
    """What the web was asked, and what came back."""

    text: str
    result: str
    model: str
