

from pydantic import BaseModel, Field

MetadataValue = str | int | float | bool
Metadata = dict[str, MetadataValue]


class Chunk(BaseModel):
    text: str
    metadata: Metadata = Field(default_factory=dict)
    id: str | None = None


class SearchHit(BaseModel):
    text: str
    score: float
    metadata: Metadata = Field(default_factory=dict)
