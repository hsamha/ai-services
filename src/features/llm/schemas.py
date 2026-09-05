from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    prompt: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str
    model: str
