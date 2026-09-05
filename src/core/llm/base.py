from typing import Protocol


class LLM(Protocol):
    async def ask(self, prompt: str) -> str:
        ...
