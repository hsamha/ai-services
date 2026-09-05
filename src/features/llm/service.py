from src.core.llm.base import LLM


async def ask(llm: LLM, prompt: str) -> str:
    message = await llm.ainvoke(prompt)
    content = message.content

    if isinstance(content, str):
        return content
    
    return "".join(part for part in content if isinstance(part, str))
