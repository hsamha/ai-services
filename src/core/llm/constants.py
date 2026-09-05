from src.core.llm.enums import LLMProvider


PROVIDER_BY_MODEL: dict[str, LLMProvider] = {
    "gpt-4o-mini": LLMProvider.OPENAI,
    "gpt-4o": LLMProvider.OPENAI,
    "gpt-4.1": LLMProvider.OPENAI,
    "gpt-4.1-mini": LLMProvider.OPENAI,
    "gpt-5": LLMProvider.OPENAI,
    "gpt-5-mini": LLMProvider.OPENAI,
}
