from src.core.llm.enums import LLMProvider


PROVIDER_BY_MODEL: dict[str, LLMProvider] = {
    "gpt-4o-mini": LLMProvider.OPENAI,
    "gpt-4o": LLMProvider.OPENAI,
    "gpt-4.1": LLMProvider.OPENAI,
    "gpt-4.1-mini": LLMProvider.OPENAI,
    "gpt-5": LLMProvider.OPENAI,
    "gpt-5-mini": LLMProvider.OPENAI,
    "openai/gpt-oss-120b": LLMProvider.GROQ,
    "openai/gpt-oss-20b": LLMProvider.GROQ,
    "llama-3.3-70b-versatile": LLMProvider.GROQ,
    "llama-3.1-8b-instant": LLMProvider.GROQ,
}
