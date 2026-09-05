"""LLM factory.

get_llm(provider, provider_key) -> LLM. Resolves the model name from settings
for the chosen provider and caches clients by (provider, model, hashed key).
"""
