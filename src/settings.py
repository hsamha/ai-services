"""Application settings, loaded from the environment via pydantic-settings.

Holds the service auth keys, Qdrant connection and collection config, chunking
parameters, the default LLM/embedding providers, and each provider's model
names. Never holds a provider API key -- that arrives per request as a header.
"""
