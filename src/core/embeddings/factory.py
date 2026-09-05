"""Embedder factory.

get_embedder(provider, provider_key) -> Embedder. Resolves the model name from
settings for the chosen provider and caches clients by
(provider, model, hashed key) so HTTP pools are reused across requests.
"""
