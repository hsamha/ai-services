"""RAG HTTP routes.

Ingest (raw text and file upload), search and query endpoints. Reads the
per-request headers (provider key, LLM provider, embedding provider) as route
parameters and assembles the RAG services from the factories -- the only place
construction happens.
"""
