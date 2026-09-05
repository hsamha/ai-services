"""RAG query service.

Embeds the question, retrieves matching chunks from the vector store, builds
the grounded prompt, calls the LLM, and returns the answer with citations.
"""
