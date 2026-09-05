"""Qdrant vector store.

Implements the VectorStore protocol on top of AsyncQdrantClient, storing the
chunk text and its document metadata in the point payload.
"""
