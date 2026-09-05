"""RAG ingestion service.

Takes raw text or an uploaded file, splits it into chunks, embeds them in
batches, and upserts them into the vector store with their document metadata.
"""
