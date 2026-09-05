"""Application error hierarchy and FastAPI exception handlers.

Defines AppError and its subclasses (auth, bad request, upstream provider
failure, embedding/collection mismatch) and maps them to a single JSON error
body shape. Provider keys must never appear in an error response.
"""
