"""Question answering routes.

Endpoints to add documents (as text or an uploaded file), search them, ask a
question, and remove a document. Reads the per-request provider settings from
the request headers and assembles the services this feature needs.
"""
