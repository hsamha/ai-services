"""Application errors and how they reach the client.

Defines the failure cases the service can report -- unauthorised, malformed
request, upstream provider failure, mismatched collection -- and renders them
all in one consistent response shape. Secrets never appear in an error.
"""
