"""Service authentication middleware.

Validates the X-API-Key header against the configured service keys using a
constant-time comparison, returning 401 otherwise. Unguarded paths: /health,
/docs, /openapi.json.
"""
