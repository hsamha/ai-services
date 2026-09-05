"""Embedding provider selection.

Chooses the provider for a request, resolves its model from configuration, and
reuses connections across requests rather than reconnecting each time.
"""
