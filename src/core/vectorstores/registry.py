"""Named knowledge bases.

Resolves a logical name to a concrete collection so several knowledge bases can
share one connection. A collection belongs to the embedding model it was built
with; reading it with a different one is refused rather than silently wrong.
"""
