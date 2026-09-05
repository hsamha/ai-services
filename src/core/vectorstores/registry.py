"""Named store registry.

Maps a logical store name plus the embedding model to a concrete collection, so
several knowledge bases share one client. A collection is bound to the
embedding model it was ingested with; querying it with another is an error.
"""
