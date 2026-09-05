"""Service authentication.

Rejects any request that does not carry a valid service key, comparing in
constant time. The health check and the interactive docs stay open.
"""
