import hmac
from hashlib import sha256
from uuid import uuid4

def hash_key(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def matches(key: str, hashed: str) -> bool:
    return hmac.compare_digest(hash_key(key), hashed)
