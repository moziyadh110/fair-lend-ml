"""Redis-backed prediction cache. Identical loan application inputs get an
identical decision without re-scoring — cheap correctness win for a service
where the same synthetic/test payload gets hit repeatedly.
"""
import hashlib
import json
import os

import redis


def build_client() -> redis.Redis:
    return redis.Redis(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=int(os.environ.get("REDIS_PORT", 6379)),
        db=0,
        decode_responses=True,
    )


def cache_key(features_dict: dict) -> str:
    return f"pred:{hashlib.md5(json.dumps(features_dict, sort_keys=True).encode()).hexdigest()}"


def get_cached(client: redis.Redis, key: str) -> dict | None:
    raw = client.get(key)
    return json.loads(raw) if raw else None


def set_cached(client: redis.Redis, key: str, value: dict, ttl_seconds: int = 3600) -> None:
    client.setex(key, ttl_seconds, json.dumps(value))
