from __future__ import annotations

from enum import Enum


class RedisFailurePolicy(str, Enum):
    CACHE_FAIL_OPEN = "cache_fail_open"
    CONTROL_PLANE_CONFIGURABLE = "control_plane_configurable"


def build_answer_cache_key(
    *,
    corpus_hash: str,
    prompt_version: str,
    llm_model: str,
    retrieval_config_hash: str,
    question_hash: str,
) -> str:
    return f"answer:{corpus_hash}:{prompt_version}:{llm_model}:{retrieval_config_hash}:{question_hash}"


def build_query_embedding_cache_key(*, embedding_model: str, embedding_config_hash: str, question_hash: str) -> str:
    return f"query_embedding:{embedding_model}:{embedding_config_hash}:{question_hash}"


def build_idempotency_key(*, request_id: str) -> str:
    return f"idempotency:{request_id}"


def build_rate_limit_key(*, subject: str, window: str) -> str:
    return f"rate_limit:{subject}:{window}"


class AsyncRedisCache:
    def __init__(self, redis_client, ttl_seconds: int, fail_open: bool = True) -> None:
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds
        self.fail_open = fail_open

    async def get(self, key: str) -> str | None:
        try:
            return await self.redis.get(key)
        except Exception:
            if self.fail_open:
                return None
            raise

    async def set(self, key: str, value: str) -> None:
        try:
            await self.redis.set(key, value, ex=self.ttl_seconds)
        except Exception:
            if self.fail_open:
                return
            raise

    async def ping(self) -> bool:
        try:
            return await self.redis.ping()
        except Exception:
            return False

    async def set_nx(self, key: str, value: str, ttl_seconds: int | None = None) -> bool:
        """Set if not exists. Returns True if key was set (no prior value)."""
        try:
            return bool(await self.redis.set(key, value, nx=True, ex=ttl_seconds or self.ttl_seconds))
        except Exception:
            if self.fail_open:
                return True  # fail-open: assume no duplicate
            raise

    async def incr(self, key: str, ttl_seconds: int | None = None) -> int:
        """Increment counter. Returns new value. Sets TTL on first creation."""
        try:
            val = await self.redis.incr(key)
            if val == 1 and ttl_seconds:
                await self.redis.expire(key, ttl_seconds)
            return val
        except Exception:
            if self.fail_open:
                return 1  # fail-open: allow request
            raise

    async def delete(self, key: str) -> None:
        """Delete a key (used for idempotency cleanup)."""
        try:
            await self.redis.delete(key)
        except Exception:
            if self.fail_open:
                return
            raise
