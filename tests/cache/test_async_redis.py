import pytest

from app.cache.redis_cache import AsyncRedisCache


class FakeRedis:
    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value

    async def ping(self):
        return True


class FailingRedis:
    async def get(self, key):
        raise ConnectionError("Redis is down")

    async def set(self, key, value, ex=None):
        raise ConnectionError("Redis is down")


@pytest.mark.anyio
async def test_async_cache_get_set_with_ttl():
    cache = AsyncRedisCache(FakeRedis(), ttl_seconds=300, fail_open=True)
    await cache.set("k1", "v1")
    assert await cache.get("k1") == "v1"


@pytest.mark.anyio
async def test_async_cache_fail_open_returns_none_on_error():
    cache = AsyncRedisCache(FailingRedis(), ttl_seconds=300, fail_open=True)
    assert await cache.get("k1") is None
    await cache.set("k1", "v1")  # no error raised


@pytest.mark.anyio
async def test_async_cache_fail_closed_raises_on_error():
    cache = AsyncRedisCache(FailingRedis(), ttl_seconds=300, fail_open=False)
    with pytest.raises(ConnectionError):
        await cache.get("k1")
