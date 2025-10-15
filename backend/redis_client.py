# backend/redis_client.py
import os, json, asyncio, logging
from typing import Any, Optional
try:
    import redis.asyncio as redis
except Exception:
    redis = None

log = logging.getLogger("redis_client")

class RedisClient:
    def __init__(self, url: Optional[str] = None):
        self.url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.enabled = os.getenv("EXDATA_REDIS", "1") != "0" and redis is not None
        self._r = None

    async def init(self):
        if not self.enabled:
            return
        try:
            self._r = await redis.from_url(self.url, decode_responses=True, max_connections=20)
            await self._r.ping()
            log.info(f"Redis connected: {self.url}")
        except Exception as e:
            log.warning(f"Redis disabled: {e}")
            self._r = None
            self.enabled = False

    async def close(self):
        if self._r:
            await self._r.close()
            self._r = None

    # JSON helpers
    async def get_json(self, key: str) -> Optional[Any]:
        if not (self.enabled and self._r): return None
        try:
            val = await self._r.get(key)
            return json.loads(val) if val else None
        except Exception as e:
            log.warning(f"get_json({key}) failed: {e}")
            return None

    async def set_json(self, key: str, value: Any, ttl: int):
        if not (self.enabled and self._r): return
        try:
            await self._r.setex(key, ttl, json.dumps(value, default=str))
        except Exception as e:
            log.warning(f"set_json({key}) failed: {e}")

    # Simple lock (SET NX PX)
    async def acquire_lock(self, key: str, ttl_ms: int = 5000) -> bool:
        if not (self.enabled and self._r): return True  # без Redis — не блокируем
        try:
            ok = await self._r.set(f"lock:{key}", "1", nx=True, px=ttl_ms)
            return bool(ok)
        except Exception as e:
            log.warning(f"acquire_lock({key}) failed: {e}")
            return True

    async def release_lock(self, key: str):
        if not (self.enabled and self._r): return
        try:
            await self._r.delete(f"lock:{key}")
        except Exception:
            pass

    # Pub/Sub (на будущее для пуш-тикеров)
    async def publish(self, channel: str, message: Any):
        if not (self.enabled and self._r): return
        try:
            await self._r.publish(channel, json.dumps(message, default=str))
        except Exception as e:
            log.warning(f"publish({channel}) failed: {e}")

# singleton
_redis_client: Optional[RedisClient] = None
_lock = asyncio.Lock()

async def get_redis() -> RedisClient:
    global _redis_client
    if _redis_client and _redis_client.enabled:
        return _redis_client
    async with _lock:
        if _redis_client is None:
            _redis_client = RedisClient()
        await _redis_client.init()
        return _redis_client

async def close_redis():
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
