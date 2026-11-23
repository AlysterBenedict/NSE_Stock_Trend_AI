"""Async concurrency runtime: pooled HTTPX client, bounded executors, GPU lock, and singleflight TTL cache."""

import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
import os
import threading
import time
from typing import Any, Awaitable, Callable

import httpx

from backend.security.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class _CacheEntry:
    value: Any
    ts: float
    ttl: float


class BoundedTTLCache:
    """Thread-safe LRU-bounded TTL cache supporting stale fallback reads."""

    def __init__(self, maxsize: int = 128, default_ttl: float = 300.0) -> None:
        self.maxsize = max(1, int(maxsize))
        self.default_ttl = float(default_ttl)
        self._store: OrderedDict[Any, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    def get_fresh(self, key: Any, ttl_override: float | None = None) -> tuple[bool, Any, float]:
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return False, None, 0.0
            eff_ttl = entry.ttl if ttl_override is None else float(ttl_override)
            if (now - entry.ts) >= eff_ttl:
                return False, None, entry.ts
            self._store.move_to_end(key)
            return True, entry.value, entry.ts

    def get_stale(self, key: Any) -> tuple[bool, Any, float]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return False, None, 0.0
            self._store.move_to_end(key)
            return True, entry.value, entry.ts

    def last_updated(self, key: Any) -> float:
        with self._lock:
            entry = self._store.get(key)
            return entry.ts if entry is not None else 0.0

    def set(self, key: Any, value: Any, ttl: float | None = None) -> None:
        eff_ttl = self.default_ttl if ttl is None else float(ttl)
        now = time.monotonic()
        with self._lock:
            self._store[key] = _CacheEntry(value=value, ts=now, ttl=eff_ttl)
            self._store.move_to_end(key)
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


class AsyncSingleflightCache:
    """Bounded TTL cache with per-event-loop async request coalescing (singleflight)."""

    def __init__(
        self,
        maxsize: int = 128,
        default_ttl: float = 300.0,
        min_refresh_interval: float = 0.0,
    ) -> None:
        self.cache = BoundedTTLCache(maxsize=maxsize, default_ttl=default_ttl)
        self.default_ttl = float(default_ttl)
        self.min_refresh_interval = float(min_refresh_interval)
        self._inflight_by_loop: dict[int, dict[Any, asyncio.Future]] = {}
        self._lock = threading.Lock()

    def clear(self) -> None:
        self.cache.clear()
        with self._lock:
            self._inflight_by_loop.clear()

    async def get_or_compute(
        self,
        key: Any,
        factory: Callable[[], Awaitable[Any]],
        ttl: float | None = None,
        force_refresh: bool = False,
        stale_on_error: bool = True,
    ) -> Any:
        eff_ttl = self.default_ttl if ttl is None else float(ttl)

        if not force_refresh:
            hit, val, _ = self.cache.get_fresh(key, ttl_override=eff_ttl)
            if hit:
                return val
        elif self.min_refresh_interval > 0.0:
            last_ts = self.cache.last_updated(key)
            if last_ts > 0.0 and (time.monotonic() - last_ts) < self.min_refresh_interval:
                hit_stale, stale_val, _ = self.cache.get_stale(key)
                if hit_stale:
                    return stale_val

        loop = asyncio.get_running_loop()
        loop_id = id(loop)

        with self._lock:
            if not force_refresh:
                hit, val, _ = self.cache.get_fresh(key, ttl_override=eff_ttl)
                if hit:
                    return val

            inflight = self._inflight_by_loop.setdefault(loop_id, {})
            existing_fut = inflight.get(key)
            if existing_fut is not None and not existing_fut.done():
                fut = existing_fut
                is_owner = False
            else:
                fut = loop.create_future()
                # Avoid "Future exception was never retrieved" warnings if owner handles error locally
                fut.add_done_callback(lambda f: f.exception() if (f.done() and not f.cancelled()) else None)
                inflight[key] = fut
                is_owner = True

        if not is_owner:
            return await asyncio.shield(fut)

        try:
            result = await factory()
            if result is not None:
                self.cache.set(key, result, ttl=eff_ttl)
            if not fut.done():
                fut.set_result(result)
            return result
        except asyncio.CancelledError as exc:
            if not fut.done():
                fut.set_exception(exc)
            raise
        except Exception as exc:
            if stale_on_error:
                ok_stale, stale_val, _ = self.cache.get_stale(key)
                if ok_stale:
                    logger.warning("Serving stale cache entry for %r after upstream error: %s", key, exc)
                    if not fut.done():
                        fut.set_result(stale_val)
                    return stale_val
            if not fut.done():
                fut.set_exception(exc)
            raise
        finally:
            with self._lock:
                inflight_map = self._inflight_by_loop.get(loop_id)
                if inflight_map is not None and inflight_map.get(key) is fut:
                    inflight_map.pop(key, None)


class RuntimeManager:
    """Manages shared HTTPX client, bounded IO/CPU thread pools, and GPU/yfinance locks."""

    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self.http_client: httpx.AsyncClient | None = None
        self.io_executor: ThreadPoolExecutor | None = None
        self.cpu_executor: ThreadPoolExecutor | None = None
        self.gpu_lock = threading.Lock()
        self.yfinance_lock = threading.Lock()

    def _ensure_executors(self) -> None:
        with self._state_lock:
            if self.io_executor is None:
                self.io_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="ns-io")
            if self.cpu_executor is None:
                cpu_workers = min(8, max(2, os.cpu_count() or 4))
                self.cpu_executor = ThreadPoolExecutor(max_workers=cpu_workers, thread_name_prefix="ns-cpu")

    async def startup(self) -> None:
        self._ensure_executors()
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(35.0, connect=10.0),
                limits=httpx.Limits(max_connections=60, max_keepalive_connections=20),
                follow_redirects=False,
                headers={"User-Agent": "NeuroStock/3.0"},
            )
        logger.info("Async runtime pools initialized.")

    async def shutdown(self) -> None:
        client = self.http_client
        self.http_client = None
        if client is not None:
            await client.aclose()

        with self._state_lock:
            io_ex = self.io_executor
            cpu_ex = self.cpu_executor
            self.io_executor = None
            self.cpu_executor = None

        if io_ex is not None:
            io_ex.shutdown(wait=True, cancel_futures=False)
        if cpu_ex is not None:
            cpu_ex.shutdown(wait=True, cancel_futures=False)
        logger.info("Async runtime pools closed.")

    def try_get_http_client(self) -> httpx.AsyncClient | None:
        return self.http_client

    async def run_io(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self._ensure_executors()
        loop = asyncio.get_running_loop()
        call = partial(fn, *args, **kwargs) if kwargs else partial(fn, *args)
        return await loop.run_in_executor(self.io_executor, call)

    async def run_cpu(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self._ensure_executors()
        loop = asyncio.get_running_loop()
        call = partial(fn, *args, **kwargs) if kwargs else partial(fn, *args)
        return await loop.run_in_executor(self.cpu_executor, call)


runtime = RuntimeManager()
