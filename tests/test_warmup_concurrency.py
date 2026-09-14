"""Tests for ChromaDB fast socket probe and background warmup concurrency synchronization."""

import time
import asyncio
import pytest
from retrieval.embedder import is_chroma_http_available, get_chroma_client, reset_chroma_client
from orchestrator.server import (
    ensure_services_initialized,
    background_warmup,
    _warmup_event,
)


def test_chroma_socket_probe_fast_fail():
    """Verify that probing an unreachable port returns False in under 300ms."""
    t0 = time.perf_counter()
    available = is_chroma_http_available("127.0.0.1", 59999, timeout=0.2)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert available is False
    assert elapsed_ms < 400.0, f"Socket probe took too long: {elapsed_ms}ms"


def test_get_chroma_client_fast_fallback():
    """Verify get_chroma_client falls back to PersistentClient without Windows 4.3s TCP hang."""
    reset_chroma_client(None)
    t0 = time.perf_counter()
    client = get_chroma_client(host="127.0.0.1", port=59998, reset=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert client is not None
    assert elapsed_ms < 1500.0, f"Fallback client creation took too long: {elapsed_ms}ms"
    reset_chroma_client(None)


@pytest.mark.asyncio
async def test_concurrent_ensure_services_initialized():
    """Verify that multiple concurrent calls to ensure_services_initialized do not collide or lock SQLite."""
    tasks = [ensure_services_initialized() for _ in range(5)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in results:
        assert not isinstance(r, Exception), f"Concurrent initialization failed with: {r}"


@pytest.mark.asyncio
async def test_background_warmup_and_ensure_services_interleaving():
    """Verify that background_warmup and ensure_services_initialized coordinate cleanly via Event and Lock."""
    warmup_task = asyncio.create_task(background_warmup())
    ensure_task = asyncio.create_task(ensure_services_initialized())

    results = await asyncio.gather(warmup_task, ensure_task, return_exceptions=True)
    for r in results:
        assert not isinstance(r, Exception), f"Interleaved warmup failed with: {r}"

    assert _warmup_event.is_set()
