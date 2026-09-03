"""Abstract LLM client interface."""

from __future__ import annotations

import abc
import asyncio
import os
import time
from collections.abc import AsyncIterator
from typing import Self

from harness.models import ChatMessage, ChatResponse, StreamChunk, ToolDefinition


class RateLimiter:
    """Simple token-bucket rate limiter for LLM API calls.

    Enforces a minimum interval between calls based on requests-per-minute.
    """

    def __init__(self, rpm: int = 50) -> None:
        self.min_interval = 60.0 / rpm  # seconds between calls
        self._last_call: float = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        """Wait until the next call is allowed."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self._last_call = time.monotonic()


# Global rate limiter — shared across all clients hitting the same gateway.
# Set LLM_RPM env var to configure (default: 50).
_global_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter | None:
    """Get the global rate limiter (created on first call if LLM_RPM is set)."""
    global _global_rate_limiter
    rpm = int(os.environ.get("LLM_RPM", "0"))
    if rpm > 0 and _global_rate_limiter is None:
        _global_rate_limiter = RateLimiter(rpm=rpm)
    return _global_rate_limiter


class LLMClient(abc.ABC):
    """Abstract interface for LLM providers.

    Implementations must provide both streaming and non-streaming chat completions
    with optional tool definitions (OpenAI function-calling format).
    """

    @abc.abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ChatResponse:
        """Non-streaming chat completion."""

    @abc.abstractmethod
    def stream_chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming chat completion yielding chunks.

        Declared as a plain ``def`` returning an ``AsyncIterator`` (not
        ``async def``): implementations are async generators, and calling an
        async-generator function returns the iterator directly — it is consumed
        with ``async for``, never awaited. Typing it ``async def`` would make it
        a coroutine-returning-an-iterator, which breaks ``async for`` at call
        sites and every override's variance check.
        """

    async def aclose(self) -> None:
        """Release any network resources held by the client.

        Default is a no-op for clients that hold no persistent connection pool.
        Implementations that own an ``httpx``/SDK client should override this to
        close it. Safe to call more than once. Supports ``async with``.
        """

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
