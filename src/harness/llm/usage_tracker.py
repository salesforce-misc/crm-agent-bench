"""Usage-tracking wrapper around LLMClient implementations.

Wraps any concrete ``LLMClient`` and transparently accumulates the token
usage reported by every ``chat()``/``stream_chat()`` call into a running
``UsageTotals`` accumulator, without changing any call sites (agents just
keep calling ``self.llm.chat(...)`` as before). This lets ``bench_runner``
snapshot the accumulator before/after a task's turn loop and diff the
two snapshots to attribute usage — and eventually cost — to that task.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import AsyncIterator
from typing import Any

from harness.llm.base import LLMClient
from harness.models import ChatMessage, ChatResponse, StreamChunk, ToolDefinition


@dataclasses.dataclass
class UsageTotals:
    """Running token totals accumulated across LLMClient calls.

    Mirrors the canonical usage keys documented on ``ChatResponse.usage``
    (see ``harness.models``), plus ``num_calls`` so callers can compute
    per-call averages.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    num_calls: int = 0
    # Wall-clock spent inside the wrapped client's chat()/stream_chat() calls,
    # in seconds. Because the agent-under-test and the user simulator each get
    # their OWN UsageTrackingClient, this accumulates ONLY the inference latency
    # of whichever component this tracker wraps — it excludes the other
    # component, tool execution, and grading. bench_runner diffs snapshots to
    # attribute per-task agent latency; sec/generation = wall_seconds/num_calls.
    wall_seconds: float = 0.0

    def add(self, usage: dict[str, int] | None, wall_seconds: float = 0.0) -> None:
        """Fold one call's usage dict + wall-clock into the running totals.

        Always counts the call (``num_calls`` increments even when the
        provider returned no usage data) and always adds ``wall_seconds``, but
        only adds token counts when ``usage`` is present. Missing keys default
        to 0 — never raises.
        """
        self.num_calls += 1
        self.wall_seconds += wall_seconds
        if not usage:
            return
        prompt = usage.get("prompt_tokens", 0) or 0
        completion = usage.get("completion_tokens", 0) or 0
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.cache_read_tokens += usage.get("cache_read_tokens", 0) or 0
        self.cache_write_tokens += usage.get("cache_write_tokens", 0) or 0
        self.reasoning_tokens += usage.get("reasoning_tokens", 0) or 0
        self.total_tokens += usage.get("total_tokens", 0) or (prompt + completion)

    def copy(self) -> UsageTotals:
        """Return an independent copy of these totals."""
        return dataclasses.replace(self)


class UsageTrackingClient(LLMClient):
    """Wraps an ``LLMClient`` and accumulates usage across calls.

    Delegates ``chat``/``stream_chat`` to the inner client unchanged, so it
    is a drop-in replacement anywhere an ``LLMClient`` is expected — the
    caller sees the same ``ChatResponse``/``StreamChunk`` objects it would
    have gotten from the inner client. Any other attribute access (e.g.
    ``.model``) falls through to the wrapped client.
    """

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.usage_totals = UsageTotals()

    def __getattr__(self, name: str):
        # Only reached for attributes not found on this instance/class —
        # delegate to the wrapped client (e.g. `.model`, provider-specific helpers).
        return getattr(self._inner, name)

    def snapshot(self) -> UsageTotals:
        """Return a point-in-time copy of the accumulated totals.

        Callers (e.g. ``bench_runner``) snapshot before and after a task's
        turn loop and diff the two copies to get that task's usage, since the
        wrapped client — and therefore this accumulator — lives for the
        whole run, not just one task.
        """
        return self.usage_totals.copy()

    def reset(self) -> None:
        """Zero out the accumulator."""
        self.usage_totals = UsageTotals()

    async def aclose(self) -> None:
        """Close the wrapped client's network resources.

        Overrides the base no-op explicitly: ``__getattr__`` delegation does not
        cover ``aclose`` because it is defined on the base class, so a plain
        attribute lookup would find the no-op here instead of the inner client's.
        """
        await self._inner.aclose()

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        # Time the whole inner call. If it raises, the exception propagates and
        # add() is never reached — a failed call contributes neither latency nor
        # a num_calls increment, matching the pre-timing behavior for tokens.
        start = time.monotonic()
        kwargs: dict[str, Any] = {
            "tools": tools,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Preserve compatibility with lightweight third-party/test clients that
        # implement the pre-structured-output interface. Only semantic-grader
        # calls need this extra keyword.
        if response_format is not None:
            kwargs["response_format"] = response_format
        result = await self._inner.chat(messages, **kwargs)
        self.usage_totals.add(result.usage, wall_seconds=time.monotonic() - start)
        return result

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        # Usage typically only arrives on the final chunk of a stream (if the
        # provider/client supports it at all); buffer it and accumulate once
        # the stream is exhausted so num_calls counts one stream_chat() call,
        # not one per chunk.
        last_usage: dict[str, int] | None = None
        start = time.monotonic()
        async for chunk in self._inner.stream_chat(
            messages, tools=tools, temperature=temperature, max_tokens=max_tokens
        ):
            if chunk.usage:
                last_usage = chunk.usage
            yield chunk
        # Whole-call wall-clock (from request to stream exhaustion). We do not
        # have a separate time-to-first-token here, so sec/generation for a
        # streamed call is the full end-to-end latency.
        self.usage_totals.add(last_usage, wall_seconds=time.monotonic() - start)
