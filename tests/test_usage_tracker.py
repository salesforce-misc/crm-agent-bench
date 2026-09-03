"""Tests for UsageTrackingClient — accumulation across chat()/stream_chat() calls."""

import asyncio
from collections.abc import AsyncIterator

import pytest

from harness.llm.base import LLMClient
from harness.llm.usage_tracker import UsageTrackingClient
from harness.models import ChatMessage, ChatResponse, StreamChunk, ToolDefinition


class _StubClient(LLMClient):
    """Fake LLMClient returning canned responses/usage for testing the wrapper."""

    def __init__(self, model: str = "stub-model") -> None:
        self.model = model
        self._chat_responses: list[ChatResponse] = []
        self._stream_chunks: list[StreamChunk] = []

    def queue_chat_response(self, response: ChatResponse) -> None:
        self._chat_responses.append(response)

    def queue_stream_chunks(self, chunks: list[StreamChunk]) -> None:
        self._stream_chunks = chunks

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ChatResponse:
        return self._chat_responses.pop(0)

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        for chunk in self._stream_chunks:
            yield chunk


@pytest.mark.asyncio
async def test_chat_accumulates_totals_across_calls():
    stub = _StubClient()
    stub.queue_chat_response(
        ChatResponse(
            content="hi",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cache_read_tokens": 2,
                "cache_write_tokens": 1,
                "reasoning_tokens": 0,
            },
        )
    )
    stub.queue_chat_response(
        ChatResponse(
            content="bye",
            usage={
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "reasoning_tokens": 4,
            },
        )
    )

    client = UsageTrackingClient(stub)
    messages = [ChatMessage(role="user", content="hi")]

    await client.chat(messages)
    await client.chat(messages)

    totals = client.usage_totals
    assert totals.num_calls == 2
    assert totals.prompt_tokens == 17
    assert totals.completion_tokens == 8
    assert totals.total_tokens == 25
    assert totals.cache_read_tokens == 2
    assert totals.cache_write_tokens == 1
    assert totals.reasoning_tokens == 4


@pytest.mark.asyncio
async def test_chat_call_with_no_usage_still_counted():
    stub = _StubClient()
    stub.queue_chat_response(ChatResponse(content="no usage here", usage=None))

    client = UsageTrackingClient(stub)
    await client.chat([ChatMessage(role="user", content="hi")])

    assert client.usage_totals.num_calls == 1
    assert client.usage_totals.prompt_tokens == 0
    assert client.usage_totals.total_tokens == 0


@pytest.mark.asyncio
async def test_snapshot_returns_independent_copy_for_per_task_diffing():
    stub = _StubClient()
    stub.queue_chat_response(
        ChatResponse(content="one", usage={"prompt_tokens": 5, "completion_tokens": 1})
    )
    stub.queue_chat_response(
        ChatResponse(content="two", usage={"prompt_tokens": 5, "completion_tokens": 1})
    )

    client = UsageTrackingClient(stub)
    messages = [ChatMessage(role="user", content="hi")]

    await client.chat(messages)
    before = client.snapshot()

    await client.chat(messages)
    after = client.snapshot()

    # Snapshots are independent copies — mutating the live accumulator must
    # not retroactively change a snapshot taken earlier.
    assert before.prompt_tokens == 5
    assert after.prompt_tokens == 10
    assert after.prompt_tokens - before.prompt_tokens == 5


@pytest.mark.asyncio
async def test_reset_zeroes_totals():
    stub = _StubClient()
    stub.queue_chat_response(
        ChatResponse(content="one", usage={"prompt_tokens": 5, "completion_tokens": 1})
    )

    client = UsageTrackingClient(stub)
    await client.chat([ChatMessage(role="user", content="hi")])
    assert client.usage_totals.num_calls == 1

    client.reset()
    assert client.usage_totals.num_calls == 0
    assert client.usage_totals.prompt_tokens == 0


@pytest.mark.asyncio
async def test_stream_chat_accumulates_from_final_chunk():
    stub = _StubClient()
    stub.queue_stream_chunks(
        [
            StreamChunk(delta_content="Hel"),
            StreamChunk(delta_content="lo"),
            StreamChunk(
                finish_reason="stop",
                usage={
                    "prompt_tokens": 20,
                    "completion_tokens": 8,
                    "total_tokens": 28,
                    "cache_read_tokens": 3,
                    "cache_write_tokens": 0,
                    "reasoning_tokens": 2,
                },
            ),
        ]
    )

    client = UsageTrackingClient(stub)
    collected = []
    async for chunk in client.stream_chat([ChatMessage(role="user", content="hi")]):
        collected.append(chunk)

    # All chunks pass through unchanged.
    assert len(collected) == 3

    totals = client.usage_totals
    assert totals.num_calls == 1
    assert totals.prompt_tokens == 20
    assert totals.completion_tokens == 8
    assert totals.cache_read_tokens == 3
    assert totals.reasoning_tokens == 2


def test_getattr_delegates_to_inner_client():
    stub = _StubClient(model="my-model")
    client = UsageTrackingClient(stub)
    assert client.model == "my-model"


# --- Agent-only latency tracking -------------------------------------------


class _SlowClient(LLMClient):
    """LLMClient that sleeps a fixed duration per chat() so wall_seconds is
    measurably non-zero and deterministic-enough to assert a lower bound on."""

    def __init__(self, delay: float = 0.02) -> None:
        self.model = "slow-model"
        self._delay = delay

    async def chat(
        self, messages, tools=None, temperature=0.0, max_tokens=4096
    ) -> ChatResponse:
        await asyncio.sleep(self._delay)
        return ChatResponse(
            content="ok", usage={"prompt_tokens": 1, "completion_tokens": 1}
        )

    async def stream_chat(
        self, messages, tools=None, temperature=0.0, max_tokens=4096
    ) -> AsyncIterator[StreamChunk]:
        await asyncio.sleep(self._delay)
        yield StreamChunk(delta_content="ok")
        yield StreamChunk(
            finish_reason="stop", usage={"prompt_tokens": 1, "completion_tokens": 1}
        )


@pytest.mark.asyncio
async def test_chat_accumulates_wall_seconds():
    client = UsageTrackingClient(_SlowClient(delay=0.02))
    messages = [ChatMessage(role="user", content="hi")]

    await client.chat(messages)
    await client.chat(messages)

    totals = client.usage_totals
    assert totals.num_calls == 2
    # Two ~20ms sleeps: total wall-clock is at least the summed sleeps.
    assert totals.wall_seconds >= 0.04
    # sec/generation is the per-call mean.
    assert totals.wall_seconds / totals.num_calls >= 0.02


@pytest.mark.asyncio
async def test_stream_chat_accumulates_wall_seconds():
    client = UsageTrackingClient(_SlowClient(delay=0.02))
    async for _ in client.stream_chat([ChatMessage(role="user", content="hi")]):
        pass
    assert client.usage_totals.num_calls == 1
    assert client.usage_totals.wall_seconds >= 0.02


@pytest.mark.asyncio
async def test_wall_seconds_diffs_across_snapshots_for_per_task_latency():
    client = UsageTrackingClient(_SlowClient(delay=0.02))
    messages = [ChatMessage(role="user", content="hi")]

    await client.chat(messages)
    before = client.snapshot()
    await client.chat(messages)
    after = client.snapshot()

    # Per-task agent latency = snapshot diff, mirroring how bench_runner
    # attributes one task's inference time.
    assert after.wall_seconds - before.wall_seconds >= 0.02


@pytest.mark.asyncio
async def test_failed_chat_adds_neither_latency_nor_call():
    class _Boom(LLMClient):
        model = "boom"

        async def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096):
            raise RuntimeError("gateway down")

        async def stream_chat(
            self, messages, tools=None, temperature=0.0, max_tokens=4096
        ):
            raise RuntimeError("gateway down")
            yield  # pragma: no cover

    client = UsageTrackingClient(_Boom())
    with pytest.raises(RuntimeError):
        await client.chat([ChatMessage(role="user", content="hi")])

    # A raised call is not a completed generation: it must not inflate
    # num_calls or wall_seconds (which would skew sec/generation).
    assert client.usage_totals.num_calls == 0
    assert client.usage_totals.wall_seconds == 0.0
