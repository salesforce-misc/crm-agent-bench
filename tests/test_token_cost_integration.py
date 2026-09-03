"""Per-task token usage + cost attribution in bench_runner.

Covers the helpers (`_usage_delta`, `_cost_for`) and the snapshot→diff→cost
flow that `run_benchmark` performs around each `run_task` call, using a stub
LLM client so no network/pricing-of-a-real-model is required.
"""

from __future__ import annotations

import pytest

from harness.bench_runner import _cost_for, _usage_delta
from harness.llm.usage_tracker import UsageTrackingClient
from harness.models import ChatResponse


class _StubLLM:
    """Minimal LLMClient stand-in returning a fixed usage per chat() call."""

    model = "stub-model"

    def __init__(self, usage: dict[str, int]) -> None:
        self._usage = usage

    async def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        return ChatResponse(content="ok", usage=dict(self._usage))

    async def stream_chat(self, *a, **k):  # pragma: no cover - not used here
        raise NotImplementedError


@pytest.mark.asyncio
async def test_usage_delta_isolates_one_tasks_tokens():
    tracker = UsageTrackingClient(
        _StubLLM({"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140})
    )
    # Simulate a prior task having already run against the long-lived client.
    await tracker.chat([])

    before = tracker.snapshot()
    # This "task" makes two agent calls.
    await tracker.chat([])
    await tracker.chat([])
    delta = _usage_delta(before, tracker.snapshot())

    assert delta["prompt_tokens"] == 200
    assert delta["completion_tokens"] == 80
    assert delta["total_tokens"] == 280
    assert delta["num_calls"] == 2  # prior call excluded by the diff


def test_cost_for_unknown_model_is_none():
    assert _cost_for("no-such-model-xyz", {"prompt_tokens": 1000}) is None


def test_cost_for_known_model_matches_pricing():
    from harness.llm.pricing import cost_usd, resolve_model

    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    card = resolve_model("gpt-4.1")
    assert card is not None  # sanity: gpt-4.1 is in pricing.yaml
    expected = cost_usd(usage, card)
    assert _cost_for("gpt-4.1", usage) == expected
    assert expected and expected > 0
