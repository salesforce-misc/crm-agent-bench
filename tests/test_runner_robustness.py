"""P0.2, P0.5, P0.6 — runner/config robustness regressions."""

from __future__ import annotations

import asyncio

import pytest

from harness.bench_runner import Task, run_task
from harness.graph_cli import _create_graph_agent
from harness.models import AgentResponse

# ---- P0.2: TASK_TIMEOUT is enforced, not cooperative -----------------------


class _HangingAgent:
    async def run(self, user_message: str, history=None) -> AgentResponse:
        await asyncio.sleep(30)  # would blow past the task timeout
        return AgentResponse(content="never", tool_calls_made=[], iterations=1)


class _SimpleUser:
    def reset(self, instruction: str) -> None: ...

    async def step(self, agent_message: str) -> str:
        return "hello"


@pytest.mark.asyncio
async def test_hanging_agent_is_timed_out():
    task = Task(id="t_hang", instruction="x", expected_actions=[])
    trace = await asyncio.wait_for(
        run_task(task, _HangingAgent(), _SimpleUser(), max_turns=5, timeout_seconds=1),
        timeout=10,  # the test itself must not hang if the fix is missing
    )
    assert trace.timed_out is True
    assert trace.error is not None


# ---- P0.5: explicit missing/invalid config fails fast ----------------------


def test_missing_explicit_config_raises(tmp_path):
    missing = str(tmp_path / "does_not_exist.yaml")
    with pytest.raises((FileNotFoundError, ValueError)):
        _create_graph_agent(missing)


def test_config_without_nodes_raises(tmp_path):
    bad = tmp_path / "no_nodes.yaml"
    bad.write_text("system_prompt: hi\n")
    with pytest.raises(ValueError):
        _create_graph_agent(str(bad))


# ---- P0.6: exhausted empty user-sim responses become an error --------------


class _StoppingUser:
    """First turn real, then empty forever — simulates a wedged user sim."""

    def reset(self, instruction: str) -> None:
        self._turn = 0

    async def step(self, agent_message: str) -> str:
        self._turn += 1
        return "start" if self._turn == 1 else ""


class _OkAgent:
    async def run(self, user_message: str, history=None) -> AgentResponse:
        return AgentResponse(content="responding", tool_calls_made=[], iterations=1)


@pytest.mark.asyncio
async def test_empty_user_response_is_error_not_clean_stop():
    task = Task(id="t_empty", instruction="x", expected_actions=[])
    trace = await run_task(
        task, _OkAgent(), _StoppingUser(), max_turns=5, timeout_seconds=30
    )
    assert trace.error is not None
    assert "user" in trace.error.lower()
