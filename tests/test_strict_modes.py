"""P1.11: opt-in stricter arg/order assertion modes.

Default behavior is unchanged (retry-tolerant): match_expected_args passes if
ANY call satisfies the args; check_order is a subsequence test. Production /
safety tasks can opt into stricter modes:

  args  mode="first": the FIRST call to the tool must satisfy the args, so a
        wrong destructive call followed by a correct one still fails.
  order mode="exact": the listed tools must occur in exactly that order with no
        extra or reordered calls among them (other tools may interleave).
"""

from __future__ import annotations

import pytest

from harness.bench_runner import Task, load_tasks, run_task
from harness.grading import check_order, match_expected_args
from harness.models import AgentResponse


def _calls(*pairs):
    return [{"name": n, "arguments": a} for n, a in pairs]


# -- args: default "any" (unchanged) -----------------------------------------


def test_args_any_tolerates_wrong_then_right():
    detail = _calls(
        ("process_refund", {"amount": "500"}), ("process_refund", {"amount": "50"})
    )
    passed, _ = match_expected_args(detail, {"process_refund": {"amount": "50"}})
    assert passed is True  # a later correct call satisfies "any"


def test_args_any_is_default():
    detail = _calls(
        ("process_refund", {"amount": "500"}), ("process_refund", {"amount": "50"})
    )
    # No mode arg → same as mode="any"
    passed, _ = match_expected_args(detail, {"process_refund": {"amount": "50"}})
    assert passed is True


# -- args: mode="first" (strict) ---------------------------------------------


def test_args_first_fails_on_wrong_first_call():
    detail = _calls(
        ("process_refund", {"amount": "500"}), ("process_refund", {"amount": "50"})
    )
    passed, failures = match_expected_args(
        detail, {"process_refund": {"amount": "50"}}, mode="first"
    )
    assert passed is False
    assert any("process_refund" in f for f in failures)


def test_args_first_passes_when_first_call_correct():
    detail = _calls(
        ("process_refund", {"amount": "50"}), ("process_refund", {"amount": "500"})
    )
    passed, _ = match_expected_args(
        detail, {"process_refund": {"amount": "50"}}, mode="first"
    )
    assert passed is True


def test_args_first_never_called_fails():
    passed, failures = match_expected_args(
        _calls(("other", {})), {"process_refund": {"amount": "50"}}, mode="first"
    )
    assert passed is False


# -- order: default "subsequence" (unchanged) --------------------------------


def test_order_subsequence_allows_extra_calls():
    called = ["lookup", "get_invoice", "log", "process_refund"]
    passed, _ = check_order(called, ["get_invoice", "process_refund"])
    assert passed is True


# -- order: mode="exact" (strict) --------------------------------------------


def test_order_exact_passes_clean_chain():
    called = ["noise", "get_invoice", "other", "process_refund"]
    passed, _ = check_order(called, ["get_invoice", "process_refund"], mode="exact")
    assert passed is True  # listed tools occur exactly once, in order


def test_order_exact_fails_on_extra_listed_call():
    # process_refund called twice — not an exact chain of the listed tools.
    called = ["get_invoice", "process_refund", "process_refund"]
    passed, failures = check_order(
        called, ["get_invoice", "process_refund"], mode="exact"
    )
    assert passed is False
    assert failures


def test_order_exact_fails_on_reorder():
    called = ["process_refund", "get_invoice"]
    passed, _ = check_order(called, ["get_invoice", "process_refund"], mode="exact")
    assert passed is False


def test_order_exact_ignores_unlisted_tools():
    called = ["a", "get_invoice", "b", "c", "process_refund", "d"]
    passed, _ = check_order(called, ["get_invoice", "process_refund"], mode="exact")
    assert passed is True


# -- invalid mode rejected ---------------------------------------------------


def test_unknown_args_mode_raises():
    with pytest.raises(ValueError):
        match_expected_args([], {"t": {"a": "1"}}, mode="bogus")


def test_unknown_order_mode_raises():
    with pytest.raises(ValueError):
        check_order([], ["t"], mode="bogus")


# -- wired through Task / run_task -------------------------------------------


class _AgentWithCalls:
    def __init__(self, calls):
        self._calls = calls
        self._n = 0

    async def run(self, user_message, history=None) -> AgentResponse:
        self._n += 1
        if self._n == 1:
            return AgentResponse(
                content="ok", tool_calls_made=self._calls, iterations=1
            )
        return AgentResponse(content="done", tool_calls_made=[], iterations=1)


class _User:
    def reset(self, instruction):
        self._n = 0

    async def step(self, agent_message):
        self._n += 1
        return "go" if self._n == 1 else "###STOP###"


@pytest.mark.asyncio
async def test_run_task_first_mode_fails_wrong_first_call():
    task = Task(
        id="refund_strict",
        instruction="x",
        expected_actions=["process_refund"],
        expected_args={"process_refund": {"amount": "50"}},
        expected_args_mode="first",
    )
    agent = _AgentWithCalls(
        [
            {
                "name": "process_refund",
                "arguments": {"amount": "500"},
                "result": "ok",
                "ok": True,
            },
            {
                "name": "process_refund",
                "arguments": {"amount": "50"},
                "result": "ok",
                "ok": True,
            },
        ]
    )
    trace = await run_task(
        task,
        agent,
        _User(),
        max_turns=5,
        timeout_seconds=30,
        agent_step_timeout_seconds=30,
        user_step_timeout_seconds=30,
    )
    assert trace.args_passed is False


def test_default_modes_when_yaml_omits_them(tmp_path):
    f = tmp_path / "t.yaml"
    f.write_text(
        "tasks:\n  - id: t1\n    instruction: hi\n    expected_actions: [get_invoice]\n"
    )
    task = load_tasks(str(f))[0]
    assert task.expected_args_mode == "any"
    assert task.expected_order_mode == "subsequence"
