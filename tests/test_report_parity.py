"""P1.1: the built-in report.md must reflect strict grading failures using the
same task_success / status_icon semantics as analysis.bench_report.

Two layers:
  1. Shared pure helpers in harness.grading are the single source of truth.
  2. analysis.bench_report.TaskResult and bench_runner._build_report both use
     them, so report.md can no longer look cleaner than the strict CSV.
"""

from __future__ import annotations

from typing import Any

from harness.bench_runner import ConversationTrace, _build_report
from harness.grading import (
    compute_action_pass,
    compute_status_icon,
    compute_task_success,
)

# -- shared helpers ----------------------------------------------------------


class TestComputeActionPass:
    def test_empty_expected_is_vacuously_true(self):
        assert compute_action_pass([], []) is True

    def test_all_expected_matched(self):
        assert compute_action_pass(["a", "b"], ["a", "b"]) is True

    def test_missing_expected_fails(self):
        assert compute_action_pass(["a", "b"], ["a"]) is False


class TestComputeTaskSuccess:
    def test_errored_is_none(self):
        assert compute_task_success(error="boom", action_pass=True) is None

    def test_action_miss_fails(self):
        assert compute_task_success(error=None, action_pass=False) is False

    def test_strict_fail_blocks(self):
        assert (
            compute_task_success(error=None, action_pass=True, args_passed=False)
            is False
        )
        assert (
            compute_task_success(error=None, action_pass=True, order_passed=False)
            is False
        )
        assert (
            compute_task_success(error=None, action_pass=True, collateral_passed=False)
            is False
        )

    def test_all_pass(self):
        assert (
            compute_task_success(
                error=None,
                action_pass=True,
                state_passed=True,
                args_passed=True,
                response_passed=True,
                order_passed=True,
                collateral_passed=True,
            )
            is True
        )

    def test_unasserted_none_dont_block(self):
        assert compute_task_success(error=None, action_pass=True) is True


class TestComputeStatusIcon:
    def test_error(self):
        assert compute_status_icon(error="x", stopped=True, action_pass=True) == "ERROR"

    def test_max_turns(self):
        assert (
            compute_status_icon(error=None, stopped=False, action_pass=True)
            == "MAX_TURNS"
        )

    def test_strict_icons(self):
        assert (
            compute_status_icon(
                error=None, stopped=True, action_pass=True, args_passed=False
            )
            == "ARGS_FAIL"
        )
        assert (
            compute_status_icon(
                error=None, stopped=True, action_pass=True, response_passed=False
            )
            == "RESPONSE_FAIL"
        )
        assert (
            compute_status_icon(
                error=None, stopped=True, action_pass=True, order_passed=False
            )
            == "ORDER_FAIL"
        )
        assert (
            compute_status_icon(
                error=None, stopped=True, action_pass=True, collateral_passed=False
            )
            == "COLLATERAL_FAIL"
        )

    def test_pass(self):
        assert compute_status_icon(error=None, stopped=True, action_pass=True) == "PASS"


# -- bench_report.TaskResult delegates to the shared helpers -----------------


def test_taskresult_matches_shared_helpers():
    from analysis.bench_report import TaskResult

    t = TaskResult(
        task_id="t",
        trial=0,
        num_turns=2,
        stopped=True,
        elapsed_seconds=1.0,
        tools_called=["a"],
        expected_actions=["a"],
        actions_matched=["a"],
        topics_visited=[],
        handoffs=[],
        state_passed=None,
        state_failures=[],
        error=None,
        args_passed=False,
    )
    assert t.task_success is False
    assert t.status_icon == "ARGS_FAIL"


# -- report.md surfaces strict failures --------------------------------------


def _trace(**kw) -> ConversationTrace:
    base: dict[str, Any] = dict(
        task_id="t1",
        expected_actions=["a"],
        actions_matched=["a"],
        tools_called=["a"],
        stopped=True,
        num_turns=2,
        elapsed_seconds=1.0,
    )
    base.update(kw)
    return ConversationTrace(**base)


_META = {
    "agent_model": "m",
    "agent_provider": "p",
    "agent_temperature": 0.0,
    "agent_max_tokens": 16384,
    "agent_reasoning_effort": "low",
    "user_sim_model": "u",
    "user_sim_strategy": "react",
    "user_sim_temperature": 0.0,
    "user_sim_max_tokens": 1086,
    "user_sim_reasoning_effort": "low",
    "mode": "Burr (state-machine ReAct)",
    "max_turns": 15,
    "task_timeout_seconds": 1200,
    "agent_step_timeout_seconds": 240,
    "user_sim_step_timeout_seconds": 90,
    "tasks_path": "tasks.yaml",
    "config_path": "cfg.yaml",
    "timestamp": "now",
}


def test_report_shows_args_fail():
    report = _build_report([_trace(args_passed=False)], _META)
    assert "ARGS_FAIL" in report


def test_report_shows_pass_for_clean_task():
    report = _build_report([_trace(state_passed=True, args_passed=True)], _META)
    assert "PASS" in report


def test_report_shows_collateral_and_order_fail():
    report = _build_report(
        [
            _trace(task_id="c1", collateral_passed=False),
            _trace(task_id="o1", order_passed=False),
        ],
        _META,
    )
    assert "COLLATERAL_FAIL" in report
    assert "ORDER_FAIL" in report


def test_report_strict_failure_not_hidden_when_actions_match():
    # The whole point: actions matched, but a strict check failed — the report
    # must NOT present this as a clean success.
    report = _build_report([_trace(response_passed=False)], _META)
    assert "RESPONSE_FAIL" in report
