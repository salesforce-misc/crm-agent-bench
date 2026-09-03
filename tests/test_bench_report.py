"""Tests for the headline Task Success Rate (TSR) metric in bench_report."""

from __future__ import annotations

from analysis.bench_report import ModelRun, TaskResult


def _task(
    task_id: str = "t",
    *,
    trial: int = 0,
    expected: list[str] | None = None,
    matched: list[str] | None = None,
    tools_called: list[str] | None = None,
    state_passed: bool | None = None,
    args_passed: bool | None = None,
    response_passed: bool | None = None,
    order_passed: bool | None = None,
    collateral_passed: bool | None = None,
    difficulty: str | None = None,
    error: str | None = None,
    error_kind: str | None = None,
    stopped: bool = True,
    num_turns: int = 3,
    elapsed_seconds: float = 1.0,
) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        trial=trial,
        num_turns=num_turns,
        stopped=stopped,
        elapsed_seconds=elapsed_seconds,
        tools_called=list(
            tools_called if tools_called is not None else (matched or [])
        ),
        expected_actions=list(expected or []),
        actions_matched=list(matched or []),
        topics_visited=[],
        handoffs=[],
        state_passed=state_passed,
        state_failures=[],
        error=error,
        error_kind=error_kind,
        args_passed=args_passed,
        response_passed=response_passed,
        order_passed=order_passed,
        collateral_passed=collateral_passed,
        difficulty=difficulty,
    )


def test_success_requires_all_expected_actions():
    t = _task(expected=["a", "b"], matched=["a"])
    assert t.task_success is False


def test_success_when_actions_and_state_pass():
    t = _task(expected=["a"], matched=["a"], state_passed=True)
    assert t.task_success is True


def test_state_fail_overrides_action_pass():
    t = _task(expected=["a"], matched=["a"], state_passed=False)
    assert t.task_success is False


def test_no_state_check_still_passes_on_actions():
    t = _task(expected=["a"], matched=["a"], state_passed=None)
    assert t.task_success is True


def test_stopped_does_not_affect_success():
    # Outcome-based: an unfinished conversation that did the work still passes.
    t = _task(expected=["a"], matched=["a"], stopped=False)
    assert t.task_success is True


def test_errored_task_is_none_not_false():
    t = _task(expected=["a"], matched=[], error="Timeout")
    assert t.task_success is None


def test_task_with_no_expected_actions_passes():
    # action_pass is vacuously True when nothing is expected.
    t = _task(expected=[], matched=[])
    assert t.task_success is True


def test_success_rate_excludes_errored_from_denominator():
    run = ModelRun(
        model="m",
        timestamp="ts",
        tasks=[
            _task("t1", expected=["a"], matched=["a"]),  # pass
            _task("t2", expected=["a"], matched=[]),  # fail
            _task("t3", expected=["a"], matched=[], error="503"),  # excluded
        ],
    )
    assert run.succeeded == 1
    assert run.scored == 2  # errored task dropped
    assert run.success_rate == 0.5


def test_success_rate_zero_when_all_errored():
    run = ModelRun(
        model="m",
        timestamp="ts",
        tasks=[_task("t1", expected=["a"], matched=[], error="x")],
    )
    assert run.scored == 0
    assert run.success_rate == 0.0


# -- pass^k -----------------------------------------------------------------


def _trials(task_id: str, outcomes: list[bool]) -> list[TaskResult]:
    """Build N trials for one task; outcomes[i]=True means that trial passed."""
    out = []
    for i, ok in enumerate(outcomes):
        out.append(
            _task(
                task_id,
                trial=i,
                expected=["a"],
                matched=["a"] if ok else [],
            )
        )
    return out


def test_pass_hat_1_equals_success_rate_single_trial():
    run = ModelRun(
        "m",
        "ts",
        tasks=[
            _task("t1", expected=["a"], matched=["a"]),
            _task("t2", expected=["a"], matched=[]),
        ],
    )
    assert run.num_trials == 1
    assert run.pass_hat_k(1) == 0.5
    assert run.pass_hat_k(1) == run.success_rate
    assert run.pass_hat_k(2) is None  # no task has 2 trials


def test_pass_hat_k_all_pass_is_one():
    run = ModelRun("m", "ts", tasks=_trials("t1", [True, True, True]))
    assert run.num_trials == 3
    assert run.pass_hat_k(1) == 1.0
    assert run.pass_hat_k(3) == 1.0


def test_pass_hat_k_decays_with_inconsistency():
    # 2 of 3 trials pass: pass^1 = 2/3, pass^2 = C(2,2)/C(3,2) = 1/3, pass^3 = 0.
    run = ModelRun("m", "ts", tasks=_trials("t1", [True, True, False]))
    assert run.pass_hat_k(1) == 2 / 3
    assert run.pass_hat_k(2) == 1 / 3
    assert run.pass_hat_k(3) == 0.0


def test_pass_hat_k_averages_across_tasks():
    # t1: always passes (pass^2 = 1). t2: 1/2 pass (pass^2 = 0). Mean = 0.5.
    run = ModelRun(
        "m", "ts", tasks=(_trials("t1", [True, True]) + _trials("t2", [True, False]))
    )
    assert run.pass_hat_k(2) == 0.5


def test_pass_hat_k_drops_errored_trials_from_n():
    # 3 trials: pass, pass, error. n=2 scored, c=2 → pass^2 = 1.0.
    tasks = [
        _task("t1", trial=0, expected=["a"], matched=["a"]),
        _task("t1", trial=1, expected=["a"], matched=["a"]),
        _task("t1", trial=2, expected=["a"], matched=[], error="503"),
    ]
    run = ModelRun("m", "ts", tasks=tasks)
    assert run.pass_hat_k(2) == 1.0


# -- efficiency -------------------------------------------------------------


def test_avg_tool_calls_excludes_errored():
    run = ModelRun(
        "m",
        "ts",
        tasks=[
            _task("t1", tools_called=["a", "b"]),
            _task("t2", tools_called=["a", "b", "c", "d"]),
            _task("t3", tools_called=[], error="x"),  # excluded
        ],
    )
    assert run.avg_tool_calls == 3.0


def test_avg_time_per_task_excludes_errored():
    run = ModelRun(
        "m",
        "ts",
        tasks=[
            _task("t1", elapsed_seconds=10.0),
            _task("t2", elapsed_seconds=20.0),
            _task("t3", elapsed_seconds=999.0, error="x"),  # excluded
        ],
    )
    assert run.avg_time_per_task == 15.0


# -- stricter checks gate task_success --------------------------------------


def test_args_fail_blocks_success():
    t = _task(expected=["a"], matched=["a"], args_passed=False)
    assert t.task_success is False
    assert t.status_icon == "ARGS_FAIL"


def test_response_fail_blocks_success():
    t = _task(expected=["a"], matched=["a"], response_passed=False)
    assert t.task_success is False
    assert t.status_icon == "RESPONSE_FAIL"


def test_order_fail_blocks_success():
    t = _task(expected=["a"], matched=["a"], order_passed=False)
    assert t.task_success is False
    assert t.status_icon == "ORDER_FAIL"


def test_collateral_fail_blocks_success():
    t = _task(expected=["a"], matched=["a"], collateral_passed=False)
    assert t.task_success is False
    assert t.status_icon == "COLLATERAL_FAIL"


def test_all_stricter_checks_pass():
    t = _task(
        expected=["a"],
        matched=["a"],
        state_passed=True,
        args_passed=True,
        response_passed=True,
        order_passed=True,
        collateral_passed=True,
    )
    assert t.task_success is True
    assert t.status_icon == "PASS"


def test_unasserted_checks_dont_block_success():
    # All None (not asserted) → still passes on actions alone.
    t = _task(expected=["a"], matched=["a"])
    assert t.task_success is True


# -- model-fault vs infra errors --------------------------------------------


def test_infra_error_is_excluded_from_denominator():
    # A timeout (no error_kind) stays None → dropped from the success rate.
    t = _task(expected=["a"], matched=[], error="APITimeoutError: timed out")
    assert t.task_success is None
    assert t.is_infra_error is True
    assert t.is_model_fault_error is False
    assert t.status_icon == "ERROR"


def test_hallucinated_tool_error_is_graded_as_failure():
    # A hallucinated-tool error is a model fault → FALSE, stays in denominator,
    # even though the actions it did make all matched before the bad call.
    t = _task(
        expected=["a"],
        matched=["a"],
        error="ValueError: Tool 'X' not found in available tools. Available tools: a",
        error_kind="hallucinated_tool",
    )
    assert t.task_success is False
    assert t.is_model_fault_error is True
    assert t.is_infra_error is False
    assert t.status_icon == "HALLUCINATED_TOOL"


def test_model_fault_error_counts_in_denominator_infra_does_not():
    run = ModelRun(
        model="m",
        timestamp="ts",
        tasks=[
            _task("t1", expected=["a"], matched=["a"]),  # pass
            _task(
                "t2",
                expected=["a"],
                matched=["a"],  # model-fault fail
                error="ValueError: Tool 'X' not found in available tools",
                error_kind="hallucinated_tool",
            ),
            _task("t3", expected=["a"], matched=[], error="503"),  # infra, excluded
        ],
    )
    assert run.succeeded == 1
    assert run.scored == 2  # pass + hallucinated; infra dropped
    assert run.errored == 1  # only the infra error
    assert run.success_rate == 0.5  # 1/2, not 1/1 or 1/3


# -- difficulty strata & turn splits ----------------------------------------


def test_success_by_difficulty():
    run = ModelRun(
        "m",
        "ts",
        tasks=[
            _task("e1", expected=["a"], matched=["a"], difficulty="easy"),
            _task("e2", expected=["a"], matched=["a"], difficulty="easy"),
            _task("h1", expected=["a"], matched=[], difficulty="hard"),
            _task("h2", expected=["a"], matched=["a"], difficulty="hard"),
        ],
    )
    by = run.success_by_difficulty()
    assert by["easy"] == (2, 2)
    assert by["hard"] == (1, 2)
