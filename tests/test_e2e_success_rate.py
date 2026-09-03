"""P1.2: an end-to-end success rate where errors count as FAILURES, for
production gating — reported alongside (not replacing) the error-excluding TSR.
"""

from __future__ import annotations

from analysis.bench_report import ModelRun, TaskResult, generate_summary_csv


def _task(task_id, *, ok=True, error=None):
    return TaskResult(
        task_id=task_id,
        trial=0,
        num_turns=2,
        stopped=True,
        elapsed_seconds=1.0,
        tools_called=["a"],
        expected_actions=["a"],
        actions_matched=["a"] if ok else [],
        topics_visited=[],
        handoffs=[],
        state_passed=None,
        state_failures=[],
        error=error,
    )


def test_end_to_end_counts_errors_as_failures():
    run = ModelRun(
        "m",
        "ts",
        tasks=[
            _task("t1", ok=True),  # pass
            _task("t2", ok=False),  # fail
            _task(
                "t3", ok=False, error="503"
            ),  # error → excluded from TSR, FAIL for e2e
        ],
    )
    # TSR excludes the errored task: 1 / 2 = 0.5
    assert run.success_rate == 0.5
    # End-to-end counts it as a failure: 1 / 3
    assert run.end_to_end_success_rate == 1 / 3


def test_e2e_equals_tsr_when_no_errors():
    run = ModelRun("m", "ts", tasks=[_task("t1", ok=True), _task("t2", ok=False)])
    assert run.end_to_end_success_rate == run.success_rate == 0.5


def test_e2e_zero_for_empty_run():
    run = ModelRun("m", "ts", tasks=[])
    assert run.end_to_end_success_rate == 0.0


def test_summary_csv_includes_e2e_column():
    # t1 pass, t2 error → TSR = 1/1 scored = 100%; e2e = 1/2 total = 50%.
    run = ModelRun(
        "m", "ts", tasks=[_task("t1", ok=True), _task("t2", ok=False, error="x")]
    )
    csv = generate_summary_csv([run], "csm_full_test")
    header = csv.splitlines()[0].split(",")
    row = dict(zip(header, csv.splitlines()[1].split(",")))
    assert "end_to_end_success_rate" in header
    assert row["success_rate"] == "100%"
    assert row["end_to_end_success_rate"] == "50%"
