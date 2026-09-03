"""Tests for token-cost-per-task reporting.

Covers ModelRun cost aggregates, back-compat loading of traces.json without
the new keys, and the new CSV columns / markdown rendering degrading
gracefully for unpriced runs.
"""

from __future__ import annotations

import json

from analysis.bench_report import (
    ModelRun,
    Report,
    TaskResult,
    generate_cost_efficiency,
    generate_detail_csv,
    generate_scorecard,
    generate_summary_csv,
    load_run,
)


def _task(
    task_id: str = "t",
    *,
    trial: int = 0,
    expected: list[str] | None = None,
    matched: list[str] | None = None,
    error: str | None = None,
    error_kind: str | None = None,
    agent_usage: dict | None = None,
    agent_cost_usd: float | None = None,
    agent_llm_seconds: float = 0.0,
) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        trial=trial,
        num_turns=3,
        stopped=True,
        elapsed_seconds=1.0,
        tools_called=list(matched or []),
        expected_actions=list(expected or []),
        actions_matched=list(matched or []),
        topics_visited=[],
        handoffs=[],
        state_passed=None,
        state_failures=[],
        error=error,
        error_kind=error_kind,
        agent_usage=agent_usage or {},
        agent_cost_usd=agent_cost_usd,
        agent_llm_seconds=agent_llm_seconds,
    )


# ---------------------------------------------------------------------------
# Agent-only latency aggregates
# ---------------------------------------------------------------------------


def test_output_tokens_per_second_excludes_infra_errors():
    run = ModelRun(
        model="m",
        timestamp="ts",
        tasks=[
            _task(
                "t1",
                expected=["a"],
                matched=["a"],
                agent_usage={
                    "num_calls": 2,
                    "completion_tokens": 200,
                    "total_tokens": 400,
                },
                agent_llm_seconds=4.0,
            ),  # 200 tok, 4s
            _task(
                "t2",
                expected=["a"],
                matched=[],
                agent_usage={
                    "num_calls": 3,
                    "completion_tokens": 100,
                    "total_tokens": 300,
                },
                agent_llm_seconds=6.0,
            ),  # 100 tok, 6s
            _task(
                "t3",
                expected=["a"],
                matched=[],
                error="503",
                agent_usage={
                    "num_calls": 5,
                    "completion_tokens": 999,
                    "total_tokens": 999,
                },
                agent_llm_seconds=99.0,
            ),  # infra error -> excluded entirely
        ],
    )
    # Non-errored only: 300 completion tokens over 10s -> 30 tok/s.
    # 10s over 2 tasks -> 5.0 s/task.
    assert run.total_agent_llm_seconds == 10.0
    assert run.total_agent_completion_tokens_scored == 300
    assert run.output_tokens_per_second == 30.0
    assert run.agent_llm_seconds_per_task == 5.0


def test_latency_metrics_none_for_pre_feature_traces():
    # No agent_llm_seconds recorded (all default 0.0) -> render "—", not fake 0.
    run = ModelRun(
        model="m",
        timestamp="ts",
        tasks=[
            _task(
                "t1",
                expected=["a"],
                matched=["a"],
                agent_usage={"num_calls": 2, "completion_tokens": 200},
            )
        ],
    )
    assert run.output_tokens_per_second is None
    assert run.agent_llm_seconds_per_task is None


def test_summary_csv_has_latency_columns():
    run = ModelRun(
        model="m",
        timestamp="ts",
        tasks=[
            _task(
                "t1",
                expected=["a"],
                matched=["a"],
                agent_usage={"num_calls": 2, "completion_tokens": 200},
                agent_llm_seconds=4.0,
            )
        ],
    )
    csv_text = generate_summary_csv([run], "csm_full_test")
    header = csv_text.splitlines()[0]
    assert "agent_output_tokens_per_second" in header
    assert "agent_llm_seconds_per_task" in header
    # 200 tok / 4s = 50.0 tok/s.
    assert "50.0" in csv_text.splitlines()[1]


# ---------------------------------------------------------------------------
# ModelRun cost aggregates
# ---------------------------------------------------------------------------


def test_cost_aggregates_hand_computed():
    run = ModelRun(
        model="m",
        timestamp="ts",
        tasks=[
            _task(
                "t1",
                expected=["a"],
                matched=["a"],
                agent_usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
                agent_cost_usd=0.01,
            ),  # success
            _task(
                "t2",
                expected=["a"],
                matched=[],
                agent_usage={
                    "prompt_tokens": 200,
                    "completion_tokens": 20,
                    "total_tokens": 220,
                },
                agent_cost_usd=0.02,
            ),  # fail (scored, not success)
            _task(
                "t3",
                expected=["a"],
                matched=[],
                error="503",
                agent_usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "total_tokens": 11,
                },
                agent_cost_usd=0.001,
            ),  # infra error, excluded from scored/succeeded
        ],
    )

    # succeeded=1, scored=2 (t3 dropped) -- same denominator as success_rate.
    assert run.succeeded == 1
    assert run.scored == 2

    assert run.total_agent_cost_usd == 0.01 + 0.02 + 0.001
    assert run.total_agent_tokens == 150 + 220 + 11

    # cost_per_attempted uses `scored` (2), NOT total (3) or successes (1).
    assert run.cost_per_attempted_task_usd == (0.01 + 0.02 + 0.001) / 2

    # cost_per_successful uses `succeeded` (1) -- the hero metric.
    assert run.cost_per_successful_task_usd == (0.01 + 0.02 + 0.001) / 1


def test_cost_per_successful_excludes_failed_tasks_from_numerator_semantics():
    # Two tasks, one succeeds one fails. Total cost is the SUM over both (the
    # failure's tokens were still spent) but the denominator is successes
    # only -- this is exactly the "effective cost" the design doc calls out.
    run = ModelRun(
        model="m",
        timestamp="ts",
        tasks=[
            _task("t1", expected=["a"], matched=["a"], agent_cost_usd=1.0),  # success
            _task("t2", expected=["a"], matched=[], agent_cost_usd=1.0),  # fail
        ],
    )
    assert run.succeeded == 1
    assert run.scored == 2
    assert run.total_agent_cost_usd == 2.0
    assert run.cost_per_attempted_task_usd == 1.0
    assert run.cost_per_successful_task_usd == 2.0  # gap vs. attempted


def test_cost_per_successful_task_is_none_when_zero_successes():
    run = ModelRun(
        model="m",
        timestamp="ts",
        tasks=[_task("t1", expected=["a"], matched=[], agent_cost_usd=5.0)],
    )
    assert run.succeeded == 0
    assert run.cost_per_successful_task_usd is None
    # attempted is still defined (scored=1, cost=5.0).
    assert run.cost_per_attempted_task_usd == 5.0


def test_unpriced_run_aggregates_are_none():
    run = ModelRun(
        model="m",
        timestamp="ts",
        tasks=[
            _task("t1", expected=["a"], matched=["a"], agent_cost_usd=None),
            _task("t2", expected=["a"], matched=["a"], agent_cost_usd=None),
        ],
    )
    assert run.total_agent_cost_usd is None
    assert run.cost_per_attempted_task_usd is None
    assert run.cost_per_successful_task_usd is None


def test_task_result_token_properties_default_to_zero():
    t = _task("t1")
    assert t.agent_prompt_tokens == 0
    assert t.agent_completion_tokens == 0
    assert t.agent_cache_read_tokens == 0
    assert t.agent_total_tokens == 0

    t2 = _task(
        "t2",
        agent_usage={
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "cache_read_tokens": 3,
            "cache_write_tokens": 2,
            "total_tokens": 15,
        },
    )
    assert t2.agent_prompt_tokens == 10
    assert t2.agent_completion_tokens == 5
    assert t2.agent_cache_read_tokens == 3
    assert t2.agent_total_tokens == 15


# ---------------------------------------------------------------------------
# Back-compat: old traces.json without the new keys
# ---------------------------------------------------------------------------


def test_load_run_backcompat_missing_cost_keys(tmp_path, monkeypatch):
    # A legacy trace dict has none of agent_usage/agent_cost_usd/user_sim_cost_usd.
    old_trace = {
        "task_id": "t1",
        "trial": 0,
        "num_turns": 2,
        "stopped": True,
        "elapsed_seconds": 3.5,
        "tools_called": ["a"],
        "expected_actions": ["a"],
        "actions_matched": ["a"],
        "topics_visited": [],
        "handoffs": [],
        "state_passed": None,
        "state_failures": [],
        "error": None,
    }

    import analysis.bench_report as br

    run_dir = tmp_path / "bench_output" / "prefix_modelX_20260101_000000"
    run_dir.mkdir(parents=True)
    (run_dir / "traces.json").write_text(json.dumps([old_trace]))
    monkeypatch.setattr(br, "BENCH_OUTPUT", tmp_path / "bench_output")

    run = load_run("prefix", "modelX", "20260101_000000")
    assert run is not None
    assert len(run.tasks) == 1
    t = run.tasks[0]
    assert t.agent_usage == {}
    assert t.agent_cost_usd is None
    assert t.user_sim_cost_usd is None
    assert t.agent_total_tokens == 0

    # Aggregates and report generation must not crash on an all-unpriced run.
    assert run.total_agent_cost_usd is None
    csv_summary = generate_summary_csv([run], "prefix")
    csv_detail = generate_detail_csv([run], "prefix")
    assert "total_agent_cost_usd" in csv_summary
    assert "agent_cost_usd" in csv_detail

    report = Report()
    generate_scorecard(report, [run])
    generate_cost_efficiency(report, [run])  # should be a no-op (all unpriced)
    rendered = report.render()
    assert "—" in rendered  # blank cost rendered as em-dash, not a crash


# ---------------------------------------------------------------------------
# CSV column presence + values
# ---------------------------------------------------------------------------


def test_detail_csv_has_new_columns_and_values():
    run = ModelRun(
        model="m",
        timestamp="ts",
        tasks=[
            _task(
                "t1",
                expected=["a"],
                matched=["a"],
                agent_usage={
                    "prompt_tokens": 11,
                    "completion_tokens": 22,
                    "cache_read_tokens": 3,
                    "total_tokens": 36,
                },
                agent_cost_usd=0.1234,
            ),
        ],
    )
    csv_text = generate_detail_csv([run], "csm")
    header, row = csv_text.strip().splitlines()[:2]
    headers = header.split(",")
    for col in (
        "agent_prompt_tokens",
        "agent_completion_tokens",
        "agent_cache_read_tokens",
        "agent_total_tokens",
        "agent_cost_usd",
    ):
        assert col in headers
    # Existing columns must still be present and in front (append-only).
    assert headers.index("elapsed_seconds") < headers.index("agent_prompt_tokens")

    row_cells = row.split(",")
    idx = {h: i for i, h in enumerate(headers)}
    assert row_cells[idx["agent_prompt_tokens"]] == "11"
    assert row_cells[idx["agent_completion_tokens"]] == "22"
    assert row_cells[idx["agent_cache_read_tokens"]] == "3"
    assert row_cells[idx["agent_total_tokens"]] == "36"
    assert row_cells[idx["agent_cost_usd"]] == "0.1234"


def test_detail_csv_blank_cost_for_unpriced_task():
    run = ModelRun(
        model="m", timestamp="ts", tasks=[_task("t1", expected=["a"], matched=["a"])]
    )
    csv_text = generate_detail_csv([run], "csm")
    header, row = csv_text.strip().splitlines()[:2]
    headers = header.split(",")
    idx = {h: i for i, h in enumerate(headers)}
    row_cells = row.split(",")
    assert row_cells[idx["agent_cost_usd"]] == ""
    assert row_cells[idx["agent_total_tokens"]] == "0"


def test_summary_csv_has_new_columns_and_values():
    run = ModelRun(
        model="m",
        timestamp="ts",
        tasks=[
            _task(
                "t1",
                expected=["a"],
                matched=["a"],
                agent_cost_usd=1.0,
                agent_usage={"total_tokens": 100},
            ),
            _task(
                "t2",
                expected=["a"],
                matched=[],
                agent_cost_usd=1.0,
                agent_usage={"total_tokens": 50},
            ),
        ],
    )
    csv_text = generate_summary_csv([run], "csm")
    header, row = csv_text.strip().splitlines()[:2]
    headers = header.split(",")
    for col in (
        "total_agent_cost_usd",
        "cost_per_attempted_task_usd",
        "cost_per_successful_task_usd",
        "total_agent_tokens",
    ):
        assert col in headers
    idx = {h: i for i, h in enumerate(headers)}
    row_cells = row.split(",")
    assert row_cells[idx["total_agent_cost_usd"]] == "2.0000"
    assert row_cells[idx["cost_per_attempted_task_usd"]] == "1.0000"  # 2.0 / scored=2
    assert (
        row_cells[idx["cost_per_successful_task_usd"]] == "2.0000"
    )  # 2.0 / succeeded=1
    assert row_cells[idx["total_agent_tokens"]] == "150"


def test_summary_csv_blank_for_unpriced_run():
    run = ModelRun(
        model="m", timestamp="ts", tasks=[_task("t1", expected=["a"], matched=["a"])]
    )
    csv_text = generate_summary_csv([run], "csm")
    header, row = csv_text.strip().splitlines()[:2]
    headers = header.split(",")
    idx = {h: i for i, h in enumerate(headers)}
    row_cells = row.split(",")
    assert row_cells[idx["total_agent_cost_usd"]] == ""
    assert row_cells[idx["cost_per_attempted_task_usd"]] == ""
    assert row_cells[idx["cost_per_successful_task_usd"]] == ""


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def test_scorecard_renders_cost_columns():
    run = ModelRun(
        model="m",
        timestamp="ts",
        tasks=[_task("t1", expected=["a"], matched=["a"], agent_cost_usd=0.5)],
    )
    report = Report()
    generate_scorecard(report, [run])
    rendered = report.render()
    assert "Cost/Success" in rendered
    assert "Total Cost" in rendered
    assert "$0.5000" in rendered


def test_scorecard_shows_dash_for_unpriced():
    run = ModelRun(
        model="m", timestamp="ts", tasks=[_task("t1", expected=["a"], matched=["a"])]
    )
    report = Report()
    generate_scorecard(report, [run])
    rendered = report.render()
    assert "—" in rendered


def test_cost_efficiency_section_skipped_when_all_unpriced():
    run = ModelRun(
        model="m", timestamp="ts", tasks=[_task("t1", expected=["a"], matched=["a"])]
    )
    report = Report()
    generate_cost_efficiency(report, [run])
    assert report.render() == ""


def test_cost_efficiency_section_reports_gap():
    run = ModelRun(
        model="m",
        timestamp="ts",
        tasks=[
            _task("t1", expected=["a"], matched=["a"], agent_cost_usd=1.0),  # success
            _task("t2", expected=["a"], matched=[], agent_cost_usd=1.0),  # fail
        ],
    )
    report = Report()
    generate_cost_efficiency(report, [run])
    rendered = report.render()
    assert "Cost Efficiency" in rendered
    assert "2.00x" in rendered  # cost_per_success (2.0) / cost_per_attempted (1.0)
