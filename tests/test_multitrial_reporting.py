"""P1.12: multi-trial and compare-mode reporting fixes.

1. The per-task matrix must aggregate ALL trials of a (model, task), not show
   the first matching trial.
2. Failure analysis must report a count of failing *trials* with the matching
   denominator, not mislabel it as "models".
3. `--compare` in CSV mode must not silently emit a non-comparison; reject it
   unless --markdown is given.
"""

from __future__ import annotations

from analysis.bench_report import ModelRun, TaskResult, aggregate_task_cell


def _t(task_id, *, ok, trial=0, error=None):
    return TaskResult(
        task_id=task_id,
        trial=trial,
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


class TestAggregateTaskCell:
    def test_all_trials_pass(self):
        trials = [_t("x", ok=True, trial=0), _t("x", ok=True, trial=1)]
        cell = aggregate_task_cell(trials)
        assert "2/2" in cell  # 2 of 2 trials passed

    def test_some_trials_fail(self):
        trials = [_t("x", ok=True, trial=0), _t("x", ok=False, trial=1)]
        cell = aggregate_task_cell(trials)
        assert "1/2" in cell

    def test_errored_trials_noted(self):
        trials = [_t("x", ok=True, trial=0), _t("x", ok=False, trial=1, error="503")]
        cell = aggregate_task_cell(trials)
        # 1 of 1 scored trial passed, with 1 error surfaced.
        assert "1/1" in cell
        assert "err" in cell.lower()

    def test_single_trial_compact(self):
        cell = aggregate_task_cell([_t("x", ok=True)])
        assert "1/1" in cell


class TestFailureTrialCounting:
    def test_failure_groups_count_trials_not_models(self):
        # One model, one task, 3 trials, 2 failing.
        run = ModelRun(
            "m",
            "ts",
            tasks=[
                _t("x", ok=True, trial=0),
                _t("x", ok=False, trial=1),
                _t("x", ok=False, trial=2),
            ],
        )
        from analysis.bench_report import failing_trials_by_task

        groups = failing_trials_by_task([run])
        # 2 failing trials recorded for task x (not capped at "1 model")
        assert len(groups["x"]) == 2


class TestCompareRequiresMarkdown:
    def test_compare_without_markdown_exits(self, monkeypatch):
        import sys

        import pytest as _pytest

        from analysis import bench_report

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "bench_report",
                "csm_full_test",
                "--compare",
                "20200101_000000",
                "20200102_000000",
            ],
        )
        # argparse error() raises SystemExit(2)
        with _pytest.raises(SystemExit) as exc:
            bench_report.main()
        assert exc.value.code == 2
