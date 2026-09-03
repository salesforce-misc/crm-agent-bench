#!/usr/bin/env python3
"""Analyze benchmark runs and generate comparison reports.

Auto-discovers models and timestamps from bench_output directory structure.
No hardcoded model lists — works with any agent/model combination.

Default output is CSV (summary + detail). Use --markdown for markdown reports.

Usage:
    # CSV output (default) — writes summary + detail CSVs:
    python -m analysis.bench_report csm_full_test
    python -m analysis.bench_report csm_full_test -o results/report.csv

    # Markdown output:
    python -m analysis.bench_report csm_full_test --markdown
    python -m analysis.bench_report csm_full_test --markdown -o results/analysis.md

    # Compare two runs:
    python -m analysis.bench_report csm_full_test --compare 20260501_200110 20260502_112910

    # Analyze a specific timestamp:
    python -m analysis.bench_report csm_full_test --timestamp 20260502_112910

    # Include conversation excerpts for failed tasks (markdown only):
    python -m analysis.bench_report csm_full_test --markdown -v

    # Filter to specific models:
    python -m analysis.bench_report csm_full_test --models sonnet4 haiku45
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from math import comb
from pathlib import Path

from harness.grading import (
    compute_action_pass,
    compute_status_icon,
    compute_task_success,
    is_model_fault_error,
)

# Root directory holding benchmark run outputs. Benchmark runs write to the
# current working directory by default, so discovery must resolve relative to
# the CWD (or an explicit override) — NOT relative to this module's location,
# which lands inside site-packages when installed as a wheel. Override with the
# BENCH_OUTPUT env var or the --bench-output CLI flag.
BENCH_OUTPUT = Path(os.environ.get("BENCH_OUTPUT", "bench_output"))

AGENT_DISPLAY = {
    "csm": "Store Manager",
    "customer_service": "Customer Service",
    "field_service": "Field Service",
    "marketing": "Marketing",
    "sdr": "Sales Development Rep",
}


MODEL_DISPLAY = {
    "sonnet4": "Sonnet 4",
    "haiku45": "Haiku 4.5",
    "gpt55": "GPT-5.5",
    "gemini3flash": "Gemini 3 Flash",
    "gemini31flite": "Gemini 3.1 Lite",
    "gemini32flash": "Gemini 3.2 Flash",
    "gpt41": "GPT-4.1",
    "nemotron3120b": "Nemotron 3 Super 120B",
    "sonnet46": "Claude 4.6 Sonnet",
}


def agent_display_name(prefix: str) -> str:
    key = prefix.replace("_full_test", "").replace("_", " ")
    return AGENT_DISPLAY.get(key.replace(" ", "_"), key)


@dataclass
class TaskResult:
    task_id: str
    trial: int
    num_turns: int
    stopped: bool
    elapsed_seconds: float
    tools_called: list[str]
    expected_actions: list[str]
    actions_matched: list[str]
    topics_visited: list[str]
    handoffs: list[str]
    state_passed: bool | None
    state_failures: list[str]
    error: str | None
    error_kind: str | None = None
    # Carried through from the trace so offline analysis treats a timeout the
    # same way the built-in report does (P: timeout parity). Defaults False so
    # pre-feature traces.json remains loadable.
    timed_out: bool = False
    turns: list[dict] = field(default_factory=list)
    # Stricter grading results (None = not asserted for this task):
    args_passed: bool | None = None
    args_failures: list[str] = field(default_factory=list)
    response_passed: bool | None = None
    response_failures: list[str] = field(default_factory=list)
    order_passed: bool | None = None
    order_failures: list[str] = field(default_factory=list)
    collateral_passed: bool | None = None
    collateral_failures: list[str] = field(default_factory=list)
    forbidden_passed: bool | None = None
    forbidden_failures: list[str] = field(default_factory=list)
    difficulty: str | None = None
    # Token cost per completed task. Defaults keep old traces.json loadable:
    # missing keys -> empty usage / no cost.
    agent_usage: dict = field(default_factory=dict)
    agent_cost_usd: float | None = None
    user_sim_cost_usd: float | None = None
    # Agent-only inference wall-clock (seconds), excluding the user simulator,
    # tool execution, and grading. 0.0 for pre-feature traces.
    agent_llm_seconds: float = 0.0

    @property
    def agent_prompt_tokens(self) -> int:
        return self.agent_usage.get("prompt_tokens", 0)

    @property
    def agent_completion_tokens(self) -> int:
        return self.agent_usage.get("completion_tokens", 0)

    @property
    def agent_cache_read_tokens(self) -> int:
        return self.agent_usage.get("cache_read_tokens", 0)

    @property
    def agent_total_tokens(self) -> int:
        return self.agent_usage.get("total_tokens", 0)

    @property
    def action_pass(self) -> bool:
        return compute_action_pass(self.expected_actions, self.actions_matched)

    @property
    def is_model_fault_error(self) -> bool:
        """True if the trace errored due to a *model* fault, not infrastructure.

        A hallucinated tool call (the model invoked a tool that does not exist)
        is the model's failure to complete the task, so it is graded as a
        failure rather than excluded from the denominator like a transient infra
        error (gateway timeout, disconnect). Keyed off the structured
        ``error_kind`` tag set by the harness.
        """
        return is_model_fault_error(self.error, self.error_kind)

    @property
    def is_infra_error(self) -> bool:
        """True if the trace errored for a reason that is NOT the model's fault.

        Infra errors (timeouts, disconnects) are excluded from the success-rate
        denominator and the descriptive averages. Model-fault errors are not:
        they are graded as failures with their real (partial) trace data.
        """
        return bool(self.error) and not self.is_model_fault_error

    @property
    def task_success(self) -> bool | None:
        """Strict per-task binary success — see harness.grading.compute_task_success.

        Outcome-based: None for infra-errored tasks (dropped from the
        success-rate denominator); a model-fault error (hallucinated tool)
        returns False and stays in the denominator. Otherwise True iff actions
        pass and no asserted strict check (state/args/response/order/collateral/
        forbidden) failed. Conversation completion and turn count are excluded —
        efficiency, not correctness.
        """
        return compute_task_success(
            error=self.error,
            error_kind=self.error_kind,
            action_pass=self.action_pass,
            state_passed=self.state_passed,
            args_passed=self.args_passed,
            response_passed=self.response_passed,
            order_passed=self.order_passed,
            collateral_passed=self.collateral_passed,
            forbidden_passed=self.forbidden_passed,
        )

    @property
    def status_icon(self) -> str:
        return compute_status_icon(
            error=self.error,
            error_kind=self.error_kind,
            stopped=self.stopped,
            action_pass=self.action_pass,
            state_passed=self.state_passed,
            args_passed=self.args_passed,
            response_passed=self.response_passed,
            order_passed=self.order_passed,
            collateral_passed=self.collateral_passed,
            forbidden_passed=self.forbidden_passed,
        )


@dataclass
class ModelRun:
    model: str
    timestamp: str
    tasks: list[TaskResult]

    @property
    def total(self) -> int:
        return len(self.tasks)

    @property
    def completed(self) -> int:
        return sum(1 for t in self.tasks if t.stopped and not t.error)

    @property
    def errored(self) -> int:
        """Infra-errored tasks (excluded from the denominator). Model-fault
        errors are graded failures, not counted here."""
        return sum(1 for t in self.tasks if t.is_infra_error)

    @property
    def hit_max(self) -> int:
        return sum(1 for t in self.tasks if not t.stopped and not t.error)

    @property
    def actions_matched_total(self) -> int:
        return sum(len(t.actions_matched) for t in self.tasks)

    @property
    def actions_expected_total(self) -> int:
        return sum(len(t.expected_actions) for t in self.tasks)

    @property
    def state_passed_total(self) -> int:
        return sum(1 for t in self.tasks if t.state_passed is True)

    @property
    def state_total(self) -> int:
        return sum(1 for t in self.tasks if t.state_passed is not None)

    @property
    def succeeded(self) -> int:
        """Tasks that strictly passed (errored tasks excluded — they are None)."""
        return sum(1 for t in self.tasks if t.task_success is True)

    @property
    def scored(self) -> int:
        """Non-errored tasks — the denominator for the success rate."""
        return sum(1 for t in self.tasks if t.task_success is not None)

    @property
    def success_rate(self) -> float:
        """Task Success Rate (TSR) — the single headline metric, 0..1.

        Strict per-task binary success averaged over non-errored tasks.
        Errored tasks (flaky endpoints, timeouts) are excluded from the
        denominator and reported separately.
        """
        return self.succeeded / self.scored if self.scored else 0.0

    @property
    def end_to_end_success_rate(self) -> float:
        """Production-gate success rate, 0..1 — errors count as FAILURES.

        Unlike ``success_rate`` (TSR), which excludes errored tasks from the
        denominator to isolate model quality, this divides successes by the
        TOTAL number of tasks. An infrastructure outage that prevents a task
        from completing is a real end-to-end failure for gating purposes, so
        report both: TSR for model quality, this for go/no-go (P1.2).
        """
        return self.succeeded / self.total if self.total else 0.0

    @property
    def avg_turns(self) -> float:
        non_error = [t for t in self.tasks if not t.is_infra_error]
        if not non_error:
            return 0
        return sum(t.num_turns for t in non_error) / len(non_error)

    @property
    def total_time(self) -> float:
        return sum(t.elapsed_seconds for t in self.tasks if not t.is_infra_error)

    # -- Efficiency metrics (per-task averages over non-errored runs) --------

    @property
    def avg_tool_calls(self) -> float:
        """Average number of tool calls per non-errored run. Lower = leaner."""
        non_error = [t for t in self.tasks if not t.is_infra_error]
        if not non_error:
            return 0.0
        return sum(len(t.tools_called) for t in non_error) / len(non_error)

    @property
    def avg_time_per_task(self) -> float:
        """Average wall-clock seconds per non-errored run."""
        non_error = [t for t in self.tasks if not t.is_infra_error]
        if not non_error:
            return 0.0
        return sum(t.elapsed_seconds for t in non_error) / len(non_error)

    # -- Token cost per task -------------------------------------------------
    #
    # Cost is agent-under-test only (design decision #1): user-sim tokens are
    # tracked separately and excluded from these headline numbers. A task
    # with agent_cost_usd is None means "unpriced" (unknown model in
    # pricing.yaml, or an older trace) — never crash, degrade to None/"—".

    @property
    def total_agent_cost_usd(self) -> float | None:
        """Sum of agent_cost_usd over tasks that have a price.

        None (unpriced run) only when every task is unpriced. A run with a
        mix of priced/unpriced tasks still reports a (partial) total rather
        than silently going to None, since a single unknown alias shouldn't
        blank out an otherwise-priced run.
        """
        priced = [t.agent_cost_usd for t in self.tasks if t.agent_cost_usd is not None]
        if not priced:
            return None
        return sum(priced)

    @property
    def total_agent_tokens(self) -> int:
        return sum(t.agent_total_tokens for t in self.tasks)

    # -- Agent-only latency --------------------------------------------------
    #
    # A model-intrinsic latency signal that EXCLUDES the user simulator, tool
    # execution, and grading — unlike avg_time_per_task (total wall-clock),
    # which is polluted by the gpt-5-mini user sim's run-to-run speed. Derived
    # from per-task agent_llm_seconds (0.0 for pre-feature traces).
    #
    # HEADLINE = output_tokens_per_second (completion tokens / agent LLM secs).
    # Chosen over sec/generation because it is turn-count independent (a 3-call
    # and an 18-call task both reduce to "tokens produced per second") and uses
    # the same unit providers publish, so it is comparable to external numbers.
    # sec/generation is confounded by context growth: as a task loops, the
    # prompt grows and each later call is slower even at constant model speed,
    # which double-penalizes loopy models that turns/cost already capture.
    #
    # Caveat: we time whole calls, not streamed token timestamps, so prefill
    # (prompt processing) is lumped into the measured seconds. This is therefore
    # EFFECTIVE end-to-end output throughput, not pure decode speed — a model
    # with large prompts reads slightly slower per output token than its raw
    # decode rate. Consistent across the suite; isolating decode would need
    # per-call TTFT instrumentation.

    @property
    def total_agent_llm_seconds(self) -> float:
        """Agent-only inference wall-clock summed over non-errored runs."""
        return sum(t.agent_llm_seconds for t in self.tasks if not t.is_infra_error)

    @property
    def total_agent_completion_tokens_scored(self) -> int:
        """Agent completion tokens summed over non-errored runs (throughput
        numerator; scoped to non-errored so a dropped task's tokens/seconds
        don't leak into the rate)."""
        return sum(
            t.agent_completion_tokens for t in self.tasks if not t.is_infra_error
        )

    @property
    def output_tokens_per_second(self) -> float | None:
        """Effective agent output throughput: completion tokens / agent LLM
        seconds, pooled over non-errored runs.

        None when no agent latency/tokens were recorded (pre-feature traces or
        an untracked client) so the analyzer renders "—" rather than a fake 0.
        """
        secs = self.total_agent_llm_seconds
        toks = self.total_agent_completion_tokens_scored
        if secs <= 0 or toks <= 0:
            return None
        return toks / secs

    @property
    def agent_llm_seconds_per_task(self) -> float | None:
        """Mean agent-only inference time per non-errored task (secondary)."""
        non_error = [t for t in self.tasks if not t.is_infra_error]
        secs = self.total_agent_llm_seconds
        if not non_error or secs <= 0:
            return None
        return secs / len(non_error)

    @property
    def cost_per_attempted_task_usd(self) -> float | None:
        """Total agent cost / number of *attempted* (scored) tasks.

        Uses the same denominator as ``success_rate`` (``self.scored`` —
        non-errored tasks) so the attempted-vs-successful cost gap is an
        apples-to-apples comparison, not an artifact of different
        denominators.
        """
        if self.total_agent_cost_usd is None or not self.scored:
            return None
        return self.total_agent_cost_usd / self.scored

    @property
    def cost_per_successful_task_usd(self) -> float | None:
        """Total agent cost / number of successes (the hero metric, P0 of the
        design doc): a model cheap per token but frequently wrong has a high
        *effective* cost. None (render as "—") when unpriced or zero
        successes, to avoid a divide-by-zero and a misleading number.
        """
        if self.total_agent_cost_usd is None or not self.succeeded:
            return None
        return self.total_agent_cost_usd / self.succeeded

    # -- pass^k reliability --------------------------------------------------

    def _trials_by_task(self) -> dict[str, list[TaskResult]]:
        """Group runs by task_id so each group is the set of trials for a task."""
        groups: dict[str, list[TaskResult]] = defaultdict(list)
        for t in self.tasks:
            groups[t.task_id].append(t)
        return groups

    @property
    def num_trials(self) -> int:
        """Max trials run for any single task (the k-axis upper bound)."""
        groups = self._trials_by_task()
        return max((len(v) for v in groups.values()), default=0)

    def pass_hat_k(self, k: int) -> float | None:
        """pass^k: probability of succeeding on ALL k of k independent trials.

            pass^k = (1/|tasks|) * Σ_task  C(c, k) / C(n, k)

        where for each task n = number of non-errored trials and c = number of
        successful trials. Errored trials are dropped from that task's n (a task
        whose every trial errored is skipped entirely). Returns None if no task
        has at least k non-errored trials. pass^1 equals the per-task-macro
        success rate; it falls as k grows, exposing non-determinism.
        """
        groups = self._trials_by_task()
        contributions: list[float] = []
        for trials in groups.values():
            scored = [t for t in trials if t.task_success is not None]
            n = len(scored)
            if n < k:
                continue
            c = sum(1 for t in scored if t.task_success is True)
            contributions.append(comb(c, k) / comb(n, k))
        if not contributions:
            return None
        return sum(contributions) / len(contributions)

    # -- difficulty strata (P2.11) -------------------------------------------

    @staticmethod
    def _rate(tasks: list[TaskResult]) -> tuple[int, int]:
        """(succeeded, scored) over a task subset."""
        succ = sum(1 for t in tasks if t.task_success is True)
        scored = sum(1 for t in tasks if t.task_success is not None)
        return succ, scored

    def success_by_difficulty(self) -> dict[str, tuple[int, int]]:
        """{difficulty: (succeeded, scored)} for tasks tagged with difficulty."""
        buckets: dict[str, list[TaskResult]] = defaultdict(list)
        for t in self.tasks:
            if t.difficulty:
                buckets[t.difficulty].append(t)
        return {d: self._rate(ts) for d, ts in buckets.items()}


def display_name(model: str) -> str:
    return MODEL_DISPLAY.get(model, model)


# ---------------------------------------------------------------------------
# Discovery — no hardcoded models/agents
# ---------------------------------------------------------------------------


def discover_models(prefix: str, timestamp: str | None = None) -> list[str]:
    """Auto-discover model names from bench_output/{prefix}_{model}_{timestamp}/ dirs."""
    models = set()
    pattern = re.compile(rf"^{re.escape(prefix)}_(.+?)_(\d{{8}}_\d{{6}})$")
    for entry in BENCH_OUTPUT.iterdir():
        if not entry.is_dir():
            continue
        m = pattern.match(entry.name)
        if m:
            model_name = m.group(1)
            ts = m.group(2)
            if timestamp is None or ts == timestamp:
                models.add(model_name)
    return sorted(models)


def discover_timestamps(prefix: str) -> list[str]:
    """Auto-discover all timestamps for a given prefix."""
    timestamps = set()
    pattern = re.compile(rf"^{re.escape(prefix)}_(.+?)_(\d{{8}}_\d{{6}})$")
    for entry in BENCH_OUTPUT.iterdir():
        if not entry.is_dir():
            continue
        m = pattern.match(entry.name)
        if m:
            timestamps.add(m.group(2))
    return sorted(timestamps)


def find_latest_timestamp(prefix: str) -> str | None:
    timestamps = discover_timestamps(prefix)
    return timestamps[-1] if timestamps else None


def load_run(prefix: str, model: str, timestamp: str) -> ModelRun | None:
    dir_name = f"{prefix}_{model}_{timestamp}"
    traces_path = BENCH_OUTPUT / dir_name / "traces.json"
    if not traces_path.exists():
        return None

    with open(traces_path) as f:
        raw = json.load(f)

    tasks = []
    for t in raw:
        tasks.append(
            TaskResult(
                task_id=t["task_id"],
                trial=t.get("trial", 0),
                num_turns=t.get("num_turns", 0),
                stopped=t.get("stopped", False),
                elapsed_seconds=t.get("elapsed_seconds", 0),
                tools_called=t.get("tools_called", []),
                expected_actions=t.get("expected_actions", []),
                actions_matched=t.get("actions_matched", []),
                topics_visited=t.get("topics_visited", []),
                handoffs=t.get("handoffs", []),
                state_passed=t.get("state_passed"),
                state_failures=t.get("state_failures", []),
                error=t.get("error"),
                error_kind=t.get("error_kind"),
                timed_out=t.get("timed_out", False),
                turns=t.get("turns", []),
                args_passed=t.get("args_passed"),
                args_failures=t.get("args_failures", []),
                response_passed=t.get("response_passed"),
                response_failures=t.get("response_failures", []),
                order_passed=t.get("order_passed"),
                order_failures=t.get("order_failures", []),
                collateral_passed=t.get("collateral_passed"),
                collateral_failures=t.get("collateral_failures", []),
                forbidden_passed=t.get("forbidden_passed"),
                forbidden_failures=t.get("forbidden_failures", []),
                difficulty=t.get("difficulty"),
                agent_usage=t.get("agent_usage", {}),
                agent_cost_usd=t.get("agent_cost_usd"),
                user_sim_cost_usd=t.get("user_sim_cost_usd"),
                agent_llm_seconds=t.get("agent_llm_seconds", 0.0),
            )
        )

    return ModelRun(model=model, timestamp=timestamp, tasks=tasks)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def h1(self, text: str) -> None:
        self.lines.append(f"# {text}\n")

    def h2(self, text: str) -> None:
        self.lines.append(f"## {text}\n")

    def h3(self, text: str) -> None:
        self.lines.append(f"### {text}\n")

    def p(self, text: str) -> None:
        self.lines.append(f"{text}\n")

    def table(
        self, headers: list[str], rows: list[list[str]], align: list[str] | None = None
    ) -> None:
        if align is None:
            align = ["l"] * len(headers)
        sep = []
        for a in align:
            if a == "r":
                sep.append("---:")
            elif a == "c":
                sep.append(":---:")
            else:
                sep.append("---")
        self.lines.append("| " + " | ".join(headers) + " |")
        self.lines.append("| " + " | ".join(sep) + " |")
        for row in rows:
            self.lines.append("| " + " | ".join(str(c) for c in row) + " |")
        self.lines.append("")

    def render(self) -> str:
        return "\n".join(self.lines)


def _fmt_cost(v: float | None) -> str:
    return f"${v:.4f}" if v is not None else "—"


def generate_scorecard(report: Report, runs: list[ModelRun]) -> None:
    report.h2("Scorecard")
    report.p(
        "**Success Rate** (TSR) is the model-quality headline: strict "
        "per-task success averaged over *non-errored* tasks. **E2E** is the "
        "production-gate rate over *all* tasks, counting errors as failures "
        "— use it for go/no-go. *Tools/Task* and *Time/Task* are efficiency "
        "metrics (lower is leaner). **Cost/Success** is agent-under-test USD "
        "cost divided by successful tasks — the hero cost metric (see Cost "
        'Efficiency below); "—" means the run is unpriced or had zero '
        "successes. Other columns are diagnostic."
    )
    report.p("")
    headers = [
        "Model",
        "Success Rate",
        "E2E",
        "Completed",
        "Errors",
        "Max Turns",
        "Actions",
        "State",
        "Avg Turns",
        "Tools/Task",
        "Time/Task",
        "Time",
        "Cost/Success",
        "Total Cost",
    ]
    align = ["l", "c", "c", "c", "c", "c", "c", "c", "r", "r", "r", "r", "r", "r"]
    rows = []
    for r in sorted(runs, key=lambda r: r.success_rate, reverse=True):
        success_str = (
            f"{r.success_rate * 100:.0f}% ({r.succeeded}/{r.scored})"
            if r.scored
            else "n/a"
        )
        e2e_str = (
            f"{r.end_to_end_success_rate * 100:.0f}% ({r.succeeded}/{r.total})"
            if r.total
            else "n/a"
        )
        rows.append(
            [
                display_name(r.model),
                f"**{success_str}**",
                e2e_str,
                f"{r.completed}/{r.total}",
                f"{r.errored}" if r.errored else "0",
                f"{r.hit_max}" if r.hit_max else "0",
                f"{r.actions_matched_total}/{r.actions_expected_total}",
                f"{r.state_passed_total}/{r.state_total}" if r.state_total else "-",
                f"{r.avg_turns:.1f}",
                f"{r.avg_tool_calls:.1f}",
                f"{r.avg_time_per_task:.0f}s",
                f"{r.total_time:.0f}s",
                f"**{_fmt_cost(r.cost_per_successful_task_usd)}**",
                _fmt_cost(r.total_agent_cost_usd),
            ]
        )
    report.table(headers, rows, align)


def generate_cost_efficiency(report: Report, runs: list[ModelRun]) -> None:
    """Cost-per-attempted-task vs. cost-per-successful-task (design doc P0).

    The gap between the two is itself a signal: a model cheap per attempted
    task but with a low success rate has a much higher *effective* cost per
    successful task — the number that should drive model choice. Skipped
    entirely when every run is unpriced (no pricing.yaml entry / older
    trace), so the section doesn't add noise to old reports.
    """
    priced_runs = [r for r in runs if r.total_agent_cost_usd is not None]
    if not priced_runs:
        return

    report.h2("Cost Efficiency")
    report.p(
        "Cost is **agent-under-test only** (ReAct loop / topic "
        "classifier); user-simulator tokens are tracked separately and "
        "excluded here. *Cost/Attempted* divides total agent cost by "
        "the same denominator as Success Rate (non-errored tasks); "
        "*Cost/Success* divides by successes only — a model that fails "
        "often looks cheap on the first number and expensive on the "
        "second. That gap is the number that should drive model choice, "
        "not raw $/token."
    )
    report.p("")
    headers = ["Model", "Cost/Attempted", "Cost/Success", "Gap (x)"]
    align = ["l", "r", "r", "r"]
    rows = []
    for r in sorted(
        runs,
        key=lambda r: (
            r.cost_per_successful_task_usd is None,
            r.cost_per_successful_task_usd or 0.0,
        ),
    ):
        attempted = r.cost_per_attempted_task_usd
        success = r.cost_per_successful_task_usd
        if attempted and success:
            gap = f"{success / attempted:.2f}x"
        else:
            gap = "—"
        rows.append(
            [
                display_name(r.model),
                _fmt_cost(attempted),
                _fmt_cost(success),
                gap,
            ]
        )
    report.table(headers, rows, align)


def generate_reliability(report: Report, runs: list[ModelRun]) -> None:
    """pass^k reliability table — only meaningful when trials > 1."""
    max_k = max((r.num_trials for r in runs), default=0)
    if max_k < 2:
        return  # single-trial run: pass^1 == success rate, already shown

    report.h2("Reliability (pass^k)")
    report.p(
        "**pass^k** = probability the model succeeds on *all k* of k "
        "independent attempts, averaged across tasks. pass^1 "
        f"is the success rate; the drop toward pass^{max_k} exposes non-determinism "
        "— a reliable model decays slowly."
    )
    report.p("")

    ks = list(range(1, max_k + 1))
    headers = ["Model"] + [f"pass^{k}" for k in ks]
    align = ["l"] + ["c"] * len(ks)
    rows = []

    # Sort by the hardest available pass^k (most reliable first), then pass^1.
    def sort_key(r: ModelRun) -> tuple[float, float]:
        return (r.pass_hat_k(max_k) or -1.0, r.pass_hat_k(1) or -1.0)

    for r in sorted(runs, key=sort_key, reverse=True):
        cells = []
        for k in ks:
            val = r.pass_hat_k(k)
            cells.append(f"{val * 100:.0f}%" if val is not None else "-")
        rows.append([display_name(r.model)] + cells)
    report.table(headers, rows, align)


def _fmt_rate(pair: tuple[int, int]) -> str:
    succ, scored = pair
    if not scored:
        return "-"
    return f"{succ / scored * 100:.0f}% ({succ}/{scored})"


def generate_difficulty_breakdown(report: Report, runs: list[ModelRun]) -> None:
    """TSR split by difficulty stratum (P2.11).

    Only emitted when at least one task carries a difficulty tag — otherwise it
    adds noise.
    """
    has_difficulty = any(t.difficulty for r in runs for t in r.tasks)
    if not has_difficulty:
        return

    # Standard ordering for difficulty strata; unknown tags appended after.
    order = ["easy", "medium", "hard"]
    report.h2("Success by Difficulty")
    report.p(
        "TSR within each difficulty stratum. A frontier model may ace "
        "*easy* yet visibly fail *hard* — this is the headroom signal."
    )
    report.p("")
    strata = sorted(
        {t.difficulty for r in runs for t in r.tasks if t.difficulty},
        key=lambda d: (order.index(d) if d in order else len(order), d),
    )
    headers = ["Model"] + [s.capitalize() for s in strata]
    align = ["l"] + ["c"] * len(strata)
    rows = []
    for r in sorted(runs, key=lambda r: r.success_rate, reverse=True):
        by = r.success_by_difficulty()
        rows.append(
            [display_name(r.model)] + [_fmt_rate(by.get(s, (0, 0))) for s in strata]
        )
    report.table(headers, rows, align)


def aggregate_task_cell(trials: list[TaskResult]) -> str:
    """Render one (model, task) matrix cell aggregating ALL its trials (P1.12).

    Shows passed/scored trial counts (e.g. "2/3") rather than the old behavior
    of displaying whichever trial happened to match first — which hid
    non-determinism across trials. Errored trials are excluded from the scored
    denominator but surfaced as a suffix.
    """
    scored = [t for t in trials if t.task_success is not None]
    errors = len(trials) - len(scored)
    passed = sum(1 for t in scored if t.task_success is True)

    if not scored:
        return f"**{errors} err**" if errors else "-"

    cell = f"{passed}/{len(scored)}"
    if passed < len(scored):
        cell = f"**{cell}**"
    if errors:
        cell += f" (+{errors} err)"
    return cell


def failing_trials_by_task(
    runs: list[ModelRun],
) -> dict[str, list[tuple[str, TaskResult]]]:
    """Group every failing TRIAL by task_id (P1.12).

    Each non-passing trial is one entry, tagged with its model — so a task that
    fails 2 of 3 trials on one model contributes 2 entries, not 1. The old code
    appended per trial but the header divided by len(runs) (models), conflating
    trial failures with model failures.
    """
    groups: dict[str, list[tuple[str, TaskResult]]] = defaultdict(list)
    for r in runs:
        for t in r.tasks:
            if t.status_icon != "PASS":
                groups[t.task_id].append((r.model, t))
    return groups


def generate_task_matrix(report: Report, runs: list[ModelRun]) -> None:
    report.h2("Per-Task Results")

    task_ids = []
    seen = set()
    for r in runs:
        for t in r.tasks:
            if t.task_id not in seen:
                task_ids.append(t.task_id)
                seen.add(t.task_id)

    headers = ["Task"] + [display_name(r.model) for r in runs]
    align = ["l"] + ["c"] * len(runs)
    rows = []
    for tid in task_ids:
        row = [f"`{tid}`"]
        for r in runs:
            trials = [t for t in r.tasks if t.task_id == tid]
            row.append(aggregate_task_cell(trials) if trials else "-")
        rows.append(row)
    report.table(headers, rows, align)


def generate_failures(report: Report, runs: list[ModelRun]) -> None:
    report.h2("Failure Analysis")

    task_failures = failing_trials_by_task(runs)

    if not task_failures:
        report.p("All tasks passed on all models.")
        return

    # Total trials per task across all models — the honest denominator for the
    # "N/M failed" header (P1.12: the old code divided by len(runs)=models even
    # when each model contributed multiple trials).
    trials_per_task: dict[str, int] = defaultdict(int)
    for r in runs:
        for t in r.tasks:
            trials_per_task[t.task_id] += 1

    for tid, failures in sorted(task_failures.items(), key=lambda x: -len(x[1])):
        fail_count = len(failures)
        total_trials = trials_per_task[tid]
        report.h3(f"`{tid}` — {fail_count}/{total_trials} trials failed")

        rows = []
        for model, task in failures:
            reason = task.status_icon
            detail = ""
            if task.error:
                detail = task.error[:80]
            elif not task.stopped:
                detail = f"{task.num_turns} turns, {len(task.tools_called)} tool calls"
            elif not task.action_pass:
                called = (
                    ", ".join(task.tools_called[:5]) if task.tools_called else "none"
                )
                detail = f"called: {called}"
            if task.state_failures:
                detail += (" | " if detail else "") + "; ".join(task.state_failures)
            rows.append([display_name(model), reason, detail])

        report.table(["Model", "Status", "Detail"], rows)


def generate_user_sim_issues(
    report: Report, runs: list[ModelRun], show_conversations: bool
) -> None:
    report.h2("User Simulator Issues")

    issues_found = False
    for r in runs:
        for t in r.tasks:
            if not t.turns or t.status_icon == "PASS":
                continue

            refusal_patterns = [
                "prefer not to",
                "don't want to",
                "don't think I need",
                "without needing my email",
                "without requiring my email",
                "I don't see why",
                "can you just",
            ]
            user_refused_email = False
            refusal_turn = None
            for turn in t.turns:
                if turn["role"] == "user":
                    content_lower = turn["content"].lower()
                    if any(p in content_lower for p in refusal_patterns):
                        user_refused_email = True
                        refusal_turn = turn["content"][:200]
                        break

            if user_refused_email:
                issues_found = True
                report.p(
                    f"**`{t.task_id}`** ({display_name(r.model)}): "
                    f"User sim refused to provide email. {t.num_turns} turns wasted."
                )
                if show_conversations and refusal_turn:
                    report.p(f"> User: {refusal_turn}")
                    report.p("")

            if not t.stopped and t.action_pass and not t.error:
                issues_found = True
                report.p(
                    f"**`{t.task_id}`** ({display_name(r.model)}): "
                    f"Actions matched but user sim didn't end conversation. "
                    f"{t.num_turns} turns, {len(t.tools_called)} tool calls."
                )

    if not issues_found:
        report.p("No user simulator issues detected.")


def generate_conversation_excerpts(report: Report, runs: list[ModelRun]) -> None:
    report.h2("Failed Task Conversations")

    for r in runs:
        for t in r.tasks:
            if t.status_icon == "PASS" or not t.turns:
                continue

            report.h3(f"`{t.task_id}` — {display_name(r.model)} ({t.status_icon})")

            max_show = 6
            turns = t.turns
            if len(turns) <= max_show:
                show_turns = list(enumerate(turns))
            else:
                show_turns = list(enumerate(turns[:3]))
                show_turns.append(
                    (
                        -1,
                        {
                            "role": "system",
                            "content": f"... ({len(turns) - 6} turns omitted) ...",
                        },
                    )
                )
                show_turns.extend(enumerate(turns[-3:], len(turns) - 3))

            for idx, turn in show_turns:
                role = turn["role"].upper()
                content = turn["content"][:300]
                if turn.get("tool_calls"):
                    tools = ", ".join(tc["name"] for tc in turn["tool_calls"])
                    report.p(f"**[{role}]** {content}")
                    report.p(f"  *Tools: {tools}*")
                else:
                    report.p(f"**[{role}]** {content}")
            report.p("")


def generate_comparison(
    report: Report,
    runs_old: list[ModelRun],
    runs_new: list[ModelRun],
    models: list[str],
) -> None:
    report.h2("Run Comparison (Old vs New)")

    headers = ["Model", "Completed", "Actions", "State", "Avg Turns", "Time", "Delta"]
    align = ["l", "c", "c", "c", "r", "r", "l"]
    rows = []

    old_by_model = {r.model: r for r in runs_old}
    new_by_model = {r.model: r for r in runs_new}

    for model in models:
        old = old_by_model.get(model)
        new = new_by_model.get(model)
        if not new:
            continue

        new_completed = f"{new.completed}/{new.total}"
        new_actions = f"{new.actions_matched_total}/{new.actions_expected_total}"
        new_state = (
            f"{new.state_passed_total}/{new.state_total}" if new.state_total else "-"
        )

        deltas = []
        if old:
            dc = new.completed - old.completed
            if dc != 0:
                deltas.append(f"Completed {'+' if dc > 0 else ''}{dc}")
            da = new.actions_matched_total - old.actions_matched_total
            if da != 0:
                deltas.append(f"Actions {'+' if da > 0 else ''}{da}")
            ds = new.state_passed_total - old.state_passed_total
            if ds != 0:
                deltas.append(f"State {'+' if ds > 0 else ''}{ds}")

        rows.append(
            [
                display_name(model),
                new_completed,
                new_actions,
                new_state,
                f"{new.avg_turns:.1f}",
                f"{new.total_time:.0f}s",
                ", ".join(deltas) if deltas else "same",
            ]
        )

    report.table(headers, rows, align)

    report.h3("Per-Task Changes")
    for model in models:
        old = old_by_model.get(model)
        new = new_by_model.get(model)
        if not old or not new:
            continue

        old_tasks = {t.task_id: t for t in old.tasks}
        new_tasks = {t.task_id: t for t in new.tasks}
        changes = []

        for tid, nt in new_tasks.items():
            ot = old_tasks.get(tid)
            if not ot:
                continue
            old_ok = ot.status_icon == "PASS"
            new_ok = nt.status_icon == "PASS"
            if old_ok and not new_ok:
                changes.append(
                    f"  REGRESSED `{tid}`: was PASS -> {nt.status_icon} ({nt.num_turns}t)"
                )
            elif not old_ok and new_ok:
                changes.append(
                    f"  IMPROVED  `{tid}`: was {ot.status_icon} -> PASS ({nt.num_turns}t)"
                )

        if changes:
            report.p(f"**{display_name(model)}:**")
            for c in changes:
                report.p(c)
            report.p("")


def generate_tool_usage(report: Report, runs: list[ModelRun]) -> None:
    report.h2("Tool Usage Patterns")

    excessive: list[tuple[str, str, int, str]] = []
    wrong_tool: list[tuple[str, str, str, str]] = []

    for r in runs:
        for t in r.tasks:
            if not t.tools_called:
                continue
            if len(t.tools_called) > 10:
                most_called = max(set(t.tools_called), key=t.tools_called.count)
                count = t.tools_called.count(most_called)
                excessive.append(
                    (
                        display_name(r.model),
                        t.task_id,
                        len(t.tools_called),
                        f"{most_called} x{count}",
                    )
                )
            if t.expected_actions and not t.action_pass:
                called_set = set(t.tools_called)
                expected_set = set(t.expected_actions)
                wrong = called_set - expected_set
                if wrong and t.tools_called:
                    wrong_tool.append(
                        (
                            display_name(r.model),
                            t.task_id,
                            ", ".join(sorted(expected_set)),
                            ", ".join(sorted(wrong)),
                        )
                    )

    if excessive:
        report.h3("Excessive Tool Calls (>10)")
        report.table(
            ["Model", "Task", "Total Calls", "Most Called"],
            [[e[0], f"`{e[1]}`", str(e[2]), e[3]] for e in excessive],
        )

    if wrong_tool:
        report.h3("Wrong Tool Used (expected action missed)")
        report.table(
            ["Model", "Task", "Expected", "Called Instead"],
            [[w[0], f"`{w[1]}`", w[2], w[3]] for w in wrong_tool],
        )


# ---------------------------------------------------------------------------
# CSV generation
# ---------------------------------------------------------------------------


def generate_summary_csv(runs: list[ModelRun], prefix: str) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    # pass^k columns are emitted only when at least one run had >1 trial.
    max_k = max((r.num_trials for r in runs), default=1)
    pass_cols = [f"pass_hat_{k}" for k in range(1, max_k + 1)] if max_k > 1 else []
    writer.writerow(
        [
            "agent",
            "model",
            "success_rate",
            "end_to_end_success_rate",
            "succeeded",
            "scored",
            "total_tasks",
            "completed",
            "errored",
            "max_turns_hit",
            "actions_matched",
            "actions_expected",
            "action_rate",
            "completion_rate",
            "state_passed",
            "state_total",
            "state_rate",
            "trials",
            *pass_cols,
            "avg_turns",
            "avg_tool_calls",
            "avg_time_per_task_seconds",
            "total_time_seconds",
            "agent_output_tokens_per_second",
            "agent_llm_seconds_per_task",
            "total_agent_cost_usd",
            "cost_per_attempted_task_usd",
            "cost_per_successful_task_usd",
            "total_agent_tokens",
        ]
    )
    for r in runs:
        action_rate = (
            f"{r.actions_matched_total / r.actions_expected_total * 100:.0f}%"
            if r.actions_expected_total
            else "n/a"
        )
        completion_rate = f"{r.completed / r.total * 100:.0f}%" if r.total else "n/a"
        state_rate = (
            f"{r.state_passed_total / r.state_total * 100:.0f}%"
            if r.state_total
            else "n/a"
        )
        success_rate = f"{r.success_rate * 100:.0f}%" if r.scored else "n/a"
        e2e_rate = f"{r.end_to_end_success_rate * 100:.0f}%" if r.total else "n/a"
        pass_vals = []
        if max_k > 1:
            for k in range(1, max_k + 1):
                val = r.pass_hat_k(k)
                pass_vals.append(f"{val * 100:.0f}%" if val is not None else "n/a")
        writer.writerow(
            [
                agent_display_name(prefix),
                display_name(r.model),
                success_rate,
                e2e_rate,
                r.succeeded,
                r.scored,
                r.total,
                r.completed,
                r.errored,
                r.hit_max,
                r.actions_matched_total,
                r.actions_expected_total,
                action_rate,
                completion_rate,
                r.state_passed_total,
                r.state_total,
                state_rate,
                r.num_trials,
                *pass_vals,
                f"{r.avg_turns:.1f}",
                f"{r.avg_tool_calls:.1f}",
                f"{r.avg_time_per_task:.1f}",
                f"{r.total_time:.1f}",
                ""
                if r.output_tokens_per_second is None
                else f"{r.output_tokens_per_second:.1f}",
                ""
                if r.agent_llm_seconds_per_task is None
                else f"{r.agent_llm_seconds_per_task:.1f}",
                ""
                if r.total_agent_cost_usd is None
                else f"{r.total_agent_cost_usd:.4f}",
                ""
                if r.cost_per_attempted_task_usd is None
                else f"{r.cost_per_attempted_task_usd:.4f}",
                ""
                if r.cost_per_successful_task_usd is None
                else f"{r.cost_per_successful_task_usd:.4f}",
                r.total_agent_tokens,
            ]
        )
    return buf.getvalue()


def generate_detail_csv(runs: list[ModelRun], prefix: str) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "agent",
            "model",
            "task_id",
            "trial",
            "difficulty",
            "success",
            "status",
            "turns",
            "tool_calls",
            "stopped",
            "error",
            "actions_matched",
            "actions_expected",
            "state_passed",
            "args_passed",
            "response_passed",
            "order_passed",
            "collateral_passed",
            "forbidden_passed",
            "failures",
            "elapsed_seconds",
            "agent_prompt_tokens",
            "agent_completion_tokens",
            "agent_cache_read_tokens",
            "agent_total_tokens",
            "agent_cost_usd",
        ]
    )
    agent_name = agent_display_name(prefix)

    def _b(v: bool | None) -> str:
        return "" if v is None else str(v)

    for r in runs:
        for t in r.tasks:
            failures = "; ".join(
                t.state_failures
                + t.args_failures
                + t.response_failures
                + t.order_failures
                + t.collateral_failures
                + t.forbidden_failures
            )
            writer.writerow(
                [
                    agent_name,
                    display_name(r.model),
                    t.task_id,
                    t.trial,
                    t.difficulty or "",
                    "" if t.task_success is None else t.task_success,
                    t.status_icon,
                    t.num_turns,
                    len(t.tools_called),
                    t.stopped,
                    t.error or "",
                    len(t.actions_matched),
                    len(t.expected_actions),
                    _b(t.state_passed),
                    _b(t.args_passed),
                    _b(t.response_passed),
                    _b(t.order_passed),
                    _b(t.collateral_passed),
                    _b(t.forbidden_passed),
                    failures,
                    f"{t.elapsed_seconds:.1f}",
                    t.agent_prompt_tokens,
                    t.agent_completion_tokens,
                    t.agent_cache_read_tokens,
                    t.agent_total_tokens,
                    "" if t.agent_cost_usd is None else f"{t.agent_cost_usd:.4f}",
                ]
            )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze CRMAgentBench benchmark runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "prefix",
        help="Run prefix — matches bench_output/{prefix}_{model}_{timestamp}/ "
        "(e.g. csm_full_test, customer_service_full_test)",
    )
    parser.add_argument(
        "--timestamp", "-t", help="Specific timestamp to analyze (default: latest)"
    )
    parser.add_argument(
        "--compare",
        "-c",
        nargs=2,
        metavar=("OLD", "NEW"),
        help="Compare two timestamps",
    )
    parser.add_argument(
        "--markdown",
        "--md",
        action="store_true",
        help="Output markdown instead of CSV (default: CSV)",
    )
    parser.add_argument(
        "--show-conversations",
        "-v",
        action="store_true",
        help="Include conversation excerpts for failures (markdown only)",
    )
    parser.add_argument("--output", "-o", help="Write report to file instead of stdout")
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        help="Filter to specific models (default: auto-discover all)",
    )
    parser.add_argument(
        "--bench-output",
        help="Root directory holding benchmark runs "
        "(default: $BENCH_OUTPUT or ./bench_output)",
    )
    args = parser.parse_args()

    # Let an explicit --bench-output override the env/CWD default for discovery.
    if args.bench_output:
        global BENCH_OUTPUT
        BENCH_OUTPUT = Path(args.bench_output)

    # --compare only produces comparison content in markdown mode. In CSV mode
    # it would silently emit an ordinary new-run summary/detail — output that
    # looks like a comparison but isn't. Reject it rather than mislead (P1.12).
    if args.compare and not args.markdown:
        parser.error(
            "--compare requires --markdown (CSV mode has no comparison output)"
        )

    available = discover_timestamps(args.prefix)
    if not available:
        print(
            f"No runs found for prefix '{args.prefix}' in {BENCH_OUTPUT}",
            file=sys.stderr,
        )
        if BENCH_OUTPUT.exists():
            prefixes = set()
            pat = re.compile(r"^(.+?)_(\d{8}_\d{6})$")
            for d in BENCH_OUTPUT.iterdir():
                m = pat.match(d.name)
                if m:
                    candidate = m.group(1)
                    # Strip model suffix to get the prefix
                    for known in discover_models(candidate, m.group(2)):
                        if candidate.endswith(f"_{known}"):
                            prefixes.add(candidate[: -len(known) - 1])
                            break
            if prefixes:
                print(f"Available prefixes: {sorted(prefixes)}", file=sys.stderr)
        sys.exit(1)

    report = Report()

    if args.compare:
        ts_old, ts_new = args.compare
        models = (
            args.models
            or discover_models(args.prefix, ts_new)
            or discover_models(args.prefix, ts_old)
        )

        runs_old = []
        runs_new = []
        for model in models:
            r = load_run(args.prefix, model, ts_old)
            if r:
                runs_old.append(r)
            r = load_run(args.prefix, model, ts_new)
            if r:
                runs_new.append(r)

        if not runs_new:
            print(f"No runs found for timestamp {ts_new}", file=sys.stderr)
            sys.exit(1)

        runs = runs_new

        if args.markdown:
            report.h1(f"Benchmark Comparison: {args.prefix}")
            report.p(f"Old: `{ts_old}` | New: `{ts_new}`")
            report.p("")
            report.p(f"Models loaded — Old: {len(runs_old)}, New: {len(runs_new)}")
            report.p("")

            generate_scorecard(report, runs_new)
            generate_cost_efficiency(report, runs_new)
            generate_reliability(report, runs_new)
            generate_difficulty_breakdown(report, runs_new)
            generate_comparison(report, runs_old, runs_new, models)
            generate_task_matrix(report, runs_new)
            generate_failures(report, runs_new)
            generate_user_sim_issues(report, runs_new, args.show_conversations)
            generate_tool_usage(report, runs_new)
            if args.show_conversations:
                generate_conversation_excerpts(report, runs_new)
    else:
        ts = args.timestamp or find_latest_timestamp(args.prefix)
        if not ts:
            print(f"No timestamps found for {args.prefix}", file=sys.stderr)
            sys.exit(1)

        models = args.models or discover_models(args.prefix, ts)

        runs = []
        for model in models:
            r = load_run(args.prefix, model, ts)
            if r:
                runs.append(r)

        if not runs:
            print(f"No runs found for timestamp {ts}", file=sys.stderr)
            sys.exit(1)

        if args.markdown:
            report.h1(f"Benchmark Analysis: {args.prefix}")
            report.p(f"Timestamp: `{ts}`")
            report.p(f"Available runs: {', '.join(available)}")
            report.p("")
            report.p(
                f"Models loaded: {len(runs)} ({', '.join(display_name(r.model) for r in runs)})"
            )
            report.p("")

            generate_scorecard(report, runs)
            generate_cost_efficiency(report, runs)
            generate_reliability(report, runs)
            generate_difficulty_breakdown(report, runs)
            generate_task_matrix(report, runs)
            generate_failures(report, runs)
            generate_user_sim_issues(report, runs, args.show_conversations)
            generate_tool_usage(report, runs)
            if args.show_conversations:
                generate_conversation_excerpts(report, runs)

    if args.markdown:
        output = report.render()
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(output)
            print(f"Markdown report written to {out_path}", file=sys.stderr)
        else:
            print(output)
    else:
        summary_csv = generate_summary_csv(runs, args.prefix)
        detail_csv = generate_detail_csv(runs, args.prefix)

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path = out_path.with_name(out_path.stem + "_summary.csv")
            detail_path = out_path.with_name(out_path.stem + "_detail.csv")
            summary_path.write_text(summary_csv)
            detail_path.write_text(detail_csv)
            print(f"Summary CSV written to {summary_path}", file=sys.stderr)
            print(f"Detail CSV written to {detail_path}", file=sys.stderr)
        else:
            print("=== Summary ===")
            print(summary_csv)
            print("=== Per-Task Detail ===")
            print(detail_csv)


if __name__ == "__main__":
    main()
