#!/usr/bin/env python3
"""Generate comparison report for user simulator benchmark runs.

Reads traces from bench_output/{prefix}_*_{timestamp}/ directories
and produces a markdown report comparing models × strategies × repeats.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_run_dir(dirname: str, prefix: str, timestamp: str) -> dict | None:
    """Extract model, strategy, run_num from directory name."""
    # Pattern: {prefix}_{model}_{strategy}_r{N}_{timestamp}
    pattern = rf"^{re.escape(prefix)}_(.+?)_(llm|react|verify|reflection)_r(\d+)_{re.escape(timestamp)}$"
    m = re.match(pattern, dirname)
    if not m:
        return None
    return {
        "model": m.group(1).replace("_", "/"),
        "strategy": m.group(2),
        "run_num": int(m.group(3)),
    }


def load_traces(traces_path: Path) -> list[dict]:
    with open(traces_path) as f:
        return json.load(f)


def analyze_traces(traces: list[dict]) -> dict:
    """Compute metrics from a single run's traces."""
    total = len(traces)
    errors = sum(1 for t in traces if t.get("error"))
    stopped = sum(1 for t in traces if t.get("stopped"))
    max_turns_hit = sum(
        1 for t in traces if not t.get("stopped") and not t.get("error")
    )

    actions_matched = sum(len(t.get("actions_matched", [])) for t in traces)
    actions_expected = sum(len(t.get("expected_actions", [])) for t in traces)
    tools_called = sum(len(t.get("tools_called", [])) for t in traces)
    total_turns = sum(t.get("num_turns", 0) for t in traces)

    state_checks = [t for t in traces if t.get("state_passed") is not None]
    state_passed = sum(1 for t in state_checks if t["state_passed"])

    # Per-task results
    task_results = {}
    for t in traces:
        tid = t["task_id"]
        matched = set(t.get("actions_matched", []))
        expected = set(t.get("expected_actions", []))
        task_results[tid] = {
            "actions_pass": matched == expected and len(expected) > 0,
            "state_pass": t.get("state_passed"),
            "stopped": t.get("stopped", False),
            "error": t.get("error"),
            "turns": t.get("num_turns", 0),
        }

    # Check for user sim issues (refusals, stuck loops)
    usersim_issues = 0
    refusal_patterns = [
        "prefer not to",
        "don't want to share",
        "rather not provide",
        "don't feel comfortable",
        "won't be sharing",
    ]
    for t in traces:
        for turn in t.get("turns", []):
            if turn.get("role") == "user":
                content = (turn.get("content") or "").lower()
                if any(p in content for p in refusal_patterns):
                    usersim_issues += 1
                    break

    return {
        "total": total,
        "errors": errors,
        "stopped": stopped,
        "max_turns_hit": max_turns_hit,
        "actions_matched": actions_matched,
        "actions_expected": actions_expected,
        "action_accuracy": actions_matched / actions_expected
        if actions_expected
        else 0,
        "tools_called": tools_called,
        "avg_turns": total_turns / total if total else 0,
        "state_passed": state_passed,
        "state_total": len(state_checks),
        "task_results": task_results,
        "usersim_issues": usersim_issues,
    }


# Config key: (model, strategy)
ConfigKey = tuple[str, str]


def build_report(
    all_runs: dict[ConfigKey, list[dict]],
    timestamp: str,
) -> str:
    """Build markdown comparison report."""
    lines = []
    lines.append(f"# User Simulator Comparison — {timestamp}")
    lines.append("")

    # Collect all models and strategies
    models = sorted({k[0] for k in all_runs})
    strategies = sorted(
        {k[1] for k in all_runs},
        key=lambda s: ["llm", "react", "verify", "reflection"].index(s),
    )
    task_ids = set()
    for runs in all_runs.values():
        for r in runs:
            task_ids.update(r["task_results"].keys())
    task_ids = sorted(task_ids)

    # --- Summary table ---
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Model | Strategy | Runs | Action Accuracy | Avg Turns | Stop Rate | User Sim Issues |"
    )
    lines.append(
        "|-------|----------|:----:|:---------------:|:---------:|:---------:|:---------------:|"
    )

    for model in models:
        for strategy in strategies:
            key = (model, strategy)
            runs = all_runs.get(key, [])
            if not runs:
                continue
            n = len(runs)
            acc_vals = [r["action_accuracy"] for r in runs]
            turn_vals = [r["avg_turns"] for r in runs]
            stop_vals = [r["stopped"] / r["total"] if r["total"] else 0 for r in runs]
            issue_vals = [r["usersim_issues"] for r in runs]

            acc_mean = sum(acc_vals) / n
            turn_mean = sum(turn_vals) / n
            stop_mean = sum(stop_vals) / n
            issue_total = sum(issue_vals)

            # Show variance if multiple runs
            if n > 1:
                acc_min, acc_max = min(acc_vals), max(acc_vals)
                acc_str = f"{acc_mean:.0%} ({acc_min:.0%}–{acc_max:.0%})"
                turn_min, turn_max = min(turn_vals), max(turn_vals)
                turn_str = f"{turn_mean:.1f} ({turn_min:.1f}–{turn_max:.1f})"
            else:
                acc_str = f"{acc_mean:.0%}"
                turn_str = f"{turn_mean:.1f}"

            lines.append(
                f"| `{model}` | `{strategy}` | {n} | {acc_str} | {turn_str} | {stop_mean:.0%} | {issue_total} |"
            )

    lines.append("")

    # --- Per-task breakdown ---
    lines.append("## Per-Task Results")
    lines.append("")
    lines.append("Pass rate across repeats (actions correct + stopped).")
    lines.append("")

    header = "| Task |"
    sep = "|------|"
    for model in models:
        for strategy in strategies:
            key = (model, strategy)
            if key not in all_runs:
                continue
            header += f" {model}/{strategy} |"
            sep += ":---:|"

    lines.append(header)
    lines.append(sep)

    for tid in task_ids:
        row = f"| `{tid}` |"
        for model in models:
            for strategy in strategies:
                key = (model, strategy)
                runs = all_runs.get(key, [])
                if not runs:
                    continue
                passes = 0
                total = 0
                for r in runs:
                    tr = r["task_results"].get(tid)
                    if tr:
                        total += 1
                        if tr["actions_pass"] and tr["stopped"] and not tr["error"]:
                            passes += 1
                if total == 0:
                    row += " - |"
                else:
                    row += f" {passes}/{total} |"
        lines.append(row)

    lines.append("")

    # --- Variance analysis ---
    lines.append("## Variance Analysis")
    lines.append("")
    lines.append(
        "Configs with inconsistent results across repeats (some pass, some fail):"
    )
    lines.append("")

    has_variance = False
    for model in models:
        for strategy in strategies:
            key = (model, strategy)
            runs = all_runs.get(key, [])
            if len(runs) <= 1:
                continue
            for tid in task_ids:
                results = []
                for r in runs:
                    tr = r["task_results"].get(tid)
                    if tr:
                        results.append(
                            tr["actions_pass"] and tr["stopped"] and not tr["error"]
                        )
                if results and not all(r == results[0] for r in results):
                    passes = sum(results)
                    total = len(results)
                    lines.append(
                        f"- **`{tid}`** with `{model}`/`{strategy}`: {passes}/{total} pass — non-deterministic"
                    )
                    has_variance = True

    if not has_variance:
        lines.append("No variance detected (all repeats consistent).")
    lines.append("")

    # --- Recommendations ---
    lines.append("## Recommendations")
    lines.append("")

    # Find best config by action accuracy
    best_key = max(
        all_runs,
        key=lambda k: sum(r["action_accuracy"] for r in all_runs[k]) / len(all_runs[k]),
    )
    best_acc = sum(r["action_accuracy"] for r in all_runs[best_key]) / len(
        all_runs[best_key]
    )

    # Find most efficient (highest accuracy with lowest turns)
    efficiency = {}
    for key, runs in all_runs.items():
        acc = sum(r["action_accuracy"] for r in runs) / len(runs)
        turns = sum(r["avg_turns"] for r in runs) / len(runs)
        efficiency[key] = acc / turns if turns > 0 else 0

    most_efficient = max(efficiency, key=lambda k: efficiency[k])

    # Find lowest variance
    variance_scores = {}
    for key, runs in all_runs.items():
        if len(runs) <= 1:
            variance_scores[key] = 0
            continue
        accs = [r["action_accuracy"] for r in runs]
        variance_scores[key] = max(accs) - min(accs)

    lowest_variance = min(variance_scores, key=lambda k: variance_scores[k])

    lines.append(
        f"- **Highest accuracy**: `{best_key[0]}`/`{best_key[1]}` — {best_acc:.0%}"
    )
    lines.append(
        f"- **Most efficient** (accuracy/turns): `{most_efficient[0]}`/`{most_efficient[1]}`"
    )
    lines.append(
        f"- **Most consistent** (lowest variance): `{lowest_variance[0]}`/`{lowest_variance[1]}`"
    )
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare user simulator benchmark runs"
    )
    parser.add_argument(
        "--prefix", required=True, help="Run prefix (e.g., usersim_compare)"
    )
    parser.add_argument("--timestamp", required=True, help="Timestamp to filter runs")
    parser.add_argument("-o", "--output", help="Output file path")
    args = parser.parse_args()

    bench_dir = Path("bench_output")
    if not bench_dir.exists():
        print("Error: bench_output/ directory not found", file=sys.stderr)
        sys.exit(1)

    # Find matching run directories
    all_runs: dict[ConfigKey, list[dict]] = defaultdict(list)
    found = 0

    for d in sorted(bench_dir.iterdir()):
        if not d.is_dir():
            continue
        info = parse_run_dir(d.name, args.prefix, args.timestamp)
        if not info:
            continue

        traces_path = d / "traces.json"
        if not traces_path.exists():
            print(f"Warning: no traces.json in {d.name}", file=sys.stderr)
            continue

        traces = load_traces(traces_path)
        metrics = analyze_traces(traces)
        key = (info["model"], info["strategy"])
        all_runs[key].append(metrics)
        found += 1

    if not found:
        print(
            f"Error: no runs found matching prefix={args.prefix} timestamp={args.timestamp}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found {found} runs across {len(all_runs)} configurations", file=sys.stderr)

    report = build_report(all_runs, args.timestamp)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report saved to {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
