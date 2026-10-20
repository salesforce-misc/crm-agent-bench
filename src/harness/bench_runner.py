"""Batch benchmark runner for automated agent evaluation.

Loads task definitions from YAML, runs agent-user conversation loops
using an LLM-powered user simulator, and logs full conversation traces.

Usage:
    python -m harness.bench_runner --tasks tasks/example.yaml --config src/agents/graph.yaml

Environment variables:
    Executor settings override the shared defaults in benchmark/benchmark.yaml.
    See .env.example for the agent, user simulator, judge, and run controls.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from harness.grading import (
    compute_action_pass,
    compute_status_icon,
    compute_task_success,
)
from harness.models import ChatMessage, Role
from harness.tools.registry import is_error_result
from harness.user_simulator import STOP_SIGNAL, UserSimulatorError, load_user_simulator

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """A single benchmark task."""

    id: str
    instruction: str
    expected_actions: list[str] = field(default_factory=list)
    domain: str | None = None
    initial_state: dict | None = None
    expected_state: dict | None = None
    # Stricter, optional assertions (see harness.grading):
    # {tool: {arg: assertion}} or {tool: [{...}, {...}]} for repeated calls.
    expected_args: dict | None = None
    # Plain-language requirements graded by the versioned LLM judge.
    expected_response_semantic: list[str] | None = None
    expected_order: list[str] | None = None  # required relative call order
    forbidden_state: dict | None = None  # paths that must NOT match (collateral damage)
    forbidden_actions: list[str] | None = (
        None  # tools that must NOT be called (refusal/policy)
    )
    forbidden_args: dict | None = None  # {tool: {arg: assertion}} that must NOT occur
    # Opt-in strictness for the checks above (P1.11). Defaults preserve the
    # retry-tolerant behavior; safety/destructive tasks can tighten them.
    expected_args_mode: str = "any"  # "any" | "first"
    expected_order_mode: str = "subsequence"  # "subsequence" | "exact"
    difficulty: str | None = None  # easy | medium | hard
    capabilities: list[str] = field(default_factory=list)
    task_set: str | None = None
    benchmark_version: str | None = None
    dataset_version: str | None = None


@dataclass
class ConversationTrace:
    """Full trace of a benchmark conversation."""

    task_id: str
    trial: int = 0
    seed: int | None = None
    benchmark_version: str | None = None
    dataset_version: str | None = None
    domain: str | None = None
    capabilities: list[str] = field(default_factory=list)
    task_set: str | None = None
    turns: list[dict[str, Any]] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    tool_calls_detail: list[dict[str, Any]] = field(default_factory=list)
    num_turns: int = 0
    stopped: bool = False
    timed_out: bool = False
    elapsed_seconds: float = 0.0
    expected_actions: list[str] = field(default_factory=list)
    actions_matched: list[str] = field(default_factory=list)
    topics_visited: list[str] = field(default_factory=list)
    handoffs: list[str] = field(default_factory=list)
    error: str | None = None
    # Stable error classification (None unless `error` is set). Drives grading:
    # "hallucinated_tool" is a model fault graded as a failure, whereas infra
    # errors (timeouts, disconnects) are excluded from the success denominator.
    error_kind: str | None = None
    state_passed: bool | None = None
    state_failures: list[str] = field(default_factory=list)
    # Stricter grading results (None = not asserted for this task):
    args_passed: bool | None = None
    args_failures: list[str] = field(default_factory=list)
    response_passed: bool | None = None
    response_failures: list[str] = field(default_factory=list)
    # Full semantic-judge audit record (model/prompt/parameters, cache key,
    # evidence, token usage, and locally computed verdict). None when the task
    # has no semantic response requirement or judging could not complete.
    response_judge: dict[str, Any] | None = None
    order_passed: bool | None = None
    order_failures: list[str] = field(default_factory=list)
    collateral_passed: bool | None = None
    collateral_failures: list[str] = field(default_factory=list)
    forbidden_passed: bool | None = None
    forbidden_failures: list[str] = field(default_factory=list)
    difficulty: str | None = None
    # Token usage + cost, split agent-under-test vs. user simulator so the
    # leaderboard can attribute cost to the model, not the eval harness. Each
    # usage dict carries the canonical keys from ChatResponse.usage
    # (prompt/completion/cache_read/cache_write/reasoning/total tokens + num_calls).
    # Cost is USD; None when the model has no pricing entry ("unpriced").
    agent_usage: dict[str, int] = field(default_factory=dict)
    user_sim_usage: dict[str, int] = field(default_factory=dict)
    agent_cost_usd: float | None = None
    user_sim_cost_usd: float | None = None
    # Inference wall-clock (seconds) spent inside the agent-under-test's LLM
    # calls for this task, EXCLUDING the user simulator, tool execution, and
    # grading. Unlike elapsed_seconds (total task wall-clock), this is a
    # model-intrinsic latency signal: sec/generation = agent_llm_seconds /
    # agent_usage["num_calls"]. 0.0 when the agent client is not usage-tracked
    # or the trace predates this feature.
    agent_llm_seconds: float = 0.0
    # Model names recorded on the trace so cost can be recomputed offline (e.g.
    # by re-running the analysis report after a pricing.yaml correction) without
    # needing the run's environment.
    agent_model: str | None = None
    user_sim_model: str | None = None
    user_sim_seconds: float = 0.0


# Every key `load_tasks` recognizes on a task mapping. Anything else is almost
# always a typo (e.g. `expected_action` singular) that would silently make the
# task assert nothing, so it is rejected at load time rather than at grading.
_KNOWN_TASK_KEYS = frozenset(
    {
        "id",
        "instruction",
        "expected_actions",
        "domain",
        "initial_state",
        "expected_state",
        "expected_args",
        "expected_response_semantic",
        "expected_order",
        "forbidden_state",
        "forbidden_actions",
        "forbidden_args",
        "expected_args_mode",
        "expected_order_mode",
        "difficulty",
        "capabilities",
    }
)
_ARGS_MODES = frozenset({"any", "first"})
_ORDER_MODES = frozenset({"subsequence", "exact"})
_DIFFICULTIES = frozenset({"easy", "medium", "hard"})
_SEMANTIC_REGEX_TOKENS = ("|", "\\", ".*", "^")


def _validate_task_item(item: dict, path: str, index: int) -> None:
    """Validate one raw task mapping before it is turned into a ``Task``.

    Catches the mistakes that would otherwise pass silently and neuter grading:
    a missing ``id``/``instruction``, an unknown key (usually a typo that drops
    the assertion), or an out-of-range enum for the mode/difficulty fields.
    Raises ``ValueError`` naming the file, task, and offending value.
    """
    where = f"{path} (task #{index})"
    if not isinstance(item, dict):
        raise TypeError(
            f"{where}: each task must be a mapping, got {type(item).__name__}"
        )
    for required in ("id", "instruction"):
        if not item.get(required):
            raise ValueError(f"{where}: missing required key '{required}'")
    tid = item["id"]
    unknown = set(item) - _KNOWN_TASK_KEYS
    if unknown:
        raise ValueError(
            f"{path}: task '{tid}' has unknown key(s) {sorted(unknown)}; "
            f"allowed keys are {sorted(_KNOWN_TASK_KEYS)}"
        )
    for field_name, allowed in (
        ("expected_args_mode", _ARGS_MODES),
        ("expected_order_mode", _ORDER_MODES),
        ("difficulty", _DIFFICULTIES),
    ):
        value = item.get(field_name)
        if value is not None and value not in allowed:
            raise ValueError(
                f"{path}: task '{tid}' has invalid {field_name}={value!r}; "
                f"expected one of {sorted(allowed)}"
            )
    for field_name in ("expected_response_semantic",):
        value = item.get(field_name)
        if value is not None and (
            not isinstance(value, list)
            or not value
            or not all(isinstance(entry, str) and entry.strip() for entry in value)
        ):
            raise ValueError(
                f"{path}: task '{tid}' {field_name} must be a non-empty list "
                "of non-empty strings"
            )
    semantic_criteria = item.get("expected_response_semantic") or []
    for criterion in semantic_criteria:
        if any(token in criterion for token in _SEMANTIC_REGEX_TOKENS):
            raise ValueError(
                f"{path}: task '{tid}' expected_response_semantic must contain "
                "plain-language requirements, not regex syntax"
            )

    expected_actions = set(item.get("expected_actions") or [])
    expected_order = set(item.get("expected_order") or [])
    expected_args_tools = set((item.get("expected_args") or {}).keys())
    forbidden_actions = set(item.get("forbidden_actions") or [])
    contradictory_actions = (
        expected_actions | expected_order | expected_args_tools
    ) & forbidden_actions
    if contradictory_actions:
        raise ValueError(
            f"{path}: task '{tid}' requires and forbids the same action(s) "
            f"{sorted(contradictory_actions)}"
        )

    declared_expected_tools = expected_actions | expected_order
    orphaned_args = expected_args_tools - declared_expected_tools
    if orphaned_args:
        raise ValueError(
            f"{path}: task '{tid}' has expected_args for undeclared expected "
            f"action(s) {sorted(orphaned_args)}"
        )


def load_tasks(path: str) -> list[Task]:
    """Load tasks from a YAML file.

    Expected format:
        tasks:
          - id: task_001
            instruction: |
              You are Alex Chen...
            expected_actions:
              - get_invoice
              - process_refund

    Refusal/policy tasks may additionally assert that restricted tools are NOT
    used (the task fails if they are, even when the final text sounds compliant):
        forbidden_actions:
          - process_refund            # must never be called
        forbidden_args:
          verify_cart_by_email:       # must never be called with this arg
            email: victim@example.com

    Safety/destructive tasks can tighten the arg/order checks (defaults are
    retry-tolerant):
        expected_args_mode: first        # first call to the tool must be correct
        expected_order_mode: exact       # listed tools must occur exactly, in order
    """
    # Frozen domain membership and base-task metadata are centralized in
    # benchmark/benchmark.yaml. Capability slices live on each task because one
    # domain catalog intentionally contains many kinds of work. Ad hoc task files
    # remain supported for tests and local development.
    from harness.benchmark_spec import _load_yaml, load_manifest, task_metadata

    data = _load_yaml(Path(path))
    manifest = load_manifest()

    tasks = []
    for i, item in enumerate(data.get("tasks", [])):
        _validate_task_item(item, path, i)
        metadata = task_metadata(path, manifest)
        if metadata and item.get("domain") != metadata["domain"]:
            raise ValueError(
                f"{path}: task '{item['id']}' declares domain "
                f"{item.get('domain')!r}, frozen domain file requires "
                f"{metadata['domain']!r}"
            )
        tasks.append(
            Task(
                id=item["id"],
                instruction=item["instruction"],
                expected_actions=item.get("expected_actions", []),
                domain=metadata.get("domain", item.get("domain")),
                initial_state=item.get("initial_state"),
                expected_state=item.get("expected_state"),
                expected_args=item.get("expected_args"),
                expected_response_semantic=item.get("expected_response_semantic"),
                expected_order=item.get("expected_order"),
                forbidden_state=item.get("forbidden_state"),
                forbidden_actions=item.get("forbidden_actions"),
                forbidden_args=item.get("forbidden_args"),
                expected_args_mode=item.get("expected_args_mode", "any"),
                expected_order_mode=item.get("expected_order_mode", "subsequence"),
                difficulty=item.get("difficulty"),
                capabilities=list(item.get("capabilities", [])),
                task_set=metadata.get("task_set"),
                benchmark_version=metadata.get("benchmark_version"),
                dataset_version=metadata.get("dataset_version"),
            )
        )
    return tasks


def _usage_delta(before, after) -> dict[str, int]:
    """Per-task usage as the field-wise diff of two UsageTotals snapshots.

    The tracked clients live for the whole run, so a single task's usage is the
    difference between accumulator snapshots taken around its turn loop. Returns
    a plain dict keyed by the canonical usage fields (+ num_calls) for JSON
    serialization into the trace.
    """
    fields = (
        "prompt_tokens",
        "completion_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
        "total_tokens",
        "num_calls",
    )
    return {f: getattr(after, f) - getattr(before, f) for f in fields}


def _llm_seconds_delta(before, after) -> float:
    """Per-task inference wall-clock as the diff of two UsageTotals snapshots.

    This is the time spent inside the tracked client's chat()/stream_chat()
    calls only — for the agent tracker it excludes the user simulator, tool
    execution, and grading, so it is a model-intrinsic latency signal. Divide by
    the same window's num_calls for sec/generation.
    """
    return after.wall_seconds - before.wall_seconds


def _cost_for(model: str, usage: dict[str, int]) -> float | None:
    """USD cost for a usage dict under `model`'s pricing, or None if unpriced."""
    from harness.llm.pricing import cost_usd, resolve_model

    return cost_usd(usage, resolve_model(model))


def _create_llm_client(
    provider: str,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    bedrock_compat: bool = False,
    reasoning_effort: str | None = None,
):
    """Create an LLM client from provider/model strings."""
    from harness.llm import create_llm_client

    return create_llm_client(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        bedrock_compat=bedrock_compat,
        reasoning_effort=reasoning_effort,
    )


def _write_text(path: Path, text: str) -> None:
    """Blocking file write — call via ``asyncio.to_thread`` from async code."""
    with open(path, "w") as f:
        f.write(text)


async def _aclose_client(obj) -> None:
    """Best-effort close of an object's LLM client (or the client itself).

    Accepts either an agent (closes its ``.llm``) or a client directly. Never
    raises — teardown failures are logged at debug and swallowed so cleanup of
    one client cannot mask a run's real result or block closing the others.
    """
    client = getattr(obj, "llm", obj)
    aclose = getattr(client, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception:
        logger.debug("Error closing LLM client during teardown", exc_info=True)


def _create_agent(config_path: str | None, environment=None):
    """Create the agent (graph or simple) from config."""
    from harness.graph_cli import _create_graph_agent

    return _create_graph_agent(config_path, environment=environment)


def _user_reasoning_effort(defaults: dict[str, str]) -> str | None:
    """Resolve only the user-specific reasoning setting and YAML fallback."""
    from harness.benchmark_spec import executor_value

    return executor_value("USER_LLM_REASONING_EFFORT", defaults) or None


async def run_task(
    task: Task,
    agent,
    user_sim,
    semantic_judge=None,
    *,
    max_turns: int,
    timeout_seconds: int,
    agent_step_timeout_seconds: int,
    user_step_timeout_seconds: int,
) -> ConversationTrace:
    """Run a single task: agent-user conversation loop.

    The caller must resolve executor configuration before invoking this function;
    ``run_task`` deliberately does not read environment variables or YAML defaults.

    Returns a ConversationTrace with the full conversation log.
    """
    trace = ConversationTrace(
        task_id=task.id,
        expected_actions=task.expected_actions,
        benchmark_version=task.benchmark_version,
        dataset_version=task.dataset_version,
        domain=task.domain,
        capabilities=task.capabilities,
        task_set=task.task_set,
        difficulty=task.difficulty,
    )
    start = time.monotonic()
    deadline = start + timeout_seconds

    def _remaining() -> float:
        """Seconds left before the task deadline (never negative)."""
        return max(0.0, deadline - time.monotonic())

    # Reset user simulator with task instruction
    user_sim.reset(task.instruction)

    # User opens the conversation (empty agent message on first turn).
    # Enforce the task deadline here too — a hung opening call must not stall.
    try:
        user_step_start = time.monotonic()
        user_message = await asyncio.wait_for(
            user_sim.step(""),
            timeout=min(_remaining(), user_step_timeout_seconds),
        )
        trace.user_sim_seconds += time.monotonic() - user_step_start
    except TimeoutError:
        logger.warning("Task %s timed out opening the conversation", task.id)
        trace.timed_out = True
        trace.error = (
            f"UserSimulatorTimeout: opening step exceeded {user_step_timeout_seconds}s"
        )
        trace.error_kind = "user_simulator_timeout"
        trace.elapsed_seconds = time.monotonic() - start
        return trace
    except UserSimulatorError as e:
        trace.error = f"{type(e).__name__}: {e}"
        trace.error_kind = "user_simulator_divergence"
        trace.elapsed_seconds = time.monotonic() - start
        return trace
    if STOP_SIGNAL in user_message:
        trace.stopped = True
        trace.elapsed_seconds = time.monotonic() - start
        return trace

    trace.turns.append({"role": "user", "content": user_message})
    history: list[ChatMessage] = []

    for _turn in range(max_turns):
        # Enforce the task deadline (cooperative check + hard wait_for below).
        elapsed = time.monotonic() - start
        if _remaining() <= 0:
            logger.warning("Task %s timed out after %.0fs", task.id, elapsed)
            trace.timed_out = True
            # Record a stable error string so EVERY timeout path (cooperative
            # deadline check + the wait_for paths below) is graded and reported
            # identically. error_kind is deliberately left unset: a bare timeout
            # is transient infra, excluded from the denominator, unless a human
            # later stamps "timeout_no_convergence".
            trace.error = (
                f"TimeoutError: task exceeded {timeout_seconds}s "
                f"(deadline reached between turns)"
            )
            break

        logger.info("  Turn %d/%d (%.0fs elapsed)", _turn + 1, max_turns, elapsed)

        # Agent responds — hard-bounded by the remaining task deadline so a
        # stuck gateway call cannot stall the whole benchmark run.
        try:
            response = await asyncio.wait_for(
                agent.run(user_message, history=history),
                timeout=min(_remaining(), agent_step_timeout_seconds),
            )
        except TimeoutError:
            logger.warning(
                "Task %s timed out during agent.run on turn %d", task.id, _turn + 1
            )
            trace.timed_out = True
            trace.error = (
                f"AgentTimeout: step exceeded {agent_step_timeout_seconds}s "
                f"on turn {_turn + 1}"
            )
            trace.error_kind = "agent_timeout"
            break
        except Exception as e:
            logger.error("Agent error on turn %d: %s", _turn + 1, e)
            trace.error = f"{type(e).__name__}: {e}"
            trace.error_kind = getattr(e, "error_kind", None)
            break

        # Collect tool call details for this turn. Current agent runtimes set
        # ``ok`` from the execution layer, independently of the business result.
        # The payload heuristic remains only for legacy/custom agents that omit
        # execution metadata.
        turn_tool_calls = []
        for tc in response.tool_calls_made:
            result = tc.get("result", "")
            ok = tc.get("ok")
            if ok is None:
                ok = not is_error_result(result)
            trace.tools_called.append(tc["name"])
            detail = {
                "turn": _turn,
                "name": tc["name"],
                "arguments": tc.get("arguments", {}),
                "result": result,
                "ok": ok,
            }
            turn_tool_calls.append({**tc, "ok": ok})
            trace.tool_calls_detail.append(detail)

        agent_content = (response.content or "").strip()
        if not agent_content:
            agent_content = "[EMPTY_RESPONSE]"
            logger.warning(
                "Task %s turn %d: agent returned empty content", task.id, _turn + 1
            )

        # Update history
        history.append(ChatMessage(role=Role.USER, content=user_message))
        history.append(ChatMessage(role=Role.ASSISTANT, content=agent_content))

        # Track topics and handoffs
        topic = getattr(response, "topic", None)
        turn_handoffs = getattr(response, "handoffs", [])
        if topic and (not trace.topics_visited or trace.topics_visited[-1] != topic):
            trace.topics_visited.append(topic)
        for h in turn_handoffs:
            trace.handoffs.append(h)

        trace.turns.append(
            {
                "role": "assistant",
                "content": agent_content,
                "tool_calls": turn_tool_calls,
                "iterations": response.iterations,
                "topic": topic,
                "handoffs": turn_handoffs,
            }
        )

        # User simulator responds — also bounded by the task deadline.
        try:
            user_step_start = time.monotonic()
            user_message = await asyncio.wait_for(
                user_sim.step(agent_content),
                timeout=min(_remaining(), user_step_timeout_seconds),
            )
            trace.user_sim_seconds += time.monotonic() - user_step_start
        except TimeoutError:
            logger.warning(
                "Task %s timed out during user_sim.step on turn %d", task.id, _turn + 1
            )
            trace.timed_out = True
            trace.error = (
                "UserSimulatorTimeout: step exceeded "
                f"{user_step_timeout_seconds}s on turn {_turn + 1}"
            )
            trace.error_kind = "user_simulator_timeout"
            break
        except UserSimulatorError as e:
            # Exhausted-retry / wedged user simulator is an infrastructure
            # failure, not a clean end of conversation (P0.6).
            logger.error("User simulator failed on turn %d: %s", _turn + 1, e)
            trace.error = f"{type(e).__name__}: {e}"
            trace.error_kind = "user_simulator_divergence"
            break

        if STOP_SIGNAL in user_message:
            trace.stopped = True
            break

        # An empty user message means the simulator produced nothing usable
        # even after its internal retries. Treat it as a user-simulator error,
        # not a clean stop — otherwise provider/gateway regressions masquerade
        # as the user politely ending the conversation (P0.6).
        if not user_message or not user_message.strip():
            logger.error(
                "User simulator returned empty message on turn %d — recording as error",
                _turn + 1,
            )
            trace.error = "UserSimulatorError: empty user message after retries"
            break

        trace.turns.append({"role": "user", "content": user_message})

    trace.num_turns = len([t for t in trace.turns if t["role"] == "user"])
    trace.elapsed_seconds = time.monotonic() - start

    # Check which expected actions were actually called (count-aware). Only
    # successfully EXECUTED calls count. Negative business outcomes such as
    # ``found: false`` still count; malformed arguments and exceptions do not.
    called_counts: dict[str, int] = {}
    for detail in trace.tool_calls_detail:
        if detail.get("ok", True):
            name = detail["name"]
            called_counts[name] = called_counts.get(name, 0) + 1
    trace.actions_matched = []
    for a in task.expected_actions:
        if called_counts.get(a, 0) > 0:
            trace.actions_matched.append(a)
            called_counts[a] -= 1

    # Argument- and order-level grading must see only successfully EXECUTED
    # calls, for the same reason as action matching (P0.1). Forbidden-action
    # checks below deliberately keep the full list — there the *attempt* is the
    # violation regardless of whether the invocation succeeded.
    ok_calls_detail = [d for d in trace.tool_calls_detail if d.get("ok", True)]
    ok_tools_called = [d["name"] for d in ok_calls_detail]

    # Stricter, optional grading (only when the task declares the assertion).
    if task.expected_args:
        from harness.grading import match_expected_args

        trace.args_passed, trace.args_failures = match_expected_args(
            ok_calls_detail, task.expected_args, mode=task.expected_args_mode
        )
    if task.expected_response_semantic and not trace.error:
        if semantic_judge is None:
            trace.error = (
                "SemanticJudgeError: task declares expected_response_semantic "
                "but no semantic judge is configured"
            )
            trace.error_kind = "grader_infrastructure"
        else:
            try:
                outcome = await semantic_judge.grade(
                    task_id=task.id,
                    task_instruction=task.instruction,
                    criteria=task.expected_response_semantic,
                    turns=trace.turns,
                )
            except Exception as exc:
                # Grader failures are benchmark infrastructure failures, never
                # evidence that the agent-under-test answered incorrectly.
                logger.error("Semantic judge failed for task %s: %s", task.id, exc)
                trace.error = f"{type(exc).__name__}: {exc}"
                trace.error_kind = "grader_infrastructure"
            else:
                trace.response_judge = outcome.audit
                trace.response_passed = outcome.passed
                trace.response_failures.extend(outcome.failures)
    if task.expected_order:
        from harness.grading import check_order

        trace.order_passed, trace.order_failures = check_order(
            ok_tools_called, task.expected_order, mode=task.expected_order_mode
        )
    if task.forbidden_actions or task.forbidden_args:
        from harness.grading import check_forbidden

        trace.forbidden_passed, trace.forbidden_failures = check_forbidden(
            trace.tool_calls_detail,
            forbidden_actions=task.forbidden_actions,
            forbidden_args=task.forbidden_args,
        )

    return trace


async def run_benchmark(
    tasks_path: str,
    config_path: str | None = None,
    output_dir: str = "bench_output",
    max_turns: int | None = None,
    filter_tasks: list[str] | None = None,
    trials: int | None = None,
    base_seed: int | None = None,
) -> list[ConversationTrace]:
    """Run all tasks and save traces.

    Args:
        tasks_path: Path to tasks YAML file.
        config_path: Path to agent graph config YAML.
        output_dir: Directory to write conversation traces.
        max_turns: Maximum turns per task; defaults to MAX_TURNS, then the
            executor config.
        filter_tasks: If provided, only run tasks whose IDs are in this list.
        trials: Number of independent runs per task (for pass^k reliability).
            Defaults to the executor config (single-shot by default).
        base_seed: Base for deterministic per-task/per-trial framework seeds;
            defaults to the executor config.

    Returns:
        List of ConversationTrace results (length = tasks * trials).
    """
    from harness.benchmark_spec import (
        executor_bool,
        executor_config,
        executor_environment,
        executor_value,
        load_manifest,
    )

    executor = executor_config(load_manifest())
    defaults = executor_environment(executor)
    max_turns = (
        int(executor_value("MAX_TURNS", defaults)) if max_turns is None else max_turns
    )
    trials = int(executor_value("TRIALS", defaults)) if trials is None else trials
    base_seed = (
        int(executor_value("BASE_SEED", defaults)) if base_seed is None else base_seed
    )
    trials = max(1, trials)
    tasks = load_tasks(tasks_path)
    if filter_tasks:
        tasks = [t for t in tasks if t.id in filter_tasks]
        logger.info("Filtered to %d tasks: %s", len(tasks), [t.id for t in tasks])
    logger.info("Loaded %d tasks from %s", len(tasks), tasks_path)

    # Create default agent (for non-stateful tasks)
    default_agent = _create_agent(config_path)

    # Create user simulator LLM (separate config from agent)
    user_provider = executor_value("USER_LLM_PROVIDER", defaults)
    user_model = executor_value("USER_LLM_MODEL", defaults)
    user_api_key = os.environ.get("USER_LLM_API_KEY", os.environ.get("LLM_API_KEY"))
    user_base_url = os.environ.get("USER_LLM_BASE_URL", os.environ.get("LLM_BASE_URL"))
    user_strategy = executor_value("USER_STRATEGY", defaults)
    user_reasoning_effort = _user_reasoning_effort(defaults)

    # Wrap the user-sim client too, but track it separately from the agent so
    # its cost is reported apart from the model-under-test (a benchmark artifact,
    # not attributable to the model's cost-per-task).
    from harness.llm.usage_tracker import UsageTrackingClient

    user_llm = UsageTrackingClient(
        _create_llm_client(
            user_provider,
            user_model,
            user_api_key,
            user_base_url,
            reasoning_effort=user_reasoning_effort,
        )
    )
    user_temperature = float(executor_value("USER_TEMPERATURE", defaults))
    user_max_tokens = int(executor_value("USER_MAX_TOKENS", defaults))
    user_step_timeout = int(executor_value("USER_STEP_TIMEOUT", defaults))
    user_max_message_chars = int(executor_value("USER_MAX_MESSAGE_CHARS", defaults))
    user_sim = load_user_simulator(
        strategy=user_strategy,
        llm=user_llm,
        max_tokens=user_max_tokens,
        temperature=user_temperature,
        max_message_chars=user_max_message_chars,
    )

    # Create a dedicated semantic judge only when the selected task set needs
    # one. It has independent, explicit configuration and is never included in
    # the agent-under-test's cost or latency metrics.
    semantic_judge = None
    judge_config = None
    if any(task.expected_response_semantic for task in tasks):
        from harness.semantic_grader import SemanticJudge, SemanticJudgeConfig

        judge_provider = executor_value("JUDGE_LLM_PROVIDER", defaults)
        judge_model = executor_value("JUDGE_LLM_MODEL", defaults)
        judge_api_key = os.environ.get(
            "JUDGE_LLM_API_KEY", user_api_key or os.environ.get("LLM_API_KEY")
        )
        judge_base_url = os.environ.get(
            "JUDGE_LLM_BASE_URL", user_base_url or os.environ.get("LLM_BASE_URL")
        )
        judge_reasoning_effort = (
            executor_value("JUDGE_LLM_REASONING_EFFORT", defaults) or None
        )
        cache_enabled = executor_bool("JUDGE_CACHE_ENABLED", defaults)
        judge_config = SemanticJudgeConfig(
            provider=judge_provider,
            model=judge_model,
            temperature=float(executor_value("JUDGE_TEMPERATURE", defaults)),
            max_tokens=int(executor_value("JUDGE_MAX_TOKENS", defaults)),
            max_attempts=int(executor_value("JUDGE_MAX_ATTEMPTS", defaults)),
            reasoning_effort=judge_reasoning_effort,
            prompt_version=executor_value("JUDGE_PROMPT_VERSION", defaults),
            endpoint_fingerprint=(
                hashlib.sha256(judge_base_url.encode()).hexdigest()[:16]
                if judge_base_url
                else "provider_default"
            ),
            cache_dir=Path(executor_value("JUDGE_CACHE_DIR", defaults)),
            cache_enabled=cache_enabled,
        )
        semantic_judge = SemanticJudge(
            _create_llm_client(
                judge_provider,
                judge_model,
                judge_api_key,
                judge_base_url,
                reasoning_effort=judge_reasoning_effort,
            ),
            judge_config,
        )

    # Prepare output directory so traces are available during the run
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    traces_file = out / "traces.json"

    # Run tasks. Each task is run `trials` times (default 1). Multiple trials
    # feed the pass^k reliability metric — they must be independent, so the
    # stateful environment and agent are recreated fresh for every trial.
    task_timeout = int(executor_value("TASK_TIMEOUT", defaults))
    agent_step_timeout = int(executor_value("AGENT_STEP_TIMEOUT", defaults))
    # Agent model name, used both for per-task cost pricing (below) and the
    # run metadata block further down.
    agent_model = executor_value("LLM_MODEL", defaults)
    agent_temperature = float(executor_value("TEMPERATURE", defaults))
    agent_max_tokens = int(executor_value("LLM_MAX_TOKENS", defaults))
    agent_reasoning_effort = executor_value("LLM_REASONING_EFFORT", defaults) or None
    agent_bedrock_compat = executor_bool("BEDROCK_COMPAT", defaults)
    traces: list[ConversationTrace] = []
    for i, task in enumerate(tasks):
        for trial in range(trials):
            from harness.benchmark_spec import derive_trial_seed

            trial_seed = derive_trial_seed(base_seed, task.id, trial)
            random.seed(trial_seed)
            if trials > 1:
                logger.info(
                    "Running task %d/%d: %s (trial %d/%d)",
                    i + 1,
                    len(tasks),
                    task.id,
                    trial + 1,
                    trials,
                )
            else:
                logger.info("Running task %d/%d: %s", i + 1, len(tasks), task.id)

            # Create per-task (per-trial) environment for stateful domains
            env = None
            if task.domain:
                import harness.tools.domains  # noqa: F401 — triggers @register_domain
                from harness.tools.environment import create_environment

                env = create_environment(task.domain, initial_state=task.initial_state)
                agent = _create_agent(config_path, environment=env)
                logger.info("  Using stateful domain: %s", task.domain)
            else:
                agent = default_agent

            # Snapshot both usage trackers before the task so we can attribute
            # exactly this task's tokens by diffing after it finishes. We resolve
            # the bound ``snapshot`` method once (via getattr, so a non-tracked
            # client — e.g. a plain agent — is a no-op) and reuse it for the
            # after-snapshot, which keeps the before/after calls provably paired.
            agent_snapshot = getattr(getattr(agent, "llm", None), "snapshot", None)
            agent_before = agent_snapshot() if agent_snapshot is not None else None
            user_snapshot = getattr(getattr(user_sim, "llm", None), "snapshot", None)
            user_before = user_snapshot() if user_snapshot is not None else None

            try:
                trace = await run_task(
                    task,
                    agent,
                    user_sim,
                    semantic_judge=semantic_judge,
                    max_turns=max_turns,
                    timeout_seconds=task_timeout,
                    agent_step_timeout_seconds=agent_step_timeout,
                    user_step_timeout_seconds=user_step_timeout,
                )
            except Exception as e:
                logger.error("  Task %s failed with error: %s", task.id, e)
                trace = ConversationTrace(
                    task_id=task.id,
                    expected_actions=task.expected_actions,
                    benchmark_version=task.benchmark_version,
                    dataset_version=task.dataset_version,
                    domain=task.domain,
                    capabilities=task.capabilities,
                    task_set=task.task_set,
                    difficulty=task.difficulty,
                    error=f"{type(e).__name__}: {e}",
                    error_kind=getattr(e, "error_kind", None),
                )
            trace.trial = trial
            trace.seed = trial_seed

            # Attribute per-task token usage + cost (agent vs. user-sim split).
            if agent_before is not None and agent_snapshot is not None:
                agent_after = agent_snapshot()
                trace.agent_usage = _usage_delta(agent_before, agent_after)
                trace.agent_llm_seconds = _llm_seconds_delta(agent_before, agent_after)
                trace.agent_model = agent_model
                trace.agent_cost_usd = _cost_for(agent_model, trace.agent_usage)
            if user_before is not None and user_snapshot is not None:
                trace.user_sim_usage = _usage_delta(user_before, user_snapshot())
                trace.user_sim_model = user_model
                trace.user_sim_cost_usd = _cost_for(user_model, trace.user_sim_usage)

            # Validate final state if expected_state is defined
            if env is not None and task.expected_state:
                from harness.tools.state_validation import validate_state

                passed, failures = validate_state(env.get_state(), task.expected_state)
                trace.state_passed = passed
                trace.state_failures = failures
                if passed:
                    logger.info("  State validation: PASSED")
                else:
                    logger.warning("  State validation: FAILED — %s", failures)

            # Collateral-damage check: assert no unintended state changes
            if env is not None and task.forbidden_state:
                from harness.tools.state_validation import validate_forbidden

                passed, failures = validate_forbidden(
                    env.get_state(), task.forbidden_state
                )
                trace.collateral_passed = passed
                trace.collateral_failures = failures
                if not passed:
                    logger.warning("  Collateral-damage check: FAILED — %s", failures)

            traces.append(trace)

            # Write traces incrementally so partial results are available during
            # the run. Offloaded to a thread so the blocking disk write doesn't
            # stall the event loop (ASYNC230); we await immediately, so the
            # snapshot passed to the thread is never mutated concurrently.
            traces_json = json.dumps([asdict(t) for t in traces], indent=2)
            await asyncio.to_thread(_write_text, traces_file, traces_json)

            if trace.error:
                logger.info("  %s: ERROR — %s", task.id, trace.error)
            else:
                matched = len(trace.actions_matched)
                expected = len(trace.expected_actions)
                logger.info(
                    "  %s: %d turns, %s, actions %d/%d",
                    task.id,
                    trace.num_turns,
                    "stopped"
                    if trace.stopped
                    else "timeout"
                    if trace.timed_out
                    else "max_turns",
                    matched,
                    expected,
                )

            # A stateful-domain task builds a fresh agent per trial; close its
            # client now so httpx connection pools don't accumulate across the
            # run. The shared default_agent is reused and closed once at the end.
            if env is not None and agent is not default_agent:
                await _aclose_client(agent)

    # Collect run metadata (agent_model already resolved above for cost pricing)
    agent_provider = executor_value("LLM_PROVIDER", defaults)
    user_sim_model = user_model
    # Record the ACTUAL engine used, derived from the constructed agent's type,
    # not from the USE_BURR env var, so the trace is labeled honestly.
    actual_mode = (
        "Burr (state-machine ReAct)"
        if type(default_agent).__name__ == "GraphAgentBurr"
        else "plain (simple ReAct)"
    )

    run_meta = {
        "benchmark_version": tasks[0].benchmark_version if tasks else None,
        "dataset_version": tasks[0].dataset_version if tasks else None,
        "domain": tasks[0].domain if tasks else None,
        "capabilities": sorted(
            {capability for task in tasks for capability in task.capabilities}
        ),
        "task_set": tasks[0].task_set if tasks else None,
        "base_seed": base_seed,
        "agent_model": agent_model,
        "agent_provider": agent_provider,
        "agent_temperature": agent_temperature,
        "agent_max_tokens": agent_max_tokens,
        "agent_reasoning_effort": agent_reasoning_effort,
        "agent_bedrock_compat": agent_bedrock_compat,
        "user_sim_model": user_sim_model,
        "user_sim_strategy": user_strategy,
        "user_sim_temperature": user_temperature,
        "user_sim_max_tokens": user_max_tokens,
        "user_sim_reasoning_effort": user_reasoning_effort,
        "task_timeout_seconds": task_timeout,
        "agent_step_timeout_seconds": agent_step_timeout,
        "user_sim_step_timeout_seconds": user_step_timeout,
        "user_sim_max_message_chars": user_max_message_chars,
        "semantic_judge": judge_config.public_dict() if judge_config else None,
        "mode": actual_mode,
        "max_turns": max_turns,
        "trials": trials,
        "tasks_path": tasks_path,
        "config_path": config_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    run_file = out / "run.json"
    await asyncio.to_thread(
        _write_text, run_file, json.dumps(run_meta, indent=2, sort_keys=True) + "\n"
    )

    # Save final report
    report = _build_report(traces, run_meta)
    report_file = out / "report.md"
    await asyncio.to_thread(_write_text, report_file, report)
    logger.info("Report saved to %s", report_file)

    # Print summary
    _print_summary(traces)

    # Release network resources: the shared default agent's client and the
    # user-simulator client both live for the whole run. Per-task agents were
    # already closed inside the loop.
    await _aclose_client(default_agent)
    await _aclose_client(user_llm)
    if semantic_judge is not None:
        await _aclose_client(semantic_judge.client)

    return traces


def _trace_action_pass(t: ConversationTrace) -> bool:
    return compute_action_pass(t.expected_actions, t.actions_matched)


def _trace_status_icon(t: ConversationTrace) -> str:
    """Status icon for a trace, using the shared grading semantics (P1.1)."""
    return compute_status_icon(
        error=t.error,
        error_kind=t.error_kind,
        stopped=t.stopped,
        action_pass=_trace_action_pass(t),
        state_passed=t.state_passed,
        args_passed=t.args_passed,
        response_passed=t.response_passed,
        order_passed=t.order_passed,
        collateral_passed=t.collateral_passed,
        forbidden_passed=t.forbidden_passed,
    )


def _trace_task_success(t: ConversationTrace) -> bool | None:
    return compute_task_success(
        error=t.error,
        error_kind=t.error_kind,
        action_pass=_trace_action_pass(t),
        state_passed=t.state_passed,
        args_passed=t.args_passed,
        response_passed=t.response_passed,
        order_passed=t.order_passed,
        collateral_passed=t.collateral_passed,
        forbidden_passed=t.forbidden_passed,
    )


def _build_report(traces: list[ConversationTrace], meta: dict[str, Any]) -> str:
    """Build a markdown report for the benchmark run."""
    lines: list[str] = []
    lines.append(f"# Benchmark Report — {meta['timestamp']}")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Setting | Value |")
    lines.append("|---------|-------|")
    lines.append(f"| Benchmark version | `{meta.get('benchmark_version')}` |")
    lines.append(f"| Dataset version | `{meta.get('dataset_version')}` |")
    lines.append(f"| Domain | `{meta.get('domain')}` |")
    lines.append(
        f"| Capabilities | `{', '.join(meta.get('capabilities', [])) or None}` |"
    )
    lines.append(f"| Task set | `{meta.get('task_set') or 'ad hoc'}` |")
    lines.append(f"| Base seed | `{meta.get('base_seed')}` |")
    lines.append(f"| Agent model | `{meta['agent_model']}` |")
    lines.append(f"| Agent provider | `{meta['agent_provider']}` |")
    lines.append(
        "| Agent sampling | "
        f"temperature `{meta.get('agent_temperature')}`, "
        f"max tokens `{meta.get('agent_max_tokens')}`, "
        f"reasoning `{meta.get('agent_reasoning_effort')}` |"
    )
    lines.append(f"| User sim model | `{meta['user_sim_model']}` |")
    lines.append(
        f"| User sim strategy | `{meta.get('user_sim_strategy', 'unknown')}` |"
    )
    lines.append(
        "| User sim sampling | "
        f"temperature `{meta.get('user_sim_temperature')}`, "
        f"max tokens `{meta.get('user_sim_max_tokens')}`, "
        f"reasoning `{meta.get('user_sim_reasoning_effort')}` |"
    )
    lines.append(f"| Task timeout | `{meta.get('task_timeout_seconds')}s` |")
    lines.append(
        "| Agent / simulator step budgets | "
        f"`{meta.get('agent_step_timeout_seconds')}s / "
        f"{meta.get('user_sim_step_timeout_seconds')}s` |"
    )
    judge_meta = meta.get("semantic_judge")
    if judge_meta:
        lines.append(f"| Semantic judge model | `{judge_meta['model']}` |")
        lines.append(f"| Semantic judge provider | `{judge_meta['provider']}` |")
        lines.append(f"| Semantic judge prompt | `{judge_meta['prompt_version']}` |")
    lines.append(f"| Mode | {meta['mode']} |")
    lines.append(f"| Max turns | {meta['max_turns']} |")
    lines.append(f"| Tasks | `{meta['tasks_path']}` |")
    lines.append(f"| Config | `{meta['config_path']}` |")
    lines.append("")

    # Summary stats
    total = len(traces)
    errored = sum(1 for t in traces if t.error)
    stopped = sum(1 for t in traces if t.stopped)
    timed_out = sum(1 for t in traces if t.timed_out)
    max_turns_hit = sum(
        1 for t in traces if not t.stopped and not t.timed_out and not t.error
    )
    total_actions_matched = sum(len(t.actions_matched) for t in traces)
    total_actions_expected = sum(len(t.expected_actions) for t in traces)
    total_time = sum(t.elapsed_seconds for t in traces)
    total_agent_llm_time = sum(t.agent_llm_seconds for t in traces)
    total_user_sim_time = sum(t.user_sim_seconds for t in traces)
    non_error = total - errored
    avg_turns = sum(t.num_turns for t in traces) / non_error if non_error else 0
    judged = [t.response_judge for t in traces if t.response_judge]
    judge_calls = sum(judge["usage"].get("num_calls", 0) for judge in judged)
    judge_cache_hits = sum(bool(judge.get("cache_hit")) for judge in judged)
    judge_costs = [
        judge["cost_usd"] for judge in judged if judge.get("cost_usd") is not None
    ]

    # Strict task success (same semantics as analysis.bench_report): a task
    # passes only if actions AND every asserted strict check passed. Errored
    # tasks are excluded from the denominator (P1.1).
    successes = [_trace_task_success(t) for t in traces]
    succeeded = sum(1 for s in successes if s is True)
    scored = sum(1 for s in successes if s is not None)
    tsr_str = (
        f"{succeeded}/{scored} ({succeeded / scored * 100:.0f}%)" if scored else "n/a"
    )

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Tasks | {total} |")
    lines.append(f"| **Task success (strict)** | **{tsr_str}** |")
    lines.append(f"| Completed (stopped) | {stopped}/{total} |")
    lines.append(f"| Errored | {errored}/{total} |")
    lines.append(f"| Timed out | {timed_out} |")
    lines.append(f"| Hit max turns | {max_turns_hit} |")
    lines.append(
        f"| Actions matched | {total_actions_matched}/{total_actions_expected} |"
    )
    lines.append(f"| Avg turns | {avg_turns:.1f} |")
    lines.append(f"| Total time | {total_time:.1f}s |")
    lines.append(f"| Agent LLM time | {total_agent_llm_time:.1f}s |")
    lines.append(f"| User simulator time | {total_user_sim_time:.1f}s |")
    if judged:
        lines.append(f"| Semantic judge calls | {judge_calls} |")
        lines.append(
            f"| Semantic judge cache hits | {judge_cache_hits}/{len(judged)} |"
        )
        lines.append(
            "| Semantic judge cost (excluded from agent cost) | "
            + (f"${sum(judge_costs):.6f}" if len(judge_costs) == len(judged) else "n/a")
            + " |"
        )
    lines.append("")

    # State validation summary (if any tasks have it)
    has_state = any(t.state_passed is not None for t in traces)
    if has_state:
        state_passed = sum(1 for t in traces if t.state_passed is True)
        state_total = sum(1 for t in traces if t.state_passed is not None)
        lines.append(f"| State validated | {state_passed}/{state_total} |")
    lines.append("")

    # Per-task table. The Status column carries the strict verdict (PASS /
    # ARGS_FAIL / ORDER_FAIL / COLLATERAL_FAIL / …) so a task whose actions
    # matched but whose strict checks failed can no longer read as clean (P1.1).
    lines.append("## Results")
    lines.append("")
    if has_state:
        lines.append(
            "| Task | Status | Turns | Stopped | Actions | State | Topics | Time |"
        )
        lines.append(
            "|------|:------:|------:|:-------:|--------:|:-----:|--------|-----:|"
        )
    else:
        lines.append("| Task | Status | Turns | Stopped | Actions | Topics | Time |")
        lines.append("|------|:------:|------:|:-------:|--------:|--------|-----:|")

    for t in traces:
        icon = _trace_status_icon(t)
        status_str = icon if icon == "PASS" else f"**{icon}**"
        if t.error:
            if has_state:
                lines.append(
                    f"| {t.task_id} | {status_str} | - | **ERROR** | - | - | - | - |"
                )
            else:
                lines.append(
                    f"| {t.task_id} | {status_str} | - | **ERROR** | - | - | - |"
                )
        else:
            matched = len(t.actions_matched)
            expected = len(t.expected_actions)
            actions_str = f"{matched}/{expected}" if expected else "n/a"
            stopped_str = (
                "yes" if t.stopped else "**timeout**" if t.timed_out else "**no**"
            )
            topics_str = " → ".join(t.topics_visited) if t.topics_visited else "-"
            if has_state:
                if t.state_passed is True:
                    state_str = "pass"
                elif t.state_passed is False:
                    state_str = "**FAIL**"
                else:
                    state_str = "-"
                lines.append(
                    f"| {t.task_id} | {status_str} | {t.num_turns} | {stopped_str} | {actions_str} | {state_str} | {topics_str} | {t.elapsed_seconds:.1f}s |"
                )
            else:
                lines.append(
                    f"| {t.task_id} | {status_str} | {t.num_turns} | {stopped_str} | {actions_str} | {topics_str} | {t.elapsed_seconds:.1f}s |"
                )

    lines.append("")

    # Per-task detail
    lines.append("## Task Details")
    lines.append("")
    for t in traces:
        lines.append(f"### {t.task_id}")
        lines.append("")
        lines.append(f"- **Status:** {_trace_status_icon(t)}")
        if t.error:
            lines.append(f"- **Error:** `{t.error}`")
        else:
            status = "yes" if t.stopped else "timeout" if t.timed_out else "no"
            lines.append(
                f"- **Turns:** {t.num_turns} | **Stopped:** {status} | **Time:** {t.elapsed_seconds:.1f}s"
            )
            if t.topics_visited:
                lines.append(f"- **Topics:** {' → '.join(t.topics_visited)}")
            if t.handoffs:
                lines.append(f"- **Handoffs:** {', '.join(t.handoffs)}")
            if t.expected_actions:
                lines.append(f"- **Expected actions:** `{t.expected_actions}`")
            if t.tools_called:
                lines.append(f"- **Tools called:** `{t.tools_called}`")
                lines.append(f"- **Actions matched:** `{t.actions_matched}`")
            if t.state_passed is not None:
                lines.append(
                    f"- **State validation:** {'PASSED' if t.state_passed else 'FAILED'}"
                )
                for failure in t.state_failures:
                    lines.append(f"  - {failure}")
            # Strict grading checks — only shown when the task asserted them.
            # Surfacing failures here is the core of P1.1: a matched-actions
            # task that flunked args/response/order/collateral is not "clean".
            for label, passed, fails in (
                ("Args", t.args_passed, t.args_failures),
                ("Response", t.response_passed, t.response_failures),
                ("Order", t.order_passed, t.order_failures),
                ("Collateral", t.collateral_passed, t.collateral_failures),
                ("Forbidden", t.forbidden_passed, t.forbidden_failures),
            ):
                if passed is not None:
                    lines.append(
                        f"- **{label} check:** {'PASSED' if passed else 'FAILED'}"
                    )
                    for failure in fails:
                        lines.append(f"  - {failure}")
            if t.response_judge:
                judge = t.response_judge
                lines.append(
                    "- **Semantic judge:** "
                    f"`{judge['provider']}/{judge['model']}` · "
                    f"prompt `{judge['prompt_version']}` · "
                    f"cache {'hit' if judge['cache_hit'] else 'miss'}"
                )
                for verdict in judge["criteria"]:
                    state = "MET" if verdict["met"] else "NOT MET"
                    lines.append(
                        f"  - `{verdict['criterion_id']}` {state}: "
                        f"{verdict['explanation']}"
                    )
                    if verdict["evidence"]:
                        lines.append(f"    - Evidence: {verdict['evidence']!r}")
        lines.append("")

    return "\n".join(lines) + "\n"


def _print_summary(traces: list[ConversationTrace]) -> None:
    """Print a summary table of benchmark results."""
    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)
    print(f"{'Task':<25} {'Status':>16} {'Turns':>6} {'Actions':>9} {'Time':>8}")
    print("-" * 70)

    for t in traces:
        icon = _trace_status_icon(t)
        if t.error:
            err_short = t.error[:40] + "..." if len(t.error) > 40 else t.error
            print(f"{t.task_id:<25} {icon:>16} {'':>6} {'':>9} {err_short}")
        else:
            matched = len(t.actions_matched)
            expected = len(t.expected_actions)
            actions_str = f"{matched}/{expected}" if expected else "n/a"
            print(
                f"{t.task_id:<25} {icon:>16} {t.num_turns:>6} "
                f"{actions_str:>9} {t.elapsed_seconds:>7.1f}s"
            )

    successes = [_trace_task_success(t) for t in traces]
    succeeded = sum(1 for s in successes if s is True)
    scored = sum(1 for s in successes if s is not None)
    total_stopped = sum(1 for t in traces if t.stopped)
    total_timed_out = sum(1 for t in traces if t.timed_out)
    total_errored = sum(1 for t in traces if t.error)
    tsr = f"{succeeded}/{scored} ({succeeded / scored * 100:.0f}%)" if scored else "n/a"
    print("-" * 70)
    print(f"Task success (strict): {tsr}")
    print(
        f"Total: {len(traces)} tasks, {total_stopped} completed, {total_timed_out} timed out, {total_errored} errored"
    )
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run benchmark tasks against agent")
    parser.add_argument("--tasks", required=True, help="Path to tasks YAML file")
    parser.add_argument(
        "--config", default=None, help="Path to agent graph config YAML"
    )
    parser.add_argument(
        "--output", default="bench_output", help="Output directory for traces"
    )
    parser.add_argument(
        "--max-turns", type=int, default=None, help="Max turns per task"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Independent runs per task for pass^k reliability (default: 1)",
    )
    parser.add_argument("--filter", nargs="+", help="Only run tasks with these IDs")
    parser.add_argument("--seed", type=int, default=None, help="Framework base seed")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Suppress noisy HTTP debug logs unless very verbose
    if log_level == logging.DEBUG:
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)

    # TRACE_LLM=1 — dump all LiteLLM requests/responses to a file
    if os.environ.get("TRACE_LLM", "").lower() in ("1", "true", "yes"):
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        llm_trace_file = out_dir / "llm_trace.log"
        llm_handler = logging.FileHandler(llm_trace_file, mode="w")
        llm_handler.setLevel(logging.DEBUG)
        llm_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(message)s"))
        # LiteLLM logs all requests/responses under these loggers
        for name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy"):
            lg = logging.getLogger(name)
            lg.setLevel(logging.DEBUG)
            lg.addHandler(llm_handler)
        import litellm

        # litellm's stubs type set_verbose as Literal[False]; assigning True is
        # the documented runtime toggle for verbose request/response logging.
        litellm.set_verbose = True  # ty: ignore[invalid-assignment]
        logger.info("LLM tracing enabled → %s", llm_trace_file)

    asyncio.run(
        run_benchmark(
            tasks_path=args.tasks,
            config_path=args.config,
            output_dir=args.output,
            max_turns=args.max_turns,
            filter_tasks=args.filter,
            trials=args.trials,
            base_seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
