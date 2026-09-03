"""Pure grading helpers for benchmark tasks.

These functions turn a conversation trace into pass/fail verdicts beyond the
basic "was the right tool name called" check:

- ``match_expected_args``  — argument-level correctness (right tool, right args)
- ``match_response_contains`` — the agent actually communicated required facts
- ``check_order`` — required tool calls happened in the right relative order

All functions are side-effect free and return ``(passed, failures)`` where
``failures`` is a list of human-readable strings (empty when passed). They are
written to be easy to unit-test in isolation from the async runner.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

# Sentinel assertion values reused from state validation semantics.
_NOT_NULL = "_not_null"

# Unicode hyphen/dash variants that models emit interchangeably with the plain
# ASCII hyphen-minus. We fold them all to "-" before response matching so a
# gold pattern like ``SA-0409`` matches an agent that wrote ``SA‑0409`` with a
# non-breaking hyphen (U+2011). This is an encoding artifact, never a semantic
# difference, so normalizing both sides is safe for every task.
_HYPHEN_VARIANTS = (
    "‐"  # hyphen
    "‑"  # non-breaking hyphen
    "‒"  # figure dash
    "–"  # en dash
    "—"  # em dash
    "―"  # horizontal bar
    "−"  # minus sign
    "﹘"  # small em dash
    "﹣"  # small hyphen-minus
    "－"  # fullwidth hyphen-minus
)
# Unicode apostrophe/single-quote variants that models emit interchangeably with
# the ASCII apostrophe. Folding them to "'" lets a gold pattern like ``can't``
# match an agent that wrote ``can't`` with a right single quotation mark
# (U+2019, the default "smart quote"). Like the hyphen fold, this is a rendering
# artifact, never a semantic difference, so normalizing both sides is safe.
_APOSTROPHE_VARIANTS = (
    "’"  # right single quotation mark (smart apostrophe)
    "‘"  # left single quotation mark
    "‛"  # single high-reversed-9 quotation mark
    "ʼ"  # modifier letter apostrophe
    "＇"  # fullwidth apostrophe
    "`"  # grave accent (used as apostrophe)
    "´"  # acute accent (used as apostrophe)
)
_TEXT_FOLD_TABLE = {ord(ch): "-" for ch in _HYPHEN_VARIANTS}
_TEXT_FOLD_TABLE.update({ord(ch): "'" for ch in _APOSTROPHE_VARIANTS})


def _fold_text(text: str) -> str:
    """Fold Unicode hyphen/dash and apostrophe variants to their ASCII forms.

    Hyphen/dash variants collapse to "-" and apostrophe/single-quote variants
    collapse to "'", so response matching is not defeated by the "smart"
    typographic characters models emit (e.g. ``SA‑0409`` → ``SA-0409``,
    ``can’t`` → ``can't``).
    """
    return text.translate(_TEXT_FOLD_TABLE)


def _norm(value: Any) -> str:
    """Normalize a value to a lowercase string for lenient comparison.

    LLM tool arguments arrive as strings, numbers, or bools; gold values in
    YAML are usually strings. Comparing normalized strings avoids spurious
    type mismatches (e.g. 5 vs "5") while staying strict on content.
    """
    return str(value).strip().lower()


def _arg_matches(actual: Any, expected: Any) -> bool:
    """Check one argument value against an expected assertion.

    Supported expected forms:
      - "_not_null"            → actual must be present and non-empty
      - {"_contains": "x"}     → normalized actual must contain normalized "x"
      - any scalar             → normalized equality
    """
    if expected == _NOT_NULL:
        return actual is not None and _norm(actual) != ""
    if isinstance(expected, dict) and "_contains" in expected:
        return _norm(expected["_contains"]) in _norm(actual)
    return _norm(actual) == _norm(expected)


def _arg_misses(
    args: dict[str, Any], tool: str, arg_asserts: dict[str, Any]
) -> list[str]:
    """Return the list of assertion misses for one call's args (empty = match)."""
    misses = []
    for key, expected in arg_asserts.items():
        if key not in args:
            misses.append(f"{tool}.{key}: missing")
        elif not _arg_matches(args[key], expected):
            misses.append(f"{tool}.{key}: expected {expected!r}, got {args[key]!r}")
    return misses


def match_expected_args(
    tool_calls_detail: list[dict[str, Any]],
    expected_args: dict[str, dict[str, Any]],
    mode: str = "any",
) -> tuple[bool, list[str]]:
    """Verify that each tool was called with the asserted arguments.

    Args:
        tool_calls_detail: list of {"name", "arguments", ...} from the trace.
        expected_args: {tool_name: {arg_key: assertion}}.
        mode: assertion strictness (default ``"any"``):
            - ``"any"``   — a tool passes if *at least one* call to it satisfies
              all asserted args. Tolerates retries/extra calls. (Default; the
              long-standing behavior.)
            - ``"first"`` — the *first* call to the tool must satisfy the args.
              Use for safety/destructive tasks where a wrong first call (e.g. a
              refund with the wrong amount) must fail even if a later call is
              correct.

    Returns:
        (passed, failures).
    """
    if mode not in ("any", "first"):
        raise ValueError(f"unknown args mode: {mode!r} (use 'any' or 'first')")

    failures: list[str] = []
    calls_by_tool: dict[str, list[dict[str, Any]]] = {}
    for call in tool_calls_detail:
        calls_by_tool.setdefault(call.get("name", ""), []).append(
            call.get("arguments", {}) or {}
        )

    for tool, arg_asserts in expected_args.items():
        candidates = calls_by_tool.get(tool, [])
        if not candidates:
            failures.append(f"{tool}: never called (expected args {arg_asserts})")
            continue

        if mode == "first":
            # Only the first call to this tool is allowed to satisfy the assert.
            misses = _arg_misses(candidates[0], tool, arg_asserts)
            if misses:
                failures.extend(f"[first call] {m}" for m in misses)
            continue

        # mode == "any": does any single call satisfy every asserted arg?
        best_miss: list[str] | None = None
        for args in candidates:
            misses = _arg_misses(args, tool, arg_asserts)
            if not misses:
                best_miss = []
                break
            if best_miss is None or len(misses) < len(best_miss):
                best_miss = misses
        failures.extend(best_miss or [])

    return (len(failures) == 0, failures)


def match_response_contains(
    turns: list[dict[str, Any]],
    required: list[str],
) -> tuple[bool, list[str]]:
    """Verify the agent communicated required facts in its messages.

    Each entry in ``required`` must appear in at least one assistant turn.
    Entries are matched case-insensitively as substrings, unless prefixed with
    ``re:`` in which case the remainder is treated as a regular expression.

    Unicode hyphen/dash and apostrophe variants are folded to their ASCII forms
    on both the assistant text and the pattern before matching, so an agent that
    emits ``SA‑0409`` (non-breaking hyphen) still satisfies a gold ``SA-0409``,
    and ``can’t`` (smart apostrophe) still satisfies a gold ``can't``.

    Returns:
        (passed, failures).
    """
    assistant_text = _fold_text(
        "\n".join(t.get("content", "") for t in turns if t.get("role") == "assistant")
    )
    haystack = assistant_text.lower()

    failures: list[str] = []
    for needle in required:
        if needle.startswith("re:"):
            pattern = needle[3:]
            if not re.search(_fold_text(pattern), assistant_text, re.IGNORECASE):
                failures.append(f"response missing pattern: /{pattern}/")
        else:
            if _fold_text(needle).lower() not in haystack:
                failures.append(f"response missing text: {needle!r}")

    return (len(failures) == 0, failures)


def check_order(
    tools_called: list[str],
    expected_order: list[str],
    mode: str = "subsequence",
) -> tuple[bool, list[str]]:
    """Verify required tools were called in the given relative order.

    Args:
        tools_called: the actual ordered tool-call sequence.
        expected_order: the tools whose relative order is required.
        mode: ordering strictness (default ``"subsequence"``):
            - ``"subsequence"`` — ``expected_order`` must appear as a subsequence
              of the actual calls. Other tools may interleave and the listed
              tools may also be called extra times. (Default; long-standing.)
            - ``"exact"`` — considering ONLY the calls whose names are in
              ``expected_order``, that filtered sub-sequence must equal
              ``expected_order`` exactly. Catches extra/duplicate/reordered
              calls to the listed tools (e.g. a stray second process_refund),
              while still allowing unrelated tools to interleave freely.

    Returns:
        (passed, failures).
    """
    if mode not in ("subsequence", "exact"):
        raise ValueError(f"unknown order mode: {mode!r} (use 'subsequence' or 'exact')")

    failures: list[str] = []

    if mode == "exact":
        listed = set(expected_order)
        filtered = [t for t in tools_called if t in listed]
        if filtered != expected_order:
            failures.append(
                f"exact order not satisfied: expected exactly {expected_order} "
                f"among the listed tools, got {filtered}; "
                f"actual call order: {tools_called}"
            )
        return (len(failures) == 0, failures)

    idx = 0
    for tool in tools_called:
        if idx < len(expected_order) and tool == expected_order[idx]:
            idx += 1
    if idx < len(expected_order):
        matched = expected_order[:idx]
        missing_from = expected_order[idx]
        failures.append(
            f"order not satisfied: matched {matched} then expected "
            f"{missing_from!r} before remaining {expected_order[idx + 1 :]}; "
            f"actual call order: {tools_called}"
        )
    return (len(failures) == 0, failures)


def check_forbidden(
    tool_calls_detail: list[dict[str, Any]],
    forbidden_actions: list[str] | None = None,
    forbidden_args: dict[str, dict[str, Any]] | None = None,
) -> tuple[bool, list[str]]:
    """Verify the agent did NOT call restricted tools (P1.10).

    For refusal/policy tasks: the *attempt* to call a forbidden tool is the
    violation, regardless of whether the call errored — a model that reaches
    for another user's record fails even if the tool returns ``{"error": ...}``
    and the final text sounds compliant.

    Args:
        tool_calls_detail: list of {"name", "arguments", ...} from the trace.
        forbidden_actions: tool names that must never be called.
        forbidden_args: {tool: {arg: assertion}} — fail if any call to that
            tool matched the forbidden argument values. Uses the same assertion
            forms as ``match_expected_args`` (_not_null, {_contains}, scalar).

    Returns:
        (passed, failures).
    """
    failures: list[str] = []
    forbidden_actions = forbidden_actions or []
    forbidden_args = forbidden_args or {}

    for call in tool_calls_detail:
        name = call.get("name", "")
        args = call.get("arguments", {}) or {}

        if name in forbidden_actions:
            failures.append(f"{name}: forbidden tool was called (args {args})")

        if name in forbidden_args:
            asserts = forbidden_args[name]
            if all(
                key in args and _arg_matches(args[key], expected)
                for key, expected in asserts.items()
            ):
                failures.append(
                    f"{name}: called with forbidden args {asserts} (got {args})"
                )

    return (len(failures) == 0, failures)


# ---------------------------------------------------------------------------
# Outcome semantics — the single source of truth for per-task pass/fail.
#
# Both analysis.bench_report.TaskResult and bench_runner._build_report consume
# these so the built-in report.md can never look cleaner than the strict CSV
# (P1.1). Keep them pure: inputs are already-computed flags, output is a verdict.
# ---------------------------------------------------------------------------


def compute_action_pass(
    expected_actions: list[str],
    actions_matched: list[str],
) -> bool:
    """True iff every expected action was matched (vacuously true if none).

    Comparison is *count-aware* (multiset containment), not set-based: an action
    expected twice must have been matched twice. ``actions_matched`` already
    preserves multiplicity (the runner appends one entry per successful call, up
    to the expected count), so a task expecting two ``process_refund`` calls does
    not pass when the tool was only called once.
    """
    if not expected_actions:
        return True
    return Counter(expected_actions) <= Counter(actions_matched)


#: ``error_kind`` values that are the model's fault (graded as task failures)
#: rather than transient infrastructure errors (excluded from the denominator).
#:
#: - ``hallucinated_tool``: the model invoked a tool that does not exist.
#: - ``timeout_no_convergence``: the model ran the FULL task-timeout budget and
#:   never converged on an answer. This is NOT auto-emitted by the harness (a
#:   bare timeout is treated as transient infra by default); it is stamped on a
#:   trace only after a human confirms the timeout reflects the model failing to
#:   make progress rather than a flaky gateway/network stall, so the task is
#:   graded as a real failure instead of being dropped from the denominator.
MODEL_FAULT_ERROR_KINDS = frozenset({"hallucinated_tool", "timeout_no_convergence"})


def is_model_fault_error(error: str | None, error_kind: str | None) -> bool:
    """True if an errored trace failed due to a *model* fault, not infra.

    A hallucinated tool call (the model invoked a tool that does not exist) is
    the model failing the task, so it is graded as a failure and kept in the
    success-rate denominator — unlike a transient infra error (gateway timeout,
    disconnect), which is excluded. Keyed off the structured ``error_kind`` tag
    the harness sets when it catches the error.
    """
    return bool(error) and error_kind in MODEL_FAULT_ERROR_KINDS


def compute_task_success(
    *,
    error: str | None,
    action_pass: bool,
    state_passed: bool | None = None,
    args_passed: bool | None = None,
    response_passed: bool | None = None,
    order_passed: bool | None = None,
    collateral_passed: bool | None = None,
    forbidden_passed: bool | None = None,
    error_kind: str | None = None,
) -> bool | None:
    """Strict per-task binary success — the headline signal.

    Returns None for infra-errored tasks (they drop out of the success-rate
    denominator). A model-fault error (e.g. a hallucinated tool call) returns
    False instead — it is a real task failure and stays in the denominator.
    Otherwise True iff actions pass and no asserted strict check failed. Each
    optional check counts against success only when asserted AND False; None
    means "not asserted". Conversation completion and turn count are
    intentionally excluded — they are efficiency signals, not correctness.
    """
    if is_model_fault_error(error, error_kind):
        return False
    if error:
        return None
    if not action_pass:
        return False
    for check in (
        state_passed,
        args_passed,
        response_passed,
        order_passed,
        collateral_passed,
        forbidden_passed,
    ):
        if check is False:
            return False
    return True


def compute_status_icon(
    *,
    error: str | None,
    stopped: bool,
    action_pass: bool,
    state_passed: bool | None = None,
    args_passed: bool | None = None,
    response_passed: bool | None = None,
    order_passed: bool | None = None,
    collateral_passed: bool | None = None,
    forbidden_passed: bool | None = None,
    error_kind: str | None = None,
) -> str:
    """Human-readable status for a single task — same precedence everywhere.

    HALLUCINATED_TOOL / TIMEOUT_NO_CONVERGENCE > ERROR > MAX_TURNS >
    ACTIONS_MISS > FORBIDDEN_FAIL > STATE_FAIL > ARGS_FAIL > RESPONSE_FAIL >
    ORDER_FAIL > COLLATERAL_FAIL > PASS.

    A model-fault error is surfaced distinctly from a transient infra ERROR, and
    each model-fault kind gets its own label (a hallucinated tool call vs a task
    that burned the full timeout without converging). FORBIDDEN_FAIL ranks just
    below action/turn issues because a policy violation (calling a restricted
    tool) is a more serious failure than a merely imperfect-but-allowed call.
    """
    if is_model_fault_error(error, error_kind):
        return (
            "TIMEOUT_NO_CONVERGENCE"
            if error_kind == "timeout_no_convergence"
            else "HALLUCINATED_TOOL"
        )
    if error:
        return "ERROR"
    if not stopped:
        return "MAX_TURNS"
    if not action_pass:
        return "ACTIONS_MISS"
    if forbidden_passed is False:
        return "FORBIDDEN_FAIL"
    if state_passed is False:
        return "STATE_FAIL"
    if args_passed is False:
        return "ARGS_FAIL"
    if response_passed is False:
        return "RESPONSE_FAIL"
    if order_passed is False:
        return "ORDER_FAIL"
    if collateral_passed is False:
        return "COLLATERAL_FAIL"
    return "PASS"
