"""Sparse state validation for stateful tool environments.

Compares actual final state against expected state assertions.
Only checks paths explicitly specified in expected — everything
else is ignored. Supports nested dicts and special assertions.
"""

from __future__ import annotations

from typing import Any


def validate_state(actual: dict, expected: dict) -> tuple[bool, list[str]]:
    """Compare actual final state against expected state assertions.

    Args:
        actual: The full state dict from ToolEnvironment.get_state().
        expected: Sparse dict — only keys that matter are specified.

    Returns:
        (passed, list_of_failure_descriptions)
    """
    failures: list[str] = []

    for path, expected_val in _flatten(expected):
        # Special assertion: _min_length for lists and dicts
        if path.endswith("._min_length"):
            list_path = path.rsplit("._min_length", 1)[0]
            actual_val = _get_nested(actual, list_path)
            if actual_val is _MISSING:
                failures.append(
                    f"{list_path}: expected collection with min length {expected_val}, but path is missing"
                )
            elif not isinstance(actual_val, (list, dict)):
                failures.append(
                    f"{list_path}: expected list or dict, got {type(actual_val).__name__}"
                )
            elif len(actual_val) < expected_val:
                failures.append(
                    f"{list_path}: expected min length {expected_val}, got {len(actual_val)}"
                )
            continue

        actual_val = _get_nested(actual, path)
        if expected_val == "_not_null":
            if actual_val is _MISSING or actual_val is None:
                failures.append(f"{path}: expected non-null value, got {actual_val!r}")
            continue

        if actual_val is _MISSING:
            failures.append(f"{path}: expected {expected_val!r}, but path is missing")
        elif actual_val != expected_val:
            failures.append(f"{path}: expected {expected_val!r}, got {actual_val!r}")

    return (len(failures) == 0, failures)


def validate_forbidden(actual: dict, forbidden: dict) -> tuple[bool, list[str]]:
    """Collateral-damage check: assert paths did NOT take forbidden values.

    The inverse of ``validate_state`` — used to catch unintended side effects
    (AppWorld-style). Each path in ``forbidden`` lists a value the final state
    must NOT have. ``"_not_null"`` means the path must remain null/absent
    (i.e. the agent must not have created/populated it).

    Args:
        actual: full state dict from ToolEnvironment.get_state().
        forbidden: sparse dict — only paths that must not change/appear.

    Returns:
        (passed, list_of_failure_descriptions). passed=True means no
        collateral damage detected.
    """
    failures: list[str] = []
    for path, bad_val in _flatten(forbidden):
        actual_val = _get_nested(actual, path)
        if bad_val == "_not_null":
            if actual_val is not _MISSING and actual_val is not None:
                failures.append(
                    f"{path}: must remain unset, but was modified to {actual_val!r}"
                )
        elif actual_val is not _MISSING and actual_val == bad_val:
            failures.append(f"{path}: must NOT equal {bad_val!r}, but it does")
    return (len(failures) == 0, failures)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MISSING = object()


def _flatten(d: dict, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten a nested dict into dot-separated (path, value) pairs."""
    items: list[tuple[str, Any]] = []
    for key, val in d.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            items.extend(_flatten(val, path))
        else:
            items.append((path, val))
    return items


def _get_nested(d: dict, path: str) -> Any:
    """Retrieve a nested value by dot-separated path.

    Handles keys that contain dots (e.g. email addresses) by trying
    progressively longer key segments at each level.

    Returns _MISSING sentinel if any key in the path doesn't exist.
    """
    return _get_nested_recursive(d, path.split("."), 0)


def _get_nested_recursive(current: Any, parts: list[str], start: int) -> Any:
    """Try matching greedy key segments from parts[start:]."""
    if start >= len(parts):
        return current
    if not isinstance(current, dict):
        return _MISSING
    for end in range(len(parts), start, -1):
        key = ".".join(parts[start:end])
        if key in current:
            result = _get_nested_recursive(current[key], parts, end)
            if result is not _MISSING:
                return result
    return _MISSING
