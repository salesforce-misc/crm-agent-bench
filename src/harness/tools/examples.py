"""Example tools for testing the agent harness."""

from __future__ import annotations

import ast
import datetime
import math
import operator
from collections.abc import Callable
from typing import Any, cast

from harness.tools.registry import ToolRegistry

# Allowlisted names for the calculator: math module functions/constants plus a
# couple of builtins. Used by the AST evaluator below instead of eval().
_CALC_NAMES: dict[str, object] = {
    k: v for k, v in math.__dict__.items() if not k.startswith("_")
}
_CALC_NAMES.update({"abs": abs, "round": round})

# Only these binary/unary operators are permitted. Keys are AST node classes;
# values are the operator functions they map to (typed loosely because the
# operands are dynamically evaluated numbers).
_CALC_BINOPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_CALC_UNARYOPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval_math(expression: str) -> float | int:
    """Evaluate an arithmetic expression via an AST allowlist (no eval()).

    Supports numeric literals, ``+ - * / // % **``, unary ``+/-``, parentheses,
    the math module's functions/constants, and ``abs``/``round``. Any other
    syntax (attribute access, comprehensions, arbitrary names, etc.) raises
    ``ValueError`` — this prevents the Python object traversal that ``eval``
    with empty builtins still permits (Bandit B307).
    """

    def _eval(node: ast.AST) -> float | int:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                # Uniform ValueError contract for every unsupported expression
                # element (not TypeError) so callers catch one exception type.
                raise ValueError(f"unsupported constant: {node.value!r}")  # noqa: TRY004
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _CALC_BINOPS:
            return _CALC_BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_UNARYOPS:
            return _CALC_UNARYOPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Name) and node.id in _CALC_NAMES:
            value = _CALC_NAMES[node.id]
            if callable(value):
                raise ValueError(f"'{node.id}' is a function, not a value")
            # Non-callable allowlisted names are numeric constants (e.g. math.pi).
            return cast("float | int", value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func = _CALC_NAMES.get(node.func.id)
            if func is None or not callable(func):
                raise ValueError(f"unknown function: {node.func.id}")
            if node.keywords:
                raise ValueError("keyword arguments are not supported")
            fn = cast("Callable[..., float | int]", func)
            return fn(*(_eval(arg) for arg in node.args))
        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    tree = ast.parse(expression, mode="eval")
    return _eval(tree.body)


def register_example_tools(registry: ToolRegistry) -> None:
    """Register a set of example tools for testing."""

    @registry.tool("Get the current weather for a city")
    def get_weather(city: str, unit: str = "celsius") -> str:
        """Get current weather for a city. This is a mock implementation."""
        temps = {"celsius": "22", "fahrenheit": "72"}
        temp = temps.get(unit, "22")
        return f"Weather in {city}: {temp} degrees {unit}, partly cloudy"

    @registry.tool("Search a knowledge base for information")
    def search_knowledge(query: str, max_results: int = 3) -> str:
        """Search the knowledge base. This is a mock implementation."""
        return (
            f"Search results for '{query}' (top {max_results}):\n"
            "1. Getting Started Guide - Overview of the platform\n"
            "2. FAQ - Common questions and answers\n"
            "3. Troubleshooting - Steps to resolve common issues"
        )

    @registry.tool("Evaluate a mathematical expression")
    def calculator(expression: str) -> str:
        """Safely evaluate a math expression via an AST allowlist (no eval())."""
        try:
            result = _safe_eval_math(expression)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    @registry.tool("Get the current date and time")
    def get_current_datetime() -> str:
        """Get the current UTC date and time in ISO format."""
        return datetime.datetime.now(datetime.UTC).isoformat()
