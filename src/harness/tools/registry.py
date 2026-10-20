"""Tool registry for registering Python functions as agent tools.

Automatically generates OpenAI function-calling tool definitions from
type hints and docstrings.
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, get_type_hints

from harness.models import ToolDefinition, ToolFunction

logger = logging.getLogger(__name__)


# Error result ``status_code`` values that represent a tool executing
# correctly and applying a business rule, NOT a failed execution. These mirror
# HTTP semantics: the tool did its job and the request was intentionally
# rejected (e.g. duplicate, already in target state, policy denial). Such a
# call still satisfies an expected action — the model called the right tool and
# the backend enforced a rule. A bare ``{"error": ...}`` with no ``status_code``
# remains a failure (unknown tool, exception, wrong args, not-found).
BUSINESS_REJECTION_STATUS_CODES = frozenset({403, 409, 422})


@dataclass(frozen=True)
class ToolExecutionResult:
    """A tool's serialized result plus its execution-layer status.

    ``ok`` describes whether Python successfully invoked the registered tool;
    it deliberately does not interpret the tool's business payload.  A lookup
    returning ``{"found": false}``, for example, is a successful execution.
    """

    result: str
    ok: bool


def is_error_result(result: str) -> bool:
    """Best-effort fallback for legacy calls without an explicit ``ok`` flag.

    A result is treated as a failure when it is a JSON object carrying an
    ``"error"`` key. This uniformly covers the three failure modes in the
    harness:
      - unknown-tool errors emitted by ``ToolRegistry.execute`` (``{"error": ...}``)
      - tool exceptions caught by ``ToolRegistry.execute`` (``{"error": ...}``)
      - domain-level failures returned by stateful tools (``{"error": ...}``,
        sometimes alongside other fields such as ``lead_id``).

    Exception: an error result whose ``"status_code"`` is a business-rejection
    code (see ``BUSINESS_REJECTION_STATUS_CODES``) is NOT a failure. The tool
    executed correctly and intentionally rejected the request (duplicate,
    already-in-target-state, policy denial) — the model called the right tool,
    so the action is satisfied even though the result carries ``"error"``.

    New execution paths should use :meth:`ToolRegistry.execute_with_status`,
    which distinguishes invocation failures from successful business outcomes
    without inspecting the payload. Plain-string results and JSON objects
    without an ``"error"`` key are treated as successful. Non-JSON strings are
    always successful.
    """
    if not isinstance(result, str):
        return False
    stripped = result.strip()
    if not stripped.startswith("{"):
        return False
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return False
    if not (isinstance(parsed, dict) and "error" in parsed):
        return False
    # A business-rejection status code means the tool ran correctly and
    # deliberately declined — not a failure.
    return parsed.get("status_code") not in BUSINESS_REJECTION_STATUS_CODES


# Map Python types to JSON Schema types
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(py_type: type) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema type."""
    origin = getattr(py_type, "__origin__", None)
    if origin is Literal:
        choices: tuple[Any, ...] = getattr(py_type, "__args__", ())
        return {
            "type": _TYPE_MAP.get(type(choices[0]), "string"),
            "enum": list(choices),
        }
    if origin is list:
        args = getattr(py_type, "__args__", (Any,))
        return {
            "type": "array",
            "items": _python_type_to_json_schema(args[0]) if args else {},
        }
    if origin is dict:
        return {"type": "object"}
    return {"type": _TYPE_MAP.get(py_type, "string")}


def _build_parameters_schema(func: Callable) -> dict[str, Any]:
    """Build a JSON Schema for a function's parameters from its type hints."""
    hints = get_type_hints(func)
    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        prop = _python_type_to_json_schema(hints.get(name, str))

        # Use docstring-style parameter descriptions if available
        # For now, use the parameter name as a basic description
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            prop["default"] = param.default

        properties[name] = prop

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


class ToolRegistry:
    """Registry for agent tools backed by Python functions."""

    def __init__(self) -> None:
        self._tools: dict[str, _RegisteredTool] = {}

    def register(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Register a function as a tool.

        Args:
            func: The Python function to register.
            name: Tool name (defaults to function name).
            description: Tool description (defaults to docstring first line).
        """
        # `functools.partial` and other callables lack ``__name__``; fall back to
        # a stable placeholder so registration never raises on an exotic callable.
        tool_name = name or getattr(func, "__name__", None) or "tool"
        tool_desc = (
            description
            or (inspect.getdoc(func) or "").split("\n")[0]
            or f"Tool: {tool_name}"
        )
        self._tools[tool_name] = _RegisteredTool(
            func=func,
            name=tool_name,
            description=tool_desc,
            parameters=_build_parameters_schema(func),
        )
        logger.debug("Registered tool: %s", tool_name)

    def tool(self, description: str | None = None) -> Callable:
        """Decorator to register a function as a tool.

        Usage:
            @registry.tool("Search the web for information")
            def search(query: str) -> str:
                ...
        """

        def decorator(func: Callable) -> Callable:
            self.register(func, description=description)
            return func

        return decorator

    def get_definitions(self) -> list[ToolDefinition]:
        """Return OpenAI function-calling tool definitions for all registered tools."""
        return [
            ToolDefinition(
                function=ToolFunction(
                    name=t.name,
                    description=t.description,
                    parameters=t.parameters,
                )
            )
            for t in self._tools.values()
        ]

    async def execute_with_status(
        self, name: str, arguments: dict[str, Any]
    ) -> ToolExecutionResult:
        """Execute a tool and report invocation status independently of output.

        A registered function that returns normally has ``ok=True`` even when
        its domain result is negative (not found, declined, unavailable, or
        unauthorized). Unknown tools, malformed arguments, and exceptions have
        ``ok=False``.

        Returns:
            The serialized tool result and execution-layer status.
        """
        if name not in self._tools:
            return ToolExecutionResult(
                result=json.dumps({"error": f"Unknown tool: {name}"}), ok=False
            )

        tool = self._tools[name]
        try:
            if inspect.iscoroutinefunction(tool.func):
                result = await tool.func(**arguments)
            else:
                result = tool.func(**arguments)

            if not isinstance(result, str):
                result = json.dumps(result, default=str)
            return ToolExecutionResult(result=result, ok=True)
        except Exception:
            # Log the full exception (with traceback) for the operator, but return
            # only a generic failure to the model. Raw exception text can carry
            # stack frames, absolute paths, or backend identifiers we do not want
            # leaking into the conversation transcript or graded output.
            logger.exception("Tool %s failed", name)
            return ToolExecutionResult(
                result=json.dumps({"error": f"Tool '{name}' execution failed"}),
                ok=False,
            )

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a registered tool by name and return its serialized result.

        This compatibility API intentionally omits execution metadata. Agent
        runtimes and graders should use :meth:`execute_with_status` instead.
        """
        return (await self.execute_with_status(name, arguments)).result

    def get_registered_tool(self, name: str) -> _RegisteredTool | None:
        """Return the internal _RegisteredTool by name, or None."""
        return self._tools.get(name)

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)


class _RegisteredTool:
    """Internal representation of a registered tool."""

    __slots__ = ("description", "func", "name", "parameters")

    def __init__(
        self, func: Callable, name: str, description: str, parameters: dict[str, Any]
    ) -> None:
        self.func = func
        self.name = name
        self.description = description
        self.parameters = parameters
