"""Structured agent/harness exceptions.

Kept dependency-free so both the runner and the graph agent can import it
cheaply and tag traces with a stable ``error_kind`` the analyzer keys off.
"""

from __future__ import annotations


class HallucinatedToolError(RuntimeError):
    """The agent called a tool that does not exist at the current graph node.

    Raised when the underlying graph runtime rejects a tool call because the
    model invented a tool name (an LLM hallucination), as opposed to a
    transient infrastructure failure. This is a *model* fault: the analyzer
    grades it as a task failure rather than excluding it from the success-rate
    denominator like an infra error (e.g. a gateway timeout).
    """

    #: Stable tag persisted onto the trace and read by the analyzer.
    error_kind = "hallucinated_tool"
