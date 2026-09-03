"""Stateful tool environment for benchmarking.

Provides a shared "database" dict that tools read/write across turns,
enabling realistic multi-turn benchmarks with state validation.

Each domain (customer_service, field_service, etc.) subclasses
ToolEnvironment and registers its own tools via closures over self.data.
"""

from __future__ import annotations

import copy
from typing import Any

from harness.tools.registry import ToolRegistry


class ToolEnvironment:
    """Base class for stateful tool environments.

    Subclasses override default_state() and register_tools().
    All tool functions are closures that capture self.data,
    so mutations are visible across all nodes within a task.
    """

    def __init__(self, initial_state: dict | None = None) -> None:
        self.data: dict[str, Any] = _deep_merge(
            self.default_state(), initial_state or {}
        )

    def default_state(self) -> dict[str, Any]:
        """Return the default initial state. Subclasses override."""
        return {}

    def register_tools(self, registry: ToolRegistry) -> None:
        """Register stateful tool functions into the registry. Subclasses override."""
        raise NotImplementedError

    def create_registry(self) -> ToolRegistry:
        """Create a fresh ToolRegistry with all domain tools registered."""
        registry = ToolRegistry()
        self.register_tools(registry)
        return registry

    def get_state(self) -> dict[str, Any]:
        """Return a deep copy of the current state for validation."""
        return copy.deepcopy(self.data)


# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------

DOMAIN_REGISTRY: dict[str, type[ToolEnvironment]] = {}


def register_domain(name: str):
    """Decorator to register a ToolEnvironment subclass as a named domain."""

    def decorator(cls: type[ToolEnvironment]) -> type[ToolEnvironment]:
        DOMAIN_REGISTRY[name] = cls
        return cls

    return decorator


def create_environment(
    domain: str, initial_state: dict | None = None
) -> ToolEnvironment:
    """Factory: instantiate a ToolEnvironment by domain name."""
    if domain not in DOMAIN_REGISTRY:
        available = list(DOMAIN_REGISTRY.keys())
        raise ValueError(f"Unknown domain: {domain!r}. Available: {available}")
    return DOMAIN_REGISTRY[domain](initial_state=initial_state)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base.

    - Dict values are merged recursively.
    - Non-dict values in override replace base values.
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result
