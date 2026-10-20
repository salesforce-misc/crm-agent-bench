"""CLI entry point for the multi-node graph agent.

Usage:
    python -m harness.graph_cli
    python -m harness.graph_cli --config path/to/graph.yaml

Environment variables:
    LLM_PROVIDER: "openai" or "litellm" (defaults to benchmark.yaml)
    LLM_MODEL: Model name (defaults to benchmark.yaml)
    LLM_API_KEY: API key for the LLM provider
    LLM_BASE_URL: Optional base URL override
    USE_BURR: "1"/"true" runs the Burr state-machine ReAct engine per node;
              "0"/"false" runs the plain-Python ReAct loop (defaults to
              benchmark.yaml). Both are dependency-free of any Salesforce package.
    TEMPERATURE: Sampling temperature for the agent LLM (defaults to benchmark.yaml)
    LLM_REASONING_EFFORT: Reasoning level for the agent (defaults to benchmark.yaml)
    GRAPH_CONFIG: Path to YAML graph config file (alternative to --config)
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from harness.graph_agent import GraphAgent, NodeConfig
from harness.models import AgentEvent, ChatMessage, Role
from harness.tools.registry import ToolRegistry
from harness.tools.stubs import stub_response

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from harness.tools.environment import ToolEnvironment


def _create_llm_client():
    """Create LLM client from environment variables."""
    from harness.benchmark_spec import (
        executor_bool,
        executor_environment,
        executor_value,
    )
    from harness.llm import create_llm_client

    defaults = executor_environment()
    return create_llm_client(
        provider=executor_value("LLM_PROVIDER", defaults),
        model=executor_value("LLM_MODEL", defaults),
        api_key=os.environ.get("LLM_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL") or None,
        bedrock_compat=executor_bool("BEDROCK_COMPAT", defaults),
        reasoning_effort=executor_value("LLM_REASONING_EFFORT", defaults) or None,
    )


def _load_graph_config(config_path: str | None) -> dict[str, Any]:
    """Load graph config from YAML file.

    Expected format:
        system_prompt: "You are a helpful agent."
        entry_node: billing
        nodes:
          billing:
            description: "Billing questions, invoices, payments"
            instructions: "Handle billing and invoice inquiries."
            tools:
              - name: get_invoice
                description: "Look up an invoice by ID"
              - name: check_payment
                description: "Check payment status"
          support:
            description: "Technical support, bug reports"
            instructions: "Handle support and troubleshooting."
            tools:
              - name: create_ticket
                description: "Create a support ticket"
    """
    path = config_path or os.environ.get("GRAPH_CONFIG")
    # No config source supplied at all → caller falls back to demo nodes.
    if not path:
        return {}
    # A config path WAS explicitly provided. A typo or bad file must fail fast
    # rather than silently running the demo agent (P0.5).
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Graph config not found: {path!r}. Set a valid --config / GRAPH_CONFIG "
            f"path, or omit it to use the built-in demo nodes."
        )
    with open(path) as f:
        config = yaml.safe_load(f)
    if not config or not isinstance(config, dict):
        raise ValueError(f"Graph config {path!r} is empty or not a YAML mapping.")
    if not config.get("nodes"):
        raise ValueError(f"Graph config {path!r} has no 'nodes' section.")
    return config


def _build_example_nodes() -> dict[str, NodeConfig]:
    """Build example graph nodes when no config is provided."""
    billing_tools = ToolRegistry()
    support_tools = ToolRegistry()
    general_tools = ToolRegistry()

    @billing_tools.tool("Look up an invoice by ID")
    def get_invoice(invoice_id: str) -> str:
        return f"Invoice {invoice_id}: $500.00, status: paid, date: 2024-01-15"

    @billing_tools.tool("Check payment status")
    def check_payment(account_id: str) -> str:
        return f"Account {account_id}: last payment $500 on 2024-01-15, balance: $0"

    @support_tools.tool("Create a support ticket")
    def create_ticket(subject: str, priority: str = "medium") -> str:
        return f"Ticket TK-{hash(subject) % 10000:04d} created: {subject} (priority: {priority})"

    @support_tools.tool("Search knowledge base")
    def search_kb(query: str) -> str:
        return f"KB results for '{query}': 1) Check network connectivity 2) Clear cache 3) Restart the app"

    @general_tools.tool("Get current date and time")
    def get_datetime() -> str:
        from datetime import datetime

        return datetime.now(UTC).isoformat()

    return {
        "billing": NodeConfig(
            tools=billing_tools,
            instructions="Handle billing inquiries, invoices, and payment questions. Be precise with amounts and dates.",
            description="Billing questions, invoices, payments",
        ),
        "support": NodeConfig(
            tools=support_tools,
            instructions="Handle technical support, troubleshooting, and bug reports. Search the knowledge base before answering.",
            description="Technical support, bug reports, troubleshooting",
        ),
        "general": NodeConfig(
            tools=general_tools,
            instructions="Handle general questions that don't fit billing or support.",
            description="General questions, greetings, other",
        ),
    }


_stub_response = stub_response  # backward compat alias


def _build_nodes_from_config(
    config: dict[str, Any],
    environment: ToolEnvironment | None = None,
) -> dict[str, NodeConfig]:
    """Build NodeConfig instances from YAML config.

    If an environment is provided, tools are stateful closures from the
    environment (shared state across nodes). Otherwise, tools are stubs.
    """
    # Build master registry from environment if available
    master_registry = None
    if environment is not None:
        master_registry = environment.create_registry()

    nodes = {}
    for name, node_cfg in config.get("nodes", {}).items():
        if master_registry is not None:
            tools = ToolRegistry()
            missing_tools = []
            for tool_spec in node_cfg.get("tools", []):
                tool_name = tool_spec["name"]
                registered = master_registry.get_registered_tool(tool_name)
                if registered:
                    tools.register(
                        registered.func,
                        name=registered.name,
                        description=registered.description,
                    )
                else:
                    missing_tools.append(tool_name)
            if missing_tools:
                available = sorted(master_registry.tool_names)
                raise ValueError(
                    f"Node '{name}' references unknown tools: {missing_tools}. "
                    f"Available tools in registry: {available}"
                )
        else:
            # Existing stub behavior
            tools = ToolRegistry()
            for tool_spec in node_cfg.get("tools", []):
                tool_name = tool_spec["name"]
                tool_desc = tool_spec.get("description", tool_name)

                def _make_stub(tn: str, td: str) -> Callable[..., Awaitable[str]]:
                    async def stub(**kwargs) -> str:
                        return _stub_response(tn, kwargs, td)

                    stub.__name__ = tn
                    return stub

                tools.register(
                    _make_stub(tool_name, tool_desc),
                    description=tool_desc,
                    name=tool_name,
                )

        nodes[name] = NodeConfig(
            tools=tools,
            instructions=node_cfg.get("instructions", ""),
            description=node_cfg.get("description", ""),
            handoff_to=node_cfg.get("handoff_to", []),
            max_iterations=node_cfg.get("max_iterations", 10),
        )

    return nodes


def _create_graph_agent(config_path: str | None = None, environment=None):
    """Create the graph agent with mode selection.

    Args:
        config_path: Path to YAML graph config file.
        environment: Optional ToolEnvironment for stateful tools.
    """
    from harness.benchmark_spec import (
        executor_bool,
        executor_environment,
        executor_value,
    )

    # Load (and validate) config first so an explicit bad path fails fast
    # before we construct the LLM client (P0.5).
    config = _load_graph_config(config_path)
    # Wrap the agent's LLM client so every downstream call — topic classifier
    # and the per-node ReAct loops, which all share this one instance —
    # accumulates token usage into a single tracker.
    from harness.llm.usage_tracker import UsageTrackingClient

    llm = UsageTrackingClient(_create_llm_client())

    if config and "nodes" in config:
        nodes = _build_nodes_from_config(config, environment=environment)
        system_prompt = config.get("system_prompt", "You are a helpful AI assistant.")
        entry_node = config.get("entry_node")
    else:
        nodes = _build_example_nodes()
        system_prompt = (
            "You are a helpful customer service agent. You can handle billing, "
            "support, and general questions. Use tools when appropriate."
        )
        entry_node = "general"

    defaults = executor_environment()
    temperature = float(executor_value("TEMPERATURE", defaults))
    max_tokens = int(executor_value("LLM_MAX_TOKENS", defaults))

    use_burr = executor_bool("USE_BURR", defaults)
    if use_burr:
        from harness.graph_agent_burr import GraphAgentBurr

        print("[Burr] Multi-node graph agent with Burr state-machine ReAct loop")
        return GraphAgentBurr(
            llm=llm,
            system_prompt=system_prompt,
            nodes=nodes,
            entry_node=entry_node,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    print("[Plain] Multi-node graph agent with simple ReAct loop")
    return GraphAgent(
        llm=llm,
        system_prompt=system_prompt,
        nodes=nodes,
        entry_node=entry_node,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _print_event(event: AgentEvent) -> None:
    """Print an agent event to stdout."""
    if event.type == "text_delta":
        print(event.content or "", end="", flush=True)
    elif event.type == "final":
        print(event.content or "", flush=True)
    elif event.type == "topic_selected":
        print(f"  [Topic: {event.content}]", flush=True)
        print("\nAgent: ", end="", flush=True)
    elif event.type == "tool_call":
        print(
            f"\n  [Calling: {event.tool_name}({event.tool_call.arguments if event.tool_call else ''})]",
            flush=True,
        )
    elif event.type == "tool_result":
        result_preview = (event.tool_result or "")[:200]
        print(f"  [Result: {result_preview}]", flush=True)
        print("\nAgent: ", end="", flush=True)
    elif event.type == "handoff":
        print(f"\n  [Handoff: {event.content}]", flush=True)
        print("\nAgent: ", end="", flush=True)
    elif event.type == "error":
        print(f"\n  [Error: {event.error}]", flush=True)


async def _run_interactive(config_path: str | None = None):
    """Run the interactive graph CLI loop."""
    agent = _create_graph_agent(config_path)
    history: list[ChatMessage] = []

    print("\nGraph Agent CLI")
    print("Type 'quit' or 'exit' to stop. Type 'clear' to reset conversation.\n")

    # Show available nodes
    nodes = agent.nodes
    print("Available nodes:")
    for name, cfg in nodes.items():
        tools_count = len(cfg.tools.get_definitions())
        handoffs = f" -> [{', '.join(cfg.handoff_to)}]" if cfg.handoff_to else ""
        print(
            f"  {name}: {cfg.description or cfg.instructions[:60]} ({tools_count} tools){handoffs}"
        )
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            history.clear()
            print("[Conversation cleared]\n")
            continue

        print("\nAgent: ", end="", flush=True)

        response_parts: list[str] = []
        try:
            async for event in agent.stream(user_input, history=history):
                _print_event(event)
                if event.type in ("text_delta", "final", "node_result"):
                    response_parts.append(event.content or "")
        except Exception as e:
            print(f"\n[Error: {e}]")
            continue

        history.append(ChatMessage(role=Role.USER, content=user_input))
        if response_parts:
            history.append(
                ChatMessage(role=Role.ASSISTANT, content="".join(response_parts))
            )

        print()  # newline after response


def main():
    config_path = None
    if len(sys.argv) > 2 and sys.argv[1] == "--config":
        config_path = sys.argv[2]
    asyncio.run(_run_interactive(config_path))


if __name__ == "__main__":
    main()
