"""CLI entry point for the agent harness.

Usage:
    python -m harness.cli

Environment variables:
    LLM_PROVIDER: "openai" (default) or "litellm"
    LLM_MODEL: Model name (default: "gpt-4o")
    LLM_API_KEY: API key for the LLM provider
    LLM_BASE_URL: Optional base URL override
    AGENT_PROMPT: Path to YAML prompt file (optional, uses default if not set)
    USE_BURR: "1" (default) runs the Burr state-machine ReAct engine; "0" runs
              the plain-Python ReAct loop.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from harness.models import AgentEvent, ChatMessage


def _create_llm_client():
    """Create LLM client from environment variables."""
    from harness.llm import create_llm_client

    return create_llm_client(
        provider=os.environ.get("LLM_PROVIDER", "openai"),
        model=os.environ.get("LLM_MODEL", "gpt-4o"),
        api_key=os.environ.get("LLM_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL"),
    )


def _load_system_prompt() -> str:
    """Load system prompt from env var or use a sensible default."""
    prompt_path = os.environ.get("AGENT_PROMPT")
    if prompt_path and Path(prompt_path).exists():
        from harness.prompts.loader import load_prompt

        return load_prompt(prompt_path)

    # Check for bundled single-node agent prompt (inside package)
    bundled = Path(__file__).parent / "prompts" / "single_node_agent.yaml"
    if bundled.exists():
        from harness.prompts.loader import load_prompt

        return load_prompt(bundled)

    return (
        "You are a helpful AI assistant. You have access to tools that you can use "
        "to help answer questions. Use tools when appropriate, and provide clear, "
        "concise answers."
    )


def _create_agent():
    """Create the agent with auto mode selection."""
    from harness.agent import AgentLoop
    from harness.tools.examples import register_example_tools
    from harness.tools.registry import ToolRegistry

    llm = _create_llm_client()
    tools = ToolRegistry()
    register_example_tools(tools)
    system_prompt = _load_system_prompt()

    use_burr = os.environ.get("USE_BURR", "1") == "1"
    if use_burr:
        from harness.agent_burr import AgentLoopBurr

        print("[Burr] Using Burr state-machine ReAct engine")
        return AgentLoopBurr(llm=llm, tools=tools, system_prompt=system_prompt)

    print("[Plain] Using simple ReAct loop")
    return AgentLoop(llm=llm, tools=tools, system_prompt=system_prompt)


async def _run_interactive():
    """Run the interactive CLI loop."""
    agent = _create_agent()
    history: list[ChatMessage] = []

    print("\nAgent Harness CLI")
    print("Type 'quit' or 'exit' to stop. Type 'clear' to reset conversation.\n")

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

        try:
            async for event in agent.stream(user_input, history=history):
                _print_event(event)
        except Exception as e:
            print(f"\n[Error: {e}]")
            continue

        print()  # newline after response

        # Add to history for multi-turn
        from harness.models import Role

        history.append(ChatMessage(role=Role.USER, content=user_input))
        # We don't have the full assistant response readily in streaming mode,
        # but for simplicity we skip history tracking of assistant messages in CLI


def _print_event(event: AgentEvent) -> None:
    """Print an agent event to stdout."""
    if event.type == "text_delta":
        print(event.content or "", end="", flush=True)
    elif event.type == "final":
        print(event.content or "", flush=True)
    elif event.type == "tool_call":
        print(
            f"\n  [Calling: {event.tool_name}({event.tool_call.arguments if event.tool_call else ''})]",
            flush=True,
        )
    elif event.type == "tool_result":
        result_preview = (event.tool_result or "")[:200]
        print(f"  [Result: {result_preview}]", flush=True)
        print("\nAgent: ", end="", flush=True)
    elif event.type == "error":
        print(f"\n  [Error: {event.error}]", flush=True)


def main():
    # Parse args first so `--help` prints usage and exits without importing or
    # initializing the LLM client (which would require credentials just to see
    # the help text).
    parser = argparse.ArgumentParser(
        prog="python -m harness",
        description="Interactive agent harness CLI. Configuration is read from "
        "environment variables (LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, "
        "LLM_BASE_URL, AGENT_PROMPT, USE_BURR).",
    )
    parser.parse_args()
    asyncio.run(_run_interactive())


if __name__ == "__main__":
    main()
