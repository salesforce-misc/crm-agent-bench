"""Plain-Python ReAct agent loop.

A dependency-free ReAct loop (no state-machine framework). The Burr-backed
equivalent lives in :mod:`harness.agent_burr`; select between them with
``USE_BURR``.

Flow:
  1. Build messages: [system] + history + [user]
  2. Call LLM with messages + tool definitions
  3. If tool_calls: execute tools, append results, loop
  4. If text: return final response
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from harness.exceptions import HallucinatedToolError
from harness.llm.base import LLMClient
from harness.models import AgentEvent, AgentResponse, ChatMessage, Role
from harness.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentLoop:
    """Simple ReAct agent loop with tool calling."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        system_prompt: str,
        max_iterations: int = 10,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def run(
        self,
        user_message: str,
        history: list[ChatMessage] | None = None,
    ) -> AgentResponse:
        """Run the ReAct loop to completion.

        Args:
            user_message: The user's input message.
            history: Optional prior conversation history.

        Returns:
            AgentResponse with final content and tool call log.
        """
        messages = self._build_initial_messages(user_message, history)
        tool_defs = self.tools.get_definitions() if len(self.tools) > 0 else None
        tool_calls_made: list[dict[str, Any]] = []

        for iteration in range(self.max_iterations):
            response = await self.llm.chat(
                messages=messages,
                tools=tool_defs,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            if not response.tool_calls:
                return AgentResponse(
                    content=response.content or "",
                    tool_calls_made=tool_calls_made,
                    iterations=iteration + 1,
                )

            # Append assistant message with tool calls
            messages.append(
                ChatMessage(
                    role=Role.ASSISTANT,
                    tool_calls=response.tool_calls,
                    content=response.content,
                )
            )

            # Execute each tool call and append results
            for tc in response.tool_calls:
                # Parity with the Burr engine: a call to a tool that isn't wired
                # to this node is a model hallucination, not an infra fault.
                # Surface it as a structured error the analyzer grades as a task
                # failure rather than silently returning an error result.
                if tc.name not in self.tools._tools:
                    raise HallucinatedToolError(
                        f"Tool '{tc.name}' not found in available tools. "
                        f"Available tools: {', '.join(sorted(self.tools.tool_names))}"
                    )
                logger.info("Executing tool: %s(%s)", tc.name, tc.arguments)
                result = await self.tools.execute(tc.name, tc.arguments)
                messages.append(
                    ChatMessage(
                        role=Role.TOOL, content=result, tool_call_id=tc.id, name=tc.name
                    )
                )
                tool_calls_made.append(
                    {"name": tc.name, "arguments": tc.arguments, "result": result}
                )

        # Max iterations reached — return last content
        logger.warning("Max iterations (%d) reached", self.max_iterations)
        return AgentResponse(
            content=response.content
            or "I've reached the maximum number of reasoning steps.",
            tool_calls_made=tool_calls_made,
            iterations=self.max_iterations,
        )

    async def stream(
        self,
        user_message: str,
        history: list[ChatMessage] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Stream the ReAct loop, yielding events for each step.

        Yields AgentEvent objects for:
        - text_delta: Incremental text from the LLM
        - tool_call: A tool call being made
        - tool_result: Result from tool execution
        - final: Final response text
        - error: Any error encountered
        """
        messages = self._build_initial_messages(user_message, history)
        tool_defs = self.tools.get_definitions() if len(self.tools) > 0 else None
        tool_calls_made: list[dict[str, Any]] = []

        for iteration in range(self.max_iterations):
            # Use non-streaming for tool-call iterations (simpler accumulation),
            # streaming only for the final text response
            response = await self.llm.chat(
                messages=messages,
                tools=tool_defs,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            if not response.tool_calls:
                # Final text response — stream it
                yield AgentEvent(type="final", content=response.content or "")
                return

            # Emit tool call events
            messages.append(
                ChatMessage(
                    role=Role.ASSISTANT,
                    tool_calls=response.tool_calls,
                    content=response.content,
                )
            )

            for tc in response.tool_calls:
                # Parity with the Burr engine (see run()): reject hallucinated
                # tools as a model fault instead of executing an unknown name.
                if tc.name not in self.tools._tools:
                    raise HallucinatedToolError(
                        f"Tool '{tc.name}' not found in available tools. "
                        f"Available tools: {', '.join(sorted(self.tools.tool_names))}"
                    )
                yield AgentEvent(type="tool_call", tool_call=tc, tool_name=tc.name)
                result = await self.tools.execute(tc.name, tc.arguments)
                messages.append(
                    ChatMessage(
                        role=Role.TOOL, content=result, tool_call_id=tc.id, name=tc.name
                    )
                )
                yield AgentEvent(
                    type="tool_result", tool_name=tc.name, tool_result=result
                )
                tool_calls_made.append(
                    {"name": tc.name, "arguments": tc.arguments, "result": result}
                )

        yield AgentEvent(
            type="final", content="I've reached the maximum number of reasoning steps."
        )

    def _build_initial_messages(
        self,
        user_message: str,
        history: list[ChatMessage] | None,
    ) -> list[ChatMessage]:
        """Build the initial message list for the LLM."""
        messages: list[ChatMessage] = [
            ChatMessage(role=Role.SYSTEM, content=self.system_prompt)
        ]
        if history:
            messages.extend(history)
        messages.append(ChatMessage(role=Role.USER, content=user_message))
        return messages
