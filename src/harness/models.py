"""Shared Pydantic data models for the agent harness."""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, enum.Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """A tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


class ChatMessage(BaseModel):
    """A single message in the conversation."""

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ToolDefinition(BaseModel):
    """OpenAI function-calling tool definition."""

    type: str = "function"
    function: ToolFunction


class ToolFunction(BaseModel):
    """Function schema within a tool definition."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


class ChatResponse(BaseModel):
    """Non-streaming LLM response."""

    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None
    # Canonical usage keys (all optional; treat a missing key as 0):
    #   prompt_tokens      — input/prompt tokens billed
    #   completion_tokens  — output/completion tokens billed
    #   total_tokens       — provider-reported total (prompt + completion, may include
    #                        cache/reasoning tokens depending on provider); recompute
    #                        as prompt+completion if the provider omits it
    #   cache_read_tokens  — tokens served from a prompt cache (discounted read)
    #   cache_write_tokens — tokens written to a prompt cache (e.g. Anthropic cache creation)
    #   reasoning_tokens   — hidden reasoning/thinking tokens billed as output (o1/o3/gpt-5 etc.)
    # None means the provider returned no usage data at all for this call.
    usage: dict[str, int] | None = None


class StreamChunk(BaseModel):
    """A single chunk from a streaming LLM response."""

    delta_content: str | None = None
    delta_tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None
    # Same canonical usage keys as ChatResponse.usage (see above); populated only on
    # the final chunk of a stream, when the provider includes trailing usage data.
    usage: dict[str, int] | None = None


class AgentEvent(BaseModel):
    """An event emitted during agent execution."""

    type: str  # "text_delta", "tool_call", "tool_result", "final", "error"
    content: str | None = None
    tool_call: ToolCall | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: str | None = None
    error: str | None = None


class AgentResponse(BaseModel):
    """Final agent response after ReAct loop completes."""

    content: str
    tool_calls_made: list[dict[str, Any]] = Field(default_factory=list)
    iterations: int = 0
    topic: str | None = None
    handoffs: list[str] = Field(default_factory=list)
