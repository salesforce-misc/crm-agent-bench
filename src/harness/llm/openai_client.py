"""OpenAI-compatible LLM client.

Works with OpenAI, Azure OpenAI, and any OpenAI-compatible endpoint (vLLM, Ollama, etc.).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx
import openai

from harness.llm.base import LLMClient, get_rate_limiter
from harness.models import (
    ChatMessage,
    ChatResponse,
    StreamChunk,
    ToolCall,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


def _messages_to_dicts(
    messages: list[ChatMessage], bedrock_compat: bool = False
) -> list[dict[str, Any]]:
    """Convert ChatMessage list to OpenAI API format.

    When bedrock_compat=True, merges system messages into the first user
    message since Bedrock requires conversations to start with a user message.
    """

    result = []
    system_prefix = ""

    for msg in messages:
        if bedrock_compat and msg.role.value == "system":
            # Accumulate system content to prepend to first user message
            system_prefix += (msg.content or "") + "\n\n"
            continue

        d: dict[str, Any] = {"role": msg.role.value, "content": msg.content or ""}

        # Prepend accumulated system content to the first user message
        if bedrock_compat and system_prefix and msg.role.value == "user":
            d["content"] = system_prefix + (msg.content or "")
            system_prefix = ""

        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]
            if not msg.content:
                d.pop("content", None)
        if msg.tool_call_id:
            d["tool_call_id"] = msg.tool_call_id
        if msg.name:
            d["name"] = msg.name
        result.append(d)
    return result


def _tools_to_dicts(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Convert ToolDefinition list to OpenAI API format."""
    return [t.model_dump() for t in tools]


def _extract_usage(usage: Any) -> dict[str, int]:
    """Build the canonical usage dict from an OpenAI SDK ``CompletionUsage`` object.

    Cache-read and reasoning tokens live on nested ``*_details`` sub-objects that
    are only present for models that support them, so every lookup is defensive —
    a missing attribute or ``None`` value collapses to 0 rather than raising.
    """
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": getattr(usage, "total_tokens", None)
        or (usage.prompt_tokens + usage.completion_tokens),
        "cache_read_tokens": getattr(prompt_details, "cached_tokens", 0) or 0,
        "cache_write_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "reasoning_tokens": getattr(completion_details, "reasoning_tokens", 0) or 0,
    }


def _parse_tool_calls(raw_tool_calls: list[Any]) -> list[ToolCall]:
    """Parse OpenAI tool call objects into ToolCall models."""
    result = []
    for tc in raw_tool_calls:
        args_str = (
            tc.function.arguments
            if hasattr(tc, "function")
            else tc["function"]["arguments"]
        )
        try:
            arguments = json.loads(args_str)
        except (json.JSONDecodeError, TypeError):
            arguments = {"raw": args_str}
        result.append(
            ToolCall(
                id=tc.id if hasattr(tc, "id") else tc["id"],
                name=tc.function.name
                if hasattr(tc, "function")
                else tc["function"]["name"],
                arguments=arguments,
            )
        )
    return result


class OpenAIClient(LLMClient):
    """OpenAI-compatible LLM client."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        bedrock_compat: bool = False,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model = model
        self.bedrock_compat = bedrock_compat
        # ``disable`` is LiteLLM's provider-neutral spelling and is not a valid
        # OpenAI Chat Completions value. Preserve the historical behavior for
        # that setting by omitting the parameter on this backend.
        self._reasoning_effort = (
            None if reasoning_effort == "disable" else reasoning_effort
        )
        skip_ssl = os.environ.get("SSL_VERIFY", "true").lower() in ("false", "0", "no")
        # Per-request timeout (seconds). Without one, a hung socket read blocks
        # forever and never surfaces as APITimeoutError, so the retry loop in
        # chat() never fires — a single transient gateway blip then wedges the
        # whole task until the outer TASK_TIMEOUT trips. Default 120s; override
        # with OPENAI_REQUEST_TIMEOUT.
        timeout = float(os.environ.get("OPENAI_REQUEST_TIMEOUT", "120"))
        http_client = httpx.AsyncClient(verify=not skip_ssl) if skip_ssl else None
        self._client = openai.AsyncOpenAI(
            api_key=api_key, base_url=base_url, http_client=http_client, timeout=timeout
        )

    def _is_reasoning_model(self) -> bool:
        """Check if the model is a reasoning model (o1, o3, GPT-5/6, etc.)."""
        m = self.model.lower()
        return any(tag in m for tag in ("o1", "o3", "o4", "gpt-5", "gpt-6"))

    async def aclose(self) -> None:
        """Close the underlying AsyncOpenAI client and its httpx connection pool."""
        await self._client.close()

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": _messages_to_dicts(
                messages, bedrock_compat=self.bedrock_compat
            ),
        }
        # Reasoning models (o1, o3, GPT-5/6) use max_completion_tokens and reject
        # an explicit temperature.
        if self._is_reasoning_model():
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["temperature"] = temperature
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = _tools_to_dicts(tools)
        if response_format:
            kwargs["response_format"] = response_format
        if self._is_reasoning_model() and self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort

        logger.debug(
            "OpenAI request [model=%s]: %d messages, bedrock_compat=%s, temperature=%s",
            self.model,
            len(kwargs["messages"]),
            self.bedrock_compat,
            temperature,
        )
        for i, m in enumerate(kwargs["messages"]):
            logger.debug(
                "  msg[%d] role=%s content_len=%d",
                i,
                m["role"],
                len(m.get("content", "") or ""),
            )

        last_err = None
        for attempt in range(3):
            limiter = get_rate_limiter()
            if limiter:
                await limiter.wait()

            try:
                response = await self._client.chat.completions.create(**kwargs)
                break
            except (openai.APIConnectionError, openai.APITimeoutError) as e:
                last_err = e
                wait_time = 2**attempt * 5  # 5s, 10s, 20s
                logger.warning(
                    "Connection error (attempt %d/3): %s — retrying in %ds",
                    attempt + 1,
                    e,
                    wait_time,
                )
                await asyncio.sleep(wait_time)
        else:
            raise last_err  # type: ignore[misc]

        choice = response.choices[0]
        msg = choice.message

        logger.debug(
            "OpenAI response [model=%s]: content=%r, finish_reason=%s, tool_calls=%s, refusal=%s",
            self.model,
            msg.content,
            choice.finish_reason,
            bool(msg.tool_calls),
            getattr(msg, "refusal", None),
        )

        return ChatResponse(
            content=msg.content,
            tool_calls=_parse_tool_calls(msg.tool_calls) if msg.tool_calls else None,
            finish_reason=choice.finish_reason,
            usage=_extract_usage(response.usage) if response.usage else None,
        )

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": _messages_to_dicts(
                messages, bedrock_compat=self.bedrock_compat
            ),
            "stream": True,
        }
        if self._is_reasoning_model():
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["temperature"] = temperature
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = _tools_to_dicts(tools)
        if self._is_reasoning_model() and self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort

        logger.debug(
            "OpenAI stream request [model=%s]: %d messages, bedrock_compat=%s, %d tools",
            self.model,
            len(kwargs["messages"]),
            self.bedrock_compat,
            len(kwargs.get("tools", [])),
        )
        for i, m in enumerate(kwargs["messages"]):
            logger.debug(
                "  msg[%d] role=%s content_len=%d",
                i,
                m["role"],
                len(m.get("content", "") or ""),
            )

        limiter = get_rate_limiter()
        if limiter:
            await limiter.wait()

        stream = await self._client.chat.completions.create(**kwargs)

        # Buffer tool calls across chunks (OpenAI sends them incrementally)
        tool_call_buffers: dict[int, dict[str, Any]] = {}

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            # Handle content deltas
            if delta.content:
                yield StreamChunk(
                    delta_content=delta.content, finish_reason=finish_reason
                )

            # Handle tool call deltas
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_buffers:
                        tool_call_buffers[idx] = {"id": "", "name": "", "arguments": ""}
                    buf = tool_call_buffers[idx]
                    if tc_delta.id:
                        buf["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            buf["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            buf["arguments"] += tc_delta.function.arguments

            # On finish, emit buffered tool calls
            if finish_reason:
                tool_calls = None
                if tool_call_buffers:
                    tool_calls = []
                    for buf in tool_call_buffers.values():
                        try:
                            args = json.loads(buf["arguments"])
                        except (json.JSONDecodeError, TypeError):
                            args = {"raw": buf["arguments"]}
                        tool_calls.append(
                            ToolCall(id=buf["id"], name=buf["name"], arguments=args)
                        )
                yield StreamChunk(
                    finish_reason=finish_reason, delta_tool_calls=tool_calls
                )
