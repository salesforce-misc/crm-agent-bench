"""LiteLLM client supporting 100+ LLM providers.

Usage:
    client = LiteLLMClient(model="gemini/gemini-2.0-flash")
    client = LiteLLMClient(model="anthropic/claude-sonnet-4-20250514")
    client = LiteLLMClient(model="openai/gpt-4o")
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import litellm

from harness.llm.base import LLMClient, get_rate_limiter
from harness.models import (
    ChatMessage,
    ChatResponse,
    StreamChunk,
    ToolCall,
    ToolDefinition,
)

logger = logging.getLogger(__name__)

# Disable LiteLLM's anonymous usage telemetry at import time. This is a
# process-wide module global; we set it explicitly (rather than relying on the
# library default) so no benchmark run phones home. Override by setting the
# LITELLM_TELEMETRY env var if you knowingly want it on.
# litellm's own stubs type this module global as ``Literal[True]``; assigning a
# runtime bool is the documented way to toggle it and is correct at runtime.
litellm.telemetry = os.environ.get("LITELLM_TELEMETRY", "").lower() in (  # ty: ignore[invalid-assignment]
    "1",
    "true",
    "yes",
)


def _extract_usage(usage: Any) -> dict[str, int]:
    """Build the canonical usage dict from a LiteLLM response's ``usage`` object.

    LiteLLM normalizes most providers into an OpenAI-shaped usage object
    (``prompt_tokens_details.cached_tokens``, ``completion_tokens_details.reasoning_tokens``),
    but Anthropic-via-LiteLLM additionally exposes cache-write tokens as a flat
    ``cache_creation_input_tokens`` attribute on the usage object itself. All lookups
    are defensive since these sub-fields only exist for providers that support them.
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


class LiteLLMClient(LLMClient):
    """LiteLLM-backed LLM client."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        # Optional thinking/reasoning control. For Gemini-3-family models, an
        # unconstrained thinking budget can consume the whole response and leave
        # an empty text part (finish_reason=stop, content=None) — fatal for the
        # user simulator, which needs a text turn every time. Passing
        # reasoning_effort="disable" (or "low") forces the model to emit text.
        # Falls back to the LLM_REASONING_EFFORT env var when not passed.
        self._reasoning_effort = (
            reasoning_effort or os.environ.get("LLM_REASONING_EFFORT") or None
        )
        # SECURITY: `litellm.ssl_verify` is a *process-wide* module global, not a
        # per-client setting — once disabled here it stays disabled for every
        # subsequent litellm call in the process (there is no per-request knob in
        # litellm to scope it). Only opt in (SSL_VERIFY=false) for a trusted local
        # proxy with a self-signed cert; never against a public endpoint, as it
        # disables TLS certificate verification globally and exposes all traffic
        # to MITM. We log loudly so this can't happen silently.
        skip_ssl = os.environ.get("SSL_VERIFY", "true").lower() in ("false", "0", "no")
        if skip_ssl:
            logger.warning(
                "SSL_VERIFY=false: disabling TLS certificate verification "
                "process-wide for ALL litellm requests. Use only for a trusted "
                "local proxy."
            )
            litellm.ssl_verify = False
        # Some providers/models reject params the client sends by default (e.g.
        # Bedrock's xAI Grok rejects `temperature`). Opt-in via LITELLM_DROP_PARAMS
        # so litellm silently drops unsupported params instead of erroring. Left
        # off by default so existing runs' params are never dropped without notice.
        if os.environ.get("LITELLM_DROP_PARAMS", "").lower() in ("true", "1", "yes"):
            litellm.drop_params = True
        # Per-request timeout (seconds) and bounded retries on transient
        # connection/timeout errors. LiteLLM's own retry knob rides in via
        # **kwargs (not its typed signature), so we retry explicitly in chat()
        # instead — same pattern as OpenAIClient. The timeout aborts a hung read
        # so a retry can fire rather than wedging until the outer task timeout.
        # LITELLM_NUM_RETRIES=0 disables retries (1 attempt, no retry).
        self._timeout = float(os.environ.get("LITELLM_REQUEST_TIMEOUT", "120"))
        self._num_retries = max(0, int(os.environ.get("LITELLM_NUM_RETRIES", "2")))

    def _is_reasoning_model(self) -> bool:
        """Check if the model is a reasoning model (o1, o3, gpt-5, etc.).

        These models reject `temperature` (only the default 1 is allowed) and
        expect `max_completion_tokens` instead of `max_tokens`.
        """
        m = self.model.lower()
        return any(tag in m for tag in ("o1", "o3", "o4", "gpt-5"))

    def _omits_temperature(self) -> bool:
        """Models that reject an explicit `temperature` but still take max_tokens.

        Reasoning models qualify. Bedrock's xAI Grok also rejects the temperature
        field outright ("This model doesn't support the temperature field"), so we
        must not send it — but unlike reasoning models it still uses `max_tokens`.
        """
        if self._is_reasoning_model():
            return True
        return "grok" in self.model.lower()

    def _build_messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        result = []
        for msg in messages:
            d: dict[str, Any] = {"role": msg.role.value}
            has_content = msg.content and msg.content.strip()
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
                if has_content:
                    d["content"] = msg.content
                # Omit content entirely for tool-call messages with no text
            elif msg.tool_call_id:
                # Tool result messages — content is the result
                d["content"] = msg.content if has_content else "(no output)"
            else:
                # Regular messages — Anthropic rejects empty/whitespace content
                if not has_content:
                    # Skip messages with no real content (e.g. empty assistant turns)
                    continue
                d["content"] = msg.content
            if msg.tool_call_id:
                d["tool_call_id"] = msg.tool_call_id
            if msg.name:
                d["name"] = msg.name
            result.append(d)
        return result

    def _build_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        return [t.model_dump() for t in tools]

    def _needs_allowed_tools(self) -> bool:
        """Some Bedrock models support tool use but litellm's static param map
        does not list `tools` as supported for them (e.g. xAI Grok via the
        converse route), so it raises UnsupportedParamsError. Forcing `tools`
        through with allowed_openai_params lets the (working) request proceed.
        """
        return "grok" in self.model.lower()

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ChatResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._build_messages(messages),
            "timeout": self._timeout,
        }
        # Reasoning models (o1, o3, gpt-5) use max_completion_tokens and reject
        # an explicit temperature. Grok-on-Bedrock also rejects temperature but
        # keeps max_tokens. Everything else gets both temperature and max_tokens.
        if self._is_reasoning_model():
            kwargs["max_completion_tokens"] = max_tokens
        elif self._omits_temperature():
            kwargs["max_tokens"] = max_tokens
        else:
            kwargs["temperature"] = temperature
            kwargs["max_tokens"] = max_tokens
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url
        if tools:
            kwargs["tools"] = self._build_tools(tools)
            if self._needs_allowed_tools():
                kwargs["allowed_openai_params"] = ["tools"]
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort

        # Bounded retries on transient connection/timeout/5xx errors. A single
        # gateway blip should not fail the whole task; a persistent error still
        # surfaces after the last attempt. self._num_retries retries => up to
        # self._num_retries + 1 total attempts, with exponential backoff.
        response = None
        for attempt in range(self._num_retries + 1):
            limiter = get_rate_limiter()
            if limiter:
                await limiter.wait()
            try:
                response = await litellm.acompletion(**kwargs)
                break
            except (
                litellm.APIConnectionError,
                litellm.Timeout,
                litellm.ServiceUnavailableError,
                litellm.InternalServerError,
            ) as e:
                # Final attempt re-raises the original transient error verbatim.
                if attempt == self._num_retries:
                    raise
                wait_time = 2**attempt * 5  # 5s, 10s, 20s, ...
                logger.warning(
                    "LiteLLM transient error (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1,
                    self._num_retries + 1,
                    e,
                    wait_time,
                )
                await asyncio.sleep(wait_time)
        if response is None:  # pragma: no cover - loop always sets or raises
            raise RuntimeError("litellm.acompletion returned no response")

        choice = response.choices[0]
        msg = choice.message

        parsed_tool_calls = None
        if msg.tool_calls:
            parsed_tool_calls = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": tc.function.arguments}
                parsed_tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                )

        return ChatResponse(
            content=msg.content,
            tool_calls=parsed_tool_calls,
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
            "messages": self._build_messages(messages),
            "stream": True,
            "timeout": self._timeout,
        }
        if self._is_reasoning_model():
            kwargs["max_completion_tokens"] = max_tokens
        elif self._omits_temperature():
            kwargs["max_tokens"] = max_tokens
        else:
            kwargs["temperature"] = temperature
            kwargs["max_tokens"] = max_tokens
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url
        if tools:
            kwargs["tools"] = self._build_tools(tools)
            if self._needs_allowed_tools():
                kwargs["allowed_openai_params"] = ["tools"]
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort

        limiter = get_rate_limiter()
        if limiter:
            await limiter.wait()

        response = await litellm.acompletion(**kwargs)
        tool_call_buffers: dict[int, dict[str, Any]] = {}

        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            if hasattr(delta, "content") and delta.content:
                yield StreamChunk(
                    delta_content=delta.content, finish_reason=finish_reason
                )

            if hasattr(delta, "tool_calls") and delta.tool_calls:
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
