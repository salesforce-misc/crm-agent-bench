"""The Burr graph agent must surface an LLM hallucinated-tool call as a
structured HallucinatedToolError so the analyzer grades it as a model-fault
task failure (not an infra error excluded from the success denominator).

When the model invokes a tool that is not wired to the current node, the Burr
``execute_tools`` action raises HallucinatedToolError rather than silently
returning an "unknown tool" string (which would let the fault hide).
"""

from __future__ import annotations

from typing import cast

import pytest

from harness.agent import AgentLoop
from harness.agent_burr import AgentLoopBurr
from harness.exceptions import HallucinatedToolError
from harness.llm.base import LLMClient
from harness.models import ChatResponse, ToolCall
from harness.tools.registry import ToolRegistry


class _FakeLLM:
    """Returns a scripted sequence of ChatResponses, one per chat() call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        resp = self._responses[self.calls]
        self.calls += 1
        return resp


def _registry_with_one_tool() -> ToolRegistry:
    tools = ToolRegistry()

    @tools.tool("A real tool that exists")
    def real_tool(x: str) -> str:
        return f"ok:{x}"

    return tools


async def test_unknown_tool_call_becomes_hallucinated_tool_error():
    tools = _registry_with_one_tool()
    llm = _FakeLLM(
        [
            ChatResponse(
                content=None,
                tool_calls=[ToolCall(id="tc_1", name="get_cart_details", arguments={})],
                finish_reason="tool_calls",
            ),
        ]
    )
    agent = AgentLoopBurr(
        llm=cast(LLMClient, llm), tools=tools, system_prompt="You are helpful."
    )

    with pytest.raises(HallucinatedToolError) as exc:
        await agent.run("do the thing")
    assert exc.value.error_kind == "hallucinated_tool"
    assert "get_cart_details" in str(exc.value)
    # The available (real) tool is named in the diagnostic.
    assert "real_tool" in str(exc.value)


async def test_real_tool_call_does_not_raise():
    tools = _registry_with_one_tool()
    llm = _FakeLLM(
        [
            ChatResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="tc_1", name="real_tool", arguments={"x": "hi"})
                ],
                finish_reason="tool_calls",
            ),
            ChatResponse(content="done", tool_calls=None, finish_reason="stop"),
        ]
    )
    agent = AgentLoopBurr(
        llm=cast(LLMClient, llm), tools=tools, system_prompt="You are helpful."
    )

    response = await agent.run("do the thing")
    assert response.content == "done"
    assert response.tool_calls_made[0]["name"] == "real_tool"
    assert response.tool_calls_made[0]["result"] == "ok:hi"


# The plain (non-Burr) engine must enforce the SAME hallucination contract, so
# the choice of engine never changes how a fabricated tool call is graded.


async def test_plain_engine_unknown_tool_call_raises():
    tools = _registry_with_one_tool()
    llm = _FakeLLM(
        [
            ChatResponse(
                content=None,
                tool_calls=[ToolCall(id="tc_1", name="get_cart_details", arguments={})],
                finish_reason="tool_calls",
            ),
        ]
    )
    agent = AgentLoop(
        llm=cast(LLMClient, llm), tools=tools, system_prompt="You are helpful."
    )

    with pytest.raises(HallucinatedToolError) as exc:
        await agent.run("do the thing")
    assert "get_cart_details" in str(exc.value)
    # The available (real) tool is named in the diagnostic.
    assert "real_tool" in str(exc.value)


async def test_plain_engine_real_tool_call_does_not_raise():
    tools = _registry_with_one_tool()
    llm = _FakeLLM(
        [
            ChatResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="tc_1", name="real_tool", arguments={"x": "hi"})
                ],
                finish_reason="tool_calls",
            ),
            ChatResponse(content="done", tool_calls=None, finish_reason="stop"),
        ]
    )
    agent = AgentLoop(
        llm=cast(LLMClient, llm), tools=tools, system_prompt="You are helpful."
    )

    response = await agent.run("do the thing")
    assert response.content == "done"
    assert response.tool_calls_made[0]["name"] == "real_tool"
    assert response.tool_calls_made[0]["result"] == "ok:hi"
