"""Tests for the Burr ReAct engine (AgentLoopBurr) — parity with AgentLoop.

These mirror tests/test_agent.py case-for-case: the Burr state-machine engine
must produce the same AgentResponse / event stream as the plain loop.
"""

import asyncio
from unittest.mock import AsyncMock

from harness.agent_burr import AgentLoopBurr
from harness.models import ChatResponse, ToolCall
from harness.tools.registry import ToolRegistry


def _make_llm_mock(responses: list[ChatResponse]) -> AsyncMock:
    mock = AsyncMock()
    mock.chat = AsyncMock(side_effect=responses)
    return mock


class TestAgentLoopBurr:
    def setup_method(self):
        self.tools = ToolRegistry()

        @self.tools.tool("Get weather")
        def get_weather(city: str) -> str:
            return f"Sunny in {city}, 22C"

    def test_simple_text_response(self):
        llm = _make_llm_mock(
            [
                ChatResponse(content="Hello! How can I help?", finish_reason="stop"),
            ]
        )
        agent = AgentLoopBurr(
            llm=llm, tools=self.tools, system_prompt="You are helpful."
        )
        response = asyncio.run(agent.run("Hi"))
        assert response.content == "Hello! How can I help?"
        assert response.iterations == 1
        assert response.tool_calls_made == []

    def test_tool_call_then_response(self):
        llm = _make_llm_mock(
            [
                ChatResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="tc_1", name="get_weather", arguments={"city": "Paris"}
                        )
                    ],
                    finish_reason="tool_calls",
                ),
                ChatResponse(content="It's sunny in Paris.", finish_reason="stop"),
            ]
        )
        agent = AgentLoopBurr(
            llm=llm, tools=self.tools, system_prompt="You are helpful."
        )
        response = asyncio.run(agent.run("Weather in Paris?"))
        assert response.iterations == 2
        assert len(response.tool_calls_made) == 1
        assert response.tool_calls_made[0]["name"] == "get_weather"
        assert response.tool_calls_made[0]["ok"] is True
        assert response.content == "It's sunny in Paris."

    def test_max_iterations_respected(self):
        """Agent stops after max_iterations even if the LLM keeps calling tools."""
        tool_resp = ChatResponse(
            content=None,
            tool_calls=[
                ToolCall(id="tc_1", name="get_weather", arguments={"city": "London"})
            ],
            finish_reason="tool_calls",
        )
        llm = _make_llm_mock([tool_resp, tool_resp, tool_resp, tool_resp, tool_resp])
        agent = AgentLoopBurr(
            llm=llm,
            tools=self.tools,
            system_prompt="You are helpful.",
            max_iterations=3,
        )
        response = asyncio.run(agent.run("Weather?"))
        assert response.iterations == 3
        # 3 LLM calls; the first two produced executed tool calls, the third hit
        # the budget after planning (so 2 tool executions recorded).
        assert len(response.tool_calls_made) == 2

    async def test_stream_text_response(self):
        llm = _make_llm_mock([ChatResponse(content="Hi there!", finish_reason="stop")])
        agent = AgentLoopBurr(
            llm=llm, tools=self.tools, system_prompt="You are helpful."
        )
        events = [ev async for ev in agent.stream("Hi")]
        assert events[0].type == "final"
        assert events[0].content == "Hi there!"

    async def test_stream_tool_call_events(self):
        llm = _make_llm_mock(
            [
                ChatResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="tc_1", name="get_weather", arguments={"city": "Tokyo"}
                        )
                    ],
                    finish_reason="tool_calls",
                ),
                ChatResponse(content="Sunny in Tokyo.", finish_reason="stop"),
            ]
        )
        agent = AgentLoopBurr(
            llm=llm, tools=self.tools, system_prompt="You are helpful."
        )
        event_types = [ev.type async for ev in agent.stream("Weather in Tokyo?")]
        assert "tool_call" in event_types
        assert "tool_result" in event_types
        assert "final" in event_types
