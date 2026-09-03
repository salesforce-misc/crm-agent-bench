"""Tests for the AgentLoop (Mode A)."""

from unittest.mock import AsyncMock

import pytest

from harness.agent import AgentLoop
from harness.models import ChatResponse, ToolCall
from harness.tools.registry import ToolRegistry


def _make_llm_mock(responses: list[ChatResponse]) -> AsyncMock:
    """Create a mock LLM client that returns responses in sequence."""
    mock = AsyncMock()
    mock.chat = AsyncMock(side_effect=responses)
    return mock


class TestAgentLoop:
    def setup_method(self):
        self.tools = ToolRegistry()

        @self.tools.tool("Get weather")
        def get_weather(city: str) -> str:
            return f"Sunny in {city}, 22C"

    def test_simple_text_response(self):
        """Agent returns text without tool calls."""
        llm = _make_llm_mock(
            [
                ChatResponse(content="Hello! How can I help?", finish_reason="stop"),
            ]
        )
        agent = AgentLoop(llm=llm, tools=self.tools, system_prompt="You are helpful.")

        import asyncio

        response = asyncio.run(agent.run("Hi"))
        assert response.content == "Hello! How can I help?"
        assert response.iterations == 1
        assert response.tool_calls_made == []

    def test_tool_call_then_response(self):
        """Agent calls a tool, then responds with text."""
        llm = _make_llm_mock(
            [
                # First call: LLM decides to call get_weather
                ChatResponse(
                    tool_calls=[
                        ToolCall(
                            id="tc_1", name="get_weather", arguments={"city": "Paris"}
                        )
                    ],
                    finish_reason="tool_calls",
                ),
                # Second call: LLM responds with final text
                ChatResponse(content="It's sunny in Paris, 22C!", finish_reason="stop"),
            ]
        )
        agent = AgentLoop(llm=llm, tools=self.tools, system_prompt="You are helpful.")

        import asyncio

        response = asyncio.run(agent.run("What's the weather in Paris?"))
        assert "sunny" in response.content.lower() or "Paris" in response.content
        assert response.iterations == 2
        assert len(response.tool_calls_made) == 1
        assert response.tool_calls_made[0]["name"] == "get_weather"

    def test_max_iterations_respected(self):
        """Agent stops after max_iterations even if LLM keeps calling tools."""
        tool_response = ChatResponse(
            tool_calls=[
                ToolCall(id="tc_1", name="get_weather", arguments={"city": "London"})
            ],
            finish_reason="tool_calls",
        )
        # Return tool calls forever
        llm = _make_llm_mock([tool_response] * 5)
        agent = AgentLoop(
            llm=llm,
            tools=self.tools,
            system_prompt="You are helpful.",
            max_iterations=3,
        )

        import asyncio

        response = asyncio.run(agent.run("Loop forever"))
        assert response.iterations == 3
        assert len(response.tool_calls_made) == 3

    @pytest.mark.asyncio
    async def test_stream_text_response(self):
        """Streaming returns final event for text response."""
        llm = _make_llm_mock(
            [
                ChatResponse(content="Hello!", finish_reason="stop"),
            ]
        )
        agent = AgentLoop(llm=llm, tools=self.tools, system_prompt="You are helpful.")

        events = []
        async for event in agent.stream("Hi"):
            events.append(event)

        assert len(events) == 1
        assert events[0].type == "final"
        assert events[0].content == "Hello!"

    @pytest.mark.asyncio
    async def test_stream_tool_call_events(self):
        """Streaming emits tool_call and tool_result events."""
        llm = _make_llm_mock(
            [
                ChatResponse(
                    tool_calls=[
                        ToolCall(
                            id="tc_1", name="get_weather", arguments={"city": "Tokyo"}
                        )
                    ],
                    finish_reason="tool_calls",
                ),
                ChatResponse(content="It's sunny in Tokyo!", finish_reason="stop"),
            ]
        )
        agent = AgentLoop(llm=llm, tools=self.tools, system_prompt="You are helpful.")

        events = []
        async for event in agent.stream("Weather in Tokyo?"):
            events.append(event)

        event_types = [e.type for e in events]
        assert "tool_call" in event_types
        assert "tool_result" in event_types
        assert "final" in event_types

    def test_no_tools_registered(self):
        """Agent works with empty tool registry."""
        llm = _make_llm_mock(
            [
                ChatResponse(content="I can't use tools.", finish_reason="stop"),
            ]
        )
        empty_tools = ToolRegistry()
        agent = AgentLoop(llm=llm, tools=empty_tools, system_prompt="You are helpful.")

        import asyncio

        response = asyncio.run(agent.run("Do something"))
        assert response.content == "I can't use tools."
