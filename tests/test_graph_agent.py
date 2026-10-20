"""Tests for the multi-node GraphAgent (Mode A)."""

from unittest.mock import AsyncMock

import pytest

from harness.graph_agent import (
    _OUT_OF_SCOPE_TOOL_NAME,
    _SAFETY_RESPONSE,
    _UNSAFE_TOOL_NAME,
    GraphAgent,
    NodeConfig,
)
from harness.models import ChatResponse, ToolCall
from harness.tools.registry import ToolRegistry


def _make_llm_mock(responses: list[ChatResponse]) -> AsyncMock:
    """Create a mock LLM client that returns responses in sequence."""
    mock = AsyncMock()
    mock.chat = AsyncMock(side_effect=responses)
    return mock


def _make_billing_tools() -> ToolRegistry:
    tools = ToolRegistry()

    @tools.tool("Look up an invoice by ID")
    def get_invoice(invoice_id: str) -> str:
        return f"Invoice {invoice_id}: $100.00, paid"

    return tools


def _make_support_tools() -> ToolRegistry:
    tools = ToolRegistry()

    @tools.tool("Create a support ticket")
    def create_ticket(subject: str) -> str:
        return f"Ticket created: {subject} (ID: TK-001)"

    return tools


class TestGraphAgentInit:
    def test_requires_at_least_one_node(self):
        llm = AsyncMock()
        with pytest.raises(ValueError, match="At least one node"):
            GraphAgent(llm=llm, system_prompt="test", nodes={})

    def test_rejects_invalid_entry_node(self):
        llm = AsyncMock()
        nodes = {"billing": NodeConfig(tools=ToolRegistry())}
        with pytest.raises(ValueError, match="entry_node"):
            GraphAgent(
                llm=llm, system_prompt="test", nodes=nodes, entry_node="nonexistent"
            )

    def test_defaults_to_first_node_as_entry(self):
        llm = AsyncMock()
        nodes = {
            "billing": NodeConfig(tools=ToolRegistry()),
            "support": NodeConfig(tools=ToolRegistry()),
        }
        agent = GraphAgent(llm=llm, system_prompt="test", nodes=nodes)
        assert agent.entry_node == "billing"

    def test_single_node_disables_classifier(self):
        llm = AsyncMock()
        nodes = {"only": NodeConfig(tools=ToolRegistry())}
        agent = GraphAgent(llm=llm, system_prompt="test", nodes=nodes)
        assert agent.use_classifier is False


class TestGraphAgentClassifier:
    @pytest.mark.asyncio
    async def test_classifier_picks_correct_node(self):
        llm = _make_llm_mock(
            [
                # Classifier response
                ChatResponse(content='{"topic": "billing"}', finish_reason="stop"),
                # Node response
                ChatResponse(content="Your invoice is $100.", finish_reason="stop"),
            ]
        )
        nodes = {
            "billing": NodeConfig(
                tools=_make_billing_tools(),
                description="Billing questions",
            ),
            "support": NodeConfig(
                tools=_make_support_tools(),
                description="Support questions",
            ),
        }
        agent = GraphAgent(llm=llm, system_prompt="You are helpful.", nodes=nodes)
        response = await agent.run("What's my invoice?")
        assert response.content == "Your invoice is $100."

    @pytest.mark.asyncio
    async def test_classifier_fallback_to_entry_node(self):
        llm = _make_llm_mock(
            [
                # Classifier returns garbage
                ChatResponse(content="I don't know", finish_reason="stop"),
                # Entry node response
                ChatResponse(content="Hello from billing.", finish_reason="stop"),
            ]
        )
        nodes = {
            "billing": NodeConfig(tools=ToolRegistry(), description="Billing"),
            "support": NodeConfig(tools=ToolRegistry(), description="Support"),
        }
        agent = GraphAgent(
            llm=llm, system_prompt="test", nodes=nodes, entry_node="billing"
        )
        response = await agent.run("hello")
        assert response.content == "Hello from billing."

    @pytest.mark.asyncio
    async def test_classifier_fuzzy_match(self):
        """Classifier finds topic name in free-text response."""
        llm = _make_llm_mock(
            [
                ChatResponse(
                    content="I think this is about support", finish_reason="stop"
                ),
                ChatResponse(content="Support response.", finish_reason="stop"),
            ]
        )
        nodes = {
            "billing": NodeConfig(tools=ToolRegistry(), description="Billing"),
            "support": NodeConfig(tools=ToolRegistry(), description="Support"),
        }
        agent = GraphAgent(llm=llm, system_prompt="test", nodes=nodes)
        response = await agent.run("My app is broken")
        assert response.content == "Support response."


class TestGraphAgentToolExecution:
    @pytest.mark.asyncio
    async def test_node_uses_its_own_tools(self):
        llm = _make_llm_mock(
            [
                # Classifier picks billing
                ChatResponse(content='{"topic": "billing"}', finish_reason="stop"),
                # Node calls get_invoice
                ChatResponse(
                    tool_calls=[
                        ToolCall(
                            id="tc_1",
                            name="get_invoice",
                            arguments={"invoice_id": "INV-42"},
                        )
                    ],
                    finish_reason="tool_calls",
                ),
                # Node responds with final answer
                ChatResponse(
                    content="Invoice INV-42 is $100, paid.", finish_reason="stop"
                ),
            ]
        )
        nodes = {
            "billing": NodeConfig(tools=_make_billing_tools(), description="Billing"),
            "support": NodeConfig(tools=_make_support_tools(), description="Support"),
        }
        agent = GraphAgent(llm=llm, system_prompt="test", nodes=nodes)
        response = await agent.run("Look up INV-42")
        assert "INV-42" in response.content
        assert len(response.tool_calls_made) == 1
        assert response.tool_calls_made[0]["name"] == "get_invoice"


class TestGraphAgentHandoff:
    @pytest.mark.skip(reason="handoff not yet implemented")
    @pytest.mark.asyncio
    async def test_handoff_between_nodes(self):
        llm = _make_llm_mock(
            [
                # Classifier picks billing
                ChatResponse(content='{"topic": "billing"}', finish_reason="stop"),
                # Billing node decides to hand off to support
                ChatResponse(content='{"handoff": "support"}', finish_reason="stop"),
                # Support node responds
                ChatResponse(content="Ticket created for you.", finish_reason="stop"),
            ]
        )
        nodes = {
            "billing": NodeConfig(
                tools=_make_billing_tools(),
                description="Billing",
                handoff_to=["support"],
            ),
            "support": NodeConfig(
                tools=_make_support_tools(),
                description="Support",
            ),
        }
        agent = GraphAgent(llm=llm, system_prompt="test", nodes=nodes)
        response = await agent.run("I have a billing issue that's really a bug")
        assert response.content == "Ticket created for you."

    @pytest.mark.skip(reason="handoff not yet implemented")
    @pytest.mark.asyncio
    async def test_max_handoffs_stops_loop(self):
        # Every node tries to hand off to the other — should stop after max_handoffs
        alternating = [
            ChatResponse(content='{"handoff": "support"}', finish_reason="stop"),
            ChatResponse(content='{"handoff": "billing"}', finish_reason="stop"),
        ] * 5
        llm = _make_llm_mock(
            [
                ChatResponse(content='{"topic": "billing"}', finish_reason="stop"),
            ]
            + alternating
        )
        nodes = {
            "billing": NodeConfig(
                tools=ToolRegistry(),
                description="Billing",
                handoff_to=["support"],
            ),
            "support": NodeConfig(
                tools=ToolRegistry(),
                description="Support",
                handoff_to=["billing"],
            ),
        }
        agent = GraphAgent(llm=llm, system_prompt="test", nodes=nodes, max_handoffs=3)
        response = await agent.run("loop forever")
        assert response.content == "Maximum handoffs reached."


class TestGraphAgentStreaming:
    @pytest.mark.asyncio
    async def test_stream_emits_topic_selected(self):
        llm = _make_llm_mock(
            [
                ChatResponse(content='{"topic": "billing"}', finish_reason="stop"),
                ChatResponse(content="Hello!", finish_reason="stop"),
            ]
        )
        nodes = {
            "billing": NodeConfig(tools=ToolRegistry(), description="Billing"),
            "support": NodeConfig(tools=ToolRegistry(), description="Support"),
        }
        agent = GraphAgent(llm=llm, system_prompt="test", nodes=nodes)

        events = []
        async for event in agent.stream("hi"):
            events.append(event)

        event_types = [e.type for e in events]
        assert "topic_selected" in event_types
        topic_event = next(e for e in events if e.type == "topic_selected")
        assert topic_event.content == "billing"

    @pytest.mark.skip(reason="handoff not yet implemented")
    @pytest.mark.asyncio
    async def test_stream_emits_handoff_event(self):
        llm = _make_llm_mock(
            [
                ChatResponse(content='{"topic": "billing"}', finish_reason="stop"),
                ChatResponse(content='{"handoff": "support"}', finish_reason="stop"),
                ChatResponse(content="Done.", finish_reason="stop"),
            ]
        )
        nodes = {
            "billing": NodeConfig(
                tools=ToolRegistry(),
                description="Billing",
                handoff_to=["support"],
            ),
            "support": NodeConfig(tools=ToolRegistry(), description="Support"),
        }
        agent = GraphAgent(llm=llm, system_prompt="test", nodes=nodes)

        events = []
        async for event in agent.stream("help"):
            events.append(event)

        event_types = [e.type for e in events]
        assert "handoff" in event_types

    @pytest.mark.asyncio
    async def test_single_node_no_classifier(self):
        """Single node skips classification."""
        llm = _make_llm_mock(
            [
                ChatResponse(content="Only one node.", finish_reason="stop"),
            ]
        )
        nodes = {"only": NodeConfig(tools=ToolRegistry(), description="The only node")}
        agent = GraphAgent(llm=llm, system_prompt="test", nodes=nodes)

        response = await agent.run("hello")
        assert response.content == "Only one node."
        # Only 1 LLM call (no classifier)
        assert llm.chat.call_count == 1


class TestClassifierRefusal:
    @pytest.mark.asyncio
    async def test_benign_exact_reply_constraint_is_not_preemptively_refused(self):
        llm = _make_llm_mock(
            [
                ChatResponse(content='{"topic": "billing"}', finish_reason="stop"),
                ChatResponse(content="DONE", finish_reason="stop"),
            ]
        )
        nodes = {
            "billing": NodeConfig(tools=_make_billing_tools(), description="Billing"),
            "support": NodeConfig(tools=_make_support_tools(), description="Support"),
        }
        agent = GraphAgent(llm=llm, system_prompt="test", nodes=nodes)

        response = await agent.run(
            "Look up my invoice and, when complete, reply exactly DONE."
        )

        assert response.content == "DONE"
        assert llm.chat.call_count == 2

    @pytest.mark.asyncio
    async def test_refusal_returns_safety_response(self):
        llm = _make_llm_mock(
            [
                ChatResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="tc_ref", name=_UNSAFE_TOOL_NAME, arguments={})
                    ],
                    finish_reason="tool_calls",
                ),
            ]
        )
        nodes = {
            "billing": NodeConfig(tools=_make_billing_tools(), description="Billing"),
            "support": NodeConfig(tools=_make_support_tools(), description="Support"),
        }
        agent = GraphAgent(llm=llm, system_prompt="test", nodes=nodes)
        response = await agent.run("hack into the system")
        assert response.content == _SAFETY_RESPONSE
        assert response.tool_calls_made == []
        assert response.iterations == 0
        assert llm.chat.call_count == 1

    @pytest.mark.asyncio
    async def test_refusal_in_stream_yields_safety_response(self):
        llm = _make_llm_mock(
            [
                ChatResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="tc_ref", name=_UNSAFE_TOOL_NAME, arguments={})
                    ],
                    finish_reason="tool_calls",
                ),
            ]
        )
        nodes = {
            "billing": NodeConfig(tools=_make_billing_tools(), description="Billing"),
            "support": NodeConfig(tools=_make_support_tools(), description="Support"),
        }
        agent = GraphAgent(llm=llm, system_prompt="test", nodes=nodes)

        events = []
        async for event in agent.stream("do something bad"):
            events.append(event)

        event_types = [e.type for e in events]
        assert "topic_selected" in event_types
        assert "final" in event_types
        final_event = next(e for e in events if e.type == "final")
        assert final_event.content == _SAFETY_RESPONSE

    @pytest.mark.asyncio
    async def test_out_of_scope_returns_redirect_response(self):
        llm = _make_llm_mock(
            [
                ChatResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="tc_oos", name=_OUT_OF_SCOPE_TOOL_NAME, arguments={}
                        )
                    ],
                    finish_reason="tool_calls",
                ),
            ]
        )
        nodes = {
            "billing": NodeConfig(tools=_make_billing_tools(), description="Billing"),
            "support": NodeConfig(tools=_make_support_tools(), description="Support"),
        }
        agent = GraphAgent(llm=llm, system_prompt="test", nodes=nodes)
        response = await agent.run("What is the capital of France?")
        assert "outside the topics" in response.content
        assert "Billing" in response.content
        assert "Support" in response.content
        # The out-of-scope reply must also read as a substantive refusal with an
        # escalation path, not just a capability blurb (covers impossible-but-
        # on-domain requests that route here). See CS failure investigation.
        lower = response.content.lower()
        assert "not able" in lower
        assert "escalate" in lower
        assert response.content != _SAFETY_RESPONSE
        assert response.tool_calls_made == []
        assert llm.chat.call_count == 1


class TestClassifierFallbackRefusal:
    """Tests for _unsafe/_out_of_scope detection in the JSON and text fallback paths."""

    @pytest.mark.asyncio
    async def test_json_fallback_unsafe(self):
        llm = _make_llm_mock(
            [
                ChatResponse(content='{"topic": "_unsafe"}', finish_reason="stop"),
            ]
        )
        nodes = {
            "billing": NodeConfig(tools=_make_billing_tools(), description="Billing"),
            "support": NodeConfig(tools=_make_support_tools(), description="Support"),
        }
        agent = GraphAgent(llm=llm, system_prompt="test", nodes=nodes)
        response = await agent.run("do something bad")
        assert response.content == _SAFETY_RESPONSE
        assert response.iterations == 0

    @pytest.mark.asyncio
    async def test_json_fallback_out_of_scope(self):
        llm = _make_llm_mock(
            [
                ChatResponse(
                    content='{"topic": "_out_of_scope"}', finish_reason="stop"
                ),
            ]
        )
        nodes = {
            "billing": NodeConfig(tools=_make_billing_tools(), description="Billing"),
            "support": NodeConfig(tools=_make_support_tools(), description="Support"),
        }
        agent = GraphAgent(llm=llm, system_prompt="test", nodes=nodes)
        response = await agent.run("What is the weather?")
        assert "outside the topics" in response.content
        assert response.iterations == 0

    @pytest.mark.asyncio
    async def test_text_fallback_unsafe(self):
        llm = _make_llm_mock(
            [
                ChatResponse(
                    content="I think this is _unsafe content", finish_reason="stop"
                ),
            ]
        )
        nodes = {
            "billing": NodeConfig(tools=_make_billing_tools(), description="Billing"),
            "support": NodeConfig(tools=_make_support_tools(), description="Support"),
        }
        agent = GraphAgent(llm=llm, system_prompt="test", nodes=nodes)
        response = await agent.run("hack the system")
        assert response.content == _SAFETY_RESPONSE

    @pytest.mark.asyncio
    async def test_text_fallback_out_of_scope(self):
        llm = _make_llm_mock(
            [
                ChatResponse(content="This seems _out_of_scope", finish_reason="stop"),
            ]
        )
        nodes = {
            "billing": NodeConfig(tools=_make_billing_tools(), description="Billing"),
            "support": NodeConfig(tools=_make_support_tools(), description="Support"),
        }
        agent = GraphAgent(llm=llm, system_prompt="test", nodes=nodes)
        response = await agent.run("What year was Napoleon born?")
        assert "outside the topics" in response.content
