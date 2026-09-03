"""Tests for the user simulator and bench runner."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from harness.models import ChatResponse
from harness.user_simulator import (
    STOP_SIGNAL,
    HumanUserSimulator,
    LLMUserSimulator,
    ReactUserSimulator,
    ReflectionUserSimulator,
    VerifyUserSimulator,
    load_user_simulator,
)


def _make_llm_mock(responses: list[str]) -> AsyncMock:
    """Create a mock LLM client that returns string responses in sequence."""
    mock = AsyncMock()
    mock.chat = AsyncMock(side_effect=[ChatResponse(content=r) for r in responses])
    return mock


class TestLLMUserSimulator:
    def test_basic_conversation(self):
        """Simulator generates user messages from LLM responses."""
        llm = _make_llm_mock(
            [
                "Hi, I need help with my invoice.",
                "My invoice ID is INV-1234.",
                STOP_SIGNAL,
            ]
        )
        sim = LLMUserSimulator(llm=llm)
        sim.reset("You are a customer with invoice INV-1234.")

        msg1 = asyncio.run(sim.step(""))
        assert "invoice" in msg1.lower()

        msg2 = asyncio.run(sim.step("Sure, what's the invoice ID?"))
        assert "INV-1234" in msg2

        msg3 = asyncio.run(sim.step("Your invoice is paid."))
        assert STOP_SIGNAL in msg3

    def test_reset_clears_history(self):
        """Reset starts fresh conversation history."""
        llm = _make_llm_mock(
            [
                "First task message.",
                "Second task message.",
            ]
        )
        sim = LLMUserSimulator(llm=llm)

        sim.reset("Task 1 instruction")
        asyncio.run(sim.step(""))
        assert (
            len(sim._messages) == 3
        )  # system + kickoff prompt + assistant (persona) response

        sim.reset("Task 2 instruction")
        assert len(sim._messages) == 1  # just system prompt

    def test_history_accumulates(self):
        """Messages accumulate in history across turns."""
        llm = _make_llm_mock(
            [
                "Hello",
                "I need a refund",
            ]
        )
        sim = LLMUserSimulator(llm=llm)
        sim.reset("You are a customer.")

        asyncio.run(sim.step(""))
        asyncio.run(sim.step("How can I help?"))

        # system + kickoff(user) + assistant("Hello") + user("How can I help?") + assistant("I need a refund")
        assert len(sim._messages) == 5


class TestReactUserSimulator:
    def test_parses_user_response(self):
        """React simulator extracts User Response from chain-of-thought."""
        llm = _make_llm_mock(
            [
                "Thought: I should ask about the invoice.\nUser Response: Can you look up invoice INV-5521?",
            ]
        )
        sim = ReactUserSimulator(llm=llm)
        sim.reset("You have invoice INV-5521.")

        msg = asyncio.run(sim.step(""))
        assert msg == "Can you look up invoice INV-5521?"

    def test_parses_stop_signal(self):
        """React simulator correctly extracts stop signal."""
        llm = _make_llm_mock(
            [
                f"Thought: The refund is done.\nUser Response: {STOP_SIGNAL}",
            ]
        )
        sim = ReactUserSimulator(llm=llm)
        sim.reset("Get a refund.")

        msg = asyncio.run(sim.step("Refund processed."))
        assert STOP_SIGNAL in msg

    def test_fallback_on_bad_format(self):
        """Falls back to last line if format is unexpected."""
        llm = _make_llm_mock(
            [
                "I just want to check my balance.",
            ]
        )
        sim = ReactUserSimulator(llm=llm)
        sim.reset("Check balance.")

        msg = asyncio.run(sim.step(""))
        assert msg == "I just want to check my balance."

    def test_full_thought_stored_in_history(self):
        """The full chain-of-thought is stored in history for context."""
        raw = "Thought: Reasoning here.\nUser Response: The actual message."
        llm = _make_llm_mock([raw])
        sim = ReactUserSimulator(llm=llm)
        sim.reset("Task instruction.")

        asyncio.run(sim.step(""))
        # History should contain the full output (with thought)
        last_user_msg = sim._messages[-1]
        assert "Thought:" in last_user_msg.content


class TestVerifyUserSimulator:
    def test_passes_on_valid_response(self):
        """Verify strategy accepts a valid response on first try."""
        llm = _make_llm_mock(
            [
                # Generate response (react format)
                "Thought: I should ask about my invoice.\nUser Response: Can you look up invoice INV-5521?",
                # Verify call — PASS
                "PASS: The response is faithful to the instruction.",
            ]
        )
        sim = VerifyUserSimulator(llm=llm)
        sim.reset("You have invoice INV-5521.")

        msg = asyncio.run(sim.step(""))
        assert msg == "Can you look up invoice INV-5521?"

    def test_retries_on_verification_failure(self):
        """Verify strategy retries when verification fails."""
        llm = _make_llm_mock(
            [
                # First generate — premature stop
                f"Thought: Done.\nUser Response: {STOP_SIGNAL}",
                # Verify — FAIL
                "FAIL: Exit conditions not met, user hasn't received invoice info yet.",
                # Second generate — correct
                "Thought: I need to ask.\nUser Response: What is the charge on my invoice?",
                # Verify — PASS
                "PASS: The response is faithful.",
            ]
        )
        sim = VerifyUserSimulator(llm=llm)
        sim.reset("Ask about invoice INV-5521. Only stop after getting invoice info.")

        msg = asyncio.run(sim.step(""))
        assert msg == "What is the charge on my invoice?"
        assert STOP_SIGNAL not in msg

    def test_uses_last_response_after_max_retries(self):
        """Falls back to last response if all retries fail."""
        llm = _make_llm_mock(
            [
                # Attempt 1
                "Thought: x\nUser Response: bad response 1",
                "FAIL: not faithful",
                # Attempt 2
                "Thought: x\nUser Response: bad response 2",
                "FAIL: still not faithful",
                # Attempt 3
                "Thought: x\nUser Response: bad response 3",
                "FAIL: nope",
            ]
        )
        sim = VerifyUserSimulator(llm=llm, max_retries=3)
        sim.reset("Some instruction.")

        msg = asyncio.run(sim.step(""))
        assert msg == "bad response 3"


class TestReflectionUserSimulator:
    def test_reflects_and_corrects(self):
        """Reflection strategy corrects response after verification failure."""
        llm = _make_llm_mock(
            [
                # Initial generate — premature stop
                f"Thought: Done.\nUser Response: {STOP_SIGNAL}",
                # Verify — FAIL
                "FAIL: User hasn't received the refund confirmation yet.",
                # Reflect — corrected response
                "I still need confirmation that my refund was processed. Can you confirm?",
                # Verify corrected — PASS
                "PASS: The response is faithful.",
            ]
        )
        sim = ReflectionUserSimulator(llm=llm)
        sim.reset("Get refund confirmation before ending.")

        msg = asyncio.run(sim.step("We'll look into it."))
        assert "refund" in msg.lower() or "confirm" in msg.lower()
        assert STOP_SIGNAL not in msg

    def test_passes_on_first_try(self):
        """Reflection strategy skips reflection if first response is valid."""
        llm = _make_llm_mock(
            [
                # Generate
                "Thought: Ask about refund.\nUser Response: Where is my refund?",
                # Verify — PASS
                "PASS: Faithful to instruction.",
            ]
        )
        sim = ReflectionUserSimulator(llm=llm)
        sim.reset("Ask about your refund.")

        msg = asyncio.run(sim.step("Hello!"))
        assert msg == "Where is my refund?"


class TestLoadUserSimulator:
    def test_load_human(self):
        sim = load_user_simulator("human")
        assert isinstance(sim, HumanUserSimulator)

    def test_load_llm(self):
        llm = AsyncMock()
        sim = load_user_simulator("llm", llm=llm)
        assert isinstance(sim, LLMUserSimulator)

    def test_load_react(self):
        llm = AsyncMock()
        sim = load_user_simulator("react", llm=llm)
        assert isinstance(sim, ReactUserSimulator)

    def test_llm_required_for_llm_strategy(self):
        with pytest.raises(ValueError, match="LLM client required"):
            load_user_simulator("llm")

    def test_load_verify(self):
        llm = AsyncMock()
        sim = load_user_simulator("verify", llm=llm)
        assert isinstance(sim, VerifyUserSimulator)

    def test_load_reflection(self):
        llm = AsyncMock()
        sim = load_user_simulator("reflection", llm=llm)
        assert isinstance(sim, ReflectionUserSimulator)

    def test_unknown_strategy_raises(self):
        llm = AsyncMock()
        with pytest.raises(ValueError, match="Unknown"):
            load_user_simulator("unknown", llm=llm)


class TestLLMUserSimulatorEmptyContent:
    """Tests for handling empty/whitespace agent messages (Anthropic compatibility)."""

    def test_empty_string_agent_message_skipped(self):
        """Empty agent message should not be added as assistant message."""
        llm = _make_llm_mock(["Hello", "Follow up"])
        sim = LLMUserSimulator(llm=llm)
        sim.reset("You are a customer.")

        asyncio.run(sim.step(""))  # first turn — kickoff
        asyncio.run(sim.step(""))  # empty agent response — should not add assistant msg

        # The agent's turn is the "user" role; an empty/blank agent message adds
        # no turn, so the only user message is the kickoff prompt.
        user_msgs = [m for m in sim._messages if m.role.value == "user"]
        assert len(user_msgs) == 1

    def test_whitespace_agent_message_skipped(self):
        """Whitespace-only agent message should not be added as assistant message."""
        llm = _make_llm_mock(["Hello", "Follow up"])
        sim = LLMUserSimulator(llm=llm)
        sim.reset("You are a customer.")

        asyncio.run(sim.step(""))
        asyncio.run(sim.step("   "))  # whitespace — should be skipped

        # The agent's turn is the "user" role; an empty/blank agent message adds
        # no turn, so the only user message is the kickoff prompt.
        user_msgs = [m for m in sim._messages if m.role.value == "user"]
        assert len(user_msgs) == 1

    def test_none_agent_message_skipped(self):
        """None-like empty agent message should not crash."""
        llm = _make_llm_mock(["Hello"])
        sim = LLMUserSimulator(llm=llm)
        sim.reset("You are a customer.")

        msg = asyncio.run(sim.step(""))
        assert msg == "Hello"

    def test_valid_agent_message_added(self):
        """Non-empty agent message should be added to history."""
        llm = _make_llm_mock(["Hello", "I need a refund"])
        sim = LLMUserSimulator(llm=llm)
        sim.reset("You are a customer.")

        asyncio.run(sim.step(""))
        asyncio.run(sim.step("How can I help you today?"))

        # The agent's message is recorded as a "user" turn (kickoff + this one).
        user_msgs = [m for m in sim._messages if m.role.value == "user"]
        assert len(user_msgs) == 2
        assert user_msgs[1].content == "How can I help you today?"


class TestBuildMessagesAnthropicCompat:
    """Tests for LiteLLM _build_messages handling of empty content (Anthropic compat)."""

    def test_skips_empty_content_messages(self):
        """Messages with empty content should be skipped."""
        from harness.llm.litellm_client import LiteLLMClient
        from harness.models import ChatMessage, Role

        client = LiteLLMClient(model="test")
        messages = [
            ChatMessage(role=Role.SYSTEM, content="You are helpful."),
            ChatMessage(role=Role.USER, content="Hi"),
            ChatMessage(role=Role.ASSISTANT, content=""),  # empty — should be skipped
            ChatMessage(role=Role.USER, content="Hello again"),
        ]
        result = client._build_messages(messages)
        assert len(result) == 3
        roles = [m["role"] for m in result]
        assert roles == ["system", "user", "user"]

    def test_skips_whitespace_only_messages(self):
        """Messages with whitespace-only content should be skipped."""
        from harness.llm.litellm_client import LiteLLMClient
        from harness.models import ChatMessage, Role

        client = LiteLLMClient(model="test")
        messages = [
            ChatMessage(role=Role.SYSTEM, content="You are helpful."),
            ChatMessage(role=Role.USER, content="Hi"),
            ChatMessage(role=Role.ASSISTANT, content="   "),  # whitespace — skip
        ]
        result = client._build_messages(messages)
        assert len(result) == 2

    def test_skips_none_content_messages(self):
        """Messages with None content should be skipped."""
        from harness.llm.litellm_client import LiteLLMClient
        from harness.models import ChatMessage, Role

        client = LiteLLMClient(model="test")
        messages = [
            ChatMessage(role=Role.SYSTEM, content="You are helpful."),
            ChatMessage(role=Role.USER, content="Hi"),
            ChatMessage(role=Role.ASSISTANT, content=None),  # None — skip
        ]
        result = client._build_messages(messages)
        assert len(result) == 2

    def test_tool_call_message_omits_empty_content(self):
        """Tool-call messages with no content should omit content field entirely."""
        from harness.llm.litellm_client import LiteLLMClient
        from harness.models import ChatMessage, Role, ToolCall

        client = LiteLLMClient(model="test")
        messages = [
            ChatMessage(role=Role.USER, content="What's the weather?"),
            ChatMessage(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[
                    ToolCall(id="tc_1", name="get_weather", arguments={"city": "Paris"})
                ],
            ),
        ]
        result = client._build_messages(messages)
        assert len(result) == 2
        assert "content" not in result[1]
        assert "tool_calls" in result[1]

    def test_tool_call_message_keeps_content_when_present(self):
        """Tool-call messages with content should keep it."""
        from harness.llm.litellm_client import LiteLLMClient
        from harness.models import ChatMessage, Role, ToolCall

        client = LiteLLMClient(model="test")
        messages = [
            ChatMessage(role=Role.USER, content="What's the weather?"),
            ChatMessage(
                role=Role.ASSISTANT,
                content="Let me check that for you.",
                tool_calls=[
                    ToolCall(id="tc_1", name="get_weather", arguments={"city": "Paris"})
                ],
            ),
        ]
        result = client._build_messages(messages)
        assert result[1]["content"] == "Let me check that for you."
        assert "tool_calls" in result[1]

    def test_tool_result_with_empty_content_gets_placeholder(self):
        """Tool result messages with empty content get a placeholder."""
        from harness.llm.litellm_client import LiteLLMClient
        from harness.models import ChatMessage, Role

        client = LiteLLMClient(model="test")
        messages = [
            ChatMessage(
                role=Role.TOOL, content="", tool_call_id="tc_1", name="get_weather"
            ),
        ]
        result = client._build_messages(messages)
        assert len(result) == 1
        assert result[0]["content"] == "(no output)"
        assert result[0]["tool_call_id"] == "tc_1"


class TestBenchRunner:
    def test_load_tasks(self):
        """Load tasks from the example YAML file."""
        from harness.bench_runner import load_tasks

        tasks = load_tasks("src/tasks/eval_customer_service_graph.yaml")
        assert len(tasks) == 13
        assert tasks[0].id == "billing_refund_01"
        assert "Alex Chen" in tasks[0].instruction
        assert "get_invoice" in tasks[0].expected_actions
