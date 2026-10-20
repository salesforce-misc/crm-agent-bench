"""Tests for OpenAIClient — request shaping, tool parsing, retries, lifecycle."""

from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from harness.llm.openai_client import OpenAIClient
from harness.models import ChatMessage, Role, ToolDefinition


def _mock_response(content="Hello", tool_calls=None, finish_reason="stop"):
    """Build a minimal OpenAI-SDK-style chat completion response."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.refusal = None
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15
    usage.prompt_tokens_details = None
    usage.completion_tokens_details = None
    usage.cache_creation_input_tokens = 0
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _raw_tool_call(id_="tc_1", name="get_invoice", arguments='{"account": "A1"}'):
    tc = MagicMock()
    tc.id = id_
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _make_client_with_create(create_mock):
    """Instantiate an OpenAIClient whose SDK create() is `create_mock`."""
    with patch("harness.llm.openai_client.openai.AsyncOpenAI") as sdk:
        instance = MagicMock()
        instance.chat.completions.create = create_mock
        instance.close = AsyncMock()
        sdk.return_value = instance
        client = OpenAIClient(model="gpt-4o")
    return client, instance


@pytest.mark.asyncio
async def test_chat_returns_content():
    create = AsyncMock(return_value=_mock_response())
    client, _ = _make_client_with_create(create)
    resp = await client.chat([ChatMessage(role=Role.USER, content="hi")])
    assert resp.content == "Hello"
    assert resp.finish_reason == "stop"
    assert resp.usage["prompt_tokens"] == 10


@pytest.mark.asyncio
async def test_reasoning_model_uses_max_completion_tokens():
    create = AsyncMock(return_value=_mock_response())
    with patch("harness.llm.openai_client.openai.AsyncOpenAI") as sdk:
        instance = MagicMock()
        instance.chat.completions.create = create
        sdk.return_value = instance
        client = OpenAIClient(model="gpt-5")
    await client.chat([ChatMessage(role=Role.USER, content="hi")], max_tokens=256)
    kwargs = create.call_args[1]
    assert kwargs["max_completion_tokens"] == 256
    assert "temperature" not in kwargs
    assert "max_tokens" not in kwargs


@pytest.mark.asyncio
async def test_gpt_6_reasoning_model_omits_temperature():
    create = AsyncMock(return_value=_mock_response())
    with patch("harness.llm.openai_client.openai.AsyncOpenAI") as sdk:
        instance = MagicMock()
        instance.chat.completions.create = create
        sdk.return_value = instance
        client = OpenAIClient(model="gpt-6-astra")
    await client.chat([ChatMessage(role=Role.USER, content="hi")], max_tokens=256)
    kwargs = create.call_args.kwargs
    assert kwargs["max_completion_tokens"] == 256
    assert "temperature" not in kwargs
    assert "max_tokens" not in kwargs


@pytest.mark.asyncio
async def test_reasoning_effort_is_forwarded_for_reasoning_model():
    create = AsyncMock(return_value=_mock_response())
    with patch("harness.llm.openai_client.openai.AsyncOpenAI") as sdk:
        instance = MagicMock()
        instance.chat.completions.create = create
        sdk.return_value = instance
        client = OpenAIClient(model="gpt-5-mini", reasoning_effort="low")
    await client.chat([ChatMessage(role=Role.USER, content="hi")])
    assert create.call_args.kwargs["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_non_reasoning_model_uses_temperature_and_max_tokens():
    create = AsyncMock(return_value=_mock_response())
    client, _ = _make_client_with_create(create)
    await client.chat(
        [ChatMessage(role=Role.USER, content="hi")], temperature=0.0, max_tokens=100
    )
    kwargs = create.call_args[1]
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == 100
    assert "max_completion_tokens" not in kwargs


@pytest.mark.asyncio
async def test_structured_response_format_is_forwarded():
    create = AsyncMock(return_value=_mock_response(content='{"ok": true}'))
    client, _ = _make_client_with_create(create)
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "result", "strict": True, "schema": {}},
    }
    await client.chat(
        [ChatMessage(role=Role.USER, content="grade")],
        response_format=response_format,
    )
    assert create.call_args.kwargs["response_format"] == response_format


@pytest.mark.asyncio
async def test_tool_calls_are_parsed():
    create = AsyncMock(
        return_value=_mock_response(
            content=None, tool_calls=[_raw_tool_call()], finish_reason="tool_calls"
        )
    )
    client, _ = _make_client_with_create(create)
    resp = await client.chat([ChatMessage(role=Role.USER, content="hi")])
    assert resp.tool_calls is not None
    assert resp.tool_calls[0].name == "get_invoice"
    assert resp.tool_calls[0].arguments == {"account": "A1"}


@pytest.mark.asyncio
async def test_tools_passed_in_kwargs():
    create = AsyncMock(return_value=_mock_response())
    client, _ = _make_client_with_create(create)
    tool = ToolDefinition.model_validate(
        {
            "type": "function",
            "function": {
                "name": "get_invoice",
                "description": "Get an invoice",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    )
    await client.chat([ChatMessage(role=Role.USER, content="hi")], tools=[tool])
    kwargs = create.call_args[1]
    assert kwargs["tools"][0]["function"]["name"] == "get_invoice"


@pytest.mark.asyncio
async def test_chat_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("harness.llm.openai_client.asyncio.sleep", AsyncMock())
    err = openai.APIConnectionError(request=MagicMock())
    create = AsyncMock(side_effect=[err, _mock_response()])
    client, _ = _make_client_with_create(create)
    resp = await client.chat([ChatMessage(role=Role.USER, content="hi")])
    assert resp.content == "Hello"
    assert create.call_count == 2


@pytest.mark.asyncio
async def test_chat_exhausts_retries_and_raises(monkeypatch):
    monkeypatch.setattr("harness.llm.openai_client.asyncio.sleep", AsyncMock())
    err = openai.APIConnectionError(request=MagicMock())
    create = AsyncMock(side_effect=[err, err, err])
    client, _ = _make_client_with_create(create)
    with pytest.raises(openai.APIConnectionError):
        await client.chat([ChatMessage(role=Role.USER, content="hi")])
    assert create.call_count == 3


@pytest.mark.asyncio
async def test_aclose_closes_sdk_client():
    create = AsyncMock(return_value=_mock_response())
    client, instance = _make_client_with_create(create)
    await client.aclose()
    instance.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_context_manager_closes_on_exit():
    create = AsyncMock(return_value=_mock_response())
    client, instance = _make_client_with_create(create)
    async with client as c:
        await c.chat([ChatMessage(role=Role.USER, content="hi")])
    instance.close.assert_awaited_once()
