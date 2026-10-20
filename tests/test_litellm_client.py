"""Tests for LiteLLMClient — per-call isolation of api_key and base_url."""

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest

from harness.llm.litellm_client import LiteLLMClient
from harness.models import ChatMessage, Role, ToolCall, ToolDefinition, ToolFunction


def test_reasoning_effort_does_not_implicitly_read_agent_env(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "high")

    client = LiteLLMClient(model="gpt-4o")

    assert client._reasoning_effort is None


ACOMPLETION_PARAMS = set(inspect.signature(litellm.acompletion).parameters.keys())


def _mock_response(content="Hello"):
    """Create a minimal litellm-style response object."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


@pytest.mark.asyncio
@patch("harness.llm.litellm_client.litellm")
async def test_base_url_passed_per_call(mock_litellm):
    mock_litellm.acompletion = AsyncMock(return_value=_mock_response())
    mock_litellm.ssl_verify = True

    client = LiteLLMClient(model="gpt-4o", base_url="https://gateway.example.com")
    messages = [ChatMessage(role=Role.USER, content="hi")]
    await client.chat(messages)

    kwargs = mock_litellm.acompletion.call_args[1]
    assert kwargs["base_url"] == "https://gateway.example.com"


@pytest.mark.asyncio
@patch("harness.llm.litellm_client.litellm")
async def test_no_base_url_omits_api_base(mock_litellm):
    mock_litellm.acompletion = AsyncMock(return_value=_mock_response())
    mock_litellm.ssl_verify = True

    client = LiteLLMClient(model="vertex_ai/gemini-3-flash")
    messages = [ChatMessage(role=Role.USER, content="hi")]
    await client.chat(messages)

    kwargs = mock_litellm.acompletion.call_args[1]
    assert "base_url" not in kwargs


@pytest.mark.asyncio
@patch("harness.llm.litellm_client.litellm")
async def test_gpt_6_reasoning_model_omits_temperature(mock_litellm):
    usage = MagicMock()
    usage.prompt_tokens = None
    usage.input_tokens = 44
    usage.completion_tokens = None
    usage.output_tokens = 17
    usage.total_tokens = 61
    usage.prompt_tokens_details = None
    usage.input_tokens_details = None
    usage.completion_tokens_details = None
    usage.output_tokens_details = None
    usage.cache_creation_input_tokens = 0
    item = MagicMock()
    item.type = "function_call"
    item.call_id = "call_1"
    item.name = "echo"
    item.arguments = '{"text":"OK"}'
    response = MagicMock()
    response.status = "completed"
    response.output_text = ""
    response.output = [item]
    response.usage = usage
    mock_litellm.aresponses = AsyncMock(return_value=response)
    mock_litellm.ssl_verify = True

    client = LiteLLMClient(
        model="gpt-6-astra",
        base_url="https://gateway.example.com/v1",
        reasoning_effort="low",
    )
    tool = ToolDefinition(
        function=ToolFunction(
            name="echo",
            description="Echo text",
            parameters={"type": "object", "properties": {}},
        )
    )
    result = await client.chat(
        [ChatMessage(role=Role.USER, content="hi")], tools=[tool], max_tokens=256
    )

    kwargs = mock_litellm.aresponses.call_args.kwargs
    assert kwargs["max_output_tokens"] == 256
    assert "temperature" not in kwargs
    assert "max_tokens" not in kwargs
    assert kwargs["reasoning"] == {"effort": "low"}
    assert kwargs["api_base"] == "https://gateway.example.com/v1"
    assert kwargs["tools"][0]["name"] == "echo"
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls == [
        ToolCall(id="call_1", name="echo", arguments={"text": "OK"})
    ]
    assert result.usage == {
        "prompt_tokens": 44,
        "completion_tokens": 17,
        "total_tokens": 61,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
    }


def test_gpt_6_responses_input_preserves_tool_history():
    client = LiteLLMClient(model="gpt-6-astra")
    items = client._build_responses_input(
        [
            ChatMessage(
                role=Role.ASSISTANT,
                tool_calls=[ToolCall(id="call_1", name="echo", arguments={"x": 1})],
            ),
            ChatMessage(
                role=Role.TOOL,
                content='{"ok":true}',
                tool_call_id="call_1",
            ),
        ]
    )
    assert items == [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "echo",
            "arguments": '{"x": 1}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"ok":true}',
        },
    ]


@pytest.mark.asyncio
@patch("harness.llm.litellm_client.litellm")
async def test_two_clients_dont_cross_contaminate(mock_litellm):
    """The bug: creating a gateway client must not affect a Vertex AI client."""
    mock_litellm.acompletion = AsyncMock(return_value=_mock_response())
    mock_litellm.ssl_verify = True

    gateway_client = LiteLLMClient(
        model="gpt-4o", api_key="gw-key", base_url="https://gateway.example.com"
    )
    vertex_client = LiteLLMClient(model="vertex_ai/gemini-3-flash")

    messages = [ChatMessage(role=Role.USER, content="hi")]

    # Gateway call should include api_base and api_key
    await gateway_client.chat(messages)
    gw_kwargs = mock_litellm.acompletion.call_args[1]
    assert gw_kwargs["base_url"] == "https://gateway.example.com"
    assert gw_kwargs["api_key"] == "gw-key"

    # Vertex call should have neither
    await vertex_client.chat(messages)
    vx_kwargs = mock_litellm.acompletion.call_args[1]
    assert "api_base" not in vx_kwargs
    assert "api_key" not in vx_kwargs


@pytest.mark.asyncio
@patch("harness.llm.litellm_client.litellm")
async def test_api_key_passed_per_call(mock_litellm):
    mock_litellm.acompletion = AsyncMock(return_value=_mock_response())
    mock_litellm.ssl_verify = True

    client = LiteLLMClient(model="gpt-4o", api_key="sk-test-123")
    messages = [ChatMessage(role=Role.USER, content="hi")]
    await client.chat(messages)

    kwargs = mock_litellm.acompletion.call_args[1]
    assert kwargs["api_key"] == "sk-test-123"


@pytest.mark.asyncio
@patch("harness.llm.litellm_client.litellm")
async def test_structured_response_format_passed_per_call(mock_litellm):
    mock_litellm.acompletion = AsyncMock(return_value=_mock_response())
    mock_litellm.ssl_verify = True
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "result", "strict": True, "schema": {}},
    }

    client = LiteLLMClient(model="gpt-4o")
    await client.chat(
        [ChatMessage(role=Role.USER, content="grade")],
        response_format=response_format,
    )

    assert (
        mock_litellm.acompletion.call_args.kwargs["response_format"] == response_format
    )


@pytest.mark.asyncio
@patch("harness.llm.litellm_client.litellm")
async def test_stream_passes_per_call_kwargs(mock_litellm):
    """stream_chat should also pass api_base/api_key per-call."""
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = MagicMock(content="hi", tool_calls=None)
    chunk.choices[0].finish_reason = "stop"

    async def mock_stream(*args, **kwargs):
        async def gen():
            yield chunk

        return gen()

    mock_litellm.acompletion = AsyncMock(side_effect=mock_stream)
    mock_litellm.ssl_verify = True

    client = LiteLLMClient(
        model="gpt-4o", api_key="sk-key", base_url="https://gw.example.com"
    )
    messages = [ChatMessage(role=Role.USER, content="hi")]
    async for _ in client.stream_chat(messages):
        pass

    kwargs = mock_litellm.acompletion.call_args[1]
    assert kwargs["base_url"] == "https://gw.example.com"
    assert kwargs["api_key"] == "sk-key"


@pytest.mark.asyncio
@patch("harness.llm.litellm_client.litellm")
async def test_kwargs_match_acompletion_signature(mock_litellm):
    """All kwargs we pass must be valid litellm.acompletion parameters."""
    mock_litellm.acompletion = AsyncMock(return_value=_mock_response())
    mock_litellm.ssl_verify = True

    client = LiteLLMClient(
        model="gpt-4o", api_key="sk-key", base_url="https://gw.example.com"
    )
    messages = [ChatMessage(role=Role.USER, content="hi")]
    await client.chat(messages)

    kwargs = mock_litellm.acompletion.call_args[1]
    invalid = set(kwargs.keys()) - ACOMPLETION_PARAMS
    assert not invalid, f"Invalid kwargs for litellm.acompletion: {invalid}"


@pytest.mark.asyncio
async def test_chat_retries_transient_error_then_succeeds(monkeypatch):
    # patch.object keeps the real exception classes — patching the whole module
    # would turn them into MagicMocks that cannot appear in an `except` clause.
    err = litellm.APIConnectionError(
        message="boom", llm_provider="openai", model="gpt-4o"
    )
    calls = {"n": 0}

    async def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise err
        return _mock_response()

    monkeypatch.setattr(litellm, "acompletion", flaky)
    monkeypatch.setattr("harness.llm.litellm_client.asyncio.sleep", AsyncMock())

    client = LiteLLMClient(model="gpt-4o")
    resp = await client.chat([ChatMessage(role=Role.USER, content="hi")])
    assert calls["n"] == 2  # one failure, one success
    assert resp.content == "Hello"


@pytest.mark.asyncio
async def test_chat_exhausts_retries_and_raises(monkeypatch):
    monkeypatch.setenv("LITELLM_NUM_RETRIES", "1")  # 1 retry => 2 attempts
    err = litellm.Timeout(message="slow", model="gpt-4o", llm_provider="openai")
    calls = {"n": 0}

    async def always_fail(**kwargs):
        calls["n"] += 1
        raise err

    monkeypatch.setattr(litellm, "acompletion", always_fail)
    monkeypatch.setattr("harness.llm.litellm_client.asyncio.sleep", AsyncMock())

    client = LiteLLMClient(model="gpt-4o")
    with pytest.raises(litellm.Timeout):
        await client.chat([ChatMessage(role=Role.USER, content="hi")])
    assert calls["n"] == 2  # exhausted both attempts


@pytest.mark.asyncio
async def test_num_retries_zero_makes_single_attempt(monkeypatch):
    monkeypatch.setenv("LITELLM_NUM_RETRIES", "0")
    err = litellm.APIConnectionError(
        message="boom", llm_provider="openai", model="gpt-4o"
    )
    calls = {"n": 0}

    async def always_fail(**kwargs):
        calls["n"] += 1
        raise err

    monkeypatch.setattr(litellm, "acompletion", always_fail)
    monkeypatch.setattr("harness.llm.litellm_client.asyncio.sleep", AsyncMock())

    client = LiteLLMClient(model="gpt-4o")
    with pytest.raises(litellm.APIConnectionError):
        await client.chat([ChatMessage(role=Role.USER, content="hi")])
    assert calls["n"] == 1  # no retry


@pytest.mark.asyncio
async def test_timeout_passed_to_acompletion(monkeypatch):
    monkeypatch.setenv("LITELLM_REQUEST_TIMEOUT", "42")
    captured = {}

    async def capture(**kwargs):
        captured.update(kwargs)
        return _mock_response()

    monkeypatch.setattr(litellm, "acompletion", capture)
    client = LiteLLMClient(model="gpt-4o")
    await client.chat([ChatMessage(role=Role.USER, content="hi")])
    assert captured["timeout"] == 42.0


@pytest.mark.asyncio
@patch("harness.llm.litellm_client.litellm")
async def test_empty_choices_returns_empty_turn(mock_litellm):
    """Vertex/Gemini can return a response with no choices (blocked/dropped
    candidate, or the whole budget spent on reasoning). chat() must yield an
    empty assistant turn rather than raising IndexError on choices[0]."""
    resp = MagicMock()
    resp.choices = []
    resp.usage = None
    mock_litellm.acompletion = AsyncMock(return_value=resp)
    mock_litellm.ssl_verify = True

    client = LiteLLMClient(model="vertex_ai/gemini-3.8-flash")
    result = await client.chat([ChatMessage(role=Role.USER, content="hi")])

    assert result.content == ""
    assert result.tool_calls is None
    assert result.finish_reason == "empty"
