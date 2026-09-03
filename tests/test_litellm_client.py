"""Tests for LiteLLMClient — per-call isolation of api_key and base_url."""

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest

from harness.llm.litellm_client import LiteLLMClient
from harness.models import ChatMessage, Role

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
