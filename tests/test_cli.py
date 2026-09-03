"""Tests for the harness CLI entry point — client/agent construction, prompt
loading, and event printing. The interactive input loop itself is not exercised
(it blocks on stdin), but every pure helper it drives is."""

from unittest.mock import MagicMock, patch

import pytest

from harness import cli
from harness.models import AgentEvent, ToolCall


def test_create_llm_client_reads_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)

    with patch("harness.llm.create_llm_client") as create:
        cli._create_llm_client()
    create.assert_called_once_with(
        provider="openai", model="gpt-4o-mini", api_key="sk-test", base_url=None
    )


def test_load_system_prompt_from_env(monkeypatch, tmp_path):
    prompt_file = tmp_path / "p.yaml"
    prompt_file.write_text("system_prompt: Be terse.\n")
    monkeypatch.setenv("AGENT_PROMPT", str(prompt_file))

    with patch("harness.prompts.loader.load_prompt", return_value="Be terse.") as lp:
        assert cli._load_system_prompt() == "Be terse."
    lp.assert_called_once()


def test_load_system_prompt_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("AGENT_PROMPT", raising=False)
    # Force every candidate path (env + bundled) to report absent so we exercise
    # the hardcoded default branch. A self-referential mock makes `.parent`,
    # `/`, and `.exists()` all resolve to a path that does not exist.
    fake_path = MagicMock()
    fake_path.exists.return_value = False
    fake_path.parent = fake_path
    fake_path.__truediv__.return_value = fake_path
    with patch("harness.cli.Path", return_value=fake_path):
        prompt = cli._load_system_prompt()
    assert "helpful AI assistant" in prompt


def test_create_agent_burr(monkeypatch):
    monkeypatch.setenv("USE_BURR", "1")
    with patch("harness.cli._create_llm_client", return_value=MagicMock()):
        agent = cli._create_agent()
    from harness.agent_burr import AgentLoopBurr

    assert isinstance(agent, AgentLoopBurr)


def test_create_agent_plain(monkeypatch):
    monkeypatch.setenv("USE_BURR", "0")
    with patch("harness.cli._create_llm_client", return_value=MagicMock()):
        agent = cli._create_agent()
    from harness.agent import AgentLoop

    assert isinstance(agent, AgentLoop)


def test_print_event_text_delta(capsys):
    cli._print_event(AgentEvent(type="text_delta", content="hello"))
    assert capsys.readouterr().out == "hello"


def test_print_event_tool_call(capsys):
    event = AgentEvent(
        type="tool_call",
        tool_name="get_invoice",
        tool_call=ToolCall(id="tc1", name="get_invoice", arguments={"a": 1}),
    )
    cli._print_event(event)
    out = capsys.readouterr().out
    assert "get_invoice" in out
    assert "Calling" in out


def test_print_event_tool_result(capsys):
    cli._print_event(AgentEvent(type="tool_result", tool_result="the answer"))
    assert "the answer" in capsys.readouterr().out


def test_print_event_error(capsys):
    cli._print_event(AgentEvent(type="error", error="boom"))
    assert "boom" in capsys.readouterr().out


def test_main_help_exits_without_client(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", "--help"])
    # --help must not require credentials / construct a client.
    with patch("harness.cli._create_agent") as ca, pytest.raises(SystemExit):
        cli.main()
    ca.assert_not_called()
