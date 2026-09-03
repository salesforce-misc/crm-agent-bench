"""P1.9: create_llm_client must validate the provider against an allowlist and
raise on unknown values, instead of silently defaulting to OpenAI — a typo in
LLM_PROVIDER otherwise switches provider behavior and invalidates comparisons.
"""

from __future__ import annotations

import pytest

from harness.llm import create_llm_client


def test_openai_provider_ok():
    client = create_llm_client(provider="openai", model="gpt-4o", api_key="x")
    assert type(client).__name__ == "OpenAIClient"


def test_unknown_provider_raises():
    with pytest.raises(ValueError) as exc:
        create_llm_client(provider="opena1", model="gpt-4o", api_key="x")
    # error should name the bad value and the allowed set
    msg = str(exc.value)
    assert "opena1" in msg
    assert "openai" in msg


def test_empty_provider_raises():
    with pytest.raises(ValueError):
        create_llm_client(provider="", model="gpt-4o", api_key="x")


@pytest.mark.parametrize("provider", ["openai", "litellm"])
def test_known_providers_do_not_raise_validation(provider):
    # Construction may fail later for other reasons, but the allowlist check
    # itself must accept all known providers.
    try:
        create_llm_client(provider=provider, model="m", api_key="x")
    except ValueError as e:
        if "Unknown LLM provider" in str(e):
            raise
