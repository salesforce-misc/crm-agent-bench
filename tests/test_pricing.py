"""Tests for the per-model pricing table + cost engine."""

import pytest

from harness.llm.pricing import PriceCard, cost_usd, load_pricing, resolve_model


class TestLoadPricing:
    def test_loads_default_table(self):
        data = load_pricing()
        assert data["version"] == 1
        assert "gpt-4.1" in data["models"]

    def test_memoized_across_calls(self):
        # load_pricing() caches by resolved path; calling twice should return
        # the exact same object (not just equal), proving no re-parse.
        first = load_pricing()
        second = load_pricing()
        assert first is second


class TestResolveModel:
    def test_exact_key_match(self):
        card = resolve_model("gpt-4.1")
        assert card is not None
        assert card.model_key == "gpt-4.1"
        assert card.input == 2.00
        assert card.output == 8.00

    def test_gpt_6_astra_official_rates(self):
        card = resolve_model("gpt-6-astra")
        assert card is not None
        assert card.input == 10.00
        assert card.output == 50.00
        assert card.cache_read == 1.00
        assert card.cache_write == 12.50

    def test_exact_alias_match(self):
        card = resolve_model("openai/gpt-4.1")
        assert card is not None
        assert card.model_key == "gpt-4.1"

    def test_regex_alias_match(self):
        # gpt-4.1 is aliased via "regex:^gpt-4\.1[-.].*$" to catch dated/preview
        # deployment ids.
        card = resolve_model("gpt-4.1-2026-04-14-preview")
        assert card is not None
        assert card.model_key == "gpt-4.1"

    def test_normalized_fallback_match(self):
        # "Claude 4.6 Sonnet" (display name from bench_report.MODEL_DISPLAY)
        # should fall back to matching the "claude-sonnet-4.6" alias
        # "Claude 4.6 Sonnet" via normalization (spaces stripped, lowercased).
        card = resolve_model("claude 4.6 sonnet")
        assert card is not None
        assert card.model_key == "claude-sonnet-4.6"

    def test_normalized_fallback_match_against_canonical_key(self):
        # Canonical key itself should match case/hyphen-insensitively too.
        card = resolve_model("GPT-4.1")
        assert card is not None
        assert card.model_key == "gpt-4.1"

    def test_unknown_model_returns_none(self):
        assert resolve_model("totally-unknown-model-xyz") is None

    def test_empty_string_returns_none(self):
        assert resolve_model("") is None


class TestCostUsd:
    def test_none_card_returns_none(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 50}
        assert cost_usd(usage, None) is None

    def test_basic_input_output_cost(self):
        card = PriceCard(
            model_key="test-model",
            input=2.00,
            output=8.00,
            cache_read=None,
            cache_write=None,
            effective_date="2026-01-01",
        )
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 500_000}
        # (1_000_000 * 2.00 / 1e6) + (500_000 * 8.00 / 1e6) = 2.00 + 4.00
        assert cost_usd(usage, card) == 6.00

    def test_missing_usage_keys_default_to_zero(self):
        card = PriceCard(
            model_key="test-model",
            input=2.00,
            output=8.00,
            cache_read=0.50,
            cache_write=2.50,
            effective_date="2026-01-01",
        )
        # No keys at all -> cost is 0, not an error.
        assert cost_usd({}, card) == 0.0

    def test_cache_read_priced_tier(self):
        card = PriceCard(
            model_key="test-model",
            input=2.00,
            output=8.00,
            cache_read=0.50,
            cache_write=2.50,
            effective_date="2026-01-01",
        )
        usage = {
            "prompt_tokens": 100_000,
            "completion_tokens": 20_000,
            "cache_read_tokens": 50_000,
            "cache_write_tokens": 10_000,
        }
        # (100_000*2.00 + 20_000*8.00 + 50_000*0.50 + 10_000*2.50) / 1e6
        # = (200_000 + 160_000 + 25_000 + 25_000) / 1e6 = 410_000 / 1e6
        expected = (
            100_000 * 2.00 + 20_000 * 8.00 + 50_000 * 0.50 + 10_000 * 2.50
        ) / 1e6
        assert cost_usd(usage, card) == pytest.approx(expected)
        assert cost_usd(usage, card) == pytest.approx(0.41)

    def test_null_cache_tier_contributes_zero_when_no_cache_tokens(self):
        card = PriceCard(
            model_key="test-model",
            input=2.00,
            output=8.00,
            cache_read=None,
            cache_write=None,
            effective_date="2026-01-01",
        )
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0}
        # No cache tokens present -> null cache price never kicks in the
        # input-price fallback; cost is exactly the input contribution.
        assert cost_usd(usage, card) == 2.00

    def test_cache_tokens_present_with_no_price_falls_back_to_input(self):
        # Documented safe-over-estimate: if a model's card has no cache price
        # defined (None) but usage reports cache tokens anyway, charge those
        # tokens at the `input` price rather than treating them as free.
        card = PriceCard(
            model_key="test-model",
            input=3.00,
            output=15.00,
            cache_read=None,
            cache_write=None,
            effective_date="2026-01-01",
        )
        usage = {
            "prompt_tokens": 100_000,
            "completion_tokens": 0,
            "cache_read_tokens": 40_000,
            "cache_write_tokens": 10_000,
        }
        # (100_000*3.00 + 40_000*3.00 + 10_000*3.00) / 1e6 -- all at input price
        expected = (100_000 * 3.00 + 40_000 * 3.00 + 10_000 * 3.00) / 1e6
        assert cost_usd(usage, card) == pytest.approx(expected)

    def test_reasoning_tokens_not_double_charged(self):
        # reasoning_tokens is informational only (already inside
        # completion_tokens per provider convention) -- adding it to usage
        # must not change the cost versus omitting it.
        card = PriceCard(
            model_key="test-model",
            input=2.00,
            output=8.00,
            cache_read=None,
            cache_write=None,
            effective_date="2026-01-01",
        )
        usage_without_reasoning = {"prompt_tokens": 1000, "completion_tokens": 500}
        usage_with_reasoning = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "reasoning_tokens": 300,
        }
        assert cost_usd(usage_without_reasoning, card) == cost_usd(
            usage_with_reasoning, card
        )
