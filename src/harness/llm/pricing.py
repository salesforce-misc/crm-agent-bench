"""Per-model pricing table + cost computation for the "token cost per completed
task" feature.

Loads `pricing.yaml` (versioned, human-maintained) and exposes:
  - `load_pricing()` — parse the table once, memoized.
  - `resolve_model(name)` — map a raw model string (display name, gateway
    model id, litellm model id, etc.) to a `PriceCard`.
  - `cost_usd(usage, card)` — compute USD cost for a usage dict against a card.

Unknown models resolve to `None`; callers must treat that as "unpriced" and
never crash — a stale/incomplete pricing table must not break a benchmark run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PRICING_PATH = Path(__file__).parent / "pricing.yaml"

# Module-level memo of the parsed pricing table, keyed by resolved path, so
# repeated load_pricing() calls (e.g. once per task) are cheap.
_PRICING_CACHE: dict[Path, dict[str, Any]] = {}


@dataclass(frozen=True)
class PriceCard:
    """USD price per 1,000,000 tokens for a single model, at a point in time."""

    model_key: str
    input: float
    output: float
    cache_read: float | None
    cache_write: float | None
    effective_date: str


def load_pricing(path: str | Path | None = None) -> dict[str, Any]:
    """Load and parse pricing.yaml, memoized by resolved path.

    Returns the raw parsed dict (`{"version": ..., "models": {...}}`).
    """
    resolved = Path(path) if path is not None else _DEFAULT_PRICING_PATH
    resolved = resolved.resolve()
    if resolved not in _PRICING_CACHE:
        with open(resolved) as f:
            data = yaml.safe_load(f) or {}
        _PRICING_CACHE[resolved] = data
    return _PRICING_CACHE[resolved]


def _normalize(name: str) -> str:
    """Loose fallback normalization: strip whitespace, lowercase, collapse
    internal spaces/underscores/hyphens to nothing, for last-resort matching."""
    return re.sub(r"[\s_-]+", "", name.strip().lower())


def _card_from_entry(model_key: str, entry: dict[str, Any]) -> PriceCard:
    return PriceCard(
        model_key=model_key,
        input=float(entry["input"]),
        output=float(entry["output"]),
        cache_read=entry.get("cache_read"),
        cache_write=entry.get("cache_write"),
        effective_date=str(entry.get("effective_date", "")),
    )


def resolve_model(name: str, path: str | Path | None = None) -> PriceCard | None:
    """Resolve a raw model name/id string to a PriceCard.

    Match order (first match wins):
      1. Exact canonical model key (e.g. "gpt-4.1").
      2. Exact alias string (e.g. "provider__ModelName").
      3. Regex alias (an alias prefixed "regex:", matched with re.search
         against the raw name).
      4. Normalized fallback: lowercase + strip spaces/underscores/hyphens,
         compared against normalized canonical keys and normalized aliases.

    Returns None if nothing matches — callers must treat this as "unpriced",
    not an error.
    """
    if not name:
        return None

    data = load_pricing(path)
    models: dict[str, Any] = data.get("models", {})

    # 1. Exact canonical key.
    if name in models:
        return _card_from_entry(name, models[name])

    # 2 & 3. Exact alias / regex alias.
    regex_candidates: list[tuple[str, str]] = []  # (model_key, pattern)
    for model_key, entry in models.items():
        for alias in entry.get("aliases") or []:
            if alias.startswith("regex:"):
                regex_candidates.append((model_key, alias[len("regex:") :]))
            elif alias == name:
                return _card_from_entry(model_key, entry)

    for model_key, pattern in regex_candidates:
        if re.search(pattern, name):
            return _card_from_entry(model_key, models[model_key])

    # 4. Normalized fallback (case/space/punctuation-insensitive).
    normalized_name = _normalize(name)
    for model_key, entry in models.items():
        if _normalize(model_key) == normalized_name:
            return _card_from_entry(model_key, entry)
        for alias in entry.get("aliases") or []:
            if alias.startswith("regex:"):
                continue
            if _normalize(alias) == normalized_name:
                return _card_from_entry(model_key, entry)

    return None


def cost_usd(usage: dict[str, int], card: PriceCard | None) -> float | None:
    """Compute USD cost for a usage dict against a price card.

    usage keys (all optional, missing -> 0): prompt_tokens, completion_tokens,
    cache_read_tokens, cache_write_tokens, reasoning_tokens, total_tokens.

    Notes:
      - reasoning_tokens are already included within completion_tokens by
        providers (OpenAI's completion_tokens_details.reasoning_tokens is a
        breakdown of completion_tokens, not additive) — do NOT charge them
        separately, or reasoning-heavy calls would be double-billed.
      - cache_read/cache_write price tiers may be null in the table when a
        provider has no separate discounted cache tier; null is treated as
        $0/token for that tier (i.e. those tokens contribute nothing beyond
        whatever else is charged).
      - If cache tokens are present in `usage` but the model's card has no
        cache price defined for that tier (None), we fall back to charging
        those tokens at the `input` price. This is a deliberate safe
        over-estimate: it is better to slightly overstate cost for a model
        whose cache pricing we haven't captured than to silently under-report
        (which could make a model look artificially cheap).

    Returns None if `card` is None (unpriced model) — never raises.
    """
    if card is None:
        return None

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    cache_read_tokens = usage.get("cache_read_tokens", 0)
    cache_write_tokens = usage.get("cache_write_tokens", 0)

    # Default: a null cache price means "no separate tier" -> $0/token. This
    # is a no-op whenever cache_*_tokens is 0 anyway. But if cache tokens
    # actually show up in usage for a model whose card has no cache price, we
    # no longer trust "no tier" and instead charge those tokens at the
    # (always-defined) input price — a deliberate safe over-estimate so an
    # unmodeled cache tier doesn't make a model look artificially cheap.
    cache_read_price = (
        card.cache_read
        if card.cache_read is not None
        else (card.input if cache_read_tokens else 0.0)
    )
    cache_write_price = (
        card.cache_write
        if card.cache_write is not None
        else (card.input if cache_write_tokens else 0.0)
    )

    return (
        (prompt_tokens * card.input / 1e6)
        + (completion_tokens * card.output / 1e6)
        + (cache_read_tokens * cache_read_price / 1e6)
        + (cache_write_tokens * cache_write_price / 1e6)
    )
