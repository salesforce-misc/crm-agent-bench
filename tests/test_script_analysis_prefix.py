"""Analysis-prefix / output-layout contract for discover_models.

The analysis prefix passed to bench_report must match the
bench_output/{prefix}_{model}_{timestamp} directory layout, i.e.
discover_models(prefix) must find the run. This locks in the rule that a prefix
must NOT already include the model segment (a prefix like
"marketing_full_test_sonnet4" could never match its own output dir).

Note: this repo does not ship the vendor-specific shell run scripts, so the
former script-scanning tests were removed; the logic guard below is what
matters and is provider-agnostic.
"""

from __future__ import annotations

import tempfile
from pathlib import Path


def test_prefix_matches_output_layout():
    """A combined {prefix} finds the model in {prefix}_{model}_{ts}."""
    import analysis.bench_report as br

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        br.BENCH_OUTPUT = root
        (root / "csm_full_test_somemodel_20260101_000000").mkdir()
        assert "somemodel" in br.discover_models("csm_full_test", "20260101_000000")


def test_no_per_model_prefix_regression():
    """A prefix that already includes the model segment must NOT match
    {prefix}_{model}_{ts} (proves the discovery contract is meaningful)."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        # Simulate the OLD broken behavior: dir is marketing_full_test_sonnet4_<ts>
        (root / "marketing_full_test_sonnet4_20260101_000000").mkdir()
        import analysis.bench_report as br2

        br2.BENCH_OUTPUT = root
        # Correct (combined) prefix finds the model:
        assert "sonnet4" in br2.discover_models(
            "marketing_full_test", "20260101_000000"
        )
        # Broken per-model prefix finds nothing:
        assert (
            br2.discover_models("marketing_full_test_sonnet4", "20260101_000000") == []
        )
