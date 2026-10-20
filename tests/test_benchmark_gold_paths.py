"""One independently reported gold-path test per frozen benchmark task."""

from __future__ import annotations

import pytest

from harness.benchmark_spec import (
    iter_frozen_tasks,
    load_manifest,
    validate_gold_task,
)

_GOLD_CASES = list(iter_frozen_tasks(load_manifest()))


@pytest.mark.parametrize(
    ("domain", "task"),
    _GOLD_CASES,
    ids=[task["id"] for _, task in _GOLD_CASES],
)
@pytest.mark.asyncio
async def test_frozen_task_gold_path(domain, task):
    assert await validate_gold_task(domain, task) == []
