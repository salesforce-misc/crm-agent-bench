"""Regression coverage for the retail failure-analysis repairs."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date

import pytest

import harness.tools.domains  # noqa: F401
from harness.bench_runner import load_tasks
from harness.benchmark_spec import load_manifest, validate_manifest
from harness.tools.environment import create_environment


def _retail_task(task_id: str):
    return next(
        task for task in load_tasks("src/tasks/retail.yaml") if task.id == task_id
    )


def test_repaired_retail_contracts_are_count_consistent():
    tasks = load_tasks("src/tasks/retail.yaml")
    for task in tasks:
        expected = Counter(task.expected_actions)
        assert Counter(task.expected_order or []) <= expected, task.id
        for tool_name, spec in (task.expected_args or {}).items():
            required = len(spec) if isinstance(spec, list) else 1
            assert expected[tool_name] >= required, (task.id, tool_name)

    assert validate_manifest(load_manifest()) == []


def test_negative_returns_use_grounded_non_mutating_eligibility_checks():
    for task_id in (
        "commerce_returns_11",
        "commerce_returns_12",
        "commerce_returns_23",
    ):
        task = _retail_task(task_id)
        assert "check_return_eligibility" in task.expected_actions
        assert "process_return" in (task.forbidden_actions or [])


@pytest.mark.asyncio
async def test_return_eligibility_and_mutation_enforce_authenticated_owner():
    env = create_environment("retail")
    registry = env.create_registry()
    await registry.execute(
        "authenticate_customer", {"customer_id": "CUST-501", "pin": "1234"}
    )

    eligibility = json.loads(
        await registry.execute("check_return_eligibility", {"order_id": "ORD-9002"})
    )
    assert eligibility["eligible"] is False
    assert "does not belong" in eligibility["reason"]

    rejected = json.loads(
        await registry.execute("process_return", {"order_id": "ORD-9002"})
    )
    assert rejected["status_code"] == 403
    assert env.data["orders"]["ORD-9002"]["status"] == "Shipped"


@pytest.mark.asyncio
async def test_cart_guard_blocks_other_customer_but_records_safe_rejection():
    env = create_environment(
        "retail", initial_state={"session": {"requester_email": "phouston@example.com"}}
    )
    registry = env.create_registry()

    own = json.loads(
        await registry.execute(
            "CSA_Product_Compatibility_Of_Cart",
            {"user_email": "phouston@example.com"},
        )
    )
    assert "error" not in own

    other = await registry.execute_with_status(
        "CSA_Product_Compatibility_Of_Cart",
        {"user_email": "ssutherland@example.com"},
    )
    payload = json.loads(other.result)
    assert other.ok is True  # executed policy denial; attempted call remains gradeable
    assert payload["status_code"] == 403
    assert "items" not in payload


@pytest.mark.asyncio
async def test_compatibility_list_is_typed_and_technically_coherent():
    env = create_environment("retail")
    registry = env.create_registry()
    definition = next(
        item
        for item in registry.get_definitions()
        if item.function.name == "CSA_List_Compatible_Motherboards"
    )
    assert definition.function.parameters["properties"]["compatibility"]["enum"] == [
        "compatible",
        "incompatible",
    ]

    result = json.loads(
        await registry.execute(
            "CSA_List_Compatible_Motherboards",
            {
                "productName": "AMD Ryzen 7 7800X3D",
                "compatibility": "compatible",
            },
        )
    )
    names = [item["product"] for item in result["motherboards"]]
    assert any("B650E-F" in name for name in names)
    assert all("B550" not in name for name in names)


def test_active_promotions_are_valid_at_frozen_clock():
    state = create_environment("retail").get_state()
    clock = date.fromisoformat(state["benchmark_clock"])
    assert clock == date(2026, 9, 21)
    for promotion in state["promotions"]:
        if promotion["active"]:
            assert date.fromisoformat(promotion["start"]) <= clock
            assert clock <= date.fromisoformat(promotion["end"])


@pytest.mark.asyncio
async def test_cart_result_contains_display_names_and_ids():
    registry = create_environment("retail").create_registry()
    result = json.loads(await registry.execute("get_cart", {"customer_id": "CUST-501"}))
    assert result["items"] == [
        {
            "product_id": "PROD-1002",
            "quantity": 1,
            "name": "Smart Fitness Watch",
        }
    ]


def test_return_tasks_grade_the_unrelated_order_through_expected_state():
    for task_id in ("commerce_returns_16", "commerce_returns_22"):
        task = _retail_task(task_id)
        assert task.expected_state["orders"]["ORD-9003"]["status"] == "Returned"
