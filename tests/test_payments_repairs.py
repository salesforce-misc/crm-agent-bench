"""Regression coverage for the payments benchmark fairness repairs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import harness.tools.domains  # noqa: F401 -- register payments
from harness.bench_runner import load_tasks
from harness.tools.environment import create_environment


def test_reference_agent_does_not_embed_seeded_lending_answers() -> None:
    prompt = Path("src/agents/payments.yaml").read_text()

    assert "APP-2003" not in prompt
    assert "credit score is 695" not in prompt


class TestFrozenTransferInvariant:
    def setup_method(self) -> None:
        self.environment = create_environment("payments")
        self.registry = self.environment.create_registry()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("from_account", "to_account", "role"),
        [
            ("ACC-1003", "ACC-1001", "Source"),
            ("ACC-1001", "ACC-1003", "Destination"),
        ],
    )
    async def test_validation_and_execution_both_reject_frozen_accounts(
        self, from_account: str, to_account: str, role: str
    ) -> None:
        before = self.environment.get_state()["accounts"]
        arguments = {
            "from_account": from_account,
            "to_account": to_account,
            "amount": 200,
        }

        validation = json.loads(
            await self.registry.execute("validate_transfer", arguments)
        )
        execution = json.loads(
            await self.registry.execute("execute_transfer", arguments)
        )

        assert validation["valid"] is False
        assert validation["reason"].startswith(f"{role} account")
        assert execution["error"].startswith(f"{role} account")
        assert self.environment.get_state()["accounts"] == before


class TestOfflineLoyaltyAdjustment:
    @pytest.mark.asyncio
    async def test_receipt_can_only_be_credited_once(self) -> None:
        environment = create_environment("payments")
        registry = environment.create_registry()
        arguments = {
            "receipt_id": "OFFLINE-1002-001",
            "account_id": "ACC-1002",
            "amount": 100,
        }

        first = json.loads(
            await registry.execute("adjust_loyalty_for_offline_purchase", arguments)
        )
        duplicate = json.loads(
            await registry.execute("adjust_loyalty_for_offline_purchase", arguments)
        )

        assert first["points_awarded"] == 1000
        assert duplicate["status_code"] == 409
        assert environment.data["loyalty"]["ACC-1002"] == 1000
        assert environment.data["offline_loyalty_adjustments"] == {
            "OFFLINE-1002-001": {
                "receipt_id": "OFFLINE-1002-001",
                "account_id": "ACC-1002",
                "amount": 100,
                "points_awarded": 1000,
            }
        }
        assert environment.data["audit_log"] == [
            {
                "type": "offline_loyalty_adjustment",
                "receipt_id": "OFFLINE-1002-001",
                "account_id": "ACC-1002",
                "amount": 100,
                "points_awarded": 1000,
            }
        ]


def test_repaired_payment_task_contracts() -> None:
    tasks = {task.id: task for task in load_tasks("src/tasks/payments.yaml")}

    assert tasks["payments_transfer_happy_basic_01"].expected_order == [
        "validate_transfer",
        "execute_transfer",
    ]
    assert tasks["payments_transfer_collateral_01"].expected_args == {
        "validate_transfer": {
            "from_account": "ACC-1002",
            "to_account": "ACC-1001",
            "amount": 60,
        },
        "execute_transfer": {
            "from_account": "ACC-1002",
            "to_account": "ACC-1001",
            "amount": 60,
        },
    }
    for task_id in (
        "payments_transfer_zero_amount_01",
        "payments_transfer_negative_amount_01",
        "payments_transfer_frozen_source_01",
        "payments_transfer_frozen_destination_01",
    ):
        assert "validate_transfer" in tasks[task_id].expected_actions
        assert "execute_transfer" in (tasks[task_id].forbidden_actions or [])
