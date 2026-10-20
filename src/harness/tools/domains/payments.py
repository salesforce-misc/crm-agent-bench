"""Payments stateful tool domain.

Provides tools simulating a payments and lending platform: account
balances, payment processing, receipts, loyalty points, an audit log,
funds transfers with validation guardrails, and credit/loan decisioning.
"""

from __future__ import annotations

import json
import random
import string
from typing import cast

from harness.tools.domains.integration import IntegrationEnvironment
from harness.tools.environment import ToolEnvironment, register_domain
from harness.tools.registry import ToolRegistry


def _ref_id() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


@register_domain("payments")
class PaymentsEnvironment(ToolEnvironment):
    def default_state(self) -> dict:
        return {
            "accounts": {
                "ACC-1001": {
                    "account_id": "ACC-1001",
                    "owner": "Dana Whitfield",
                    "balance": 2450.75,
                    "status": "Active",
                },
                "ACC-1002": {
                    "account_id": "ACC-1002",
                    "owner": "Marcus Reyes",
                    "balance": 180.00,
                    "status": "Active",
                },
                "ACC-1003": {
                    "account_id": "ACC-1003",
                    "owner": "Priya Anand",
                    "balance": 9800.50,
                    "status": "Frozen",
                },
            },
            "applicants": {
                "APP-2001": {
                    "applicant_id": "APP-2001",
                    "credit_score": 720,
                    "income": 65000.0,
                },
                "APP-2002": {
                    "applicant_id": "APP-2002",
                    "credit_score": 640,
                    "income": 48000.0,
                },
                "APP-2003": {
                    "applicant_id": "APP-2003",
                    "credit_score": 695,
                    "income": 30000.0,
                },
            },
            "transactions": {},
            "loyalty": {},
            "offline_loyalty_adjustments": {},
            "audit_log": [],
        }

    def register_tools(self, registry: ToolRegistry) -> None:
        data = self.data

        @registry.tool("Get an account's current balance and status")
        def get_account_balance(account_id: str) -> str:
            if account_id not in data["accounts"]:
                return json.dumps({"error": f"Account {account_id} not found"})
            acct = data["accounts"][account_id]
            return json.dumps(
                {
                    "account_id": account_id,
                    "balance": acct["balance"],
                    "status": acct["status"],
                }
            )

        @registry.tool("Process a payment and create a transaction record")
        def process_payment(amount: float, method: str) -> str:
            txn_id = f"TXN-{_ref_id()}"
            txn = {
                "transaction_id": txn_id,
                "amount": amount,
                "method": method,
                "success": True,
                "receipt_sent": False,
            }
            data["transactions"][txn_id] = txn
            return json.dumps(
                {
                    "transaction_id": txn_id,
                    "amount": amount,
                    "method": method,
                    "success": True,
                }
            )

        @registry.tool("Send a receipt for a completed transaction")
        def send_receipt(transaction_id: str, amount: float) -> str:
            if transaction_id not in data["transactions"]:
                return json.dumps({"error": f"Transaction {transaction_id} not found"})
            txn = data["transactions"][transaction_id]
            txn["receipt_sent"] = True
            txn["receipt_amount"] = amount
            return json.dumps(
                {
                    "transaction_id": transaction_id,
                    "amount": amount,
                    "receipt_sent": True,
                }
            )

        @registry.tool("Award loyalty points to an account based on a spend amount")
        def award_loyalty_points(account_id: str, amount: float) -> str:
            points_awarded = int(amount * 10)
            current = data["loyalty"].get(account_id, 0)
            new_balance = current + points_awarded
            data["loyalty"][account_id] = new_balance
            return json.dumps(
                {
                    "account_id": account_id,
                    "points_awarded": points_awarded,
                    "points_balance": new_balance,
                }
            )

        @registry.tool("Credit loyalty points for a verified offline purchase receipt")
        def adjust_loyalty_for_offline_purchase(
            receipt_id: str, account_id: str, amount: float
        ) -> str:
            """Apply one auditable loyalty credit per offline receipt."""
            if account_id not in data["accounts"]:
                return json.dumps(
                    {
                        "error": f"Account {account_id} not found",
                        "status_code": 422,
                    }
                )
            receipt_id = receipt_id.strip()
            if not receipt_id:
                return json.dumps(
                    {"error": "Receipt ID is required", "status_code": 422}
                )
            if amount <= 0:
                return json.dumps(
                    {
                        "error": "Purchase amount must be greater than zero",
                        "status_code": 422,
                    }
                )
            adjustments = data["offline_loyalty_adjustments"]
            if receipt_id in adjustments:
                existing = adjustments[receipt_id]
                return json.dumps(
                    {
                        "error": f"Receipt {receipt_id} was already credited",
                        "status_code": 409,
                        "receipt_id": receipt_id,
                        "account_id": existing["account_id"],
                        "points_awarded": existing["points_awarded"],
                    }
                )

            points_awarded = int(amount * 10)
            new_balance = data["loyalty"].get(account_id, 0) + points_awarded
            data["loyalty"][account_id] = new_balance
            adjustment = {
                "receipt_id": receipt_id,
                "account_id": account_id,
                "amount": amount,
                "points_awarded": points_awarded,
            }
            adjustments[receipt_id] = adjustment
            data["audit_log"].append(
                {"type": "offline_loyalty_adjustment", **adjustment}
            )
            return json.dumps({**adjustment, "points_balance": new_balance})

        @registry.tool("Append a transaction ID to the audit log")
        def log_transaction(transaction_id: str) -> str:
            if transaction_id not in data["transactions"]:
                return json.dumps({"error": f"Transaction {transaction_id} not found"})
            data["audit_log"].append(transaction_id)
            return json.dumps(
                {
                    "transaction_id": transaction_id,
                    "logged": True,
                    "audit_log_size": len(data["audit_log"]),
                }
            )

        @registry.tool(
            "Validate whether a funds transfer between two accounts would be allowed"
        )
        def validate_transfer(from_account: str, to_account: str, amount: float) -> str:
            if from_account not in data["accounts"]:
                return json.dumps(
                    {
                        "valid": False,
                        "reason": f"Account {from_account} not found",
                        "status_code": 422,
                    }
                )
            if to_account not in data["accounts"]:
                return json.dumps(
                    {
                        "valid": False,
                        "reason": f"Account {to_account} not found",
                        "status_code": 422,
                    }
                )
            if amount <= 0:
                return json.dumps(
                    {
                        "valid": False,
                        "reason": "Transfer amount must be greater than zero",
                        "status_code": 422,
                    }
                )
            from_status = data["accounts"][from_account]["status"]
            if from_status != "Active":
                return json.dumps(
                    {
                        "valid": False,
                        "reason": f"Source account {from_account} is {from_status.lower()}",
                        "status_code": 422,
                    }
                )
            to_status = data["accounts"][to_account]["status"]
            if to_status != "Active":
                return json.dumps(
                    {
                        "valid": False,
                        "reason": f"Destination account {to_account} is {to_status.lower()}",
                        "status_code": 422,
                    }
                )
            if data["accounts"][from_account]["balance"] < amount:
                return json.dumps(
                    {
                        "valid": False,
                        "reason": f"Insufficient funds in {from_account}",
                        "status_code": 422,
                    }
                )
            return json.dumps({"valid": True})

        @registry.tool("Execute a funds transfer between two accounts")
        def execute_transfer(from_account: str, to_account: str, amount: float) -> str:
            if from_account not in data["accounts"]:
                return json.dumps(
                    {
                        "error": f"Account {from_account} not found",
                        "status_code": 422,
                    }
                )
            if to_account not in data["accounts"]:
                return json.dumps(
                    {
                        "error": f"Account {to_account} not found",
                        "status_code": 422,
                    }
                )
            if amount <= 0:
                return json.dumps(
                    {
                        "error": "Transfer amount must be greater than zero",
                        "status_code": 422,
                    }
                )
            from_acct = data["accounts"][from_account]
            if from_acct["status"] != "Active":
                return json.dumps(
                    {
                        "error": f"Source account {from_account} is {from_acct['status'].lower()}",
                        "status_code": 422,
                    }
                )
            to_acct = data["accounts"][to_account]
            if to_acct["status"] != "Active":
                return json.dumps(
                    {
                        "error": f"Destination account {to_account} is {to_acct['status'].lower()}",
                        "status_code": 422,
                    }
                )
            if from_acct["balance"] < amount:
                return json.dumps(
                    {
                        "error": f"Insufficient funds in {from_account}",
                        "status_code": 422,
                    }
                )
            from_acct["balance"] -= amount
            to_acct["balance"] += amount
            transfer_id = f"XFER-{_ref_id()}"
            return json.dumps(
                {
                    "transfer_id": transfer_id,
                    "from_account": from_account,
                    "to_account": to_account,
                    "amount": amount,
                    "success": True,
                }
            )

        @registry.tool("Get an applicant's credit score and income")
        def get_credit_score(applicant_id: str) -> str:
            if applicant_id not in data["applicants"]:
                return json.dumps(
                    {
                        "found": False,
                        "applicant_id": applicant_id,
                        "message": f"Applicant {applicant_id} not found",
                    }
                )
            applicant = data["applicants"][applicant_id]
            return json.dumps(
                {
                    "applicant_id": applicant_id,
                    "credit_score": applicant["credit_score"],
                    "income": applicant["income"],
                }
            )

        @registry.tool("Evaluate a loan application and return an approval decision")
        def evaluate_loan(applicant_id: str, amount: float) -> str:
            if applicant_id not in data["applicants"]:
                return json.dumps({"error": f"Applicant {applicant_id} not found"})
            applicant = data["applicants"][applicant_id]
            credit_score = applicant["credit_score"]
            income = applicant["income"]
            if credit_score >= 680 and amount <= income * 3:
                return json.dumps(
                    {
                        "applicant_id": applicant_id,
                        "decision": "approved",
                        "reason": "Credit score and requested amount meet lending criteria",
                    }
                )
            if credit_score < 680:
                reason = f"Credit score {credit_score} is below the minimum of 680"
            else:
                reason = (
                    f"Requested amount {amount} exceeds allowed maximum of "
                    f"{income * 3} (3x income)"
                )
            return json.dumps(
                {
                    "applicant_id": applicant_id,
                    "decision": "denied",
                    "reason": reason,
                }
            )

        IntegrationEnvironment.register_gateway_tools(
            cast(IntegrationEnvironment, self), registry
        )
