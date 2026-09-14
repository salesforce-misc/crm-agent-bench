# Ported from sierra-research/tau2-bench (telecom domain), MIT. Reimplemented as pure-Python ToolEnvironment (no upstream code copied).
"""Telecom stateful tool domain.

Provides 13 tools that read/write a shared state dict simulating a mobile
carrier's support systems: customers, lines, devices, plans, and bills.
Covers two workflows — network/device troubleshooting (line status,
diagnostics, resets, restarts, roaming) and billing (bill lookup, plan
changes, payments, refunds) — with policy-gated business rejections for
disallowed actions (e.g. refunds over the authorization limit, plan changes
on non-active lines).
"""

from __future__ import annotations

import json
import random
import string

from harness.tools.environment import ToolEnvironment, register_domain
from harness.tools.registry import ToolRegistry

# Maximum refund an agent may issue without supervisor escalation.
REFUND_LIMIT = 60.0


def _ref_id(prefix: str) -> str:
    return f"{prefix}-" + "".join(
        random.choices(string.ascii_uppercase + string.digits, k=6)
    )


@register_domain("telecom")
class TelecomEnvironment(ToolEnvironment):
    def default_state(self) -> dict:
        return {
            "plans": {
                "PLAN-BASIC": {
                    "plan_id": "PLAN-BASIC",
                    "name": "Basic 5GB",
                    "data_limit_gb": 5.0,
                    "price_per_month": 25.0,
                    "data_refueling_price_per_gb": 10.0,
                },
                "PLAN-PLUS": {
                    "plan_id": "PLAN-PLUS",
                    "name": "Plus 15GB",
                    "data_limit_gb": 15.0,
                    "price_per_month": 45.0,
                    "data_refueling_price_per_gb": 8.0,
                },
                "PLAN-UNLIMITED": {
                    "plan_id": "PLAN-UNLIMITED",
                    "name": "Unlimited",
                    "data_limit_gb": 100.0,
                    "price_per_month": 70.0,
                    "data_refueling_price_per_gb": 5.0,
                },
            },
            "customers": {
                "CUST-1": {
                    "customer_id": "CUST-1",
                    "full_name": "Alex Rivera",
                    "email": "alex.rivera@example.com",
                    "phone_number": "555-0101",
                    "account_status": "Active",
                    "line_ids": ["LINE-1", "LINE-2"],
                },
                "CUST-2": {
                    "customer_id": "CUST-2",
                    "full_name": "Morgan Chen",
                    "email": "morgan.chen@example.com",
                    "phone_number": "555-0202",
                    "account_status": "Active",
                    "line_ids": ["LINE-3"],
                },
                "CUST-3": {
                    "customer_id": "CUST-3",
                    "full_name": "Sam Okafor",
                    "email": "sam.okafor@example.com",
                    "phone_number": "555-0303",
                    "account_status": "Active",
                    "line_ids": ["LINE-4"],
                },
            },
            "lines": {
                "LINE-1": {
                    "line_id": "LINE-1",
                    "customer_id": "CUST-1",
                    "phone_number": "555-1001",
                    "status": "Active",
                    "plan_id": "PLAN-PLUS",
                    "device_id": "DEV-1",
                    "data_used_gb": 9.5,
                    "roaming_enabled": False,
                    "network_status": "Online",
                },
                "LINE-2": {
                    "line_id": "LINE-2",
                    "customer_id": "CUST-1",
                    "phone_number": "555-1002",
                    "status": "Suspended",
                    "plan_id": "PLAN-BASIC",
                    "device_id": "DEV-2",
                    "data_used_gb": 4.8,
                    "roaming_enabled": False,
                    "network_status": "No Signal",
                },
                "LINE-3": {
                    "line_id": "LINE-3",
                    "customer_id": "CUST-2",
                    "phone_number": "555-1003",
                    "status": "Active",
                    "plan_id": "PLAN-UNLIMITED",
                    "device_id": "DEV-3",
                    "data_used_gb": 42.0,
                    "roaming_enabled": True,
                    "network_status": "Online",
                },
                "LINE-4": {
                    "line_id": "LINE-4",
                    "customer_id": "CUST-3",
                    "phone_number": "555-1004",
                    "status": "Active",
                    "plan_id": "PLAN-BASIC",
                    "device_id": "DEV-4",
                    "data_used_gb": 2.1,
                    "roaming_enabled": False,
                    "network_status": "Degraded",
                },
            },
            "devices": {
                "DEV-1": {
                    "device_id": "DEV-1",
                    "line_id": "LINE-1",
                    "model": "Pixel 9",
                    "device_type": "phone",
                    "status": "Online",
                },
                "DEV-2": {
                    "device_id": "DEV-2",
                    "line_id": "LINE-2",
                    "model": "iPhone 13",
                    "device_type": "phone",
                    "status": "Offline",
                },
                "DEV-3": {
                    "device_id": "DEV-3",
                    "line_id": "LINE-3",
                    "model": "Galaxy S24",
                    "device_type": "phone",
                    "status": "Online",
                },
                "DEV-4": {
                    "device_id": "DEV-4",
                    "line_id": "LINE-4",
                    "model": "Router X200",
                    "device_type": "router",
                    "status": "Needs Replacement",
                },
            },
            "bills": {
                "BILL-1": {
                    "bill_id": "BILL-1",
                    "customer_id": "CUST-1",
                    "period": "2026-07",
                    "total_due": 45.0,
                    "due_date": "2026-07-20",
                    "status": "Paid",
                },
                "BILL-2": {
                    "bill_id": "BILL-2",
                    "customer_id": "CUST-1",
                    "period": "2026-08",
                    "total_due": 25.0,
                    "due_date": "2026-08-20",
                    "status": "Overdue",
                },
                "BILL-3": {
                    "bill_id": "BILL-3",
                    "customer_id": "CUST-2",
                    "period": "2026-08",
                    "total_due": 70.0,
                    "due_date": "2026-08-25",
                    "status": "Awaiting Payment",
                },
                "BILL-4": {
                    "bill_id": "BILL-4",
                    "customer_id": "CUST-3",
                    "period": "2026-08",
                    "total_due": 25.0,
                    "due_date": "2026-08-22",
                    "status": "Issued",
                },
                "BILL-5": {
                    "bill_id": "BILL-5",
                    "customer_id": "CUST-2",
                    "period": "2026-07",
                    "total_due": 70.0,
                    "due_date": "2026-07-25",
                    "status": "Paid",
                },
            },
        }

    def register_tools(self, registry: ToolRegistry) -> None:
        data = self.data

        def _line_for_customer(customer_id: str, line_id: str):
            if customer_id not in data["customers"]:
                return None, json.dumps({"error": f"Customer {customer_id} not found"})
            if line_id not in data["lines"]:
                return None, json.dumps({"error": f"Line {line_id} not found"})
            line = data["lines"][line_id]
            if line["customer_id"] != customer_id:
                return None, json.dumps(
                    {
                        "error": f"Line {line_id} does not belong to customer {customer_id}"
                    }
                )
            return line, None

        # -- Troubleshooting tools ------------------------------------------------

        @registry.tool("Get a line's status, plan, and network connectivity state")
        def get_line_status(line_id: str) -> str:
            if line_id not in data["lines"]:
                return json.dumps({"error": f"Line {line_id} not found"})
            return json.dumps(data["lines"][line_id])

        @registry.tool("Get a device's model, type, and current operating status")
        def get_device_status(device_id: str) -> str:
            if device_id not in data["devices"]:
                return json.dumps({"error": f"Device {device_id} not found"})
            return json.dumps(data["devices"][device_id])

        @registry.tool(
            "Run a network diagnostic on a line, reporting connectivity and device "
            "health and a recommended next action"
        )
        def run_network_diagnostic(line_id: str) -> str:
            if line_id not in data["lines"]:
                return json.dumps({"error": f"Line {line_id} not found"})
            line = data["lines"][line_id]
            device = data["devices"].get(line.get("device_id"), {})
            if line["status"] != "Active":
                recommendation = f"Line is {line['status']}; resume/reactivate the line before further diagnostics."
            elif device.get("status") == "Needs Replacement":
                recommendation = (
                    "Device hardware fault detected; dispatch a replacement device."
                )
            elif device.get("status") == "Offline":
                recommendation = "Device appears offline; try restart_device."
            elif line["network_status"] != "Online":
                recommendation = (
                    "Line network status is degraded; try reset_line_connection."
                )
            else:
                recommendation = "No issues detected."
            return json.dumps(
                {
                    "line_id": line_id,
                    "line_status": line["status"],
                    "network_status": line["network_status"],
                    "device_status": device.get("status", "Unknown"),
                    "recommendation": recommendation,
                }
            )

        @registry.tool("Reset a line's network connection to clear connectivity issues")
        def reset_line_connection(customer_id: str, line_id: str) -> str:
            line, err = _line_for_customer(customer_id, line_id)
            if err:
                return err
            if line["status"] != "Active":
                return json.dumps(
                    {
                        "error": (
                            f"Line {line_id} is {line['status']}; it must be Active "
                            "to reset its connection"
                        ),
                        "status_code": 422,
                    }
                )
            if line["network_status"] == "Online":
                return json.dumps(
                    {
                        "error": f"Line {line_id} already has an online connection",
                        "status_code": 409,
                    }
                )
            line["network_status"] = "Online"
            return json.dumps(
                {"status": "success", "line_id": line_id, "network_status": "Online"}
            )

        @registry.tool("Restart a device to resolve connectivity or performance issues")
        def restart_device(device_id: str) -> str:
            if device_id not in data["devices"]:
                return json.dumps({"error": f"Device {device_id} not found"})
            device = data["devices"][device_id]
            if device["status"] == "Needs Replacement":
                return json.dumps(
                    {
                        "error": (
                            f"Device {device_id} has a hardware fault; a restart will "
                            "not resolve this, dispatch a replacement instead"
                        ),
                        "status_code": 422,
                    }
                )
            if device["status"] == "Online":
                return json.dumps(
                    {
                        "error": f"Device {device_id} is already online",
                        "status_code": 409,
                    }
                )
            device["status"] = "Online"
            return json.dumps(
                {"status": "success", "device_id": device_id, "device_status": "Online"}
            )

        @registry.tool("Enable or disable international roaming on a line")
        def set_roaming(customer_id: str, line_id: str, enabled: bool) -> str:
            line, err = _line_for_customer(customer_id, line_id)
            if err:
                return err
            if line["status"] != "Active":
                return json.dumps(
                    {
                        "error": f"Line {line_id} is {line['status']}; it must be Active to change roaming",
                        "status_code": 422,
                    }
                )
            if line["roaming_enabled"] == enabled:
                return json.dumps(
                    {
                        "error": f"Roaming is already {'enabled' if enabled else 'disabled'} for line {line_id}",
                        "status_code": 409,
                    }
                )
            line["roaming_enabled"] = enabled
            return json.dumps(
                {"status": "success", "line_id": line_id, "roaming_enabled": enabled}
            )

        # -- Billing tools ----------------------------------------------------------

        @registry.tool("Look up a customer by customer ID")
        def get_customer(customer_id: str) -> str:
            if customer_id not in data["customers"]:
                return json.dumps({"error": f"Customer {customer_id} not found"})
            return json.dumps(data["customers"][customer_id])

        @registry.tool("Get a bill by bill ID")
        def get_bill(bill_id: str) -> str:
            if bill_id not in data["bills"]:
                return json.dumps({"error": f"Bill {bill_id} not found"})
            return json.dumps(data["bills"][bill_id])

        @registry.tool("List all bills belonging to a customer")
        def list_bills(customer_id: str) -> str:
            if customer_id not in data["customers"]:
                return json.dumps({"error": f"Customer {customer_id} not found"})
            bills = [
                b for b in data["bills"].values() if b["customer_id"] == customer_id
            ]
            return json.dumps({"customer_id": customer_id, "bills": bills})

        @registry.tool("Change a line's service plan")
        def change_plan(customer_id: str, line_id: str, new_plan_id: str) -> str:
            line, err = _line_for_customer(customer_id, line_id)
            if err:
                return err
            if new_plan_id not in data["plans"]:
                return json.dumps({"error": f"Plan {new_plan_id} not found"})
            if line["status"] != "Active":
                return json.dumps(
                    {
                        "error": f"Line {line_id} is {line['status']}; plan changes require an Active line",
                        "status_code": 422,
                    }
                )
            if line["plan_id"] == new_plan_id:
                return json.dumps(
                    {
                        "error": f"Line {line_id} is already on plan {new_plan_id}",
                        "status_code": 409,
                    }
                )
            line["plan_id"] = new_plan_id
            return json.dumps(
                {"status": "success", "line_id": line_id, "plan_id": new_plan_id}
            )

        @registry.tool("Record a payment against a customer's bill")
        def make_payment(customer_id: str, bill_id: str, amount: float) -> str:
            if customer_id not in data["customers"]:
                return json.dumps({"error": f"Customer {customer_id} not found"})
            if bill_id not in data["bills"]:
                return json.dumps({"error": f"Bill {bill_id} not found"})
            bill = data["bills"][bill_id]
            if bill["customer_id"] != customer_id:
                return json.dumps(
                    {
                        "error": f"Bill {bill_id} does not belong to customer {customer_id}"
                    }
                )
            if bill["status"] == "Paid":
                return json.dumps(
                    {
                        "error": f"Bill {bill_id} has already been paid",
                        "status_code": 409,
                    }
                )
            if round(amount, 2) != round(bill["total_due"], 2):
                return json.dumps(
                    {
                        "error": (
                            f"Payment amount {amount} does not match the amount due "
                            f"{bill['total_due']} for bill {bill_id}"
                        ),
                        "status_code": 422,
                    }
                )
            bill["status"] = "Paid"
            return json.dumps(
                {
                    "status": "success",
                    "bill_id": bill_id,
                    "amount_paid": amount,
                    "bill_status": "Paid",
                }
            )

        @registry.tool(
            "Process a refund against a paid bill, subject to the agent's authorization limit"
        )
        def process_refund(
            customer_id: str, bill_id: str, amount: float, reason: str = ""
        ) -> str:
            if customer_id not in data["customers"]:
                return json.dumps({"error": f"Customer {customer_id} not found"})
            if bill_id not in data["bills"]:
                return json.dumps({"error": f"Bill {bill_id} not found"})
            bill = data["bills"][bill_id]
            if bill["customer_id"] != customer_id:
                return json.dumps(
                    {
                        "error": f"Bill {bill_id} does not belong to customer {customer_id}"
                    }
                )
            if bill["status"] != "Paid":
                return json.dumps(
                    {
                        "error": f"Bill {bill_id} is {bill['status']}; only Paid bills can be refunded",
                        "status_code": 422,
                    }
                )
            if amount > bill["total_due"]:
                return json.dumps(
                    {
                        "error": f"Refund amount {amount} exceeds bill total {bill['total_due']}",
                        "status_code": 422,
                    }
                )
            if amount > REFUND_LIMIT:
                return json.dumps(
                    {
                        "error": (
                            f"Refund amount {amount} exceeds the agent authorization limit "
                            f"of ${REFUND_LIMIT}; escalate to a supervisor"
                        ),
                        "status_code": 403,
                    }
                )
            refund_id = _ref_id("REF")
            bill["status"] = "Refunded"
            bill["refund_id"] = refund_id
            bill["refund_reason"] = reason
            return json.dumps(
                {
                    "status": "success",
                    "refund_id": refund_id,
                    "bill_id": bill_id,
                    "amount": amount,
                    "reason": reason,
                }
            )
