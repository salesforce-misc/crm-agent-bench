# Ported from openai/openai-cookbook (dispute_agent), MIT. Reimplemented as
# pure-Python ToolEnvironment (no upstream code copied).
"""Dispute (payment chargeback) stateful tool domain.

Provides tools that read/write a shared state dict simulating a payment
dispute / chargeback intake pipeline: customers, orders, phone-call logs,
email threads, and disputes (chargebacks) that must be investigated and
resolved via accept/reject/refund decisions.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from harness.tools.environment import ToolEnvironment, register_domain
from harness.tools.registry import ToolRegistry


def _now() -> str:
    return datetime.now(UTC).isoformat()


@register_domain("dispute")
class DisputeEnvironment(ToolEnvironment):
    def default_state(self) -> dict:
        return {
            "customers": {
                "CUST-01": {
                    "customer_id": "CUST-01",
                    "name": "Alex Rivera",
                    "email": "alex.rivera@example.com",
                    "phone": "+1-555-0101",
                },
                "CUST-02": {
                    "customer_id": "CUST-02",
                    "name": "Morgan Chase",
                    "email": "morgan.chase@example.com",
                    "phone": "+1-555-0102",
                },
                "CUST-03": {
                    "customer_id": "CUST-03",
                    "name": "Sam Okafor",
                    "email": "sam.okafor@example.com",
                    "phone": "+1-555-0103",
                },
                "CUST-04": {
                    "customer_id": "CUST-04",
                    "name": "Robin Patel",
                    "email": "robin.patel@example.com",
                    "phone": "+1-555-0104",
                },
            },
            "orders": {
                "ORD-2001": {
                    "order_id": "ORD-2001",
                    "customer_id": "CUST-01",
                    "product": "Espresso Machine",
                    "amount": 349.00,
                    "order_date": "2026-08-01",
                    "shipped": True,
                    "delivery_status": "Delivered",
                    "tracking_number": "TRK-8801",
                    "refunded_amount": 0.0,
                },
                "ORD-2002": {
                    "order_id": "ORD-2002",
                    "customer_id": "CUST-02",
                    "product": "Standing Desk",
                    "amount": 519.00,
                    "order_date": "2026-08-05",
                    "shipped": False,
                    "delivery_status": "Processing",
                    "tracking_number": None,
                    "refunded_amount": 0.0,
                },
                "ORD-2003": {
                    "order_id": "ORD-2003",
                    "customer_id": "CUST-03",
                    "product": "Noise-Cancelling Earbuds",
                    "amount": 129.00,
                    "order_date": "2026-07-20",
                    "shipped": True,
                    "delivery_status": "Delivered",
                    "tracking_number": "TRK-8802",
                    "refunded_amount": 0.0,
                },
                "ORD-2004": {
                    "order_id": "ORD-2004",
                    "customer_id": "CUST-01",
                    "product": "Air Fryer",
                    "amount": 89.99,
                    "order_date": "2026-08-10",
                    "shipped": True,
                    "delivery_status": "In Transit",
                    "tracking_number": "TRK-8803",
                    "refunded_amount": 0.0,
                },
                "ORD-2005": {
                    "order_id": "ORD-2005",
                    "customer_id": "CUST-04",
                    "product": "Gaming Monitor",
                    "amount": 429.00,
                    "order_date": "2026-06-15",
                    "shipped": True,
                    "delivery_status": "Delivered",
                    "tracking_number": "TRK-8804",
                    "refunded_amount": 429.00,
                },
                "ORD-2006": {
                    "order_id": "ORD-2006",
                    "customer_id": "CUST-02",
                    "product": "Blender",
                    "amount": 59.50,
                    "order_date": "2026-07-28",
                    "shipped": True,
                    "delivery_status": "Delivered",
                    "tracking_number": "TRK-8805",
                    "refunded_amount": 0.0,
                },
            },
            "disputes": {
                "DSP-3001": {
                    "dispute_id": "DSP-3001",
                    "order_id": "ORD-2001",
                    "customer_id": "CUST-01",
                    "amount": 349.00,
                    "reason": "product_not_received",
                    "status": "open",
                    "created_at": "2026-09-02",
                    "evidence_notes": [],
                    "resolution": None,
                    "resolved_at": None,
                },
                "DSP-3002": {
                    "dispute_id": "DSP-3002",
                    "order_id": "ORD-2002",
                    "customer_id": "CUST-02",
                    "amount": 519.00,
                    "reason": "duplicate_charge",
                    "status": "open",
                    "created_at": "2026-09-04",
                    "evidence_notes": [],
                    "resolution": None,
                    "resolved_at": None,
                },
                "DSP-3003": {
                    "dispute_id": "DSP-3003",
                    "order_id": "ORD-2003",
                    "customer_id": "CUST-03",
                    "amount": 129.00,
                    "reason": "product_unacceptable",
                    "status": "open",
                    "created_at": "2026-08-25",
                    "evidence_notes": [
                        "Customer called support describing defective unit."
                    ],
                    "resolution": None,
                    "resolved_at": None,
                },
                "DSP-3004": {
                    "dispute_id": "DSP-3004",
                    "order_id": "ORD-2005",
                    "customer_id": "CUST-04",
                    "amount": 429.00,
                    "reason": "product_not_received",
                    "status": "accepted",
                    "created_at": "2026-06-20",
                    "evidence_notes": ["Tracking showed no delivery scan."],
                    "resolution": "Refund issued in full.",
                    "resolved_at": "2026-06-25",
                },
                "DSP-3005": {
                    "dispute_id": "DSP-3005",
                    "order_id": "ORD-2006",
                    "customer_id": "CUST-02",
                    "amount": 59.50,
                    "reason": "fraudulent",
                    "status": "open",
                    "created_at": "2026-09-08",
                    "evidence_notes": [],
                    "resolution": None,
                    "resolved_at": None,
                },
            },
            "phone_logs": {
                "LOG-4001": {
                    "log_id": "LOG-4001",
                    "customer_id": "CUST-01",
                    "order_id": "ORD-2001",
                    "phone_number": "+1-555-0101",
                    "timestamp": "2026-09-01T14:12:00Z",
                    "duration_minutes": 6,
                    "notes": "Customer reports package never arrived despite delivered status.",
                },
                "LOG-4002": {
                    "log_id": "LOG-4002",
                    "customer_id": "CUST-02",
                    "order_id": "ORD-2002",
                    "phone_number": "+1-555-0102",
                    "timestamp": "2026-09-03T09:47:00Z",
                    "duration_minutes": 3,
                    "notes": "Customer noticed two charges of the same amount on statement.",
                },
                "LOG-4003": {
                    "log_id": "LOG-4003",
                    "customer_id": "CUST-03",
                    "order_id": "ORD-2003",
                    "phone_number": "+1-555-0103",
                    "timestamp": "2026-08-24T16:30:00Z",
                    "duration_minutes": 9,
                    "notes": "Customer says earbuds arrived with cracked casing.",
                },
                "LOG-4004": {
                    "log_id": "LOG-4004",
                    "customer_id": "CUST-02",
                    "order_id": "ORD-2006",
                    "phone_number": "+1-555-0102",
                    "timestamp": "2026-09-07T11:05:00Z",
                    "duration_minutes": 2,
                    "notes": "Customer states they never made this purchase.",
                },
            },
            "emails": {
                "EML-5001": {
                    "email_id": "EML-5001",
                    "customer_id": "CUST-01",
                    "order_id": "ORD-2001",
                    "email": "alex.rivera@example.com",
                    "subject": "Where is my espresso machine?",
                    "body": (
                        "I was charged for order ORD-2001 but tracking shows no "
                        "movement in 10 days. I never received it."
                    ),
                    "timestamp": "2026-09-01T10:02:00Z",
                },
                "EML-5002": {
                    "email_id": "EML-5002",
                    "customer_id": "CUST-02",
                    "order_id": "ORD-2002",
                    "email": "morgan.chase@example.com",
                    "subject": "Charged twice for standing desk",
                    "body": (
                        "My card statement shows two identical charges for "
                        "order ORD-2002. Please refund the duplicate."
                    ),
                    "timestamp": "2026-09-03T08:55:00Z",
                },
                "EML-5003": {
                    "email_id": "EML-5003",
                    "customer_id": "CUST-03",
                    "order_id": "ORD-2003",
                    "email": "sam.okafor@example.com",
                    "subject": "Earbuds arrived broken",
                    "body": (
                        "The noise-cancelling earbuds I ordered (ORD-2003) "
                        "arrived with a cracked case and do not power on."
                    ),
                    "timestamp": "2026-08-24T15:40:00Z",
                },
                "EML-5004": {
                    "email_id": "EML-5004",
                    "customer_id": "CUST-02",
                    "order_id": "ORD-2006",
                    "email": "morgan.chase@example.com",
                    "subject": "Unrecognized charge",
                    "body": (
                        "I do not recognize the charge for order ORD-2006. "
                        "I did not authorize this purchase."
                    ),
                    "timestamp": "2026-09-07T10:15:00Z",
                },
            },
        }

    def register_tools(self, registry: ToolRegistry) -> None:
        data = self.data

        @registry.tool("Look up an order by ID")
        def get_order(order_id: str) -> str:
            if order_id not in data["orders"]:
                return json.dumps({"error": f"Order {order_id} not found"})
            return json.dumps(data["orders"][order_id])

        @registry.tool("Look up a customer by ID")
        def get_customer(customer_id: str) -> str:
            if customer_id not in data["customers"]:
                return json.dumps({"error": f"Customer {customer_id} not found"})
            return json.dumps(data["customers"][customer_id])

        @registry.tool("Look up a dispute (chargeback) by ID")
        def get_dispute(dispute_id: str) -> str:
            if dispute_id not in data["disputes"]:
                return json.dumps({"error": f"Dispute {dispute_id} not found"})
            return json.dumps(data["disputes"][dispute_id])

        @registry.tool("Find the dispute (chargeback) associated with an order")
        def get_dispute_for_order(order_id: str) -> str:
            for dispute in data["disputes"].values():
                if dispute["order_id"] == order_id:
                    return json.dumps(dispute)
            return json.dumps({"error": f"No dispute found for order {order_id}"})

        @registry.tool(
            "List disputes, optionally filtered by status "
            "(open, escalated, accepted, rejected)"
        )
        def list_disputes(status: str = "") -> str:
            disputes = list(data["disputes"].values())
            if status:
                disputes = [d for d in disputes if d["status"] == status]
            return json.dumps({"disputes": disputes})

        @registry.tool("Get phone call logs for a customer")
        def get_phone_logs(customer_id: str) -> str:
            logs = [
                log
                for log in data["phone_logs"].values()
                if log["customer_id"] == customer_id
            ]
            return json.dumps({"phone_logs": logs})

        @registry.tool("Get email threads for a customer")
        def get_emails(customer_id: str) -> str:
            emails = [
                email
                for email in data["emails"].values()
                if email["customer_id"] == customer_id
            ]
            return json.dumps({"emails": emails})

        @registry.tool(
            "Attach an evidence note (e.g. summary of a call or email finding) "
            "to a dispute's investigation record"
        )
        def add_evidence_note(dispute_id: str, note: str) -> str:
            if dispute_id not in data["disputes"]:
                return json.dumps({"error": f"Dispute {dispute_id} not found"})
            dispute = data["disputes"][dispute_id]
            if dispute["status"] in ("accepted", "rejected"):
                return json.dumps(
                    {
                        "error": f"Dispute {dispute_id} is already closed",
                        "status_code": 409,
                    }
                )
            dispute["evidence_notes"].append(note)
            return json.dumps(dispute)

        @registry.tool(
            "Escalate a dispute for further review, attaching investigation notes"
        )
        def escalate_dispute(dispute_id: str, notes: str = "") -> str:
            if dispute_id not in data["disputes"]:
                return json.dumps({"error": f"Dispute {dispute_id} not found"})
            dispute = data["disputes"][dispute_id]
            if dispute["status"] in ("accepted", "rejected"):
                return json.dumps(
                    {
                        "error": f"Dispute {dispute_id} is already closed",
                        "status_code": 409,
                    }
                )
            if notes:
                dispute["evidence_notes"].append(notes)
            dispute["status"] = "escalated"
            return json.dumps(dispute)

        @registry.tool(
            "Accept a dispute (chargeback) in the customer's favor; requires "
            "the dispute to be open/escalated and to have gathered evidence"
        )
        def accept_dispute(dispute_id: str, reason: str) -> str:
            if dispute_id not in data["disputes"]:
                return json.dumps({"error": f"Dispute {dispute_id} not found"})
            dispute = data["disputes"][dispute_id]
            if dispute["status"] in ("accepted", "rejected"):
                return json.dumps(
                    {
                        "error": f"Dispute {dispute_id} is already closed",
                        "status_code": 409,
                    }
                )
            if not dispute["evidence_notes"]:
                return json.dumps(
                    {
                        "error": (
                            f"Dispute {dispute_id} cannot be accepted without "
                            "gathered evidence (phone logs, emails, or order "
                            "history)"
                        ),
                        "status_code": 422,
                    }
                )
            dispute["status"] = "accepted"
            dispute["resolution"] = reason
            dispute["resolved_at"] = _now()
            return json.dumps(dispute)

        @registry.tool(
            "Reject a dispute (chargeback), siding with the merchant; requires "
            "the dispute to be open/escalated and to have gathered evidence"
        )
        def reject_dispute(dispute_id: str, reason: str) -> str:
            if dispute_id not in data["disputes"]:
                return json.dumps({"error": f"Dispute {dispute_id} not found"})
            dispute = data["disputes"][dispute_id]
            if dispute["status"] in ("accepted", "rejected"):
                return json.dumps(
                    {
                        "error": f"Dispute {dispute_id} is already closed",
                        "status_code": 409,
                    }
                )
            if not dispute["evidence_notes"]:
                return json.dumps(
                    {
                        "error": (
                            f"Dispute {dispute_id} cannot be rejected without "
                            "gathered evidence (phone logs, emails, or order "
                            "history)"
                        ),
                        "status_code": 422,
                    }
                )
            dispute["status"] = "rejected"
            dispute["resolution"] = reason
            dispute["resolved_at"] = _now()
            return json.dumps(dispute)

        @registry.tool(
            "Issue a refund on an order, typically after accepting a dispute; "
            "amount cannot exceed the order total minus any prior refund"
        )
        def issue_refund(order_id: str, amount: float) -> str:
            if order_id not in data["orders"]:
                return json.dumps({"error": f"Order {order_id} not found"})
            order = data["orders"][order_id]
            remaining = round(order["amount"] - order["refunded_amount"], 2)
            if remaining <= 0:
                return json.dumps(
                    {
                        "error": f"Order {order_id} has already been fully refunded",
                        "status_code": 409,
                    }
                )
            if amount > remaining:
                return json.dumps(
                    {
                        "error": (
                            f"Refund amount {amount} exceeds remaining refundable "
                            f"balance {remaining} for order {order_id}"
                        ),
                        "status_code": 422,
                    }
                )
            order["refunded_amount"] = round(order["refunded_amount"] + amount, 2)
            return json.dumps(
                {
                    "order_id": order_id,
                    "refunded_amount": order["refunded_amount"],
                    "remaining_balance": round(
                        order["amount"] - order["refunded_amount"], 2
                    ),
                    "status": "refund_issued",
                }
            )
