"""Customer service stateful tool domain.

Provides 9 tools that read/write a shared state dict simulating
a customer service database: accounts, invoices, tickets, payments,
knowledge base articles, and service statuses.
"""

from __future__ import annotations

import json
import random
import string

from harness.tools.environment import ToolEnvironment, register_domain
from harness.tools.registry import ToolRegistry


def _ref_id() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


@register_domain("customer_service")
class CustomerServiceEnvironment(ToolEnvironment):
    def default_state(self) -> dict:
        return {
            "accounts": {},
            "invoices": {},
            "tickets": {},
            "payments": {},
            "kb_articles": [
                {
                    "id": "KB-1042",
                    "title": "Password Reset Troubleshooting",
                    "summary": (
                        "Try clearing browser cache and cookies, then use the "
                        "'Forgot Password' link. If SSO, contact your IT admin."
                    ),
                },
                {
                    "id": "KB-1087",
                    "title": "Account Lockout Policy",
                    "summary": (
                        "Accounts lock after 5 failed attempts. Wait 30 minutes "
                        "or contact support to unlock."
                    ),
                },
            ],
            "services": {
                "Dashboard": {
                    "status": "Degraded Performance",
                    "message": "Engineering team investigating. ETA for resolution: 1 hour.",
                },
                "API": {
                    "status": "Operational",
                    "message": "All systems normal.",
                },
                "Reporting": {
                    "status": "Degraded Performance",
                    "message": "Linked to Dashboard incident.",
                },
            },
        }

    def register_tools(self, registry: ToolRegistry) -> None:
        data = self.data

        @registry.tool("Look up an invoice by ID")
        def get_invoice(invoice_id: str) -> str:
            if invoice_id not in data["invoices"]:
                return json.dumps({"error": f"Invoice {invoice_id} not found"})
            return json.dumps(data["invoices"][invoice_id])

        @registry.tool("Check payment status for an account")
        def check_payment(account_id: str) -> str:
            payments = [
                p
                for p in data["payments"].values()
                if p.get("account_id") == account_id
            ]
            if not payments:
                return json.dumps({"error": f"No payments found for {account_id}"})
            return json.dumps({"payments": payments})

        @registry.tool("Process a refund for an invoice")
        def process_refund(invoice_id: str, reason: str = "") -> str:
            if invoice_id not in data["invoices"]:
                return json.dumps({"error": f"Invoice {invoice_id} not found"})
            invoice = data["invoices"][invoice_id]
            if invoice.get("status") == "Refunded":
                return json.dumps(
                    {
                        "error": f"Invoice {invoice_id} already refunded",
                        "status_code": 409,
                    }
                )
            ref = f"RFD-{_ref_id()}"
            invoice["status"] = "Refunded"
            invoice["refund_id"] = ref
            return json.dumps(
                {
                    "refund_id": ref,
                    "invoice_id": invoice_id,
                    "amount": invoice.get("amount", "unknown"),
                    "status": "Processed",
                }
            )

        @registry.tool("Check current account balance")
        def check_account_balance(account_id: str) -> str:
            if account_id not in data["accounts"]:
                return json.dumps({"error": f"Account {account_id} not found"})
            acct = data["accounts"][account_id]
            return json.dumps(
                {
                    "account_id": account_id,
                    "current_balance": acct.get("balance", "$0.00"),
                    "status": acct.get("status", "Active"),
                }
            )

        @registry.tool("Create a support ticket")
        def create_ticket(subject: str, priority: str = "medium") -> str:
            ticket_id = f"TKT-{_ref_id()}"
            ticket = {
                "ticket_id": ticket_id,
                "subject": subject,
                "priority": priority,
                "status": "Open",
                "assigned_to": "Support Engineering Team",
            }
            data["tickets"][ticket_id] = ticket
            return json.dumps(ticket)

        @registry.tool("Search the knowledge base")
        def search_kb(query: str) -> str:
            query_lower = query.lower()
            results = [
                a
                for a in data["kb_articles"]
                if query_lower in a["title"].lower()
                or query_lower in a["summary"].lower()
            ]
            if not results:
                results = data["kb_articles"][:2]
            return json.dumps({"results": results})

        @registry.tool("Check service status")
        def check_service_status(service: str) -> str:
            # Match the service name case-insensitively (and trim whitespace) so
            # a correct intent with imperfect casing — e.g. "dashboard" instead
            # of "Dashboard" — resolves instead of erroring out. The canonical
            # name is echoed back in the result.
            requested = (service or "").strip().lower()
            canonical = next(
                (k for k in data["services"] if k.lower() == requested), None
            )
            if canonical is None:
                return json.dumps({"error": f"Unknown service: {service}"})
            svc = data["services"][canonical]
            return json.dumps({"service": canonical, **svc})

        @registry.tool("Look up account information")
        def get_account_info(account_id: str) -> str:
            if account_id not in data["accounts"]:
                return json.dumps({"error": f"Account {account_id} not found"})
            return json.dumps(data["accounts"][account_id])

        @registry.tool("Get current date and time")
        def get_datetime() -> str:
            return json.dumps({"datetime": "2024-03-15T14:30:00Z"})
