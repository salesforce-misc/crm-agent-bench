"""Stub tool responses for YAML-defined tools.

When tools are defined in a YAML graph config (rather than registered
programmatically), they use these stub responses to return realistic
mock data. This enables benchmarking and testing without real backends.

To customize responses, either:
1. Override STUB_RESPONSES entries before building nodes
2. Register real tools programmatically via ToolRegistry
"""

from __future__ import annotations

import json
import random
import string

STUB_RESPONSES: dict[str, dict] = {
    "get_invoice": {
        "invoice_id": "{input}",
        "amount": "$247.50",
        "status": "Paid",
        "date": "2024-03-15",
        "items": "1x Enterprise License",
    },
    "check_payment": {
        "transaction_ref": "TXN-{ref_id}",
        "amount": "$247.50",
        "date": "2024-03-15",
        "status": "Received and Applied",
    },
    "process_refund": {
        "refund_id": "RFD-{ref_id}",
        "amount": "$247.50",
        "method": "Original payment method",
        "expected": "3-5 business days",
        "status": "Processed successfully",
    },
    "check_account_balance": {
        "account_id": "{input}",
        "current_balance": "$0.00",
        "last_payment": "$247.50 on 2024-03-15",
        "last_invoice": "INV-9921 — $247.50 (Paid)",
        "status": "Account is current, no outstanding balance",
        "note": "Balance display may take up to 24 hours to update after payment.",
    },
    "create_ticket": {
        "ticket_id": "TKT-{ref_id}",
        "priority": "P1 - High",
        "status": "Open",
        "assigned_to": "Support Engineering Team",
        "estimated_response": "Within 2 hours",
    },
    "search_kb": {
        "results": [
            {
                "id": "KB-1042",
                "title": "Password Reset Troubleshooting",
                "summary": "Try clearing browser cache and cookies, then use the 'Forgot Password' link. If SSO, contact your IT admin.",
            },
            {
                "id": "KB-1087",
                "title": "Account Lockout Policy",
                "summary": "Accounts lock after 5 failed attempts. Wait 30 minutes or contact support to unlock.",
            },
        ]
    },
    "check_service_status": {
        "service": "Dashboard",
        "status": "Degraded Performance",
        "incident_id": "INC-{ref_id}",
        "started": "2024-03-15 09:30 UTC",
        "affected": ["Dashboard", "Reporting"],
        "message": "Engineering team investigating. ETA for resolution: 1 hour.",
    },
    "get_account_info": {
        "account_id": "{input}",
        "full_name": "Morgan Davies",
        "email": "morgan.davies@example.com",
        "phone": "+1-555-0142",
        "billing_address": "123 Main St, Suite 400, San Francisco, CA 94102",
        "plan": "Enterprise",
        "status": "Active",
        "account_created": "2023-01-15",
        "last_login": "2024-03-14 09:22 UTC",
    },
    "get_datetime": {
        "datetime": "2024-03-15T14:30:00Z",
    },
}


def stub_response(tool_name: str, kwargs: dict, description: str = "") -> str:
    """Generate a realistic stub response for a tool call.

    Returns JSON string. Replaces {ref_id} with a random ID and
    {input} with the first kwarg value.
    """
    ref_id = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
    input_val = kwargs.get("kwargs", "") or ""

    if tool_name in STUB_RESPONSES:
        template = STUB_RESPONSES[tool_name]
        # Deep-replace placeholders in the template
        rendered = json.dumps(template)
        rendered = rendered.replace("{ref_id}", ref_id)
        rendered = rendered.replace("{input}", input_val or "unknown")
        return rendered

    return json.dumps({"status": "success", "ref": ref_id, "description": description})
