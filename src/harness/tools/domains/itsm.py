"""IT service management (ITSM) / ticketing stateful tool domain.

Provides 12 tools simulating an IT service desk: ticket lifecycle
management (assignment, escalation, status updates, comments), guarded
deletion (dependency and permission checks), and business-hours lookup.
"""

from __future__ import annotations

import json

from harness.tools.environment import ToolEnvironment, register_domain
from harness.tools.registry import ToolRegistry


@register_domain("itsm")
class ITSMEnvironment(ToolEnvironment):
    def default_state(self) -> dict:
        return {
            "tickets": {
                "TCK-1001": {
                    "ticket_id": "TCK-1001",
                    "subject": "VPN client fails to connect from home network",
                    "status": "Open",
                    "priority": "High",
                    "assigned_to": "agent.jsmith",
                    "dependencies": [],
                    "comments": [],
                    "escalated": False,
                },
                "TCK-1002": {
                    "ticket_id": "TCK-1002",
                    "subject": "New hire laptop provisioning",
                    "status": "In Progress",
                    "priority": "Medium",
                    "assigned_to": "agent.jsmith",
                    "dependencies": ["TCK-1004"],
                    "comments": [
                        "Waiting on asset tag from procurement.",
                    ],
                    "escalated": False,
                },
                "TCK-1003": {
                    "ticket_id": "TCK-1003",
                    "subject": "Shared printer on 3rd floor offline",
                    "status": "Closed",
                    "priority": "Low",
                    "assigned_to": "admin.rlee",
                    "dependencies": [],
                    "comments": [
                        "Replaced network cable, printer back online.",
                    ],
                    "escalated": False,
                },
                "TCK-1004": {
                    "ticket_id": "TCK-1004",
                    "subject": "Procurement request for new-hire hardware",
                    "status": "Open",
                    "priority": "Medium",
                    "assigned_to": "",
                    "dependencies": [],
                    "comments": [],
                    "escalated": False,
                },
            },
            "users": {
                "admin.rlee": {
                    "user_id": "admin.rlee",
                    "role": "admin",
                    "permissions": ["view", "edit", "assign", "delete", "escalate"],
                },
                "agent.jsmith": {
                    "user_id": "agent.jsmith",
                    "role": "agent",
                    "permissions": ["view", "edit", "assign", "escalate"],
                },
            },
            "session": {
                "can_delete": False,
            },
        }

    def register_tools(self, registry: ToolRegistry) -> None:
        data = self.data

        @registry.tool("Look up a ticket by ID")
        def get_ticket(ticket_id: str) -> str:
            if ticket_id not in data["tickets"]:
                return json.dumps({"error": f"Ticket {ticket_id} not found"})
            return json.dumps(data["tickets"][ticket_id])

        @registry.tool("List all tickets, optionally filtered by status")
        def list_tickets(status: str = "") -> str:
            tickets = list(data["tickets"].values())
            if status:
                status_lower = status.lower()
                tickets = [
                    t for t in tickets if t.get("status", "").lower() == status_lower
                ]
            return json.dumps({"tickets": tickets})

        @registry.tool("Assign a ticket to a user")
        def assign_ticket(ticket_id: str, assignee: str) -> str:
            if ticket_id not in data["tickets"]:
                return json.dumps({"error": f"Ticket {ticket_id} not found"})
            if assignee not in data["users"]:
                return json.dumps({"error": f"Unknown user: {assignee}"})
            ticket = data["tickets"][ticket_id]
            ticket["assigned_to"] = assignee
            return json.dumps(ticket)

        @registry.tool("Reassign an already-assigned ticket to a different user")
        def reassign_ticket(ticket_id: str, assignee: str) -> str:
            if ticket_id not in data["tickets"]:
                return json.dumps({"error": f"Ticket {ticket_id} not found"})
            if assignee not in data["users"]:
                return json.dumps({"error": f"Unknown user: {assignee}"})
            ticket = data["tickets"][ticket_id]
            if not ticket.get("assigned_to"):
                return json.dumps(
                    {
                        "error": f"Ticket {ticket_id} is not currently assigned; use assign_ticket instead",
                        "status_code": 422,
                    }
                )
            ticket["assigned_to"] = assignee
            return json.dumps(ticket)

        @registry.tool("Close a ticket")
        def close_ticket(ticket_id: str) -> str:
            if ticket_id not in data["tickets"]:
                return json.dumps({"error": f"Ticket {ticket_id} not found"})
            ticket = data["tickets"][ticket_id]
            if ticket.get("status") == "Closed":
                return json.dumps(
                    {
                        "error": f"Ticket {ticket_id} is already Closed",
                        "status_code": 409,
                    }
                )
            ticket["status"] = "Closed"
            return json.dumps(ticket)

        @registry.tool("Escalate a ticket to Critical priority")
        def escalate_ticket(ticket_id: str) -> str:
            if ticket_id not in data["tickets"]:
                return json.dumps({"error": f"Ticket {ticket_id} not found"})
            ticket = data["tickets"][ticket_id]
            ticket["priority"] = "Critical"
            ticket["escalated"] = True
            return json.dumps(ticket)

        @registry.tool("Update the status of a ticket")
        def update_ticket_status(ticket_id: str, status: str) -> str:
            if ticket_id not in data["tickets"]:
                return json.dumps({"error": f"Ticket {ticket_id} not found"})
            ticket = data["tickets"][ticket_id]
            ticket["status"] = status
            return json.dumps(ticket)

        @registry.tool("Add a comment to a ticket")
        def add_ticket_comment(ticket_id: str, comment: str) -> str:
            if ticket_id not in data["tickets"]:
                return json.dumps({"error": f"Ticket {ticket_id} not found"})
            ticket = data["tickets"][ticket_id]
            ticket["comments"].append(comment)
            return json.dumps(ticket)

        @registry.tool(
            "Delete a ticket (requires no open dependencies and delete permission)"
        )
        def delete_ticket(ticket_id: str) -> str:
            if ticket_id not in data["tickets"]:
                return json.dumps({"error": f"Ticket {ticket_id} not found"})
            ticket = data["tickets"][ticket_id]
            if ticket.get("dependencies"):
                return json.dumps(
                    {
                        "error": f"Ticket {ticket_id} has open dependencies and cannot be deleted",
                        "status_code": 409,
                    }
                )
            if not data["session"].get("can_delete"):
                return json.dumps(
                    {
                        "error": "Insufficient permission to delete tickets",
                        "status_code": 403,
                    }
                )
            del data["tickets"][ticket_id]
            return json.dumps({"status": "deleted", "ticket_id": ticket_id})

        @registry.tool("Check the dependent ticket IDs for a ticket")
        def check_dependencies(ticket_id: str) -> str:
            if ticket_id not in data["tickets"]:
                return json.dumps({"error": f"Ticket {ticket_id} not found"})
            ticket = data["tickets"][ticket_id]
            return json.dumps(
                {
                    "ticket_id": ticket_id,
                    "dependencies": ticket.get("dependencies", []),
                }
            )

        @registry.tool("Check whether a user's role permits a given action")
        def check_permission(user_id: str, action: str) -> str:
            if user_id not in data["users"]:
                return json.dumps({"error": f"User {user_id} not found"})
            user = data["users"][user_id]
            allowed = action in user.get("permissions", [])
            if action == "delete" and user.get("role") == "admin":
                data["session"]["can_delete"] = True
            return json.dumps(
                {
                    "user_id": user_id,
                    "action": action,
                    "role": user.get("role"),
                    "allowed": allowed,
                }
            )

        @registry.tool("Get current business hours and whether the desk is open now")
        def get_business_hours() -> str:
            return json.dumps(
                {
                    "open": "09:00",
                    "close": "17:00",
                    "timezone": "America/Los_Angeles",
                    "currently_open": True,
                }
            )
