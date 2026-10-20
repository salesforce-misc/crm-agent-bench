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
                # Deletion approval is deliberately scoped to one ticket rather
                # than being a session-wide admin switch. The two preflight
                # records below are consumed by delete_ticket.
                "delete_authorizations": {},
                "dependency_checks": {},
            },
        }

    def register_tools(self, registry: ToolRegistry) -> None:
        data = self.data

        @registry.tool("Look up a ticket by ID")
        def get_ticket(ticket_id: str) -> str:
            if ticket_id not in data["tickets"]:
                return json.dumps(
                    {
                        "found": False,
                        "ticket_id": ticket_id,
                        "message": f"Ticket {ticket_id} not found",
                    }
                )
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
            "Delete a ticket after a clean dependency check and ticket-specific delete permission"
        )
        def delete_ticket(ticket_id: str) -> str:
            if ticket_id not in data["tickets"]:
                return json.dumps({"error": f"Ticket {ticket_id} not found"})
            ticket = data["tickets"][ticket_id]
            dependency_checks = data["session"].setdefault("dependency_checks", {})
            if ticket_id not in dependency_checks:
                return json.dumps(
                    {
                        "error": (
                            f"Ticket {ticket_id} requires a dependency check before deletion"
                        ),
                        "status_code": 409,
                    }
                )
            checked_open_dependencies = dependency_checks[ticket_id].get(
                "open_dependencies", []
            )
            current_open_dependencies = [
                dependency_id
                for dependency_id in ticket.get("dependencies", [])
                if data["tickets"].get(dependency_id, {}).get("status") != "Closed"
            ]
            if checked_open_dependencies or current_open_dependencies:
                return json.dumps(
                    {
                        "error": f"Ticket {ticket_id} has open dependencies and cannot be deleted",
                        "status_code": 409,
                        "open_dependencies": current_open_dependencies
                        or checked_open_dependencies,
                    }
                )
            authorizations = data["session"].setdefault("delete_authorizations", {})
            if ticket_id not in authorizations:
                return json.dumps(
                    {
                        "error": (
                            f"Insufficient ticket-specific delete permission for {ticket_id}"
                        ),
                        "status_code": 403,
                    }
                )
            del data["tickets"][ticket_id]
            dependency_checks.pop(ticket_id, None)
            authorizations.pop(ticket_id, None)
            return json.dumps({"status": "deleted", "ticket_id": ticket_id})

        @registry.tool(
            "Check a ticket's dependencies and the status of open dependencies"
        )
        def check_dependencies(ticket_id: str) -> str:
            if ticket_id not in data["tickets"]:
                return json.dumps({"error": f"Ticket {ticket_id} not found"})
            ticket = data["tickets"][ticket_id]
            dependencies = ticket.get("dependencies", [])
            open_dependencies = [
                {
                    "ticket_id": dependency_id,
                    "status": data["tickets"]
                    .get(dependency_id, {})
                    .get("status", "Unknown"),
                }
                for dependency_id in dependencies
                if data["tickets"].get(dependency_id, {}).get("status") != "Closed"
            ]
            data["session"].setdefault("dependency_checks", {})[ticket_id] = {
                "open_dependencies": [item["ticket_id"] for item in open_dependencies]
            }
            return json.dumps(
                {
                    "ticket_id": ticket_id,
                    "dependencies": dependencies,
                    "open_dependencies": open_dependencies,
                }
            )

        @registry.tool(
            "Check whether a user's role permits an action, optionally for a ticket"
        )
        def check_permission(user_id: str, action: str, ticket_id: str = "") -> str:
            if user_id not in data["users"]:
                return json.dumps({"error": f"User {user_id} not found"})
            if ticket_id and ticket_id not in data["tickets"]:
                return json.dumps({"error": f"Ticket {ticket_id} not found"})
            user = data["users"][user_id]
            allowed = action in user.get("permissions", [])
            if action == "delete" and ticket_id and allowed:
                data["session"].setdefault("delete_authorizations", {})[ticket_id] = (
                    user_id
                )
            return json.dumps(
                {
                    "user_id": user_id,
                    "action": action,
                    "ticket_id": ticket_id,
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
