# Ported from olly-styles/WorkBench, MIT. Reimplemented as pure-Python ToolEnvironment (no upstream code copied).
"""Workspace stateful tool domain.

Provides tools that read/write a shared state dict simulating an office
productivity workspace: an employee directory, an email inbox/outbox, a
shared calendar, and a project task board. Modeled after the tool groups in
olly-styles/WorkBench (email, calendar, project management), reimplemented
from scratch as closures over a plain dict rather than a pandas DataFrame.
"""

from __future__ import annotations

import json

from harness.tools.environment import ToolEnvironment, register_domain
from harness.tools.registry import ToolRegistry

CURRENT_TIME = "2026-09-11 09:00:00"
CURRENT_USER_EMAIL = "sam.rivera@atlas.com"

VALID_TASK_STATUSES = ["Backlog", "In Progress", "In Review", "Completed"]
VALID_TASK_BOARDS = ["Back end", "Front end", "Design"]
VALID_EVENT_FIELDS = ["title", "participant_email", "start_time", "duration_minutes"]
VALID_TASK_FIELDS = ["title", "assignee_email", "status", "due_date", "board"]

_SEARCH_RESULT_LIMIT = 5


def _is_valid_email(email: str) -> bool:
    return "@" in email and "." in email.split("@")[-1]


def _next_id(prefix: str, collection: dict) -> str:
    numbers = [
        int(key.split("-")[1])
        for key in collection
        if key.startswith(f"{prefix}-") and key.split("-")[1].isdigit()
    ]
    next_number = (max(numbers) + 1) if numbers else 1001
    return f"{prefix}-{next_number}"


def _minutes_since_midnight(hhmmss: str) -> int:
    hours, minutes, _seconds = hhmmss.split(":")
    return int(hours) * 60 + int(minutes)


def _events_overlap(
    a_start: str, a_duration: int, b_start: str, b_duration: int
) -> bool:
    """Return True if two same-day events (start "YYYY-MM-DD HH:MM:SS") overlap."""
    a_date, a_time = a_start.split(" ")
    b_date, b_time = b_start.split(" ")
    if a_date != b_date:
        return False
    a_from = _minutes_since_midnight(a_time)
    a_to = a_from + int(a_duration)
    b_from = _minutes_since_midnight(b_time)
    b_to = b_from + int(b_duration)
    return a_from < b_to and b_from < a_to


@register_domain("workspace")
class WorkspaceEnvironment(ToolEnvironment):
    def default_state(self) -> dict:
        return {
            "employees": {
                "sam.rivera@atlas.com": {
                    "email": "sam.rivera@atlas.com",
                    "name": "Sam Rivera",
                    "department": "Engineering Management",
                },
                "aisha.chen@atlas.com": {
                    "email": "aisha.chen@atlas.com",
                    "name": "Aisha Chen",
                    "department": "Backend Engineering",
                },
                "carlos.rodriguez@atlas.com": {
                    "email": "carlos.rodriguez@atlas.com",
                    "name": "Carlos Rodriguez",
                    "department": "Backend Engineering",
                },
                "fatima.khan@atlas.com": {
                    "email": "fatima.khan@atlas.com",
                    "name": "Fatima Khan",
                    "department": "Backend Engineering",
                },
                "leila.azizi@atlas.com": {
                    "email": "leila.azizi@atlas.com",
                    "name": "Leila Azizi",
                    "department": "Frontend Engineering",
                },
                "kofi.mensah@atlas.com": {
                    "email": "kofi.mensah@atlas.com",
                    "name": "Kofi Mensah",
                    "department": "Frontend Engineering",
                },
                "nadia.moreau@atlas.com": {
                    "email": "nadia.moreau@atlas.com",
                    "name": "Nadia Moreau",
                    "department": "Product Design",
                },
                "yuki.tanaka@atlas.com": {
                    "email": "yuki.tanaka@atlas.com",
                    "name": "Yuki Tanaka",
                    "department": "Product Design",
                },
            },
            "emails": {
                "EMAIL-1001": {
                    "email_id": "EMAIL-1001",
                    "folder": "inbox",
                    "sender": "carlos.rodriguez@atlas.com",
                    "recipient": "sam.rivera@atlas.com",
                    "subject": "Task update: Add authentication for email notification",
                    "body": (
                        "Sam, I completed the auth work for email notifications "
                        "ahead of schedule. Ready for review."
                    ),
                    "sent_datetime": "2026-09-08 09:15:00",
                    "protected": False,
                },
                "EMAIL-1002": {
                    "email_id": "EMAIL-1002",
                    "folder": "inbox",
                    "sender": "fatima.khan@atlas.com",
                    "recipient": "sam.rivera@atlas.com",
                    "subject": "Question about Q3 budget planning",
                    "body": "Do you have five minutes to go over the Q3 budget numbers?",
                    "sent_datetime": "2026-09-08 10:05:00",
                    "protected": False,
                },
                "EMAIL-1003": {
                    "email_id": "EMAIL-1003",
                    "folder": "outbox",
                    "sender": "sam.rivera@atlas.com",
                    "recipient": "nadia.moreau@atlas.com",
                    "subject": "Re: Design review feedback",
                    "body": "Thanks for the mockups, left a few comments inline.",
                    "sent_datetime": "2026-09-07 14:30:00",
                    "protected": False,
                },
                "EMAIL-1004": {
                    "email_id": "EMAIL-1004",
                    "folder": "inbox",
                    "sender": "legal-archive@atlas.com",
                    "recipient": "sam.rivera@atlas.com",
                    "subject": "Signed NDA - Project Falcon (compliance record)",
                    "body": "Attached is the fully executed NDA for Project Falcon.",
                    "sent_datetime": "2026-08-01 09:00:00",
                    "protected": True,
                },
                "EMAIL-1005": {
                    "email_id": "EMAIL-1005",
                    "folder": "inbox",
                    "sender": "kofi.mensah@atlas.com",
                    "recipient": "sam.rivera@atlas.com",
                    "subject": "Blocked on homepage alignment issue",
                    "body": "Hit a snag with a third-party API. Could use a brainstorm session.",
                    "sent_datetime": "2026-09-09 11:00:00",
                    "protected": False,
                },
                "EMAIL-1006": {
                    "email_id": "EMAIL-1006",
                    "folder": "outbox",
                    "sender": "sam.rivera@atlas.com",
                    "recipient": "aisha.chen@atlas.com",
                    "subject": "Sprint planning notes",
                    "body": "Notes from today's sprint planning are attached.",
                    "sent_datetime": "2026-09-06 16:00:00",
                    "protected": False,
                },
                "EMAIL-1007": {
                    "email_id": "EMAIL-1007",
                    "folder": "inbox",
                    "sender": "yuki.tanaka@atlas.com",
                    "recipient": "sam.rivera@atlas.com",
                    "subject": "Year-end performance review timeline",
                    "body": "Could you confirm the timeline for year-end reviews?",
                    "sent_datetime": "2026-09-05 13:20:00",
                    "protected": False,
                },
                "EMAIL-1008": {
                    "email_id": "EMAIL-1008",
                    "folder": "inbox",
                    "sender": "leila.azizi@atlas.com",
                    "recipient": "sam.rivera@atlas.com",
                    "subject": "Carousel animation demo ready",
                    "body": "The carousel animation demo is ready for your review.",
                    "sent_datetime": "2026-09-09 15:45:00",
                    "protected": False,
                },
            },
            "calendar_events": {
                "CAL-2001": {
                    "event_id": "CAL-2001",
                    "title": "Sprint planning",
                    "participant_email": "aisha.chen@atlas.com",
                    "start_time": "2026-09-12 09:00:00",
                    "duration_minutes": 60,
                    "locked": False,
                },
                "CAL-2002": {
                    "event_id": "CAL-2002",
                    "title": "1:1 with Kofi",
                    "participant_email": "kofi.mensah@atlas.com",
                    "start_time": "2026-09-12 13:00:00",
                    "duration_minutes": 30,
                    "locked": False,
                },
                "CAL-2003": {
                    "event_id": "CAL-2003",
                    "title": "Design review",
                    "participant_email": "nadia.moreau@atlas.com",
                    "start_time": "2026-09-14 11:00:00",
                    "duration_minutes": 45,
                    "locked": False,
                },
                "CAL-2004": {
                    "event_id": "CAL-2004",
                    "title": "All-Hands Q3 Review",
                    "participant_email": "yuki.tanaka@atlas.com",
                    "start_time": "2026-09-15 10:00:00",
                    "duration_minutes": 60,
                    "locked": True,
                },
                "CAL-2005": {
                    "event_id": "CAL-2005",
                    "title": "Backend sync",
                    "participant_email": "carlos.rodriguez@atlas.com",
                    "start_time": "2026-09-16 15:00:00",
                    "duration_minutes": 30,
                    "locked": False,
                },
                "CAL-2006": {
                    "event_id": "CAL-2006",
                    "title": "Performance review: Fatima",
                    "participant_email": "fatima.khan@atlas.com",
                    "start_time": "2026-09-17 14:00:00",
                    "duration_minutes": 30,
                    "locked": False,
                },
            },
            "project_tasks": {
                "TASK-3001": {
                    "task_id": "TASK-3001",
                    "title": "Add authentication for email notification",
                    "assignee_email": "carlos.rodriguez@atlas.com",
                    "status": "In Progress",
                    "due_date": "2026-09-20",
                    "board": "Back end",
                },
                "TASK-3002": {
                    "task_id": "TASK-3002",
                    "title": "Fix alignment issue in homepage",
                    "assignee_email": "kofi.mensah@atlas.com",
                    "status": "In Review",
                    "due_date": "2026-09-14",
                    "board": "Front end",
                },
                "TASK-3003": {
                    "task_id": "TASK-3003",
                    "title": "Add animation to carousel",
                    "assignee_email": "leila.azizi@atlas.com",
                    "status": "Completed",
                    "due_date": "2026-09-10",
                    "board": "Front end",
                },
                "TASK-3004": {
                    "task_id": "TASK-3004",
                    "title": "Optimize database query for search",
                    "assignee_email": "fatima.khan@atlas.com",
                    "status": "Backlog",
                    "due_date": "2026-09-25",
                    "board": "Back end",
                },
                "TASK-3005": {
                    "task_id": "TASK-3005",
                    "title": "Design onboarding flow mockups",
                    "assignee_email": "nadia.moreau@atlas.com",
                    "status": "In Progress",
                    "due_date": "2026-09-18",
                    "board": "Design",
                },
                "TASK-3006": {
                    "task_id": "TASK-3006",
                    "title": "Update Flask to latest version",
                    "assignee_email": "aisha.chen@atlas.com",
                    "status": "Completed",
                    "due_date": "2026-09-05",
                    "board": "Back end",
                },
            },
        }

    def register_tools(self, registry: ToolRegistry) -> None:
        data = self.data

        # -------------------------------------------------------------
        # Directory
        # -------------------------------------------------------------

        @registry.tool("Find an employee's email address by name")
        def find_employee_email(name: str) -> str:
            query = name.lower()
            results = [
                emp
                for emp in data["employees"].values()
                if query in emp["name"].lower()
            ]
            return json.dumps({"results": results})

        # -------------------------------------------------------------
        # Email
        # -------------------------------------------------------------

        @registry.tool(
            "Search emails by keyword across subject/body/sender/recipient, "
            "optionally filtered by folder ('inbox' or 'outbox')"
        )
        def search_emails(query: str = "", folder: str = "") -> str:
            words = query.lower().split()

            def matches(email: dict) -> bool:
                haystack = (
                    f"{email['subject']} {email['body']} "
                    f"{email['sender']} {email['recipient']}"
                ).lower()
                if not all(word in haystack for word in words):
                    return False
                return not (folder and email["folder"] != folder)

            results = [e for e in data["emails"].values() if matches(e)]
            results.sort(key=lambda e: e["sent_datetime"], reverse=True)
            return json.dumps({"results": results[:_SEARCH_RESULT_LIMIT]})

        @registry.tool("Read the full contents of an email by ID")
        def get_email(email_id: str) -> str:
            email = data["emails"].get(email_id)
            if email is None:
                return json.dumps({"error": f"Email {email_id} not found"})
            return json.dumps(email)

        @registry.tool("Send a new email to a recipient")
        def send_email(recipient: str, subject: str, body: str) -> str:
            if not _is_valid_email(recipient):
                return json.dumps(
                    {
                        "error": f"Invalid recipient email address: {recipient}",
                        "status_code": 422,
                    }
                )
            email_id = _next_id("EMAIL", data["emails"])
            data["emails"][email_id] = {
                "email_id": email_id,
                "folder": "outbox",
                "sender": CURRENT_USER_EMAIL,
                "recipient": recipient,
                "subject": subject,
                "body": body,
                "sent_datetime": CURRENT_TIME,
                "protected": False,
            }
            return json.dumps({"status": "sent", "email_id": email_id})

        @registry.tool("Reply to an existing email by ID with new body text")
        def reply_email(email_id: str, body: str) -> str:
            original = data["emails"].get(email_id)
            if original is None:
                return json.dumps({"error": f"Email {email_id} not found"})
            reply_to = (
                original["sender"]
                if original["folder"] == "inbox"
                else original["recipient"]
            )
            subject = original["subject"]
            if not subject.lower().startswith("re:"):
                subject = f"Re: {subject}"
            new_id = _next_id("EMAIL", data["emails"])
            data["emails"][new_id] = {
                "email_id": new_id,
                "folder": "outbox",
                "sender": CURRENT_USER_EMAIL,
                "recipient": reply_to,
                "subject": subject,
                "body": body,
                "sent_datetime": CURRENT_TIME,
                "protected": False,
            }
            return json.dumps(
                {"status": "sent", "email_id": new_id, "in_reply_to": email_id}
            )

        @registry.tool("Delete an email by ID")
        def delete_email(email_id: str) -> str:
            email = data["emails"].get(email_id)
            if email is None:
                return json.dumps({"error": f"Email {email_id} not found"})
            if email.get("protected"):
                return json.dumps(
                    {
                        "error": (
                            f"Email {email_id} is flagged as a compliance record "
                            "and cannot be deleted"
                        ),
                        "status_code": 403,
                    }
                )
            del data["emails"][email_id]
            return json.dumps({"status": "deleted", "email_id": email_id})

        # -------------------------------------------------------------
        # Calendar
        # -------------------------------------------------------------

        @registry.tool(
            "Search calendar events by keyword in title/participant email, "
            "optionally filtered by a start-time date range"
        )
        def search_events(
            query: str = "", date_min: str = "", date_max: str = ""
        ) -> str:
            query_lower = query.lower()
            results = [
                e
                for e in data["calendar_events"].values()
                if query_lower in e["title"].lower()
                or query_lower in e["participant_email"].lower()
            ]
            if date_min:
                results = [e for e in results if e["start_time"][:10] >= date_min]
            if date_max:
                results = [e for e in results if e["start_time"][:10] <= date_max]
            results.sort(key=lambda e: e["start_time"])
            return json.dumps({"results": results[:_SEARCH_RESULT_LIMIT]})

        @registry.tool("Create a new calendar event with a participant")
        def create_event(
            title: str, participant_email: str, start_time: str, duration_minutes: int
        ) -> str:
            if participant_email not in data["employees"]:
                return json.dumps(
                    {
                        "error": f"Unknown participant email: {participant_email}",
                        "status_code": 422,
                    }
                )
            for existing in data["calendar_events"].values():
                if existing["participant_email"] != participant_email:
                    continue
                if _events_overlap(
                    start_time,
                    duration_minutes,
                    existing["start_time"],
                    existing["duration_minutes"],
                ):
                    return json.dumps(
                        {
                            "error": (
                                f"{participant_email} already has a conflicting event "
                                f"'{existing['title']}' ({existing['event_id']})"
                            ),
                            "status_code": 409,
                        }
                    )
            event_id = _next_id("CAL", data["calendar_events"])
            data["calendar_events"][event_id] = {
                "event_id": event_id,
                "title": title,
                "participant_email": participant_email,
                "start_time": start_time,
                "duration_minutes": duration_minutes,
                "locked": False,
            }
            return json.dumps({"status": "created", "event_id": event_id})

        @registry.tool(
            "Update a field on an existing calendar event "
            f"(one of {VALID_EVENT_FIELDS})"
        )
        def update_event(event_id: str, field: str, new_value: str) -> str:
            event = data["calendar_events"].get(event_id)
            if event is None:
                return json.dumps({"error": f"Event {event_id} not found"})
            if event.get("locked"):
                return json.dumps(
                    {
                        "error": f"Event {event_id} is locked and cannot be modified",
                        "status_code": 403,
                    }
                )
            if field not in VALID_EVENT_FIELDS:
                return json.dumps(
                    {
                        "error": f"Field must be one of {VALID_EVENT_FIELDS}",
                        "status_code": 422,
                    }
                )
            if field == "participant_email" and new_value not in data["employees"]:
                return json.dumps(
                    {
                        "error": f"Unknown participant email: {new_value}",
                        "status_code": 422,
                    }
                )
            if field == "duration_minutes":
                event[field] = int(new_value)
            else:
                event[field] = new_value
            return json.dumps({"status": "updated", "event_id": event_id})

        @registry.tool("Delete a calendar event by ID")
        def delete_event(event_id: str) -> str:
            event = data["calendar_events"].get(event_id)
            if event is None:
                return json.dumps({"error": f"Event {event_id} not found"})
            if event.get("locked"):
                return json.dumps(
                    {
                        "error": f"Event {event_id} is locked and cannot be deleted",
                        "status_code": 403,
                    }
                )
            del data["calendar_events"][event_id]
            return json.dumps({"status": "deleted", "event_id": event_id})

        # -------------------------------------------------------------
        # Project tasks
        # -------------------------------------------------------------

        @registry.tool(
            "Create a new project task "
            f"(status one of {VALID_TASK_STATUSES}, board one of {VALID_TASK_BOARDS})"
        )
        def create_task(
            title: str,
            assignee_email: str,
            status: str = "Backlog",
            due_date: str = "",
            board: str = "Back end",
        ) -> str:
            if assignee_email not in data["employees"]:
                return json.dumps(
                    {
                        "error": f"Unknown assignee email: {assignee_email}",
                        "status_code": 422,
                    }
                )
            if status not in VALID_TASK_STATUSES:
                return json.dumps(
                    {
                        "error": f"Status must be one of {VALID_TASK_STATUSES}",
                        "status_code": 422,
                    }
                )
            if board not in VALID_TASK_BOARDS:
                return json.dumps(
                    {
                        "error": f"Board must be one of {VALID_TASK_BOARDS}",
                        "status_code": 422,
                    }
                )
            task_id = _next_id("TASK", data["project_tasks"])
            data["project_tasks"][task_id] = {
                "task_id": task_id,
                "title": title,
                "assignee_email": assignee_email,
                "status": status,
                "due_date": due_date,
                "board": board,
            }
            return json.dumps({"status": "created", "task_id": task_id})

        @registry.tool(
            "List project tasks, optionally filtered by assignee, status, or board"
        )
        def list_tasks(
            assignee_email: str = "", status: str = "", board: str = ""
        ) -> str:
            results = list(data["project_tasks"].values())
            if assignee_email:
                results = [t for t in results if t["assignee_email"] == assignee_email]
            if status:
                results = [t for t in results if t["status"] == status]
            if board:
                results = [t for t in results if t["board"] == board]
            return json.dumps({"results": results})

        @registry.tool(
            f"Update a field on an existing project task (one of {VALID_TASK_FIELDS})"
        )
        def update_task(task_id: str, field: str, new_value: str) -> str:
            task = data["project_tasks"].get(task_id)
            if task is None:
                return json.dumps({"error": f"Task {task_id} not found"})
            if task["status"] == "Completed":
                return json.dumps(
                    {
                        "error": f"Task {task_id} is completed and locked from further edits",
                        "status_code": 409,
                    }
                )
            if field not in VALID_TASK_FIELDS:
                return json.dumps(
                    {
                        "error": f"Field must be one of {VALID_TASK_FIELDS}",
                        "status_code": 422,
                    }
                )
            if field == "assignee_email" and new_value not in data["employees"]:
                return json.dumps(
                    {
                        "error": f"Unknown assignee email: {new_value}",
                        "status_code": 422,
                    }
                )
            if field == "status" and new_value not in VALID_TASK_STATUSES:
                return json.dumps(
                    {
                        "error": f"Status must be one of {VALID_TASK_STATUSES}",
                        "status_code": 422,
                    }
                )
            if field == "board" and new_value not in VALID_TASK_BOARDS:
                return json.dumps(
                    {
                        "error": f"Board must be one of {VALID_TASK_BOARDS}",
                        "status_code": 422,
                    }
                )
            task[field] = new_value
            return json.dumps({"status": "updated", "task_id": task_id})

        @registry.tool("Mark a project task as completed")
        def complete_task(task_id: str) -> str:
            task = data["project_tasks"].get(task_id)
            if task is None:
                return json.dumps({"error": f"Task {task_id} not found"})
            if task["status"] == "Completed":
                return json.dumps(
                    {
                        "error": f"Task {task_id} is already completed",
                        "status_code": 409,
                    }
                )
            task["status"] = "Completed"
            return json.dumps({"status": "completed", "task_id": task_id})

        @registry.tool("Delete a project task by ID")
        def delete_task(task_id: str) -> str:
            task = data["project_tasks"].get(task_id)
            if task is None:
                return json.dumps({"error": f"Task {task_id} not found"})
            if task["status"] == "Completed":
                return json.dumps(
                    {
                        "error": (
                            f"Task {task_id} is completed and retained for records; "
                            "it cannot be deleted"
                        ),
                        "status_code": 409,
                    }
                )
            del data["project_tasks"][task_id]
            return json.dumps({"status": "deleted", "task_id": task_id})
