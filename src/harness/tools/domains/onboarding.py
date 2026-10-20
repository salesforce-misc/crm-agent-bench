"""Onboarding stateful tool domain.

Provides 8 tools that read/write a shared state dict simulating a customer
onboarding system: customer accounts (create/configure/verify/finalize) and
task lists (add/complete/list/stats) used for onboarding checklists.
"""

from __future__ import annotations

import json
import random
import string

from harness.tools.environment import ToolEnvironment, register_domain
from harness.tools.registry import ToolRegistry


def _ref_id() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


@register_domain("onboarding")
class OnboardingEnvironment(ToolEnvironment):
    def default_state(self) -> dict:
        return {
            "accounts": {
                "ACC-SEED01": {
                    "account_id": "ACC-SEED01",
                    "name": "Alice Johnson",
                    "email": "alice.johnson@example.com",
                    "plan": "starter",
                    "verified": True,
                    "configured": True,
                    "finalized": True,
                },
            },
            "task_lists": {
                "LIST-ONBOARD1": {
                    "list_id": "LIST-ONBOARD1",
                    "name": "New Hire Onboarding",
                    "tasks": {
                        "TSK-SEED01": {
                            "task_id": "TSK-SEED01",
                            "title": "Set up laptop",
                            "done": True,
                        },
                        "TSK-SEED02": {
                            "task_id": "TSK-SEED02",
                            "title": "Complete HR paperwork",
                            "done": False,
                        },
                        "TSK-SEED03": {
                            "task_id": "TSK-SEED03",
                            "title": "Schedule orientation",
                            "done": False,
                        },
                    },
                },
            },
        }

    def register_tools(self, registry: ToolRegistry) -> None:
        data = self.data

        @registry.tool("Create a new customer account")
        def create_customer_account(name: str, email: str) -> str:
            for acct in data["accounts"].values():
                if acct.get("email") == email:
                    return json.dumps(
                        {
                            "error": f"Account with email {email} already exists",
                            "status_code": 409,
                        }
                    )
            account_id = f"ACC-{_ref_id()}"
            account = {
                "account_id": account_id,
                "name": name,
                "email": email,
                "plan": None,
                "verified": False,
                "configured": False,
                "finalized": False,
            }
            data["accounts"][account_id] = account
            return json.dumps(account)

        @registry.tool("Configure account settings (plan) for a customer account")
        def configure_account_settings(account_id: str, plan: str) -> str:
            if account_id not in data["accounts"]:
                return json.dumps({"error": f"Account {account_id} not found"})
            acct = data["accounts"][account_id]
            acct["plan"] = plan
            acct["configured"] = True
            return json.dumps(acct)

        @registry.tool("Send a verification email and mark the account verified")
        def send_verification_email(account_id: str) -> str:
            if account_id not in data["accounts"]:
                return json.dumps({"error": f"Account {account_id} not found"})
            acct = data["accounts"][account_id]
            acct["verified"] = True
            return json.dumps(acct)

        @registry.tool("Finalize onboarding for a customer account")
        def finalize_onboarding(account_id: str) -> str:
            if account_id not in data["accounts"]:
                return json.dumps({"error": f"Account {account_id} not found"})
            acct = data["accounts"][account_id]
            missing = []
            if not acct.get("configured"):
                missing.append("account configuration")
            if not acct.get("verified"):
                missing.append("email verification")
            if missing:
                return json.dumps(
                    {
                        "error": (
                            "Cannot finalize onboarding: missing "
                            f"{' and '.join(missing)}"
                        ),
                        "status_code": 422,
                    }
                )
            acct["finalized"] = True
            return json.dumps(acct)

        @registry.tool("Add a task to a task list, creating the list if needed")
        def add_task(list_id: str, title: str) -> str:
            if list_id not in data["task_lists"]:
                data["task_lists"][list_id] = {
                    "list_id": list_id,
                    "name": list_id,
                    "tasks": {},
                }
            task_id = f"TSK-{_ref_id()}"
            task = {"task_id": task_id, "title": title, "done": False}
            data["task_lists"][list_id]["tasks"][task_id] = task
            return json.dumps(task)

        @registry.tool("Mark a task as complete by task ID")
        def complete_task(task_id: str) -> str:
            for task_list in data["task_lists"].values():
                if task_id in task_list["tasks"]:
                    task = task_list["tasks"][task_id]
                    task["done"] = True
                    return json.dumps(task)
            return json.dumps({"error": f"Task {task_id} not found"})

        @registry.tool("List all tasks in a task list")
        def list_tasks(list_id: str) -> str:
            if list_id not in data["task_lists"]:
                return json.dumps({"error": f"Task list {list_id} not found"})
            tasks = list(data["task_lists"][list_id]["tasks"].values())
            return json.dumps({"list_id": list_id, "tasks": tasks})

        @registry.tool("Calculate completion stats for a task list")
        def calculate_task_stats(list_id: str) -> str:
            if list_id not in data["task_lists"]:
                return json.dumps({"error": f"Task list {list_id} not found"})
            tasks = list(data["task_lists"][list_id]["tasks"].values())
            total = len(tasks)
            completed = sum(1 for t in tasks if t.get("done"))
            pending = total - completed
            percent_complete = round((completed / total) * 100, 2) if total else 0.0
            return json.dumps(
                {
                    "total": total,
                    "completed": completed,
                    "pending": pending,
                    "percent_complete": percent_complete,
                }
            )
