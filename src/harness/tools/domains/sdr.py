"""Sales Development Rep (SDR) Agent stateful tool domain.

Provides tools simulating an SDR workflow: lead management, outreach emails,
lead response handling, person account warranty outreach, opportunity creation,
and opt-out management.
"""

from __future__ import annotations

import json
import random
import string

from harness.tools.environment import ToolEnvironment
from harness.tools.registry import ToolRegistry


def _lead_id() -> str:
    return f"00Q{''.join(random.choices(string.ascii_uppercase + string.digits, k=12))}"


def _opp_id() -> str:
    return f"006{''.join(random.choices(string.ascii_uppercase + string.digits, k=12))}"


class SDREnvironment(ToolEnvironment):
    def default_state(self) -> dict:
        return {
            "leads": {
                "lead-001": {
                    "id": "lead-001",
                    "first_name": "Alice",
                    "last_name": "Thompson",
                    "email": "alice.thompson@example.com",
                    "company": "TechCorp",
                    "status": "New",
                    "rating": "Warm",
                    "interest": "",
                    "product_interest": "Dell XPS 13 9310 2-in-1",
                },
                "lead-002": {
                    "id": "lead-002",
                    "first_name": "Marcus",
                    "last_name": "Chen",
                    "email": "marcus.chen@example.com",
                    "company": "Innovate.io",
                    "status": "Open",
                    "rating": "Hot",
                    "interest": "Enterprise laptop fleet",
                    "product_interest": "Lenovo ThinkPad X1 Carbon Gen 12 (2024)",
                },
                "lead-003": {
                    "id": "lead-003",
                    "first_name": "Sarah",
                    "last_name": "Mitchell",
                    "email": "sarah.mitchell@example.com",
                    "company": "DesignHub",
                    "status": "New",
                    "rating": "Warm",
                    "interest": "",
                    "product_interest": "Apple MacBook Air M4 (2025)",
                },
                "lead-004": {
                    "id": "lead-004",
                    "first_name": "James",
                    "last_name": "Rodriguez",
                    "email": "james.rodriguez@example.com",
                    "company": "StartupXYZ",
                    "status": "Working",
                    "rating": "Cold",
                    "interest": "Budget laptops for team",
                    "product_interest": "Acer Swift 5",
                },
                "lead-005": {
                    "id": "lead-005",
                    "first_name": "Emily",
                    "last_name": "Park",
                    "email": "emily.park@example.com",
                    "company": "GlobalFin",
                    "status": "New",
                    "rating": "Warm",
                    "interest": "",
                    "product_interest": 'HP Spectre x360 13.3" 2-in-1',
                },
            },
            "person_accounts": {
                "joy.hernandez@example.com": {
                    "first_name": "Joy",
                    "last_name": "Hernandez",
                    "email": "joy.hernandez@example.com",
                    "city": "Sullivantown",
                    "state": "New York",
                    "warranty_status": "Active",
                },
                "brian.gates@example.com": {
                    "first_name": "Brian",
                    "last_name": "Gates",
                    "email": "brian.gates@example.com",
                    "city": "Rachelland",
                    "state": "Wisconsin",
                    "warranty_status": "Active",
                },
                "samantha.butler@example.com": {
                    "first_name": "Samantha",
                    "last_name": "Butler",
                    "email": "samantha.butler@example.com",
                    "city": "Rachelburgh",
                    "state": "Massachusetts",
                    "warranty_status": "About to Expire",
                },
                "katherine.prince@example.com": {
                    "first_name": "Katherine",
                    "last_name": "Prince",
                    "email": "katherine.prince@example.com",
                    "city": "Gainesland",
                    "state": "Utah",
                    "warranty_status": "Active",
                },
                "donald.miller@example.com": {
                    "first_name": "Donald",
                    "last_name": "Miller",
                    "email": "donald.miller@example.com",
                    "city": "Port Joshua",
                    "state": "Kentucky",
                    "warranty_status": "Expired",
                },
                "hannah.hansen@example.com": {
                    "first_name": "Hannah",
                    "last_name": "Hansen",
                    "email": "hannah.hansen@example.com",
                    "city": "Williamsonmouth",
                    "state": "Arizona",
                    "warranty_status": "Active",
                },
                "robert.miller@example.com": {
                    "first_name": "Robert",
                    "last_name": "Miller",
                    "email": "robert.miller@example.com",
                    "city": "Melindaborough",
                    "state": "Kentucky",
                    "warranty_status": "About to Expire",
                },
                "samantha.jones@example.com": {
                    "first_name": "Samantha",
                    "last_name": "Jones",
                    "email": "samantha.jones@example.com",
                    "city": "South Tanya",
                    "state": "Massachusetts",
                    "warranty_status": "Active",
                },
            },
            "products": [
                {"name": "Warranty"},
                {"name": "Huawei MateBook X Pro"},
                {"name": "Samsung Galaxy Book3 Pro"},
                {"name": "LG Gram 17 (2024)"},
                {"name": "MSI Stealth 15M"},
                {"name": "Razer Blade 15"},
                {"name": "Microsoft Surface Laptop 5"},
                {"name": "Acer Swift 5"},
                {"name": "Asus ROG Zephyrus G14 (2024)"},
                {"name": "Lenovo ThinkPad X1 Carbon Gen 12 (2024)"},
                {"name": 'HP Spectre x360 13.3" 2-in-1'},
                {"name": "Dell XPS 13 9310 2-in-1"},
                {"name": "Apple MacBook Air M4 (2025)"},
            ],
            "opportunities": [],
            "emails_drafted": [],
            "emails_scheduled": [],
        }

    def _find_lead_by_email(self, email: str) -> dict | None:
        for lead in self.data["leads"].values():
            if lead["email"].lower() == email.lower():
                return lead
        return None

    def _find_lead_by_id(self, lead_id: str) -> dict | None:
        return self.data["leads"].get(lead_id)

    def register_tools(self, registry: ToolRegistry) -> None:
        env = self

        @registry.tool("Create a new lead record with first name, last name, and email")
        def MCA_Create_Lead_record(FirstName: str, LastName: str, Email: str) -> str:
            existing = env._find_lead_by_email(Email)
            if existing:
                return json.dumps(
                    {
                        "error": f"Lead with email {Email} already exists",
                        "lead_id": existing["id"],
                        "status_code": 409,
                    }
                )

            lead_id = _lead_id()
            lead = {
                "id": lead_id,
                "first_name": FirstName,
                "last_name": LastName,
                "email": Email,
                "company": "",
                "status": "New",
                "rating": "Warm",
                "interest": "",
                "product_interest": "",
            }
            env.data["leads"][lead_id] = lead
            return json.dumps(
                {
                    "status": "Success",
                    "lead_id": lead_id,
                    "message": f"Lead {FirstName} {LastName} created successfully.",
                }
            )

        @registry.tool("Update the status of a lead record by email")
        def MCA_Update_Lead_Status(Email: str, LeadStatus: str) -> str:
            lead = env._find_lead_by_email(Email)
            if not lead:
                return json.dumps({"error": f"Lead with email {Email} not found"})

            old_status = lead["status"]
            lead["status"] = LeadStatus
            return json.dumps(
                {
                    "status": "Success",
                    "message": f"Lead status updated from '{old_status}' to '{LeadStatus}' for {lead['first_name']} {lead['last_name']}.",
                }
            )

        @registry.tool(
            "Draft an initial outreach email for a lead with personalized product information"
        )
        def Sales_SDR_Draft_Initial_Outreach_Email_With_Products(lead_id: str) -> str:
            lead = env._find_lead_by_id(lead_id)
            if not lead:
                lead = env._find_lead_by_email(lead_id)
            if not lead:
                return json.dumps({"error": f"Lead {lead_id} not found"})

            product = lead.get("product_interest", "our premium laptop solutions")
            email_draft = {
                "subject": f"Exclusive Offer: {product} for {lead['company'] or 'your team'}",
                "body": (
                    f"Hi {lead['first_name']},\n\n"
                    f"I noticed your interest in premium technology solutions. "
                    f"I'd love to introduce you to the {product}, which could be a great fit "
                    f"for {lead['company'] or 'your needs'}.\n\n"
                    f"Would you be open to a brief conversation this week?\n\n"
                    f"Best regards,\nVanessa Central\nSales Development Representative"
                ),
                "recipient": lead["email"],
            }
            env.data["emails_drafted"].append(
                {"type": "lead_outreach", "lead_id": lead["id"], "draft": email_draft}
            )
            return json.dumps({"status": "Drafted", **email_draft})

        @registry.tool(
            "Draft an outreach email for a person account regarding warranty expiration"
        )
        def SDR_Draft_OutReach_Email_for_Account(email: str) -> str:
            account = env.data["person_accounts"].get(email)
            if not account:
                return json.dumps(
                    {"error": f"Person Account with email {email} not found"}
                )

            if account["warranty_status"] != "About to Expire":
                return json.dumps(
                    {
                        "error": f"Account warranty status is '{account['warranty_status']}', not 'About to Expire'"
                    }
                )

            email_draft = {
                "subject": f"Important: Your Warranty is About to Expire, {account['first_name']}",
                "body": (
                    f"Hi {account['first_name']},\n\n"
                    f"We wanted to let you know that your product warranty is about to expire. "
                    f"To ensure continued coverage and peace of mind, we recommend extending "
                    f"your warranty before it lapses.\n\n"
                    f"Would you like to discuss your warranty extension options?\n\n"
                    f"Best regards,\nVanessa Central\nSales Development Representative"
                ),
                "recipient": email,
            }
            env.data["emails_drafted"].append(
                {"type": "account_outreach", "email": email, "draft": email_draft}
            )
            return json.dumps({"status": "Drafted", **email_draft})

        @registry.tool(
            "Schedule a drafted email for sending at an optimal engagement time"
        )
        def SDR_Schedule_Email(recipient_email: str, subject: str = "") -> str:
            env.data["emails_scheduled"].append(
                {
                    "recipient": recipient_email,
                    "subject": subject,
                    "scheduled_time": "2025-04-08T09:00:00Z",
                }
            )
            return json.dumps(
                {
                    "status": "Scheduled",
                    "recipient": recipient_email,
                    "scheduled_time": "2025-04-08T09:00:00Z",
                    "message": "Email scheduled for optimal engagement time.",
                }
            )

        @registry.tool(
            "Update the lead interest field based on their email response content"
        )
        def SDR_Update_Lead_Interest(TargetID: str, EmailResponse: str) -> str:
            lead = env._find_lead_by_id(TargetID)
            if not lead:
                lead = env._find_lead_by_email(TargetID)
            if not lead:
                return json.dumps({"error": f"Lead {TargetID} not found"})

            lead["interest"] = EmailResponse[:200]
            return json.dumps(
                {
                    "status": "Success",
                    "message": f"Lead interest updated for {lead['first_name']} {lead['last_name']}.",
                }
            )

        @registry.tool("Update a person account's warranty status to 'About to Expire'")
        def SDR_Update_Warranty_Status_to_About_to_Expire(email: str) -> str:
            account = env.data["person_accounts"].get(email)
            if not account:
                return json.dumps(
                    {"error": f"Person Account with email {email} not found"}
                )

            if account["warranty_status"] == "Expired":
                return json.dumps(
                    {
                        "error": f"Account for {account['first_name']} is already marked as 'Expired'. Cannot update.",
                        "status_code": 409,
                    }
                )

            if account["warranty_status"] == "About to Expire":
                return json.dumps(
                    {
                        "error": f"Account for {account['first_name']} is already marked as 'About to Expire'.",
                        "status_code": 409,
                    }
                )

            account["warranty_status"] = "About to Expire"
            return json.dumps(
                {
                    "status": "Success",
                    "message": f"Warranty status updated to 'About to Expire' for {account['first_name']} {account['last_name']}.",
                }
            )

        @registry.tool(
            "Create a new opportunity record for a person account interested in warranty extension"
        )
        def SDR_Create_Opportunity_New(AccountName: str, accountId: str = "") -> str:
            # accountId is an optional alternate identifier in the tool contract;
            # this mock keys the created opportunity off AccountName only.
            opp_id = _opp_id()
            opportunity = {
                "id": opp_id,
                "account_name": AccountName,
                "name": f"Warranty Extension - {AccountName}",
                "stage": "Prospecting",
                "close_date": "2025-05-08",
            }
            env.data["opportunities"].append(opportunity)
            return json.dumps(
                {
                    "status": "Success",
                    "opportunity_id": opp_id,
                    "name": opportunity["name"],
                    "message": f"Opportunity created for {AccountName}.",
                }
            )

        @registry.tool(
            "Draft a warranty extension confirmation email for a person account"
        )
        def DraftWarrantyExtensionConfirmationEmail(email: str) -> str:
            account = env.data["person_accounts"].get(email)
            if not account:
                return json.dumps(
                    {"error": f"Person Account with email {email} not found"}
                )

            email_draft = {
                "subject": f"Warranty Extension Confirmation - {account['first_name']} {account['last_name']}",
                "body": (
                    f"Hi {account['first_name']},\n\n"
                    f"Thank you for your interest in extending your warranty! "
                    f"We've created a warranty extension opportunity for your account. "
                    f"Our team will reach out shortly with the next steps.\n\n"
                    f"Best regards,\nVanessa Central\nSales Development Representative"
                ),
                "recipient": email,
            }
            env.data["emails_drafted"].append(
                {"type": "warranty_confirmation", "email": email, "draft": email_draft}
            )
            return json.dumps({"status": "Drafted", **email_draft})

        @registry.tool("Opt out a lead by setting their rating to Cold")
        def Sales_SDR_Opt_Out_Target_Rating_as_Cold(leadId: str) -> str:
            lead = env._find_lead_by_id(leadId)
            if not lead:
                lead = env._find_lead_by_email(leadId)
            if not lead:
                return json.dumps({"error": f"Lead {leadId} not found"})

            lead["rating"] = "Cold"
            return json.dumps(
                {
                    "status": "Success",
                    "message": f"Lead {lead['first_name']} {lead['last_name']} has been opted out (rating set to Cold).",
                }
            )

        @registry.tool("Get product details and availability information")
        def SDR_Get_Product_Details(productName: str = "") -> str:
            if productName:
                matches = [
                    p
                    for p in env.data["products"]
                    if productName.lower() in p["name"].lower()
                ]
                if matches:
                    return json.dumps({"products": matches})
                return json.dumps({"error": f"Product '{productName}' not found"})
            return json.dumps({"products": env.data["products"]})
