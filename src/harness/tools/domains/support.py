"""Advanced customer support stateful tool domain.

Provides 16 tools simulating an advanced customer support platform:
customer/account/order lookups, issue classification, knowledge-base
search, case management (creation and escalation), churn-risk and
customer-value analysis, retention recommendations, specialist
consultation/routing, and human handoff.
"""

from __future__ import annotations

import json
import random
import string
from typing import Literal

from harness.tools.environment import ToolEnvironment
from harness.tools.registry import ToolRegistry


def _ref_id() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


class SupportEnvironment(ToolEnvironment):
    def default_state(self) -> dict:
        return {
            "customers": {
                "CUST-1001": {
                    "customer_id": "CUST-1001",
                    "name": "Dana Whitfield",
                    "tier": "Enterprise",
                    "account_id": "ACC-2001",
                },
                "CUST-1002": {
                    "customer_id": "CUST-1002",
                    "name": "Marcus Yee",
                    "tier": "Standard",
                    "account_id": "ACC-2002",
                },
                "CUST-1003": {
                    "customer_id": "CUST-1003",
                    "name": "Priya Nadkarni",
                    "tier": "Premium",
                    "account_id": "ACC-2003",
                },
            },
            "accounts": {
                "ACC-2001": {
                    "account_id": "ACC-2001",
                    "plan": "Enterprise",
                    "mrr": 4200.0,
                    "tenure_months": 30,
                    "status": "Active",
                },
                "ACC-2002": {
                    "account_id": "ACC-2002",
                    "plan": "Standard",
                    "mrr": 99.0,
                    "tenure_months": 4,
                    "status": "Active",
                },
                "ACC-2003": {
                    "account_id": "ACC-2003",
                    "plan": "Premium",
                    "mrr": 650.0,
                    "tenure_months": 14,
                    "status": "Active",
                },
            },
            "orders": {
                "CUST-1001": [
                    {
                        "order_id": "ORD-5001",
                        "item": "Enterprise Seats (50)",
                        "amount": 4200.0,
                        "date": "2026-06-01",
                    },
                    {
                        "order_id": "ORD-5002",
                        "item": "Premium Support Add-on",
                        "amount": 500.0,
                        "date": "2026-08-01",
                    },
                ],
                "CUST-1002": [
                    {
                        "order_id": "ORD-5003",
                        "item": "Standard Plan Renewal",
                        "amount": 99.0,
                        "date": "2026-08-15",
                    },
                ],
                "CUST-1003": [
                    {
                        "order_id": "ORD-5004",
                        "item": "Premium Plan Upgrade",
                        "amount": 650.0,
                        "date": "2026-05-20",
                    },
                    {
                        "order_id": "ORD-5005",
                        "item": "API Rate Limit Increase",
                        "amount": 120.0,
                        "date": "2026-07-10",
                    },
                ],
            },
            "support_history": {
                "CUST-1001": [
                    {
                        "case_id": "CASE-OLD001",
                        "subject": "SSO login failures",
                        "resolution": "Resolved - SAML config fixed",
                        "date": "2026-07-02",
                    },
                ],
                "CUST-1002": [
                    {
                        "case_id": "CASE-OLD002",
                        "subject": "Billed twice in one month",
                        "resolution": "Refunded duplicate charge",
                        "date": "2026-08-20",
                    },
                    {
                        "case_id": "CASE-OLD003",
                        "subject": "Cannot export reports",
                        "resolution": "Escalated to engineering, unresolved",
                        "date": "2026-09-01",
                    },
                ],
                "CUST-1003": [],
            },
            "profiles": {
                "CUST-1001": {"segment": "Enterprise", "engagement_score": 82},
                "CUST-1002": {"segment": "SMB", "engagement_score": 34},
                "CUST-1003": {"segment": "Mid-Market", "engagement_score": 61},
            },
            "cases": {},
            "kb_articles": [
                {
                    "id": "KB-2001",
                    "title": "Troubleshooting SSO and Login Failures",
                    "summary": (
                        "Verify SAML/SSO configuration, check certificate expiry, "
                        "and confirm the identity provider metadata is up to date."
                    ),
                },
                {
                    "id": "KB-2002",
                    "title": "Understanding Your Invoice and Billing Cycle",
                    "summary": (
                        "Explains billing cycles, proration, duplicate charges, "
                        "and how to request a refund for a billing error."
                    ),
                },
                {
                    "id": "KB-2003",
                    "title": "General Account Setup Guide",
                    "summary": (
                        "Covers general account setup, user provisioning, and "
                        "basic navigation of the support portal."
                    ),
                },
            ],
        }

    def register_tools(self, registry: ToolRegistry) -> None:
        data = self.data

        @registry.tool("Fetch a customer record by customer ID")
        def fetch_customer(customer_id: str) -> str:
            if customer_id not in data["customers"]:
                return json.dumps({"error": f"Customer {customer_id} not found"})
            return json.dumps(data["customers"][customer_id])

        @registry.tool(
            "Heuristically classify a support issue description into a type and priority"
        )
        def classify_issue(description: str) -> str:
            text = description.lower()
            billing_kw = (
                "bill",
                "invoice",
                "charge",
                "refund",
                "payment",
                "subscription",
                "price",
            )
            technical_kw = (
                "error",
                "bug",
                "crash",
                "not working",
                "fail",
                "login",
                "sso",
                "api",
                "outage",
                "broken",
            )
            urgent_kw = (
                "urgent",
                "immediately",
                "asap",
                "down",
                "outage",
                "critical",
                "cannot access",
                "furious",
                "angry",
            )

            if any(kw in text for kw in billing_kw):
                issue_type = "billing"
            elif any(kw in text for kw in technical_kw):
                issue_type = "technical"
            else:
                issue_type = "general"

            if any(kw in text for kw in urgent_kw):
                priority = "high"
            elif issue_type in ("billing", "technical"):
                priority = "medium"
            else:
                priority = "low"

            return json.dumps({"type": issue_type, "priority": priority})

        @registry.tool("Search the support knowledge base for matching articles")
        def search_knowledge_base(query: str) -> str:
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

        @registry.tool("Create a support case for a customer")
        def create_case(
            customer_id: str,
            subject: str,
            priority: Literal["high", "medium", "low"] = "medium",
        ) -> str:
            if customer_id not in data["customers"]:
                return json.dumps({"error": f"Customer {customer_id} not found"})
            case_id = f"CASE-{_ref_id()}"
            case = {
                "case_id": case_id,
                "customer_id": customer_id,
                "subject": subject,
                "priority": priority,
                "status": "Open",
                "escalated_to": None,
            }
            data["cases"][case_id] = case
            return json.dumps(case)

        @registry.tool("Escalate an existing case to a specialist team")
        def escalate_case(case_id: str, team: str) -> str:
            if case_id not in data["cases"]:
                return json.dumps({"error": f"Case {case_id} not found"})
            case = data["cases"][case_id]
            case["status"] = "Escalated"
            case["escalated_to"] = team
            return json.dumps(case)

        @registry.tool("Fetch account data including plan, MRR, tenure, and status")
        def fetch_account_data(account_id: str) -> str:
            if account_id not in data["accounts"]:
                return json.dumps({"error": f"Account {account_id} not found"})
            return json.dumps(data["accounts"][account_id])

        @registry.tool("Fetch a customer's past order history")
        def fetch_order_history(customer_id: str) -> str:
            if customer_id not in data["customers"]:
                return json.dumps({"error": f"Customer {customer_id} not found"})
            return json.dumps({"orders": data["orders"].get(customer_id, [])})

        @registry.tool("Fetch a customer's past support interaction history")
        def fetch_support_history(customer_id: str) -> str:
            if customer_id not in data["customers"]:
                return json.dumps({"error": f"Customer {customer_id} not found"})
            return json.dumps({"history": data["support_history"].get(customer_id, [])})

        @registry.tool("Fetch a customer's profile (segment and engagement score)")
        def fetch_user_profile(customer_id: str) -> str:
            if customer_id not in data["profiles"]:
                return json.dumps({"error": f"Customer {customer_id} not found"})
            return json.dumps(data["profiles"][customer_id])

        @registry.tool(
            "Assess churn risk for a customer based on support history and engagement"
        )
        def assess_churn_risk(customer_id: str) -> str:
            if customer_id not in data["customers"]:
                return json.dumps({"error": f"Customer {customer_id} not found"})
            history = data["support_history"].get(customer_id, [])
            profile = data["profiles"].get(customer_id, {"engagement_score": 50})
            engagement = profile.get("engagement_score", 50)
            unresolved = sum(
                1 for h in history if "unresolved" in h.get("resolution", "").lower()
            )

            score = 100 - engagement + len(history) * 5 + unresolved * 15
            score = max(0, min(100, score))

            if score >= 60:
                risk = "high"
            elif score >= 30:
                risk = "medium"
            else:
                risk = "low"

            return json.dumps({"risk": risk, "score": score})

        @registry.tool(
            "Calculate a customer's lifetime value and value tier from account data"
        )
        def calculate_customer_value(customer_id: str) -> str:
            if customer_id not in data["customers"]:
                return json.dumps({"error": f"Customer {customer_id} not found"})
            account_id = data["customers"][customer_id].get("account_id")
            account = data["accounts"].get(account_id)
            if not account:
                return json.dumps({"error": f"Account for {customer_id} not found"})

            lifetime_value = round(account["mrr"] * account["tenure_months"], 2)
            if lifetime_value >= 20000:
                tier = "platinum"
            elif lifetime_value >= 5000:
                tier = "gold"
            elif lifetime_value >= 1000:
                tier = "silver"
            else:
                tier = "bronze"

            return json.dumps({"lifetime_value": lifetime_value, "tier": tier})

        @registry.tool(
            "Generate retention recommendations for a customer based on their profile"
        )
        def generate_recommendations(customer_id: str) -> str:
            if customer_id not in data["customers"]:
                return json.dumps({"error": f"Customer {customer_id} not found"})
            profile = data["profiles"].get(customer_id, {"engagement_score": 50})
            engagement = profile.get("engagement_score", 50)

            recommendations = []
            if engagement < 40:
                recommendations.append(
                    "Schedule a proactive check-in call to re-engage the customer."
                )
                recommendations.append(
                    "Offer a personalized onboarding or feature-adoption session."
                )
            else:
                recommendations.append(
                    "Highlight underused premium features in the next touchpoint."
                )
            recommendations.append(
                "Consider a loyalty discount or plan review at renewal time."
            )

            return json.dumps({"recommendations": recommendations})

        @registry.tool(
            "Consult a specialist for a given topic and question (simulated expert answer)"
        )
        def consult_specialist(topic: str, question: str) -> str:
            answer = (
                f"Specialist ({topic}) guidance: Based on standard {topic} "
                f"procedures, recommend reviewing the customer's recent "
                f"activity and applying the documented remediation steps for "
                f"'{question}'."
            )
            return json.dumps({"topic": topic, "question": question, "answer": answer})

        @registry.tool("Escalate the current conversation to a human support agent")
        def escalate_to_human(reason: str) -> str:
            return json.dumps(
                {"escalated": True, "reason": reason, "handoff": "human agent"}
            )

        @registry.tool("Analyze and diagnose a billing issue for a customer")
        def analyze_billing_issue(customer_id: str) -> str:
            if customer_id not in data["customers"]:
                return json.dumps({"error": f"Customer {customer_id} not found"})
            account_id = data["customers"][customer_id].get("account_id")
            account = data["accounts"].get(account_id, {})
            diagnosis = (
                f"Billing review for {customer_id}: plan={account.get('plan', 'unknown')}, "
                f"mrr={account.get('mrr', 'unknown')}, status={account.get('status', 'unknown')}. "
                "No unauthorized charges detected; verify recent invoices against orders."
            )
            return json.dumps({"customer_id": customer_id, "diagnosis": diagnosis})

        @registry.tool("Analyze and diagnose a technical issue for a customer")
        def analyze_technical_issue(customer_id: str) -> str:
            if customer_id not in data["customers"]:
                return json.dumps({"error": f"Customer {customer_id} not found"})
            history = data["support_history"].get(customer_id, [])
            diagnosis = (
                f"Technical review for {customer_id}: {len(history)} prior "
                "support interaction(s) on file. Recommend checking recent "
                "error logs, SSO/API configuration, and service status."
            )
            return json.dumps({"customer_id": customer_id, "diagnosis": diagnosis})
