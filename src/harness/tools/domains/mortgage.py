# Ported from awslabs/amazon-bedrock-agent-samples (mortgage_assistant), Apache-2.0.
# Reimplemented as pure-Python ToolEnvironment (no upstream code copied).
"""Mortgage stateful tool domain.

Provides tools simulating a mortgage servicing and origination platform:
customer profiles, existing (serviced) mortgages, in-progress loan
applications, a current rate table with historical trend, a required-
document checklist per application, payment estimation, rate locking, and
payoff quoting.
"""

from __future__ import annotations

import json
import random
import string
from datetime import date, timedelta

from harness.tools.environment import ToolEnvironment, register_domain
from harness.tools.registry import ToolRegistry

# Fixed anchor date for deterministic, reproducible rate-history generation.
# (Matches the benchmark's "current date" so seeded data reads as realistic.)
_ANCHOR_DATE = date(2026, 9, 11)

_VALID_DOCUMENT_TYPES = frozenset(
    {
        "proof_of_income",
        "employment_verification",
        "proof_of_assets",
        "credit_authorization",
        "homeowners_insurance",
        "appraisal_report",
    }
)

_CLOSED_APPLICATION_STATUSES = frozenset({"DENIED", "WITHDRAWN", "CLOSED"})


def _ref_id(prefix: str) -> str:
    return f"{prefix}-" + "".join(
        random.choices(string.ascii_uppercase + string.digits, k=6)
    )


@register_domain("mortgage")
class MortgageEnvironment(ToolEnvironment):
    def default_state(self) -> dict:
        return {
            "customers": {
                "CUST-3001": {
                    "customer_id": "CUST-3001",
                    "name": "Alice Nguyen",
                    "email": "alice.nguyen@example.com",
                },
                "CUST-3002": {
                    "customer_id": "CUST-3002",
                    "name": "Brian Cole",
                    "email": "brian.cole@example.com",
                },
                "CUST-3003": {
                    "customer_id": "CUST-3003",
                    "name": "Maria Gomez",
                    "email": "maria.gomez@example.com",
                },
                "CUST-3004": {
                    "customer_id": "CUST-3004",
                    "name": "David Kim",
                    "email": "david.kim@example.com",
                },
                "CUST-3005": {
                    "customer_id": "CUST-3005",
                    "name": "Sophia Turner",
                    "email": "sophia.turner@example.com",
                },
            },
            # Existing, already-closed/serviced mortgages, keyed by customer_id.
            "mortgages": {
                "CUST-3001": {
                    "customer_id": "CUST-3001",
                    "account_number": "MTG-70011",
                    "outstanding_principal": 312450.00,
                    "interest_rate": 5.75,
                    "original_issue_date": "2019-08-01",
                    "maturity_date": "2049-08-01",
                    "payments_remaining": 276,
                    "last_payment_date": "2026-08-28",
                    "next_payment_due": "2026-09-28",
                    "next_payment_amount": 1823.44,
                    "status": "Active",
                },
                "CUST-3002": {
                    "customer_id": "CUST-3002",
                    "account_number": "MTG-70012",
                    "outstanding_principal": 198320.10,
                    "interest_rate": 4.90,
                    "original_issue_date": "2021-02-15",
                    "maturity_date": "2036-02-15",
                    "payments_remaining": 114,
                    "last_payment_date": "2026-08-30",
                    "next_payment_due": "2026-09-30",
                    "next_payment_amount": 2110.02,
                    "status": "Active",
                },
                "CUST-3004": {
                    "customer_id": "CUST-3004",
                    "account_number": "MTG-70014",
                    "outstanding_principal": 0.0,
                    "interest_rate": 6.10,
                    "original_issue_date": "2015-03-01",
                    "maturity_date": "2025-03-01",
                    "payments_remaining": 0,
                    "last_payment_date": "2025-02-15",
                    "next_payment_due": None,
                    "next_payment_amount": 0.0,
                    "status": "Paid Off",
                },
                "CUST-3005": {
                    "customer_id": "CUST-3005",
                    "account_number": "MTG-70015",
                    "outstanding_principal": 421900.55,
                    "interest_rate": 7.25,
                    "original_issue_date": "2020-11-10",
                    "maturity_date": "2050-11-10",
                    "payments_remaining": 288,
                    "last_payment_date": "2026-07-10",
                    "next_payment_due": "2026-08-10",
                    "next_payment_amount": 2879.15,
                    "status": "Delinquent",
                },
            },
            "applications": {
                "APP-9001": {
                    "application_id": "APP-9001",
                    "customer_id": "CUST-3003",
                    "application_type": "NEW_MORTGAGE",
                    "application_date": "2026-08-10",
                    "application_status": "IN_PROGRESS",
                    "application_amount": 450000.0,
                    "application_term_years": 30,
                    "application_rate_type": "fixed",
                    "application_tentative_rate": 6.125,
                    "rate_locked": False,
                    "locked_rate": None,
                },
                "APP-9002": {
                    "application_id": "APP-9002",
                    "customer_id": "CUST-3005",
                    "application_type": "REFINANCE",
                    "application_date": "2026-08-20",
                    "application_status": "IN_PROGRESS",
                    "application_amount": 275000.0,
                    "application_term_years": 15,
                    "application_rate_type": "fixed",
                    "application_tentative_rate": 5.5,
                    "rate_locked": False,
                    "locked_rate": None,
                },
                "APP-9003": {
                    "application_id": "APP-9003",
                    "customer_id": "CUST-3001",
                    "application_type": "REFINANCE",
                    "application_date": "2026-07-01",
                    "application_status": "APPROVED",
                    "application_amount": 300000.0,
                    "application_term_years": 30,
                    "application_rate_type": "fixed",
                    "application_tentative_rate": 5.875,
                    "rate_locked": True,
                    "locked_rate": 5.875,
                },
                "APP-9004": {
                    "application_id": "APP-9004",
                    "customer_id": "CUST-3002",
                    "application_type": "NEW_MORTGAGE",
                    "application_date": "2026-06-15",
                    "application_status": "DENIED",
                    "application_amount": 900000.0,
                    "application_term_years": 30,
                    "application_rate_type": "fixed",
                    "application_tentative_rate": 6.5,
                    "rate_locked": False,
                    "locked_rate": None,
                },
                "APP-9005": {
                    "application_id": "APP-9005",
                    "customer_id": "CUST-3004",
                    "application_type": "NEW_MORTGAGE",
                    "application_date": "2026-05-01",
                    "application_status": "WITHDRAWN",
                    "application_amount": 500000.0,
                    "application_term_years": 30,
                    "application_rate_type": "fixed",
                    "application_tentative_rate": 6.75,
                    "rate_locked": False,
                    "locked_rate": None,
                },
            },
            # Current published rate table, keyed by loan_type.
            "rate_table": {
                "30-year-fixed": {
                    "loan_type": "30-year-fixed",
                    "rate": 6.35,
                    "as_of": "2026-09-08",
                },
                "15-year-fixed": {
                    "loan_type": "15-year-fixed",
                    "rate": 5.65,
                    "as_of": "2026-09-08",
                },
                "5/1-arm": {
                    "loan_type": "5/1-arm",
                    "rate": 5.85,
                    "as_of": "2026-09-08",
                },
                "7/1-arm": {
                    "loan_type": "7/1-arm",
                    "rate": 5.95,
                    "as_of": "2026-09-08",
                },
            },
            # Required-document checklists, keyed by application_id.
            "documents": {
                "APP-9001": [
                    {"document_type": "proof_of_income", "status": "MISSING"},
                    {"document_type": "employment_verification", "status": "MISSING"},
                    {"document_type": "proof_of_assets", "status": "COMPLETED"},
                    {"document_type": "credit_authorization", "status": "COMPLETED"},
                ],
                "APP-9002": [
                    {"document_type": "proof_of_income", "status": "COMPLETED"},
                    {"document_type": "homeowners_insurance", "status": "MISSING"},
                    {"document_type": "credit_authorization", "status": "COMPLETED"},
                ],
                "APP-9003": [
                    {"document_type": "proof_of_income", "status": "COMPLETED"},
                    {"document_type": "employment_verification", "status": "COMPLETED"},
                    {"document_type": "proof_of_assets", "status": "COMPLETED"},
                    {"document_type": "credit_authorization", "status": "COMPLETED"},
                ],
                "APP-9004": [
                    {"document_type": "proof_of_income", "status": "COMPLETED"},
                    {"document_type": "credit_authorization", "status": "COMPLETED"},
                ],
                "APP-9005": [
                    {"document_type": "proof_of_income", "status": "COMPLETED"},
                ],
            },
        }

    def register_tools(self, registry: ToolRegistry) -> None:
        data = self.data

        @registry.tool("Look up a mortgage customer's profile by customer ID")
        def get_customer(customer_id: str) -> str:
            if customer_id not in data["customers"]:
                return json.dumps({"error": f"Customer {customer_id} not found"})
            return json.dumps(data["customers"][customer_id])

        @registry.tool(
            "Get the details of a customer's existing (already-issued) mortgage"
        )
        def get_mortgage_status(customer_id: str) -> str:
            if customer_id not in data["customers"]:
                return json.dumps({"error": f"Customer {customer_id} not found"})
            if customer_id not in data["mortgages"]:
                return json.dumps(
                    {"error": f"No existing mortgage found for {customer_id}"}
                )
            return json.dumps(data["mortgages"][customer_id])

        @registry.tool("Get the status and details of a mortgage application by ID")
        def get_application_status(application_id: str) -> str:
            if application_id not in data["applications"]:
                return json.dumps({"error": f"Application {application_id} not found"})
            return json.dumps(data["applications"][application_id])

        @registry.tool("List all mortgage applications submitted by a customer")
        def list_customer_applications(customer_id: str) -> str:
            if customer_id not in data["customers"]:
                return json.dumps({"error": f"Customer {customer_id} not found"})
            apps = [
                app
                for app in data["applications"].values()
                if app["customer_id"] == customer_id
            ]
            return json.dumps({"customer_id": customer_id, "applications": apps})

        @registry.tool(
            "Get the current published mortgage rate table, optionally filtered "
            "to a single loan type (e.g. '30-year-fixed', '15-year-fixed', "
            "'5/1-arm', '7/1-arm')"
        )
        def get_current_rates(loan_type: str = "") -> str:
            if loan_type:
                if loan_type not in data["rate_table"]:
                    return json.dumps({"error": f"Unknown loan type {loan_type!r}"})
                return json.dumps(data["rate_table"][loan_type])
            return json.dumps({"rates": list(data["rate_table"].values())})

        @registry.tool(
            "Get recent historical daily rate trend for a loan type over the "
            "last N business days"
        )
        def get_rate_history(loan_type: str, day_count: int = 30) -> str:
            if loan_type not in data["rate_table"]:
                return json.dumps({"error": f"Unknown loan type {loan_type!r}"})
            if day_count <= 0:
                return json.dumps(
                    {
                        "error": "day_count must be a positive integer",
                        "status_code": 422,
                    }
                )
            base_rate = data["rate_table"][loan_type]["rate"]
            history = []
            day_offset = 1
            while len(history) < day_count:
                day = _ANCHOR_DATE - timedelta(days=day_offset)
                if day.weekday() < 5:  # skip weekends, business days only
                    drift = ((len(history) % 7) - 3) * 0.02
                    history.append(
                        {
                            "date": day.isoformat(),
                            "rate": round(base_rate + drift, 2),
                        }
                    )
                day_offset += 1
            return json.dumps({"loan_type": loan_type, "history": history})

        @registry.tool(
            "Estimate the monthly principal-and-interest payment for a given "
            "loan principal, annual interest rate (percent), and term in years"
        )
        def estimate_monthly_payment(
            principal: float, annual_rate: float, term_years: int
        ) -> str:
            if principal <= 0 or annual_rate < 0 or term_years <= 0:
                return json.dumps(
                    {
                        "error": (
                            "principal and term_years must be positive, and "
                            "annual_rate must be non-negative"
                        ),
                        "status_code": 422,
                    }
                )
            monthly_rate = annual_rate / 100 / 12
            num_payments = term_years * 12
            if monthly_rate == 0:
                monthly_payment = principal / num_payments
            else:
                monthly_payment = (
                    principal
                    * monthly_rate
                    * (1 + monthly_rate) ** num_payments
                    / ((1 + monthly_rate) ** num_payments - 1)
                )
            total_paid = monthly_payment * num_payments
            return json.dumps(
                {
                    "principal": principal,
                    "annual_rate": annual_rate,
                    "term_years": term_years,
                    "monthly_payment": round(monthly_payment, 2),
                    "total_paid": round(total_paid, 2),
                    "total_interest": round(total_paid - principal, 2),
                }
            )

        @registry.tool("Get the required-document checklist for a mortgage application")
        def get_required_documents(application_id: str) -> str:
            if application_id not in data["applications"]:
                return json.dumps({"error": f"Application {application_id} not found"})
            return json.dumps(
                {
                    "application_id": application_id,
                    "documents": data["documents"].get(application_id, []),
                }
            )

        @registry.tool(
            "Submit (mark completed) a required document for a mortgage application"
        )
        def submit_document(application_id: str, document_type: str) -> str:
            if application_id not in data["applications"]:
                return json.dumps({"error": f"Application {application_id} not found"})
            if document_type not in _VALID_DOCUMENT_TYPES:
                return json.dumps(
                    {
                        "error": f"Unknown document type {document_type!r}",
                        "status_code": 422,
                    }
                )
            app = data["applications"][application_id]
            if app["application_status"] in _CLOSED_APPLICATION_STATUSES:
                return json.dumps(
                    {
                        "error": (
                            f"Application {application_id} is "
                            f"{app['application_status']} and no longer "
                            f"accepts document submissions"
                        ),
                        "status_code": 409,
                    }
                )
            checklist = data["documents"].setdefault(application_id, [])
            entry = next(
                (d for d in checklist if d["document_type"] == document_type), None
            )
            if entry is not None:
                entry["status"] = "COMPLETED"
            else:
                checklist.append(
                    {"document_type": document_type, "status": "COMPLETED"}
                )
            return json.dumps(
                {
                    "application_id": application_id,
                    "document_type": document_type,
                    "status": "COMPLETED",
                }
            )

        @registry.tool(
            "Lock in an interest rate for a mortgage application, subject to "
            "eligibility rules"
        )
        def lock_rate(application_id: str, rate: float) -> str:
            if application_id not in data["applications"]:
                return json.dumps({"error": f"Application {application_id} not found"})
            app = data["applications"][application_id]
            if app["application_status"] in _CLOSED_APPLICATION_STATUSES:
                return json.dumps(
                    {
                        "error": (
                            f"Application {application_id} is "
                            f"{app['application_status']} and is not eligible "
                            f"for a rate lock"
                        ),
                        "status_code": 403,
                    }
                )
            if app.get("rate_locked"):
                return json.dumps(
                    {
                        "error": (
                            f"Application {application_id} already has a "
                            f"locked rate of {app.get('locked_rate')}"
                        ),
                        "status_code": 409,
                    }
                )
            checklist = data["documents"].get(application_id, [])
            incomplete = [
                d["document_type"] for d in checklist if d["status"] != "COMPLETED"
            ]
            if incomplete:
                return json.dumps(
                    {
                        "error": (
                            "Cannot lock rate until all required documents are "
                            f"completed. Outstanding: {incomplete}"
                        ),
                        "status_code": 422,
                    }
                )
            app["rate_locked"] = True
            app["locked_rate"] = rate
            return json.dumps(
                {
                    "application_id": application_id,
                    "rate_locked": True,
                    "locked_rate": rate,
                }
            )

        @registry.tool("Withdraw an in-progress or approved mortgage application")
        def withdraw_application(application_id: str) -> str:
            if application_id not in data["applications"]:
                return json.dumps({"error": f"Application {application_id} not found"})
            app = data["applications"][application_id]
            if app["application_status"] in _CLOSED_APPLICATION_STATUSES:
                return json.dumps(
                    {
                        "error": (
                            f"Application {application_id} is already "
                            f"{app['application_status']}"
                        ),
                        "status_code": 409,
                    }
                )
            app["application_status"] = "WITHDRAWN"
            return json.dumps(
                {"application_id": application_id, "application_status": "WITHDRAWN"}
            )

        @registry.tool(
            "Get a payoff quote for a customer's existing mortgage, including "
            "estimated accrued interest"
        )
        def get_payoff_quote(customer_id: str) -> str:
            if customer_id not in data["customers"]:
                return json.dumps({"error": f"Customer {customer_id} not found"})
            if customer_id not in data["mortgages"]:
                return json.dumps(
                    {"error": f"No existing mortgage found for {customer_id}"}
                )
            mortgage = data["mortgages"][customer_id]
            if mortgage["status"] == "Paid Off":
                return json.dumps(
                    {
                        "error": (
                            f"Mortgage {mortgage['account_number']} is already "
                            f"paid off; no payoff quote is needed"
                        ),
                        "status_code": 409,
                    }
                )
            principal = mortgage["outstanding_principal"]
            daily_interest = principal * (mortgage["interest_rate"] / 100) / 365
            payoff_amount = round(principal + daily_interest * 30, 2)
            quote_id = _ref_id("PQ")
            return json.dumps(
                {
                    "quote_id": quote_id,
                    "customer_id": customer_id,
                    "account_number": mortgage["account_number"],
                    "outstanding_principal": principal,
                    "estimated_payoff_amount": payoff_amount,
                    "good_through": (_ANCHOR_DATE + timedelta(days=30)).isoformat(),
                }
            )
