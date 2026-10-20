"""Tests for the stateful tool environment."""

import json

import pytest

# Ensure domains are registered
import harness.tools.domains  # noqa: F401
from harness.tools.environment import (
    DOMAIN_REGISTRY,
    ToolEnvironment,
    _deep_merge,
    create_environment,
)
from harness.tools.state_validation import validate_state

# ---------------------------------------------------------------------------
# ToolEnvironment base class
# ---------------------------------------------------------------------------


class TestToolEnvironment:
    def test_default_state_is_empty(self):
        env = ToolEnvironment()
        assert env.data == {}

    def test_initial_state_override(self):
        env = ToolEnvironment(initial_state={"key": "value"})
        assert env.data == {"key": "value"}

    def test_get_state_returns_deep_copy(self):
        env = ToolEnvironment(initial_state={"nested": {"a": 1}})
        state = env.get_state()
        state["nested"]["a"] = 999
        assert env.data["nested"]["a"] == 1

    def test_create_registry_raises_not_implemented(self):
        env = ToolEnvironment()
        with pytest.raises(NotImplementedError):
            env.create_registry()


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_simple_merge(self):
        result = _deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_override_value(self):
        result = _deep_merge({"a": 1}, {"a": 2})
        assert result == {"a": 2}

    def test_nested_merge(self):
        base = {"x": {"a": 1, "b": 2}}
        override = {"x": {"b": 3, "c": 4}}
        result = _deep_merge(base, override)
        assert result == {"x": {"a": 1, "b": 3, "c": 4}}

    def test_does_not_mutate_base(self):
        base = {"x": {"a": 1}}
        _deep_merge(base, {"x": {"b": 2}})
        assert base == {"x": {"a": 1}}


# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------


class TestDomainRegistry:
    def test_customer_support_registered(self):
        assert "customer_support" in DOMAIN_REGISTRY

    def test_create_environment_customer_support(self):
        env = create_environment("customer_support")
        assert "invoices" in env.data
        assert "accounts" in env.data
        assert "tickets" in env.data

    def test_create_environment_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown domain"):
            create_environment("nonexistent_domain")

    def test_initial_state_merged_onto_defaults(self):
        env = create_environment(
            "customer_support",
            initial_state={
                "invoices": {"INV-001": {"amount": "$100", "status": "Paid"}},
            },
        )
        # Merged: invoices has INV-001, AND defaults like kb_articles still exist
        assert "INV-001" in env.data["invoices"]
        assert len(env.data["kb_articles"]) == 5


# ---------------------------------------------------------------------------
# State validation
# ---------------------------------------------------------------------------


class TestStateValidation:
    def test_matching_state(self):
        actual = {"invoices": {"INV-1": {"status": "Refunded"}}}
        expected = {"invoices": {"INV-1": {"status": "Refunded"}}}
        passed, failures = validate_state(actual, expected)
        assert passed
        assert failures == []

    def test_mismatched_value(self):
        actual = {"invoices": {"INV-1": {"status": "Paid"}}}
        expected = {"invoices": {"INV-1": {"status": "Refunded"}}}
        passed, failures = validate_state(actual, expected)
        assert not passed
        assert len(failures) == 1
        assert "Refunded" in failures[0]

    def test_missing_path(self):
        actual = {"invoices": {}}
        expected = {"invoices": {"INV-1": {"status": "Refunded"}}}
        passed, failures = validate_state(actual, expected)
        assert not passed
        assert "missing" in failures[0]

    def test_sparse_validation(self):
        actual = {
            "invoices": {
                "INV-1": {"status": "Refunded", "amount": "$100", "date": "2024-01-01"}
            },
            "tickets": {"TKT-1": {"status": "Open"}},
        }
        expected = {"invoices": {"INV-1": {"status": "Refunded"}}}
        passed, failures = validate_state(actual, expected)
        assert passed

    def test_min_length_pass(self):
        actual = {"search_history": [{"q": "a"}, {"q": "b"}, {"q": "c"}]}
        expected = {"search_history": {"_min_length": 3}}
        passed, failures = validate_state(actual, expected)
        assert passed

    def test_min_length_fail(self):
        actual = {"search_history": [{"q": "a"}]}
        expected = {"search_history": {"_min_length": 3}}
        passed, failures = validate_state(actual, expected)
        assert not passed
        assert "min length 3" in failures[0]

    def test_min_length_missing_path(self):
        actual = {}
        expected = {"search_history": {"_min_length": 1}}
        passed, failures = validate_state(actual, expected)
        assert not passed
        assert "missing" in failures[0]

    def test_min_length_not_a_collection(self):
        actual = {"search_history": "not a list"}
        expected = {"search_history": {"_min_length": 1}}
        passed, failures = validate_state(actual, expected)
        assert not passed
        assert "expected list or dict" in failures[0]

    def test_min_length_dict_pass(self):
        actual = {
            "tickets": {"TKT-001": {"id": "TKT-001"}, "TKT-002": {"id": "TKT-002"}}
        }
        expected = {"tickets": {"_min_length": 1}}
        passed, failures = validate_state(actual, expected)
        assert passed

    def test_min_length_dict_fail(self):
        actual = {"tickets": {}}
        expected = {"tickets": {"_min_length": 1}}
        passed, failures = validate_state(actual, expected)
        assert not passed
        assert "min length 1" in failures[0]


# ---------------------------------------------------------------------------
# Customer service domain
# ---------------------------------------------------------------------------


class TestCustomerSupportDomain:
    def setup_method(self):
        self.env = create_environment(
            "customer_support",
            initial_state={
                "accounts": {
                    "ACC-5521": {
                        "account_id": "ACC-5521",
                        "full_name": "Alex Chen",
                        "email": "alex@example.com",
                        "plan": "Enterprise",
                        "status": "Active",
                        "balance": "$0.00",
                    },
                },
                "invoices": {
                    "INV-8834": {
                        "invoice_id": "INV-8834",
                        "account_id": "ACC-5521",
                        "amount": "$247.50",
                        "status": "Paid",
                        "date": "2024-03-15",
                    },
                },
                "payments": {
                    "PAY-001": {
                        "account_id": "ACC-5521",
                        "amount": "$247.50",
                        "date": "2024-03-15",
                        "status": "Received",
                    },
                },
            },
        )
        self.registry = self.env.create_registry()

    @pytest.mark.asyncio
    async def test_get_invoice(self):
        result = await self.registry.execute("get_invoice", {"invoice_id": "INV-8834"})
        data = json.loads(result)
        assert data["invoice_id"] == "INV-8834"
        assert data["status"] == "Paid"

    @pytest.mark.asyncio
    async def test_get_invoice_not_found(self):
        result = await self.registry.execute("get_invoice", {"invoice_id": "INV-9999"})
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_process_refund_mutates_state(self):
        result = await self.registry.execute(
            "process_refund", {"invoice_id": "INV-8834"}
        )
        data = json.loads(result)
        assert data["status"] == "Processed"
        assert "refund_id" in data

        # State should be mutated
        assert self.env.data["invoices"]["INV-8834"]["status"] == "Refunded"

    @pytest.mark.asyncio
    async def test_process_refund_then_get_invoice(self):
        await self.registry.execute("process_refund", {"invoice_id": "INV-8834"})
        result = await self.registry.execute("get_invoice", {"invoice_id": "INV-8834"})
        data = json.loads(result)
        assert data["status"] == "Refunded"
        assert "refund_id" in data

    @pytest.mark.asyncio
    async def test_process_refund_already_refunded(self):
        await self.registry.execute("process_refund", {"invoice_id": "INV-8834"})
        result = await self.registry.execute(
            "process_refund", {"invoice_id": "INV-8834"}
        )
        data = json.loads(result)
        assert "error" in data
        assert "already refunded" in data["error"]

    @pytest.mark.asyncio
    async def test_create_ticket_adds_to_state(self):
        result = await self.registry.execute(
            "create_ticket",
            {
                "subject": "Dashboard broken",
                "priority": "high",
            },
        )
        data = json.loads(result)
        assert data["status"] == "Open"
        assert data["priority"] == "high"

        # Ticket should be in state
        ticket_id = data["ticket_id"]
        assert ticket_id in self.env.data["tickets"]

    @pytest.mark.asyncio
    async def test_check_payment(self):
        result = await self.registry.execute(
            "check_payment", {"account_id": "ACC-5521"}
        )
        data = json.loads(result)
        assert len(data["payments"]) == 1
        assert data["payments"][0]["amount"] == "$247.50"

    @pytest.mark.asyncio
    async def test_check_payment_not_found(self):
        result = await self.registry.execute(
            "check_payment", {"account_id": "ACC-0000"}
        )
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_get_account_info(self):
        result = await self.registry.execute(
            "get_account_info", {"account_id": "ACC-5521"}
        )
        data = json.loads(result)
        assert data["full_name"] == "Alex Chen"

    @pytest.mark.asyncio
    async def test_check_service_status(self):
        result = await self.registry.execute(
            "check_service_status", {"service": "Dashboard"}
        )
        data = json.loads(result)
        assert data["status"] == "Degraded Performance"

    @pytest.mark.asyncio
    async def test_check_service_status_case_insensitive(self):
        # Imperfect casing / whitespace must still resolve, and the canonical
        # service name is echoed back.
        for variant in ("dashboard", "DASHBOARD", "  Dashboard  "):
            result = await self.registry.execute(
                "check_service_status", {"service": variant}
            )
            data = json.loads(result)
            assert "error" not in data, f"{variant!r} should resolve"
            assert data["service"] == "Dashboard"
            assert data["status"] == "Degraded Performance"

    @pytest.mark.asyncio
    async def test_check_service_status_unknown_service_errors(self):
        result = await self.registry.execute(
            "check_service_status", {"service": "Nonexistent"}
        )
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_search_kb(self):
        result = await self.registry.execute("search_kb", {"query": "password"})
        data = json.loads(result)
        assert len(data["results"]) >= 1
        assert "Password" in data["results"][0]["title"]

    @pytest.mark.asyncio
    async def test_state_validation_after_refund(self):
        await self.registry.execute("process_refund", {"invoice_id": "INV-8834"})
        passed, failures = validate_state(
            self.env.get_state(),
            {"invoices": {"INV-8834": {"status": "Refunded"}}},
        )
        assert passed
        assert failures == []

    def test_tool_count(self):
        assert len(self.registry) == 25


# ---------------------------------------------------------------------------
# Cross-domain isolation
# ---------------------------------------------------------------------------


class TestCrossDomainIsolation:
    def test_different_tool_sets(self):
        cs_env = create_environment("customer_support")
        csm_env = create_environment("retail")
        cs_tools = cs_env.create_registry().tool_names
        csm_tools = csm_env.create_registry().tool_names
        assert set(cs_tools) != set(csm_tools)
        assert "get_invoice" in cs_tools
        assert "get_invoice" not in csm_tools
        assert "CSA_Knowledge_FAQ" in csm_tools
        assert "CSA_Knowledge_FAQ" not in cs_tools

    def test_independent_state(self):
        cs_env = create_environment("customer_support")
        csm_env = create_environment("retail")
        assert "invoices" in cs_env.data
        assert "invoices" not in csm_env.data
        assert "products" in csm_env.data
        assert "products" not in cs_env.data
