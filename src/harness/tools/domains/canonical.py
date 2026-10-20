"""Canonical paper-benchmark domains.

The source environments remain separated by implementation lineage, while the
registered public domains follow the nine business boundaries frozen for v1.
Each composite exposes only tools that belong to that business domain.
"""

from __future__ import annotations

from harness.tools.domains.commerce import CommerceEnvironment
from harness.tools.domains.csm import CSMEnvironment
from harness.tools.domains.customer_service import CustomerServiceEnvironment
from harness.tools.domains.integration import IntegrationEnvironment
from harness.tools.domains.sdr import SDREnvironment
from harness.tools.domains.support import SupportEnvironment
from harness.tools.environment import _deep_merge, register_domain
from harness.tools.registry import ToolRegistry


def _merge_states(*states: dict) -> dict:
    merged: dict = {}
    for state in states:
        merged = _deep_merge(merged, state)
    return merged


@register_domain("retail")
class RetailEnvironment(CSMEnvironment, CommerceEnvironment, IntegrationEnvironment):
    """Products, inventory, carts, checkout, returns, and shipments."""

    def default_state(self) -> dict:
        integration = IntegrationEnvironment.default_state(self)
        return _merge_states(
            CommerceEnvironment.default_state(self),
            CSMEnvironment.default_state(self),
            {"shipments": integration["shipments"]},
        )

    def register_tools(self, registry: ToolRegistry) -> None:
        CommerceEnvironment.register_tools(self, registry)
        CSMEnvironment.register_tools(self, registry)
        IntegrationEnvironment.register_shipment_tools(self, registry)


@register_domain("customer_support")
class CustomerSupportEnvironment(CustomerServiceEnvironment, SupportEnvironment):
    """Billing, technical support, cases, escalation, and churn."""

    def default_state(self) -> dict:
        customer_service = CustomerServiceEnvironment.default_state(self)
        support = SupportEnvironment.default_state(self)
        merged = _merge_states(customer_service, support)
        merged["kb_articles"] = [
            *customer_service["kb_articles"],
            *support["kb_articles"],
        ]
        return merged

    def register_tools(self, registry: ToolRegistry) -> None:
        CustomerServiceEnvironment.register_tools(self, registry)
        SupportEnvironment.register_tools(self, registry)


@register_domain("sales")
class SalesEnvironment(SDREnvironment):
    """Lead management, outreach, warranty follow-up, and opportunities."""
