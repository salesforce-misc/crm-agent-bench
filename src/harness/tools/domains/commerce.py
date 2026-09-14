"""Commerce stateful tool domain.

Provides 9 tools that read/write a shared state dict simulating an online
retail storefront: products, shopping carts, orders, customers, and an
authentication-gated returns flow.
"""

from __future__ import annotations

import json
import random
import string

from harness.tools.environment import ToolEnvironment, register_domain
from harness.tools.registry import ToolRegistry


def _ref_id() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


@register_domain("commerce")
class CommerceEnvironment(ToolEnvironment):
    def default_state(self) -> dict:
        return {
            "products": {
                "PROD-1001": {
                    "id": "PROD-1001",
                    "name": "Wireless Noise-Cancelling Headphones",
                    "category": "Audio",
                    "price": 249.99,
                    "description": (
                        "Over-ear Bluetooth headphones with active noise "
                        "cancellation and 30-hour battery life."
                    ),
                    "in_stock": True,
                },
                "PROD-1002": {
                    "id": "PROD-1002",
                    "name": "Smart Fitness Watch",
                    "category": "Wearables",
                    "price": 179.99,
                    "description": (
                        "Fitness tracker with heart-rate monitoring, GPS, and "
                        "a 7-day battery life."
                    ),
                    "in_stock": True,
                },
                "PROD-1003": {
                    "id": "PROD-1003",
                    "name": "Portable Bluetooth Speaker",
                    "category": "Audio",
                    "price": 89.99,
                    "description": (
                        "Compact waterproof speaker with 12-hour playtime and "
                        "rich bass."
                    ),
                    "in_stock": True,
                },
                "PROD-1004": {
                    "id": "PROD-1004",
                    "name": "4K Streaming Media Player",
                    "category": "Electronics",
                    "price": 49.99,
                    "description": (
                        "Streams video in 4K HDR with voice remote and built-in apps."
                    ),
                    "in_stock": False,
                },
            },
            "carts": {
                "CUST-501": {
                    "customer_id": "CUST-501",
                    "items": [{"product_id": "PROD-1002", "quantity": 1}],
                    "total": 179.99,
                },
            },
            "orders": {
                "ORD-9001": {
                    "order_id": "ORD-9001",
                    "customer_id": "CUST-501",
                    "items": [{"product_id": "PROD-1001", "quantity": 1}],
                    "total": 249.99,
                    "status": "Delivered",
                },
                "ORD-9002": {
                    "order_id": "ORD-9002",
                    "customer_id": "CUST-502",
                    "items": [{"product_id": "PROD-1003", "quantity": 2}],
                    "total": 179.98,
                    "status": "Shipped",
                },
                "ORD-9003": {
                    "order_id": "ORD-9003",
                    "customer_id": "CUST-501",
                    "items": [{"product_id": "PROD-1004", "quantity": 1}],
                    "total": 49.99,
                    "status": "Returned",
                },
            },
            "customers": {
                "CUST-501": {
                    "customer_id": "CUST-501",
                    "name": "Jordan Lee",
                    "email": "jordan.lee@example.com",
                    "pin": "1234",
                },
                "CUST-502": {
                    "customer_id": "CUST-502",
                    "name": "Priya Nair",
                    "email": "priya.nair@example.com",
                    "pin": "5678",
                },
            },
            "session": {"authenticated": False},
        }

    def register_tools(self, registry: ToolRegistry) -> None:
        data = self.data

        @registry.tool("Look up a product by ID")
        def get_product(product_id: str) -> str:
            if product_id not in data["products"]:
                return json.dumps({"error": f"Product {product_id} not found"})
            return json.dumps(data["products"][product_id])

        @registry.tool("List all products in the catalog")
        def list_products() -> str:
            return json.dumps({"products": list(data["products"].values())})

        @registry.tool("Search products by name or category (case-insensitive)")
        def search_products(query: str) -> str:
            query_lower = query.lower()
            results = [
                p
                for p in data["products"].values()
                if query_lower in p["name"].lower()
                or query_lower in p["category"].lower()
            ]
            return json.dumps({"results": results})

        @registry.tool("Get related product recommendations for a product")
        def get_recommendations(product_id: str) -> str:
            if product_id not in data["products"]:
                return json.dumps({"error": f"Product {product_id} not found"})
            category = data["products"][product_id]["category"]
            recommendations = [
                p
                for pid, p in data["products"].items()
                if pid != product_id and p["category"] == category
            ][:3]
            if not recommendations:
                recommendations = [
                    p for pid, p in data["products"].items() if pid != product_id
                ][:3]
            return json.dumps(
                {"product_id": product_id, "recommendations": recommendations}
            )

        @registry.tool("Get a customer's cart contents and total")
        def get_cart(customer_id: str) -> str:
            cart = data["carts"].setdefault(
                customer_id, {"customer_id": customer_id, "items": [], "total": 0.0}
            )
            return json.dumps(cart)

        @registry.tool("Add a product to a customer's cart")
        def add_to_cart(customer_id: str, product_id: str, quantity: int = 1) -> str:
            if product_id not in data["products"]:
                return json.dumps({"error": f"Product {product_id} not found"})
            cart = data["carts"].setdefault(
                customer_id, {"customer_id": customer_id, "items": [], "total": 0.0}
            )
            existing = next(
                (item for item in cart["items"] if item["product_id"] == product_id),
                None,
            )
            if existing is not None:
                existing["quantity"] += quantity
            else:
                cart["items"].append({"product_id": product_id, "quantity": quantity})
            cart["total"] = round(
                sum(
                    data["products"][item["product_id"]]["price"] * item["quantity"]
                    for item in cart["items"]
                    if item["product_id"] in data["products"]
                ),
                2,
            )
            return json.dumps(cart)

        @registry.tool("Get the status of an order by ID")
        def get_order_status(order_id: str) -> str:
            if order_id not in data["orders"]:
                return json.dumps({"error": f"Order {order_id} not found"})
            return json.dumps(data["orders"][order_id])

        @registry.tool("Authenticate a customer using their PIN")
        def authenticate_customer(customer_id: str, pin: str) -> str:
            if customer_id not in data["customers"]:
                return json.dumps({"error": f"Customer {customer_id} not found"})
            if data["customers"][customer_id].get("pin") != pin:
                return json.dumps(
                    {"error": "authentication failed", "status_code": 403}
                )
            data["session"]["authenticated"] = True
            return json.dumps(
                {"status": "success", "customer_id": customer_id, "authenticated": True}
            )

        @registry.tool("Process a return for a delivered order")
        def process_return(order_id: str, reason: str = "") -> str:
            if not data["session"]["authenticated"]:
                return json.dumps({"error": "not authenticated", "status_code": 403})
            if order_id not in data["orders"]:
                return json.dumps({"error": f"Order {order_id} not found"})
            order = data["orders"][order_id]
            if order.get("status") == "Returned":
                return json.dumps(
                    {"error": f"Order {order_id} already returned", "status_code": 409}
                )
            if order.get("status") != "Delivered":
                return json.dumps(
                    {
                        "error": (
                            f"Order {order_id} is not eligible for return "
                            f"(status: {order.get('status')})"
                        ),
                        "status_code": 422,
                    }
                )
            return_id = f"RET-{_ref_id()}"
            order["status"] = "Returned"
            order["return_id"] = return_id
            order["return_reason"] = reason
            return json.dumps(
                {
                    "return_id": return_id,
                    "order_id": order_id,
                    "status": "Returned",
                    "reason": reason,
                }
            )
