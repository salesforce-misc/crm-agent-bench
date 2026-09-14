"""Integration (external-API) stateful tool domain.

Provides tools that simulate calling out to third-party external APIs:
a weather service, a shipment tracking service, and a payment gateway.
State is kept in a shared dict simulating cached external-API responses.
"""

from __future__ import annotations

import json
import random
import string

from harness.tools.environment import ToolEnvironment, register_domain
from harness.tools.registry import ToolRegistry


def _ref_id() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


@register_domain("integration")
class IntegrationEnvironment(ToolEnvironment):
    def default_state(self) -> dict:
        return {
            "weather": {
                "San Francisco": {
                    "temp_f": 62,
                    "condition": "Foggy",
                    "humidity": 78,
                },
                "Phoenix": {
                    "temp_f": 104,
                    "condition": "Sunny",
                    "humidity": 12,
                },
                "Miami": {
                    "temp_f": 89,
                    "condition": "Thunderstorms",
                    "humidity": 85,
                },
            },
            "forecasts": {
                "San Francisco": [
                    {"day": 1, "temp_f": 61, "condition": "Foggy"},
                    {"day": 2, "temp_f": 64, "condition": "Partly Cloudy"},
                    {"day": 3, "temp_f": 66, "condition": "Sunny"},
                    {"day": 4, "temp_f": 63, "condition": "Foggy"},
                    {"day": 5, "temp_f": 65, "condition": "Sunny"},
                ],
                "Phoenix": [
                    {"day": 1, "temp_f": 103, "condition": "Sunny"},
                    {"day": 2, "temp_f": 106, "condition": "Sunny"},
                    {"day": 3, "temp_f": 101, "condition": "Sunny"},
                    {"day": 4, "temp_f": 99, "condition": "Sunny"},
                    {"day": 5, "temp_f": 100, "condition": "Sunny"},
                ],
                "Miami": [
                    {"day": 1, "temp_f": 88, "condition": "Thunderstorms"},
                    {"day": 2, "temp_f": 87, "condition": "Thunderstorms"},
                    {"day": 3, "temp_f": 90, "condition": "Humid"},
                    {"day": 4, "temp_f": 91, "condition": "Sunny"},
                    {"day": 5, "temp_f": 89, "condition": "Humid"},
                ],
            },
            "alerts": {
                "Miami": {
                    "severity": "Severe",
                    "headline": "Severe Thunderstorm Warning",
                    "description": (
                        "A severe thunderstorm warning is in effect for the Miami "
                        "area until 8:00 PM local time. Expect heavy rain and "
                        "damaging winds."
                    ),
                },
            },
            "shipments": {
                "1Z999AA10123456784": {
                    "status": "In Transit",
                    "location": "Memphis, TN",
                    "eta": "2026-09-13",
                },
                "1Z999AA10198765432": {
                    "status": "Delivered",
                    "location": "Front porch",
                    "eta": "2026-09-10",
                },
            },
        }

    def register_tools(self, registry: ToolRegistry) -> None:
        data = self.data

        @registry.tool("Get the current weather for a city")
        def get_current_weather(city: str) -> str:
            if city not in data["weather"]:
                return json.dumps({"error": f"Unknown city: {city}"})
            return json.dumps({"city": city, **data["weather"][city]})

        @registry.tool("Get a multi-day weather forecast for a city")
        def get_weather_forecast(city: str, days: int = 3) -> str:
            if city not in data["forecasts"]:
                return json.dumps({"error": f"Unknown city: {city}"})
            forecast = data["forecasts"][city][:days]
            return json.dumps({"city": city, "forecast": forecast})

        @registry.tool("Get any active weather alert for a city")
        def get_weather_alert(city: str) -> str:
            if city not in data["weather"]:
                return json.dumps({"error": f"Unknown city: {city}"})
            alert = data["alerts"].get(city)
            return json.dumps({"city": city, "alert": alert})

        @registry.tool("Track a shipment by tracking number")
        def track_shipment(tracking_number: str) -> str:
            if tracking_number not in data["shipments"]:
                return json.dumps(
                    {"error": f"Unknown tracking number: {tracking_number}"}
                )
            shipment = data["shipments"][tracking_number]
            return json.dumps({"tracking_number": tracking_number, **shipment})

        @registry.tool("Process a payment through the external payment gateway")
        def process_gateway_payment(amount: float, card_token: str) -> str:
            if amount <= 0:
                return json.dumps(
                    {
                        "error": "Amount must be greater than zero",
                        "status_code": 422,
                    }
                )
            if card_token == "tok_declined":
                return json.dumps(
                    {
                        "success": False,
                        "reason": "card_declined",
                        "status_code": 402,
                    }
                )
            txn_id = f"GTX-{_ref_id()}"
            return json.dumps(
                {
                    "success": True,
                    "gateway_txn_id": txn_id,
                    "amount": amount,
                }
            )
