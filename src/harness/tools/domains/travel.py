"""Travel stateful tool domain.

Provides 10 tools that read/write a shared state dict simulating a travel
booking platform: hotels, flights, bookings (hotel + flight), car rentals,
and guest reservations/itineraries.
"""

from __future__ import annotations

import json
import random
import string

from harness.tools.environment import ToolEnvironment, register_domain
from harness.tools.registry import ToolRegistry


def _ref_id() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


@register_domain("travel")
class TravelEnvironment(ToolEnvironment):
    def default_state(self) -> dict:
        return {
            "hotels": {
                "HTL-1": {
                    "hotel_id": "HTL-1",
                    "name": "Grand Central Hotel",
                    "city": "New York",
                    "price_per_night": 220.0,
                    "rooms_available": 5,
                },
                "HTL-2": {
                    "hotel_id": "HTL-2",
                    "name": "Riverside Inn",
                    "city": "New York",
                    "price_per_night": 150.0,
                    "rooms_available": 0,
                },
                "HTL-3": {
                    "hotel_id": "HTL-3",
                    "name": "Golden Gate Suites",
                    "city": "San Francisco",
                    "price_per_night": 260.0,
                    "rooms_available": 3,
                },
            },
            "flights": {
                "FL-1": {
                    "flight_id": "FL-1",
                    "origin": "New York",
                    "destination": "San Francisco",
                    "price": 350.0,
                    "seats_available": 10,
                },
                "FL-2": {
                    "flight_id": "FL-2",
                    "origin": "San Francisco",
                    "destination": "New York",
                    "price": 340.0,
                    "seats_available": 8,
                },
            },
            "bookings": {},
            "guests": {
                "GST-1": {
                    "guest_id": "GST-1",
                    "name": "Alice Johnson",
                    "hotel_id": "HTL-1",
                    "check_in": "2026-09-15",
                    "interests": ["museums", "fine dining"],
                },
                "GST-2": {
                    "guest_id": "GST-2",
                    "name": "Bob Martinez",
                    "hotel_id": "HTL-3",
                    "check_in": "2026-09-20",
                    "interests": ["hiking", "wine tasting"],
                },
            },
            "cars": {},
        }

    def register_tools(self, registry: ToolRegistry) -> None:
        data = self.data

        @registry.tool("Search hotels available in a given city")
        def search_hotels(city: str) -> str:
            city_lower = city.lower()
            results = [
                h for h in data["hotels"].values() if h["city"].lower() == city_lower
            ]
            return json.dumps({"hotels": results})

        @registry.tool("Look up a hotel by ID")
        def get_hotel(hotel_id: str) -> str:
            if hotel_id not in data["hotels"]:
                return json.dumps({"error": f"Hotel {hotel_id} not found"})
            return json.dumps(data["hotels"][hotel_id])

        @registry.tool("Create a hotel booking for a guest")
        def create_hotel_booking(hotel_id: str, guest_name: str, nights: int) -> str:
            if hotel_id not in data["hotels"]:
                return json.dumps({"error": f"Hotel {hotel_id} not found"})
            hotel = data["hotels"][hotel_id]
            if hotel.get("rooms_available", 0) <= 0:
                return json.dumps(
                    {
                        "error": f"No rooms available at {hotel_id}",
                        "status_code": 409,
                    }
                )
            booking_id = f"BKG-{_ref_id()}"
            hotel["rooms_available"] -= 1
            booking = {
                "booking_id": booking_id,
                "type": "hotel",
                "hotel_id": hotel_id,
                "guest_name": guest_name,
                "nights": nights,
                "status": "Reserved",
            }
            data["bookings"][booking_id] = booking
            return json.dumps(booking)

        @registry.tool("Confirm a pending booking")
        def confirm_booking(booking_id: str) -> str:
            if booking_id not in data["bookings"]:
                return json.dumps({"error": f"Booking {booking_id} not found"})
            booking = data["bookings"][booking_id]
            if booking.get("status") == "Confirmed":
                return json.dumps(
                    {
                        "error": f"Booking {booking_id} already confirmed",
                        "status_code": 409,
                    }
                )
            booking["status"] = "Confirmed"
            return json.dumps(booking)

        @registry.tool("Search flights between an origin and destination")
        def search_flights(origin: str, destination: str) -> str:
            origin_lower = origin.lower()
            destination_lower = destination.lower()
            results = [
                f
                for f in data["flights"].values()
                if f["origin"].lower() == origin_lower
                and f["destination"].lower() == destination_lower
            ]
            return json.dumps({"flights": results})

        @registry.tool("Book a flight for a passenger")
        def book_flight(flight_id: str, passenger_name: str) -> str:
            if flight_id not in data["flights"]:
                return json.dumps({"error": f"Flight {flight_id} not found"})
            flight = data["flights"][flight_id]
            if flight.get("seats_available", 0) <= 0:
                return json.dumps(
                    {
                        "error": f"No seats available on {flight_id}",
                        "status_code": 409,
                    }
                )
            booking_id = f"BKG-{_ref_id()}"
            flight["seats_available"] -= 1
            booking = {
                "booking_id": booking_id,
                "type": "flight",
                "flight_id": flight_id,
                "passenger_name": passenger_name,
                "status": "Confirmed",
            }
            data["bookings"][booking_id] = booking
            return json.dumps(booking)

        @registry.tool("Book a rental car in a given city")
        def book_car(city: str, car_type: str = "economy") -> str:
            car_id = f"CAR-{_ref_id()}"
            car = {
                "car_id": car_id,
                "city": city,
                "car_type": car_type,
                "status": "Confirmed",
            }
            data["cars"][car_id] = car
            return json.dumps(car)

        @registry.tool("Send a trip itinerary to a guest's email")
        def send_itinerary(booking_reference: str, email: str) -> str:
            return json.dumps(
                {
                    "sent": True,
                    "booking_reference": booking_reference,
                    "email": email,
                }
            )

        @registry.tool("Look up a guest's reservation and profile")
        def get_guest_reservation(guest_id: str) -> str:
            if guest_id not in data["guests"]:
                return json.dumps({"error": f"Guest {guest_id} not found"})
            guest = dict(data["guests"][guest_id])
            hotel = data["hotels"].get(guest.get("hotel_id"))
            if hotel:
                guest["hotel"] = hotel
            return json.dumps(guest)

        @registry.tool("Generate a personalized day schedule for a guest")
        def generate_schedule(guest_id: str) -> str:
            if guest_id not in data["guests"]:
                return json.dumps({"error": f"Guest {guest_id} not found"})
            guest = data["guests"][guest_id]
            interests = guest.get("interests", [])
            hotel = data["hotels"].get(guest.get("hotel_id"), {})
            city = hotel.get("city", "the area")
            slots = ["09:00", "12:00", "15:00", "18:00"]
            activities = [f"Explore {interest} in {city}" for interest in interests]
            if not activities:
                activities = [f"Sightseeing in {city}"]
            schedule = [
                {"time": slots[i % len(slots)], "activity": activities[i]}
                for i in range(len(activities))
            ]
            return json.dumps(
                {
                    "guest_id": guest_id,
                    "guest_name": guest.get("name"),
                    "check_in": guest.get("check_in"),
                    "schedule": schedule,
                }
            )
