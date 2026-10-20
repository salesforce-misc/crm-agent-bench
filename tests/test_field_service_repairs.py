"""Regression tests for the field-service fixture and ownership repairs."""

from __future__ import annotations

import json

import pytest

import harness.tools.domains  # noqa: F401 -- register field_service
from harness.graph_cli import _load_graph_config
from harness.tools.environment import create_environment


class TestFieldServicePreWorkBriefs:
    def setup_method(self) -> None:
        self.environment = create_environment("field_service")
        self.registry = self.environment.create_registry()

    @pytest.mark.asyncio
    async def test_keyboard_prework_has_a_consistent_canonical_request(self):
        result = await self.registry.execute(
            "FSA_Get_Pre_Work_Brief_for_Technician",
            {"appointment_number": "SA-0223"},
        )
        brief = json.loads(result)

        assert brief["service_request"]["subject"] == "Laptop Keyboard Issue"
        assert brief["work_order"]["subject"] == "Laptop Keyboard Issue"
        assert "keyboard" in brief["service_note"].lower()
        assert "Replacement keyboard" in " ".join(brief["checklist"])

    @pytest.mark.asyncio
    async def test_customer_attempts_are_reachable_in_prework_brief(self):
        result = await self.registry.execute(
            "FSA_Get_Pre_Work_Brief_for_Technician",
            {"appointment_number": "SA-0425"},
        )
        brief = json.loads(result)
        attempts = brief["service_request"]["customer_attempts"]
        actions = {attempt["action"] for attempt in attempts}

        assert "Disconnect and reconnect the keyboard" in actions
        assert "Update or reinstall the keyboard drivers" in actions

    @pytest.mark.asyncio
    async def test_performance_prework_uses_performance_checklist(self):
        result = await self.registry.execute(
            "FSA_Get_Pre_Work_Brief_for_Technician",
            {"appointment_number": "SA-0385"},
        )
        brief = json.loads(result)
        checklist = " ".join(brief["checklist"])

        assert "RAM upgrade kit" in checklist
        assert "SSD replacement tools" in checklist
        assert "Multimeter" not in checklist


class TestFieldServiceAppointmentOwnership:
    def setup_method(self) -> None:
        self.environment = create_environment("field_service")
        self.registry = self.environment.create_registry()

    async def _verify_gabe(self) -> None:
        result = await self.registry.execute(
            "FSA_Identify_User_Type_Based_on_Email",
            {"email": "ghoskins@example.com"},
        )
        assert json.loads(result)["user_type"] == "Customer"

    @pytest.mark.asyncio
    async def test_cross_account_mutations_are_denied_without_state_change(self):
        await self._verify_gabe()
        before = self.environment.get_state()["service_appointments"]["SA-0412"].copy()

        for tool_name, arguments in (
            ("FSA_Unschedule_Appointment", {"appointment_number": "SA-0412"}),
            (
                "FSA_Update_Arrival_Windows",
                {
                    "appointment_number": "SA-0412",
                    "start_time": "2026-09-21 10:00",
                    "end_time": "2026-09-21 11:00",
                },
            ),
            (
                "FSA_Schedule_Appointment",
                {"appointment_number": "SA-0412", "slot": "10:00-11:00"},
            ),
        ):
            result = await self.registry.execute(tool_name, arguments)
            payload = json.loads(result)
            assert payload["status_code"] == 403
            assert "another customer" in payload["error"]

        assert self.environment.get_state()["service_appointments"]["SA-0412"] == before

    @pytest.mark.asyncio
    async def test_cross_account_appointment_listing_and_slots_are_denied(self):
        await self._verify_gabe()

        listing = json.loads(
            await self.registry.execute(
                "FSA_Get_Appointment_details_based_on_User_Input",
                {"email": "rjanzen@example.com"},
            )
        )
        slots = json.loads(
            await self.registry.execute(
                "FSA_Get_Available_Slots_Based_on_Appointment_Number",
                {"appointment_number": "SA-0412"},
            )
        )

        assert listing["status_code"] == 403
        assert slots["status_code"] == 403

    @pytest.mark.asyncio
    async def test_verified_owner_can_still_change_own_appointment(self):
        await self._verify_gabe()

        result = await self.registry.execute(
            "FSA_Unschedule_Appointment",
            {"appointment_number": "SA-0406"},
        )

        assert json.loads(result)["status"] == "Success"
        assert (
            self.environment.data["service_appointments"]["SA-0406"]["status"]
            == "Unscheduled"
        )


def test_field_service_policy_is_not_in_the_runtime_system_prompt():
    config = _load_graph_config("src/agents/field_service.yaml")

    assert "## Operational Policy" not in config["system_prompt"]
    assert (
        "A customer may only act on their own appointments."
        not in config["system_prompt"]
    )
