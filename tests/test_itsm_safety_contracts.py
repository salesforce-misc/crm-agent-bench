"""Regression tests for ITSM safety and precondition invariants."""

from __future__ import annotations

import json

import pytest

from harness.tools.domains.itsm import ITSMEnvironment


async def _call(registry, name: str, **arguments) -> dict:
    return json.loads(await registry.execute(name, arguments))


@pytest.mark.asyncio
async def test_reassigning_an_unassigned_ticket_fails_without_assigning_it():
    environment = ITSMEnvironment()
    registry = environment.create_registry()

    result = await _call(
        registry,
        "reassign_ticket",
        ticket_id="TCK-1004",
        assignee="admin.rlee",
    )

    assert result["status_code"] == 422
    assert environment.get_state()["tickets"]["TCK-1004"]["assigned_to"] == ""


@pytest.mark.asyncio
async def test_closing_an_already_closed_ticket_fails_without_mutation():
    environment = ITSMEnvironment()
    registry = environment.create_registry()
    before = environment.get_state()["tickets"]["TCK-1003"]

    result = await _call(registry, "close_ticket", ticket_id="TCK-1003")

    assert result["status_code"] == 409
    assert environment.get_state()["tickets"]["TCK-1003"] == before


@pytest.mark.asyncio
async def test_delete_requires_dependency_check_without_mutating_ticket():
    environment = ITSMEnvironment()
    registry = environment.create_registry()

    result = await _call(registry, "delete_ticket", ticket_id="TCK-1004")

    assert result["status_code"] == 409
    assert "dependency check" in result["error"]
    assert "TCK-1004" in environment.get_state()["tickets"]


@pytest.mark.asyncio
async def test_delete_requires_ticket_specific_permission_without_mutating_ticket():
    environment = ITSMEnvironment()
    registry = environment.create_registry()
    await _call(registry, "check_dependencies", ticket_id="TCK-1004")

    result = await _call(registry, "delete_ticket", ticket_id="TCK-1004")

    assert result["status_code"] == 403
    assert "ticket-specific" in result["error"]
    assert "TCK-1004" in environment.get_state()["tickets"]


@pytest.mark.asyncio
async def test_delete_permission_is_bound_to_the_checked_ticket():
    environment = ITSMEnvironment()
    registry = environment.create_registry()
    await _call(registry, "check_dependencies", ticket_id="TCK-1001")
    await _call(
        registry,
        "check_permission",
        user_id="admin.rlee",
        action="delete",
        ticket_id="TCK-1004",
    )

    result = await _call(registry, "delete_ticket", ticket_id="TCK-1001")

    assert result["status_code"] == 403
    assert {"TCK-1001", "TCK-1004"} <= set(environment.get_state()["tickets"])


@pytest.mark.asyncio
async def test_full_preflight_deletes_ticket_and_consumes_preflight_records():
    environment = ITSMEnvironment()
    registry = environment.create_registry()
    dependency_result = await _call(
        registry, "check_dependencies", ticket_id="TCK-1004"
    )
    permission_result = await _call(
        registry,
        "check_permission",
        user_id="admin.rlee",
        action="delete",
        ticket_id="TCK-1004",
    )

    result = await _call(registry, "delete_ticket", ticket_id="TCK-1004")

    assert dependency_result["open_dependencies"] == []
    assert permission_result["allowed"] is True
    assert result == {"status": "deleted", "ticket_id": "TCK-1004"}
    state = environment.get_state()
    assert "TCK-1004" not in state["tickets"]
    assert "TCK-1004" not in state["session"]["dependency_checks"]
    assert "TCK-1004" not in state["session"]["delete_authorizations"]


@pytest.mark.asyncio
async def test_open_dependency_is_exposed_and_blocks_deletion():
    environment = ITSMEnvironment()
    registry = environment.create_registry()
    dependency_result = await _call(
        registry, "check_dependencies", ticket_id="TCK-1002"
    )
    await _call(
        registry,
        "check_permission",
        user_id="admin.rlee",
        action="delete",
        ticket_id="TCK-1002",
    )

    result = await _call(registry, "delete_ticket", ticket_id="TCK-1002")

    assert dependency_result["open_dependencies"] == [
        {"ticket_id": "TCK-1004", "status": "Open"}
    ]
    assert result["status_code"] == 409
    assert "open dependencies" in result["error"]
    assert "TCK-1002" in environment.get_state()["tickets"]


@pytest.mark.asyncio
async def test_denied_permission_never_authorizes_deletion():
    environment = ITSMEnvironment()
    registry = environment.create_registry()
    await _call(registry, "check_dependencies", ticket_id="TCK-1001")
    permission_result = await _call(
        registry,
        "check_permission",
        user_id="agent.jsmith",
        action="delete",
        ticket_id="TCK-1001",
    )

    result = await _call(registry, "delete_ticket", ticket_id="TCK-1001")

    assert permission_result["allowed"] is False
    assert result["status_code"] == 403
    assert "TCK-1001" in environment.get_state()["tickets"]
