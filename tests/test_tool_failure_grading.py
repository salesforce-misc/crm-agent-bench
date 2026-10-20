"""P0.1 regression: failed tool executions must not satisfy expected actions.

Execution failures such as an unknown tool, malformed arguments, or a raised
exception must not count toward ``actions_matched``. Negative business results
such as ``{"found": false}`` are successful invocations and must count.
"""

from __future__ import annotations

import pytest

from harness.bench_runner import Task, run_task
from harness.models import AgentResponse
from harness.tools.registry import is_error_result


class _ScriptedAgent:
    """Agent stub that replays a fixed sequence of tool calls then stops."""

    def __init__(self, tool_calls_made: list[dict]):
        self._tool_calls_made = tool_calls_made
        self._calls = 0

    async def run(self, user_message: str, history=None) -> AgentResponse:
        self._calls += 1
        if self._calls == 1:
            return AgentResponse(
                content="working on it",
                tool_calls_made=self._tool_calls_made,
                iterations=1,
            )
        return AgentResponse(content="all done", tool_calls_made=[], iterations=1)


class _ScriptedUser:
    """User sim that says one thing then stops."""

    def reset(self, instruction: str) -> None:
        self._turn = 0

    async def step(self, agent_message: str) -> str:
        self._turn += 1
        if self._turn == 1:
            return "please refund my invoice"
        return "###STOP###"


class TestIsErrorResult:
    def test_domain_error_json_is_failure(self):
        assert is_error_result('{"error": "Customer not found"}') is True

    def test_unknown_tool_error_is_failure(self):
        assert is_error_result('{"error": "Unknown tool: foo"}') is True

    def test_plain_success_string_is_ok(self):
        assert is_error_result("Invoice INV-1: $500, status paid") is False

    def test_success_json_object_is_ok(self):
        assert is_error_result('{"invoice_id": "INV-1", "status": "paid"}') is False

    def test_non_json_is_ok(self):
        assert is_error_result("done") is False

    def test_error_among_other_fields_is_failure(self):
        # An error result without a status_code is still a failure even with
        # other fields present (unknown tool, exception, not-found, bad args).
        assert (
            is_error_result('{"error": "Lead already exists", "lead_id": "L-1"}')
            is True
        )

    def test_business_rejection_status_code_is_not_failure(self):
        # A tool that executed correctly and intentionally rejected the request
        # (409/403/422) is NOT a failed execution — the action is satisfied.
        assert (
            is_error_result(
                '{"error": "Lead already exists", "lead_id": "L-1", "status_code": 409}'
            )
            is False
        )
        assert (
            is_error_result('{"error": "not your campaign", "status_code": 403}')
            is False
        )
        assert (
            is_error_result('{"error": "validation failed", "status_code": 422}')
            is False
        )

    def test_server_and_client_error_status_codes_remain_failures(self):
        # Genuine failures keep failing even when a status_code is present.
        assert is_error_result('{"error": "boom", "status_code": 500}') is True
        assert is_error_result('{"error": "bad request", "status_code": 400}') is True
        assert is_error_result('{"error": "not found", "status_code": 404}') is True


class TestFailedCallsDoNotMatch:
    @pytest.mark.asyncio
    async def test_errored_tool_call_does_not_satisfy_expected_action(self):
        task = Task(
            id="t_err",
            instruction="refund the invoice",
            expected_actions=["process_refund"],
        )
        # The agent called the right tool name, but it errored.
        agent = _ScriptedAgent(
            [
                {
                    "name": "process_refund",
                    "arguments": {"id": "X"},
                    "result": '{"error": "Invoice not found"}',
                    "ok": False,
                },
            ]
        )
        trace = await run_task(
            task,
            agent,
            _ScriptedUser(),
            max_turns=5,
            timeout_seconds=30,
            agent_step_timeout_seconds=30,
            user_step_timeout_seconds=30,
        )
        assert "process_refund" in trace.tools_called  # still recorded for diagnostics
        assert trace.actions_matched == []  # but does NOT count as matched

    @pytest.mark.asyncio
    async def test_successful_tool_call_satisfies_expected_action(self):
        task = Task(
            id="t_ok",
            instruction="refund the invoice",
            expected_actions=["process_refund"],
        )
        agent = _ScriptedAgent(
            [
                {
                    "name": "process_refund",
                    "arguments": {"id": "X"},
                    "result": '{"refund_id": "R-1", "status": "ok"}',
                    "ok": True,
                },
            ]
        )
        trace = await run_task(
            task,
            agent,
            _ScriptedUser(),
            max_turns=5,
            timeout_seconds=30,
            agent_step_timeout_seconds=30,
            user_step_timeout_seconds=30,
        )
        assert trace.actions_matched == ["process_refund"]

    @pytest.mark.asyncio
    async def test_not_found_business_result_satisfies_expected_action(self):
        task = Task(
            id="t_not_found",
            instruction="look up the missing invoice",
            expected_actions=["get_invoice"],
            expected_args={"get_invoice": {"invoice_id": "INV-MISSING"}},
        )
        agent = _ScriptedAgent(
            [
                {
                    "name": "get_invoice",
                    "arguments": {"invoice_id": "INV-MISSING"},
                    "result": '{"found": false, "invoice_id": "INV-MISSING"}',
                },
            ]
        )

        trace = await run_task(
            task,
            agent,
            _ScriptedUser(),
            max_turns=5,
            timeout_seconds=30,
            agent_step_timeout_seconds=30,
            user_step_timeout_seconds=30,
        )

        assert trace.tool_calls_detail[0]["ok"] is True
        assert trace.actions_matched == ["get_invoice"]
        assert trace.args_passed is True

    @pytest.mark.asyncio
    async def test_business_rejection_call_satisfies_expected_action(self):
        # Consolidated approach: a negative task is credited purely via the
        # tool's business-rejection status_code (403/409/422) — no per-task
        # flag. The agent called the right tool; the backend intentionally
        # rejected the request, which still satisfies the expected action.
        task = Task(
            id="t_reject",
            instruction="apply an invalid coupon",
            expected_actions=["CSA_Apply_Promotion"],
        )
        agent = _ScriptedAgent(
            [
                {
                    "name": "CSA_Apply_Promotion",
                    "arguments": {"Coupon": "BOGUS50OFF"},
                    "result": '{"error": "Invalid coupon code: BOGUS50OFF", "status_code": 422}',
                },
            ]
        )
        trace = await run_task(
            task,
            agent,
            _ScriptedUser(),
            max_turns=5,
            timeout_seconds=30,
            agent_step_timeout_seconds=30,
            user_step_timeout_seconds=30,
        )
        # ok is derived from is_error_result; the 422 makes it a success.
        assert trace.tool_calls_detail[0]["ok"] is True
        assert trace.actions_matched == ["CSA_Apply_Promotion"]
