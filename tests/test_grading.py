"""Tests for stricter grading: arguments, communicated info, call order."""

from __future__ import annotations

from harness.grading import (
    check_order,
    compute_action_pass,
    match_expected_args,
)
from harness.tools.state_validation import validate_forbidden

# -- compute_action_pass (count-aware) --------------------------------------


def test_action_pass_no_expected_is_vacuously_true():
    assert compute_action_pass([], []) is True


def test_action_pass_all_matched():
    assert compute_action_pass(["get_invoice"], ["get_invoice"]) is True


def test_action_pass_missing_action_fails():
    assert compute_action_pass(["get_invoice"], []) is False


def test_action_pass_duplicate_expected_needs_duplicate_match():
    # Two refunds expected but only one matched must FAIL (multiplicity), even
    # though a set comparison would spuriously pass.
    assert (
        compute_action_pass(["process_refund", "process_refund"], ["process_refund"])
        is False
    )


def test_action_pass_duplicate_expected_with_duplicate_match():
    assert (
        compute_action_pass(
            ["process_refund", "process_refund"],
            ["process_refund", "process_refund"],
        )
        is True
    )


# -- match_expected_args ----------------------------------------------------


def _call(name, **args):
    return {"name": name, "arguments": args}


def test_args_exact_match_passes():
    calls = [_call("process_refund", invoice_id="INV-8834", amount="247.50")]
    ok, fails = match_expected_args(
        calls, {"process_refund": {"invoice_id": "INV-8834"}}
    )
    assert ok and fails == []


def test_args_wrong_value_fails():
    calls = [_call("process_refund", invoice_id="INV-0000")]
    ok, fails = match_expected_args(
        calls, {"process_refund": {"invoice_id": "INV-8834"}}
    )
    assert not ok
    assert "invoice_id" in fails[0]


def test_args_tool_never_called_fails():
    ok, fails = match_expected_args([], {"process_refund": {"invoice_id": "X"}})
    assert not ok
    assert "never called" in fails[0]


def test_args_numeric_vs_string_normalized():
    calls = [_call("set_qty", qty=5)]
    ok, _ = match_expected_args(calls, {"set_qty": {"qty": "5"}})
    assert ok


def test_args_int_vs_float_compare_equal():
    # An LLM emitting the int 70 must satisfy an expected float 70.0 (and the
    # reverse), since they are numerically equal.
    calls = [_call("process_refund", amount=70)]
    ok, fails = match_expected_args(calls, {"process_refund": {"amount": 70.0}})
    assert ok and fails == []
    calls = [_call("process_refund", amount=70.0)]
    ok, _ = match_expected_args(calls, {"process_refund": {"amount": 70}})
    assert ok
    # Numeric strings coerce too: "70.00" == 70.
    calls = [_call("process_refund", amount="70.00")]
    ok, _ = match_expected_args(calls, {"process_refund": {"amount": 70}})
    assert ok


def test_args_numerically_unequal_still_fails():
    calls = [_call("process_refund", amount=70.5)]
    ok, fails = match_expected_args(calls, {"process_refund": {"amount": 70.0}})
    assert not ok
    assert "amount" in fails[0]


def test_args_bool_not_coerced_to_number():
    # True must not equal 1 — bools compare on the string path.
    calls = [_call("set_flag", enabled=True)]
    ok, _ = match_expected_args(calls, {"set_flag": {"enabled": 1}})
    assert not ok
    ok, _ = match_expected_args(calls, {"set_flag": {"enabled": True}})
    assert ok


def test_args_not_null_assertion():
    ok, _ = match_expected_args(
        [_call("create_lead", email="a@b.com")], {"create_lead": {"email": "_not_null"}}
    )
    assert ok
    bad, fails = match_expected_args(
        [_call("create_lead", email="")], {"create_lead": {"email": "_not_null"}}
    )
    assert not bad


def test_args_contains_assertion():
    calls = [_call("draft_email", body="Your warranty is about to expire soon")]
    ok, _ = match_expected_args(
        calls, {"draft_email": {"body": {"_contains": "warranty"}}}
    )
    assert ok


def test_args_one_good_call_among_retries_passes():
    calls = [
        _call("process_refund", invoice_id="WRONG"),
        _call("process_refund", invoice_id="INV-8834"),
    ]
    ok, fails = match_expected_args(
        calls, {"process_refund": {"invoice_id": "INV-8834"}}
    )
    assert ok and fails == []


def test_args_ordered_assertions_cover_repeated_calls():
    calls = [
        _call("charge", amount=40, method="credit card"),
        _call("charge", amount=60, method="PayPal"),
    ]
    ok, failures = match_expected_args(
        calls,
        {
            "charge": [
                {"amount": 40, "method": {"_contains": "card"}},
                {"amount": 60, "method": {"_contains": "paypal"}},
            ]
        },
    )
    assert ok is True
    assert failures == []


def test_args_ordered_assertions_reject_swapped_repeated_calls():
    calls = [_call("charge", amount=60), _call("charge", amount=40)]
    ok, failures = match_expected_args(
        calls, {"charge": [{"amount": 40}, {"amount": 60}]}
    )
    assert ok is False
    assert all("[call" in failure for failure in failures)


# -- check_order ------------------------------------------------------------


def test_order_satisfied_as_subsequence():
    called = ["get_invoice", "search_kb", "process_refund"]
    ok, _ = check_order(called, ["get_invoice", "process_refund"])
    assert ok


def test_order_violated():
    called = ["process_refund", "get_invoice"]
    ok, fails = check_order(called, ["get_invoice", "process_refund"])
    assert not ok
    assert "order not satisfied" in fails[0]


def test_order_missing_tool():
    called = ["get_invoice"]
    ok, _ = check_order(called, ["get_invoice", "process_refund"])
    assert not ok


# -- validate_forbidden (collateral damage) ---------------------------------


def test_forbidden_no_damage_passes():
    state = {"invoices": {"INV-1": {"status": "Refunded"}}}
    ok, _ = validate_forbidden(state, {"invoices": {"INV-2": {"status": "Refunded"}}})
    assert ok  # INV-2 untouched (missing), so no collateral damage


def test_forbidden_damage_detected():
    state = {"invoices": {"INV-2": {"status": "Refunded"}}}
    ok, fails = validate_forbidden(
        state, {"invoices": {"INV-2": {"status": "Refunded"}}}
    )
    assert not ok
    assert "must NOT equal" in fails[0]


def test_forbidden_not_null_means_must_stay_unset():
    state = {"tickets": {"T-9": {"id": "T-9"}}}
    ok, fails = validate_forbidden(state, {"tickets": {"T-9": "_not_null"}})
    assert not ok  # a ticket was created that should not have been
