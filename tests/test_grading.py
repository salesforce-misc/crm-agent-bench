"""Tests for stricter grading: arguments, communicated info, call order."""

from __future__ import annotations

from harness.grading import (
    check_order,
    compute_action_pass,
    match_expected_args,
    match_response_contains,
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


# -- match_response_contains ------------------------------------------------


def _turns(*assistant_msgs):
    return [{"role": "assistant", "content": m} for m in assistant_msgs]


def test_response_substring_present():
    ok, _ = match_response_contains(
        _turns("Your ticket number is TICK-123."), ["TICK-123"]
    )
    assert ok


def test_response_substring_missing():
    ok, fails = match_response_contains(_turns("All done!"), ["TICK-123"])
    assert not ok
    assert "TICK-123" in fails[0]


def test_response_case_insensitive():
    ok, _ = match_response_contains(_turns("Refund PROCESSED"), ["refund processed"])
    assert ok


def test_response_regex():
    ok, _ = match_response_contains(_turns("CTR is 3.4%"), [r"re:\d+\.\d+%"])
    assert ok


def test_response_ignores_user_turns():
    turns = [
        {"role": "user", "content": "TICK-123"},
        {"role": "assistant", "content": "Let me check."},
    ]
    ok, _ = match_response_contains(turns, ["TICK-123"])
    assert not ok  # only the user said it, not the agent


def test_response_unicode_hyphen_folded_substring():
    # Agent emits a non-breaking hyphen (U+2011); gold uses ASCII "-".
    ok, _ = match_response_contains(_turns("Your appointment is SA‑0409."), ["SA-0409"])
    assert ok


def test_response_unicode_hyphen_folded_regex():
    # En dash (U+2013) in the agent text still matches an ASCII regex.
    ok, _ = match_response_contains(
        _turns("Options: SA–0409 or SA–0412."),
        [r"re:SA-0409|SA-0412"],
    )
    assert ok


def test_response_unicode_hyphen_in_pattern_folded():
    # Non-breaking hyphen inside the gold pattern matches ASCII agent text.
    ok, _ = match_response_contains(_turns("Ticket SA-0409 created."), ["SA‑0409"])
    assert ok


def test_response_smart_apostrophe_folded_regex():
    # Agent emits a curly apostrophe (U+2019); gold regex uses ASCII "'".
    ok, _ = match_response_contains(
        _turns("I can’t retrieve that; it isn’t associated with your account."),
        [r"re:can't|isn't associated"],
    )
    assert ok


def test_response_smart_apostrophe_folded_substring():
    # Curly apostrophe in the agent text matches an ASCII substring needle.
    ok, _ = match_response_contains(
        _turns("That doesn’t belong to you."), ["doesn't belong"]
    )
    assert ok


def test_response_smart_apostrophe_in_pattern_folded():
    # Curly apostrophe inside the gold pattern matches ASCII agent text.
    ok, _ = match_response_contains(
        _turns("It doesn't belong to you."), ["doesn’t belong"]
    )
    assert ok


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
