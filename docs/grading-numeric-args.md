# Numeric argument comparison in the args check

The **args** check (`match_expected_args` in `src/harness/grading.py`) verifies
that a tool was called with the expected argument values. Values arrive from the
model as JSON, so a numeric argument can show up as an `int`, a `float`, or a
numeric string depending on the model and provider — while the gold value in the
task YAML is written however the author spelled it (`70`, `70.0`, `"70"`).

## The problem

The comparison normalized both sides to lowercase strings and compared those.
That made `5` and `"5"` compare equal (good) but treated **`70` and `70.0` as
different strings** (`"70" != "70.0"`), so a deterministically-correct refund
would be flagged `ARGS_FAIL` purely because the model emitted `70` where the task
declared `70.0`. This surfaced in a real run:

```
process_refund.amount: expected 70.0, got 70   # numerically equal, wrongly failed
```

## The fix

`_arg_matches` now compares **by numeric value when both sides are numbers**,
falling back to the previous normalized-string equality otherwise:

```python
actual_num, expected_num = _as_number(actual), _as_number(expected)
if actual_num is not None and expected_num is not None:
    return actual_num == expected_num
return _norm(actual) == _norm(expected)
```

`_as_number` coerces `int`, `float`, and numeric strings (`"70.00"`) to `float`,
and returns `None` for anything else so the string path still handles ids, codes,
and free text.

### Bools are deliberately excluded

`_as_number` returns `None` for `bool`, so a boolean argument compares on the
string path and **`True` never equals `1`**. Tool flags like `enabled: true`
keep strict boolean semantics rather than silently matching a numeric `1`.

### What still fails

Only numerically-**equal** values now match. Genuinely different numbers still
fail as before — `70.5` does not satisfy an expected `70.0`. This loosens
representation, not correctness.

Covered by `tests/test_grading.py`
(`test_args_int_vs_float_compare_equal`, `test_args_numerically_unequal_still_fails`,
`test_args_bool_not_coerced_to_number`).
