# CRMAgentBench — Metric Definitions

What each CRMAgentBench metric means and how it is computed. This document is
**model-agnostic**: it describes the metrics, not any one model's results.

CRMAgentBench v2 runs the model-under-test across **9 canonical domains and all
555 tasks in one `base` task set**. The versioned domain, task-level capability,
and base-task membership is frozen in `benchmark/benchmark.yaml`. Each task is a scripted,
multi-turn conversation between the agent and a simulated user, with canonical
tools the agent may call. A task is graded pass/fail by a fixed set of checks.

The shared executor configuration in `benchmark/benchmark.yaml` fixes the
simulated user and semantic response judge across models-under-test so the
comparison is fair. Their cost and latency are always **excluded** from the
metrics below, which price and time the agent-under-test only.

---

# Part 1 — Headline metrics

Reports show three numbers per model. Only the first one sets the ranking; the
other two are shown for context.

## pass^1 (the ranking)
```
pass^1 = succeeded / scored
```
The share of tasks the agent solved on its single attempt. This is the accuracy
number and the **only metric that sets the ranking**. Reported per domain and
pooled across all nine domains.

- **succeeded** = tasks that passed every check (see Part 2 for the check list).
- **scored** = tasks that produced a gradeable result. Infra-errored tasks (gateway
  errors, timeouts) are excluded — they count neither for nor against the model.
- The executor runs one trial per task by default. For repeated-trial
  reliability, set `--trials` and report pass^1 separately from pass^k.

## Cost / task (context only)
```
cost / task = mean over tasks of ( agent_cost_usd )
```
Average dollars a single task costs to run. `agent_cost_usd` is per-task and already
**sums every LLM call the agent made across all turns of that task**, so this is the
whole-task cost averaged over the run. Cost prices the **agent-under-test only** —
the fixed user simulator is excluded — using the per-model prices in the pricing
table (`src/harness/llm/pricing.yaml`).

- The denominator is **every task in the run, not just wins**. This answers "what
  does an average task cost," including spend on failed or looping runs. It differs
  from Cost / success (Part 2), which divides total cost by wins only.
- Does not affect the ranking.

## Avg s / turn (context only)
```
avg s / turn = mean over tasks of ( agent_llm_seconds / num_turns )
```
Agent-only latency per conversation turn, **macro-averaged**: compute each task's own
seconds-per-turn first, then take the plain mean across tasks. Every task counts
equally regardless of length, so a few long looping tasks cannot dominate the number
(unlike a pooled Σseconds / Σturns, which is turn-weighted).

- `agent_llm_seconds` is the wall-clock spent inside the model's own chat/stream
  calls, so it **excludes the user simulator, tool calls, and grading** by
  construction. A "turn" (`num_turns`) is one user↔agent exchange; in multi-node
  graph mode a turn expands to ~2.5 LLM calls (router + reasoning + arg calls), so
  per-call latency is roughly this figure ÷ 2.5.
- This includes prompt processing / TTFT, so it is **user-perceived per-turn
  latency, not decode-only speed**. Tasks with `num_turns == 0` (infra-errored before
  any turn) are skipped.
- Lower is snappier. Does not affect the ranking.

## `n/a` values
`n/a` means that model's per-task traces do not carry the field. Older or merged
multi-model runs recorded accuracy but not per-task `agent_cost_usd` or
`agent_llm_seconds`, so their Cost / task and Avg s / turn show `n/a`.

---

# Part 2 — Other available metrics

These are computed and stored in the run's summary/detail CSVs and reports, but are
**not** among the headline numbers. Use them for deeper analysis.

## The pass/fail checks
A task is solved only when **every** check passes. There is no partial credit.

| Check | What it verifies |
|---|---|
| **actions** | The agent called the required tools. |
| **state** | The resulting system state is correct (e.g. a ticket was created). |
| **args** | The tool arguments were correct (right ids, fields, values). |
| **response** | User-facing messages communicate every semantic criterion, including exact atomic facts. |
| **order** | Steps happened in the required sequence, where order matters. |
| **collateral** | Required side effects exist (e.g. a drafted email). |
| **forbidden** | The agent did **not** do something it should have refused. |

All response grading uses declarative plain-language criteria evaluated by the
frozen semantic LLM judge. This includes IDs, amounts, dates, codes, paraphrases,
refusals, comparisons, availability conclusions, and negation-sensitive
requirements. Regex syntax is not allowed in the semantic field. Each judge
verdict is strict structured JSON and records criterion-level evidence copied
from assistant messages. Each criterion receives only a binary `true`/`false`
verdict, with no partial credit. The harness computes the final response pass
locally as a strict logical AND, so one unmet criterion fails the whole response.
It caches validated verdicts by content hash and treats judge request or
validation failures as infrastructure errors rather than agent failures.

## Success by difficulty
Tasks are tagged **easy / medium / hard**. Reporting pass^1 per band shows where a
model breaks down — strong models still tend to lose points on *hard*.

## Outcome status
Every task ends in one status:

| Status | Meaning |
|---|---|
| **PASS** | Finished and passed every check. |
| **MAX_TURNS** | Hit the turn cap before finishing. |
| **RESPONSE_FAIL** | Final message missed required content. |
| **ARGS_FAIL** | A tool was called with wrong arguments. |
| **ACTIONS_MISS** | A required tool was never called (or the wrong one was). |
| **FORBIDDEN_FAIL** | The agent did something it should have refused. |
| **STATE_FAIL** | The end state was wrong. |

`MAX_TURNS` is a **stopping condition, not a verdict**. A task that hits the turn cap
can still be a PASS if its checks already passed. Reconcile:
`PASS_total = PASS-status tasks + max-turns tasks that still passed`.

## pass^k
Solved within k attempts. Only applies when a run uses more than 1 trial per task.
When trials > 1, report pass^k separately — do not conflate it with single-trial
pass^1.

## Avg turns
Mean conversation turns per task. Lower means the agent reaches the goal with less
back-and-forth.

## Avg tool calls
Mean tool invocations per task. Lower usually means fewer tokens, but too low can
mean the agent skipped a required step — read it alongside pass^1.

## Total agent cost
- **Total agent cost** = sum of token cost over all tasks in a run. Absolute spend
  to run the selected domain. (Cost / task, in Part 1, is the mean of the per-task cost.)

## Cost / success
```
cost / success = total agent cost / tasks succeeded
```
Average dollars spent per **solved** task. Charges all spend — including tokens
burned on failed or looping runs — against the tasks that actually succeeded, so a
model that is cheap per token but loops a lot will not look artificially good. This
is a **value** metric, useful for value comparisons but not used to rank.

## Tokens / task
```
tokens / task = total agent tokens / tasks scored
```
Average total tokens (prompt + completion, agent only) the model spent per task —
the **efficiency** metric. **Lower is better**: a model that reaches the answer in
few turns spends fewer tokens; one that loops or over-explains spends more. It
catches looping that a raw speed metric hides — a model can decode tokens fast yet
still be inefficient because it generates far more of them.

## Output tokens / second
```
output tokens / second = scored completion tokens / agent inference seconds
```
Agent-only **decode speed**. Agent inference seconds (`agent_llm_seconds`) is the
wall-clock inside the model's own chat/stream calls, excluding the user simulator,
tool calls, and grading. Pool as a ratio of the summed totals across all 9 domains —
do not average the per-domain rates. This is **raw speed, not efficiency**: a
fast-but-looping model scores well here even though it does more total work, so read
it alongside Tokens / task, not instead of it. (This differs from Avg s / turn in
Part 1, which measures per-turn latency including prompt processing, not decode-only
throughput.)

## Retired metrics
- **Avg time per task** — mean end-to-end wall-clock per task. Removed because it
  includes the user simulator's latency, tool round-trips, and grading, so a slow
  user-sim run made a fast model look slow. For an agent-only latency signal, use
  Avg s / turn (Part 1), which times only the model's own inference calls.

---

## How to read the numbers together

- **Accuracy first.** Rank on pass^1. Use Cost / success and Tokens / task to judge
  value and efficiency, not to rank.
- **Speed and efficiency are different axes.** Output tokens / second is decode
  speed; Tokens / task is total work. A model can be fast (high tok/s) yet wasteful
  (high tokens / task) if it loops — read the two together.
- **Low turns or tool calls is only good if pass^1 stays high** — otherwise the
  agent is skipping work, not being efficient.
- **Watch `scored` vs total tasks.** If many tasks were infra-errored and dropped,
  pass^1 is over a smaller base and is less comparable. Aim for `scored` = total;
  re-run infra-errored tasks.
- **`MAX_TURNS` failures often mean "found the answer, never committed it"** — the
  agent gathered the right data but did not fire the final mutating call before the
  cap. Different from getting the answer wrong.
- **User-simulator variance is real.** The simulated user can drift from its script
  and starve the agent of a needed turn. A single `MAX_TURNS` failure on a long or
  cross-topic task may be a sim artifact — confirm with a re-run before blaming the
  model.
