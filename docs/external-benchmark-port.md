# Porting External Tool-Use Benchmarks & Agent Galleries into crm-agent-bench

## Why (and how this differs from the agent-script-recipes port)

[`docs/agent-script-recipes-port.md`](agent-script-recipes-port.md) covered
porting one Salesforce source (`trailheadapps/agent-script-recipes`). This doc
covers a *second, distinct* expansion: porting from **public tool-use benchmarks
and agent-example galleries** to add business domains that source repo did not
cover. It is committed separately from the agent-script-recipes port.

The selection bar is deliberately narrow, because we need **reimplementable data
and tools**, not prompts:

1. **Concrete seed data** — a JSON/CSV "database" of records we can drop into a
   domain's `default_state()`.
2. **Explicit tool/function signatures** — typed methods we can reimplement as
   pure-Python closures over `self.data`.
3. **Deterministic grading** — the source grades on tool-call sequence and/or
   final state (not an LLM judge), so its tasks map onto our assertions.
4. **A permissive license** — MIT / Apache-2.0 / MIT-0. Reimplementation with
   attribution; no copying of code under non-commercial or research-only terms.

## Selected sources

| Source | License | Domain added | Agents |
|---|---|---|---|
| [`sierra-research/tau2-bench`](https://github.com/sierra-research/tau2-bench) | MIT | `telecom` | telecom_troubleshooting, telecom_billing |
| [`olly-styles/WorkBench`](https://github.com/olly-styles/WorkBench) | MIT | `workspace` | workspace_email, workspace_calendar, workspace_project |
| [`ShishirPatil/gorilla`](https://github.com/ShishirPatil/gorilla) (BFCL multi-turn `trading_bot.py`) | Apache-2.0 | `finance` | finance_brokerage |
| [`awslabs/amazon-bedrock-agent-samples`](https://github.com/awslabs/amazon-bedrock-agent-samples) (`mortgage_assistant`) | Apache-2.0 | `mortgage` | mortgage_advisor |
| [`anthropics/anthropic-cookbook`](https://github.com/anthropics/anthropic-cookbook) (`threat_intel_enrichment_agent`) | MIT | `security` | security_threat_intel |
| [`openai/openai-cookbook`](https://github.com/openai/openai-cookbook) (`dispute_agent`) | MIT | `dispute` | dispute_investigator |

WorkBench ships an integrated workspace (email + calendar + project management +
analytics + CRM) that share one seeded database, so it is modeled as a **single
`workspace` domain with three topic-routed agents**, not three domains — faithful
to how the source's tools actually share state.

### Rejected sources (and why)
- **CRMArena / CRMArena-Pro** — CC BY-NC 4.0 (non-commercial); data lives in a
  live Salesforce org, not a portable seed dict; heavy overlap with existing
  domains. (Its CPQ + confidentiality *task-design* ideas are worth borrowing as
  inspiration, but nothing is copied.)
- **ToolBench / ToolAlpaca** — research-only / LLM-simulated data + judge-based
  eval; not deterministic.
- **AppWorld** — Apache-2.0 code but the seed databases ship *encrypted* (an
  anti-contamination measure), so data cannot be copied; deferred as a
  higher-effort future source (reimplement from public `.pyi` signatures only).

## How an external benchmark maps to native format

| External-benchmark concept | crm-agent-bench native |
|---|---|
| A domain's seed `db.json` / CSV database | domain `default_state()` + per-task `initial_state` overrides |
| A typed tool method (`tools.py` / simulator class) | a **tool** — Python closure over `self.data`, schema auto-derived from type hints |
| A tool group / subagent topic | a graph **node** (topic-routed by the classifier) |
| `evaluation_criteria.actions` (ordered ground-truth calls) | `expected_actions` + `expected_order` (+`expected_order_mode: exact`) |
| Final-state diff / "harmful side effect" checks | `expected_state` / `forbidden_state` + `forbidden_actions` |
| A benchmark task's user goal | task `instruction` (2nd-person persona with stop condition) |

## Native conventions (identical to the recipe port)

Same as [`docs/agent-script-recipes-port.md`](agent-script-recipes-port.md):

- **Domain** `src/harness/tools/domains/<domain>.py`: subclass `ToolEnvironment`,
  `@register_domain("<domain>")`, `default_state() -> dict`, `register_tools`.
  Write tools return a business-rejection JSON with `status_code` in
  `{403, 409, 422}` when refusing a valid-but-disallowed request — such calls
  still count as a successful action for `expected_actions`.
- **Agent** `src/agents/<domain>_<recipe>_graph.yaml`: `system_prompt`,
  `entry_node`, `nodes.<name>` with `description` / `instructions` /
  `tools: [{name, description}]`. Every tool name must exist in the domain
  registry.
- **Task** `src/tasks/eval_<domain>_<recipe>_graph.yaml`: `tasks:` list; required
  `id` + `instruction`; optional `domain`, `initial_state`, `difficulty`, and the
  assertion keys. **≥20 tasks per agent, mixed easy/medium/hard.** Plus one
  per-domain smoke file `eval_<domain>_graph_smoke_test.yaml`.

### Grading gotcha to respect (same as before)
`expected_state` / `forbidden_state` support ONLY: scalar equality, `"_not_null"`,
and the path form `<key>._min_length: N`. **Do NOT use `{_contains: ...}` inside
state blocks** — that flattens to an unsatisfiable path. `{_contains}` is valid
ONLY inside `expected_args` / `forbidden_args`.

## CI wiring
- `src/harness/tools/domains/__init__.py` — import each new domain module.
- `tests/test_config_validation.py` — add each agent graph to `_DOMAIN_CONFIGS`
  and each task/smoke file to `_TASK_FILES`.

## Verification
1. `uv run ruff check src tests && uv run ruff format --check`
2. `uv run pytest tests/test_config_validation.py -q` (primary gate)
3. `uv run pytest -q` (full suite)
4. Per-domain end-to-end tool-execution spot check against merged `initial_state`.
5. Counts: `ls src/agents/*.yaml | wc -l`; total task count.

## Attribution
Each ported domain module carries a header comment naming its upstream source
and license. No upstream code is copied verbatim — data shapes and tool
signatures are reimplemented as pure-Python `ToolEnvironment` closures.

## Out of scope
- No runtime/grading/Burr changes.
- No live external API calls — all tool side effects are pure-Python mutations on
  `self.data`.
