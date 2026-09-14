# crm-agent-bench

Multi-node ReAct agent benchmark with automated evaluation and LLM-powered user
simulation. The per-node reasoning/tool loop runs on
[Apache Burr](https://burr.dagworks.io/) — this repo has **no proprietary /
vendor agent-runtime dependencies** and installs entirely from public PyPI.

```
                          Task YAML
                             |
                             v
                     [Bench Runner]
                      /          \
              [User Sim]      [Graph Agent]
                  |                |
                  v                v
            "I need a        [Topic Classifier]
             refund..."           |
                  |          +---------+---------+
                  |          |         |         |
                  |       [billing] [support] [general]
                  |          |         |         |
                  |          v         v         v
                  |       [Burr ReAct loop per node]
                  |          |
                  v          v
              ###STOP###   Response
```

## Engine

Each selected graph node runs a ReAct loop — reason with the LLM, execute the
requested tools, feed results back, repeat — implemented as an explicit Apache
Burr state machine (`reason → execute_tools → reason … → finish`). See
`src/harness/agent_burr.py` and `src/harness/graph_agent_burr.py`.

| | Burr engine (default) | Plain engine |
|---|---|---|
| **ReAct loop** | Apache Burr state machine | Plain Python loop |
| **Activate** | Default (`USE_BURR=1`) | `USE_BURR=0` |
| **Dependencies** | `apache-burr[pydantic]` (public PyPI) | none beyond core |

Both engines share the same topic router, per-node tool isolation, and refusal
handling, and drive the same `LLMClient` / `ToolRegistry`.

## Setup

```bash
uv venv
uv pip install -e ".[dev]"     # core + Burr + test deps

cp .env.example .env            # then fill in your API keys
```

Configuration is read from environment variables. The application does **not**
auto-load `.env`; either export the variables into your shell, or pass the file
explicitly on each run with `uv run --env-file .env ...` (shown below).

`LLM_PROVIDER` is `openai` (any OpenAI-compatible endpoint) or `litellm`. Point
`LLM_BASE_URL` at any compatible gateway, or leave it unset for OpenAI.

## Running a benchmark

```bash
uv run --env-file .env python -m harness.bench_runner \
    --tasks src/tasks/eval_csm_graph.yaml \
    --config src/agents/csm_graph.yaml \
    --output bench_output/my_run \
    --max-turns 15 \
    --trials 3        # run each task k times for pass^k reliability (default: 1)
```

Each run produces `traces.json` (conversation logs) and `report.md` (summary
stats, per-task results, tool accuracy). See [`docs/METRICS.md`](docs/METRICS.md)
for what each metric means and how it is computed. Analyze with:

```bash
uv run python -m analysis --help
```

### Agents & tasks

**27 agents across 12 domains, 583 graded tasks.** The seven `commerce`…
`integration` domains below are ported from
[`trailheadapps/agent-script-recipes`](https://github.com/trailheadapps/agent-script-recipes)
into this harness's native format — see
[`docs/agent-script-recipes-port.md`](docs/agent-script-recipes-port.md). Each
ported agent carries ≥20 tasks with a mixed easy/medium/hard spread, plus a
per-domain smoke file (`eval_<domain>_graph_smoke_test.yaml`).

| Agent | Config | Tasks | Domain |
|-------|--------|------:|--------|
| Customer Service | `src/agents/customer_service_graph.yaml` | 13 | `customer_service` |
| Store Manager | `src/agents/csm_graph.yaml` | 21 | `csm` |
| Field Service | `src/agents/field_service_graph.yaml` | 23 | `field_service` |
| SDR | `src/agents/sdr_graph.yaml` | 19 | `sdr` |
| Marketing | `src/agents/marketing_graph.yaml` | 21 | `marketing` |
| Commerce · Product QA | `src/agents/commerce_product_qa_graph.yaml` | 22 | `commerce` |
| Commerce · Cart | `src/agents/commerce_cart_graph.yaml` | 22 | `commerce` |
| Commerce · Returns | `src/agents/commerce_returns_graph.yaml` | 23 | `commerce` |
| Payments · Checkout | `src/agents/payments_checkout_graph.yaml` | 22 | `payments` |
| Payments · Transfer | `src/agents/payments_transfer_graph.yaml` | 22 | `payments` |
| Payments · Loan | `src/agents/payments_loan_graph.yaml` | 22 | `payments` |
| Travel · Hotel Booking | `src/agents/travel_hotel_booking_graph.yaml` | 22 | `travel` |
| Travel · Trip | `src/agents/travel_trip_graph.yaml` | 22 | `travel` |
| Travel · Guest Schedule | `src/agents/travel_guest_schedule_graph.yaml` | 23 | `travel` |
| ITSM · Ticket Ops | `src/agents/itsm_ticket_ops_graph.yaml` | 22 | `itsm` |
| ITSM · Record Safety | `src/agents/itsm_record_safety_graph.yaml` | 22 | `itsm` |
| ITSM · Case Hours | `src/agents/itsm_case_hours_graph.yaml` | 22 | `itsm` |
| Onboarding · Account Setup | `src/agents/onboarding_account_setup_graph.yaml` | 22 | `onboarding` |
| Onboarding · Task Manager | `src/agents/onboarding_task_manager_graph.yaml` | 22 | `onboarding` |
| Support · Triage | `src/agents/support_triage_graph.yaml` | 22 | `support` |
| Support · Churn | `src/agents/support_churn_graph.yaml` | 22 | `support` |
| Support · Specialist Routing | `src/agents/support_specialist_routing_graph.yaml` | 22 | `support` |
| Support · Escalation | `src/agents/support_escalation_graph.yaml` | 22 | `support` |
| Support · Delegation | `src/agents/support_delegation_graph.yaml` | 22 | `support` |
| Support · Case Submission | `src/agents/support_case_submission_graph.yaml` | 22 | `support` |
| Integration · Weather | `src/agents/integration_weather_graph.yaml` | 22 | `integration` |
| Integration · External APIs | `src/agents/integration_external_apis_graph.yaml` | 22 | `integration` |

## Interactive usage

```bash
uv run python -m harness.cli                                          # single-node
uv run python -m harness.graph_cli --config src/agents/customer_service_graph.yaml   # multi-node
```

## Utilities

Optional helper CLIs, run with `python -m`:

```bash
# Convert an AFScript/AgentScript DSL file into a graph config YAML that
# `harness.graph_cli` can run. Prints to stdout, or use -o to write a file.
uv run python -m harness.convert_afscript input.afscript -o agent_graph.yaml

# Compare user-simulator benchmark runs (model × strategy × repeats) into a
# single markdown report. Point it at a run prefix and timestamp.
uv run python -m analysis.usersim_report --prefix bench --timestamp 20260101_120000
```

## Extending the benchmark

The framework is driven by declarative YAML, so the common cases — a new task
or a new agent — need no harness changes. Only the harder case (a new stateful
tool world) requires Python, and it's a single class.

### Add a task (YAML, no code)

A task is a persona instruction plus gold assertions. Only the checks you
assert are graded, so a minimal task is genuinely minimal. Drop it into the
matching suite file in `src/tasks/`:

```yaml
- id: cart_promo_and_apply_01
  instruction: |
    You are Steve Sutherland, email ssutherland@example.com.
    Ask if there are promotions for the items in your cart, then ask the
    agent to apply the coupon for the first promotion it lists.
    When the coupon is applied, end the conversation.
  domain: csm
  difficulty: medium
  expected_actions:            # these tools must be called
    - CSA_Get_Promotion_For_Cart
    - CSA_Apply_Promotion
  expected_order:              # ...in this relative order
    - CSA_Get_Promotion_For_Cart
    - CSA_Apply_Promotion
  expected_args:               # ...with these arguments
    CSA_Apply_Promotion:
      user_email: ssutherland@example.com
      Coupon: SAVE20
  expected_state:              # ...and the cart must end up changed
    carts:
      CSA_Cart9:
        coupon_applied: "SAVE20"
  forbidden_state:             # ...without touching anyone else's cart
    carts:
      CSA_Cart8:
        coupon_applied: _not_null
```

Available assertion keys (all optional; a check counts only when asserted):

| Key | Verifies |
|-----|----------|
| `expected_actions` | The listed tools were called (by name) |
| `expected_args` | A call to a tool matched the given arguments |
| `expected_order` | Listed tools were called in the given relative order |
| `expected_response_contains` | The agent's reply contained a fact (substring, or `re:` regex) |
| `expected_state` | The final environment state matches (the side effect happened) |
| `forbidden_state` | A record that must not change was left untouched |
| `forbidden_actions` | A restricted tool was never called (the attempt is the failure) |

Assertion value forms: a scalar (equality), `_not_null` (must be present and
non-empty), `{_contains: "x"}` (substring), and `_min_length` on a collection.

### Add an agent (two YAML files)

1. Define the agent as a graph config in `src/agents/` — a `system_prompt`, an
   `entry_node`, and one or more `nodes`, each with a `description`,
   `instructions`, and a list of `tools` (by name). See
   `src/agents/csm_graph.yaml` for a full example.
2. Write its tasks in `src/tasks/` (as above), tagged with the same `domain`.
3. Optionally add a one-line run script.

Tools named in the config that are not backed by a registered domain get an
auto-generated stub, so you can stand up and benchmark a new agent before
writing any tool code.

### Add a stateful tool world (one class)

For real side effects (writes, cross-turn state, `expected_state` checks),
register a domain. Subclass `ToolEnvironment`, return the starting data from
`default_state()`, and register tools in `register_tools()`. Tool schemas are
generated from the Python type hints, so there is no separate schema to
maintain. `expected_state` / `forbidden_state` are validated against
`get_state()` (which returns `self.data`).

```python
import json
from harness.tools.environment import ToolEnvironment, register_domain
from harness.tools.registry import ToolRegistry


@register_domain("my_domain")
class MyDomainEnvironment(ToolEnvironment):
    def default_state(self) -> dict:
        return {"records": {"REC-1": {"status": "open"}}}

    def register_tools(self, registry: ToolRegistry) -> None:
        env = self

        @registry.tool("Look up a record by id")
        def get_record(record_id: str) -> str:
            return json.dumps(
                env.data["records"].get(record_id, {"error": "Not found"})
            )

        @registry.tool("Close a record")
        def close_record(record_id: str) -> str:
            rec = env.data["records"].get(record_id)
            if rec is None:
                return json.dumps({"error": "Not found"})
            rec["status"] = "closed"  # persisted write — visible to expected_state
            return json.dumps({"record_id": record_id, "status": "closed"})
```

A task can then assert the write happened:

```yaml
  expected_state:
    records:
      REC-1:
        status: closed
```

See the shipped domains in `src/harness/tools/domains/` for larger examples.

## Tests

```bash
uv run pytest
```

## Community & contributing

- [Contributing guide](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md)
- [License (Apache-2.0)](LICENSE.txt)
