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

Credentials and optional overrides are read from environment variables. Shared
benchmark-executor defaults are in
[`benchmark/benchmark.yaml`](benchmark/benchmark.yaml). The application does
**not** auto-load `.env`; either export its variables into your shell, or pass
the file explicitly on each run with `uv run --env-file .env ...` (shown below).

`LLM_PROVIDER` is `litellm` (the default) or `openai` (the direct
OpenAI-compatible client). Point `LLM_BASE_URL` at any compatible gateway, or
leave it unset to use the provider default.

### OpenAI-compatible gateways

Any OpenAI-compatible gateway can be used through the default `litellm`
provider. Keep the gateway key out of source control and set:

```bash
export LLM_PROVIDER=litellm
export LLM_BASE_URL=https://your-gateway.example.com/v1
export LLM_API_KEY="your-gateway-key"
export LLM_MODEL="your-model-id"
```

To see the model IDs enabled for your key:

```bash
curl -sS \
  -H "Authorization: Bearer ${LLM_API_KEY}" \
  "${LLM_BASE_URL}/models" | jq -r '.data[].id'
```

Use one of those IDs as `LLM_MODEL`. A benchmark agent also needs the selected
model to support OpenAI-style chat completions and function/tool calling; a
model appearing in `/models` does not by itself guarantee those capabilities.

### Executor configuration

Every benchmark run uses the `executor` section of
[`benchmark/benchmark.yaml`](benchmark/benchmark.yaml). Precedence is explicit
CLI argument (where available), then `.env`/environment variable, then the YAML
default. The user simulator and judge inherit the agent API key and base URL
only when their own `*_API_KEY` or `*_BASE_URL` variable is unset.

| Component | YAML default | `.env` overrides |
|---|---|---|
| Agent | `litellm` / `gpt-4o`; temperature `0.0`; reasoning `low`; 16,384 tokens; 240-second step budget | `LLM_PROVIDER`, `LLM_MODEL`, `TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_REASONING_EFFORT`, `AGENT_STEP_TIMEOUT` |
| User simulator | `litellm` / `gpt-5-mini`; structured JSON turns; 1,086 tokens; 4,000 characters; 90-second step budget | `USER_LLM_PROVIDER`, `USER_LLM_MODEL`, `USER_TEMPERATURE`, `USER_MAX_TOKENS`, `USER_MAX_MESSAGE_CHARS`, `USER_STEP_TIMEOUT`, `USER_LLM_REASONING_EFFORT` |
| Semantic judge | `litellm` / `gpt-5-mini`; temperature `0.0`; reasoning `low`; 16,384 tokens; three attempts | `JUDGE_LLM_PROVIDER`, `JUDGE_LLM_MODEL`, `JUDGE_TEMPERATURE`, `JUDGE_MAX_TOKENS`, `JUDGE_MAX_ATTEMPTS`, `JUDGE_LLM_REASONING_EFFORT` |
| Run controls | 30 turns, one trial, seed `20260918`, 1,200-second task timeout | `MAX_TURNS`, `BASE_SEED`, `TRIALS`, `TASK_TIMEOUT`; or `--seed`, `--trials`, and `--max-turns` |

`LLM_REASONING_EFFORT` applies only to the agent. The user simulator and judge
have separate reasoning controls. With the default `gpt-5-mini` reasoning model,
the client omits the temperature parameter because the provider controls it.

For comparable runs, keep the user-simulator and judge connection settings
fixed. Judge calls use strict JSON-schema output, validate quoted evidence
against assistant messages, and are cached by all verdict-affecting inputs under
`bench_output/judge_cache` by default. Judge tokens, cost, and latency are
recorded separately and never attributed to the agent-under-test.

## Running a benchmark

```bash
uv run --env-file .env python -m harness.bench_runner \
    --tasks src/tasks/retail.yaml \
    --config src/agents/retail.yaml \
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

**9 canonical domains, 9 task catalogs, 9 reference multi-agent systems, and
555 graded tasks.** Observable tool behavior and state changes use deterministic
checks; all 465 response criteria across 402 tasks use the semantic LLM judge.
The capability-specific recipes are ported from
[`trailheadapps/agent-script-recipes`](https://github.com/trailheadapps/agent-script-recipes)
into this harness's native format — see
[`docs/agent-script-recipes-port.md`](docs/agent-script-recipes-port.md). The
benchmark manifest is authoritative for domain membership. Capabilities remain task
metadata for filtering and reporting; they are not separate execution suites.

| Domain | Tasks | Capability slices |
|--------|------:|-------------------|
| `retail` | 95 | product QA, inventory, promotions, cart, returns, shipment, external API |
| `customer_support` | 125 | billing, technical support, triage, case submission, specialist routing, churn, delegation, escalation |
| `field_service` | 23 | appointments, work orders, troubleshooting |
| IT service management (ITSM; `itsm`) | 66 | ticket operations, record safety, case hours |
| `payments` | 73 | checkout, transfer, lending, gateway payment, external API |
| `travel` | 93 | hotel booking, trip orchestration, guest schedule, weather, external API |
| `onboarding` | 41 | account setup, task management |
| `sales` | 18 | lead management, outreach, opportunities |
| `marketing` | 21 | campaign analytics, campaign creation |

Each `src/agents/<domain>.yaml` is an optional reference implementation with
multiple specialist nodes. Task correctness is architecture-neutral. To test a
compatible agent of your own, keep the domain task file and replace `--config`.

The frozen contract, shared executor settings, and complete `base` task set
live in [`benchmark/benchmark.yaml`](benchmark/benchmark.yaml). Validate the
lock and all fresh-environment gold paths before a benchmark run:

```bash
uv run python -m harness.benchmark_spec
uv run python -m harness.benchmark_spec --gold
```

All benchmark execution uses the same `harness.bench_runner` command and its
CLI/environment settings.

## Interactive usage

```bash
uv run python -m harness.cli                                          # single-node
uv run python -m harness.graph_cli --config src/agents/customer_support.yaml   # multi-node
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
matching domain file in `src/tasks/` and give it one or more capability tags:

```yaml
- id: cart_promo_and_apply_01
  instruction: |
    You are Steve Sutherland, email ssutherland@example.com.
    Ask if there are promotions for the items in your cart, then ask the
    agent to apply the coupon for the first promotion it lists.
    When the coupon is applied, end the conversation.
  domain: retail
  capabilities: [cart, promotions]
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
  expected_response_semantic:  # meaning may be phrased naturally
    - The assistant must confirm that the SAVE20 promotion was applied to the cart.
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
| `expected_response_semantic` | Plain-language response requirements judged by the strict semantic LLM judge, including exact facts; regex syntax is rejected |
| `expected_state` | The final environment state matches (the side effect happened) |
| `forbidden_state` | A record that must not change was left untouched |
| `forbidden_actions` | A restricted tool was never called (the attempt is the failure) |

Assertion value forms: a scalar (equality), `_not_null` (must be present and
non-empty), `{_contains: "x"}` (substring), and `_min_length` on a collection.

### Customize a domain agent

Define one graph config in `src/agents/<domain>.yaml`: a `system_prompt`, an
`entry_node`, and one or more specialist `nodes`, each with a `description`,
`instructions`, and a list of tools. See `src/agents/retail.yaml`. The benchmark
does not grade graph shape, routing labels, subagent count, or prompt wording.
For a new business domain, add the matching `src/tasks/<domain>.yaml`, policy,
and stateful environment, then register them in `benchmark/benchmark.yaml`.

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
