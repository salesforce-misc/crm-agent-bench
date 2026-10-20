# Porting Agent Script Recipes into crm-agent-bench

> The implementation-lineage names in this document predate the paper freeze.
> In benchmark v1, `commerce` is exposed as `retail`, `support` as
> `customer_support`, and former `integration` tasks are distributed across
> travel weather, retail shipment, and payments gateway capabilities. The canonical
> mapping is `benchmark/benchmark.yaml`.

## Why

To grow the benchmark's **agent and task count**, we port the substantive,
tool-using recipes from the public
[`trailheadapps/agent-script-recipes`](https://github.com/trailheadapps/agent-script-recipes)
repo into this harness's native format. That repo has 32 AgentScript (`.agent`)
recipes across commerce, payments, travel, IT service, onboarding, customer
support, and external-integration domains.

At the time of the original recipe port, the benchmark grew from **5 domains,
5 agent graphs, and ~11 task files** by adding **7 implementation-lineage
domains, ~22 agent graphs, and 500+ tasks** (≥20 per agent, mixed
easy/medium/hard), plus smoke coverage. The paper-freeze consolidation described
above now exposes those assets as **9 canonical domain task catalogs and 9
reference multi-agent graphs**. The former recipe boundaries survive only as
capability tags and specialist nodes.

The ~9 pure AgentScript-*syntax* demos (HelloWorld, LanguageSettings,
ListVariables, VariableManagement, SystemInstructionOverrides,
ActionDescriptionOverrides, AdvancedInputBindings, AfterReasoning,
ReasoningInstructions, ContextHandling) are intentionally **skipped** — they
teach language features, not tool use, so they don't grade as deterministic
tool-call tasks.

## How a recipe maps to native format

| AgentScript recipe concept | crm-agent-bench native |
|---|---|
| A recipe (agent + subagents) | specialist nodes inside `src/agents/<domain>.yaml` |
| A recipe *subagent* | a namespaced graph **node** (topic-routed by the classifier) |
| A recipe action (`flow://` / `apex://`) | a **tool** (Python closure over `self.data`) |
| `ASR_*` objects + `data/*.json` seed records | domain `default_state()` + per-task `initial_state` |
| README "Try It Out" dialogue | task `instruction` persona + deterministic assertions |

## Original recipe grouping

| Domain | Agents (recipe → graph) | Representative tools |
|---|---|---|
| `commerce` | product_qa (SimpleQA), cart (TemplateExpressions), returns (OpenGateRouter) | get_product, list_products, search_products, add_to_cart, get_cart, get_recommendations, get_order_status, authenticate_customer, process_return |
| `payments` | checkout (ActionChaining), transfer (ErrorHandling), loan (ConditionalLogicPatterns) | process_payment, send_receipt, award_loyalty_points, log_transaction, get_account_balance, validate_transfer, execute_transfer, get_credit_score, evaluate_loan |
| `travel` | hotel_booking (MultiSubagentNavigation), trip (MultiSubagentOrchestration), guest_schedule (PromptTemplateActions) | search_hotels, get_hotel, create_hotel_booking, confirm_booking, search_flights, book_flight, book_car, send_itinerary, generate_schedule |
| IT service management (ITSM; `itsm`) | ticket_ops (AvailableWhenFiltering), record_safety (SafetyAndGuardrails), case_hours (InstructionActionReferences) | get_ticket, list_tickets, assign_ticket, reassign_ticket, close_ticket, escalate_ticket, update_ticket_status, add_ticket_comment, delete_ticket, check_dependencies, check_permission, get_business_hours |
| `onboarding` | account_setup (MultiStepWorkflows), task_manager (ComplexStateManagement) | create_customer_account, configure_account_settings, send_verification_email, finalize_onboarding, add_task, complete_task, list_tasks, calculate_task_stats |
| `support` | triage (CustomerServiceAgent), churn (AdvancedReasoningPatterns), specialist_routing (BidirectionalNavigation), escalation (EscalationPatterns), delegation (TrueDelegation), case_submission (CustomLightningTypes) | fetch_customer, classify_issue, search_knowledge_base, create_case, escalate_case, fetch_account_data, fetch_order_history, fetch_support_history, fetch_user_profile, assess_churn_risk, calculate_customer_value, generate_recommendations, submit_case, consult_specialist, escalate_to_human |
| `integration` | weather (ActionDefinitions), external_apis (ExternalAPIIntegration) | get_current_weather, get_weather_forecast, get_weather_alert, track_shipment, process_gateway_payment |

## Native conventions

**Domain** (`src/harness/tools/domains/<domain>.py`) — subclass `ToolEnvironment`,
decorate `@register_domain("<domain>")`, implement `default_state() -> dict` and
`register_tools(self, registry)` with tools as closures over `self.data`. Tool
JSON-schema is auto-derived from **type hints** + the docstring/decorator
description. Write tools return a business-rejection JSON with `status_code` in
`{403, 409, 422}` when refusing a valid-but-disallowed request — such calls still
count as a successful action for `expected_actions`. Model:
`src/harness/tools/domains/customer_service.py`; base:
`src/harness/tools/environment.py`.

**Agent** (`src/agents/<domain>.yaml`) — `system_prompt`,
`entry_node` (must be a node key), and `nodes.<name>` with `description`
(classifier routing hint), `instructions`, and `tools: [{name, description}]`.
Every tool name must exist in the bound domain registry. Models:
`src/agents/sales.yaml`, `src/agents/customer_support.yaml`.

**Task** (`src/tasks/<domain>.yaml`, a `tasks:` list) —
required `id` and `instruction` (2nd-person persona for the user simulator, with
a stop condition). Frozen tasks also declare `domain`, one or more `capabilities`,
and `difficulty`. Optional `initial_state` is deep-merged over
defaults. Supported assertions include `expected_actions`,
`expected_args` (+`expected_args_mode`), `expected_order`
(+`expected_order_mode`), `expected_response_semantic` (plain-language LLM-judge
requirements), `expected_state`,
`forbidden_actions`/`forbidden_args`/`forbidden_state`. Unknown
keys fail to load. Model: `src/tasks/customer_support.yaml`.

## CI wiring

New domains must be registered so nothing escapes validation:

- `src/harness/tools/domains/__init__.py` — import each new domain module so its
  `@register_domain` runs.
- `benchmark/benchmark.yaml` — add the domain's one task file, one reference
  agent, and policy. Parametrized guards derive the canonical files and assert
  that every tool reference exists, each `entry_node` is valid, and each task's
  `domain` matches its file.

## Verification

1. `uv run ruff check src tests && uv run ruff format --check`
2. `uv run pytest tests/test_config_validation.py -q` (primary correctness gate)
3. `uv run pytest -q` (full suite)
4. End-to-end spot check per domain:
   `uv run --env-file .env python -m harness.bench_runner --tasks src/tasks/retail.yaml --config src/agents/retail.yaml --trials 1`
5. Confirm exactly 9 agent YAMLs, 9 task YAMLs, and 555 unique task IDs.

## Out of scope

- No changes to the runtime engine, grading, or Burr wiring.
- No Salesforce Flow/Apex — tool side effects are pure-Python mutations on
  `self.data` (the harness has no vendor runtime, by design).
