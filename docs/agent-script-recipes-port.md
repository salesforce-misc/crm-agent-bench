# Porting Agent Script Recipes into crm-agent-bench

## Why

To grow the benchmark's **agent and task count**, we port the substantive,
tool-using recipes from the public
[`trailheadapps/agent-script-recipes`](https://github.com/trailheadapps/agent-script-recipes)
repo into this harness's native format. That repo has 32 AgentScript (`.agent`)
recipes across commerce, payments, travel, IT service, onboarding, customer
support, and external-integration domains.

Before this change the benchmark shipped **5 domains, 5 agent graphs, ~11 task
files**. After: **+7 domains, ~22 new agent graphs (5 → ~27), 500+ new tasks**
(≥20 per agent, mixed easy/medium/hard), plus a smoke file per domain.

The ~9 pure AgentScript-*syntax* demos (HelloWorld, LanguageSettings,
ListVariables, VariableManagement, SystemInstructionOverrides,
ActionDescriptionOverrides, AdvancedInputBindings, AfterReasoning,
ReasoningInstructions, ContextHandling) are intentionally **skipped** — they
teach language features, not tool use, so they don't grade as deterministic
tool-call tasks.

## How a recipe maps to native format

| AgentScript recipe concept | crm-agent-bench native |
|---|---|
| A recipe (agent + subagents) | one **agent graph YAML** `src/agents/<domain>_<recipe>_graph.yaml` |
| A recipe *subagent* | a graph **node** (topic-routed by the classifier) |
| A recipe action (`flow://` / `apex://`) | a **tool** (Python closure over `self.data`) |
| `ASR_*` objects + `data/*.json` seed records | domain `default_state()` + per-task `initial_state` |
| README "Try It Out" dialogue | task `instruction` persona + deterministic assertions |

## Domain grouping (7 new domains, 22 agents)

| Domain | Agents (recipe → graph) | Representative tools |
|---|---|---|
| `commerce` | product_qa (SimpleQA), cart (TemplateExpressions), returns (OpenGateRouter) | get_product, list_products, search_products, add_to_cart, get_cart, get_recommendations, get_order_status, authenticate_customer, process_return |
| `payments` | checkout (ActionChaining), transfer (ErrorHandling), loan (ConditionalLogicPatterns) | process_payment, send_receipt, award_loyalty_points, log_transaction, get_account_balance, validate_transfer, execute_transfer, get_credit_score, evaluate_loan |
| `travel` | hotel_booking (MultiSubagentNavigation), trip (MultiSubagentOrchestration), guest_schedule (PromptTemplateActions) | search_hotels, get_hotel, create_hotel_booking, confirm_booking, search_flights, book_flight, book_car, send_itinerary, generate_schedule |
| `itsm` | ticket_ops (AvailableWhenFiltering), record_safety (SafetyAndGuardrails), case_hours (InstructionActionReferences) | get_ticket, list_tickets, assign_ticket, reassign_ticket, close_ticket, escalate_ticket, update_ticket_status, add_ticket_comment, delete_ticket, check_dependencies, check_permission, get_business_hours |
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

**Agent** (`src/agents/<domain>_<recipe>_graph.yaml`) — `system_prompt`,
`entry_node` (must be a node key), and `nodes.<name>` with `description`
(classifier routing hint), `instructions`, and `tools: [{name, description}]`.
Every tool name must exist in the bound domain registry. Model:
`src/agents/sdr_graph.yaml`, `src/agents/customer_service_graph.yaml`.

**Task** (`src/tasks/eval_<domain>_<recipe>_graph.yaml`, a `tasks:` list) —
required `id` and `instruction` (2nd-person persona for the user simulator, with
a stop condition). Optional `domain`, `initial_state` (deep-merged over
defaults), `difficulty` (`easy|medium|hard`), and assertions: `expected_actions`,
`expected_args` (+`expected_args_mode`), `expected_order`
(+`expected_order_mode`), `expected_response_contains` (substring or `re:`),
`expected_state`, `forbidden_actions`/`forbidden_args`/`forbidden_state`. Unknown
keys fail to load. Model: `src/tasks/eval_customer_service_graph.yaml`.

## CI wiring

New domains, agents, and task files must be registered so nothing escapes
validation:

- `src/harness/tools/domains/__init__.py` — import each new domain module so its
  `@register_domain` runs.
- `tests/test_config_validation.py` — add each new agent graph to
  `_DOMAIN_CONFIGS` and each task file to `_TASK_FILES`. The parametrized guards
  then assert every tool referenced by an agent YAML or task exists in the
  domain registry, that `entry_node` is valid, and that each task's `domain`
  matches its file.

## Verification

1. `uv run ruff check src tests && uv run ruff format --check`
2. `uv run pytest tests/test_config_validation.py -q` (primary correctness gate)
3. `uv run pytest -q` (full suite)
4. End-to-end spot check per domain:
   `uv run --env-file .env python -m harness.bench_runner --tasks src/tasks/eval_commerce_product_qa_graph.yaml --config src/agents/commerce_product_qa_graph.yaml --trials 1`
5. Confirm counts: `ls src/agents/*.yaml | wc -l` (~27); total task count 500+.

## Out of scope

- No changes to the runtime engine, grading, or Burr wiring.
- No Salesforce Flow/Apex — tool side effects are pure-Python mutations on
  `self.data` (the harness has no vendor runtime, by design).
