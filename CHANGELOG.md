# Changelog

## Benchmark 2.5.0 / Dataset 1.4.0

- Raise the global per-task wall-clock budget from 600 to 1,200 seconds so
  slower LLM infrastructure and long conversations can complete without
  weakening the separate agent- and simulator-step timeout safeguards.
- Centralize executor setting resolution as environment-over-YAML, require
  explicit task budgets at the low-level runner boundary, and prevent agent
  reasoning settings from leaking into LiteLLM clients used by other roles.
- Fix spurious semantic-judge infrastructure errors caused by exact-quote
  validation rejecting faithful evidence: normalize typographic punctuation
  (curly quotes, en/em dashes, the minus sign) and markdown emphasis before
  the containment check, accept evidence that spans a turn boundary, and let
  the judge cite a criterion's supporting facts as multiple independent
  quotes instead of splicing non-adjacent excerpts into one fabricated
  string. Bumps the judge prompt to `semantic-communication-v5`.
- payments: enforce frozen source and destination status in transfer validation
  and execution, and require auditable validation for invalid transfers.
- payments: add an idempotent, receipt-backed offline loyalty adjustment instead
  of laundering the direct-loyalty task through a normal checkout.
- payments: repair malformed transfer assertions and make checkout stop points
  observable between payment, receipt, and later side effects.
- payments: clarify the existing classifier node boundaries for account
  information, account transfers, bank-funded checkout, and gateway tokens.

## Benchmark 2.4.0 / Dataset 1.3.0

- itsm: repair contradictory or misaligned action contracts, invalid
  case-management redirects, redundant closed-ticket mutations, and deletion
  tasks whose gold paths omitted required safety checks.
- itsm: enforce dependency and ticket-scoped authorization preconditions in the
  deletion backend, and expose dependency status so agents can make the guarded
  decision from tool evidence.
- Clarify classifier safety criteria for internal-instruction probes while
  preserving benign user formatting requests, and constrain the user simulator
  from inventing new acceptance criteria after the requested work is complete.
- Reject task definitions that both require and forbid an action or attach
  expected arguments to an undeclared expected action.
- Use the authoritative dependency-status check for dependency-aware closure,
  assert successful deletion through target absence, grade both confidentiality
  probes in the reveal-then-hours task, and reject duplicate YAML keys during
  direct task loading.

## Benchmark 2.3.0 / Dataset 1.2.0

- field_service: make technician pre-work briefs expose a canonical service
  request and structured customer troubleshooting attempts; repair the
  inconsistent keyboard fixture and select the performance checklist from its
  explicit issue category instead of ambiguous keyword precedence.
- field_service: enforce verified appointment ownership in the tool backend.
  Policy files remain evaluation artifacts; the cross-account refusal task
  retains attempt-level grading because the user explicitly identifies the
  appointment as belonging to another customer.

## Benchmark 2.2.0 / Dataset 1.1.0

- customer_support: consolidate the `case_submission` node onto `create_case`
  (dropping the redundant `submit_case` tool) and give the `escalation` node
  `escalate_case` alongside `escalate_to_human`, so the two nodes' tool
  surfaces overlap with the rest of the reference graph instead of exposing
  disjoint, near-duplicate case-management tools.
- Constrain `create_case`'s `priority` argument to a `high`/`medium`/`low`
  enum at the tool-schema layer (registry now emits JSON Schema `enum` for
  `Literal` type hints), structurally nudging models away from non-canonical
  paraphrases like "urgent" or "Critical / P0".
- Add a `_contains_any` args-grading assertion for genuinely free-text
  arguments (e.g. a knowledge-base search query) where several phrasings are
  equally valid and a single literal substring is too brittle.

## Benchmark 2.0.0 / Dataset 1.1.0

- Replace every response assertion with 502 declarative, plain-language semantic
  requirements across 429 tasks, all evaluated by a frozen LLM judge; task
  loading rejects both the legacy response field and regex syntax.
- Keep tool calls, arguments, order, final state, and forbidden behavior under
  deterministic grading.
- Add strict structured judge output, exact evidence validation, prompt-injection
  isolation, content-addressed caching, retry/parse safeguards, and trace-level
  judge provenance, usage, cost, and criterion evidence.
- Require a strict binary verdict for every response criterion and compute the
  response result as a logical AND: one unmet criterion fails the whole response.
- Clarify that conjunctions require every stated fact while explicit alternatives
  require any one allowed fact, avoiding accidental over-grading of `or` criteria.
- Exclude semantic-judge infrastructure failures from the scored denominator
  instead of attributing them to the agent-under-test.

## Benchmark 1.1.0 / Dataset 1.0.0

- Consolidate the benchmark into one canonical task YAML and one optional
  reference-agent YAML for each of the nine business domains.
- Replace the 29 suite execution units with task-level capability tags; all 583
  semantically unchanged tasks remain in the single `base` task set.
- Allow paper-protocol runs to use a compatible agent config instead of
  requiring the shipped reference agent.
- Move smoke-only task YAMLs to `tests/fixtures/tasks/` so `src/tasks/` is the
  canonical public dataset surface.

## Benchmark 1.0.0 / Dataset 1.0.0

- Freeze nine canonical business domains and 29 capability-oriented suites.
- Freeze all 583 globally unique paper tasks in one `base` task set; paper
  evaluation runs the complete set without development, validation, or training
  partitions.
- Version policies, canonical tool contracts, initial state, grading, simulator,
  repeated-trial protocol, and result metadata in a deterministic lockfile.
- Add full fresh-environment gold-path validation with an explicit `583/583`
  success report and a stratified human audit.
