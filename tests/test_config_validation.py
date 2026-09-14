"""Tests for config validation — tool name checking at graph startup."""

from pathlib import Path

import pytest
import yaml

from harness.bench_runner import load_tasks
from harness.graph_cli import _build_nodes_from_config
from harness.tools.registry import ToolRegistry


def _write_tasks(tmp_path: Path, task: dict) -> str:
    path = tmp_path / "tasks.yaml"
    with open(path, "w") as f:
        yaml.safe_dump({"tasks": [task]}, f)
    return str(path)


_MINIMAL_TASK = {"id": "t1", "instruction": "do the thing"}


class TestTaskLoadValidation:
    def test_minimal_task_loads(self, tmp_path):
        tasks = load_tasks(_write_tasks(tmp_path, dict(_MINIMAL_TASK)))
        assert len(tasks) == 1
        assert tasks[0].id == "t1"

    def test_unknown_key_raises(self, tmp_path):
        # `expected_action` (singular) is a common typo that would otherwise
        # silently make the task assert nothing.
        task = {**_MINIMAL_TASK, "expected_action": ["get_invoice"]}
        with pytest.raises(ValueError, match="unknown key"):
            load_tasks(_write_tasks(tmp_path, task))

    def test_missing_instruction_raises(self, tmp_path):
        with pytest.raises(ValueError, match="instruction"):
            load_tasks(_write_tasks(tmp_path, {"id": "t1"}))

    def test_invalid_difficulty_raises(self, tmp_path):
        task = {**_MINIMAL_TASK, "difficulty": "trivial"}
        with pytest.raises(ValueError, match="difficulty"):
            load_tasks(_write_tasks(tmp_path, task))

    def test_invalid_args_mode_raises(self, tmp_path):
        task = {**_MINIMAL_TASK, "expected_args_mode": "all"}
        with pytest.raises(ValueError, match="expected_args_mode"):
            load_tasks(_write_tasks(tmp_path, task))

    def test_invalid_order_mode_raises(self, tmp_path):
        task = {**_MINIMAL_TASK, "expected_order_mode": "strict"}
        with pytest.raises(ValueError, match="expected_order_mode"):
            load_tasks(_write_tasks(tmp_path, task))

    def test_valid_enums_pass(self, tmp_path):
        task = {
            **_MINIMAL_TASK,
            "difficulty": "hard",
            "expected_args_mode": "first",
            "expected_order_mode": "exact",
        }
        tasks = load_tasks(_write_tasks(tmp_path, task))
        assert tasks[0].difficulty == "hard"
        assert tasks[0].expected_args_mode == "first"
        assert tasks[0].expected_order_mode == "exact"


def _make_registry_with_tools(*names: str) -> ToolRegistry:
    registry = ToolRegistry()
    for name in names:

        def _stub(**kwargs):
            return "stub"

        _stub.__name__ = name
        registry.register(_stub, name=name, description=f"Stub {name}")
    return registry


class _FakeEnvironment:
    def create_registry(self):
        return _make_registry_with_tools("get_invoice", "check_payment")


class TestToolNameValidation:
    def test_valid_tools_pass(self):
        config = {
            "nodes": {
                "billing": {
                    "tools": [{"name": "get_invoice"}, {"name": "check_payment"}],
                    "instructions": "Handle billing",
                    "description": "Billing node",
                },
            },
        }
        env = _FakeEnvironment()
        nodes = _build_nodes_from_config(config, environment=env)
        assert "billing" in nodes
        assert len(nodes["billing"].tools.get_definitions()) == 2

    def test_unknown_tool_raises_value_error(self):
        config = {
            "nodes": {
                "billing": {
                    "tools": [
                        {"name": "get_invoice"},
                        {"name": "nonexistent_tool"},
                    ],
                    "instructions": "Handle billing",
                    "description": "Billing node",
                },
            },
        }
        env = _FakeEnvironment()
        with pytest.raises(ValueError, match="nonexistent_tool"):
            _build_nodes_from_config(config, environment=env)

    def test_error_message_lists_available_tools(self):
        config = {
            "nodes": {
                "billing": {
                    "tools": [{"name": "bad_tool"}],
                },
            },
        }
        env = _FakeEnvironment()
        with pytest.raises(ValueError, match="get_invoice"):
            _build_nodes_from_config(config, environment=env)

    def test_stub_mode_skips_validation(self):
        config = {
            "nodes": {
                "billing": {
                    "tools": [{"name": "anything_goes", "description": "stub"}],
                },
            },
        }
        nodes = _build_nodes_from_config(config, environment=None)
        assert "billing" in nodes
        assert len(nodes["billing"].tools.get_definitions()) == 1


# ---------------------------------------------------------------------------
# Integration: validate real agent YAMLs against their domain registries
# ---------------------------------------------------------------------------

_AGENTS_DIR = Path(__file__).resolve().parent.parent / "src" / "agents"
_TASKS_DIR = Path(__file__).resolve().parent.parent / "src" / "tasks"

# Every first-class domain that ships an agent config. Discovered statically so
# a NEW domain agent/task file fails CI until it's added here AND validates,
# rather than silently going unchecked (P1.3 — was only csm + customer_service).
_DOMAIN_CONFIGS = [
    ("csm", _AGENTS_DIR / "csm_graph.yaml"),
    ("customer_service", _AGENTS_DIR / "customer_service_graph.yaml"),
    ("field_service", _AGENTS_DIR / "field_service_graph.yaml"),
    ("marketing", _AGENTS_DIR / "marketing_graph.yaml"),
    ("sdr", _AGENTS_DIR / "sdr_graph.yaml"),
    # Ported agent-script recipes (7 new domains, 22 new agent graphs). Multiple
    # graphs share one domain, so the parametrize ids below key on file stem.
    ("commerce", _AGENTS_DIR / "commerce_product_qa_graph.yaml"),
    ("commerce", _AGENTS_DIR / "commerce_cart_graph.yaml"),
    ("commerce", _AGENTS_DIR / "commerce_returns_graph.yaml"),
    ("payments", _AGENTS_DIR / "payments_checkout_graph.yaml"),
    ("payments", _AGENTS_DIR / "payments_transfer_graph.yaml"),
    ("payments", _AGENTS_DIR / "payments_loan_graph.yaml"),
    ("travel", _AGENTS_DIR / "travel_hotel_booking_graph.yaml"),
    ("travel", _AGENTS_DIR / "travel_trip_graph.yaml"),
    ("travel", _AGENTS_DIR / "travel_guest_schedule_graph.yaml"),
    ("itsm", _AGENTS_DIR / "itsm_ticket_ops_graph.yaml"),
    ("itsm", _AGENTS_DIR / "itsm_record_safety_graph.yaml"),
    ("itsm", _AGENTS_DIR / "itsm_case_hours_graph.yaml"),
    ("onboarding", _AGENTS_DIR / "onboarding_account_setup_graph.yaml"),
    ("onboarding", _AGENTS_DIR / "onboarding_task_manager_graph.yaml"),
    ("support", _AGENTS_DIR / "support_triage_graph.yaml"),
    ("support", _AGENTS_DIR / "support_churn_graph.yaml"),
    ("support", _AGENTS_DIR / "support_specialist_routing_graph.yaml"),
    ("support", _AGENTS_DIR / "support_escalation_graph.yaml"),
    ("support", _AGENTS_DIR / "support_delegation_graph.yaml"),
    ("support", _AGENTS_DIR / "support_case_submission_graph.yaml"),
    ("integration", _AGENTS_DIR / "integration_weather_graph.yaml"),
    ("integration", _AGENTS_DIR / "integration_external_apis_graph.yaml"),
]

# All task files, paired with the domain whose registry their tools must exist
# in. Smoke + full variants for each of the five evaluated domains.
_TASK_FILES = [
    ("csm", _TASKS_DIR / "eval_csm_graph.yaml"),
    ("csm", _TASKS_DIR / "eval_csm_graph_smoke_test.yaml"),
    ("customer_service", _TASKS_DIR / "eval_customer_service_graph.yaml"),
    ("customer_service", _TASKS_DIR / "eval_customer_service_graph_smoke_test.yaml"),
    ("field_service", _TASKS_DIR / "eval_field_service_graph.yaml"),
    ("field_service", _TASKS_DIR / "eval_field_service_graph_smoke_test.yaml"),
    ("marketing", _TASKS_DIR / "eval_marketing_graph.yaml"),
    ("marketing", _TASKS_DIR / "eval_marketing_graph_smoke_test.yaml"),
    ("marketing", _TASKS_DIR / "eval_marketing_graph_hard_smoke_test.yaml"),
    ("sdr", _TASKS_DIR / "eval_sdr_graph.yaml"),
    ("sdr", _TASKS_DIR / "eval_sdr_graph_smoke_test.yaml"),
    # Ported agent-script recipes: 22 per-agent eval files + 7 per-domain smoke
    # files across the 7 new domains. Tool refs and declared domain are checked
    # against each domain's registry by the parametrized guards below.
    ("commerce", _TASKS_DIR / "eval_commerce_product_qa_graph.yaml"),
    ("commerce", _TASKS_DIR / "eval_commerce_cart_graph.yaml"),
    ("commerce", _TASKS_DIR / "eval_commerce_returns_graph.yaml"),
    ("commerce", _TASKS_DIR / "eval_commerce_graph_smoke_test.yaml"),
    ("payments", _TASKS_DIR / "eval_payments_checkout_graph.yaml"),
    ("payments", _TASKS_DIR / "eval_payments_transfer_graph.yaml"),
    ("payments", _TASKS_DIR / "eval_payments_loan_graph.yaml"),
    ("payments", _TASKS_DIR / "eval_payments_graph_smoke_test.yaml"),
    ("travel", _TASKS_DIR / "eval_travel_hotel_booking_graph.yaml"),
    ("travel", _TASKS_DIR / "eval_travel_trip_graph.yaml"),
    ("travel", _TASKS_DIR / "eval_travel_guest_schedule_graph.yaml"),
    ("travel", _TASKS_DIR / "eval_travel_graph_smoke_test.yaml"),
    ("itsm", _TASKS_DIR / "eval_itsm_ticket_ops_graph.yaml"),
    ("itsm", _TASKS_DIR / "eval_itsm_record_safety_graph.yaml"),
    ("itsm", _TASKS_DIR / "eval_itsm_case_hours_graph.yaml"),
    ("itsm", _TASKS_DIR / "eval_itsm_graph_smoke_test.yaml"),
    ("onboarding", _TASKS_DIR / "eval_onboarding_account_setup_graph.yaml"),
    ("onboarding", _TASKS_DIR / "eval_onboarding_task_manager_graph.yaml"),
    ("onboarding", _TASKS_DIR / "eval_onboarding_graph_smoke_test.yaml"),
    ("support", _TASKS_DIR / "eval_support_triage_graph.yaml"),
    ("support", _TASKS_DIR / "eval_support_churn_graph.yaml"),
    ("support", _TASKS_DIR / "eval_support_specialist_routing_graph.yaml"),
    ("support", _TASKS_DIR / "eval_support_escalation_graph.yaml"),
    ("support", _TASKS_DIR / "eval_support_delegation_graph.yaml"),
    ("support", _TASKS_DIR / "eval_support_case_submission_graph.yaml"),
    ("support", _TASKS_DIR / "eval_support_graph_smoke_test.yaml"),
    ("integration", _TASKS_DIR / "eval_integration_weather_graph.yaml"),
    ("integration", _TASKS_DIR / "eval_integration_external_apis_graph.yaml"),
    ("integration", _TASKS_DIR / "eval_integration_graph_smoke_test.yaml"),
]


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _domain_tool_names(domain_name: str) -> set[str]:
    import harness.tools.domains  # noqa: F401 — triggers @register_domain
    from harness.tools.environment import create_environment

    registry = create_environment(domain_name).create_registry()
    return set(registry.tool_names)


def test_every_domain_with_an_agent_config_is_covered():
    """Guard against a new domain agent silently escaping validation: every
    registered domain that has a *_graph.yaml must appear in _DOMAIN_CONFIGS."""
    import harness.tools.domains  # noqa: F401
    from harness.tools.environment import DOMAIN_REGISTRY

    covered = {d for d, _ in _DOMAIN_CONFIGS}
    for domain in DOMAIN_REGISTRY:
        cfg = _AGENTS_DIR / f"{domain}_graph.yaml"
        if cfg.exists():
            assert domain in covered, (
                f"Domain '{domain}' has agent config {cfg.name} but is not in "
                f"_DOMAIN_CONFIGS — add it so its YAML is validated."
            )


@pytest.mark.parametrize(
    "domain_name,config_path",
    _DOMAIN_CONFIGS,
    ids=[f"{d}:{p.stem}" for d, p in _DOMAIN_CONFIGS],
)
def test_real_yaml_tools_exist_in_domain(domain_name, config_path):
    """Every tool referenced in an agent YAML must exist in its domain registry."""
    from harness.tools.environment import create_environment

    assert config_path.exists(), f"missing agent config: {config_path}"
    env = create_environment(domain_name)
    config = _load_yaml(config_path)
    nodes = _build_nodes_from_config(config, environment=env)
    for node_name, node_cfg in config.get("nodes", {}).items():
        for tool_spec in node_cfg.get("tools", []):
            tool_name = tool_spec["name"]
            assert tool_name in [
                d.function.name for d in nodes[node_name].tools.get_definitions()
            ], (
                f"Tool '{tool_name}' in node '{node_name}' of {config_path.name} "
                f"not found in {domain_name} domain registry"
            )


@pytest.mark.parametrize(
    "domain_name,config_path",
    _DOMAIN_CONFIGS,
    ids=[f"{d}:{p.stem}" for d, p in _DOMAIN_CONFIGS],
)
def test_agent_config_entry_node_exists(domain_name, config_path):
    """If an agent config names an entry_node, it must be one of its nodes."""
    config = _load_yaml(config_path)
    entry = config.get("entry_node")
    if entry is not None:
        assert entry in config.get("nodes", {}), (
            f"{config_path.name}: entry_node '{entry}' is not among nodes "
            f"{sorted(config.get('nodes', {}))}"
        )


@pytest.mark.parametrize(
    "domain_name,task_path", _TASK_FILES, ids=[f"{d}:{p.stem}" for d, p in _TASK_FILES]
)
def test_task_tool_references_exist_in_domain(domain_name, task_path):
    """Every tool named in a task's expected_actions / expected_args /
    expected_order / forbidden_actions / forbidden_args must exist in the
    domain registry — a typo otherwise grades against a tool that never runs."""
    assert task_path.exists(), f"missing task file: {task_path}"
    known = _domain_tool_names(domain_name)
    data = _load_yaml(task_path)

    for task in data.get("tasks", []):
        tid = task.get("id", "<no-id>")
        referenced: set[str] = set()
        referenced.update(task.get("expected_actions", []) or [])
        referenced.update(task.get("expected_order", []) or [])
        referenced.update((task.get("expected_args") or {}).keys())
        referenced.update(task.get("forbidden_actions", []) or [])
        referenced.update((task.get("forbidden_args") or {}).keys())

        unknown = referenced - known
        assert not unknown, (
            f"{task_path.name} task '{tid}' references tools not in the "
            f"'{domain_name}' domain registry: {sorted(unknown)}. "
            f"Known tools: {sorted(known)}"
        )


@pytest.mark.parametrize(
    "domain_name,task_path", _TASK_FILES, ids=[f"{d}:{p.stem}" for d, p in _TASK_FILES]
)
def test_task_declared_domain_matches_file(domain_name, task_path):
    """Each task's declared `domain` (when present) must match the file's domain
    and be a registered domain."""
    import harness.tools.domains  # noqa: F401
    from harness.tools.environment import DOMAIN_REGISTRY

    data = _load_yaml(task_path)
    for task in data.get("tasks", []):
        d = task.get("domain")
        if d is not None:
            assert d in DOMAIN_REGISTRY, (
                f"{task_path.name} task '{task.get('id')}' declares unknown "
                f"domain '{d}'"
            )
            assert d == domain_name, (
                f"{task_path.name} task '{task.get('id')}' declares domain "
                f"'{d}' but the file is for '{domain_name}'"
            )
