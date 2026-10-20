"""Tests for config validation — tool name checking at graph startup."""

import json
from pathlib import Path
from typing import cast

import pytest
import yaml

from harness.bench_runner import load_tasks
from harness.grading import compute_action_pass, match_expected_args
from harness.graph_cli import _build_nodes_from_config
from harness.tools.environment import ToolEnvironment
from harness.tools.registry import ToolRegistry, is_error_result


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

    def test_semantic_response_criteria_load(self, tmp_path):
        task = {
            **_MINIMAL_TASK,
            "expected_response_semantic": [
                "The assistant must state that the record does not exist."
            ],
        }
        loaded = load_tasks(_write_tasks(tmp_path, task))[0]
        assert loaded.expected_response_semantic == [
            "The assistant must state that the record does not exist."
        ]

    def test_semantic_response_criteria_reject_empty_list(self, tmp_path):
        task = {**_MINIMAL_TASK, "expected_response_semantic": []}
        with pytest.raises(ValueError, match="expected_response_semantic"):
            load_tasks(_write_tasks(tmp_path, task))

    def test_semantic_response_criteria_reject_regex_syntax(self, tmp_path):
        task = {
            **_MINIMAL_TASK,
            "expected_response_semantic": ["not found|does not exist"],
        }
        with pytest.raises(ValueError, match="plain-language requirements"):
            load_tasks(_write_tasks(tmp_path, task))

    def test_legacy_response_contains_is_rejected(self, tmp_path):
        task = {**_MINIMAL_TASK, "expected_response_contains": ["PROD-1"]}
        with pytest.raises(ValueError, match="unknown key"):
            load_tasks(_write_tasks(tmp_path, task))

    def test_duplicate_yaml_key_is_rejected(self, tmp_path):
        path = tmp_path / "tasks.yaml"
        path.write_text(
            "tasks:\n"
            "- id: t1\n"
            "  instruction: do the thing\n"
            "  forbidden_state: {session: {can_delete: true}}\n"
            "  forbidden_state: {tickets: {TCK-1001: _not_null}}\n"
        )

        with pytest.raises(ValueError, match="duplicate YAML key 'forbidden_state'"):
            load_tasks(str(path))

    def test_expected_action_cannot_also_be_forbidden(self, tmp_path):
        task = {
            **_MINIMAL_TASK,
            "expected_actions": ["delete_ticket"],
            "forbidden_actions": ["delete_ticket"],
        }
        with pytest.raises(ValueError, match="requires and forbids"):
            load_tasks(_write_tasks(tmp_path, task))

    def test_ordered_action_cannot_also_be_forbidden(self, tmp_path):
        task = {
            **_MINIMAL_TASK,
            "expected_order": ["delete_ticket"],
            "forbidden_actions": ["delete_ticket"],
        }
        with pytest.raises(ValueError, match="requires and forbids"):
            load_tasks(_write_tasks(tmp_path, task))

    def test_expected_args_action_cannot_also_be_forbidden(self, tmp_path):
        task = {
            **_MINIMAL_TASK,
            "expected_args": {"delete_ticket": {"ticket_id": "TCK-1001"}},
            "forbidden_actions": ["delete_ticket"],
        }
        with pytest.raises(ValueError, match="requires and forbids"):
            load_tasks(_write_tasks(tmp_path, task))

    def test_expected_args_require_a_declared_expected_action(self, tmp_path):
        task = {
            **_MINIMAL_TASK,
            "expected_args": {"delete_ticket": {"ticket_id": "TCK-1001"}},
        }
        with pytest.raises(ValueError, match="undeclared expected action"):
            load_tasks(_write_tasks(tmp_path, task))

    def test_expected_order_declares_action_for_expected_args(self, tmp_path):
        task = {
            **_MINIMAL_TASK,
            "expected_order": ["delete_ticket"],
            "expected_args": {"delete_ticket": {"ticket_id": "TCK-1001"}},
        }
        loaded = load_tasks(_write_tasks(tmp_path, task))[0]
        assert loaded.expected_order == ["delete_ticket"]


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
        nodes = _build_nodes_from_config(config, environment=cast(ToolEnvironment, env))
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
            _build_nodes_from_config(config, environment=cast(ToolEnvironment, env))

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
            _build_nodes_from_config(config, environment=cast(ToolEnvironment, env))

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
_FIXTURE_TASKS_DIR = Path(__file__).resolve().parent / "fixtures" / "tasks"
_CANONICAL_DOMAINS = (
    "retail",
    "customer_support",
    "field_service",
    "itsm",
    "payments",
    "travel",
    "onboarding",
    "sales",
    "marketing",
)

_DOMAIN_CONFIGS = [
    (domain, _AGENTS_DIR / f"{domain}.yaml") for domain in _CANONICAL_DOMAINS
]

_TASK_FILES = [(domain, _TASKS_DIR / f"{domain}.yaml") for domain in _CANONICAL_DOMAINS]
for fixture_path in sorted(_FIXTURE_TASKS_DIR.glob("*.yaml")):
    fixture = yaml.safe_load(fixture_path.read_text())
    domains = {task.get("domain") for task in fixture.get("tasks", [])}
    assert len(domains) == 1, f"{fixture_path}: smoke fixture mixes domains"
    _TASK_FILES.append((domains.pop(), fixture_path))


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _domain_tool_names(domain_name: str) -> set[str]:
    import harness.tools.domains  # noqa: F401 — triggers @register_domain
    from harness.tools.environment import create_environment

    registry = create_environment(domain_name).create_registry()
    return set(registry.tool_names)


def test_every_domain_with_an_agent_config_is_covered():
    """Exactly one canonical agent and task YAML must exist per domain."""
    import harness.tools.domains  # noqa: F401
    from harness.tools.environment import DOMAIN_REGISTRY

    assert set(DOMAIN_REGISTRY) == set(_CANONICAL_DOMAINS)
    assert {path.name for path in _AGENTS_DIR.glob("*.yaml")} == {
        f"{domain}.yaml" for domain in _CANONICAL_DOMAINS
    }
    assert {path.name for path in _TASKS_DIR.glob("*.yaml")} == {
        f"{domain}.yaml" for domain in _CANONICAL_DOMAINS
    }


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


def test_itsm_successful_deletions_assert_target_is_absent():
    """A successful delete must prove the record disappeared, not just call a tool."""
    tasks = _load_yaml(_TASKS_DIR / "itsm.yaml")["tasks"]

    for task in tasks:
        if "delete_ticket" not in (task.get("expected_actions") or []):
            continue
        target = task["expected_args"]["delete_ticket"]["ticket_id"]
        forbidden_tickets = (task.get("forbidden_state") or {}).get("tickets", {})
        assert forbidden_tickets.get(target) == "_not_null", (
            f"{task['id']}: successful deletion of {target} must assert that "
            "the target ticket is absent from final state"
        )


# Gold cases whose intended result is a successful lookup that observes no
# matching business record. These need an execution audit, not merely the
# static tool-name validation above: an ``{"error": ...}`` payload used to be
# filtered out before expected_actions and expected_args were graded.
_NOT_FOUND_GOLD_CASES = [
    (
        "retail",
        _TASKS_DIR / "retail.yaml",
        "integration_ext_07",
        "track_shipment",
    ),
    (
        "retail",
        _TASKS_DIR / "retail.yaml",
        "commerce_qa_unknown_product_id_10",
        "get_product",
    ),
    (
        "payments",
        _TASKS_DIR / "payments.yaml",
        "payments_loan_008",
        "get_credit_score",
    ),
    (
        "itsm",
        _TASKS_DIR / "itsm.yaml",
        "itsm_safety_006_lookup_nonexistent_ticket",
        "get_ticket",
    ),
    (
        "travel",
        _TASKS_DIR / "travel.yaml",
        "travel_hotel_016_invalid_hotel_id_lookup",
        "get_hotel",
    ),
    (
        "travel",
        _TASKS_DIR / "travel.yaml",
        "integration_weather_unknown_city_current_01",
        "get_current_weather",
    ),
    (
        "travel",
        _TASKS_DIR / "travel.yaml",
        "integration_weather_unknown_city_forecast_01",
        "get_weather_forecast",
    ),
    (
        "travel",
        _TASKS_DIR / "travel.yaml",
        "travel_guest_unknown_guest_lookup_01",
        "get_guest_reservation",
    ),
    (
        "travel",
        _TASKS_DIR / "travel.yaml",
        "travel_guest_unknown_guest_schedule_01",
        "generate_schedule",
    ),
    (
        "itsm",
        _TASKS_DIR / "itsm.yaml",
        "itsm_ops_014_lookup_nonexistent_ticket",
        "get_ticket",
    ),
    (
        "itsm",
        _TASKS_DIR / "itsm.yaml",
        "itsm_hours_016_nonexistent_ticket",
        "get_ticket",
    ),
    (
        "travel",
        _TASKS_DIR / "travel.yaml",
        "travel_hotel_019_confirm_nonexistent_booking_hallucination_guard",
        "confirm_booking",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "domain_name,task_path,task_id,tool_name",
    _NOT_FOUND_GOLD_CASES,
    ids=[case[2] for case in _NOT_FOUND_GOLD_CASES],
)
async def test_not_found_gold_case_is_executable_and_gradeable(
    domain_name, task_path, task_id, tool_name
):
    """Execute each not-found gold call and prove all structural checks pass."""
    import harness.tools.domains  # noqa: F401
    from harness.tools.environment import create_environment

    task = next(
        item for item in _load_yaml(task_path).get("tasks", []) if item["id"] == task_id
    )
    arguments = task["expected_args"][tool_name]
    registry = create_environment(
        domain_name, initial_state=task.get("initial_state")
    ).create_registry()

    execution = await registry.execute_with_status(tool_name, arguments)
    payload = json.loads(execution.result)
    call_detail = [
        {
            "name": tool_name,
            "arguments": arguments,
            "result": execution.result,
            "ok": execution.ok,
        }
    ]
    args_passed, args_failures = match_expected_args(
        call_detail, task["expected_args"], mode=task.get("expected_args_mode", "any")
    )

    assert execution.ok is True
    assert payload["found"] is False
    assert "error" not in payload
    assert is_error_result(execution.result) is False
    assert compute_action_pass(task["expected_actions"], [tool_name]) is True
    assert args_passed is True, args_failures


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
