"""Versioned benchmark manifest, lockfile, and freeze validation.

The manifest names the nine canonical domains, their policies, task catalogs,
optional reference agents, and the single ``base`` task set. Capability tags
live on individual tasks. The lockfile fingerprints the executable pieces of
that contract and snapshots canonical tool schemas plus initial environment
data.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, cast

import yaml
from yaml.resolver import BaseResolver

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "benchmark" / "benchmark.yaml"
CANONICAL_DOMAINS = frozenset(
    {
        "retail",
        "customer_support",
        "field_service",
        "itsm",
        "payments",
        "travel",
        "onboarding",
        "sales",
        "marketing",
    }
)


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(
                f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _load_yaml(path: Path) -> Any:
    return yaml.load(path.read_text(), Loader=_UniqueKeyLoader)


def _absolute(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def load_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    manifest_path = _absolute(path)
    data = _load_yaml(manifest_path)
    if not isinstance(data, dict):
        raise TypeError(f"{manifest_path}: manifest must be a mapping")
    return data


def domain_for_tasks(
    tasks_path: str | Path, manifest: dict[str, Any] | None = None
) -> tuple[str, dict[str, Any]] | None:
    """Return frozen domain metadata for a task file, or ``None`` if ad hoc."""
    manifest = manifest or load_manifest()
    target = _absolute(tasks_path)
    for domain, spec in manifest.get("domains", {}).items():
        if _absolute(spec["tasks"]) == target:
            return domain, spec
    return None


def task_metadata(
    tasks_path: str | Path, manifest: dict[str, Any] | None = None
) -> dict[str, Any]:
    manifest = manifest or load_manifest()
    domain_entry = domain_for_tasks(tasks_path, manifest)
    if domain_entry is None:
        return {}
    domain, _ = domain_entry
    return {
        "domain": domain,
        "task_set": manifest["task_set"],
        "benchmark_version": str(manifest["benchmark_version"]),
        "dataset_version": str(manifest["dataset_version"]),
    }


_EXECUTOR_SETTING_TYPES: dict[tuple[str, ...], type | tuple[type, ...]] = {
    ("max_turns",): int,
    ("trials",): int,
    ("base_seed",): int,
    ("task_timeout_seconds",): int,
    ("agent", "provider"): str,
    ("agent", "model"): str,
    ("agent", "temperature"): (int, float),
    ("agent", "max_tokens"): int,
    ("agent", "reasoning_effort"): str,
    ("agent", "step_timeout_seconds"): int,
    ("agent", "bedrock_compat"): bool,
    ("agent", "use_burr"): bool,
    ("user_simulator", "strategy"): str,
    ("user_simulator", "provider"): str,
    ("user_simulator", "model"): str,
    ("user_simulator", "temperature"): (int, float),
    ("user_simulator", "max_tokens"): int,
    ("user_simulator", "reasoning_effort"): str,
    ("user_simulator", "step_timeout_seconds"): int,
    ("user_simulator", "max_message_chars"): int,
    ("semantic_judge", "provider"): str,
    ("semantic_judge", "model"): str,
    ("semantic_judge", "temperature"): (int, float),
    ("semantic_judge", "max_tokens"): int,
    ("semantic_judge", "max_attempts"): int,
    ("semantic_judge", "reasoning_effort"): str,
    ("semantic_judge", "cache_enabled"): bool,
    ("semantic_judge", "cache_dir"): str,
    ("semantic_judge", "prompt_version"): str,
}


def _executor_config_errors(config: Any) -> list[str]:
    if not isinstance(config, dict):
        return ["executor must be a mapping"]

    errors: list[str] = []
    for path, expected_type in _EXECUTOR_SETTING_TYPES.items():
        value: Any = config
        for key in path:
            if not isinstance(value, dict) or key not in value:
                errors.append(f"executor.{'.'.join(path)} is required")
                break
            value = value[key]
        else:
            if (
                not isinstance(value, expected_type)
                or isinstance(value, bool)
                and (expected_type is not bool)
            ):
                errors.append(f"executor.{'.'.join(path)} has an invalid type")
    return errors


def executor_config(manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the shared runtime defaults used by the benchmark executor."""
    config = (manifest or load_manifest()).get("executor")
    errors = _executor_config_errors(config)
    if errors:
        raise ValueError("invalid executor configuration: " + "; ".join(errors))
    return cast(dict[str, Any], config)


def executor_environment(config: dict[str, Any] | None = None) -> dict[str, str]:
    """Serialize executor settings for use as environment-value fallbacks."""
    if config is None:
        config = executor_config()
    agent = config["agent"]
    user = config["user_simulator"]
    judge = config["semantic_judge"]

    def _serialize(value: Any) -> str:
        return str(value).lower() if isinstance(value, bool) else str(value)

    values = {
        "MAX_TURNS": config["max_turns"],
        "TASK_TIMEOUT": config["task_timeout_seconds"],
        "TRIALS": config["trials"],
        "BASE_SEED": config["base_seed"],
        "LLM_PROVIDER": agent["provider"],
        "LLM_MODEL": agent["model"],
        "TEMPERATURE": agent["temperature"],
        "LLM_MAX_TOKENS": agent["max_tokens"],
        "LLM_REASONING_EFFORT": agent["reasoning_effort"],
        "AGENT_STEP_TIMEOUT": agent["step_timeout_seconds"],
        "BEDROCK_COMPAT": agent["bedrock_compat"],
        "USE_BURR": agent["use_burr"],
        "USER_STRATEGY": user["strategy"],
        "USER_LLM_PROVIDER": user["provider"],
        "USER_LLM_MODEL": user["model"],
        "USER_TEMPERATURE": user["temperature"],
        "USER_MAX_TOKENS": user["max_tokens"],
        "USER_LLM_REASONING_EFFORT": user["reasoning_effort"],
        "USER_STEP_TIMEOUT": user["step_timeout_seconds"],
        "USER_MAX_MESSAGE_CHARS": user["max_message_chars"],
        "JUDGE_LLM_PROVIDER": judge["provider"],
        "JUDGE_LLM_MODEL": judge["model"],
        "JUDGE_TEMPERATURE": judge["temperature"],
        "JUDGE_MAX_TOKENS": judge["max_tokens"],
        "JUDGE_MAX_ATTEMPTS": judge["max_attempts"],
        "JUDGE_LLM_REASONING_EFFORT": judge["reasoning_effort"],
        "JUDGE_CACHE_ENABLED": judge["cache_enabled"],
        "JUDGE_CACHE_DIR": judge["cache_dir"],
        "JUDGE_PROMPT_VERSION": judge["prompt_version"],
    }
    return {name: _serialize(value) for name, value in values.items()}


def executor_value(name: str, defaults: dict[str, str] | None = None) -> str:
    """Resolve one executor setting with environment-over-YAML precedence."""
    values = defaults if defaults is not None else executor_environment()
    if name not in values:
        raise KeyError(f"unknown executor environment setting: {name}")
    return os.environ.get(name, values[name])


def executor_bool(name: str, defaults: dict[str, str] | None = None) -> bool:
    """Resolve and strictly parse one boolean executor setting."""
    value = executor_value(name, defaults).strip().lower()
    if value in {"1", "true", "yes"}:
        return True
    if value in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def derive_trial_seed(base_seed: int, task_id: str, trial: int) -> int:
    """Derive a stable 32-bit seed independent of Python's salted ``hash``."""
    payload = f"{base_seed}:{task_id}:{trial}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _load_raw_tasks(path: Path) -> list[dict[str, Any]]:
    data = _load_yaml(path)
    if not isinstance(data, dict) or not isinstance(data.get("tasks"), list):
        raise TypeError(f"{path}: expected a top-level tasks list")
    return data["tasks"]


def validate_manifest(
    manifest: dict[str, Any] | None = None,
) -> list[str]:
    """Return all freeze-contract violations without stopping at the first one."""
    manifest = manifest or load_manifest()
    errors: list[str] = []

    if manifest.get("schema_version") != 3:
        errors.append("schema_version must be 3")
    for key in ("benchmark_version", "dataset_version"):
        if not manifest.get(key):
            errors.append(f"{key} is required")
    if manifest.get("status") != "frozen":
        errors.append("manifest status must be 'frozen'")
    if manifest.get("task_set") != "base":
        errors.append("task_set must be 'base'")
    if manifest.get("task_count") != 555:
        errors.append("task_count must be 555")
    if "paper_run" in manifest:
        errors.append(
            "paper_run is no longer supported; configure runs through the executor"
        )
    errors.extend(_executor_config_errors(manifest.get("executor")))
    if "split_summary" in manifest:
        errors.append("legacy split_summary is not allowed; all tasks belong to base")
    if "suites" in manifest:
        errors.append("legacy suites are not allowed; tasks are organized by domain")

    domains = set(manifest.get("domains", {}))
    if domains != CANONICAL_DOMAINS:
        errors.append(
            f"domains must be exactly {sorted(CANONICAL_DOMAINS)}, got {sorted(domains)}"
        )

    expected_task_files = {f"{domain}.yaml" for domain in CANONICAL_DOMAINS}
    actual_task_files = {
        path.name for path in (REPO_ROOT / "src" / "tasks").glob("*.yaml")
    }
    if actual_task_files != expected_task_files:
        errors.append(
            "src/tasks must contain exactly one YAML per domain: "
            f"expected {sorted(expected_task_files)}, got {sorted(actual_task_files)}"
        )
    expected_agent_files = {f"{domain}.yaml" for domain in CANONICAL_DOMAINS}
    actual_agent_files = {
        path.name for path in (REPO_ROOT / "src" / "agents").glob("*.yaml")
    }
    if actual_agent_files != expected_agent_files:
        errors.append(
            "src/agents must contain exactly one YAML per domain: "
            f"expected {sorted(expected_agent_files)}, got {sorted(actual_agent_files)}"
        )

    import harness.tools.domains  # noqa: F401
    from harness.graph_cli import _build_nodes_from_config
    from harness.tools.environment import DOMAIN_REGISTRY, create_environment

    if set(DOMAIN_REGISTRY) != CANONICAL_DOMAINS:
        errors.append(
            f"registered domains do not match manifest: {sorted(DOMAIN_REGISTRY)}"
        )

    task_ids: dict[str, str] = {}
    total_task_count = 0
    semantic_task_count = 0
    semantic_criterion_count = 0
    for domain, spec in manifest.get("domains", {}).items():
        expected_tasks = f"src/tasks/{domain}.yaml"
        expected_agent = f"src/agents/{domain}.yaml"
        if spec.get("tasks") != expected_tasks:
            errors.append(
                f"domain {domain}: tasks must be the canonical file {expected_tasks}"
            )
        if spec.get("reference_agent") != expected_agent:
            errors.append(f"domain {domain}: reference_agent must be {expected_agent}")

        policy = _absolute(spec.get("policy", ""))
        tasks_path = _absolute(spec.get("tasks", ""))
        config_path = _absolute(spec.get("reference_agent", ""))
        if not policy.is_file():
            errors.append(f"domain {domain}: missing policy {policy}")
        if not tasks_path.is_file():
            errors.append(f"domain {domain}: missing tasks file {tasks_path}")
            continue
        if not config_path.is_file():
            errors.append(f"domain {domain}: missing reference agent {config_path}")
            continue

        try:
            raw_tasks = _load_raw_tasks(tasks_path)
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            errors.append(f"domain {domain}: cannot load tasks: {exc}")
            continue
        total_task_count += len(raw_tasks)

        environment = create_environment(domain)
        tool_names = set(environment.create_registry().tool_names)
        try:
            config = _load_yaml(config_path)
            _build_nodes_from_config(config, environment=environment)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            errors.append(f"domain {domain}: invalid reference agent: {exc}")

        state_roots = set(environment.default_state())
        for task in raw_tasks:
            tid = task.get("id")
            if not tid:
                errors.append(f"domain {domain}: task missing id")
                continue
            if tid in task_ids:
                errors.append(
                    f"task id {tid!r} is duplicated in {task_ids[tid]} and {domain}"
                )
            task_ids[tid] = domain
            if task.get("domain") != domain:
                errors.append(
                    f"domain {domain} task {tid}: declares {task.get('domain')!r}"
                )
            capabilities = task.get("capabilities")
            if (
                not isinstance(capabilities, list)
                or not capabilities
                or not all(isinstance(value, str) and value for value in capabilities)
            ):
                errors.append(
                    f"domain {domain} task {tid}: capabilities must be non-empty strings"
                )
            if task.get("difficulty") not in {"easy", "medium", "hard"}:
                errors.append(f"domain {domain} task {tid}: difficulty is not frozen")
            semantic_criteria = task.get("expected_response_semantic") or []
            if "expected_response_contains" in task:
                errors.append(
                    f"domain {domain} task {tid}: expected_response_contains is "
                    "not allowed; all response assertions must use "
                    "expected_response_semantic"
                )
            if semantic_criteria:
                semantic_task_count += 1
                semantic_criterion_count += len(semantic_criteria)
                if not all(
                    isinstance(value, str) and value.strip()
                    for value in semantic_criteria
                ):
                    errors.append(
                        f"domain {domain} task {tid}: semantic response criteria "
                        "must be non-empty strings"
                    )
                if any(
                    any(token in value for token in ("|", "\\", ".*", "^"))
                    for value in semantic_criteria
                    if isinstance(value, str)
                ):
                    errors.append(
                        f"domain {domain} task {tid}: semantic response criteria "
                        "must be plain-language requirements without regex syntax"
                    )
                if any(
                    not value.startswith("The assistant must ")
                    or not value.endswith(".")
                    for value in semantic_criteria
                    if isinstance(value, str)
                ):
                    errors.append(
                        f"domain {domain} task {tid}: frozen semantic response "
                        "criteria must be declarative requirement sentences"
                    )
            expected_actions = set(task.get("expected_actions") or [])
            expected_order = set(task.get("expected_order") or [])
            expected_args_tools = set((task.get("expected_args") or {}).keys())
            forbidden_actions = set(task.get("forbidden_actions") or [])
            contradictory_actions = (
                expected_actions | expected_order | expected_args_tools
            ) & forbidden_actions
            if contradictory_actions:
                errors.append(
                    f"domain {domain} task {tid}: requires and forbids the same "
                    f"actions {sorted(contradictory_actions)}"
                )
            declared_expected_tools = expected_actions | expected_order
            orphaned_args = expected_args_tools - declared_expected_tools
            if orphaned_args:
                errors.append(
                    f"domain {domain} task {tid}: expected_args reference "
                    f"undeclared expected actions {sorted(orphaned_args)}"
                )
            asserted_tools = set(task.get("expected_actions", [])) | set(
                task.get("forbidden_actions", [])
            )
            asserted_tools |= set(task.get("expected_order", []))
            asserted_tools |= set((task.get("expected_args") or {}).keys())
            asserted_tools |= set((task.get("forbidden_args") or {}).keys())
            unknown = asserted_tools - tool_names
            if unknown:
                errors.append(
                    f"domain {domain} task {tid}: unknown canonical tools {sorted(unknown)}"
                )
            for field in ("initial_state", "expected_state", "forbidden_state"):
                unknown_roots = set(task.get(field) or {}) - state_roots
                if unknown_roots:
                    errors.append(
                        f"domain {domain} task {tid}: {field} has unknown state roots "
                        f"{sorted(unknown_roots)}"
                    )

    expected_count = manifest.get("task_count")
    if total_task_count != expected_count:
        errors.append(
            f"domain task count {total_task_count} != task_count {expected_count}"
        )
    if total_task_count != 555:
        errors.append(f"paper dataset must contain 555 tasks, got {total_task_count}")
    if semantic_task_count != 402 or semantic_criterion_count != 465:
        errors.append(
            "semantic response migration must cover 402 tasks / 465 criteria, got "
            f"{semantic_task_count} tasks / {semantic_criterion_count} criteria"
        )
    return errors


def build_lock(manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a deterministic snapshot of the frozen executable contract."""
    manifest = manifest or load_manifest()
    import harness.tools.domains  # noqa: F401
    from harness.tools.environment import create_environment

    artifact_paths: set[str] = {
        "benchmark/benchmark.yaml",
        "pyproject.toml",
        "uv.lock",
        "src/analysis/bench_report.py",
    }
    for spec in manifest.get("domains", {}).values():
        artifact_paths.add(spec["policy"])
        artifact_paths.add(spec["tasks"])
        artifact_paths.add(spec["reference_agent"])
    artifact_paths.update(
        str(path.relative_to(REPO_ROOT))
        for pattern in ("*.py", "*.yaml")
        for path in (REPO_ROOT / "src" / "harness").rglob(pattern)
    )

    artifacts = {path: _sha256_file(_absolute(path)) for path in sorted(artifact_paths)}
    domain_contracts: dict[str, Any] = {}
    for domain in manifest["domains"]:
        environment = create_environment(domain)
        definitions = [
            definition.model_dump(mode="json")
            for definition in environment.create_registry().get_definitions()
        ]
        definitions.sort(key=lambda item: item["function"]["name"])
        state = environment.default_state()
        domain_contracts[domain] = {
            "initial_state_sha256": _sha256_bytes(_canonical_json(state)),
            "tool_contract_sha256": _sha256_bytes(_canonical_json(definitions)),
            "tools": definitions,
        }

    return {
        "lock_schema_version": 1,
        "benchmark_version": str(manifest["benchmark_version"]),
        "dataset_version": str(manifest["dataset_version"]),
        "artifacts": artifacts,
        "domains": domain_contracts,
    }


def write_lock(manifest: dict[str, Any] | None = None) -> Path:
    manifest = manifest or load_manifest()
    lock_path = _absolute(manifest["lock_file"])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(build_lock(manifest), indent=2) + "\n")
    return lock_path


def validate_lock(manifest: dict[str, Any] | None = None) -> list[str]:
    manifest = manifest or load_manifest()
    lock_path = _absolute(manifest["lock_file"])
    if not lock_path.is_file():
        return [f"missing lockfile {lock_path}"]
    expected = build_lock(manifest)
    try:
        actual = json.loads(lock_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read lockfile {lock_path}: {exc}"]
    return [] if actual == expected else ["benchmark lockfile is stale; regenerate it"]


def _plain_assertion_value(value: Any) -> Any | None:
    if not isinstance(value, dict):
        return value
    if "_contains" in value:
        return value["_contains"]
    return None


def _walk_key_values(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(child, (dict, list)):
                yield str(key), child
            yield from _walk_key_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_key_values(child)


def _normal_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


_PARAM_ALIASES = {
    "emailid": "email",
    "useremail": "email",
    "recipientemail": "email",
    "customerquery": "query",
    "productname": "name",
    "inputcampaignname": "name",
    "campaignid": "id",
    "appointmentnumber": "id",
    "bookingreference": "bookingid",
    "issuedescription": "description",
}

_PARAM_DEFAULTS: dict[str, Any] = {
    "amount": 50.0,
    "quantity": 1,
    "nights": 1,
    "days": 3,
    "pin": "1234",
    "reason": "requested by customer",
    "method": "credit_card",
    "priority": "medium",
    "plan": "standard",
    "team": "technical",
    "type": "Most",
    "relationshiptype": "Compatible",
    "query": "account setup",
    "question": "Please investigate the customer issue",
    "topic": "technical support",
    "description": "Customer needs assistance",
    "subject": "Customer support request",
    "casedata": "Customer support request",
    "inputdescription": "Frozen benchmark gold campaign",
    "inputstartdate": "2026-10-01",
    "inputenddate": "2026-10-31",
    "inputcampaignname": "Gold Validation Campaign",
    "emailresponse": "Interested in learning more",
    "action": "delete",
    "status": "in_progress",
    "comment": "Gold-path validation update",
    "slot": "2026-09-21 10:00",
    "starttime": "2026-09-21 10:00",
    "endtime": "2026-09-21 12:00",
    "citationmode": "brief",
    "location": "San Francisco",
    "city": "San Francisco",
    "cartype": "economy",
    "guestname": "Alex Chen",
    "passengername": "Alex Chen",
    "name": "Alex Chen",
    "firstname": "Alex",
    "lastname": "Chen",
}


def _instruction_candidate(param: str, instruction: str) -> Any | None:
    normal = _normal_name(param)
    account_ids = re.findall(r"\bACC-[A-Za-z0-9-]+\b", instruction)
    if normal == "fromaccount" and account_ids:
        return account_ids[0]
    if normal == "toaccount" and len(account_ids) > 1:
        return account_ids[1]
    patterns: list[tuple[set[str], str, Any]] = [
        (
            {"email", "emailid", "useremail", "recipientemail"},
            r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
            str,
        ),
        ({"trackingnumber"}, r"\b1Z[A-Z0-9]{15,}\b", str),
        ({"cardtoken"}, r"\btok_[A-Za-z0-9_]+\b", str),
        ({"customerid"}, r"\bCUST-[A-Za-z0-9-]+\b", str),
        ({"accountid"}, r"\bACC-[A-Za-z0-9-]+\b", str),
        ({"applicantid"}, r"\bAPP-[A-Za-z0-9-]+\b", str),
        ({"invoiceid"}, r"\bINV-[A-Za-z0-9-]+\b", str),
        ({"orderid"}, r"\bORD-[A-Za-z0-9-]+\b", str),
        ({"productid"}, r"\bPROD-[A-Za-z0-9-]+\b", str),
        ({"hotelid"}, r"\bHTL-[A-Za-z0-9-]+\b", str),
        ({"flightid"}, r"\bFL-[A-Za-z0-9-]+\b", str),
        ({"guestid"}, r"\bGST-[A-Za-z0-9-]+\b", str),
        ({"ticketid"}, r"\b(?:INC|TKT|TICKET)-[A-Za-z0-9-]+\b", str),
        ({"campaignid"}, r"\b(?:MA|CMP|CAM)-[A-Za-z0-9-]+\b", str),
        ({"listid"}, r"\bLIST-[A-Za-z0-9-]+\b", str),
        ({"taskid"}, r"\bTASK-[A-Za-z0-9-]+\b", str),
        ({"appointmentnumber"}, r"\bAPPT-[A-Za-z0-9-]+\b", str),
        ({"leadid", "targetid"}, r"\bLEAD-[A-Za-z0-9-]+\b", str),
        ({"amount"}, r"\$\s*([0-9]+(?:\.[0-9]+)?)", float),
        ({"days", "nights", "quantity"}, r"\b([1-9][0-9]*)\b", int),
    ]
    for names, pattern, convert in patterns:
        if normal not in names:
            continue
        match = re.search(pattern, instruction, flags=re.IGNORECASE)
        if match:
            raw = match.group(1) if match.lastindex else match.group(0)
            return convert(raw)
    return None


def _gold_arguments(
    task: dict[str, Any],
    tool: Any,
    state: dict[str, Any],
    context: dict[str, Any],
    occurrence: int,
) -> dict[str, Any]:
    expected_spec = (task.get("expected_args") or {}).get(tool.name, {})
    if isinstance(expected_spec, list):
        expected = expected_spec[min(occurrence, len(expected_spec) - 1)]
    else:
        expected = expected_spec
    args: dict[str, Any] = {}
    instruction = task["instruction"]
    state_values = list(_walk_key_values(state))
    # Prefer identifiers returned by the most recent call (e.g. the booking or
    # transaction just created) when a later call consumes generated output.
    context_values = [
        item
        for payload in reversed(list(context.values()))
        for item in _walk_key_values(payload)
    ]

    for param in tool.parameters.get("properties", {}):
        if param in expected:
            value = _plain_assertion_value(expected[param])
            if value is not None:
                schema_type = tool.parameters["properties"][param].get("type")
                if schema_type == "number":
                    value = float(value)
                elif schema_type == "integer":
                    value = int(value)
                args[param] = value
                continue
        normal = _normal_name(param)
        alias = _PARAM_ALIASES.get(normal, normal)

        candidate = _instruction_candidate(param, instruction)
        if candidate is not None:
            args[param] = candidate
            continue

        keyed = [
            value
            for key, value in context_values + state_values
            if _PARAM_ALIASES.get(_normal_name(key), _normal_name(key)) == alias
        ]
        mentioned = [
            value
            for value in keyed
            if isinstance(value, (str, int, float))
            and str(value).lower() in instruction.lower()
        ]
        if mentioned:
            args[param] = mentioned[0]
            continue
        if keyed:
            args[param] = keyed[0]
            continue

        if normal in _PARAM_DEFAULTS or alias in _PARAM_DEFAULTS:
            args[param] = _PARAM_DEFAULTS.get(normal, _PARAM_DEFAULTS.get(alias))

    return args


def iter_frozen_tasks(manifest: dict[str, Any] | None = None):
    """Yield every ``(domain, task)`` pair in the canonical base task set."""
    manifest = manifest or load_manifest()
    for domain, spec in manifest["domains"].items():
        for task in _load_raw_tasks(_absolute(spec["tasks"])):
            yield domain, task


async def validate_gold_task(domain: str, task: dict[str, Any]) -> list[str]:
    """Execute and validate one architecture-independent canonical gold path."""
    import harness.tools.domains  # noqa: F401
    from harness.tools.environment import create_environment
    from harness.tools.state_validation import validate_state

    errors: list[str] = []
    environment = create_environment(domain, initial_state=task.get("initial_state"))
    registry = environment.create_registry()
    sequence = task.get("expected_order") or task.get("expected_actions", [])
    context: dict[str, Any] = {}
    occurrences: dict[str, int] = {}
    for index, tool_name in enumerate(sequence):
        registered = registry.get_registered_tool(tool_name)
        if registered is None:
            errors.append(
                f"{domain} task {task['id']}: gold tool {tool_name!r} missing"
            )
            break
        occurrence = occurrences.get(tool_name, 0)
        args = _gold_arguments(
            task,
            registered,
            environment.get_state(),
            context,
            occurrence,
        )
        occurrences[tool_name] = occurrence + 1
        required = set(registered.parameters.get("required", []))
        missing = required - set(args)
        if missing:
            errors.append(
                f"{domain} task {task['id']}: gold call {index + 1} "
                f"{tool_name} missing args {sorted(missing)}"
            )
            break
        result = await registry.execute_with_status(tool_name, args)
        if not result.ok:
            errors.append(
                f"{domain} task {task['id']}: gold call {index + 1} "
                f"{tool_name} failed to execute"
            )
            break
        try:
            payload = json.loads(result.result)
        except (json.JSONDecodeError, TypeError):
            payload = {"result": result.result}
        context[f"call_{index}"] = payload
    else:
        if task.get("expected_state"):
            passed, failures = validate_state(
                environment.get_state(), task["expected_state"]
            )
            if not passed:
                errors.append(
                    f"{domain} task {task['id']}: gold final state failed: "
                    + "; ".join(failures)
                )
    return errors


async def validate_gold_paths(
    manifest: dict[str, Any] | None = None,
) -> list[str]:
    """Execute every base task's declared successful tool path in fresh state.

    The validator is intentionally architecture-independent: it executes only
    canonical tools, then checks asserted final state. Generated identifiers
    are carried from one tool result into later calls. Refusal tasks have an
    empty gold path and are validated structurally by :func:`validate_manifest`.
    """
    errors: list[str] = []
    for domain, task in iter_frozen_tasks(manifest):
        errors.extend(await validate_gold_task(domain, task))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen benchmark")
    parser.add_argument(
        "--write-lock", action="store_true", help="Regenerate benchmark.lock.json"
    )
    parser.add_argument(
        "--gold", action="store_true", help="Execute all canonical gold paths"
    )
    args = parser.parse_args()
    manifest = load_manifest()
    errors = validate_manifest(manifest)
    if errors:
        raise SystemExit("Freeze validation failed:\n- " + "\n- ".join(errors))
    if args.gold:
        errors = asyncio.run(validate_gold_paths(manifest))
        if errors:
            raise SystemExit("Gold validation failed:\n- " + "\n- ".join(errors))
        print(
            f"Gold validation: {manifest['task_count']}/{manifest['task_count']} "
            "benchmark tasks passed."
        )
    if args.write_lock:
        path = write_lock(manifest)
        print(f"Wrote {path.relative_to(REPO_ROOT)}")
    else:
        errors = validate_lock(manifest)
        if errors:
            raise SystemExit("Freeze validation failed:\n- " + "\n- ".join(errors))
        print(
            f"Benchmark {manifest['benchmark_version']} / dataset "
            f"{manifest['dataset_version']} is frozen and valid."
        )


if __name__ == "__main__":
    main()
