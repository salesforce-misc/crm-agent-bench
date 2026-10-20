"""Guards for the canonical benchmark v2 freeze contract."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from harness.bench_runner import (
    _user_reasoning_effort,
    load_tasks,
    main,
)
from harness.benchmark_spec import (
    CANONICAL_DOMAINS,
    derive_trial_seed,
    executor_bool,
    executor_config,
    executor_environment,
    executor_value,
    load_manifest,
    validate_lock,
    validate_manifest,
)


def test_frozen_manifest_and_lock_are_current():
    manifest = load_manifest()
    assert validate_manifest(manifest) == []
    assert validate_lock(manifest) == []
    assert set(manifest["domains"]) == CANONICAL_DOMAINS
    assert manifest["schema_version"] == 3
    assert "suites" not in manifest
    assert manifest["task_set"] == "base"
    assert manifest["task_count"] == 555
    assert "paper_run" not in manifest
    executor = executor_config(manifest)
    assert executor["agent"]["provider"] == "litellm"
    assert executor["user_simulator"]["provider"] == "litellm"
    assert executor["agent"]["reasoning_effort"] == "low"
    assert executor["task_timeout_seconds"] == 1200
    assert executor["agent"]["step_timeout_seconds"] == 240
    assert executor["user_simulator"]["reasoning_effort"] == "low"
    assert executor["user_simulator"]["max_tokens"] == 1086
    assert executor["user_simulator"]["step_timeout_seconds"] == 90
    assert executor["user_simulator"]["max_message_chars"] == 4000
    assert executor["semantic_judge"] == {
        "provider": "litellm",
        "model": "gpt-5-mini",
        "temperature": 0.0,
        "max_tokens": 16384,
        "max_attempts": 3,
        "reasoning_effort": "low",
        "cache_enabled": True,
        "cache_dir": "bench_output/judge_cache",
        "prompt_version": "semantic-communication-v6",
    }
    assert {Path(spec["tasks"]).name for spec in manifest["domains"].values()} == {
        f"{domain}.yaml" for domain in CANONICAL_DOMAINS
    }
    assert {
        Path(spec["reference_agent"]).name for spec in manifest["domains"].values()
    } == {f"{domain}.yaml" for domain in CANONICAL_DOMAINS}


def test_frozen_metadata_is_attached_to_loaded_tasks():
    tasks = load_tasks("src/tasks/retail.yaml")
    assert tasks
    assert {task.domain for task in tasks} == {"retail"}
    assert {task.benchmark_version for task in tasks} == {"2.6.0"}
    assert {task.dataset_version for task in tasks} == {"1.5.0"}
    assert {capability for task in tasks for capability in task.capabilities} == {
        "cart",
        "external_api",
        "inventory",
        "product_qa",
        "promotions",
        "returns",
        "shipment",
    }
    assert {task.task_set for task in tasks} == {"base"}


def test_manifest_rejects_removed_paper_run():
    manifest = load_manifest()
    manifest["paper_run"] = {}

    assert validate_manifest(manifest) == [
        "paper_run is no longer supported; configure runs through the executor"
    ]


def test_executor_settings_apply_unless_an_environment_override_exists(monkeypatch):
    for name in (
        "TASK_TIMEOUT",
        "MAX_TURNS",
        "TEMPERATURE",
        "LLM_REASONING_EFFORT",
        "USER_LLM_MODEL",
        "USER_LLM_REASONING_EFFORT",
        "JUDGE_TEMPERATURE",
        "JUDGE_LLM_REASONING_EFFORT",
    ):
        monkeypatch.delenv(name, raising=False)

    defaults = executor_environment(executor_config())

    assert executor_value("TASK_TIMEOUT", defaults) == "1200"
    assert executor_value("MAX_TURNS", defaults) == "30"
    assert executor_value("LLM_PROVIDER", defaults) == "litellm"
    assert executor_value("USER_LLM_PROVIDER", defaults) == "litellm"
    assert executor_value("JUDGE_LLM_PROVIDER", defaults) == "litellm"
    assert executor_value("TEMPERATURE", defaults) == "0.0"
    assert executor_value("USER_LLM_MODEL", defaults) == "gpt-5-mini"
    assert executor_value("USER_TEMPERATURE", defaults) == "0.0"
    assert executor_value("LLM_REASONING_EFFORT", defaults) == "low"
    assert executor_value("USER_LLM_REASONING_EFFORT", defaults) == "low"
    assert executor_value("JUDGE_TEMPERATURE", defaults) == "0.0"
    assert executor_value("JUDGE_LLM_REASONING_EFFORT", defaults) == "low"
    assert executor_bool("USE_BURR", defaults) is True
    assert executor_bool("BEDROCK_COMPAT", defaults) is False

    monkeypatch.setenv("JUDGE_TEMPERATURE", "0.7")
    assert executor_value("JUDGE_TEMPERATURE", defaults) == "0.7"
    monkeypatch.setenv("USE_BURR", "0")
    assert executor_bool("USE_BURR", defaults) is False
    monkeypatch.setenv("MAX_TURNS", "12")
    assert executor_value("MAX_TURNS", defaults) == "12"


def test_agent_reasoning_override_does_not_change_user_reasoning(monkeypatch):
    defaults = executor_environment(executor_config())
    monkeypatch.delenv("USER_LLM_REASONING_EFFORT", raising=False)
    monkeypatch.setenv("LLM_REASONING_EFFORT", "high")

    assert executor_value("LLM_REASONING_EFFORT", defaults) == "high"
    assert _user_reasoning_effort(defaults) == "low"

    monkeypatch.setenv("USER_LLM_REASONING_EFFORT", "disable")
    assert _user_reasoning_effort(defaults) == "disable"


def test_executor_bool_rejects_ambiguous_values(monkeypatch):
    defaults = executor_environment(executor_config())
    monkeypatch.setenv("USE_BURR", "sometimes")

    with pytest.raises(ValueError, match="USE_BURR must be a boolean"):
        executor_bool("USE_BURR", defaults)


def test_runner_rejects_removed_paper_run_flag(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_runner",
            "--tasks",
            "src/tasks/retail.yaml",
            "--paper-run",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    assert "unrecognized arguments: --paper-run" in capsys.readouterr().err


def test_every_frozen_task_belongs_to_base():
    manifest = load_manifest()
    tasks = [
        task
        for spec in manifest["domains"].values()
        for task in load_tasks(spec["tasks"])
    ]
    assert len(tasks) == 555
    assert len({task.id for task in tasks}) == 555
    assert {task.task_set for task in tasks} == {"base"}

    semantic_tasks = [task for task in tasks if task.expected_response_semantic]
    assert len(semantic_tasks) == 402
    assert sum(len(task.expected_response_semantic or []) for task in tasks) == 465
    assert all(
        criterion.startswith("The assistant must ")
        and criterion.endswith(".")
        and not any(token in criterion for token in ("|", "\\", ".*", "^"))
        for task in tasks
        for criterion in (task.expected_response_semantic or [])
    )
    assert not any(
        "expected_response_contains:" in Path(spec["tasks"]).read_text()
        for spec in manifest["domains"].values()
    )


def test_trial_seed_is_stable_and_trial_specific():
    first = derive_trial_seed(20260918, "commerce_cart_001", 0)
    assert first == derive_trial_seed(20260918, "commerce_cart_001", 0)
    assert first != derive_trial_seed(20260918, "commerce_cart_001", 1)
