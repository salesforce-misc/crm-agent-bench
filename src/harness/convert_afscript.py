"""Convert AFScript/AgentScript files to agent-harness graph YAML format.

Parses the indentation-based AFScript DSL and emits a graph config YAML
compatible with `python -m harness.graph_cli --config <output.yaml>`.

Handles:
- system block (instructions, messages)
- config block (description, label, etc.)
- language block (default_locale)
- variables block (linked, mutable)
- knowledge block
- start_agent / subagent / topic blocks (Agentforce dialect)
- reasoning blocks (instructions + actions)
- action definitions (inputs, outputs, target)
- transitions (@utils.transition to @topic.X / @subagent.X)

Usage:
    python -m harness.convert_afscript input.afscript -o output.yaml
    python -m harness.convert_afscript input.afscript  # prints to stdout
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ActionDef:
    name: str
    description: str = ""
    label: str = ""
    inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    target: str = ""
    require_user_confirmation: bool = False


@dataclass
class ReasoningBlock:
    instructions: str = ""
    action_refs: list[str] = field(default_factory=list)
    transitions: list[str] = field(default_factory=list)


@dataclass
class AgentNode:
    name: str
    node_type: str = "subagent"  # start_agent, subagent, topic
    label: str = ""
    description: str = ""
    reasoning: ReasoningBlock | None = None
    local_actions: dict[str, ActionDef] = field(default_factory=dict)
    handoff_to: list[str] = field(default_factory=list)


@dataclass
class AFScriptSpec:
    system_instructions: str = ""
    system_messages: dict[str, str] = field(default_factory=dict)
    config: dict[str, str] = field(default_factory=dict)
    language: dict[str, Any] = field(default_factory=dict)
    variables: dict[str, dict[str, Any]] = field(default_factory=dict)
    knowledge: dict[str, Any] = field(default_factory=dict)
    global_actions: dict[str, ActionDef] = field(default_factory=dict)
    nodes: dict[str, AgentNode] = field(default_factory=dict)
    entry_node: str | None = None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_TRANSITION_RE = re.compile(r"@utils\.transition\s+to\s+@(?:topic|subagent)\.(\w+)")
_GOTO_RE = re.compile(
    r"go_to_(\w+):\s*@utils\.transition\s+to\s+@(?:topic|subagent)\.(\w+)"
)
_ACTION_REF_RE = re.compile(r"(\w+):\s*@actions\.(\w+)")
_MULTILINE_ARROW_RE = re.compile(r"->\s*$")


def _get_indent(line: str) -> int:
    """Count leading spaces."""
    return len(line) - len(line.lstrip())


def _strip_comment(line: str) -> str:
    """Remove trailing comments (naive — doesn't handle strings)."""
    idx = line.find("#")
    if idx >= 0:
        # Only strip if # is not inside quotes (simplified check)
        before = line[:idx]
        if before.count('"') % 2 == 0 and before.count("'") % 2 == 0:
            return line[:idx]
    return line


def _collect_block(
    lines: list[str], start: int, base_indent: int
) -> tuple[list[str], int]:
    """Collect all lines belonging to a block at base_indent+.

    Returns (block_lines, next_line_index).
    """
    block = []
    i = start
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            block.append(line)
            i += 1
            continue
        if _get_indent(line) <= base_indent:
            break
        block.append(line)
        i += 1
    return block, i


def _extract_string_value(value: str) -> str:
    """Extract a string value, stripping quotes and multiline markers."""
    value = value.strip()
    if value.startswith(("|", "->")):
        return ""  # Multiline — handled by block collection
    for quote in ('"', "'"):
        if value.startswith(quote) and value.endswith(quote):
            return value[1:-1]
    return value


def _collect_multiline_string(
    lines: list[str], start: int, base_indent: int
) -> tuple[str, int]:
    """Collect a multiline string (| or ->) block."""
    parts = []
    i = start
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            parts.append("")
            i += 1
            continue
        if _get_indent(line) <= base_indent:
            break
        parts.append(stripped)
        i += 1
    return "\n".join(parts), i


def _parse_action_def(name: str, lines: list[str], base_indent: int) -> ActionDef:
    """Parse an action definition block."""
    action = ActionDef(name=name)
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = _strip_comment(line).strip()
        if not stripped:
            i += 1
            continue

        indent = _get_indent(lines[i])

        if stripped.startswith("description:"):
            val = stripped.split(":", 1)[1].strip()
            if val in ("|", "->", ""):
                text, j = _collect_multiline_string(lines, i + 1, indent)
                action.description = text
                i = j
            else:
                action.description = _extract_string_value(val)
                i += 1
        elif stripped.startswith("label:"):
            action.label = _extract_string_value(stripped.split(":", 1)[1])
            i += 1
        elif stripped.startswith("target:"):
            action.target = _extract_string_value(stripped.split(":", 1)[1])
            i += 1
        elif stripped.startswith("require_user_confirmation:"):
            action.require_user_confirmation = (
                stripped.split(":", 1)[1].strip().lower() == "true"
            )
            i += 1
        elif stripped == "inputs:":
            block, j = _collect_block(lines, i + 1, indent)
            action.inputs = _parse_params(block, indent + 4)
            i = j
        elif stripped == "outputs:":
            block, j = _collect_block(lines, i + 1, indent)
            action.outputs = _parse_params(block, indent + 4)
            i = j
        else:
            i += 1

    return action


def _parse_params(lines: list[str], base_indent: int) -> dict[str, dict[str, Any]]:
    """Parse input/output parameter blocks."""
    params: dict[str, dict[str, Any]] = {}
    current_name = None
    current_param: dict[str, Any] = {}
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = _strip_comment(line).strip()
        if not stripped:
            i += 1
            continue

        indent = _get_indent(lines[i])

        # Check if this is a new parameter name (has a colon and type or sub-block)
        if ":" in stripped and indent <= base_indent:
            # Save previous
            if current_name:
                params[current_name] = current_param

            parts = stripped.split(":", 1)
            current_name = parts[0].strip()
            type_val = parts[1].strip() if len(parts) > 1 else ""

            # Handle "param_name: type" on same line
            current_param = {}
            if type_val and type_val not in ("|", "->"):
                current_param["type"] = type_val
            i += 1
        elif current_name and ":" in stripped:
            # Sub-property of current param
            key, val = stripped.split(":", 1)
            key = key.strip()
            val = val.strip()
            if val in ("|", "->"):
                text, j = _collect_multiline_string(lines, i + 1, indent)
                current_param[key] = text
                i = j
            else:
                current_param[key] = _extract_string_value(val)
                i += 1
        else:
            i += 1

    if current_name:
        params[current_name] = current_param

    return params


def _parse_reasoning(lines: list[str], base_indent: int) -> ReasoningBlock:
    """Parse a reasoning block."""
    reasoning = ReasoningBlock()
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = _strip_comment(line).strip()
        if not stripped:
            i += 1
            continue

        indent = _get_indent(lines[i])

        if stripped.startswith("instructions:"):
            val = stripped.split(":", 1)[1].strip()
            if val in ("|", "->", ""):
                text, j = _collect_multiline_string(lines, i + 1, indent)
                reasoning.instructions = text
                i = j
            else:
                reasoning.instructions = _extract_string_value(val)
                i += 1
        elif stripped == "actions:":
            block, j = _collect_block(lines, i + 1, indent)
            # Parse action references and transitions
            for bline in block:
                bs = bline.strip()

                # Transition: go_to_X: @utils.transition to @topic.X
                goto_m = _GOTO_RE.search(bs)
                if goto_m:
                    reasoning.transitions.append(goto_m.group(2))
                    reasoning.action_refs.append(goto_m.group(1))
                    continue

                # Standalone transition
                trans_m = _TRANSITION_RE.search(bs)
                if trans_m:
                    reasoning.transitions.append(trans_m.group(1))
                    continue

                # Action reference: Name: @actions.ActionName
                action_m = _ACTION_REF_RE.search(bs)
                if action_m:
                    reasoning.action_refs.append(action_m.group(2))
                    continue

                # Simple action name with colon
                if ":" in bs and not bs.startswith("#"):
                    name_part = bs.split(":")[0].strip()
                    if name_part and name_part[0].isupper():
                        reasoning.action_refs.append(name_part)

            i = j
        else:
            i += 1

    return reasoning


def _normalize_node_name(name: str) -> str:
    """Normalize a node name to valid Python identifier (snake_case)."""
    # Replace spaces and hyphens with underscores
    name = re.sub(r"[\s\-]+", "_", name)
    # Remove non-alphanumeric except underscores
    name = re.sub(r"[^\w]", "", name)
    return name.lower()


def parse_afscript(text: str) -> AFScriptSpec:
    """Parse AFScript text into an AFScriptSpec."""
    spec = AFScriptSpec()
    lines = text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = _strip_comment(line).strip()

        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        indent = _get_indent(line)

        # Top-level blocks (indent == 0)
        if indent == 0:
            # system:
            if stripped == "system:":
                block, j = _collect_block(lines, i + 1, 0)
                _parse_system_block(spec, block)
                i = j
                continue

            # config:
            if stripped == "config:":
                block, j = _collect_block(lines, i + 1, 0)
                _parse_config_block(spec, block)
                i = j
                continue

            # language:
            if stripped == "language:":
                block, j = _collect_block(lines, i + 1, 0)
                _parse_simple_block(spec.language, block)
                i = j
                continue

            # variables:
            if stripped == "variables:":
                block, j = _collect_block(lines, i + 1, 0)
                _parse_variables_block(spec, block)
                i = j
                continue

            # knowledge:
            if stripped == "knowledge:":
                block, j = _collect_block(lines, i + 1, 0)
                _parse_simple_block(spec.knowledge, block)
                i = j
                continue

            # actions: (global)
            if stripped == "actions:":
                block, j = _collect_block(lines, i + 1, 0)
                _parse_global_actions(spec, block)
                i = j
                continue

            # start_agent [name]:
            m = re.match(r"start_agent\s*(\w+)?:", stripped)
            if m:
                name = m.group(1) or "start"
                name = _normalize_node_name(name)
                block, j = _collect_block(lines, i + 1, 0)
                node = _parse_agent_node(name, "start_agent", block)
                spec.nodes[name] = node
                spec.entry_node = name
                i = j
                continue

            # subagent name:
            m = re.match(r"subagent\s+(\w+):", stripped)
            if m:
                name = _normalize_node_name(m.group(1))
                block, j = _collect_block(lines, i + 1, 0)
                node = _parse_agent_node(name, "subagent", block)
                spec.nodes[name] = node
                i = j
                continue

            # topic name: (Agentforce dialect)
            m = re.match(r"topic\s+(\w+):", stripped)
            if m:
                name = _normalize_node_name(m.group(1))
                block, j = _collect_block(lines, i + 1, 0)
                node = _parse_agent_node(name, "topic", block)
                spec.nodes[name] = node
                i = j
                continue

            # connected_subagent name:
            m = re.match(r"connected_subagent\s+(\w+):", stripped)
            if m:
                name = _normalize_node_name(m.group(1))
                block, j = _collect_block(lines, i + 1, 0)
                node = _parse_agent_node(name, "connected_subagent", block)
                spec.nodes[name] = node
                i = j
                continue

        i += 1

    # Resolve transitions into handoff_to
    _resolve_transitions(spec)

    return spec


def _parse_system_block(spec: AFScriptSpec, lines: list[str]) -> None:
    """Parse the system block."""
    i = 0
    while i < len(lines):
        stripped = _strip_comment(lines[i]).strip()
        if not stripped:
            i += 1
            continue

        indent = _get_indent(lines[i])

        if stripped.startswith("instructions:"):
            val = stripped.split(":", 1)[1].strip()
            if val in ("|", "->", ""):
                text, j = _collect_multiline_string(lines, i + 1, indent)
                spec.system_instructions = text
                i = j
            else:
                spec.system_instructions = _extract_string_value(val)
                i += 1
        elif stripped == "messages:":
            block, j = _collect_block(lines, i + 1, indent)
            for mline in block:
                ms = _strip_comment(mline).strip()
                if ":" in ms and ms[0].isalpha():
                    key, val = ms.split(":", 1)
                    key = key.strip()
                    val = val.strip()
                    if val in ("|", "->"):
                        mind = _get_indent(mline)
                        text, _ = _collect_multiline_string(
                            block[block.index(mline) + 1 :], 0, mind
                        )
                        spec.system_messages[key] = text
                    else:
                        spec.system_messages[key] = _extract_string_value(val)
            i = j
        else:
            i += 1


def _parse_config_block(spec: AFScriptSpec, lines: list[str]) -> None:
    """Parse the config block."""
    for line in lines:
        stripped = _strip_comment(line).strip()
        if not stripped or ":" not in stripped:
            continue
        key, val = stripped.split(":", 1)
        spec.config[key.strip()] = _extract_string_value(val)


def _parse_simple_block(target: dict, lines: list[str]) -> None:
    """Parse a simple key-value block."""
    for line in lines:
        stripped = _strip_comment(line).strip()
        if not stripped or ":" not in stripped:
            continue
        key, val = stripped.split(":", 1)
        target[key.strip()] = _extract_string_value(val)


def _parse_variables_block(spec: AFScriptSpec, lines: list[str]) -> None:
    """Parse the variables block."""
    i = 0
    while i < len(lines):
        stripped = _strip_comment(lines[i]).strip()
        if not stripped:
            i += 1
            continue

        indent = _get_indent(lines[i])

        # Variable definition: Name: modifier type [= default]
        if ":" in stripped:
            parts = stripped.split(":", 1)
            var_name = parts[0].strip()
            var_rest = parts[1].strip()

            var_info: dict[str, Any] = {}

            # Parse modifier + type
            tokens = var_rest.split()
            if tokens:
                if tokens[0] in ("linked", "mutable"):
                    var_info["modifier"] = tokens[0]
                    tokens = tokens[1:]
                if tokens:
                    var_info["type"] = tokens[0]

            # Check for default value
            if "=" in var_rest:
                var_info["default"] = var_rest.split("=", 1)[1].strip()

            # Collect sub-properties
            sub_block, j = _collect_block(lines, i + 1, indent)
            for sline in sub_block:
                ss = _strip_comment(sline).strip()
                if ":" in ss:
                    sk, sv = ss.split(":", 1)
                    var_info[sk.strip()] = _extract_string_value(sv)

            spec.variables[var_name] = var_info
            i = j
        else:
            i += 1


def _parse_global_actions(spec: AFScriptSpec, lines: list[str]) -> None:
    """Parse global action definitions."""
    i = 0
    while i < len(lines):
        stripped = _strip_comment(lines[i]).strip()
        if not stripped:
            i += 1
            continue

        indent = _get_indent(lines[i])

        # Action name:
        if stripped.endswith(":") and not stripped.startswith(
            ("description", "label", "inputs", "outputs", "target")
        ):
            action_name = stripped[:-1].strip()
            block, j = _collect_block(lines, i + 1, indent)
            action = _parse_action_def(action_name, block, indent + 4)
            spec.global_actions[action_name] = action
            i = j
        else:
            i += 1


def _parse_agent_node(name: str, node_type: str, lines: list[str]) -> AgentNode:
    """Parse a start_agent, subagent, or topic block."""
    node = AgentNode(name=name, node_type=node_type)
    i = 0

    while i < len(lines):
        stripped = _strip_comment(lines[i]).strip()
        if not stripped:
            i += 1
            continue

        indent = _get_indent(lines[i])

        if stripped.startswith("description:"):
            val = stripped.split(":", 1)[1].strip()
            if val in ("|", "->", ""):
                text, j = _collect_multiline_string(lines, i + 1, indent)
                node.description = text
                i = j
            else:
                node.description = _extract_string_value(val)
                i += 1
        elif stripped.startswith("label:"):
            node.label = _extract_string_value(stripped.split(":", 1)[1])
            i += 1
        elif stripped == "reasoning:":
            block, j = _collect_block(lines, i + 1, indent)
            node.reasoning = _parse_reasoning(block, indent + 4)
            i = j
        elif stripped == "actions:":
            block, j = _collect_block(lines, i + 1, indent)
            _parse_node_actions(node, block, indent + 4)
            i = j
        else:
            i += 1

    return node


def _parse_node_actions(node: AgentNode, lines: list[str], base_indent: int) -> None:
    """Parse local action definitions within a node."""
    i = 0
    while i < len(lines):
        stripped = _strip_comment(lines[i]).strip()
        if not stripped:
            i += 1
            continue

        indent = _get_indent(lines[i])

        if stripped.endswith(":") and not stripped.startswith(
            ("description", "label", "inputs", "outputs", "target")
        ):
            action_name = stripped[:-1].strip()
            block, j = _collect_block(lines, i + 1, indent)
            action = _parse_action_def(action_name, block, indent + 4)
            node.local_actions[action_name] = action
            i = j
        else:
            i += 1


def _resolve_transitions(spec: AFScriptSpec) -> None:
    """Resolve transitions from reasoning blocks into handoff_to lists."""
    all_node_names = set(spec.nodes.keys())

    for node in spec.nodes.values():
        if node.reasoning and node.reasoning.transitions:
            for target in node.reasoning.transitions:
                normalized = _normalize_node_name(target)
                if (
                    normalized in all_node_names
                    and normalized != node.name
                    and normalized not in node.handoff_to
                ):
                    node.handoff_to.append(normalized)


# ---------------------------------------------------------------------------
# YAML emitter
# ---------------------------------------------------------------------------


def to_graph_yaml(spec: AFScriptSpec) -> dict[str, Any]:
    """Convert parsed AFScriptSpec to graph YAML dict."""
    # Build system prompt
    system_prompt = spec.system_instructions or "You are an AI Agent."
    if spec.config.get("description"):
        system_prompt = spec.config["description"] + "\n\n" + system_prompt

    # Build nodes
    nodes: dict[str, Any] = {}
    for name, node in spec.nodes.items():
        node_yaml: dict[str, Any] = {}

        node_yaml["description"] = node.description or node.label or name

        # Build instructions from reasoning
        if node.reasoning and node.reasoning.instructions:
            node_yaml["instructions"] = node.reasoning.instructions

        # Handoffs
        if node.handoff_to:
            node_yaml["handoff_to"] = node.handoff_to

        # Collect tools — from reasoning action refs + local actions + global actions
        tools: list[dict[str, str]] = []
        seen_tools: set[str] = set()

        # Actions referenced in reasoning
        if node.reasoning:
            for action_name in node.reasoning.action_refs:
                if action_name in seen_tools:
                    continue
                seen_tools.add(action_name)

                # Look up action definition (local first, then global)
                action_def = node.local_actions.get(
                    action_name
                ) or spec.global_actions.get(action_name)
                if action_def:
                    tool_entry: dict[str, str] = {"name": action_name}
                    desc = action_def.description
                    if action_def.label:
                        desc = desc or action_def.label
                    if desc:
                        tool_entry["description"] = desc
                    tools.append(tool_entry)
                else:
                    # Referenced but not defined — add as stub
                    tools.append({"name": action_name, "description": action_name})

        # Local actions not already referenced
        for action_name, action_def in node.local_actions.items():
            if action_name in seen_tools:
                continue
            seen_tools.add(action_name)
            tool_entry = {"name": action_name}
            desc = action_def.description or action_def.label
            if desc:
                tool_entry["description"] = desc
            tools.append(tool_entry)

        node_yaml["tools"] = tools
        nodes[name] = node_yaml

    result: dict[str, Any] = {
        "system_prompt": system_prompt,
    }

    if spec.entry_node:
        result["entry_node"] = spec.entry_node

    result["nodes"] = nodes

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _yaml_str_representer(dumper: yaml.Dumper, data: str) -> Any:
    """Use block scalar style for multiline strings."""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert AFScript/AgentScript files to agent-harness graph YAML"
    )
    parser.add_argument("input", help="Path to AFScript file")
    parser.add_argument("-o", "--output", help="Output YAML path (default: stdout)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    text = input_path.read_text()
    spec = parse_afscript(text)
    graph_yaml = to_graph_yaml(spec)

    # Custom YAML dumper for multiline strings
    yaml.add_representer(str, _yaml_str_representer)

    yaml_str = yaml.dump(
        graph_yaml, default_flow_style=False, sort_keys=False, width=120
    )

    # Add header comment
    header = (
        f"# Converted from AFScript: {input_path.name}\n"
        f"#\n"
        f"# Usage:\n"
        f"#   uv run python -m harness.graph_cli --config {args.output or 'output.yaml'}\n"
        f"\n"
    )
    yaml_str = header + yaml_str

    if args.output:
        Path(args.output).write_text(yaml_str)
        print(f"Written to {args.output}")
    else:
        print(yaml_str)


if __name__ == "__main__":
    main()
