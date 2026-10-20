"""Prompt template loader with placeholder replacement.

Handles ``{context.*}`` placeholder substitution and locale-code resolution for
system-prompt generation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

# Locale code to display name mapping.
LANGUAGE_CODE_MAP: dict[str, str] = {
    "en_US": "English",
    "en_GB": "English (UK)",
    "en_AU": "English (Australian)",
    "fr": "French",
    "fr_CA": "French (Canadian)",
    "it": "Italian",
    "de": "German",
    "es": "Spanish",
    "es_MX": "Spanish (Mexico)",
    "ca": "Catalan",
    "nl_NL": "Dutch (Netherlands)",
    "da": "Danish",
    "no": "Norwegian",
    "sv": "Swedish",
    "fi": "Finnish",
    "ja": "Japanese",
    "zh_CN": "Chinese (Simplified)",
    "zh_TW": "Chinese (Traditional)",
    "ko": "Korean",
    "hi": "Hindi",
    "in": "Indonesian",
    "id": "Indonesian",
    "tl": "Filipino",
    "th": "Thai",
    "vi": "Vietnamese",
    "ms": "Malay",
    "pt_PT": "Portuguese (European)",
    "pt_BR": "Portuguese (Brazil)",
    "iw": "Hebrew",
    "he": "Hebrew",
    "ar": "Arabic",
    "tr": "Turkish",
    "bg": "Bulgarian",
    "hr": "Croatian",
}

# Context variable names used in prompt templates
CONTEXT_VARS = [
    "agent_description",
    "agent_label",
    "agent_company",
    "agent_locale",
    "agent_role",
    "agent_user_location",
    "current_date_time",
]


def replace_context_placeholders(system_prompt: str, **kwargs: Any) -> str:
    """Replace {var_name:-"default_value"} placeholders in a prompt template.

    Args:
        system_prompt: Template string with {var:-"default"} placeholders.
        **kwargs: Values to substitute. Keys should match variable names.

    Returns:
        The prompt with all placeholders resolved.
    """
    for var in CONTEXT_VARS:
        pattern = "{" + var + ':-"([^"]*)"}'
        match = re.search(pattern, system_prompt)
        if match:
            replacement = kwargs.get(var)
            if replacement is not None:
                if isinstance(replacement, list):
                    replacement = ", ".join(str(item) for item in replacement)
                replacement = str(replacement)
                system_prompt = re.sub(pattern, replacement, system_prompt)
            else:
                default_value = match.group(1)
                system_prompt = re.sub(pattern, default_value, system_prompt)
    return system_prompt


def load_yaml_prompt(yaml_path: str | Path) -> dict[str, Any]:
    """Load a YAML prompt config file.

    Args:
        yaml_path: Path to YAML file.

    Returns:
        Parsed YAML as a dict.
    """
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def load_prompt(
    yaml_path: str | Path,
    context: dict[str, str] | None = None,
    primary_language: str = "English",
    allowed_languages: str = "English",
) -> str:
    """Load a YAML prompt template, replace placeholders, and format language vars.

    This is the main entry point for prompt loading. It:
    1. Loads the YAML config
    2. Extracts the system_prompt field
    3. Replaces {var:-"default"} context placeholders
    4. Formats {primary_language} and {allowed_languages}

    Args:
        yaml_path: Path to the YAML prompt config.
        context: Context variables (agent_label, agent_description, etc.).
        primary_language: Primary language name (default: "English").
        allowed_languages: Comma-separated allowed languages (default: "English").

    Returns:
        The fully resolved system prompt string.
    """
    config = load_yaml_prompt(yaml_path)
    system_prompt = config.get("system_prompt", "")

    # If the caller left primary_language at the default but supplied a locale
    # code in the context (e.g. agent_locale="ja"), resolve the code to its
    # display name ("Japanese") so the prompt speaks the right language.
    if primary_language == "English" and context and context.get("agent_locale"):
        primary_language = resolve_language(context["agent_locale"])

    if context:
        system_prompt = replace_context_placeholders(system_prompt, **context)

    # Format language placeholders (these use simple {var} format, not {var:-"default"})
    try:
        system_prompt = system_prompt.format(
            primary_language=primary_language,
            allowed_languages=allowed_languages,
        )
    except KeyError:
        pass  # Template may not have language placeholders

    return system_prompt


def resolve_language(locale_code: str | None) -> str:
    """Resolve a locale code to a display language name.

    Args:
        locale_code: Locale code like "en_US", "ja", etc.

    Returns:
        Display language name, defaults to "English".
    """
    if not locale_code:
        return "English"
    return LANGUAGE_CODE_MAP.get(locale_code, "English")
