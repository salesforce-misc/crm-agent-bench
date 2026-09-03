"""Tests for the prompt loader."""

from harness.prompts.loader import (
    LANGUAGE_CODE_MAP,
    replace_context_placeholders,
    resolve_language,
)


class TestReplaceContextPlaceholders:
    def test_replace_with_value(self):
        template = 'You are {agent_label:-"Default Agent"}'
        result = replace_context_placeholders(template, agent_label="My Agent")
        assert result == "You are My Agent"

    def test_use_default_when_no_value(self):
        template = 'You are {agent_label:-"Default Agent"}'
        result = replace_context_placeholders(template)
        assert result == "You are Default Agent"

    def test_multiple_placeholders(self):
        template = 'I am {agent_label:-"Agent"} from {agent_company:-"Company"}'
        result = replace_context_placeholders(
            template, agent_label="Nova", agent_company="Salesforce"
        )
        assert result == "I am Nova from Salesforce"

    def test_partial_replacement(self):
        template = 'I am {agent_label:-"Agent"} from {agent_company:-"Company"}'
        result = replace_context_placeholders(template, agent_label="Nova")
        assert result == "I am Nova from Company"

    def test_list_value_joined(self):
        template = 'Languages: {agent_locale:-"English"}'
        result = replace_context_placeholders(
            template, agent_locale=["English", "French"]
        )
        assert result == "Languages: English, French"

    def test_no_placeholders(self):
        template = "No placeholders here"
        result = replace_context_placeholders(template)
        assert result == "No placeholders here"

    def test_all_context_vars(self):
        template = (
            '{agent_description:-"desc"} '
            '{agent_label:-"label"} '
            '{agent_company:-"company"} '
            '{agent_locale:-"locale"} '
            '{agent_role:-"role"} '
            '{agent_user_location:-"location"} '
            '{current_date_time:-"now"}'
        )
        result = replace_context_placeholders(
            template,
            agent_description="A helpful assistant",
            agent_label="Nova",
            agent_company="SF",
            agent_locale="en",
            agent_role="helper",
            agent_user_location="USA",
            current_date_time="2024-01-01",
        )
        assert result == "A helpful assistant Nova SF en helper USA 2024-01-01"


class TestResolveLanguage:
    def test_known_locale(self):
        assert resolve_language("en_US") == "English"
        assert resolve_language("ja") == "Japanese"
        assert resolve_language("fr") == "French"

    def test_unknown_locale_defaults_english(self):
        assert resolve_language("xx_XX") == "English"

    def test_none_defaults_english(self):
        assert resolve_language(None) == "English"


class TestLanguageCodeMap:
    def test_has_common_locales(self):
        assert "en_US" in LANGUAGE_CODE_MAP
        assert "ja" in LANGUAGE_CODE_MAP
        assert "es" in LANGUAGE_CODE_MAP
        assert "de" in LANGUAGE_CODE_MAP
        assert "fr" in LANGUAGE_CODE_MAP
