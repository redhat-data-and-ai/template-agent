"""Unit tests for Jinja2 system-prompt rendering (config/prompts.py)."""

from datetime import date

import pytest

from deep_agent.src.agent.config.prompts import (
    DEFAULT_TEMPLATE_PATH,
    TEMPLATE_ENV_VAR,
    build_prompt_context,
    render_system_prompt,
)


class TestRenderSystemPrompt:
    def test_current_date_placeholder_is_rendered(self):
        out = render_system_prompt("Today's date is {{current_date}}.")
        assert out == f"Today's date is {date.today().isoformat()}."
        assert "{{" not in out

    def test_body_without_placeholders_is_unchanged(self):
        body = "You are an orchestrator.\n\nDelegate, never calculate."
        assert render_system_prompt(body) == body

    def test_agent_name_variable(self):
        out = render_system_prompt(
            "You are {{ agent_name }}.", agent_name="orchestrator"
        )
        assert out == "You are orchestrator."

    def test_extra_context_variables(self):
        out = render_system_prompt(
            "Hello {{ user_name }}.", extra_context={"user_name": "Sam"}
        )
        assert out == "Hello Sam."

    def test_unknown_variable_renders_empty_instead_of_raising(self):
        out = render_system_prompt("Value: [{{ never_supplied }}]")
        assert out == "Value: []"

    def test_invalid_template_falls_back_to_verbatim(self):
        body = "Broken {% if %} template"
        assert render_system_prompt(body) == body

    def test_runtime_render_error_falls_back_to_verbatim(self):
        body = "Result: {{ 1 / 0 }}"
        assert render_system_prompt(body) == body

    def test_empty_body_falls_back_to_bundled_default(self):
        out = render_system_prompt("")
        assert date.today().isoformat() in out
        assert "{{" not in out

    def test_bundled_default_template_exists_and_documents_variables(self):
        text = DEFAULT_TEMPLATE_PATH.read_text()
        assert "current_date" in text
        assert "agent_name" in text


class TestCustomTemplatePath:
    def test_template_path_argument_replaces_body(self, tmp_path):
        tpl = tmp_path / "custom.j2"
        tpl.write_text("Custom prompt for {{ agent_name }} on {{ current_date }}.")
        out = render_system_prompt(
            "ignored body", agent_name="orchestrator", template_path=tpl
        )
        assert out == f"Custom prompt for orchestrator on {date.today().isoformat()}."

    def test_env_var_selects_custom_template(self, tmp_path, monkeypatch):
        tpl = tmp_path / "env.j2"
        tpl.write_text("From env template.")
        monkeypatch.setenv(TEMPLATE_ENV_VAR, str(tpl))
        assert render_system_prompt("ignored body") == "From env template."

    def test_missing_custom_template_falls_back_to_body(self, tmp_path, monkeypatch):
        monkeypatch.setenv(TEMPLATE_ENV_VAR, str(tmp_path / "nope.j2"))
        assert render_system_prompt("the body") == "the body"


class TestBuildPromptContext:
    def test_defaults(self):
        ctx = build_prompt_context(agent_name="orchestrator")
        assert ctx["current_date"] == date.today().isoformat()
        assert ctx["date"] == ctx["current_date"]
        assert ctx["agent_name"] == "orchestrator"

    def test_extra_context_overrides(self):
        ctx = build_prompt_context(extra_context={"current_date": "2000-01-01"})
        assert ctx["current_date"] == "2000-01-01"


@pytest.mark.parametrize("body", ["", "   "])
def test_blank_body_uses_default_template(body):
    assert date.today().isoformat() in render_system_prompt(body)
