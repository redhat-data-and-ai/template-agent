"""Tests for platform memory path mapping and instruction loading."""

from deep_agent.src.memory.instructions import (
    USER_MEMORY_FILE,
    append_memory_instructions,
    load_memory_instructions,
    resolve_memory_namespaces,
)


class TestResolveMemoryNamespaces:
    """Stock agent.yaml ``memories`` is rewritten; custom paths pass through."""

    def test_stock_memories_maps_to_user_profile(self):
        assert resolve_memory_namespaces(["memories"]) == [USER_MEMORY_FILE]

    def test_stock_with_slashes_maps_to_user_profile(self):
        assert resolve_memory_namespaces(["/memories/"]) == [USER_MEMORY_FILE]

    def test_missing_uses_platform_file(self):
        assert resolve_memory_namespaces(None) == [USER_MEMORY_FILE]
        assert resolve_memory_namespaces([]) == [USER_MEMORY_FILE]

    def test_custom_paths_pass_through(self):
        custom = ["/memories/user_profile.md"]
        assert resolve_memory_namespaces(custom) == custom

    def test_multiple_custom_namespaces_pass_through(self):
        custom = ["user_mem", "shared"]
        assert resolve_memory_namespaces(custom) == custom


class TestLoadMemoryInstructions:
    """Instructions ship in the runtime package, not the config volume."""

    def test_loads_user_profile_path(self):
        import deep_agent.src.memory.instructions as instr

        instr._instructions_cache = None
        text = load_memory_instructions()
        assert USER_MEMORY_FILE in text
        assert "write_file" in text
        assert "edit_file" in text
        assert "complete sentence" in text.lower() or "complete" in text

    def test_append_adds_separator(self):
        result = append_memory_instructions("You are a helper.")
        assert result.startswith("You are a helper.\n\n---\n\n")
        assert USER_MEMORY_FILE in result

    def test_missing_template_returns_empty_then_leaves_prompt(
        self, tmp_path, monkeypatch
    ):
        import deep_agent.src.memory.instructions as instr

        missing = tmp_path / "missing.j2"
        monkeypatch.setattr(instr, "_instructions_path", lambda: missing)
        monkeypatch.setattr(instr, "_instructions_cache", None)
        monkeypatch.setattr(instr, "_instructions_source", None)
        assert instr.load_memory_instructions() == ""
        assert instr.append_memory_instructions("keep me") == "keep me"


class TestIsMemoryStoreKey:
    def test_store_key_and_full_path(self):
        from deep_agent.src.memory.instructions import (
            USER_MEMORY_FILE,
            USER_MEMORY_STORE_KEY,
            is_memory_store_key,
        )

        assert is_memory_store_key(USER_MEMORY_STORE_KEY)
        assert is_memory_store_key(USER_MEMORY_FILE)
        assert is_memory_store_key("/user_profile.md")
        assert not is_memory_store_key("quarterly.md")
        assert not is_memory_store_key("")


class TestMemoryFileText:
    def test_string_content(self):
        from deep_agent.src.memory.instructions import memory_file_text

        assert memory_file_text(
            {"content": "The user's date of birth is June 14, 2003."}
        ) == ("The user's date of birth is June 14, 2003.")

    def test_list_content(self):
        from deep_agent.src.memory.instructions import memory_file_text

        assert (
            memory_file_text({"content": ["line one", "line two"]})
            == "line one\nline two"
        )

    def test_missing_or_invalid(self):
        from deep_agent.src.memory.instructions import memory_file_text

        assert memory_file_text(None) == ""
        assert memory_file_text({}) == ""
        assert memory_file_text({"content": 3}) == ""


class TestInstructionsOverride:
    """Memory instructions are a prompt, so they are domain-specific.

    The bundled default is written around the demo agent — it names BMI
    and height/weight. An agent built from this template must be able to
    replace it by dropping a file in its config directory, with no code
    change.
    """

    def test_bundled_default_is_used_when_no_override_exists(
        self, monkeypatch, tmp_path
    ):
        import deep_agent.src.memory.instructions as instr

        monkeypatch.setenv("CONFIG_PATH", str(tmp_path))
        assert instr._instructions_path() == instr._BUNDLED_INSTRUCTIONS_PATH

    def test_config_override_wins(self, monkeypatch, tmp_path):
        import deep_agent.src.memory.instructions as instr

        override = tmp_path / "memory_instructions.md"
        override.write_text("## Remember only invoice totals.")
        monkeypatch.setenv("CONFIG_PATH", str(tmp_path))

        assert instr._instructions_path() == override

    def test_override_content_reaches_the_system_prompt(self, monkeypatch, tmp_path):
        import deep_agent.src.memory.instructions as instr

        override = tmp_path / "memory_instructions.md"
        override.write_text("## Remember only invoice totals.")
        monkeypatch.setenv("CONFIG_PATH", str(tmp_path))
        monkeypatch.setattr(instr, "_instructions_cache", None)
        monkeypatch.setattr(instr, "_instructions_source", None)

        prompt = instr.append_memory_instructions("You are an invoice agent.")

        assert "invoice totals" in prompt
        # The demo domain must not leak into an agent that overrode it.
        assert "BMI" not in prompt

    def test_switching_source_busts_the_cache(self, monkeypatch, tmp_path):
        """Source is part of the cache key: an override added later has no
        reason to differ in mtime from the bundled default."""
        import deep_agent.src.memory.instructions as instr

        monkeypatch.setenv("CONFIG_PATH", str(tmp_path))
        monkeypatch.setattr(instr, "_instructions_cache", None)
        monkeypatch.setattr(instr, "_instructions_source", None)

        bundled = instr.load_memory_instructions()
        assert "BMI" in bundled

        (tmp_path / "memory_instructions.md").write_text("## Custom rules.")
        assert instr.load_memory_instructions() == "## Custom rules."
