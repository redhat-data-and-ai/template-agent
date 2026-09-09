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
        monkeypatch.setattr(instr, "_INSTRUCTIONS_PATH", missing)
        monkeypatch.setattr(instr, "_instructions_cache", None)
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
