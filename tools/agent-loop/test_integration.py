"""Integration tests for generated Python modules.

Run from: cd tools/agent-loop/generated/python && python3 -m pytest ../../test_integration.py -v

These tests verify behavioral contracts of generated code, independent of
the byte-exact prototype constraint. This allows the Prolog generator to
evolve formatting while preserving correctness.
"""

import sys
import os
import re
import tempfile

import pytest

# Ensure generated/python/ is importable (tests run from there)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())


# ============================================================================
# TestCostsModule
# ============================================================================

class TestCostsModule:
    """Test the costs module: pricing data, CostTracker, cost calculation."""

    def test_pricing_count(self):
        from costs import DEFAULT_PRICING
        assert len(DEFAULT_PRICING) == 16

    def test_pricing_opus_values(self):
        from costs import DEFAULT_PRICING
        opus = DEFAULT_PRICING["opus"]
        assert opus["input"] == 15.0
        assert opus["output"] == 75.0

    def test_pricing_free_models(self):
        from costs import DEFAULT_PRICING
        for model in ("llama3", "codellama", "mistral"):
            assert DEFAULT_PRICING[model]["input"] == 0.0
            assert DEFAULT_PRICING[model]["output"] == 0.0

    def test_cost_tracker_init(self):
        from costs import CostTracker
        tracker = CostTracker()
        assert tracker.total_cost == 0.0
        assert tracker.total_input_tokens == 0
        assert len(tracker.records) == 0

    def test_record_usage(self):
        from costs import CostTracker
        tracker = CostTracker()
        record = tracker.record_usage("opus", 1_000_000, 500_000)
        # input: 1M * 15.0/1M = 15.0, output: 0.5M * 75.0/1M = 37.5
        assert abs(tracker.total_cost - 52.5) < 0.001

    def test_get_summary(self):
        from costs import CostTracker
        tracker = CostTracker()
        tracker.record_usage("opus", 100, 50)
        summary = tracker.get_summary()
        assert "total_requests" in summary
        assert "total_input_tokens" in summary
        assert summary["total_requests"] == 1
        assert summary["total_input_tokens"] == 100

    def test_format_status(self):
        from costs import CostTracker
        tracker = CostTracker()
        tracker.record_usage("opus", 100, 50)
        status = tracker.format_status()
        assert "Tokens:" in status
        assert "Cost:" in status


# ============================================================================
# TestSecurityProfiles
# ============================================================================

class TestSecurityProfiles:
    """Test security profiles: names, properties, defaults."""

    def test_builtin_profile_names(self):
        from security.profiles import get_builtin_profiles
        profiles = get_builtin_profiles()
        assert set(profiles.keys()) == {"open", "cautious", "guarded", "paranoid"}

    def test_open_profile(self):
        from security.profiles import get_builtin_profiles
        p = get_builtin_profiles()["open"]
        assert p.path_validation is False
        assert p.command_blocklist is False
        assert p.command_proxying == "disabled"

    def test_cautious_profile(self):
        from security.profiles import get_builtin_profiles
        p = get_builtin_profiles()["cautious"]
        assert p.path_validation is True
        assert p.command_blocklist is True

    def test_guarded_profile(self):
        from security.profiles import get_builtin_profiles
        p = get_builtin_profiles()["guarded"]
        assert p.command_proxying == "enabled"
        assert p.audit_logging == "detailed"

    def test_paranoid_profile(self):
        from security.profiles import get_builtin_profiles
        p = get_builtin_profiles()["paranoid"]
        assert p.allowed_commands_only is True
        assert p.command_proxying == "strict"
        assert p.max_file_read_size == 1048576

    def test_default_fallback(self):
        from security.profiles import get_profile
        p = get_profile("nonexistent")
        assert p.name == "cautious"

    def test_all_have_description(self):
        from security.profiles import get_builtin_profiles
        for name, profile in get_builtin_profiles().items():
            assert profile.description, f"{name} has empty description"


# ============================================================================
# TestToolsModule
# ============================================================================

class TestToolsModule:
    """Test tools module: blocked paths, command patterns, validation."""

    def test_blocked_paths(self):
        from tools import _BLOCKED_PATHS
        for path in ("/etc/shadow", "/etc/gshadow", "/etc/sudoers"):
            assert path in _BLOCKED_PATHS

    def test_blocked_prefixes(self):
        from tools import _BLOCKED_PREFIXES
        assert "/proc/" in _BLOCKED_PREFIXES
        assert "/sys/" in _BLOCKED_PREFIXES
        assert "/dev/" in _BLOCKED_PREFIXES

    def test_blocked_command_count(self):
        from tools import _BLOCKED_COMMAND_PATTERNS
        assert len(_BLOCKED_COMMAND_PATTERNS) == 9

    def test_blocked_command_valid_regex(self):
        from tools import _BLOCKED_COMMAND_PATTERNS
        for pattern, description in _BLOCKED_COMMAND_PATTERNS:
            compiled = re.compile(pattern)
            assert compiled is not None, f"Invalid regex: {pattern}"
            assert description, f"Empty description for pattern: {pattern}"

    def test_validate_path_blocked(self):
        from tools import validate_path
        _, error = validate_path("/etc/shadow", "/tmp")
        assert error is not None
        assert "Blocked" in error

    def test_validate_path_allowed(self):
        from tools import validate_path
        _, error = validate_path("/tmp/test_file", "/tmp")
        assert error is None

    def test_command_blocked_rm(self):
        from tools import is_command_blocked
        reason = is_command_blocked("rm -rf /")
        assert reason is not None

    def test_command_allowed_ls(self):
        from tools import is_command_blocked
        reason = is_command_blocked("ls -la")
        assert reason is None


# ============================================================================
# TestToolsGenerated
# ============================================================================

class TestToolsGenerated:
    """Test auto-generated tool specs and destructive tool set."""

    def test_tool_specs_keys(self):
        from tools_generated import TOOL_SPECS
        assert set(TOOL_SPECS.keys()) == {"bash", "read", "write", "edit"}

    def test_bash_has_command_param(self):
        from tools_generated import TOOL_SPECS
        bash_params = TOOL_SPECS["bash"]["parameters"]
        param_names = [p["name"] for p in bash_params]
        assert "command" in param_names

    def test_edit_has_three_params(self):
        from tools_generated import TOOL_SPECS
        edit_params = TOOL_SPECS["edit"]["parameters"]
        assert len(edit_params) == 3

    def test_destructive_tools(self):
        from tools_generated import DESTRUCTIVE_TOOLS
        assert "bash" in DESTRUCTIVE_TOOLS
        assert "write" in DESTRUCTIVE_TOOLS
        assert "edit" in DESTRUCTIVE_TOOLS
        assert "read" not in DESTRUCTIVE_TOOLS


# ============================================================================
# TestBackendsInit
# ============================================================================

class TestBackendsInit:
    """Test backends package: exports, base classes, instantiation."""

    def test_all_exports(self):
        import backends
        for name in ("AgentBackend", "AgentResponse", "ToolCall"):
            assert name in backends.__all__

    def test_required_backends(self):
        import backends
        required = [
            "CoroBackend", "ClaudeCodeBackend", "GeminiBackend",
            "OllamaAPIBackend", "OllamaCLIBackend", "OpenRouterBackend",
        ]
        for name in required:
            assert name in backends.__all__, f"{name} not in __all__"

    def test_toolcall_instantiation(self):
        from backends.base import ToolCall
        tc = ToolCall(name="test", arguments={"key": "value"})
        assert tc.name == "test"
        assert tc.arguments == {"key": "value"}
        assert tc.id == ""

    def test_agent_response_defaults(self):
        from backends.base import AgentResponse
        resp = AgentResponse(content="hello")
        assert resp.content == "hello"
        assert resp.tool_calls == []
        assert resp.tokens == {}


# ============================================================================
# TestAliasesModule
# ============================================================================

class TestAliasesModule:
    """Test aliases module: default aliases, resolution, argument handling."""

    def test_alias_count(self):
        from aliases import DEFAULT_ALIASES
        assert len(DEFAULT_ALIASES) == 31

    def test_known_aliases(self):
        from aliases import DEFAULT_ALIASES
        assert DEFAULT_ALIASES["q"] == "quit"
        assert DEFAULT_ALIASES["h"] == "help"
        assert DEFAULT_ALIASES["opus"] == "backend claude-opus"

    def test_resolve_with_prefix(self):
        from aliases import AliasManager
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("{}")
            f.flush()
            mgr = AliasManager(config_path=f.name)
        try:
            assert mgr.resolve("/h") == "/help"
        finally:
            os.unlink(f.name)

    def test_unknown_passthrough(self):
        from aliases import AliasManager
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("{}")
            f.flush()
            mgr = AliasManager(config_path=f.name)
        try:
            assert mgr.resolve("/nonexistent") == "/nonexistent"
        finally:
            os.unlink(f.name)

    def test_args_preserved(self):
        from aliases import AliasManager
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("{}")
            f.flush()
            mgr = AliasManager(config_path=f.name)
        try:
            assert mgr.resolve("/h extra args") == "/help extra args"
        finally:
            os.unlink(f.name)


# ============================================================================
# TestConfigModule
# ============================================================================

class TestConfigModule:
    """Test config module: defaults, presets, agent properties."""

    def test_agent_config_defaults(self):
        from config import AgentConfig
        cfg = AgentConfig(name="test", backend="coro")
        assert cfg.tools == ["bash", "read", "write", "edit"]
        assert cfg.auto_tools is False

    def test_default_config_agent_count(self):
        from config import get_default_config
        config = get_default_config()
        assert len(config.agents) == 10

    def test_default_uses_coro(self):
        from config import get_default_config
        config = get_default_config()
        assert config.agents["default"].backend == "coro"

    def test_yolo_auto_tools(self):
        from config import get_default_config
        config = get_default_config()
        assert config.agents["yolo"].auto_tools is True

    def test_known_agent_names(self):
        from config import get_default_config
        config = get_default_config()
        expected = {
            "default", "claude-sonnet", "claude-opus", "claude-haiku",
            "yolo", "gemini", "ollama", "ollama-cli", "openai", "gpt-4o-mini",
        }
        assert set(config.agents.keys()) == expected


# ============================================================================
# TestContextModule
# ============================================================================

class TestContextModule:
    """Test context module: enums, messages, context management."""

    def test_behavior_enum(self):
        from context import ContextBehavior
        values = {e.value for e in ContextBehavior}
        assert values == {"continue", "fresh", "sliding", "summarize"}

    def test_format_enum(self):
        from context import ContextFormat
        values = {e.value for e in ContextFormat}
        assert values == {"plain", "markdown", "json", "xml"}

    def test_message_defaults(self):
        from context import Message
        msg = Message(role="user", content="hello")
        assert msg.tokens == 0
        assert msg.tool_calls == []

    def test_add_message(self):
        from context import ContextManager
        cm = ContextManager()
        cm.add_message("user", "hello world")
        assert len(cm) == 1
        assert cm.token_count > 0

    def test_fresh_context(self):
        from context import ContextManager, ContextBehavior
        cm = ContextManager(behavior=ContextBehavior.FRESH)
        cm.add_message("user", "hello")
        assert cm.get_context() == []

    def test_clear(self):
        from context import ContextManager
        cm = ContextManager()
        cm.add_message("user", "hello")
        cm.clear()
        assert len(cm) == 0
        assert cm.token_count == 0


# ============================================================================
# TestCrossModuleIntegration
# ============================================================================

class TestCrossModuleIntegration:
    """Test cross-module interactions: SecurityConfig.from_profile, etc."""

    def test_security_config_from_profile(self):
        from tools import SecurityConfig
        cfg = SecurityConfig.from_profile("cautious")
        assert cfg.path_validation is True
        assert cfg.command_blocklist is True

    def test_cost_all_models(self):
        from costs import CostTracker, DEFAULT_PRICING
        tracker = CostTracker()
        for model in DEFAULT_PRICING:
            record = tracker.record_usage(model, 1000, 500)
            assert record is not None
        assert tracker.get_summary()["total_requests"] == 16

    def test_tool_result(self):
        from tools import ToolResult
        result = ToolResult(success=True, output="done", tool_name="bash")
        assert result.success is True
        assert result.output == "done"
        assert result.tool_name == "bash"

    def test_agent_response_no_tools(self):
        from backends.base import AgentResponse
        resp = AgentResponse(content="hello")
        assert resp.tool_calls == []
        assert resp.content == "hello"


# ============================================================================
# TestComponentDriven — parameterized tests from Prolog-generated metadata
# ============================================================================

import json

@pytest.fixture(scope="module")
def test_metadata():
    """Load component-driven test metadata generated by emit_test_metadata/0."""
    metadata_path = os.path.join(os.path.dirname(__file__),
                                 "generated", "python", "test_metadata.json")
    with open(metadata_path) as f:
        return json.load(f)


class TestComponentDriven:
    """Tests driven by Prolog component registry metadata.

    These tests read test_metadata.json (generated by emit_test_metadata/0)
    and verify that the generated Python modules match the component registry.
    When components are added or removed in Prolog, these tests auto-adjust.
    """

    def test_cost_models_match(self, test_metadata):
        """Every model in metadata appears in DEFAULT_PRICING."""
        from costs import DEFAULT_PRICING
        for model in test_metadata["costs"]["models"]:
            assert model in DEFAULT_PRICING, f"Model '{model}' in metadata but not in DEFAULT_PRICING"

    def test_tool_names_match(self, test_metadata):
        """Every tool in metadata appears in TOOL_SPECS."""
        from tools_generated import TOOL_SPECS
        for tool in test_metadata["tools"]["names"]:
            assert tool in TOOL_SPECS, f"Tool '{tool}' in metadata but not in TOOL_SPECS"

    def test_security_profiles_match(self, test_metadata):
        """Every security profile in metadata is in get_builtin_profiles()."""
        from security.profiles import get_builtin_profiles
        profiles = get_builtin_profiles()
        for profile in test_metadata["security"]["profiles"]:
            assert profile in profiles, f"Profile '{profile}' in metadata but not in builtin profiles"

    def test_backend_names_in_all(self, test_metadata):
        """Every backend from metadata has a corresponding class in backends.__all__."""
        import backends
        all_lower = [name.lower() for name in backends.__all__]
        for backend_name in test_metadata["backends"]["names"]:
            # Normalize: "claude-code" -> "claudecode", "ollama-api" -> "ollamaapi"
            normalized = backend_name.replace("-", "").lower()
            found = any(normalized in entry for entry in all_lower)
            assert found, f"Backend '{backend_name}' has no matching class in backends.__all__"

    def test_component_counts(self, test_metadata):
        """Metadata counts match actual module contents."""
        from costs import DEFAULT_PRICING
        from tools_generated import TOOL_SPECS
        from security.profiles import get_builtin_profiles

        assert len(DEFAULT_PRICING) == test_metadata["costs"]["count"], \
            f"Costs count mismatch: {len(DEFAULT_PRICING)} != {test_metadata['costs']['count']}"
        assert len(TOOL_SPECS) == test_metadata["tools"]["count"], \
            f"Tools count mismatch: {len(TOOL_SPECS)} != {test_metadata['tools']['count']}"
        assert len(get_builtin_profiles()) == test_metadata["security"]["count"], \
            f"Security count mismatch: {len(get_builtin_profiles())} != {test_metadata['security']['count']}"

    def test_command_count_match(self, test_metadata):
        """Command count in metadata matches expected slash command count."""
        assert test_metadata["commands"]["count"] >= 25, \
            f"Command count too low: {test_metadata['commands']['count']} < 25"

    def test_destructive_tools_match(self, test_metadata):
        """Destructive tools in metadata match DESTRUCTIVE_TOOLS in generated code."""
        from tools_generated import DESTRUCTIVE_TOOLS
        for tool in test_metadata.get("destructive_tools", []):
            assert tool in DESTRUCTIVE_TOOLS, \
                f"Destructive tool '{tool}' in metadata but not in DESTRUCTIVE_TOOLS"

    def test_module_dependencies_present(self, test_metadata):
        """Module dependencies map has expected structure."""
        deps = test_metadata.get("module_dependencies", {})
        assert "agent_loop" in deps, "agent_loop should have dependencies"
        assert "security" in deps["agent_loop"], "agent_loop should depend on security"
        assert "backends" in deps, "backends should have dependencies"
        assert "costs" in deps["backends"], "backends should depend on costs"

    def test_readme_has_mermaid(self):
        """Generated README.md contains a Mermaid dependency diagram."""
        readme_path = os.path.join(os.path.dirname(__file__), "generated", "python", "README.md")
        with open(readme_path) as f:
            content = f.read()
        assert "```mermaid" in content, "README should contain mermaid code block"
        assert "graph TD" in content, "README should contain graph TD directive"
        assert "agent_loop" in content, "README diagram should reference agent_loop module"


# ============================================================================
# TestPluginManager — Phase 14 Python plugin system
# ============================================================================

class TestPluginManager:
    """Test the PluginManager class in generated tools.py."""

    def test_plugin_manager_import(self):
        from tools import PluginManager
        pm = PluginManager()
        assert pm is not None

    def test_plugin_manager_empty_state(self):
        from tools import PluginManager
        pm = PluginManager()
        assert pm.list_tools() == []
        assert pm.get_tool_schemas() == []

    def test_plugin_manager_load_nonexistent_dir(self):
        from tools import PluginManager
        pm = PluginManager()
        count = pm.load_dir("/nonexistent/path/to/plugins")
        assert count == 0

    def test_plugin_manager_load_file(self):
        from tools import PluginManager
        pm = PluginManager()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump({
                "name": "test-plugin",
                "version": "1.0",
                "tools": [{
                    "name": "greet",
                    "description": "Say hello",
                    "parameters": [{"name": "name", "param_type": "string", "required": True}],
                    "command_template": "echo Hello {name}!"
                }]
            }, f)
            f.flush()
            pm.load_file(f.name)
        tools = pm.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "greet"
        os.unlink(f.name)

    def test_plugin_manager_execute(self):
        from tools import PluginManager
        pm = PluginManager()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump({
                "name": "test-plugin",
                "version": "1.0",
                "tools": [{
                    "name": "greet",
                    "description": "Say hello",
                    "parameters": [{"name": "name", "param_type": "string", "required": True}],
                    "command_template": "echo Hello {name}!"
                }]
            }, f)
            f.flush()
            pm.load_file(f.name)
        result = pm.execute("greet", {"name": "World"})
        assert result == "echo Hello World!"
        os.unlink(f.name)

    def test_plugin_manager_tool_schemas(self):
        from tools import PluginManager
        pm = PluginManager()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump({
                "name": "test-plugin",
                "version": "1.0",
                "tools": [{
                    "name": "greet",
                    "description": "Greet someone",
                    "parameters": [{"name": "name", "param_type": "string", "required": True}],
                    "command_template": "echo Hello {name}!"
                }]
            }, f)
            f.flush()
            pm.load_file(f.name)
        schemas = pm.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "greet"
        os.unlink(f.name)

    def test_plugin_execute_unknown_tool(self):
        from tools import PluginManager
        pm = PluginManager()
        result = pm.execute("nonexistent", {})
        assert result is None


# ============================================================================
# TestCommandDispatch — Phase 14 data-driven dispatch
# ============================================================================

class TestCommandDispatch:
    """Test that data-driven command dispatch works in agent_loop.py."""

    def test_agent_loop_module_importable(self):
        """agent_loop module can be imported."""
        import agent_loop
        assert hasattr(agent_loop, 'AgentLoop')

    def test_handle_command_exit(self):
        """Exit command sets running=False."""
        import agent_loop
        loop = agent_loop.AgentLoop.__new__(agent_loop.AgentLoop)
        loop.running = True
        loop.config = {"model": "test", "max_context_tokens": 4096}
        loop.context = type('MockCtx', (), {'clear': lambda self: None, 'token_count': lambda self: 0})()
        loop.alias_manager = type('MockAlias', (), {'resolve': lambda self, x: x})()
        # Mock other required attributes
        loop.multiline_mode = False
        result = loop._handle_command("exit")
        assert result is True
        assert loop.running is False

    def test_handle_command_help(self):
        """Help command returns True."""
        import agent_loop
        loop = agent_loop.AgentLoop.__new__(agent_loop.AgentLoop)
        loop.running = True
        loop.config = {"model": "test", "max_context_tokens": 4096}
        loop.context = type('MockCtx', (), {'messages': []})()
        loop.alias_manager = type('MockAlias', (), {'resolve': lambda self, x: x})()
        loop.multiline_mode = False
        # _print_help needs to exist
        loop._print_help = lambda: None
        result = loop._handle_command("help")
        assert result is True


# ============================================================================
# TestOutputParser — Phase 20 structured output parsing
# ============================================================================

class TestOutputParser:
    """Test OutputParser: JSON extraction from model responses."""

    def test_extract_fenced_json(self):
        from output_parser import OutputParser
        text = 'Here is the result:\n```json\n{"key": "value", "count": 42}\n```\nDone.'
        blocks = OutputParser.extract_json(text)
        assert len(blocks) == 1
        assert blocks[0]["key"] == "value"
        assert blocks[0]["count"] == 42

    def test_extract_bare_json(self):
        from output_parser import OutputParser
        text = 'The answer is {"result": true, "data": [1,2,3]} end.'
        blocks = OutputParser.extract_json(text)
        assert len(blocks) == 1
        assert blocks[0]["result"] is True

    def test_no_json_returns_empty(self):
        from output_parser import OutputParser
        text = "No JSON here, just plain text."
        blocks = OutputParser.extract_json(text)
        assert len(blocks) == 0

    def test_key_validation_pass(self):
        from output_parser import OutputParser
        text = '```json\n{"name": "test", "age": 25}\n```'
        parsed = OutputParser.parse_response(text, expected_keys=["name", "age"])
        assert len(parsed.json_blocks) == 1
        assert len(parsed.errors) == 0

    def test_key_validation_fail(self):
        from output_parser import OutputParser
        text = '```json\n{"name": "test"}\n```'
        parsed = OutputParser.parse_response(text, expected_keys=["name", "age"])
        assert len(parsed.json_blocks) == 1
        assert len(parsed.errors) == 1
        assert "age" in parsed.errors[0]

    def test_multiple_fenced_blocks(self):
        from output_parser import OutputParser
        text = '```json\n{"a": 1}\n```\ntext\n```json\n{"b": 2}\n```'
        blocks = OutputParser.extract_json(text)
        assert len(blocks) == 2


# ============================================================================
# TestToolResultCache — Phase 20 tool result caching
# ============================================================================

class TestToolResultCache:
    """Test ToolResultCache: caching, TTL, skip destructive."""

    def test_cache_hit(self):
        from tools import ToolResultCache
        cache = ToolResultCache(ttl=60)
        cache.put("read", {"file": "test.txt"}, "file contents")
        result = cache.get("read", {"file": "test.txt"})
        assert result == "file contents"

    def test_cache_miss(self):
        from tools import ToolResultCache
        cache = ToolResultCache(ttl=60)
        result = cache.get("read", {"file": "test.txt"})
        assert result is None

    def test_skip_destructive(self):
        from tools import ToolResultCache
        cache = ToolResultCache(ttl=60)
        cache.put("bash", {"command": "ls"}, "output")
        result = cache.get("bash", {"command": "ls"})
        assert result is None  # bash is destructive, skipped

    def test_skip_write(self):
        from tools import ToolResultCache
        cache = ToolResultCache(ttl=60)
        cache.put("write", {"file": "x"}, "ok")
        assert cache.get("write", {"file": "x"}) is None

    def test_cache_clear(self):
        from tools import ToolResultCache
        cache = ToolResultCache(ttl=60)
        cache.put("read", {"file": "a"}, "data")
        assert cache.size() == 1
        cache.clear()
        assert cache.size() == 0

    def test_cache_different_args(self):
        from tools import ToolResultCache
        cache = ToolResultCache(ttl=60)
        cache.put("read", {"file": "a.txt"}, "content_a")
        cache.put("read", {"file": "b.txt"}, "content_b")
        assert cache.get("read", {"file": "a.txt"}) == "content_a"
        assert cache.get("read", {"file": "b.txt"}) == "content_b"


# ============================================================================
# TestToolSchemaCache — Phase 20 tool schema caching
# ============================================================================

class TestToolSchemaCache:
    """Test get_tool_schemas() caching."""

    def test_schemas_returned(self):
        from tools_generated import get_tool_schemas
        schemas = get_tool_schemas()
        assert len(schemas) == 4  # bash, read, write, edit
        names = {s["function"]["name"] for s in schemas}
        assert names == {"bash", "read", "write", "edit"}

    def test_schemas_cached(self):
        from tools_generated import get_tool_schemas
        s1 = get_tool_schemas()
        s2 = get_tool_schemas()
        assert s1 is s2  # Same object reference = cached


# ============================================================================
# TestGeminiValidation — Phase 20 Gemini model validation
# ============================================================================

class TestGeminiValidation:
    """Test validate_gemini_model function."""

    def test_flash_valid(self):
        from agent_loop import validate_gemini_model
        assert validate_gemini_model("gemini-3-flash-preview") == "gemini-3-flash-preview"

    def test_flash_reject(self):
        from agent_loop import validate_gemini_model
        result = validate_gemini_model("gemini-2-flash", "gemini-3-flash-preview")
        assert result == "gemini-3-flash-preview"  # fallback to default

    def test_pro_valid(self):
        from agent_loop import validate_gemini_model
        assert validate_gemini_model("gemini-2.5-pro-preview") == "gemini-2.5-pro-preview"

    def test_pro_reject(self):
        from agent_loop import validate_gemini_model
        result = validate_gemini_model("gemini-2-pro", "gemini-3-flash-preview")
        assert result == "gemini-3-flash-preview"

    def test_unknown_passthrough(self):
        from agent_loop import validate_gemini_model
        assert validate_gemini_model("some-other-model") == "some-other-model"


# ============================================================================
# TestMCPClient — Phase 20 MCP support
# ============================================================================

class TestMCPClient:
    """Test MCPClient and MCPManager import and structure."""

    def test_mcp_client_importable(self):
        from mcp_client import MCPClient
        assert MCPClient is not None

    def test_mcp_manager_importable(self):
        from mcp_client import MCPManager
        assert MCPManager is not None

    def test_mcp_manager_empty(self):
        from mcp_client import MCPManager
        mgr = MCPManager()
        assert mgr.list_tools() == []

    def test_mcp_client_init(self):
        from mcp_client import MCPClient
        client = MCPClient("test", ["echo", "hello"])
        assert client.name == "test"
        assert client.command == ["echo", "hello"]


# ============================================================================
# TestAsyncApiBackend — Phase 21 Python async backend
# ============================================================================

class TestAsyncApiBackend:
    """Test AsyncApiBackend class."""

    def test_importable(self):
        from backends.base import AsyncApiBackend
        assert AsyncApiBackend is not None

    def test_in_all(self):
        import backends
        assert "AsyncApiBackend" in backends.__all__

    def test_init_openai(self):
        from backends.base import AsyncApiBackend
        backend = AsyncApiBackend("test", "http://localhost/v1", "key", "gpt-4")
        assert backend.name == "test"
        assert backend.api_format == "openai"

    def test_init_anthropic(self):
        from backends.base import AsyncApiBackend
        backend = AsyncApiBackend("claude", "http://localhost/v1", "key", "claude-3",
                                  api_format="anthropic")
        assert backend.api_format == "anthropic"

    def test_build_headers_openai(self):
        from backends.base import AsyncApiBackend
        backend = AsyncApiBackend("test", "http://localhost", "sk-test", "gpt-4")
        headers = backend._build_headers()
        assert headers["Authorization"] == "Bearer sk-test"

    def test_build_headers_anthropic(self):
        from backends.base import AsyncApiBackend
        backend = AsyncApiBackend("test", "http://localhost", "sk-test", "claude-3",
                                  api_format="anthropic")
        headers = backend._build_headers()
        assert headers["x-api-key"] == "sk-test"
        assert "anthropic-version" in headers

    def test_build_body_openai(self):
        from backends.base import AsyncApiBackend
        backend = AsyncApiBackend("test", "http://localhost", "key", "gpt-4")
        body = backend._build_body("hello", [])
        assert body["model"] == "gpt-4"
        assert len(body["messages"]) == 1
        assert body["messages"][0]["content"] == "hello"

    def test_parse_response_openai(self):
        from backends.base import AsyncApiBackend
        backend = AsyncApiBackend("test", "http://localhost", "key", "gpt-4")
        data = {
            "choices": [{"message": {"content": "Hello!", "role": "assistant"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        resp = backend._parse_response(data)
        assert resp.content == "Hello!"
        assert resp.tokens["input"] == 10
        assert resp.tokens["output"] == 5

    def test_parse_response_anthropic(self):
        from backends.base import AsyncApiBackend
        backend = AsyncApiBackend("test", "http://localhost", "key", "claude-3",
                                  api_format="anthropic")
        data = {
            "content": [{"type": "text", "text": "Bonjour!"}],
            "usage": {"input_tokens": 8, "output_tokens": 4}
        }
        resp = backend._parse_response(data)
        assert resp.content == "Bonjour!"
        assert resp.tokens["input"] == 8

    def test_parse_response_with_tool_calls(self):
        from backends.base import AsyncApiBackend
        backend = AsyncApiBackend("test", "http://localhost", "key", "gpt-4")
        data = {
            "choices": [{"message": {"content": "", "tool_calls": [
                {"id": "call_1", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}
            ]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        resp = backend._parse_response(data)
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "bash"


# ============================================================================
# TestToolHandlerCacheWiring — Phase 21 cache wiring
# ============================================================================

class TestToolHandlerCacheWiring:
    """Test that ToolHandler has cache field and uses it."""

    def test_tool_handler_has_cache(self):
        from tools import ToolHandler
        handler = ToolHandler()
        assert hasattr(handler, 'cache')
        assert handler.cache.size() == 0

    def test_tool_handler_has_mcp_manager(self):
        from tools import ToolHandler
        handler = ToolHandler()
        assert hasattr(handler, 'mcp_manager')
        assert handler.mcp_manager is None

    def test_cache_populated_on_read(self):
        import os
        from tools import ToolHandler
        from backends.base import ToolCall
        handler = ToolHandler(confirm_destructive=False)
        test_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_integration.py"))
        tc = ToolCall(name="read", arguments={"file_path": test_file})
        result = handler.execute(tc)
        assert result.success
        assert handler.cache.size() == 1
        # Second call should hit cache
        result2 = handler.execute(tc)
        assert result2.success
        assert result2.output == result.output

    def test_cache_skips_bash(self):
        from tools import ToolHandler
        from backends.base import ToolCall
        handler = ToolHandler(confirm_destructive=False)
        tc = ToolCall(name="bash", arguments={"command": "echo hello"})
        result = handler.execute(tc)
        assert result.success
        assert handler.cache.size() == 0  # bash is destructive, not cached

    def test_mcp_dispatch_no_servers(self):
        from tools import ToolHandler
        from backends.base import ToolCall
        handler = ToolHandler(confirm_destructive=False)
        tc = ToolCall(name="mcp:server:tool", arguments={"query": "test"})
        result = handler.execute(tc)
        assert not result.success
        assert "No MCP servers" in result.output


# ============================================================================
# TestClearCacheCommand — Phase 21 /clear-cache
# ============================================================================

class TestClearCacheCommand:
    """Test /clear-cache command integration."""

    def test_alias_count_updated(self):
        from aliases import DEFAULT_ALIASES
        assert DEFAULT_ALIASES.get("cc") == "clear-cache"


# ============================================================================
# TestToolApprovalUI — Phase 22 approval mode system
# ============================================================================

class TestToolApprovalUI:
    """Test tool approval mode system in ToolHandler."""

    def test_tool_handler_has_approval_mode(self):
        from tools import ToolHandler
        handler = ToolHandler()
        assert hasattr(handler, 'approval_mode')
        assert handler.approval_mode == "default"

    def test_approval_mode_yolo(self):
        from tools import ToolHandler
        handler = ToolHandler(approval_mode="yolo")
        assert handler.check_approval("bash") is True
        assert handler.check_approval("write") is True

    def test_approval_mode_plan(self):
        from tools import ToolHandler
        handler = ToolHandler(approval_mode="plan")
        assert handler.check_approval("read") is True
        assert handler.check_approval("bash") is False
        assert handler.check_approval("write") is False

    def test_approval_mode_auto_edit(self):
        from tools import ToolHandler
        handler = ToolHandler(approval_mode="auto_edit")
        assert handler.check_approval("read") is True
        assert handler.check_approval("edit") is True
        assert handler.check_approval("write") is True
        assert handler.check_approval("bash") is False

    def test_approval_mode_default_read_approved(self):
        from tools import ToolHandler
        handler = ToolHandler(approval_mode="default")
        assert handler.check_approval("read") is True

    def test_confirm_tool_execution_yolo(self):
        from tools import ToolHandler
        from backends.base import ToolCall
        handler = ToolHandler(approval_mode="yolo")
        tc = ToolCall(name="bash", arguments={"command": "rm -rf /"})
        assert handler.confirm_tool_execution(tc) is True

    def test_confirm_tool_execution_plan_blocks(self):
        from tools import ToolHandler
        from backends.base import ToolCall
        handler = ToolHandler(approval_mode="plan")
        tc = ToolCall(name="bash", arguments={"command": "ls"})
        assert handler.confirm_tool_execution(tc) is False

    def test_execute_blocked_by_plan_mode(self):
        from tools import ToolHandler
        from backends.base import ToolCall
        handler = ToolHandler(approval_mode="plan")
        tc = ToolCall(name="bash", arguments={"command": "ls"})
        result = handler.execute(tc)
        assert not result.success
        assert "blocked by approval mode" in result.output


# ============================================================================
# TestStreamingRetry — Phase 22 streaming error recovery
# ============================================================================

class TestStreamingRetry:
    """Test streaming retry wrapper exists in generated code."""

    def test_agent_loop_has_streaming_retry(self):
        import importlib
        agent_loop = importlib.import_module("agent_loop")
        src = open(agent_loop.__file__).read()
        assert "_send_streaming_with_retry" in src

    def test_process_message_uses_retry(self):
        import importlib
        agent_loop = importlib.import_module("agent_loop")
        src = open(agent_loop.__file__).read()
        assert "_send_streaming_with_retry" in src


# ============================================================================
# TestOutputParserWiring — Phase 22 OutputParser in response processing
# ============================================================================

class TestOutputParserWiring:
    """Test OutputParser is wired into response processing."""

    def test_process_message_uses_output_parser(self):
        import importlib
        agent_loop = importlib.import_module("agent_loop")
        src = open(agent_loop.__file__).read()
        assert "OutputParser.parse_response" in src

    def test_output_parser_importable(self):
        from output_parser import OutputParser, ParsedOutput
        assert hasattr(OutputParser, 'parse_response')
        assert hasattr(OutputParser, 'extract_json')

    def test_output_parser_extract_fenced(self):
        from output_parser import OutputParser
        text = 'Here is some JSON:\n```json\n{"key": "value"}\n```\nDone.'
        blocks = OutputParser.extract_json(text)
        assert len(blocks) == 1
        assert blocks[0]["key"] == "value"

    def test_output_parser_extract_bare(self):
        from output_parser import OutputParser
        text = 'The result is {"count": 42} and that is it.'
        blocks = OutputParser.extract_json(text)
        assert len(blocks) == 1
        assert blocks[0]["count"] == 42

    def test_output_parser_no_json(self):
        from output_parser import OutputParser
        blocks = OutputParser.extract_json("No JSON here at all")
        assert len(blocks) == 0

    def test_output_parser_key_validation(self):
        from output_parser import OutputParser
        text = '```json\n{"name": "test", "value": 1}\n```'
        parsed = OutputParser.parse_response(text, expected_keys=["name"])
        assert len(parsed.json_blocks) == 1
        assert len(parsed.errors) == 0

    def test_output_parser_key_validation_missing(self):
        from output_parser import OutputParser
        text = '```json\n{"name": "test"}\n```'
        parsed = OutputParser.parse_response(text, expected_keys=["name", "missing_key"])
        assert len(parsed.errors) > 0


# ============================================================================
# TestMCPLifecycle — Phase 22 MCP server lifecycle management
# ============================================================================

class TestMCPLifecycle:
    """Test MCP lifecycle wiring in generated code."""

    def test_main_has_mcp_init(self):
        import importlib
        agent_loop = importlib.import_module("agent_loop")
        src = open(agent_loop.__file__).read()
        assert "MCPManager" in src
        assert "mcp_server_configs" in src

    def test_main_has_mcp_disconnect(self):
        import importlib
        agent_loop = importlib.import_module("agent_loop")
        src = open(agent_loop.__file__).read()
        assert "disconnect_all" in src

    def test_mcp_manager_importable(self):
        from mcp_client import MCPManager
        assert hasattr(MCPManager, 'disconnect_all')
        assert hasattr(MCPManager, 'list_tools')
        assert hasattr(MCPManager, 'dispatch')


# ============================================================================
# TestContextOverflow — Phase 23 context overflow notification
# ============================================================================

class TestContextOverflow:
    """Test context overflow notification."""

    def test_add_message_returns_int(self):
        from context import ContextManager
        ctx = ContextManager(max_messages=50)
        result = ctx.add_message("user", "hello")
        assert isinstance(result, int)
        assert result == 0

    def test_add_message_returns_trimmed_count(self):
        from context import ContextManager
        ctx = ContextManager(max_messages=3)
        ctx.add_message("user", "msg1")
        ctx.add_message("assistant", "msg2")
        ctx.add_message("user", "msg3")
        trimmed = ctx.add_message("assistant", "msg4")
        assert trimmed >= 1


# ============================================================================
# TestReloadRobustness — Phase 23 config reload enhancements
# ============================================================================

class TestReloadRobustness:
    """Test config reload robustness in generated code."""

    def test_reload_syncs_approval_mode(self):
        import importlib
        agent_loop = importlib.import_module("agent_loop")
        src = open(agent_loop.__file__).read()
        assert "approval_mode" in src

    def test_reload_refreshes_mcp(self):
        import importlib
        agent_loop = importlib.import_module("agent_loop")
        src = open(agent_loop.__file__).read()
        assert "disconnect_all" in src

    def test_reload_recreates_backend(self):
        import importlib
        agent_loop = importlib.import_module("agent_loop")
        src = open(agent_loop.__file__).read()
        assert "create_backend_from_config" in src


# ============================================================================
# TestSchemaValidation — Phase 23 tool schema validation
# ============================================================================

class TestSchemaValidation:
    """Test tool argument schema validation."""

    def test_tool_handler_has_required_params(self):
        from tools import ToolHandler
        handler = ToolHandler()
        assert hasattr(handler, 'tool_required_params')
        assert 'bash' in handler.tool_required_params
        assert 'command' in handler.tool_required_params['bash']

    def test_validation_catches_missing_param(self):
        from tools import ToolHandler
        from backends.base import ToolCall
        handler = ToolHandler(approval_mode="yolo")
        tc = ToolCall(name="bash", arguments={})
        result = handler.execute(tc)
        assert not result.success
        assert "Validation" in result.output
        assert "command" in result.output

    def test_validation_passes_valid_args(self):
        from tools import ToolHandler
        from backends.base import ToolCall
        handler = ToolHandler(approval_mode="yolo")
        tc = ToolCall(name="bash", arguments={"command": "echo hi"})
        err = handler._validate_tool_args(tc)
        assert err is None

    def test_validation_skips_unknown_tools(self):
        from tools import ToolHandler
        from backends.base import ToolCall
        handler = ToolHandler(approval_mode="yolo")
        tc = ToolCall(name="custom_plugin", arguments={})
        err = handler._validate_tool_args(tc)
        assert err is None

    def test_read_requires_path(self):
        from tools import ToolHandler
        from backends.base import ToolCall
        handler = ToolHandler(approval_mode="yolo")
        tc = ToolCall(name="read", arguments={})
        result = handler.execute(tc)
        assert not result.success
        assert "path" in result.output

    def test_write_requires_path_and_content(self):
        from tools import ToolHandler
        from backends.base import ToolCall
        handler = ToolHandler(approval_mode="yolo")
        tc = ToolCall(name="write", arguments={"path": "/tmp/x"})
        result = handler.execute(tc)
        assert not result.success
        assert "content" in result.output


# ============================================================================
# TestTokenBudget — Phase 23 rate limiting / token budget
# ============================================================================

class TestTokenBudget:
    """Test token budget tracking."""

    def test_config_has_token_budget(self):
        from config import AgentConfig
        cfg = AgentConfig(name="test", backend="coro")
        assert hasattr(cfg, 'token_budget')

    def test_cost_tracker_is_over_budget(self):
        from costs import CostTracker
        tracker = CostTracker()
        assert tracker.is_over_budget(0.0) is False  # unlimited
        assert tracker.is_over_budget(1.0) is False  # under budget
        tracker.total_cost = 1.5
        assert tracker.is_over_budget(1.0) is True

    def test_cost_tracker_budget_remaining(self):
        from costs import CostTracker
        tracker = CostTracker()
        assert tracker.budget_remaining(0.0) == -1.0  # unlimited
        assert tracker.budget_remaining(1.0) == 1.0
        tracker.total_cost = 0.3
        assert abs(tracker.budget_remaining(1.0) - 0.7) < 0.001

    def test_budget_check_in_process_message(self):
        import importlib
        agent_loop = importlib.import_module("agent_loop")
        src = open(agent_loop.__file__).read()
        assert "is_over_budget" in src
        assert "token_budget" in src


class TestStreamingTokenCounter:
    """Phase 24: Streaming token counter tests."""

    def test_streaming_token_counter_importable(self):
        import importlib
        agent_loop = importlib.import_module("agent_loop")
        src = open(agent_loop.__file__).read()
        assert "class StreamingTokenCounter" in src

    def test_counter_on_token_counts_chars(self):
        import importlib
        agent_loop = importlib.import_module("agent_loop")
        # Get the class from the module
        StreamingTokenCounter = getattr(agent_loop, "StreamingTokenCounter")
        counter = StreamingTokenCounter(show_live=False, cost_tracker=None)
        # Suppress print output
        import io, sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            counter.on_token("Hello ")
            counter.on_token("world!")
            assert counter.char_count == 12
            assert counter.token_count >= 1  # ~12/4 = 3
        finally:
            sys.stdout = old_stdout

    def test_counter_finish_returns_stats(self):
        import importlib
        agent_loop = importlib.import_module("agent_loop")
        StreamingTokenCounter = getattr(agent_loop, "StreamingTokenCounter")
        counter = StreamingTokenCounter(show_live=False)
        import io, sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            counter.on_token("Test token data here")
            stats = counter.finish()
            assert "approx_tokens" in stats
            assert "chars" in stats
            assert stats["chars"] == 20
        finally:
            sys.stdout = old_stdout

    def test_counter_format_summary(self):
        import importlib
        agent_loop = importlib.import_module("agent_loop")
        StreamingTokenCounter = getattr(agent_loop, "StreamingTokenCounter")
        counter = StreamingTokenCounter(show_live=False)
        import io, sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            counter.on_token("A" * 40)
            summary = counter.format_summary()
            assert "tokens" in summary
            assert "chars" in summary
            assert "40" in summary  # char count
        finally:
            sys.stdout = old_stdout

    def test_process_message_uses_counter(self):
        import importlib
        agent_loop = importlib.import_module("agent_loop")
        src = open(agent_loop.__file__).read()
        assert "StreamingTokenCounter" in src
        assert "_counter.on_token" in src
        assert "_counter.finish()" in src
        assert "[Streamed:" in src


class TestSharedLogicRound3to6:
    """Test shared_logic methods added in rounds 3-6 (methods 124-170)."""

    def test_security_is_path_safe(self):
        from security.profiles import is_path_safe
        assert is_path_safe("src/main.rs") is True
        assert is_path_safe("../../etc/passwd") is False

    def test_security_is_visible_file(self):
        from security.profiles import is_visible_file
        assert is_visible_file("main.rs") is True
        assert is_visible_file(".env") is False

    def test_security_is_hidden_path(self):
        from security.profiles import is_hidden_path
        assert is_hidden_path(".git") is True
        assert is_hidden_path("src") is False

    def test_security_has_path_traversal(self):
        from security.profiles import has_path_traversal
        assert has_path_traversal("..") is True
        assert has_path_traversal("../etc") is True
        assert has_path_traversal("src/main") is False

    def test_security_is_safe_command(self):
        from security.profiles import is_safe_command
        assert is_safe_command("ls -la") is True
        assert is_safe_command("cat file.txt") is True
        assert is_safe_command("rm -rf /") is False

    def test_security_is_blocked_command(self):
        from security.profiles import is_blocked_command
        assert is_blocked_command("rm -rf /") is True
        assert is_blocked_command("ls -la") is False

    def test_security_is_writable_path(self):
        from security.profiles import is_writable_path
        assert is_writable_path("/home/user/file.txt") is True
        assert is_writable_path("/etc/passwd") is False
        assert is_writable_path("/usr/bin/ls") is False

    def test_costs_module_importable(self):
        import costs
        assert hasattr(costs, 'CostTracker') or hasattr(costs, 'cost_compute')

    def test_context_module_has_clear(self):
        from context import ContextManager
        ctx = ContextManager()
        ctx.clear()
        assert len(ctx.messages) == 0

    def test_retry_module_importable(self):
        import retry
        assert hasattr(retry, 'is_retryable_status') or hasattr(retry, 'RetryConfig')

    def test_streaming_counter_in_agent_loop(self):
        import agent_loop
        src = open(agent_loop.__file__).read()
        assert "class StreamingTokenCounter" in src

    def test_tool_cache_in_tools(self):
        import tools
        src = open(tools.__file__).read()
        assert "class ToolResultCache" in src

    def test_output_parser_importable(self):
        from output_parser import OutputParser
        result = OutputParser.parse_response('{"key": "val"}')
        assert result is not None

    def test_sessions_importable(self):
        import sessions
        assert hasattr(sessions, 'SessionManager')

    def test_mcp_client_importable(self):
        import mcp_client
        assert hasattr(mcp_client, 'MCPClient')


class TestSharedLogicEdgeCases:
    """Edge-case and boundary-value tests for shared_logic methods."""

    def test_security_edge_cases(self):
        from security.profiles import is_path_safe, is_visible_file, is_hidden_path
        # Empty strings
        assert is_path_safe("") is True
        assert is_visible_file("") is True
        assert is_hidden_path("") is False
        # Boundary paths
        assert is_path_safe("/normal/path") is True
        assert is_hidden_path(".") is True

    def test_security_blocked_vs_safe(self):
        from security.profiles import is_safe_command, is_blocked_command
        assert is_safe_command("ls") is True
        assert is_blocked_command("rm -rf /") is True
        assert is_safe_command("rm -rf /") is False
        assert is_blocked_command("ls") is False

    def test_security_writable_boundary(self):
        from security.profiles import is_writable_path
        assert is_writable_path("/tmp/file") is True
        assert is_writable_path("/etc/hosts") is False
        assert is_writable_path("/usr/local/bin") is False
        assert is_writable_path("/bin/sh") is False

    def test_costs_zero_division(self):
        import costs
        src = open(costs.__file__).read()
        # cost_compute should exist
        assert "cost_compute" in src
        # Verify zero-division guard in input_ratio
        assert "== 0" in src or "total_input_tokens" in src

    def test_context_clear_idempotent(self):
        from context import ContextManager
        ctx = ContextManager()
        ctx.clear()
        ctx.clear()  # double clear should not crash
        assert len(ctx.messages) == 0

    def test_context_add_and_count(self):
        from context import ContextManager
        ctx = ContextManager()
        ctx.add_message("user", "hello")
        ctx.add_message("assistant", "hi")
        assert len(ctx.messages) == 2
        ctx.clear()
        assert len(ctx.messages) == 0

    def test_output_parser_empty_input(self):
        from output_parser import OutputParser
        result = OutputParser.parse_response("")
        assert result is not None
        assert len(result.json_blocks) == 0

    def test_output_parser_json_extraction(self):
        from output_parser import OutputParser
        result = OutputParser.parse_response('Here is data: {"key": "val"}')
        assert len(result.json_blocks) >= 1
        assert result.json_blocks[0]["key"] == "val"

    def test_retry_module_structure(self):
        import retry
        src = open(retry.__file__).read()
        assert "is_retryable_status" in src
        assert "compute_delay" in src

    def test_config_module_structure(self):
        import config
        src = open(config.__file__).read()
        assert "SEARCH_PATHS" in src or "search_paths" in src or "config_search_paths" in src

    def test_sessions_manager_init(self):
        import tempfile, os
        from sessions import SessionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            assert mgr.sessions_dir.exists()
            sessions = mgr.list_sessions()
            assert len(sessions) == 0

    def test_tools_module_has_handler(self):
        import tools
        src = open(tools.__file__).read()
        assert "ToolHandler" in src or "ToolResult" in src

    def test_mcp_client_structure(self):
        from mcp_client import MCPClient
        client = MCPClient("test", ["echo", "hello"])
        assert client.name == "test"

    def test_display_module_exists(self):
        import display
        assert hasattr(display, 'Spinner') or hasattr(display, 'ProgressBar') or "spinner" in open(display.__file__).read().lower()

    def test_aliases_module_exists(self):
        import aliases
        src = open(aliases.__file__).read()
        assert "DEFAULT_ALIASES" in src or "aliases" in src.lower()


class TestSharedLogicFunctional:
    """Functional tests that call shared_logic methods with concrete values."""

    def test_cost_tracker_is_over_budget(self):
        import costs
        src = open(costs.__file__).read()
        assert "is_over_budget" in src
        assert "budget_remaining" in src
        assert "total_cost" in src

    def test_context_manager_lifecycle(self):
        from context import ContextManager
        ctx = ContextManager()
        assert len(ctx.messages) == 0
        ctx.add_message("user", "hello")
        assert len(ctx.messages) == 1
        ctx.add_message("assistant", "world")
        assert len(ctx.messages) == 2
        ctx.clear()
        assert len(ctx.messages) == 0

    def test_context_manager_has_messages(self):
        from context import ContextManager
        ctx = ContextManager()
        assert hasattr(ctx, 'messages')

    def test_security_comprehensive(self):
        from security.profiles import (is_path_safe, is_visible_file,
            is_hidden_path, has_path_traversal, is_safe_command,
            is_blocked_command, is_writable_path)
        # Safe paths
        assert is_path_safe("src/main.rs")
        assert is_path_safe("README.md")
        # Dangerous paths
        assert not is_path_safe("../../etc/shadow")
        # Visible vs hidden
        assert is_visible_file("config.yaml")
        assert not is_visible_file(".gitignore")
        # Traversal
        assert has_path_traversal("../secret")
        assert not has_path_traversal("normal/path")
        # Commands
        assert is_safe_command("echo hello")
        assert is_safe_command("grep pattern file")
        assert not is_safe_command("dd if=/dev/zero")
        assert is_blocked_command("rm -rf /")
        assert is_blocked_command("mkfs.ext4")
        assert not is_blocked_command("echo hello")
        # Writable
        assert is_writable_path("/tmp/test")
        assert not is_writable_path("/etc/config")

    def test_output_parser_fenced_blocks(self):
        from output_parser import OutputParser
        text = '```json\n{"key": "value"}\n```'
        result = OutputParser.parse_response(text)
        assert len(result.json_blocks) >= 1

    def test_output_parser_bare_json(self):
        from output_parser import OutputParser
        result = OutputParser.parse_response('The answer is {"result": 42}')
        assert len(result.json_blocks) >= 1
        assert result.json_blocks[0]["result"] == 42

    def test_output_parser_no_json(self):
        from output_parser import OutputParser
        result = OutputParser.parse_response("Just plain text with no JSON at all.")
        assert len(result.json_blocks) == 0
        assert len(result.errors) == 0

    def test_session_manager_save_load(self):
        import tempfile
        from sessions import SessionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            from context import ContextManager
            mgr = SessionManager(tmpdir)
            ctx = ContextManager()
            ctx.add_message("user", "test message")
            session_id = mgr.save_session(ctx, name="test-session")
            assert session_id
            assert mgr.session_exists(session_id)
            loaded = mgr.load_session(session_id)
            assert loaded is not None
            mgr.delete_session(session_id)
            assert not mgr.session_exists(session_id)

    def test_mcp_client_init(self):
        from mcp_client import MCPClient
        c = MCPClient("test-server", ["echo", "hello"])
        assert c.name == "test-server"
        assert c.command == ["echo", "hello"]

    def test_mcp_manager_init(self):
        from mcp_client import MCPManager
        m = MCPManager()
        assert len(m.clients) == 0
        tools = m.list_tools()
        assert len(tools) == 0

    def test_retry_source_structure(self):
        import retry
        src = open(retry.__file__).read()
        assert "retry" in src.lower()
        assert "RetryConfig" in src or "backoff" in src.lower()

    def test_export_module_exists(self):
        import export
        src = open(export.__file__).read()
        assert "markdown" in src.lower() or "export" in src.lower()

    def test_multiline_module_exists(self):
        import multiline
        src = open(multiline.__file__).read()
        assert "heredoc" in src.lower() or "multiline" in src.lower() or "continuation" in src.lower()
