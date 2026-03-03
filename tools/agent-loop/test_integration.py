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
        assert len(DEFAULT_ALIASES) == 30

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
