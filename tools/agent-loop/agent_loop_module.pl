%% agent_loop_module.pl - Prolog generator for UnifyWeaver Agent Loop
%%
%% This module defines agent backends, tools, security profiles, and
%% model pricing declaratively, then generates Python code from specs.
%%
%% Usage:
%%   swipl -g "generate_all, halt" agent_loop_module.pl

:- module(agent_loop_module, [
    generate_all/0,
    generate_module/1,
    agent_backend/2,
    tool_spec/2,
    security_profile/2,
    model_pricing/3,
    security_module/3
]).

%% =============================================================================
%% Agent Backend Definitions
%% =============================================================================

%% agent_backend(Name, Properties)
%% class_name/1 overrides the auto-generated CamelCase class name.
%% optional_import/1 marks backends that need try/except imports.
%% file_name/1 overrides the auto-generated snake_case file name.

agent_backend(coro, [
    type(cli),
    command("claude"),
    args(["--verbose"]),
    description("Coro-code CLI backend using single-task mode"),
    context_format(conversation_history),
    output_parser(coro_parser),
    supports_streaming(false)
]).

agent_backend(claude_code, [
    type(cli),
    command("claude"),
    args(["-p", "--model"]),
    description("Claude Code CLI backend using print mode"),
    default_model("sonnet"),
    models(["sonnet", "opus", "haiku"]),
    context_format(conversation_history),
    supports_streaming(false)
]).

agent_backend(gemini, [
    type(cli),
    command("gemini"),
    args(["-p", "-m", "--output-format", "text"]),
    description("Gemini CLI backend"),
    default_model("gemini-2.5-flash"),
    context_format(conversation_history),
    supports_streaming(false)
]).

agent_backend(claude_api, [
    type(api),
    class_name('ClaudeAPIBackend'),
    endpoint("https://api.anthropic.com/v1/messages"),
    model("claude-sonnet-4-20250514"),
    auth_env("ANTHROPIC_API_KEY"),
    auth_header("x-api-key"),
    description("Anthropic Claude API backend"),
    context_format(messages_array),
    supports_tools(true),
    supports_streaming(true),
    optional_import(true)
]).

agent_backend(openai_api, [
    type(api),
    class_name('OpenAIBackend'),
    file_name(openai_api),
    endpoint("https://api.openai.com/v1/chat/completions"),
    model("gpt-4o"),
    auth_env("OPENAI_API_KEY"),
    auth_header("Authorization"),
    auth_prefix("Bearer "),
    description("OpenAI API backend"),
    context_format(messages_array),
    supports_tools(true),
    supports_streaming(true),
    optional_import(true)
]).

agent_backend(ollama_api, [
    type(api),
    class_name('OllamaAPIBackend'),
    endpoint("http://localhost:11434/api/chat"),
    model("llama3"),
    description("Ollama REST API backend for local models"),
    default_host("localhost"),
    default_port(11434),
    context_format(messages_array),
    auth_required(false),
    supports_streaming(true)
]).

agent_backend(ollama_cli, [
    type(cli),
    class_name('OllamaCLIBackend'),
    command("ollama"),
    args(["run"]),
    description("Ollama CLI backend using 'ollama run' command"),
    default_model("llama3"),
    context_format(conversation_history),
    supports_streaming(false)
]).

agent_backend(openrouter_api, [
    type(api),
    class_name('OpenRouterBackend'),
    endpoint("https://openrouter.ai/api/v1/chat/completions"),
    model("anthropic/claude-sonnet-4-20250514"),
    auth_env("OPENROUTER_API_KEY"),
    auth_header("Authorization"),
    auth_prefix("Bearer "),
    description("OpenRouter API backend with model routing"),
    context_format(messages_array),
    supports_tools(true),
    supports_streaming(true)
]).

%% =============================================================================
%% Tool Definitions
%% =============================================================================

tool_spec(bash, [
    description("Execute a bash command"),
    parameters([
        param(command, string, required, "The command to execute")
    ]),
    confirmation_required(true),
    timeout(120)
]).

tool_spec(read, [
    description("Read a file"),
    parameters([
        param(path, string, required, "Path to file")
    ]),
    confirmation_required(false)
]).

tool_spec(write, [
    description("Write content to a file"),
    parameters([
        param(path, string, required, "Path to file"),
        param(content, string, required, "Content to write")
    ]),
    confirmation_required(true)
]).

tool_spec(edit, [
    description("Edit a file with search/replace"),
    parameters([
        param(path, string, required, "Path to file"),
        param(old_string, string, required, "Text to find"),
        param(new_string, string, required, "Replacement text")
    ]),
    confirmation_required(true)
]).

%% =============================================================================
%% Security Module Registry
%% =============================================================================

%% security_module(ModuleName, PrimaryExport, ExtraExports)
security_module(audit,        'AuditLogger',        []).
security_module(profiles,     'SecurityProfile',    [get_profile, get_builtin_profiles]).
security_module(proxy,        'CommandProxyManager', []).
security_module(path_proxy,   'PathProxyManager',   []).
security_module(proot_sandbox,'ProotSandbox',       ['ProotConfig']).

%% =============================================================================
%% Security Profile Definitions
%% =============================================================================

%% security_profile(Name, Properties)
security_profile(open, [
    description("No restrictions - for trusted manual use"),
    path_validation(false),
    command_blocklist(false),
    command_proxying(disabled),
    audit_logging(disabled)
]).

security_profile(cautious, [
    description("Basic safety for well-behaved agents like Claude Code"),
    path_validation(true),
    command_blocklist(true),
    command_proxying(disabled),
    audit_logging(basic)
]).

security_profile(guarded, [
    description("Actively protected and monitored for semi-autonomous agents"),
    path_validation(true),
    command_blocklist(true),
    blocked_commands(guarded_extra_blocks),
    safe_commands(paranoid_safe),
    command_proxying(enabled),
    audit_logging(detailed),
    network_isolation(localhost_only)
]).

security_profile(paranoid, [
    description("Maximum security for chaotic/untrusted agents"),
    path_validation(true),
    command_blocklist(true),
    allowed_commands_only(true),
    allowed_commands(paranoid_allowed),
    safe_commands(paranoid_safe),
    command_proxying(strict),
    audit_logging(forensic),
    network_isolation(blocked),
    anomaly_detection(true),
    max_file_read_size(1048576),
    max_file_write_size(10485760)
]).

%% Regex lists referenced by profiles
regex_list(guarded_extra_blocks, [
    "r'^sudo\\s'",
    "r'\\bbase64\\b.*\\|\\s*(bash|sh)'",
    "r'\\beval\\s'",
    "r'\\bnohup\\s'",
    "r'\\bdisown\\s'",
    "r'&\\s*$'",
    "r'\\bpython[23]?\\s+-c\\s.*os\\.system'",
    "r'\\bpython[23]?\\s+-c\\s.*subprocess'",
    "r'\\bpython[23]?\\s+-c\\s.*__import__'",
    "r'\\bnode\\s+-e\\s.*child_process'"
]).

regex_list(paranoid_safe, [
    "r'^ls(\\s|$)'",
    "r'^cat\\s'",
    "r'^head\\s'",
    "r'^tail\\s'",
    "r'^grep\\s'",
    "r'^echo\\s'",
    "r'^pwd$'",
    "r'^cd\\s'",
    "r'^wc\\s'",
    "r'^sort\\s'",
    "r'^diff\\s'",
    "r'^git\\s+(status|log|diff|show|branch)'"
]).

regex_list(paranoid_confirm, [
    "r'^find\\s+(?!.*(-exec|-execdir|-delete|-ok)\\b)[^;|&]*$'",
    "r'^python3\\s+[^-].*\\.py$'",
    "r'^node\\s+[^-].*\\.js$'"
]).

%% =============================================================================
%% Model Pricing Definitions
%% =============================================================================

%% model_pricing(ModelName, InputPricePerMillion, OutputPricePerMillion)
%% Claude models
model_pricing("claude-opus-4-20250514",    15.0,   75.0).
model_pricing("claude-sonnet-4-20250514",   3.0,   15.0).
model_pricing("claude-haiku-3-5-20241022",  0.80,   4.0).
%% Aliases
model_pricing("opus",    15.0,  75.0).
model_pricing("sonnet",   3.0,  15.0).
model_pricing("haiku",    0.80,  4.0).
%% OpenAI models
model_pricing("gpt-4o",        2.50, 10.0).
model_pricing("gpt-4o-mini",   0.15,  0.60).
model_pricing("gpt-4-turbo",  10.0,  30.0).
model_pricing("gpt-4",        30.0,  60.0).
model_pricing("gpt-3.5-turbo", 0.50,  1.50).
%% Gemini
model_pricing("gemini-2.5-flash", 0.075, 0.30).
model_pricing("gemini-2.5-pro",   1.25,  5.0).
%% Local models (free)
model_pricing("llama3",    0.0, 0.0).
model_pricing("codellama", 0.0, 0.0).
model_pricing("mistral",  0.0, 0.0).

%% =============================================================================
%% Loop Configuration
%% =============================================================================

loop_config([
    max_context_tokens(100000),
    max_messages(50),
    trim_strategy(oldest_first),
    display_tokens(true),
    display_tool_calls(true),
    confirm_destructive_tools(true)
]).

%% =============================================================================
%% Code Generation — Master Entry
%% =============================================================================

generate_all :-
    write('Generating agent loop files...'), nl,
    make_directory_path('generated/backends'),
    make_directory_path('generated/security'),
    generate_module(backends_init),
    generate_module(backends_base),
    forall(agent_backend(Name, _), generate_backend(Name)),
    generate_module(security_init),
    generate_module(security_profiles),
    generate_module(costs),
    generate_module(tools_generated),
    generate_module(readme),
    write('Done.'), nl.

%% Generate a single module by name
generate_module(backends_init)      :- generate_backends_init.
generate_module(backends_base)      :- generate_backends_base.
generate_module(security_init)      :- generate_security_init.
generate_module(security_profiles)  :- generate_security_profiles.
generate_module(costs)              :- generate_costs.
generate_module(tools_generated)    :- generate_tools.
generate_module(readme)             :- generate_readme.

%% =============================================================================
%% Generator: backends/__init__.py
%% =============================================================================

generate_backends_init :-
    open('generated/backends/__init__.py', write, S),
    write(S, '# Auto-generated by agent_loop_module.pl\n'),
    write(S, 'from .base import AgentBackend, AgentResponse, ToolCall\n'),
    %% Direct imports (non-optional backends)
    forall((agent_backend(Name, Props), \+ member(optional_import(true), Props)), (
        resolve_class_name(Name, ClassName),
        resolve_file_name(Name, FileName),
        format(S, 'from .~w import ~w~n', [FileName, ClassName])
    )),
    %% __all__ with direct backends
    write(S, '\n__all__ = [\n'),
    write(S, '    \'AgentBackend\', \'AgentResponse\', \'ToolCall\',\n'),
    findall(Name, (agent_backend(Name, Props), \+ member(optional_import(true), Props)), DirectBackends),
    write_all_entries(S, DirectBackends),
    write(S, ']\n'),
    %% Optional imports with try/except
    forall((agent_backend(Name, Props), member(optional_import(true), Props)), (
        resolve_class_name(Name, ClassName),
        resolve_file_name(Name, FileName),
        member(description(Desc), Props),
        format(S, '~n# ~w (optional - requires pip package)~n', [Desc]),
        format(S, 'try:~n', []),
        format(S, '    from .~w import ~w~n', [FileName, ClassName]),
        format(S, '    __all__.append(\'~w\')~n', [ClassName]),
        format(S, 'except ImportError:~n', []),
        format(S, '    pass~n', [])
    )),
    close(S),
    format('  Generated backends/__init__.py~n', []).

write_all_entries(_, []).
write_all_entries(S, [Name|Rest]) :-
    resolve_class_name(Name, ClassName),
    (Rest = [] ->
        format(S, '    \'~w\'~n', [ClassName])
    ;
        format(S, '    \'~w\',~n', [ClassName])
    ),
    write_all_entries(S, Rest).

%% =============================================================================
%% Generator: backends/base.py
%% =============================================================================

generate_backends_base :-
    open('generated/backends/base.py', write, S),
    write(S, '"""Abstract base class for agent backends."""\n\n'),
    write(S, 'from dataclasses import dataclass, field\n'),
    write(S, 'from typing import Any\n'),
    write(S, 'from abc import ABC, abstractmethod\n\n\n'),
    %% ToolCall dataclass
    write(S, '@dataclass\n'),
    write(S, 'class ToolCall:\n'),
    write(S, '    """Represents a tool call from the agent."""\n'),
    write(S, '    name: str\n'),
    write(S, '    arguments: dict[str, Any]\n'),
    write(S, '    id: str = ""\n\n\n'),
    %% AgentResponse dataclass
    write(S, '@dataclass\n'),
    write(S, 'class AgentResponse:\n'),
    write(S, '    """Response from an agent backend."""\n'),
    write(S, '    content: str\n'),
    write(S, '    tool_calls: list[ToolCall] = field(default_factory=list)\n'),
    write(S, '    tokens: dict[str, int] = field(default_factory=dict)\n'),
    write(S, '    raw: Any = None\n\n\n'),
    %% AgentBackend ABC
    write(S, 'class AgentBackend(ABC):\n'),
    write(S, '    """Abstract interface for agent backends."""\n\n'),
    write(S, '    @abstractmethod\n'),
    write(S, '    def send_message(self, message: str, context: list[dict], **kwargs) -> AgentResponse:\n'),
    write(S, '        """Send a message with context, get response.\n\n'),
    write(S, '        Optional kwargs:\n'),
    write(S, '            on_status: Callback for status updates (e.g. tool call progress)\n'),
    write(S, '        """\n'),
    write(S, '        raise NotImplementedError\n\n'),
    write(S, '    def parse_tool_calls(self, response: str) -> list[ToolCall]:\n'),
    write(S, '        """Extract tool calls from response. Override in subclasses."""\n'),
    write(S, '        return []\n\n'),
    write(S, '    def supports_streaming(self) -> bool:\n'),
    write(S, '        """Whether this backend supports streaming output."""\n'),
    write(S, '        return False\n\n'),
    write(S, '    @property\n'),
    write(S, '    def name(self) -> str:\n'),
    write(S, '        """Backend name for display."""\n'),
    write(S, '        return self.__class__.__name__\n'),
    close(S),
    format('  Generated backends/base.py~n', []).

%% =============================================================================
%% Generator: security/__init__.py
%% =============================================================================

generate_security_init :-
    open('generated/security/__init__.py', write, S),
    write(S, '"""Security subsystem for UnifyWeaver Agent Loop.\n\n'),
    write(S, 'Provides audit logging, enhanced security profiles, command proxying,\n'),
    write(S, 'PATH-based wrapper scripts, and proot filesystem isolation.\n'),
    write(S, '"""\n\n'),
    %% Imports — one import line per module, comma-separated exports
    forall(security_module(Mod, Primary, Extras), (
        AllNames = [Primary|Extras],
        atomic_list_concat(AllNames, ', ', NamesStr),
        format(S, 'from .~w import ~w~n', [Mod, NamesStr])
    )),
    %% __all__
    write(S, '\n__all__ = [\n'),
    findall(Export, (
        security_module(_, Primary, Extras),
        member(Export, [Primary|Extras])
    ), AllExports),
    write_quoted_list(S, AllExports),
    write(S, ']\n'),
    close(S),
    format('  Generated security/__init__.py~n', []).

%% =============================================================================
%% Generator: security/profiles.py
%% =============================================================================

generate_security_profiles :-
    open('generated/security/profiles.py', write, S),
    write(S, '"""Enhanced security profiles for the agent loop.\n\n'),
    write(S, 'Each profile defines a complete security posture: what\'s blocked, what\'s\n'),
    write(S, 'proxied, what\'s logged, and what isolation is applied.\n'),
    write(S, '"""\n\n'),
    write(S, 'from dataclasses import dataclass, field\n\n\n'),
    %% SecurityProfile dataclass
    write(S, '@dataclass\n'),
    write(S, 'class SecurityProfile:\n'),
    write(S, '    """Full security profile with all layer settings."""\n'),
    write(S, '    name: str\n'),
    write(S, '    description: str = \'\'\n\n'),
    write(S, '    # Layer 1: Path validation\n'),
    write(S, '    path_validation: bool = True\n'),
    write(S, '    blocked_paths: list[str] = field(default_factory=list)\n'),
    write(S, '    allowed_paths: list[str] = field(default_factory=list)\n\n'),
    write(S, '    # Layer 2: Command blocklist / allowlist\n'),
    write(S, '    command_blocklist: bool = True\n'),
    write(S, '    blocked_commands: list[str] = field(default_factory=list)\n'),
    write(S, '    allowed_commands: list[str] = field(default_factory=list)\n'),
    write(S, '    allowed_commands_only: bool = False  # If True, only allowed_commands may run\n\n'),
    write(S, '    # Layer 3: Command proxying\n'),
    write(S, '    command_proxying: str = \'disabled\'  # disabled, optional, enabled, strict\n\n'),
    write(S, '    # Layer 3.5: PATH-based wrapper scripts\n'),
    write(S, '    path_proxying: bool = False\n\n'),
    write(S, '    # Layer 4: Filesystem isolation\n'),
    write(S, '    proot_isolation: bool = False\n'),
    write(S, '    proot_allowed_dirs: list[str] = field(default_factory=list)\n\n'),
    write(S, '    # Layer 5: Audit logging\n'),
    write(S, '    audit_logging: str = \'disabled\'  # disabled, basic, detailed, forensic\n\n'),
    write(S, '    # Layer 6: Network isolation (future)\n'),
    write(S, '    network_isolation: str = \'disabled\'  # disabled, localhost_only, blocked\n\n'),
    write(S, '    # Layer 7: Anomaly detection (future)\n'),
    write(S, '    anomaly_detection: bool = False\n\n'),
    write(S, '    # Safe commands — subset of allowed_commands that skip confirmation\n'),
    write(S, '    # (read-only / harmless commands the user doesn\'t need to approve)\n'),
    write(S, '    safe_commands: list[str] = field(default_factory=list)\n\n'),
    write(S, '    # Resource limits\n'),
    write(S, '    max_file_read_size: int | None = None   # bytes, None = unlimited\n'),
    write(S, '    max_file_write_size: int | None = None\n\n\n'),
    %% Regex lists
    write(S, '# ── Built-in profiles ─────────────────────────────────────────────────────\n\n'),
    generate_regex_list_py(S, guarded_extra_blocks, '_GUARDED_EXTRA_BLOCKS'),
    write(S, '\n'),
    generate_regex_list_py(S, paranoid_safe, '_PARANOID_SAFE'),
    write(S, '\n'),
    generate_regex_list_py(S, paranoid_confirm, '_PARANOID_CONFIRM'),
    write(S, '\n'),
    write(S, '# Combined allowlist (safe + confirm)\n'),
    write(S, '_PARANOID_ALLOWED = _PARANOID_SAFE + _PARANOID_CONFIRM\n\n\n'),
    %% get_builtin_profiles()
    write(S, 'def get_builtin_profiles() -> dict[str, SecurityProfile]:\n'),
    write(S, '    """Return all built-in security profiles."""\n'),
    write(S, '    return {\n'),
    forall(security_profile(Name, Props), (
        generate_profile_entry(S, Name, Props)
    )),
    write(S, '    }\n\n\n'),
    %% get_profile()
    write(S, 'def get_profile(name: str) -> SecurityProfile:\n'),
    write(S, '    """Get a built-in profile by name, defaulting to cautious."""\n'),
    write(S, '    profiles = get_builtin_profiles()\n'),
    write(S, '    return profiles.get(name, profiles[\'cautious\'])\n'),
    close(S),
    format('  Generated security/profiles.py~n', []).

%% Write a regex_list as a Python list variable
generate_regex_list_py(S, ListName, PyVarName) :-
    regex_list(ListName, Patterns),
    format(S, '~w = [~n', [PyVarName]),
    forall(member(P, Patterns), (
        format(S, '    ~w,~n', [P])
    )),
    write(S, ']\n').

%% Write a single profile entry in get_builtin_profiles()
generate_profile_entry(S, Name, Props) :-
    member(description(Desc), Props),
    format(S, '        \'~w\': SecurityProfile(~n', [Name]),
    format(S, '            name=\'~w\',~n', [Name]),
    format(S, '            description=\'~w\',~n', [Desc]),
    %% Boolean fields with defaults
    (member(path_validation(PV), Props) ->
        py_bool(PV, PVPy), format(S, '            path_validation=~w,~n', [PVPy]) ; true),
    (member(command_blocklist(CB), Props) ->
        py_bool(CB, CBPy), format(S, '            command_blocklist=~w,~n', [CBPy]) ; true),
    (member(allowed_commands_only(ACO), Props) ->
        py_bool(ACO, ACOPy), format(S, '            allowed_commands_only=~w,~n', [ACOPy]) ; true),
    %% List fields (reference named regex lists)
    (member(blocked_commands(_), Props) ->
        format(S, '            blocked_commands=list(_GUARDED_EXTRA_BLOCKS),~n', []) ; true),
    (member(allowed_commands(ACRef), Props) ->
        (ACRef = paranoid_allowed ->
            format(S, '            allowed_commands=list(_PARANOID_ALLOWED),~n', [])
        ;
            true
        ) ; true),
    (member(safe_commands(_), Props) ->
        format(S, '            safe_commands=list(_PARANOID_SAFE),~n', []) ; true),
    %% String fields
    (member(command_proxying(CP), Props) ->
        format(S, '            command_proxying=\'~w\',~n', [CP]) ; true),
    (member(audit_logging(AL), Props) ->
        format(S, '            audit_logging=\'~w\',~n', [AL]) ; true),
    (member(network_isolation(NI), Props) ->
        format(S, '            network_isolation=\'~w\',~n', [NI]) ; true),
    (member(anomaly_detection(AD), Props) ->
        py_bool(AD, ADPy), format(S, '            anomaly_detection=~w,~n', [ADPy]) ; true),
    %% Resource limits
    (member(max_file_read_size(MFR), Props) ->
        format(S, '            max_file_read_size=~w,~n', [MFR]) ; true),
    (member(max_file_write_size(MFW), Props) ->
        format(S, '            max_file_write_size=~w,~n', [MFW]) ; true),
    write(S, '        ),\n').

%% =============================================================================
%% Generator: costs.py
%% =============================================================================

generate_costs :-
    open('generated/costs.py', write, S),
    %% Header and imports
    write(S, '"""Cost tracking for API usage."""\n\n'),
    write(S, 'from dataclasses import dataclass, field\n'),
    write(S, 'from datetime import datetime\n'),
    write(S, 'from typing import Any\n'),
    write(S, 'import json\n'),
    write(S, 'import os\n'),
    write(S, 'import sys\n'),
    write(S, 'import time\n'),
    write(S, 'from pathlib import Path\n'),
    write(S, 'from urllib.request import urlopen, Request\n'),
    write(S, 'from urllib.error import URLError\n\n\n'),
    %% DEFAULT_PRICING dict — generated from model_pricing/3 facts
    write(S, '# Pricing per 1M tokens (auto-generated from Prolog facts)\n'),
    write(S, 'DEFAULT_PRICING = {\n'),
    forall(model_pricing(Model, InPrice, OutPrice), (
        format(S, '    "~w": {"input": ~w, "output": ~w},~n', [Model, InPrice, OutPrice])
    )),
    write(S, '}\n\n\n'),
    %% UsageRecord dataclass
    write(S, '@dataclass\n'),
    write(S, 'class UsageRecord:\n'),
    write(S, '    """Record of a single API call."""\n'),
    write(S, '    timestamp: str\n'),
    write(S, '    model: str\n'),
    write(S, '    input_tokens: int\n'),
    write(S, '    output_tokens: int\n'),
    write(S, '    input_cost: float\n'),
    write(S, '    output_cost: float\n'),
    write(S, '    total_cost: float\n\n\n'),
    %% CostTracker class — hybrid (structure from facts, methods embedded)
    write(S, '@dataclass\n'),
    write(S, 'class CostTracker:\n'),
    write(S, '    """Track API costs for a session."""\n\n'),
    write(S, '    pricing: dict = field(default_factory=lambda: DEFAULT_PRICING.copy())\n'),
    write(S, '    records: list[UsageRecord] = field(default_factory=list)\n'),
    write(S, '    total_input_tokens: int = 0\n'),
    write(S, '    total_output_tokens: int = 0\n'),
    write(S, '    total_cost: float = 0.0\n\n'),
    write(S, '    def record_usage(self, model: str, input_tokens: int, output_tokens: int) -> UsageRecord:\n'),
    write(S, '        """Record token usage and calculate cost."""\n'),
    write(S, '        pricing = self.pricing.get(model, {"input": 0.0, "output": 0.0})\n'),
    write(S, '        input_cost = (input_tokens / 1_000_000) * pricing["input"]\n'),
    write(S, '        output_cost = (output_tokens / 1_000_000) * pricing["output"]\n'),
    write(S, '        total_cost = input_cost + output_cost\n'),
    write(S, '        record = UsageRecord(\n'),
    write(S, '            timestamp=datetime.now().isoformat(),\n'),
    write(S, '            model=model,\n'),
    write(S, '            input_tokens=input_tokens,\n'),
    write(S, '            output_tokens=output_tokens,\n'),
    write(S, '            input_cost=input_cost,\n'),
    write(S, '            output_cost=output_cost,\n'),
    write(S, '            total_cost=total_cost\n'),
    write(S, '        )\n'),
    write(S, '        self.records.append(record)\n'),
    write(S, '        self.total_input_tokens += input_tokens\n'),
    write(S, '        self.total_output_tokens += output_tokens\n'),
    write(S, '        self.total_cost += total_cost\n'),
    write(S, '        return record\n\n'),
    write(S, '    def get_summary(self) -> dict:\n'),
    write(S, '        """Get a summary of costs."""\n'),
    write(S, '        return {\n'),
    write(S, '            "total_requests": len(self.records),\n'),
    write(S, '            "total_input_tokens": self.total_input_tokens,\n'),
    write(S, '            "total_output_tokens": self.total_output_tokens,\n'),
    write(S, '            "total_tokens": self.total_input_tokens + self.total_output_tokens,\n'),
    write(S, '            "total_cost_usd": round(self.total_cost, 6),\n'),
    write(S, '            "cost_formatted": f"${self.total_cost:.4f}"\n'),
    write(S, '        }\n\n'),
    write(S, '    def format_status(self) -> str:\n'),
    write(S, '        """Format cost status for display."""\n'),
    write(S, '        summary = self.get_summary()\n'),
    write(S, '        return (\n'),
    write(S, '            f"Tokens: {summary[\'total_input_tokens\']:,} in / "\n'),
    write(S, '            f"{summary[\'total_output_tokens\']:,} out | "\n'),
    write(S, '            f"Cost: {summary[\'cost_formatted\']}"\n'),
    write(S, '        )\n\n'),
    write(S, '    def reset(self) -> None:\n'),
    write(S, '        """Reset all tracking."""\n'),
    write(S, '        self.records.clear()\n'),
    write(S, '        self.total_input_tokens = 0\n'),
    write(S, '        self.total_output_tokens = 0\n'),
    write(S, '        self.total_cost = 0.0\n\n'),
    write(S, '    def to_dict(self) -> dict:\n'),
    write(S, '        """Convert to dictionary for serialization."""\n'),
    write(S, '        return {\n'),
    write(S, '            "summary": self.get_summary(),\n'),
    write(S, '            "records": [\n'),
    write(S, '                {\n'),
    write(S, '                    "timestamp": r.timestamp,\n'),
    write(S, '                    "model": r.model,\n'),
    write(S, '                    "input_tokens": r.input_tokens,\n'),
    write(S, '                    "output_tokens": r.output_tokens,\n'),
    write(S, '                    "input_cost": r.input_cost,\n'),
    write(S, '                    "output_cost": r.output_cost,\n'),
    write(S, '                    "total_cost": r.total_cost\n'),
    write(S, '                }\n'),
    write(S, '                for r in self.records\n'),
    write(S, '            ]\n'),
    write(S, '        }\n\n'),
    write(S, '    def save(self, path: str | Path) -> None:\n'),
    write(S, '        """Save cost data to JSON file."""\n'),
    write(S, '        path = Path(path)\n'),
    write(S, '        path.write_text(json.dumps(self.to_dict(), indent=2))\n\n'),
    write(S, '    @classmethod\n'),
    write(S, '    def load(cls, path: str | Path) -> "CostTracker":\n'),
    write(S, '        """Load cost data from JSON file."""\n'),
    write(S, '        path = Path(path)\n'),
    write(S, '        data = json.loads(path.read_text())\n'),
    write(S, '        tracker = cls()\n'),
    write(S, '        for r in data.get("records", []):\n'),
    write(S, '            record = UsageRecord(\n'),
    write(S, '                timestamp=r["timestamp"],\n'),
    write(S, '                model=r["model"],\n'),
    write(S, '                input_tokens=r["input_tokens"],\n'),
    write(S, '                output_tokens=r["output_tokens"],\n'),
    write(S, '                input_cost=r["input_cost"],\n'),
    write(S, '                output_cost=r["output_cost"],\n'),
    write(S, '                total_cost=r["total_cost"]\n'),
    write(S, '            )\n'),
    write(S, '            tracker.records.append(record)\n'),
    write(S, '            tracker.total_input_tokens += record.input_tokens\n'),
    write(S, '            tracker.total_output_tokens += record.output_tokens\n'),
    write(S, '            tracker.total_cost += record.total_cost\n'),
    write(S, '        return tracker\n\n'),
    write(S, '    def set_pricing(self, model: str, input_price: float, output_price: float) -> None:\n'),
    write(S, '        """Set custom pricing for a model (per 1M tokens)."""\n'),
    write(S, '        self.pricing[model] = {"input": input_price, "output": output_price}\n\n'),
    write(S, '    def ensure_pricing(self, model: str) -> bool:\n'),
    write(S, '        """Ensure pricing exists for a model. Fetch from OpenRouter if needed."""\n'),
    write(S, '        if model in self.pricing:\n'),
    write(S, '            return True\n'),
    write(S, '        pricing = fetch_openrouter_pricing(model)\n'),
    write(S, '        if pricing:\n'),
    write(S, '            self.pricing[model] = pricing\n'),
    write(S, '            return True\n'),
    write(S, '        return False\n\n\n'),
    %% OpenRouter pricing helper
    write(S, '# --- OpenRouter pricing ---\n\n'),
    write(S, '_OPENROUTER_CACHE_DIR = Path(os.environ.get(\n'),
    write(S, '    \'AGENT_LOOP_CACHE\', os.path.expanduser(\'~/.agent-loop/cache\')\n'),
    write(S, '))\n'),
    write(S, '_OPENROUTER_CACHE_FILE = _OPENROUTER_CACHE_DIR / \'openrouter_pricing.json\'\n'),
    write(S, '_OPENROUTER_CACHE_TTL = 86400  # 1 day\n\n\n'),
    write(S, 'def _load_openrouter_cache() -> dict | None:\n'),
    write(S, '    """Load cached OpenRouter pricing if fresh enough."""\n'),
    write(S, '    try:\n'),
    write(S, '        if not _OPENROUTER_CACHE_FILE.exists():\n'),
    write(S, '            return None\n'),
    write(S, '        age = time.time() - _OPENROUTER_CACHE_FILE.stat().st_mtime\n'),
    write(S, '        if age > _OPENROUTER_CACHE_TTL:\n'),
    write(S, '            return None\n'),
    write(S, '        return json.loads(_OPENROUTER_CACHE_FILE.read_text())\n'),
    write(S, '    except Exception:\n'),
    write(S, '        return None\n\n\n'),
    write(S, 'def _save_openrouter_cache(pricing: dict) -> None:\n'),
    write(S, '    """Save OpenRouter pricing to cache."""\n'),
    write(S, '    try:\n'),
    write(S, '        _OPENROUTER_CACHE_DIR.mkdir(parents=True, exist_ok=True)\n'),
    write(S, '        _OPENROUTER_CACHE_FILE.write_text(json.dumps(pricing))\n'),
    write(S, '    except Exception:\n'),
    write(S, '        pass\n\n\n'),
    write(S, 'def fetch_openrouter_pricing(model_id: str) -> dict | None:\n'),
    write(S, '    """Fetch pricing for a model from OpenRouter\'s API."""\n'),
    write(S, '    cache = _load_openrouter_cache()\n'),
    write(S, '    if cache and model_id in cache:\n'),
    write(S, '        return cache[model_id]\n'),
    write(S, '    try:\n'),
    write(S, '        req = Request(\n'),
    write(S, '            \'https://openrouter.ai/api/v1/models\',\n'),
    write(S, '            headers={\'Content-Type\': \'application/json\'}\n'),
    write(S, '        )\n'),
    write(S, '        with urlopen(req, timeout=10) as resp:\n'),
    write(S, '            data = json.loads(resp.read().decode())\n'),
    write(S, '    except (URLError, json.JSONDecodeError, OSError) as e:\n'),
    write(S, '        print(f"  [OpenRouter pricing fetch failed: {e}]", file=sys.stderr)\n'),
    write(S, '        return None\n'),
    write(S, '    pricing_cache = {}\n'),
    write(S, '    for m in data.get(\'data\', []):\n'),
    write(S, '        mid = m.get(\'id\', \'\')\n'),
    write(S, '        p = m.get(\'pricing\', {})\n'),
    write(S, '        prompt_per_token = float(p.get(\'prompt\', \'0\') or \'0\')\n'),
    write(S, '        completion_per_token = float(p.get(\'completion\', \'0\') or \'0\')\n'),
    write(S, '        pricing_cache[mid] = {\n'),
    write(S, '            \'input\': round(prompt_per_token * 1_000_000, 4),\n'),
    write(S, '            \'output\': round(completion_per_token * 1_000_000, 4),\n'),
    write(S, '        }\n'),
    write(S, '    _save_openrouter_cache(pricing_cache)\n'),
    write(S, '    return pricing_cache.get(model_id)\n'),
    close(S),
    format('  Generated costs.py~n', []).

%% =============================================================================
%% Generator: tools_generated.py (unchanged from before)
%% =============================================================================

generate_tools :-
    open('generated/tools_generated.py', write, S),
    write(S, '"""Auto-generated tool definitions from Prolog specs."""\n\n'),
    write(S, 'TOOL_SPECS = {\n'),
    findall(Name, tool_spec(Name, _), Tools),
    generate_tool_specs(S, Tools),
    write(S, '}\n\n'),
    write(S, 'DESTRUCTIVE_TOOLS = {\n'),
    generate_destructive_list(S, Tools),
    write(S, '}\n'),
    close(S),
    format('  Generated tools_generated.py~n', []).

generate_tool_specs(_, []).
generate_tool_specs(S, [Name|Rest]) :-
    tool_spec(Name, Props),
    member(description(Desc), Props),
    member(parameters(Params), Props),
    format(S, '    "~w": {~n', [Name]),
    format(S, '        "description": "~w",~n', [Desc]),
    format(S, '        "parameters": [~n', []),
    generate_params(S, Params),
    format(S, '        ]~n', []),
    format(S, '    },~n', []),
    generate_tool_specs(S, Rest).

generate_params(_, []).
generate_params(S, [param(Name, Type, Required, Desc)|Rest]) :-
    (Required = required -> PyReq = 'True' ; PyReq = 'False'),
    format(S, '            {"name": "~w", "type": "~w", "required": ~w, "description": "~w"},~n',
           [Name, Type, PyReq, Desc]),
    generate_params(S, Rest).

generate_destructive_list(_, []).
generate_destructive_list(S, [Name|Rest]) :-
    tool_spec(Name, Props),
    (member(confirmation_required(true), Props) ->
        format(S, '    "~w",~n', [Name])
    ;
        true
    ),
    generate_destructive_list(S, Rest).

%% =============================================================================
%% Generator: README.md
%% =============================================================================

generate_readme :-
    open('generated/README.md', write, S),
    write(S, '# UnifyWeaver Agent Loop - Generated Code\n\n'),
    write(S, 'This code was generated by `agent_loop_module.pl`.\n\n'),
    write(S, '## Backends\n\n'),
    forall(agent_backend(Name, Props), (
        member(description(Desc), Props),
        format(S, '- **~w**: ~w~n', [Name, Desc])
    )),
    write(S, '\n## Tools\n\n'),
    forall(tool_spec(Name, Props), (
        member(description(Desc), Props),
        format(S, '- **~w**: ~w~n', [Name, Desc])
    )),
    write(S, '\n## Usage\n\n'),
    write(S, '```bash\n'),
    write(S, 'python3 agent_loop.py              # interactive\n'),
    write(S, 'python3 agent_loop.py "prompt"     # single prompt\n'),
    write(S, 'python3 agent_loop.py -b claude    # use Claude API\n'),
    write(S, '```\n'),
    close(S),
    format('  Generated README.md~n', []).

%% =============================================================================
%% Generator: Individual backend stubs
%% =============================================================================

generate_backend(Name) :-
    agent_backend(Name, Props),
    resolve_file_name(Name, FileName),
    atom_concat('generated/backends/', FileName, Path1),
    atom_concat(Path1, '.py', Path),
    open(Path, write, S),
    generate_backend_code(S, Name, Props),
    close(S),
    format('  Generated backends/~w.py~n', [FileName]).

generate_backend_code(S, Name, Props) :-
    resolve_class_name(Name, ClassName),
    member(type(Type), Props),
    member(description(Desc), Props),
    format(S, '"""~w"""~n~n', [Desc]),
    (Type = cli ->
        generate_cli_backend(S, ClassName, Props)
    ;
        generate_api_backend(S, ClassName, Props)
    ).

generate_cli_backend(S, ClassName, Props) :-
    member(command(Cmd), Props),
    format(S, 'import subprocess~n', []),
    format(S, 'from .base import AgentBackend, AgentResponse~n~n', []),
    format(S, 'class ~w(AgentBackend):~n', [ClassName]),
    format(S, '    """CLI backend using ~w command."""~n~n', [Cmd]),
    format(S, '    def __init__(self, command: str = "~w"):~n', [Cmd]),
    format(S, '        self.command = command~n~n', []),
    format(S, '    def send_message(self, message: str, context: list[dict]) -> AgentResponse:~n', []),
    format(S, '        raise NotImplementedError("See prototype for full implementation")~n', []).

generate_api_backend(S, ClassName, Props) :-
    member(endpoint(Endpoint), Props),
    member(model(Model), Props),
    format(S, 'import os~n', []),
    format(S, 'import requests~n', []),
    format(S, 'from .base import AgentBackend, AgentResponse~n~n', []),
    format(S, 'class ~w(AgentBackend):~n', [ClassName]),
    format(S, '    """API backend for ~w"""~n~n', [Endpoint]),
    format(S, '    ENDPOINT = "~w"~n', [Endpoint]),
    format(S, '    DEFAULT_MODEL = "~w"~n~n', [Model]),
    format(S, '    def __init__(self, api_key: str = None, model: str = None):~n', []),
    (member(auth_env(EnvVar), Props) ->
        format(S, '        self.api_key = api_key or os.environ.get("~w")~n', [EnvVar])
    ;
        format(S, '        self.api_key = api_key~n', [])
    ),
    format(S, '        self.model = model or self.DEFAULT_MODEL~n~n', []),
    format(S, '    def send_message(self, message: str, context: list[dict]) -> AgentResponse:~n', []),
    format(S, '        raise NotImplementedError("See prototype for full implementation")~n', []).

%% =============================================================================
%% Helper Predicates
%% =============================================================================

%% Resolve class name: use class_name/1 override if present, else auto-generate
resolve_class_name(Name, ClassName) :-
    agent_backend(Name, Props),
    (member(class_name(ClassName), Props) ->
        true
    ;
        backend_class_name(Name, ClassName)
    ).

%% Resolve file name: use file_name/1 override if present, else auto-generate
resolve_file_name(Name, FileName) :-
    agent_backend(Name, Props),
    (member(file_name(FN), Props) ->
        atom_string(FN, FNStr),
        atom_string(FileName, FNStr)
    ;
        backend_file_name(Name, FileName)
    ).

%% Auto-generate CamelCase class name from snake_case
backend_class_name(Name, ClassName) :-
    atom_codes(Name, Codes),
    to_camel_case(Codes, CamelCodes),
    atom_codes('Backend', BackendCodes),
    append(CamelCodes, BackendCodes, ClassCodes),
    atom_codes(ClassName, ClassCodes).

%% Auto-generate snake_case file name
backend_file_name(Name, FileName) :-
    atom_codes(Name, Codes),
    to_snake_case(Codes, SnakeCodes),
    atom_codes(FileName, SnakeCodes).

to_camel_case([], []).
to_camel_case([C|Rest], [Upper|CamelRest]) :-
    C >= 0'a, C =< 0'z,
    Upper is C - 32,
    to_camel_case_rest(Rest, CamelRest).
to_camel_case([C|Rest], [C|CamelRest]) :-
    to_camel_case_rest(Rest, CamelRest).

to_camel_case_rest([], []).
to_camel_case_rest([0'_|Rest], CamelRest) :-
    to_camel_case(Rest, CamelRest).
to_camel_case_rest([C|Rest], [C|CamelRest]) :-
    C \= 0'_,
    to_camel_case_rest(Rest, CamelRest).

to_snake_case([], []).
to_snake_case([C|Rest], [Lower|SnakeRest]) :-
    C >= 0'A, C =< 0'Z,
    Lower is C + 32,
    to_snake_case(Rest, SnakeRest).
to_snake_case([C|Rest], [C|SnakeRest]) :-
    to_snake_case(Rest, SnakeRest).

%% Convert Prolog true/false to Python True/False
py_bool(true, 'True').
py_bool(false, 'False').

%% Write a list of quoted strings for __all__
write_quoted_list(_, []).
write_quoted_list(S, [Item]) :-
    format(S, '    \'~w\',~n', [Item]).
write_quoted_list(S, [Item|Rest]) :-
    Rest \= [],
    format(S, '    \'~w\',~n', [Item]),
    write_quoted_list(S, Rest).

%% Query helpers for introspection
list_backends :-
    write('Available backends:'), nl,
    forall(agent_backend(Name, Props), (
        member(description(Desc), Props),
        format('  ~w: ~w~n', [Name, Desc])
    )).

list_tools :-
    write('Available tools:'), nl,
    forall(tool_spec(Name, Props), (
        member(description(Desc), Props),
        format('  ~w: ~w~n', [Name, Desc])
    )).
