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

:- discontiguous generate_backend_full/3.

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
%% Context Module Data
%% =============================================================================

%% context_enum(EnumName, DocString, Values)
%% Values = list of value(PythonName, StringVal, Comment)
context_enum('ContextBehavior', "How to handle context across messages.", [
    value('CONTINUE', "continue", "Full context, continue conversation"),
    value('FRESH',    "fresh",    "No context, each query independent"),
    value('SLIDING',  "sliding",  "Keep only last N messages"),
    value('SUMMARIZE',"summarize","Summarize old context (future)")
]).

context_enum('ContextFormat', "How to format context for the backend.", [
    value('PLAIN',    "plain",    "Simple text with role prefixes"),
    value('MARKDOWN', "markdown", "Markdown formatted"),
    value('JSON',     "json",     "JSON array of messages"),
    value('XML',      "xml",      "XML structure")
]).

%% message_field(Name, Type, Default)
message_field(role,         'Literal["user", "assistant", "system", "tool"]', none).
message_field(content,      'str',   none).
message_field(tokens,       'int',   '0').
message_field(tool_call_id, 'str',   '""').
message_field(tool_calls,   'list',  'field(default_factory=list)').

%% =============================================================================
%% Config Module Data
%% =============================================================================

%% api_key_env_var(BackendType, EnvVarName)
api_key_env_var(openrouter, 'OPENROUTER_API_KEY').
api_key_env_var(claude,     'ANTHROPIC_API_KEY').
api_key_env_var(openai,     'OPENAI_API_KEY').
api_key_env_var(gemini,     'GEMINI_API_KEY').

%% api_key_file(BackendType, FilePath)
api_key_file(claude, '~/.anthropic/api_key').
api_key_file(openai, '~/.openai/api_key').

%% agent_config_field(Name, Type, Default, Comment)
agent_config_field(name,                'str',            none,      "").
agent_config_field(backend,             'str',            none,      "coro, claude-code, gemini, claude, ollama-api, ollama-cli").
agent_config_field(model,               'str | None',     'None',    "").
agent_config_field(host,                'str | None',     'None',    "For network backends (ollama-api)").
agent_config_field(port,                'int | None',     'None',    "").
agent_config_field(api_key,             'str | None',     'None',    "Or env var name like $ANTHROPIC_API_KEY").
agent_config_field(command,             'str | None',     'None',    "For CLI backends").
agent_config_field(system_prompt,       'str | None',     'None',    "").
agent_config_field(agent_md,            'str | None',     'None',    "Path to agent.md file").
agent_config_field(tools,               'list[str]',      'field(default_factory=lambda: [\'bash\', \'read\', \'write\', \'edit\'])', "").
agent_config_field(auto_tools,          'bool',           'False',   "Skip confirmation").
agent_config_field(context_mode,        'str',            '"continue"', "continue, fresh, sliding").
agent_config_field(max_context_tokens,  'int',            '100000',  "").
agent_config_field(max_messages,        'int',            '50',      "").
agent_config_field(skills,              'list[str]',      'field(default_factory=list)', "Paths to skill files").
agent_config_field(max_iterations,      'int',            '0',       "0 = unlimited, N = pause after N tool iterations").
agent_config_field(timeout,             'int',            '300',     "").
agent_config_field(show_tokens,         'bool',           'True',    "").
agent_config_field(extra,               'dict',           'field(default_factory=dict)', "").

%% default_agent_preset(Name, BackendAtom, Properties)
default_agent_preset(default, coro, [command='claude']).
default_agent_preset('claude-sonnet', 'claude-code', [model=sonnet]).
default_agent_preset('claude-opus', 'claude-code', [model=opus]).
default_agent_preset('claude-haiku', 'claude-code', [model=haiku]).
default_agent_preset(yolo, 'claude-code', [
    model=haiku, auto_tools=true,
    system_prompt="Be concise and take action. Execute tools without asking."
]).
default_agent_preset(gemini, gemini, [model='gemini-3-flash-preview']).
default_agent_preset(ollama, 'ollama-api', [model=llama3, host=localhost, port=11434]).
default_agent_preset('ollama-cli', 'ollama-cli', [model=llama3]).
default_agent_preset(openai, openai, [model='gpt-4o']).
default_agent_preset('gpt-4o-mini', openai, [model='gpt-4o-mini']).

%% =============================================================================
%% Tools Module Data
%% =============================================================================

%% blocked_path(Path)
blocked_path('/etc/shadow').
blocked_path('/etc/gshadow').
blocked_path('/etc/sudoers').

%% blocked_path_prefix(Prefix)
blocked_path_prefix('/proc/').
blocked_path_prefix('/sys/').
blocked_path_prefix('/dev/').

%% blocked_home_pattern(Pattern)
blocked_home_pattern('.ssh/').
blocked_home_pattern('.gnupg/').
blocked_home_pattern('.aws/').
blocked_home_pattern('.config/gcloud/').
blocked_home_pattern('.env').
blocked_home_pattern('.netrc').
blocked_home_pattern('.npmrc').

%% blocked_command_pattern(Regex, Description)
blocked_command_pattern("\\brm\\s+-[rf]*\\s+/", "rm with absolute path and force/recursive flags").
blocked_command_pattern("\\bmkfs\\b", "filesystem format").
blocked_command_pattern("\\bdd\\s+.*of=/dev/", "write to block device").
blocked_command_pattern(">\\s*/dev/sd", "redirect to block device").
blocked_command_pattern("\\bcurl\\b.*\\|\\s*(?:ba)?sh", "pipe remote script to shell").
blocked_command_pattern("\\bwget\\b.*\\|\\s*(?:ba)?sh", "pipe remote script to shell").
blocked_command_pattern("\\bchmod\\s+777\\b", "world-writable permissions").
blocked_command_pattern(":\\(\\)\\s*\\{\\s*:\\|:\\s*&\\s*\\}\\s*;", "fork bomb").
blocked_command_pattern("\\b>\\s*/etc/", "overwrite system config").

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
    generate_module(context),
    generate_module(config),
    generate_module(display),
    generate_module(tools),
    generate_module(readme),
    write('Done.'), nl.

%% Generate a single module by name
generate_module(backends_init)      :- generate_backends_init.
generate_module(backends_base)      :- generate_backends_base.
generate_module(security_init)      :- generate_security_init.
generate_module(security_profiles)  :- generate_security_profiles.
generate_module(costs)              :- generate_costs.
generate_module(tools_generated)    :- generate_tools.
generate_module(context)            :- generate_context.
generate_module(config)             :- generate_config.
generate_module(display)            :- generate_display.
generate_module(tools)              :- generate_tools_module.
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
%% Generator: context.py
%% =============================================================================

generate_context :-
    open('generated/context.py', write, S),
    write(S, '"""Context manager for conversation history."""\n\n'),
    write(S, 'from dataclasses import dataclass, field\n'),
    write(S, 'from typing import Literal\n'),
    write(S, 'from enum import Enum\n\n\n'),
    %% Generate enums from context_enum/3 facts
    forall(context_enum(EnumName, DocStr, Values), (
        format(S, 'class ~w(Enum):~n', [EnumName]),
        format(S, '    """~w"""~n', [DocStr]),
        forall(member(value(PyName, StrVal, Comment), Values), (
            format(S, '    ~w = "~w"    # ~w~n',
                   [PyName, StrVal, Comment])
        )),
        write(S, '\n\n')
    )),
    %% Generate Message dataclass from message_field/3 facts
    write(S, '@dataclass\nclass Message:\n    """A single message in the conversation."""\n'),
    forall(message_field(Name, Type, Default), (
        (Default = none ->
            format(S, '    ~w: ~w~n', [Name, Type])
        ;
            format(S, '    ~w: ~w = ~w~n', [Name, Type, Default])
        )
    )),
    write(S, '\n\n'),
    %% Helper functions (imperative — embedded)
    write_py(S, context_helpers),
    write(S, '\n\n'),
    %% ContextManager class (imperative — embedded)
    write_py(S, context_manager_class),
    write(S, '\n'),
    close(S),
    format('  Generated context.py~n', []).

%% =============================================================================
%% Generator: config.py
%% =============================================================================

generate_config :-
    open('generated/config.py', write, S),
    write(S, '"""Configuration system for agent loop variants."""\n\n'),
    write(S, 'import os\nimport json\nfrom pathlib import Path\n'),
    write(S, 'from dataclasses import dataclass, field\nfrom typing import Any\n\n\n'),
    %% Config cascade function (imperative)
    write_py(S, config_read_cascade),
    write(S, '\n\n'),
    %% resolve_api_key with data-driven env vars and file paths
    write_py(S, config_resolve_api_key_header),
    %% Generate env_vars dict from api_key_env_var/2 facts
    write(S, '    env_vars = {\n'),
    forall(api_key_env_var(Backend, Var), (
        format(S, '        \'~w\': \'~w\',~n', [Backend, Var])
    )),
    write(S, '    }\n'),
    write_py(S, config_resolve_api_key_middle),
    %% Generate file_locations dict from api_key_file/2 facts
    write(S, '    file_locations = {\n'),
    forall(api_key_file(Backend, Path), (
        format(S, '        \'~w\': \'~w\',~n', [Backend, Path])
    )),
    write(S, '    }\n'),
    write_py(S, config_resolve_api_key_footer),
    write(S, '\n'),
    %% YAML import guard
    write(S, '# Try to import yaml, fall back to JSON only\n'),
    write(S, 'try:\n    import yaml\n    HAS_YAML = True\n'),
    write(S, 'except ImportError:\n    HAS_YAML = False\n\n\n'),
    %% Generate AgentConfig dataclass from agent_config_field/4 facts
    write(S, '@dataclass\nclass AgentConfig:\n    """Configuration for an agent variant."""\n'),
    forall(agent_config_field(Name, Type, Default, Comment), (
        (Comment = "" ->
            CommentStr = ""
        ;
            format(atom(CommentStr), '  # ~w', [Comment])
        ),
        (Default = none ->
            format(S, '    ~w: ~w~w~n', [Name, Type, CommentStr])
        ;
            format(S, '    ~w: ~w = ~w~w~n', [Name, Type, Default, CommentStr])
        )
    )),
    write(S, '\n\n'),
    %% Config dataclass (simple, inline)
    write(S, '@dataclass\nclass Config:\n    """Root configuration with multiple agent variants."""\n'),
    write(S, '    default: str = "default"\n'),
    write(S, '    agents: dict[str, AgentConfig] = field(default_factory=dict)\n\n'),
    write(S, '    # Global settings\n'),
    write(S, '    config_dir: str = ""\n'),
    write(S, '    skills_dir: str = ""\n\n\n'),
    %% _resolve_env_var, _load_agent_config, load_config, etc. (imperative)
    write_py(S, config_imperative_functions),
    write(S, '\n\n'),
    %% get_default_config — data-driven from default_agent_preset/3 facts
    write(S, 'def get_default_config() -> Config:\n'),
    write(S, '    """Get a default configuration with common presets."""\n'),
    write(S, '    config = Config()\n\n'),
    forall(default_agent_preset(Name, Backend, Props), (
        format(S, '    config.agents[\'~w\'] = AgentConfig(~n', [Name]),
        format(S, '        name=\'~w\',~n', [Name]),
        format(S, '        backend=\'~w\',~n', [Backend]),
        forall(member(Key=Val, Props), (
            (Val = true -> format(S, '        ~w=True,~n', [Key])
            ; Val = false -> format(S, '        ~w=False,~n', [Key])
            ; number(Val) -> format(S, '        ~w=~w,~n', [Key, Val])
            ; format(S, '        ~w=\'~w\',~n', [Key, Val])
            )
        )),
        write(S, '    )\n\n')
    )),
    write(S, '    return config\n\n\n'),
    %% save_example_config (imperative)
    write_py(S, config_save_example),
    write(S, '\n'),
    close(S),
    format('  Generated config.py~n', []).

%% =============================================================================
%% Generator: display.py
%% =============================================================================

generate_display :-
    open('generated/display.py', write, S),
    write_py(S, display_module),
    close(S),
    format('  Generated display.py~n', []).

%% =============================================================================
%% Generator: tools.py (main tools module, not tools_generated.py)
%% =============================================================================

generate_tools_module :-
    open('generated/tools.py', write, S),
    write(S, '"""Tool handler for executing agent tool calls."""\n\n'),
    write(S, 'import os\nimport re\nimport subprocess\n'),
    write(S, 'from dataclasses import dataclass, field\nfrom pathlib import Path\n'),
    write(S, 'from backends.base import ToolCall\n'),
    write(S, 'from security.audit import AuditLogger\n'),
    write(S, 'from security.profiles import SecurityProfile, get_profile\n'),
    write(S, 'from security.proxy import CommandProxyManager\n\n\n'),
    %% ToolResult dataclass
    write(S, '@dataclass\nclass ToolResult:\n    """Result of executing a tool."""\n'),
    write(S, '    success: bool\n    output: str\n    tool_name: str\n\n\n'),
    %% Path validation — data-driven blocked lists
    write(S, '# ── Path validation ────────────────────────────────────────────────────────\n\n'),
    write(S, '# Sensitive paths that should never be accessed by agent tools\n'),
    write(S, '_BLOCKED_PATHS = {\n'),
    forall(blocked_path(P), format(S, '    \'~w\',~n', [P])),
    write(S, '}\n'),
    write(S, '_BLOCKED_PREFIXES = (\n'),
    forall(blocked_path_prefix(P), format(S, '    \'~w\',~n', [P])),
    write(S, ')\n'),
    write(S, '_BLOCKED_HOME_PATTERNS = (\n'),
    forall(blocked_home_pattern(P), format(S, '    \'~w\',~n', [P])),
    write(S, ')\n\n\n'),
    %% validate_path function (imperative)
    write_py(S, tools_validate_path),
    write(S, '\n\n'),
    %% Command blocklist — data-driven
    write(S, '# ── Command blocklist ──────────────────────────────────────────────────────\n\n'),
    write(S, '_BLOCKED_COMMAND_PATTERNS = [\n'),
    forall(blocked_command_pattern(Regex, Desc), (
        format(S, '    (r\'~w\', "~w"),~n', [Regex, Desc])
    )),
    write(S, ']\n\n\n'),
    %% is_command_blocked (imperative)
    write_py(S, tools_is_command_blocked),
    write(S, '\n\n'),
    %% SecurityConfig (imperative — has from_profile method)
    write_py(S, tools_security_config),
    write(S, '\n\n'),
    %% ToolHandler class (imperative)
    write_py(S, tools_handler_class),
    write(S, '\n'),
    close(S),
    format('  Generated tools.py~n', []).

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
%% Python Code Fragment System
%% =============================================================================
%%
%% py_fragment(Name, Code) — named Python code fragments that can be composed.
%% Fragments are multiline strings written verbatim to output streams.
%%
%% write_py(S, Name) — write a named fragment to stream S.
%% write_py(S, Name, Subs) — write fragment with {{Key}} substitutions.
%%
%% This decomposes the 2000+ lines of verbatim Python in backend generators
%% into ~30 reusable, composable units.

:- discontiguous py_fragment/2.

%% Write a fragment to a stream
write_py(S, Name) :-
    (py_fragment(Name, Code) ->
        write(S, Code)
    ;
        format(atom(Msg), 'Unknown py_fragment: ~w', [Name]),
        throw(error(existence_error(py_fragment, Name), context(write_py/2, Msg)))
    ).

%% Write a fragment with {{Key}} -> Value substitutions
write_py(S, Name, Subs) :-
    py_fragment(Name, Code),
    apply_subs(Code, Subs, Result),
    write(S, Result).

%% Apply substitutions: list of Key=Value pairs
apply_subs(Code, [], Code).
apply_subs(Code, [Key=Value|Rest], Result) :-
    format(atom(Placeholder), '{{~w}}', [Key]),
    atom_string(Code, CodeStr),
    atom_string(Placeholder, PlaceholderStr),
    (atom(Value) -> atom_string(Value, ValueStr) ; term_string(Value, ValueStr)),
    replace_all_sub(CodeStr, PlaceholderStr, ValueStr, Mid),
    atom_string(MidAtom, Mid),
    apply_subs(MidAtom, Rest, Result).

%% Replace all occurrences of Sub in String with Rep
replace_all_sub(String, Sub, Rep, Result) :-
    string_length(Sub, SubLen),
    (sub_string(String, Before, SubLen, _, Sub) ->
        sub_string(String, 0, Before, _, Prefix),
        Start is Before + SubLen,
        sub_string(String, Start, _, 0, Suffix),
        replace_all_sub(Suffix, Sub, Rep, RestResult),
        string_concat(Prefix, Rep, Part1),
        string_concat(Part1, RestResult, Result)
    ;
        Result = String
    ).

%% --- Fragment: _format_prompt (shared by 5 CLI backends) ---

py_fragment(format_prompt, '    def _format_prompt(self, message: str, context: list[dict]) -> str:
        """Format message with conversation context."""
        if not context:
            return message

        history_lines = []
        for msg in context[-6:]:
            role = "User" if msg.get(\'role\') == \'user\' else "Assistant"
            content = msg.get(\'content\', \'\')
            if len(content) > 500:
                content = content[:500] + "..."
            history_lines.append(f"{role}: {content}")

        history = "\\n".join(history_lines)

        return f"""Previous conversation:
{history}

Current request: {message}"""
').

%% --- Fragment: Error handlers ---

py_fragment(error_handler_cli, '        except subprocess.TimeoutExpired:
            {{on_timeout}}return AgentResponse(
                content={{timeout_msg}},
                tokens={}
            )
        except FileNotFoundError:
            return AgentResponse(
                content=f"[Error: Command \'{self.command}\' not found. {{install_hint}}]",
                tokens={}
            )
        except Exception as e:
            return AgentResponse(
                content=f"[Error: {e}]",
                tokens={}
            )
').

py_fragment(error_handler_api_sdk, '        except {{sdk}}.APIError as e:
            return AgentResponse(
                content=f"[API Error: {e}]",
                tokens={}
            )
        except Exception as e:
            return AgentResponse(
                content=f"[Error: {e}]",
                tokens={}
            )
').

py_fragment(error_handler_urllib, '        except HTTPError as e:
            error_body = e.read().decode(\'utf-8\', errors=\'replace\')
            try:
                err_data = json.loads(error_body)
                err_msg = err_data.get(\'error\', {}).get(\'message\', error_body[:200])
            except json.JSONDecodeError:
                err_msg = error_body[:200]
            return AgentResponse(
                content=f"[API Error {e.code}: {err_msg}]",
                tokens={}
            )
        except URLError as e:
            return AgentResponse(
                content=f"[Network Error: {e.reason}]",
                tokens={}
            )
        except Exception as e:
            return AgentResponse(
                content=f"[Error: {e}]",
                tokens={}
            )
').

%% --- Fragment: SDK import guard ---

py_fragment(sdk_import_guard, '# Try to import {{sdk}}, but don\'t fail if not installed
try:
    import {{sdk}}
    HAS_{{SDK_UPPER}} = True
except ImportError:
    HAS_{{SDK_UPPER}} = False

').

py_fragment(sdk_init_guard, '        if not HAS_{{SDK_UPPER}}:
            raise ImportError(
                "{{sdk}} package not installed. "
                "Install with: pip install {{sdk}}"
            )

').

%% --- Fragment: API key validation ---

py_fragment(api_key_from_env, '        # Get API key from parameter or environment
        self.api_key = api_key or os.environ.get(\'{{env_var}}\')
        if not self.api_key:
            raise ValueError(
                "API key required. Set {{env_var}} environment variable "
                "or pass api_key parameter."
            )
').

%% --- Fragment: supports_streaming ---

py_fragment(supports_streaming_true, '    def supports_streaming(self) -> bool:
        """{{backend_name}} supports streaming."""
        return True
').

%% --- Fragment: name property ---

py_fragment(name_property, '    @property
    def name(self) -> str:
        return f"{{display_name}}"
').

%% --- Fragment: Messages array builder (OpenAI-style) ---

py_fragment(messages_builder_system, '        # Build messages array
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Add context messages
        for msg in context:
            if msg.get(\'role\') in (\'user\', \'assistant\'):
                messages.append({
                    "role": msg[\'role\'],
                    "content": msg[\'content\']
                })
').

py_fragment(messages_builder_openrouter, '        # Build messages array
        messages = [{"role": "system", "content": self.system_prompt}]

        for msg in context:
            role = msg.get(\'role\')
            if role == \'tool\':
                messages.append({
                    "role": "tool",
                    "tool_call_id": msg.get(\'tool_call_id\', \'\'),
                    "content": msg[\'content\'],
                })
            elif role == \'assistant\' and msg.get(\'tool_calls\'):
                messages.append({
                    "role": "assistant",
                    "content": msg.get(\'content\') or \'\',
                    "tool_calls": msg[\'tool_calls\'],
                })
            elif role in (\'user\', \'assistant\'):
                messages.append({"role": role, "content": msg[\'content\']})

        # Add current message (only if non-empty and not already in context)
        if message and (not context or context[-1].get(\'content\') != message):
            messages.append({"role": "user", "content": message})
').

%% --- Fragment: Messages array builder (Anthropic-style, no system in messages) ---

py_fragment(messages_builder_anthropic, '        # Build messages array
        messages = []

        # Add context messages
        for msg in context:
            if msg.get(\'role\') in (\'user\', \'assistant\'):
                messages.append({
                    "role": msg[\'role\'],
                    "content": msg[\'content\']
                })

        # Note: current message is already in context (added by agent_loop)
').

%% --- Fragment: Ollama messages builder (with system prompt) ---

py_fragment(messages_builder_ollama, '        # Build messages array
        messages = []

        # Add system prompt
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        # Add context messages
        for msg in context:
            if msg.get(\'role\') in (\'user\', \'assistant\'):
                messages.append({
                    "role": msg[\'role\'],
                    "content": msg[\'content\']
                })

        # Add current message
        messages.append({
            "role": "user",
            "content": message
        })
').

%% --- Fragment: OpenAI-style tool call extraction ---

py_fragment(extract_tool_calls_openai, '    def _extract_tool_calls(self, response) -> list[ToolCall]:
        """Extract tool calls from response."""
        tool_calls = []

        if not response.choices or len(response.choices) == 0:
            return tool_calls

        choice = response.choices[0]
        if not choice.message or not choice.message.tool_calls:
            return tool_calls

        for tc in choice.message.tool_calls:
            if tc.type == "function":
                import json
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": tc.function.arguments}

                tool_calls.append(ToolCall(
                    name=tc.function.name,
                    arguments=arguments,
                    id=tc.id
                ))

        return tool_calls
').

%% --- Fragment: Anthropic-style tool call extraction ---

py_fragment(extract_tool_calls_anthropic, '    def _extract_tool_calls(self, response) -> list[ToolCall]:
        """Extract tool use blocks from response."""
        tool_calls = []

        for block in response.content:
            if hasattr(block, \'type\') and block.type == \'tool_use\':
                tool_calls.append(ToolCall(
                    name=block.name,
                    arguments=block.input,
                    id=block.id
                ))

        return tool_calls
').

%% --- Fragment: _describe_tool_call variants ---

py_fragment(describe_tool_call_gemini, '    def _describe_tool_call(self, tool_name: str, params: dict) -> str:
        """Create a short description of a tool call."""
        if tool_name == \'read_file\':
            return f"reading {os.path.basename(params.get(\'file_path\', \'?\'))}"
        elif tool_name == \'glob\':
            return f"searching {params.get(\'pattern\', \'?\')}"
        elif tool_name == \'grep\':
            return f"grep {params.get(\'pattern\', \'?\')}"
        elif tool_name == \'run_shell_command\':
            cmd = params.get(\'command\', \'?\')
            if len(cmd) > 72:
                cmd = cmd[:69] + \'...\'
            return f"$ {cmd}"
        elif tool_name == \'write_file\':
            return f"writing {os.path.basename(params.get(\'file_path\', \'?\'))}"
        elif tool_name == \'edit\':
            return f"editing {os.path.basename(params.get(\'file_path\', \'?\'))}"
        elif tool_name == \'list_directory\':
            return f"ls {params.get(\'path\', \'?\')}"
        else:
            return tool_name
').

py_fragment(describe_tool_call_claude_code, '    def _describe_tool_call(self, tool_name: str, params: dict) -> str:
        """Create a short description of a tool call."""
        if tool_name == \'Read\':
            return f"reading {os.path.basename(params.get(\'file_path\', \'?\'))}"
        elif tool_name == \'Glob\':
            return f"searching {params.get(\'pattern\', \'?\')}"
        elif tool_name == \'Grep\':
            return f"grep {params.get(\'pattern\', \'?\')}"
        elif tool_name == \'Bash\':
            cmd = params.get(\'command\', \'?\')
            if len(cmd) > 72:
                cmd = cmd[:69] + \'...\'
            return f"$ {cmd}"
        elif tool_name == \'Write\':
            return f"writing {os.path.basename(params.get(\'file_path\', \'?\'))}"
        elif tool_name == \'Edit\':
            return f"editing {os.path.basename(params.get(\'file_path\', \'?\'))}"
        elif tool_name == \'Task\':
            return f"agent: {params.get(\'description\', \'?\')}"
        elif tool_name == \'WebFetch\':
            return f"fetching {params.get(\'url\', \'?\')}"
        elif tool_name == \'WebSearch\':
            return f"searching: {params.get(\'query\', \'?\')}"
        else:
            return tool_name
').

py_fragment(describe_tool_call_openrouter, '    def _describe_tool_call(self, tool_name: str, params: dict) -> str:
        """Create a short description of a tool call."""
        if tool_name == \'read\':
            return f"reading {os.path.basename(params.get(\'path\', \'?\'))}"
        elif tool_name == \'write\':
            return f"writing {os.path.basename(params.get(\'path\', \'?\'))}"
        elif tool_name == \'edit\':
            return f"editing {os.path.basename(params.get(\'path\', \'?\'))}"
        elif tool_name == \'bash\':
            cmd = params.get(\'command\', \'?\')
            if len(cmd) > 72:
                cmd = cmd[:69] + \'...\'
            return f"$ {cmd}"
        return tool_name
').

%% --- Fragment: _clean_output (simple, for ollama_cli) ---

py_fragment(clean_output_simple, '    def _clean_output(self, output: str) -> str:
        """Clean up output."""
        result = output.strip()

        # Remove common prefixes
        for prefix in [\'A:\', \'Assistant:\']:
            if result.startswith(prefix):
                result = result[len(prefix):].strip()
                break

        return result
').

%% --- Fragment: list_models (CLI variant) ---

py_fragment(list_models_cli, '    def list_models(self) -> list[str]:
        """List available models using \'ollama list\'."""
        try:
            result = subprocess.run(
                [self.command, "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split(\'\\n\')[1:]  # Skip header
                return [line.split()[0] for line in lines if line.strip()]
        except Exception:
            pass
        return []
').

%% --- Fragment: list_models (API variant) ---

py_fragment(list_models_api, '    def list_models(self) -> list[str]:
        """List available models on the Ollama server."""
        try:
            url = f"{self.base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")

            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode(\'utf-8\'))

            return [m["name"] for m in result.get("models", [])]

        except Exception:
            return []
').

%% --- Fragment: Stream-JSON parser (gemini variant) ---

py_fragment(stream_json_parser_gemini, '            content_parts = []
            tokens = {}
            tool_count = 0
            last_tool_desc = None

            for line in proc.stdout:
                line = line.strip()
                if not line or not line.startswith(\'{\'):
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                etype = event.get(\'type\', \'\')

                if etype == \'message\' and event.get(\'role\') == \'assistant\':
                    text = event.get(\'content\', \'\')
                    content_parts.append(text)

                elif etype == \'tool_use\':
                    tool_count += 1
                    tool_name = event.get(\'tool_name\', \'?\')
                    params = event.get(\'parameters\', {})
                    last_tool_desc = self._describe_tool_call(tool_name, params)
                    if self._on_status:
                        self._on_status(f"[{tool_count}] {last_tool_desc}")

                elif etype == \'tool_result\':
                    status = event.get(\'status\', \'\')
                    if self._on_status:
                        self._on_status(f"[{tool_count}] {last_tool_desc} done ({status})")

                elif etype == \'result\':
                    stats = event.get(\'stats\', {})
                    tokens = {
                        \'input\': stats.get(\'input_tokens\', 0),
                        \'output\': stats.get(\'output_tokens\', 0),
                    }

            proc.wait(timeout=10)
            content = \'\'.join(content_parts).strip()

            if not content and proc.returncode != 0:
                stderr = proc.stderr.read()
                content = f"[Error: {stderr.strip()}]"
').

%% --- Fragment: Stream-JSON parser (claude_code variant) ---

py_fragment(stream_json_parser_claude_code, '            content_parts = []
            tokens = {}
            tool_count = 0
            last_tool_desc = None

            for line in proc.stdout:
                line = line.strip()
                if not line or not line.startswith(\'{\'):
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                etype = event.get(\'type\', \'\')

                if etype == \'assistant\':
                    # Assistant messages contain content blocks
                    message_data = event.get(\'message\', {})
                    content_blocks = message_data.get(\'content\', [])

                    for block in content_blocks:
                        btype = block.get(\'type\', \'\')

                        if btype == \'text\':
                            content_parts.append(block.get(\'text\', \'\'))

                        elif btype == \'tool_use\':
                            tool_count += 1
                            tool_name = block.get(\'name\', \'?\')
                            tool_input = block.get(\'input\', {})
                            last_tool_desc = self._describe_tool_call(tool_name, tool_input)
                            if self._on_status:
                                self._on_status(f"[{tool_count}] {last_tool_desc}")

                elif etype == \'user\':
                    # Tool result \u2014 include tool description so it\'s visible
                    if self._on_status and tool_count > 0:
                        self._on_status(f"[{tool_count}] {last_tool_desc} done")

                elif etype == \'result\':
                    # Final result with usage stats
                    usage = event.get(\'usage\', {})
                    tokens = {
                        \'input\': usage.get(\'input_tokens\', 0),
                        \'output\': usage.get(\'output_tokens\', 0),
                    }
                    cache_read = usage.get(\'cache_read_input_tokens\', 0)
                    cache_create = usage.get(\'cache_creation_input_tokens\', 0)
                    if cache_read:
                        tokens[\'cache_read\'] = cache_read
                    if cache_create:
                        tokens[\'cache_create\'] = cache_create

                    # Use result text if we didn\'t collect content from events
                    if not content_parts:
                        result_text = event.get(\'result\', \'\')
                        if result_text:
                            content_parts.append(result_text)

            proc.wait(timeout=10)
            content = \'\'.join(content_parts).strip()

            if not content and proc.returncode != 0:
                stderr = proc.stderr.read()
                content = f"[Error: {stderr.strip()}]"
').

%% --- Fragment: Popen setup for CLI stream-json ---

py_fragment(popen_setup, '            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                stdin=subprocess.DEVNULL
            )

').

%% --- Fragment: Stream-JSON return ---

py_fragment(stream_json_return, '        return AgentResponse(
            content=content,
            tool_calls=[],
            tokens=tokens,
        )
').

%% --- Fragment: SSE streaming (openrouter) ---

py_fragment(sse_streaming_openrouter, '    def send_message_streaming(self, message: str, context: list[dict],
                               on_token=None, **kwargs) -> AgentResponse:
        """Send message with SSE streaming. Tokens arrive via on_token callback."""
        # Build messages array (same as non-streaming)
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in context:
            role = msg.get(\'role\')
            if role == \'tool\':
                messages.append({
                    "role": "tool",
                    "tool_call_id": msg.get(\'tool_call_id\', \'\'),
                    "content": msg[\'content\'],
                })
            elif role == \'assistant\' and msg.get(\'tool_calls\'):
                messages.append({
                    "role": "assistant",
                    "content": msg.get(\'content\') or \'\',
                    "tool_calls": msg[\'tool_calls\'],
                })
            elif role in (\'user\', \'assistant\'):
                messages.append({"role": role, "content": msg[\'content\']})
        if message and (not context or context[-1].get(\'content\') != message):
            messages.append({"role": "user", "content": message})

        # Build request body with stream=True
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }
        if self.tool_schemas:
            body["tools"] = self.tool_schemas
            body["tool_choice"] = "auto"

        url = f"{self.base_url.rstrip(\'/\')}/chat/completions"
        req = Request(
            url,
            data=json.dumps(body).encode(\'utf-8\'),
            headers={
                \'Content-Type\': \'application/json\',
                \'Authorization\': f\'Bearer {self.api_key}\',
                \'HTTP-Referer\': \'https://github.com/s243a/UnifyWeaver\',
                \'X-Title\': \'UnifyWeaver Agent Loop\',
            },
            method=\'POST\'
        )

        content = ""
        tool_calls_acc = {}  # index -> {id, name, arguments}
        tokens = {}

        try:
            with urlopen(req, timeout=300) as resp:
                for raw_line in resp:
                    line = raw_line.decode(\'utf-8\', errors=\'replace\').rstrip(\'\\n\\r\')
                    if not line.startswith(\'data: \'):
                        continue
                    payload = line[6:]
                    if payload == \'[DONE]\':
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get(\'choices\', [])
                    if not choices:
                        # Usage-only chunk (some providers send this)
                        if chunk.get(\'usage\'):
                            usage = chunk[\'usage\']
                            tokens = {
                                \'input\': usage.get(\'prompt_tokens\', 0),
                                \'output\': usage.get(\'completion_tokens\', 0),
                                \'total\': usage.get(\'total_tokens\', 0),
                            }
                        continue

                    delta = choices[0].get(\'delta\', {})

                    # Text content
                    text = delta.get(\'content\')
                    if text:
                        content += text
                        if on_token:
                            on_token(text)

                    # Tool call deltas \u2014 accumulate by index
                    for tc_delta in delta.get(\'tool_calls\', []):
                        idx = tc_delta.get(\'index\', 0)
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                \'id\': \'\', \'name\': \'\', \'arguments\': \'\'
                            }
                        acc = tool_calls_acc[idx]
                        if tc_delta.get(\'id\'):
                            acc[\'id\'] = tc_delta[\'id\']
                        func = tc_delta.get(\'function\', {})
                        if func.get(\'name\'):
                            acc[\'name\'] = func[\'name\']
                        if func.get(\'arguments\'):
                            acc[\'arguments\'] += func[\'arguments\']

                    # Usage in final chunk
                    if chunk.get(\'usage\'):
                        usage = chunk[\'usage\']
                        tokens = {
                            \'input\': usage.get(\'prompt_tokens\', 0),
                            \'output\': usage.get(\'completion_tokens\', 0),
                            \'total\': usage.get(\'total_tokens\', 0),
                        }

        except HTTPError as e:
            error_body = e.read().decode(\'utf-8\', errors=\'replace\')
            try:
                err_data = json.loads(error_body)
                err_msg = err_data.get(\'error\', {}).get(\'message\', error_body[:200])
            except json.JSONDecodeError:
                err_msg = error_body[:200]
            return AgentResponse(content=f"[API Error {e.code}: {err_msg}]", tokens={})
        except URLError as e:
            return AgentResponse(content=f"[Network Error: {e.reason}]", tokens={})
        except Exception as e:
            return AgentResponse(content=f"[Error: {e}]", tokens={})

        # Convert accumulated tool call deltas to ToolCall objects
        tool_calls = []
        for acc in tool_calls_acc.values():
            try:
                arguments = json.loads(acc[\'arguments\']) if acc[\'arguments\'] else {}
            except json.JSONDecodeError:
                arguments = {"raw": acc[\'arguments\']}
            tool_calls.append(ToolCall(
                name=acc[\'name\'],
                arguments=arguments,
                id=acc[\'id\']
            ))

        return AgentResponse(
            content=content,
            tool_calls=tool_calls,
            tokens=tokens,
        )
').

%% =============================================================================
%% Fragments: context.py
%% =============================================================================

py_fragment(context_helpers, '
def _count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text (chars / 4 heuristic)."""
    return max(1, len(text) // 4)
').

py_fragment(context_manager_class, 'class ContextManager:
    """Manages conversation history and context window."""

    def __init__(
        self,
        max_tokens: int = 100000,
        max_messages: int = 50,
        max_chars: int = 0,
        max_words: int = 0,
        behavior: ContextBehavior = ContextBehavior.CONTINUE,
        format: ContextFormat = ContextFormat.PLAIN
    ):
        self.messages: list[Message] = []
        self.max_tokens = max_tokens
        self.max_messages = max_messages
        self.max_chars = max_chars
        self.max_words = max_words
        self.behavior = behavior
        self.format = format
        self._token_count = 0
        self._char_count = 0
        self._word_count = 0

    def add_message(self, role: str, content: str, tokens: int = 0,
                    tool_call_id: str = "", tool_calls: list | None = None) -> None:
        """Add a message to the context."""
        if tokens == 0:
            tokens = _estimate_tokens(content)

        msg = Message(role=role, content=content, tokens=tokens,
                      tool_call_id=tool_call_id,
                      tool_calls=tool_calls or [])
        self.messages.append(msg)
        self._token_count += tokens
        self._char_count += len(content)
        self._word_count += _count_words(content)
        self._trim_if_needed()

    def _trim_if_needed(self) -> None:
        """Remove old messages if any context limit is exceeded."""
        while len(self.messages) > self.max_messages:
            self._remove_oldest()

        while self._token_count > self.max_tokens and len(self.messages) > 1:
            self._remove_oldest()

        if self.max_chars > 0:
            while self._char_count > self.max_chars and len(self.messages) > 1:
                self._remove_oldest()

        if self.max_words > 0:
            while self._word_count > self.max_words and len(self.messages) > 1:
                self._remove_oldest()

    def _remove_oldest(self) -> None:
        """Remove the oldest message and update counters."""
        removed = self.messages.pop(0)
        self._token_count -= removed.tokens
        self._char_count -= len(removed.content)
        self._word_count -= _count_words(removed.content)

    @staticmethod
    def _msg_to_dict(msg: \'Message\') -> dict:
        """Convert a Message to a dict, including tool fields when present."""
        d = {"role": msg.role, "content": msg.content}
        if msg.role == \'tool\' and msg.tool_call_id:
            d["tool_call_id"] = msg.tool_call_id
        if msg.role == \'assistant\' and msg.tool_calls:
            d["tool_calls"] = msg.tool_calls
        return d

    def get_context(self) -> list[dict]:
        """Get context formatted for the backend."""
        if self.behavior == ContextBehavior.FRESH:
            return []

        messages = self.messages
        if self.behavior == ContextBehavior.SLIDING:
            messages = self.messages[-self.max_messages:]

        if self.max_chars <= 0 and self.max_words <= 0:
            return [self._msg_to_dict(msg) for msg in messages]

        result = []
        total_chars = 0
        total_words = 0
        for msg in reversed(messages):
            msg_chars = len(msg.content)
            msg_words = _count_words(msg.content)

            if self.max_chars > 0 and total_chars + msg_chars > self.max_chars and result:
                break
            if self.max_words > 0 and total_words + msg_words > self.max_words and result:
                break

            result.append(self._msg_to_dict(msg))
            total_chars += msg_chars
            total_words += msg_words

        result.reverse()
        return result

    def get_formatted_context(self) -> str:
        """Get context as a formatted string."""
        context = self.get_context()

        if self.format == ContextFormat.PLAIN:
            return self._format_plain(context)
        elif self.format == ContextFormat.MARKDOWN:
            return self._format_markdown(context)
        elif self.format == ContextFormat.JSON:
            return self._format_json(context)
        elif self.format == ContextFormat.XML:
            return self._format_xml(context)

        return self._format_plain(context)

    def _format_plain(self, context: list[dict]) -> str:
        """Format context as plain text."""
        lines = []
        for msg in context:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg[\'content\']}")
        return "\\n".join(lines)

    def _format_markdown(self, context: list[dict]) -> str:
        """Format context as markdown."""
        lines = []
        for msg in context:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"**{role}:**\\n{msg[\'content\']}\\n")
        return "\\n".join(lines)

    def _format_json(self, context: list[dict]) -> str:
        """Format context as JSON."""
        import json
        return json.dumps(context, indent=2)

    def _format_xml(self, context: list[dict]) -> str:
        """Format context as XML."""
        lines = ["<conversation>"]
        for msg in context:
            lines.append(f\'  <message role="{msg["role"]}">\'  )
            lines.append(f"    {msg[\'content\']}")
            lines.append("  </message>")
        lines.append("</conversation>")
        return "\\n".join(lines)

    def clear(self) -> None:
        """Clear all messages from context."""
        self.messages.clear()
        self._token_count = 0
        self._char_count = 0
        self._word_count = 0

    @property
    def token_count(self) -> int:
        """Current total token count."""
        return self._token_count

    @property
    def char_count(self) -> int:
        """Current total character count."""
        return self._char_count

    @property
    def word_count(self) -> int:
        """Current total word count."""
        return self._word_count

    @property
    def message_count(self) -> int:
        """Current message count."""
        return len(self.messages)

    def __len__(self) -> int:
        return len(self.messages)
').

%% =============================================================================
%% Fragments: config.py
%% =============================================================================

py_fragment(config_read_cascade, 'def read_config_cascade(no_fallback: bool = False) -> dict:
    """Read config from uwsal.json, falling back to coro.json."""
    candidates = [
        \'uwsal.json\',
        os.path.expanduser(\'~/uwsal.json\'),
    ]
    if not no_fallback:
        candidates += [
            \'coro.json\',
            os.path.expanduser(\'~/coro.json\'),
        ]
    for path in candidates:
        try:
            with open(path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    return {}
').

py_fragment(config_resolve_api_key_header, '
def resolve_api_key(backend_type: str, cli_key: str | None = None,
                    no_fallback: bool = False) -> str | None:
    """Resolve API key with full priority chain."""
    if cli_key:
        return cli_key

').

py_fragment(config_resolve_api_key_middle, '    env_var = env_vars.get(backend_type)
    if env_var:
        val = os.environ.get(env_var)
        if val:
            return val

    config = read_config_cascade(no_fallback)
    provider_key = config.get(\'keys\', {}).get(backend_type)
    if provider_key:
        return provider_key
    if config.get(\'api_key\'):
        return config[\'api_key\']

').

py_fragment(config_resolve_api_key_footer, '    loc = file_locations.get(backend_type)
    if loc:
        path = os.path.expanduser(loc)
        try:
            with open(path) as f:
                return f.read().strip()
        except FileNotFoundError:
            pass

    return None

').

py_fragment(config_imperative_functions, 'def _resolve_env_var(value: str) -> str:
    """Resolve environment variable references like $VAR or ${VAR}."""
    if not isinstance(value, str):
        return value
    if value.startswith(\'$\'):
        var_name = value[1:]
        if var_name.startswith(\'{\') and var_name.endswith(\'}\'):
            var_name = var_name[1:-1]
        return os.environ.get(var_name, value)
    return value


def _load_agent_config(name: str, data: dict) -> AgentConfig:
    """Load an agent config from a dictionary."""
    if \'api_key\' in data:
        data[\'api_key\'] = _resolve_env_var(data[\'api_key\'])

    return AgentConfig(
        name=name,
        backend=data.get(\'backend\', \'coro\'),
        model=data.get(\'model\'),
        host=data.get(\'host\'),
        port=data.get(\'port\'),
        api_key=data.get(\'api_key\'),
        command=data.get(\'command\'),
        system_prompt=data.get(\'system_prompt\'),
        agent_md=data.get(\'agent_md\'),
        tools=data.get(\'tools\', [\'bash\', \'read\', \'write\', \'edit\']),
        auto_tools=data.get(\'auto_tools\', False),
        context_mode=data.get(\'context_mode\', \'continue\'),
        max_context_tokens=data.get(\'max_context_tokens\', 100000),
        max_messages=data.get(\'max_messages\', 50),
        skills=data.get(\'skills\', []),
        max_iterations=data.get(\'max_iterations\', 0),
        timeout=data.get(\'timeout\', 300),
        show_tokens=data.get(\'show_tokens\', True),
        extra=data.get(\'extra\', {})
    )


def load_config(path: str | Path) -> Config:
    """Load configuration from a YAML or JSON file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    content = path.read_text()

    if path.suffix in (\'.yaml\', \'.yml\'):
        if not HAS_YAML:
            raise ImportError("PyYAML not installed. Use JSON config or: pip install pyyaml")
        data = yaml.safe_load(content)
    else:
        data = json.loads(content)

    config = Config(
        default=data.get(\'default\', \'default\'),
        config_dir=str(path.parent),
        skills_dir=data.get(\'skills_dir\', \'\')
    )

    for name, agent_data in data.get(\'agents\', {}).items():
        config.agents[name] = _load_agent_config(name, agent_data)

    if \'default\' not in config.agents:
        config.agents[\'default\'] = AgentConfig(name=\'default\', backend=\'coro\')

    return config


def load_config_from_dir(dir_path: str | Path = None) -> Config | None:
    """Load config from standard locations in a directory."""
    if dir_path is None:
        dir_path = Path.cwd()
    else:
        dir_path = Path(dir_path)

    for name in [\'agents.yaml\', \'agents.yml\', \'agents.json\',
                 \'.agents.yaml\', \'.agents.yml\', \'.agents.json\']:
        config_path = dir_path / name
        if config_path.exists():
            return load_config(config_path)

    return None
').

py_fragment(config_save_example, 'def save_example_config(path: str | Path):
    """Save an example configuration file."""
    path = Path(path)

    example = {
        "default": "claude-sonnet",
        "skills_dir": "./skills",
        "agents": {
            "claude-sonnet": {
                "backend": "claude-code",
                "model": "sonnet",
                "tools": ["bash", "read", "write", "edit"],
                "context_mode": "continue"
            },
            "claude-opus": {
                "backend": "claude-code",
                "model": "opus",
                "system_prompt": "You are a senior software engineer. Be thorough."
            },
            "yolo": {
                "backend": "claude-code",
                "model": "haiku",
                "auto_tools": True,
                "system_prompt": "Be fast and take action without asking."
            },
            "gemini": {
                "backend": "gemini",
                "model": "gemini-2.5-flash"
            },
            "ollama-local": {
                "backend": "ollama-api",
                "model": "llama3",
                "host": "localhost",
                "port": 11434
            },
            "ollama-remote": {
                "backend": "ollama-api",
                "model": "codellama",
                "host": "192.168.1.100",
                "port": 11434
            },
            "claude-api": {
                "backend": "claude",
                "model": "claude-sonnet-4-20250514",
                "api_key": "$ANTHROPIC_API_KEY"
            },
            "openai": {
                "backend": "openai",
                "model": "gpt-4o",
                "api_key": "$OPENAI_API_KEY"
            },
            "openai-mini": {
                "backend": "openai",
                "model": "gpt-4o-mini",
                "api_key": "$OPENAI_API_KEY"
            },
            "coding-assistant": {
                "backend": "claude-code",
                "model": "sonnet",
                "agent_md": "./agents/coding.md",
                "skills": ["./skills/git.md", "./skills/testing.md"],
                "tools": ["bash", "read", "write", "edit"],
                "system_prompt": "You are a coding assistant focused on clean, tested code."
            }
        }
    }

    if path.suffix in (\'.yaml\', \'.yml\'):
        if not HAS_YAML:
            raise ImportError("PyYAML not installed for YAML output")
        content = yaml.dump(example, default_flow_style=False, sort_keys=False)
    else:
        content = json.dumps(example, indent=2)

    path.write_text(content)
').

%% =============================================================================
%% Fragments: display.py (mostly imperative, single large fragment)
%% =============================================================================

py_fragment(display_module, '"""Display module using tput for Termux-compatible terminal control."""

import os
import subprocess
import sys
import threading
import time
from typing import Optional


def _get_tput_sequence(cap: str) -> str:
    """Get escape sequence for a terminfo capability."""
    try:
        result = subprocess.run(
            [\'tput\', cap],
            capture_output=True,
            text=True,
            timeout=1
        )
        return result.stdout
    except Exception:
        return \'\'


_SEQUENCES = {
    \'cr\': \'\\r\',
    \'el\': _get_tput_sequence(\'el\'),
    \'civis\': _get_tput_sequence(\'civis\'),
    \'cnorm\': _get_tput_sequence(\'cnorm\'),
    \'sc\': _get_tput_sequence(\'sc\'),
    \'rc\': _get_tput_sequence(\'rc\'),
}


def tput_write(cap: str) -> None:
    """Write cached tput sequence to stdout."""
    seq = _SEQUENCES.get(cap, \'\')
    if seq:
        os.write(sys.stdout.fileno(), seq.encode())


class TerminalControl:
    """Terminal control using tput commands."""

    @staticmethod
    def cr() -> None:
        tput_write(\'cr\')

    @staticmethod
    def el() -> None:
        tput_write(\'el\')

    @staticmethod
    def clear_line() -> None:
        tput_write(\'cr\')
        tput_write(\'el\')

    @staticmethod
    def save_cursor() -> None:
        tput_write(\'sc\')

    @staticmethod
    def restore_cursor() -> None:
        tput_write(\'rc\')

    @staticmethod
    def hide_cursor() -> None:
        tput_write(\'civis\')

    @staticmethod
    def show_cursor() -> None:
        tput_write(\'cnorm\')

    @staticmethod
    def move_up(n: int = 1) -> None:
        os.write(sys.stdout.fileno(), f"\\033[{n}A".encode())

    @staticmethod
    def move_down(n: int = 1) -> None:
        os.write(sys.stdout.fileno(), f"\\033[{n}B".encode())

    @staticmethod
    def cols() -> int:
        try:
            result = subprocess.run([\'tput\', \'cols\'], capture_output=True, text=True, timeout=1)
            return int(result.stdout.strip() or \'80\')
        except (ValueError, Exception):
            return 80

    @staticmethod
    def lines() -> int:
        try:
            result = subprocess.run([\'tput\', \'lines\'], capture_output=True, text=True, timeout=1)
            return int(result.stdout.strip() or \'24\')
        except (ValueError, Exception):
            return 24


class Spinner:
    """Animated spinner that updates in place using tput."""

    FRAMES = [\'\\u280b\', \'\\u2819\', \'\\u2839\', \'\\u2838\', \'\\u283c\', \'\\u2834\', \'\\u2826\', \'\\u2827\', \'\\u2807\', \'\\u280f\']
    INTERVAL = 0.1

    def __init__(self, message: str = "Working"):
        self.message = message
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame = 0
        self._tc = TerminalControl()
        self._cols = self._tc.cols()
        self._lock = threading.Lock()
        self._start_time = 0.0

    def _truncate(self, text: str, width: int) -> str:
        max_len = width - 2
        if len(text) > max_len:
            return text[:max_len - 3] + \'...\'
        return text

    def _animate(self) -> None:
        self._tc.hide_cursor()
        try:
            while self._running:
                with self._lock:
                    elapsed = time.time() - self._start_time
                    elapsed_str = f" ({elapsed:.0f}s)" if elapsed >= 2 else ""
                    frame = self.FRAMES[self._frame % len(self.FRAMES)]
                    display = self._truncate(
                        self.message + elapsed_str, self._cols
                    )
                    self._tc.clear_line()
                    sys.stdout.write(f"{frame} {display}")
                    sys.stdout.flush()
                    self._frame += 1
                time.sleep(self.INTERVAL)
        finally:
            self._tc.show_cursor()

    def start(self) -> None:
        if self._running:
            return
        self._start_time = time.time()
        self._running = True
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self, final_message: Optional[str] = None) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
            self._thread = None

        with self._lock:
            self._tc.clear_line()
            sys.stdout.write(f"  {self.message}\\n")
            if final_message:
                sys.stdout.write(final_message + \'\\n\')
            sys.stdout.flush()

    def update(self, message: str) -> None:
        with self._lock:
            if message != self.message:
                self._tc.clear_line()
                sys.stdout.write(f"  {self.message}\\n")
                sys.stdout.flush()
                self.message = message

    def __enter__(self) -> \'Spinner\':
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


class ProgressBar:
    """Progress bar for streaming that updates in place."""

    def __init__(self, total: int = 100, width: int = 30,
                 prefix: str = "", suffix: str = ""):
        self.total = total
        self.width = width
        self.prefix = prefix
        self.suffix = suffix
        self.current = 0
        self._tc = TerminalControl()

    def update(self, current: Optional[int] = None,
               increment: int = 0, suffix: str = None) -> None:
        if current is not None:
            self.current = current
        else:
            self.current += increment

        if suffix is not None:
            self.suffix = suffix

        self._render()

    def _render(self) -> None:
        if self.total <= 0:
            pct = 0
        else:
            pct = min(100, int(100 * self.current / self.total))

        filled = int(self.width * pct / 100)
        bar = \'\\u2588\' * filled + \'\\u2591\' * (self.width - filled)

        self._tc.clear_line()
        sys.stdout.write(f"{self.prefix}[{bar}] {pct}% {self.suffix}")
        sys.stdout.flush()

    def finish(self, message: str = "") -> None:
        self.current = self.total
        self._render()
        sys.stdout.write(\'\\n\')
        if message:
            sys.stdout.write(message + \'\\n\')
        sys.stdout.flush()


class StatusLine:
    """Status line at the bottom of the terminal."""

    def __init__(self):
        self._tc = TerminalControl()
        self._saved = False
        self._status = ""

    def show(self, status: str) -> None:
        self._status = status
        self._tc.clear_line()
        cols = self._tc.cols()
        if len(status) > cols - 1:
            status = status[:cols - 4] + \'...\'
        sys.stdout.write(status)
        sys.stdout.flush()

    def clear(self) -> None:
        self._tc.clear_line()
        sys.stdout.flush()
        self._status = ""

    def update(self, status: str) -> None:
        self.show(status)


class StreamDisplay:
    """Display for streaming output with progress indication."""

    def __init__(self, show_tokens: bool = True):
        self.show_tokens = show_tokens
        self._tc = TerminalControl()
        self._char_count = 0
        self._line_count = 0

    def start(self, label: str = "Streaming") -> None:
        sys.stdout.write(f"\\n{label}:\\n")
        sys.stdout.flush()
        self._char_count = 0
        self._line_count = 0

    def chunk(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()
        self._char_count += len(text)
        self._line_count += text.count(\'\\n\')

    def finish(self, tokens: Optional[dict] = None) -> None:
        sys.stdout.write(\'\\n\')
        if self.show_tokens and tokens:
            input_t = tokens.get(\'input\', 0)
            output_t = tokens.get(\'output\', 0)
            sys.stdout.write(f"  [Tokens: input={input_t}, output={output_t}]\\n")
        sys.stdout.flush()


class DisplayMode:
    """Factory for display components based on mode."""

    APPEND_ONLY = \'append\'
    NCURSES = \'ncurses\'

    @classmethod
    def supports_ncurses(cls) -> bool:
        return bool(_SEQUENCES.get(\'el\'))

    @classmethod
    def get_spinner(cls, message: str, mode: str = None) -> \'Spinner\':
        if mode is None:
            mode = cls.NCURSES if cls.supports_ncurses() else cls.APPEND_ONLY

        if mode == cls.APPEND_ONLY:
            return AppendOnlySpinner(message)
        return Spinner(message)

    @classmethod
    def get_progress(cls, total: int, mode: str = None, **kwargs) -> \'ProgressBar\':
        if mode is None:
            mode = cls.NCURSES if cls.supports_ncurses() else cls.APPEND_ONLY

        if mode == cls.APPEND_ONLY:
            return AppendOnlyProgress(total, **kwargs)
        return ProgressBar(total, **kwargs)


class AppendOnlySpinner:
    """Fallback spinner for append-only mode."""

    def __init__(self, message: str = "Working"):
        self.message = message
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _animate(self) -> None:
        count = 0
        while self._running:
            if count % 10 == 0:
                sys.stdout.write(\'.\')
                sys.stdout.flush()
            count += 1
            time.sleep(0.1)

    def start(self) -> None:
        if self._running:
            return
        sys.stdout.write(f"[{self.message}]")
        sys.stdout.flush()
        self._running = True
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self, final_message: Optional[str] = None) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
            self._thread = None

        if final_message:
            sys.stdout.write(f" {final_message}\\n")
        else:
            sys.stdout.write(" done\\n")
        sys.stdout.flush()

    def update(self, message: str) -> None:
        pass

    def __enter__(self) -> \'AppendOnlySpinner\':
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


class AppendOnlyProgress:
    """Fallback progress bar for append-only mode."""

    def __init__(self, total: int = 100, width: int = 30,
                 prefix: str = "", suffix: str = ""):
        self.total = total
        self.prefix = prefix
        self._last_pct = -1

    def update(self, current: Optional[int] = None,
               increment: int = 0, suffix: str = None) -> None:
        if current is None:
            current = increment

        if self.total <= 0:
            return

        pct = min(100, int(100 * current / self.total))
        milestone = (pct // 25) * 25
        if milestone > self._last_pct:
            sys.stdout.write(f"[{milestone}%]")
            sys.stdout.flush()
            self._last_pct = milestone

    def finish(self, message: str = "") -> None:
        if self._last_pct < 100:
            sys.stdout.write("[100%]")
        sys.stdout.write(f" {message}\\n" if message else "\\n")
        sys.stdout.flush()
').

%% =============================================================================
%% Fragments: tools.py
%% =============================================================================

py_fragment(tools_validate_path, 'def validate_path(raw_path: str, working_dir: str,
                  extra_blocked: list[str] | None = None,
                  extra_allowed: list[str] | None = None) -> tuple[str, str | None]:
    """Resolve and validate a file path.

    Returns (resolved_path, error_message).
    error_message is None if the path is safe.
    """
    if raw_path.startswith(\'/\'):
        resolved = os.path.realpath(raw_path)
    else:
        resolved = os.path.realpath(os.path.join(working_dir, raw_path))

    raw_abs = os.path.abspath(os.path.join(working_dir, raw_path)) if not raw_path.startswith(\'/\') else raw_path
    check_paths = {resolved, raw_abs}

    if extra_allowed:
        for pattern in extra_allowed:
            expanded = os.path.realpath(os.path.expanduser(pattern))
            for p in check_paths:
                if p == expanded or p.startswith(expanded.rstrip(\'/\') + \'/\'):
                    return resolved, None

    if extra_blocked:
        for pattern in extra_blocked:
            expanded = os.path.realpath(os.path.expanduser(pattern))
            for p in check_paths:
                if p == expanded or p.startswith(expanded.rstrip(\'/\') + \'/\'):
                    return resolved, f"Blocked by config: {raw_path}"

    for p in check_paths:
        if p in _BLOCKED_PATHS:
            return resolved, f"Blocked: {raw_path} is a sensitive system file"
        for prefix in _BLOCKED_PREFIXES:
            if p.startswith(prefix):
                return resolved, f"Blocked: {raw_path} is in a system directory"

    home = os.path.expanduser(\'~\')
    for p in check_paths:
        if p.startswith(home + \'/\'):
            rel = p[len(home) + 1:]
            for pattern in _BLOCKED_HOME_PATTERNS:
                if rel == pattern.rstrip(\'/\') or rel.startswith(pattern):
                    return resolved, f"Blocked: ~/{pattern} may contain credentials"

    return resolved, None
').

py_fragment(tools_is_command_blocked, 'def is_command_blocked(command: str,
                       extra_blocked: list[str] | None = None,
                       extra_allowed: list[str] | None = None) -> str | None:
    """Return reason if command is blocked, None if allowed."""
    if extra_allowed:
        for pattern in extra_allowed:
            if re.search(pattern, command, re.IGNORECASE):
                return None

    if extra_blocked:
        for pattern in extra_blocked:
            if re.search(pattern, command, re.IGNORECASE):
                return f"Blocked by config: matches \'{pattern}\'"

    for pattern, description in _BLOCKED_COMMAND_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return f"Blocked: {description}"
    return None
').

py_fragment(tools_security_config, '# ── Security config ────────────────────────────────────────────────────────

@dataclass
class SecurityConfig:
    """Security settings for tool execution."""
    path_validation: bool = True
    command_blocklist: bool = True
    blocked_paths: list[str] = field(default_factory=list)
    blocked_commands: list[str] = field(default_factory=list)
    allowed_paths: list[str] = field(default_factory=list)
    allowed_commands: list[str] = field(default_factory=list)

    allowed_commands_only: bool = False
    safe_commands: list[str] = field(default_factory=list)
    command_proxying: str = \'disabled\'
    max_file_read_size: int | None = None
    max_file_write_size: int | None = None

    path_proxying: bool = False
    proot_sandbox: bool = False
    proot_allowed_dirs: list[str] = field(default_factory=list)

    @classmethod
    def from_profile(cls, profile: str) -> \'SecurityConfig\':
        """Create config from a named security profile."""
        sp = get_profile(profile)
        return cls(
            path_validation=sp.path_validation,
            command_blocklist=sp.command_blocklist,
            blocked_paths=list(sp.blocked_paths),
            blocked_commands=list(sp.blocked_commands),
            allowed_paths=list(sp.allowed_paths),
            allowed_commands=list(sp.allowed_commands),
            allowed_commands_only=sp.allowed_commands_only,
            safe_commands=list(sp.safe_commands),
            command_proxying=sp.command_proxying,
            max_file_read_size=sp.max_file_read_size,
            max_file_write_size=sp.max_file_write_size,
            path_proxying=sp.path_proxying,
            proot_sandbox=sp.proot_isolation,
            proot_allowed_dirs=list(sp.proot_allowed_dirs),
        )
').

py_fragment(tools_handler_class, 'class ToolHandler:
    """Handles execution of tool calls from agent responses."""

    def __init__(
        self,
        allowed_tools: list[str] | None = None,
        confirm_destructive: bool = True,
        working_dir: str | None = None,
        security: SecurityConfig | None = None,
        audit: AuditLogger | None = None
    ):
        self.allowed_tools = allowed_tools or [\'bash\', \'read\', \'write\', \'edit\']
        self.confirm_destructive = confirm_destructive
        self.working_dir = working_dir or os.getcwd()
        self.security = security or SecurityConfig()
        self.audit = audit or AuditLogger(level=\'disabled\')

        self.proxy: CommandProxyManager | None = None
        if self.security.command_proxying != \'disabled\':
            self.proxy = CommandProxyManager()

        self.path_proxy = None
        if self.security.path_proxying:
            from security.path_proxy import PathProxyManager
            self.path_proxy = PathProxyManager()
            if self.proxy:
                generated = self.path_proxy.generate_wrappers(self.proxy)
                if generated:
                    self.audit.log_proxy_action(
                        \'\', \'path_proxy\', \'init\',
                        f\'Generated wrappers: {", ".join(generated)}\')

        self.proot = None
        if self.security.proot_sandbox:
            from security.proot_sandbox import ProotSandbox, ProotConfig
            proot_cfg = ProotConfig(
                allowed_dirs=self.security.proot_allowed_dirs)
            sandbox = ProotSandbox(self.working_dir, proot_cfg)
            if sandbox.is_available():
                self.proot = sandbox
            else:
                import sys
                print("[Warning] proot sandbox requested but proot not "
                      "found. Install with: pkg install proot",
                      file=sys.stderr)

        self.tools = {
            \'bash\': self._execute_bash,
            \'read\': self._read_file,
            \'write\': self._write_file,
            \'edit\': self._edit_file,
        }

        self.destructive_tools = {\'bash\', \'write\', \'edit\'}

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result."""
        if tool_call.name not in self.allowed_tools:
            return ToolResult(
                success=False,
                output=f"Tool \'{tool_call.name}\' is not allowed",
                tool_name=tool_call.name
            )

        if tool_call.name not in self.tools:
            return ToolResult(
                success=False,
                output=f"Unknown tool: {tool_call.name}",
                tool_name=tool_call.name
            )

        if tool_call.name == \'bash\':
            blocked = self._pre_check_bash(tool_call.arguments)
            if blocked:
                self.audit.log_tool_call(
                    tool_call.name, False,
                    args_summary=str(tool_call.arguments)[:256]
                )
                return blocked

        if self.confirm_destructive and tool_call.name in self.destructive_tools:
            if not self._is_safe_command(tool_call):
                if not self._confirm_execution(tool_call):
                    return ToolResult(
                        success=False,
                        output="User declined to execute tool",
                        tool_name=tool_call.name
                    )

        try:
            result = self.tools[tool_call.name](tool_call.arguments)
            self.audit.log_tool_call(
                tool_call.name, result.success,
                args_summary=str(tool_call.arguments)[:256]
            )
            return result
        except Exception as e:
            self.audit.log_tool_call(tool_call.name, False, args_summary=str(e))
            return ToolResult(
                success=False,
                output=f"Error executing {tool_call.name}: {e}",
                tool_name=tool_call.name
            )

    def _confirm_execution(self, tool_call: ToolCall) -> bool:
        """Ask user to confirm execution of a destructive tool."""
        print(f"\\n[Tool: {tool_call.name}]")

        if tool_call.name == \'bash\':
            cmd = tool_call.arguments.get(\'command\', \'\')
            print(f"  Command: {cmd}")
        elif tool_call.name == \'write\':
            path = tool_call.arguments.get(\'path\', \'\')
            print(f"  File: {path}")
        elif tool_call.name == \'edit\':
            path = tool_call.arguments.get(\'path\', \'\')
            old = tool_call.arguments.get(\'old_string\', \'\')[:50]
            print(f"  File: {path}")
            print(f"  Replace: {old}...")

        try:
            response = input("Execute? [y/N]: ").strip().lower()
            return response in (\'y\', \'yes\')
        except (EOFError, KeyboardInterrupt):
            return False

    def _is_safe_command(self, tool_call: ToolCall) -> bool:
        """Check if a tool call is safe enough to skip confirmation."""
        if tool_call.name != \'bash\':
            return False
        if not self.security.safe_commands:
            return False

        command = tool_call.arguments.get(\'command\', \'\').strip()
        return any(
            re.search(pat, command, re.IGNORECASE)
            for pat in self.security.safe_commands
        )

    def _pre_check_bash(self, args: dict) -> ToolResult | None:
        """Run security checks on a bash command BEFORE confirmation."""
        command = args.get(\'command\', \'\').strip()
        if not command:
            return ToolResult(
                success=False,
                output="No command provided",
                tool_name=\'bash\'
            )

        if self.security.allowed_commands_only:
            matched = any(
                re.search(pat, command, re.IGNORECASE)
                for pat in self.security.allowed_commands
            )
            if not matched:
                reason = "Command not in allowlist"
                self.audit.log_command(command, allowed=False, reason=reason)
                return ToolResult(
                    success=False,
                    output=f"[Security] {reason}",
                    tool_name=\'bash\'
                )

        if self.security.command_blocklist:
            reason = is_command_blocked(
                command,
                extra_blocked=self.security.blocked_commands,
                extra_allowed=self.security.allowed_commands if not self.security.allowed_commands_only else None,
            )
            if reason:
                self.audit.log_command(command, allowed=False, reason=reason)
                return ToolResult(
                    success=False,
                    output=f"[Security] {reason}",
                    tool_name=\'bash\'
                )

        if self.proxy:
            proxy_mode = self.security.command_proxying
            allowed, reason = self.proxy.check(command, proxy_mode)
            if not allowed:
                self.audit.log_command(command, allowed=False, reason=reason)
                self.audit.log_proxy_action(command, \'\', \'block\', reason or \'\')
                return ToolResult(
                    success=False,
                    output=f"[Security/Proxy] {reason}",
                    tool_name=\'bash\'
                )

        return None

    def _execute_bash(self, args: dict) -> ToolResult:
        """Execute a bash command."""
        command = args.get(\'command\', \'\')
        if not command:
            return ToolResult(
                success=False,
                output="No command provided",
                tool_name=\'bash\'
            )

        exec_env = None
        exec_cmd = command

        if self.path_proxy:
            log_file = self.audit._log_file if self.audit._log_file else None
            exec_env = self.path_proxy.build_env(
                proxy_mode=self.security.command_proxying,
                audit_log=str(log_file) if log_file else None,
            )

        if self.proot:
            exec_cmd = self.proot.wrap_command(command)
            proot_env = self.proot.build_env_overrides()
            if exec_env is None:
                exec_env = dict(os.environ)
            exec_env.update(proot_env)

        try:
            result = subprocess.run(
                exec_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.working_dir,
                env=exec_env,
            )

            output = result.stdout
            if result.stderr:
                output += "\\n" + result.stderr

            self.audit.log_command(
                command, allowed=True,
                output=output.strip() if output else None
            )

            return ToolResult(
                success=result.returncode == 0,
                output=output.strip() or "(no output)",
                tool_name=\'bash\'
            )

        except subprocess.TimeoutExpired:
            self.audit.log_command(command, allowed=True, reason=\'timeout\')
            return ToolResult(
                success=False,
                output="Command timed out after 120 seconds",
                tool_name=\'bash\'
            )
        except Exception as e:
            self.audit.log_command(command, allowed=True, reason=str(e))
            return ToolResult(
                success=False,
                output=f"Error: {e}",
                tool_name=\'bash\'
            )

    def _validate_file_path(self, raw_path: str, tool_name: str) -> tuple[Path, ToolResult | None]:
        """Validate and resolve a file path."""
        if not raw_path:
            return Path(), ToolResult(False, "No path provided", tool_name)

        if self.security.path_validation:
            resolved, error = validate_path(
                raw_path, self.working_dir,
                extra_blocked=self.security.blocked_paths,
                extra_allowed=self.security.allowed_paths,
            )
            if error:
                self.audit.log_file_access(raw_path, tool_name, allowed=False, reason=error)
                return Path(resolved), ToolResult(False, f"[Security] {error}", tool_name)
            return Path(resolved), None

        if raw_path.startswith(\'/\'):
            return Path(raw_path), None
        return Path(self.working_dir) / raw_path, None

    def _read_file(self, args: dict) -> ToolResult:
        """Read a file."""
        path = args.get(\'path\', \'\')
        file_path, error = self._validate_file_path(path, \'read\')
        if error:
            return error

        try:
            if self.security.max_file_read_size is not None:
                try:
                    file_size = file_path.stat().st_size
                    if file_size > self.security.max_file_read_size:
                        reason = (f"File too large: {file_size} bytes "
                                  f"(limit: {self.security.max_file_read_size})")
                        self.audit.log_file_access(path, \'read\', allowed=False, reason=reason)
                        return ToolResult(False, f"[Security] {reason}", \'read\')
                except OSError:
                    pass

            content = file_path.read_text()
            self.audit.log_file_access(
                path, \'read\', allowed=True, bytes_count=len(content)
            )
            return ToolResult(
                success=True,
                output=content,
                tool_name=\'read\'
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                output=f"File not found: {path}",
                tool_name=\'read\'
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=f"Error reading file: {e}",
                tool_name=\'read\'
            )

    def _write_file(self, args: dict) -> ToolResult:
        """Write content to a file."""
        path = args.get(\'path\', \'\')
        content = args.get(\'content\', \'\')

        file_path, error = self._validate_file_path(path, \'write\')
        if error:
            return error

        if self.security.max_file_write_size is not None:
            if len(content) > self.security.max_file_write_size:
                reason = (f"Content too large: {len(content)} bytes "
                          f"(limit: {self.security.max_file_write_size})")
                self.audit.log_file_access(path, \'write\', allowed=False, reason=reason)
                return ToolResult(False, f"[Security] {reason}", \'write\')

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            self.audit.log_file_access(
                path, \'write\', allowed=True, bytes_count=len(content)
            )
            return ToolResult(
                success=True,
                output=f"Wrote {len(content)} bytes to {path}",
                tool_name=\'write\'
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=f"Error writing file: {e}",
                tool_name=\'write\'
            )

    def _edit_file(self, args: dict) -> ToolResult:
        """Edit a file with search/replace."""
        path = args.get(\'path\', \'\')
        old_string = args.get(\'old_string\', \'\')
        new_string = args.get(\'new_string\', \'\')

        file_path, error = self._validate_file_path(path, \'edit\')
        if error:
            return error

        if not old_string:
            return ToolResult(
                success=False,
                output="No old_string provided",
                tool_name=\'edit\'
            )

        try:
            content = file_path.read_text()

            if old_string not in content:
                return ToolResult(
                    success=False,
                    output=f"old_string not found in {path}",
                    tool_name=\'edit\'
                )

            count = content.count(old_string)
            if count > 1:
                return ToolResult(
                    success=False,
                    output=f"old_string found {count} times - must be unique",
                    tool_name=\'edit\'
                )

            new_content = content.replace(old_string, new_string, 1)
            file_path.write_text(new_content)
            self.audit.log_file_access(
                path, \'edit\', allowed=True,
                bytes_count=len(new_content)
            )

            return ToolResult(
                success=True,
                output=f"Edited {path}",
                tool_name=\'edit\'
            )

        except FileNotFoundError:
            return ToolResult(
                success=False,
                output=f"File not found: {path}",
                tool_name=\'edit\'
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=f"Error editing file: {e}",
                tool_name=\'edit\'
            )

    def format_result_for_agent(self, result: ToolResult) -> str:
        """Format a tool result for sending back to the agent."""
        status = "Success" if result.success else "Failed"
        return f"[Tool {result.tool_name} - {status}]\\n{result.output}"
').

%% =============================================================================
%% Generator: Individual backends (full implementations)
%% =============================================================================

generate_backend(Name) :-
    agent_backend(Name, Props),
    resolve_file_name(Name, FileName),
    atom_concat('generated/backends/', FileName, Path1),
    atom_concat(Path1, '.py', Path),
    open(Path, write, S),
    generate_backend_full(S, Name, Props),
    close(S),
    format('  Generated backends/~w.py~n', [FileName]).

%% --- ollama_cli (uses: format_prompt, clean_output_simple, list_models_cli, name_property) ---

generate_backend_full(S, ollama_cli, Props) :-
    member(description(Desc), Props),
    format(S, '"""~w"""~n~n', [Desc]),
    write(S, 'import subprocess\n'),
    write(S, 'from .base import AgentBackend, AgentResponse, ToolCall\n\n\n'),
    write(S, 'class OllamaCLIBackend(AgentBackend):\n'),
    write(S, '    """Ollama CLI backend using \'ollama run\' command."""\n\n'),
    write(S, '    def __init__(\n'),
    write(S, '        self,\n'),
    write(S, '        command: str = "ollama",\n'),
    write(S, '        model: str = "llama3",\n'),
    write(S, '        timeout: int = 300\n'),
    write(S, '    ):\n'),
    write(S, '        self.command = command\n'),
    write(S, '        self.model = model\n'),
    write(S, '        self.timeout = timeout\n\n'),
    write(S, '    def send_message(self, message: str, context: list[dict], **kwargs) -> AgentResponse:\n'),
    write(S, '        """Send message to Ollama CLI and get response."""\n'),
    write(S, '        prompt = self._format_prompt(message, context)\n\n'),
    write(S, '        cmd = [self.command, "run", self.model]\n\n'),
    write(S, '        try:\n'),
    write(S, '            result = subprocess.run(\n'),
    write(S, '                cmd,\n'),
    write(S, '                input=prompt,\n'),
    write(S, '                capture_output=True,\n'),
    write(S, '                text=True,\n'),
    write(S, '                timeout=self.timeout\n'),
    write(S, '            )\n\n'),
    write(S, '            output = result.stdout\n'),
    write(S, '            if result.returncode != 0 and result.stderr:\n'),
    write(S, '                output = f"[Error: {result.stderr.strip()}]"\n\n'),
    write(S, '        except subprocess.TimeoutExpired:\n'),
    write(S, '            return AgentResponse(\n'),
    write(S, '                content=f"[Error: Command timed out after {self.timeout} seconds]",\n'),
    write(S, '                tokens={}\n'),
    write(S, '            )\n'),
    write(S, '        except FileNotFoundError:\n'),
    write(S, '            return AgentResponse(\n'),
    write(S, '                content=f"[Error: Command \'{self.command}\' not found. Install Ollama from https://ollama.ai]",\n'),
    write(S, '                tokens={}\n'),
    write(S, '            )\n'),
    write(S, '        except Exception as e:\n'),
    write(S, '            return AgentResponse(\n'),
    write(S, '                content=f"[Error: {e}]",\n'),
    write(S, '                tokens={}\n'),
    write(S, '            )\n\n'),
    write(S, '        content = self._clean_output(output)\n\n'),
    write(S, '        return AgentResponse(\n'),
    write(S, '            content=content,\n'),
    write(S, '            tool_calls=[],\n'),
    write(S, '            tokens={},\n'),
    write(S, '            raw=output\n'),
    write(S, '        )\n\n'),
    write_py(S, format_prompt), write(S, '\n'),
    write_py(S, clean_output_simple), write(S, '\n'),
    write_py(S, list_models_cli), write(S, '\n'),
    write_py(S, name_property, [display_name='Ollama CLI ({self.model})']).

%% --- ollama_api (uses: messages_builder_ollama, list_models_api, name_property) ---

generate_backend_full(S, ollama_api, Props) :-
    member(description(Desc), Props),
    format(S, '"""~w"""~n~n', [Desc]),
    write(S, 'import json\nimport urllib.request\nimport urllib.error\n'),
    write(S, 'from .base import AgentBackend, AgentResponse, ToolCall\n\n\n'),
    write(S, 'class OllamaAPIBackend(AgentBackend):\n'),
    write(S, '    """Ollama REST API backend for local models."""\n\n'),
    write(S, '    def __init__(\n        self,\n        host: str = "localhost",\n'),
    write(S, '        port: int = 11434,\n        model: str = "llama3",\n'),
    write(S, '        system_prompt: str | None = None,\n        timeout: int = 300\n    ):\n'),
    write(S, '        self.host = host\n        self.port = port\n        self.model = model\n'),
    write(S, '        self.system_prompt = system_prompt or "You are a helpful assistant."\n'),
    write(S, '        self.timeout = timeout\n'),
    write(S, '        self.base_url = f"http://{host}:{port}"\n\n'),
    write(S, '    def send_message(self, message: str, context: list[dict], **kwargs) -> AgentResponse:\n'),
    write(S, '        """Send message to Ollama API and get response."""\n'),
    write_py(S, messages_builder_ollama),
    write(S, '\n        # Build request\n'),
    write(S, '        payload = {\n            "model": self.model,\n'),
    write(S, '            "messages": messages,\n            "stream": False\n        }\n\n'),
    write(S, '        try:\n            url = f"{self.base_url}/api/chat"\n'),
    write(S, '            data = json.dumps(payload).encode(\'utf-8\')\n'),
    write(S, '            req = urllib.request.Request(\n                url,\n                data=data,\n'),
    write(S, '                headers={"Content-Type": "application/json"},\n                method="POST"\n            )\n\n'),
    write(S, '            with urllib.request.urlopen(req, timeout=self.timeout) as response:\n'),
    write(S, '                result = json.loads(response.read().decode(\'utf-8\'))\n\n'),
    write(S, '            content = result.get("message", {}).get("content", "")\n\n'),
    write(S, '            tokens = {}\n'),
    write(S, '            if "eval_count" in result:\n                tokens["output"] = result["eval_count"]\n'),
    write(S, '            if "prompt_eval_count" in result:\n                tokens["input"] = result["prompt_eval_count"]\n\n'),
    write(S, '            return AgentResponse(\n                content=content,\n                tool_calls=[],\n'),
    write(S, '                tokens=tokens,\n                raw=result\n            )\n\n'),
    write(S, '        except urllib.error.URLError as e:\n'),
    write(S, '            return AgentResponse(\n'),
    write(S, '                content=f"[Error: Cannot connect to Ollama at {self.base_url}. Is Ollama running? Error: {e}]",\n'),
    write(S, '                tokens={}\n            )\n'),
    write(S, '        except json.JSONDecodeError as e:\n'),
    write(S, '            return AgentResponse(\n'),
    write(S, '                content=f"[Error: Invalid JSON response from Ollama: {e}]",\n'),
    write(S, '                tokens={}\n            )\n'),
    write(S, '        except Exception as e:\n            return AgentResponse(\n'),
    write(S, '                content=f"[Error: {e}]",\n                tokens={}\n            )\n\n'),
    write_py(S, list_models_api), write(S, '\n'),
    write_py(S, name_property, [display_name='Ollama API ({self.model}@{self.host}:{self.port})']).

%% --- gemini (uses: popen_setup, stream_json_parser_gemini, describe_tool_call_gemini, format_prompt) ---

generate_backend_full(S, gemini, Props) :-
    member(description(Desc), Props),
    format(S, '"""~w"""~n~n', [Desc]),
    write(S, 'import json\nimport os\nimport subprocess\nimport sys\n'),
    write(S, 'from .base import AgentBackend, AgentResponse, ToolCall\n\n\n'),
    write(S, 'class GeminiBackend(AgentBackend):\n'),
    write(S, '    """Gemini CLI backend with streaming JSON output."""\n\n'),
    write(S, '    def __init__(self, command: str = "gemini", model: str = "gemini-3-flash-preview",\n'),
    write(S, '                 sandbox: bool = False, approval_mode: str = "yolo",\n'),
    write(S, '                 allowed_tools: list[str] | None = None):\n'),
    write(S, '        self.command = command\n        self.model = model\n'),
    write(S, '        self.sandbox = sandbox\n        self.approval_mode = approval_mode\n'),
    write(S, '        self.allowed_tools = allowed_tools or []\n'),
    write(S, '        self._on_status = None\n\n'),
    write(S, '    def send_message(self, message: str, context: list[dict], **kwargs) -> AgentResponse:\n'),
    write(S, '        """Send message to Gemini CLI with live streaming output."""\n'),
    write(S, '        prompt = self._format_prompt(message, context)\n'),
    write(S, '        self._on_status = kwargs.get(\'on_status\')\n\n'),
    write(S, '        cmd = [\n            self.command,\n            "-m", self.model,\n'),
    write(S, '            "-o", "stream-json",\n        ]\n'),
    write(S, '        if self.sandbox:\n            cmd.append("-s")\n'),
    write(S, '        if self.approval_mode:\n            cmd.extend(["--approval-mode", self.approval_mode])\n'),
    write(S, '        for tool in self.allowed_tools:\n            cmd.extend(["--allowed-tools", tool])\n'),
    write(S, '        cmd.extend(["-p", prompt])\n\n        try:\n'),
    write_py(S, popen_setup),
    write_py(S, stream_json_parser_gemini),
    write(S, '\n'),
    write(S, '        except subprocess.TimeoutExpired:\n            proc.kill()\n'),
    write(S, '            return AgentResponse(content="[Error: Command timed out]", tokens={})\n'),
    write(S, '        except FileNotFoundError:\n'),
    write(S, '            return AgentResponse(\n'),
    write(S, '                content=f"[Error: Command \'{self.command}\' not found. Install Gemini CLI.]",\n'),
    write(S, '                tokens={})\n'),
    write(S, '        except Exception as e:\n'),
    write(S, '            return AgentResponse(content=f"[Error: {e}]", tokens={})\n\n'),
    write_py(S, stream_json_return), write(S, '\n'),
    write_py(S, describe_tool_call_gemini), write(S, '\n'),
    write_py(S, format_prompt), write(S, '\n'),
    write_py(S, name_property, [display_name='Gemini ({self.model})']).

%% --- claude_code (uses: popen_setup, stream_json_parser_claude_code, describe_tool_call_claude_code, format_prompt) ---

generate_backend_full(S, claude_code, Props) :-
    member(description(Desc), Props),
    format(S, '"""~w"""~n~n', [Desc]),
    write(S, 'import json\nimport os\nimport subprocess\n'),
    write(S, 'from .base import AgentBackend, AgentResponse, ToolCall\n\n\n'),
    write(S, 'class ClaudeCodeBackend(AgentBackend):\n'),
    write(S, '    """Claude Code CLI backend with streaming JSON output."""\n\n'),
    write(S, '    def __init__(self, command: str = "claude", model: str = "sonnet"):\n'),
    write(S, '        self.command = command\n        self.model = model\n'),
    write(S, '        self._on_status = None\n\n'),
    write(S, '    def send_message(self, message: str, context: list[dict], **kwargs) -> AgentResponse:\n'),
    write(S, '        """Send message to Claude Code CLI with live streaming output."""\n'),
    write(S, '        prompt = self._format_prompt(message, context)\n'),
    write(S, '        self._on_status = kwargs.get(\'on_status\')\n\n'),
    write(S, '        cmd = [\n            self.command,\n            "-p",\n            "--verbose",\n'),
    write(S, '            "--output-format", "stream-json",\n            "--model", self.model,\n'),
    write(S, '            prompt\n        ]\n\n        try:\n'),
    write_py(S, popen_setup),
    write_py(S, stream_json_parser_claude_code),
    write(S, '\n'),
    write(S, '        except subprocess.TimeoutExpired:\n            proc.kill()\n'),
    write(S, '            return AgentResponse(content="[Error: Command timed out]", tokens={})\n'),
    write(S, '        except FileNotFoundError:\n'),
    write(S, '            return AgentResponse(\n'),
    write(S, '                content=f"[Error: Command \'{self.command}\' not found. Install with: npm install -g @anthropic-ai/claude-code]",\n'),
    write(S, '                tokens={})\n'),
    write(S, '        except Exception as e:\n'),
    write(S, '            return AgentResponse(content=f"[Error: {e}]", tokens={})\n\n'),
    write_py(S, stream_json_return), write(S, '\n'),
    write_py(S, describe_tool_call_claude_code), write(S, '\n'),
    write_py(S, format_prompt), write(S, '\n'),
    write_py(S, name_property, [display_name='Claude Code ({self.model})']).

%% --- claude_api (uses: sdk_import_guard, api_key_from_env, messages_builder_anthropic, extract_tool_calls_anthropic) ---

generate_backend_full(S, claude_api, Props) :-
    member(description(Desc), Props),
    format(S, '"""~w"""~n~n', [Desc]),
    write(S, 'import os\nfrom .base import AgentBackend, AgentResponse, ToolCall\n\n'),
    write_py(S, sdk_import_guard, [sdk=anthropic, 'SDK_UPPER'='ANTHROPIC']),
    write(S, '\nclass ClaudeAPIBackend(AgentBackend):\n'),
    write(S, '    """Anthropic Claude API backend."""\n\n'),
    write(S, '    def __init__(\n        self,\n        api_key: str | None = None,\n'),
    write(S, '        model: str = "claude-sonnet-4-20250514",\n        max_tokens: int = 4096,\n'),
    write(S, '        system_prompt: str | None = None\n    ):\n'),
    write_py(S, sdk_init_guard, ['SDK_UPPER'='ANTHROPIC', sdk=anthropic]),
    write_py(S, api_key_from_env, [env_var='ANTHROPIC_API_KEY']),
    write(S, '\n        self.model = model\n        self.max_tokens = max_tokens\n'),
    write(S, '        self.system_prompt = system_prompt or (\n'),
    write(S, '            "You are a helpful AI assistant. Be concise and direct."\n        )\n\n'),
    write(S, '        self.client = anthropic.Anthropic(api_key=self.api_key)\n\n'),
    write(S, '    def send_message(self, message: str, context: list[dict], **kwargs) -> AgentResponse:\n'),
    write(S, '        """Send message to Claude API and get response."""\n'),
    write_py(S, messages_builder_anthropic),
    write(S, '\n        try:\n'),
    write(S, '            response = self.client.messages.create(\n'),
    write(S, '                model=self.model,\n                max_tokens=self.max_tokens,\n'),
    write(S, '                system=self.system_prompt,\n                messages=messages\n            )\n\n'),
    write(S, '            content = ""\n            for block in response.content:\n'),
    write(S, '                if hasattr(block, \'text\'):\n                    content += block.text\n\n'),
    write(S, '            tokens = {}\n            if hasattr(response, \'usage\'):\n'),
    write(S, '                tokens = {\n                    \'input\': response.usage.input_tokens,\n'),
    write(S, '                    \'output\': response.usage.output_tokens\n                }\n\n'),
    write(S, '            tool_calls = self._extract_tool_calls(response)\n\n'),
    write(S, '            return AgentResponse(\n                content=content,\n'),
    write(S, '                tool_calls=tool_calls,\n                tokens=tokens,\n'),
    write(S, '                raw=response\n            )\n\n'),
    write_py(S, error_handler_api_sdk, [sdk=anthropic]),
    write(S, '\n'),
    write_py(S, extract_tool_calls_anthropic),
    write(S, '\n'),
    %% Streaming method
    write(S, '    def send_message_streaming(\n        self,\n        message: str,\n'),
    write(S, '        context: list[dict],\n        on_token: callable = None\n    ) -> AgentResponse:\n'),
    write(S, '        """Send message with streaming response.\n\n'),
    write(S, '        Args:\n            message: The message to send\n'),
    write(S, '            context: Conversation context\n'),
    write(S, '            on_token: Callback called for each token chunk (str) -> None\n        """\n'),
    write(S, '        messages = []\n        for msg in context:\n'),
    write(S, '            if msg.get(\'role\') in (\'user\', \'assistant\'):\n'),
    write(S, '                messages.append({\n                    "role": msg[\'role\'],\n'),
    write(S, '                    "content": msg[\'content\']\n                })\n'),
    write(S, '        # Note: current message is already in context (added by agent_loop)\n\n'),
    write(S, '        try:\n            content_parts = []\n'),
    write(S, '            input_tokens = 0\n            output_tokens = 0\n\n'),
    write(S, '            with self.client.messages.stream(\n                model=self.model,\n'),
    write(S, '                max_tokens=self.max_tokens,\n                system=self.system_prompt,\n'),
    write(S, '                messages=messages\n            ) as stream:\n'),
    write(S, '                for text in stream.text_stream:\n                    content_parts.append(text)\n'),
    write(S, '                    if on_token:\n                        on_token(text)\n\n'),
    write(S, '                response = stream.get_final_message()\n'),
    write(S, '                if hasattr(response, \'usage\'):\n'),
    write(S, '                    input_tokens = response.usage.input_tokens\n'),
    write(S, '                    output_tokens = response.usage.output_tokens\n\n'),
    write(S, '            content = "".join(content_parts)\n'),
    write(S, '            return AgentResponse(\n                content=content,\n'),
    write(S, '                tool_calls=[],\n'),
    write(S, '                tokens={\'input\': input_tokens, \'output\': output_tokens},\n'),
    write(S, '                raw=response\n            )\n\n'),
    write_py(S, error_handler_api_sdk, [sdk=anthropic]),
    write(S, '\n'),
    write_py(S, supports_streaming_true, [backend_name='Claude API']),
    write(S, '\n'),
    write_py(S, name_property, [display_name='Claude API ({self.model})']).

%% --- openai_api (uses: sdk_import_guard, api_key_from_env, messages_builder_system, extract_tool_calls_openai) ---

generate_backend_full(S, openai_api, Props) :-
    member(description(Desc), Props),
    format(S, '"""~w"""~n~n', [Desc]),
    write(S, 'import os\nfrom .base import AgentBackend, AgentResponse, ToolCall\n\n'),
    write_py(S, sdk_import_guard, [sdk=openai, 'SDK_UPPER'='OPENAI']),
    write(S, '\nclass OpenAIBackend(AgentBackend):\n'),
    write(S, '    """OpenAI API backend (GPT-4, GPT-3.5, etc.)."""\n\n'),
    write(S, '    def __init__(\n        self,\n        api_key: str | None = None,\n'),
    write(S, '        model: str = "gpt-4o",\n        max_tokens: int = 4096,\n'),
    write(S, '        system_prompt: str | None = None,\n        base_url: str | None = None\n    ):\n'),
    write_py(S, sdk_init_guard, ['SDK_UPPER'='OPENAI', sdk=openai]),
    write_py(S, api_key_from_env, [env_var='OPENAI_API_KEY']),
    write(S, '\n        self.model = model\n        self.max_tokens = max_tokens\n'),
    write(S, '        self.system_prompt = system_prompt or (\n'),
    write(S, '            "You are a helpful AI assistant. Be concise and direct."\n        )\n'),
    write(S, '        self.base_url = base_url\n\n'),
    write(S, '        client_kwargs = {"api_key": self.api_key}\n'),
    write(S, '        if self.base_url:\n            client_kwargs["base_url"] = self.base_url\n\n'),
    write(S, '        self.client = openai.OpenAI(**client_kwargs)\n\n'),
    write(S, '    def send_message(self, message: str, context: list[dict], **kwargs) -> AgentResponse:\n'),
    write(S, '        """Send message to OpenAI API and get response."""\n'),
    write_py(S, messages_builder_system),
    write(S, '\n        # Note: current message is already in context (added by agent_loop)\n\n'),
    write(S, '        try:\n'),
    write(S, '            response = self.client.chat.completions.create(\n'),
    write(S, '                model=self.model,\n                max_tokens=self.max_tokens,\n'),
    write(S, '                messages=messages\n            )\n\n'),
    write(S, '            content = ""\n            if response.choices and len(response.choices) > 0:\n'),
    write(S, '                choice = response.choices[0]\n'),
    write(S, '                if choice.message and choice.message.content:\n'),
    write(S, '                    content = choice.message.content\n\n'),
    write(S, '            tokens = {}\n            if response.usage:\n'),
    write(S, '                tokens = {\n                    \'input\': response.usage.prompt_tokens,\n'),
    write(S, '                    \'output\': response.usage.completion_tokens,\n'),
    write(S, '                    \'total\': response.usage.total_tokens\n                }\n\n'),
    write(S, '            tool_calls = self._extract_tool_calls(response)\n\n'),
    write(S, '            return AgentResponse(\n                content=content,\n'),
    write(S, '                tool_calls=tool_calls,\n                tokens=tokens,\n'),
    write(S, '                raw=response\n            )\n\n'),
    write_py(S, error_handler_api_sdk, [sdk=openai]),
    write(S, '\n'),
    write_py(S, extract_tool_calls_openai),
    write(S, '\n'),
    %% Streaming method
    write(S, '    def send_message_streaming(\n        self,\n        message: str,\n'),
    write(S, '        context: list[dict],\n        on_token: callable = None\n    ) -> AgentResponse:\n'),
    write(S, '        """Send message with streaming response.\n\n'),
    write(S, '        Args:\n            message: The message to send\n'),
    write(S, '            context: Conversation context\n'),
    write(S, '            on_token: Callback called for each token chunk (str) -> None\n        """\n'),
    write(S, '        messages = [{"role": "system", "content": self.system_prompt}]\n'),
    write(S, '        for msg in context:\n'),
    write(S, '            if msg.get(\'role\') in (\'user\', \'assistant\'):\n'),
    write(S, '                messages.append({\n                    "role": msg[\'role\'],\n'),
    write(S, '                    "content": msg[\'content\']\n                })\n'),
    write(S, '        messages.append({"role": "user", "content": message})\n\n'),
    write(S, '        try:\n            content_parts = []\n\n'),
    write(S, '            stream = self.client.chat.completions.create(\n'),
    write(S, '                model=self.model,\n                max_tokens=self.max_tokens,\n'),
    write(S, '                messages=messages,\n                stream=True,\n'),
    write(S, '                stream_options={"include_usage": True}\n            )\n\n'),
    write(S, '            input_tokens = 0\n            output_tokens = 0\n\n'),
    write(S, '            for chunk in stream:\n'),
    write(S, '                if chunk.choices and len(chunk.choices) > 0:\n'),
    write(S, '                    delta = chunk.choices[0].delta\n'),
    write(S, '                    if delta and delta.content:\n'),
    write(S, '                        content_parts.append(delta.content)\n'),
    write(S, '                        if on_token:\n                            on_token(delta.content)\n\n'),
    write(S, '                if chunk.usage:\n'),
    write(S, '                    input_tokens = chunk.usage.prompt_tokens\n'),
    write(S, '                    output_tokens = chunk.usage.completion_tokens\n\n'),
    write(S, '            content = "".join(content_parts)\n'),
    write(S, '            return AgentResponse(\n                content=content,\n'),
    write(S, '                tool_calls=[],\n'),
    write(S, '                tokens={\'input\': input_tokens, \'output\': output_tokens},\n'),
    write(S, '                raw=None\n            )\n\n'),
    write_py(S, error_handler_api_sdk, [sdk=openai]),
    write(S, '\n'),
    write_py(S, supports_streaming_true, [backend_name='OpenAI API']),
    write(S, '\n'),
    write_py(S, name_property, [display_name='OpenAI ({self.model})']).

%% --- coro (uses: format_prompt, name_property — mostly unique imperative logic) ---

generate_backend_full(S, coro, Props) :-
    member(description(Desc), Props),
    format(S, '"""~w"""~n~n', [Desc]),
    write(S, 'import json as _json\nimport os\nimport subprocess\nimport re\nimport tempfile\n'),
    write(S, 'from .base import AgentBackend, AgentResponse, ToolCall\n\n\n'),
    write(S, 'class CoroBackend(AgentBackend):\n'),
    write(S, '    """Coro-code CLI backend using single-task mode."""\n\n'),
    write(S, '    def __init__(self, command: str = "coro", verbose: bool = True,\n'),
    write(S, '                 debug: bool = True, max_steps: int = 0,\n'),
    write(S, '                 config: str | None = None, no_fallback: bool = False,\n'),
    write(S, '                 max_context_tokens: int = 0):\n'),
    write(S, '        self.command = command\n        self.verbose = verbose\n'),
    write(S, '        self.debug = debug\n        self.max_steps = max_steps\n'),
    write(S, '        self.config = config or self._find_config(no_fallback=no_fallback)\n'),
    write(S, '        self.max_context_tokens = max_context_tokens\n'),
    write(S, '        self._temp_config = None\n'),
    write(S, '        self.model = self._read_model_from_config()\n\n'),
    write(S, '        if self.max_context_tokens > 0:\n'),
    write(S, '            self._temp_config = self._create_limited_config()\n'),
    write(S, '        self.token_pattern = re.compile(\n'),
    write(S, '            r\'(?:Input|Output|Total):\\s*([\\d,]+)\\s*tokens?\',\n'),
    write(S, '            re.IGNORECASE\n        )\n'),
    write(S, '        self._coro_token_pattern = re.compile(\n'),
    write(S, '            r\'Tokens:\\s*(\\d+)\\s*input\\s*\\+\\s*(\\d+)\\s*output\\s*=\\s*(\\d+)\\s*total\'\n        )\n'),
    write(S, '        self._duration_pattern = re.compile(r\'Duration:\\s*([\\d.]+)s\')\n'),
    write(S, '        self._ansi_pattern = re.compile(r\'\\x1b\\[[0-9;]*[A-Za-z]\')\n\n'),
    %% Config discovery methods
    write(S, '    def _find_config(self, no_fallback: bool = False) -> str | None:\n'),
    write(S, '        """Find coro-native config (coro.json only)."""\n'),
    write(S, '        if os.path.isfile(\'coro.json\'):\n            return None\n'),
    write(S, '        if no_fallback:\n            return None\n'),
    write(S, '        path = os.path.expanduser(\'~/coro.json\')\n'),
    write(S, '        if os.path.isfile(path):\n            return path\n'),
    write(S, '        return None\n\n'),
    write(S, '    def _read_coro_config(self) -> dict:\n'),
    write(S, '        """Read the full coro config from the best available path."""\n'),
    write(S, '        candidates = [self.config] if self.config else []\n'),
    write(S, '        for name in [\'uwsal.json\', \'coro.json\']:\n'),
    write(S, '            if os.path.isfile(name):\n                candidates.append(name)\n'),
    write(S, '            home = os.path.expanduser(f\'~/{name}\')\n'),
    write(S, '            if os.path.isfile(home):\n                candidates.append(home)\n'),
    write(S, '        for path in candidates:\n            try:\n'),
    write(S, '                with open(path) as f:\n                    return _json.load(f)\n'),
    write(S, '            except Exception:\n                continue\n'),
    write(S, '        return {}\n\n'),
    write(S, '    def _read_model_from_config(self) -> str:\n'),
    write(S, '        """Read model name from coro config file."""\n'),
    write(S, '        return self._read_coro_config().get(\'model\', \'unknown\')\n\n'),
    write(S, '    def _create_limited_config(self) -> str | None:\n'),
    write(S, '        """Create a temp coro.json with max_token set to limit context."""\n'),
    write(S, '        base = self._read_coro_config()\n'),
    write(S, '        if not base:\n            return None\n'),
    write(S, '        base[\'max_token\'] = self.max_context_tokens\n'),
    write(S, '        try:\n            tf = tempfile.NamedTemporaryFile(\n'),
    write(S, '                mode=\'w\', suffix=\'.json\', prefix=\'coro_\',\n'),
    write(S, '                delete=False\n            )\n'),
    write(S, '            _json.dump(base, tf, indent=2)\n            tf.close()\n'),
    write(S, '            return tf.name\n        except Exception:\n            return None\n\n'),
    %% send_message
    write(S, '    def send_message(self, message: str, context: list[dict], **kwargs) -> AgentResponse:\n'),
    write(S, '        """Send message to coro CLI and get response."""\n'),
    write(S, '        prompt = self._format_prompt(message, context)\n\n'),
    write(S, '        cmd = [self.command]\n'),
    write(S, '        config_path = self._temp_config or self.config\n'),
    write(S, '        if config_path:\n            cmd.extend([\'-c\', config_path])\n'),
    write(S, '        if self.debug:\n            cmd.append(\'--debug\')\n'),
    write(S, '        elif self.verbose:\n            cmd.append(\'--verbose\')\n'),
    write(S, '        if self.max_steps > 0:\n'),
    write(S, '            cmd.extend([\'--max-steps\', str(self.max_steps)])\n'),
    write(S, '        cmd.append(prompt)\n\n'),
    write(S, '        try:\n            result = subprocess.run(\n                cmd,\n'),
    write(S, '                capture_output=True,\n                text=True,\n'),
    write(S, '                timeout=300\n            )\n'),
    write(S, '            output = result.stdout\n'),
    write(S, '            if result.stderr:\n                output += "\\n" + result.stderr\n\n'),
    write(S, '        except subprocess.TimeoutExpired:\n'),
    write(S, '            return AgentResponse(content="[Error: Command timed out after 5 minutes]", tokens={})\n'),
    write(S, '        except FileNotFoundError:\n'),
    write(S, '            return AgentResponse(content=f"[Error: Command \'{self.command}\' not found]", tokens={})\n'),
    write(S, '        except Exception as e:\n'),
    write(S, '            return AgentResponse(content=f"[Error: {e}]", tokens={})\n\n'),
    write(S, '        content = self._clean_output(output)\n'),
    write(S, '        tokens = self._parse_tokens(output)\n'),
    write(S, '        tool_calls = self.parse_tool_calls(output)\n\n'),
    write(S, '        return AgentResponse(\n            content=content,\n'),
    write(S, '            tool_calls=tool_calls,\n            tokens=tokens,\n'),
    write(S, '            raw=output\n        )\n\n'),
    write(S, '    def _strip_ansi(self, text: str) -> str:\n'),
    write(S, '        """Remove ANSI escape codes from text."""\n'),
    write(S, '        return self._ansi_pattern.sub(\'\', text)\n\n'),
    write_py(S, format_prompt), write(S, '\n'),
    %% _clean_output (coro-specific, too unique for a generic fragment)
    write(S, '    def _clean_output(self, output: str) -> str:\n'),
    write(S, '        """Clean up coro output, removing noise."""\n'),
    write(S, '        lines = output.split(\'\\n\')\n        cleaned = []\n        blank_count = 0\n\n'),
    write(S, '        for line in lines:\n            plain = self._strip_ansi(line)\n'),
    write(S, '            if self.token_pattern.search(plain):\n                continue\n'),
    write(S, '            if self._coro_token_pattern.search(plain):\n                continue\n'),
    write(S, '            if self._duration_pattern.search(plain):\n                continue\n'),
    write(S, '            if re.match(r\'^\\s*\\d{4}-\\d{2}-\\d{2}T\', plain):\n                continue\n'),
    write(S, '            stripped = plain.strip()\n'),
    write(S, '            if stripped and all(c in \'[0123456789ABCDJKHfm;\' for c in stripped):\n'),
    write(S, '                continue\n'),
    write(S, '            if stripped in [\'Previous conversation:\', \'Current request:\'] or \\\n'),
    write(S, '               stripped.startswith(\'Current request: \') or \\\n'),
    write(S, '               stripped.startswith(\'User: \') or \\\n'),
    write(S, '               stripped.startswith(\'Assistant: \'):\n                continue\n'),
    write(S, '            if stripped in [\'\', \'...\', \''),
    write(S, '\u2500'),
    write(S, '\' * 10]:\n                blank_count += 1\n'),
    write(S, '                if blank_count <= 1:\n                    cleaned.append(line)\n'),
    write(S, '                continue\n\n            blank_count = 0\n            cleaned.append(line)\n\n'),
    write(S, '        result = \'\\n\'.join(cleaned).strip()\n'),
    write(S, '        result = self._strip_ansi(result)\n\n'),
    write(S, '        for prefix in [\'A:\', \'Assistant:\', \'Claude:\']:\n'),
    write(S, '            if result.startswith(prefix):\n'),
    write(S, '                result = result[len(prefix):].strip()\n                break\n\n'),
    write(S, '        return result\n\n'),
    %% _parse_tokens
    write(S, '    def _parse_tokens(self, output: str) -> dict[str, int]:\n'),
    write(S, '        """Extract token counts from output."""\n'),
    write(S, '        plain = self._strip_ansi(output)\n        tokens = {}\n\n'),
    write(S, '        match = self._coro_token_pattern.search(plain)\n'),
    write(S, '        if match:\n'),
    write(S, '            tokens[\'input\'] = int(match.group(1))\n'),
    write(S, '            tokens[\'output\'] = int(match.group(2))\n'),
    write(S, '            tokens[\'total\'] = int(match.group(3))\n'),
    write(S, '            dur = self._duration_pattern.search(plain)\n'),
    write(S, '            if dur:\n                tokens[\'duration\'] = float(dur.group(1))\n'),
    write(S, '            return tokens\n\n'),
    write(S, '        for match in self.token_pattern.finditer(plain):\n'),
    write(S, '            line = plain[max(0, match.start()-20):match.end()]\n'),
    write(S, '            count = int(match.group(1).replace(\',\', \'\'))\n\n'),
    write(S, '            if \'input\' in line.lower():\n'),
    write(S, '                tokens[\'input\'] = tokens.get(\'input\', 0) + count\n'),
    write(S, '            elif \'output\' in line.lower():\n'),
    write(S, '                tokens[\'output\'] = tokens.get(\'output\', 0) + count\n'),
    write(S, '            elif \'total\' in line.lower():\n'),
    write(S, '                tokens[\'total\'] = count\n\n'),
    write(S, '        return tokens\n\n'),
    write(S, '    def parse_tool_calls(self, response: str) -> list[ToolCall]:\n'),
    write(S, '        """Extract tool calls from coro output."""\n'),
    write(S, '        return []\n\n'),
    write_py(S, name_property, [display_name='Coro ({self.command})']).

%% --- openrouter_api (uses: messages_builder_openrouter, describe_tool_call_openrouter, sse_streaming_openrouter) ---

generate_backend_full(S, openrouter_api, Props) :-
    member(description(Desc), Props),
    format(S, '"""~w"""~n~n', [Desc]),
    write(S, 'import json\nimport os\nimport sys\n'),
    write(S, 'from urllib.request import urlopen, Request\n'),
    write(S, 'from urllib.error import HTTPError, URLError\n'),
    write(S, 'from .base import AgentBackend, AgentResponse, ToolCall\n\n\n'),
    %% Generate DEFAULT_TOOL_SCHEMAS from tool_spec/2 facts
    write(S, '# Default tool schemas for function calling (OpenAI format)\n'),
    write(S, 'DEFAULT_TOOL_SCHEMAS = [\n'),
    forall(tool_spec(ToolName, ToolProps), (
        generate_tool_schema_py(S, ToolName, ToolProps)
    )),
    write(S, ']\n\n\n'),
    write(S, 'class OpenRouterBackend(AgentBackend):\n'),
    write(S, '    """OpenRouter API backend (OpenAI-compatible, no pip deps)."""\n\n'),
    write(S, '    def __init__(\n        self,\n        api_key: str | None = None,\n'),
    write(S, '        model: str | None = None,\n'),
    write(S, '        base_url: str = "https://openrouter.ai/api/v1",\n'),
    write(S, '        max_tokens: int = 4096,\n        temperature: float = 0.7,\n'),
    write(S, '        system_prompt: str | None = None,\n'),
    write(S, '        tools: list[dict] | None = None,\n    ):\n'),
    write(S, '        self.api_key = api_key\n'),
    write(S, '        self.model = model or \'moonshotai/kimi-k2.5\'\n'),
    write(S, '        self.base_url = base_url or \'https://openrouter.ai/api/v1\'\n'),
    write(S, '        self.max_tokens = max_tokens\n        self.temperature = temperature\n'),
    write(S, '        self.system_prompt = system_prompt or (\n'),
    write(S, '            "You are a helpful AI coding assistant. "\n'),
    write(S, '            "Answer questions directly and concisely. "\n'),
    write(S, '            "When asked to perform tasks, use the available tools."\n        )\n'),
    write(S, '        self.tool_schemas = tools\n\n'),
    write(S, '        if not self.api_key:\n            raise ValueError(\n'),
    write(S, '                "OpenRouter API key required. Set OPENROUTER_API_KEY, "\n'),
    write(S, '                "provide --api-key, or add api_key to uwsal.json / coro.json."\n            )\n\n'),
    write(S, '    def send_message(self, message: str, context: list[dict], **kwargs) -> AgentResponse:\n'),
    write(S, '        """Send message to OpenRouter API."""\n'),
    write(S, '        on_status = kwargs.get(\'on_status\')\n\n'),
    write_py(S, messages_builder_openrouter),
    write(S, '\n        body = {\n            "model": self.model,\n'),
    write(S, '            "messages": messages,\n            "max_tokens": self.max_tokens,\n'),
    write(S, '            "temperature": self.temperature,\n        }\n\n'),
    write(S, '        if self.tool_schemas:\n            body["tools"] = self.tool_schemas\n'),
    write(S, '            body["tool_choice"] = "auto"\n\n'),
    write(S, '        url = f"{self.base_url.rstrip(\'/\')}/chat/completions"\n'),
    write(S, '        req_data = json.dumps(body).encode(\'utf-8\')\n\n'),
    write(S, '        req = Request(\n            url,\n            data=req_data,\n'),
    write(S, '            headers={\n                \'Content-Type\': \'application/json\',\n'),
    write(S, '                \'Authorization\': f\'Bearer {self.api_key}\',\n'),
    write(S, '                \'HTTP-Referer\': \'https://github.com/s243a/UnifyWeaver\',\n'),
    write(S, '                \'X-Title\': \'UnifyWeaver Agent Loop\',\n'),
    write(S, '            },\n            method=\'POST\'\n        )\n\n'),
    write(S, '        if on_status:\n            on_status("Waiting for response...")\n\n'),
    write(S, '        try:\n            with urlopen(req, timeout=300) as resp:\n'),
    write(S, '                data = json.loads(resp.read().decode(\'utf-8\'))\n'),
    write_py(S, error_handler_urllib),
    write(S, '\n        content = ""\n        tool_calls = []\n        tokens = {}\n\n'),
    write(S, '        if data.get(\'choices\'):\n            choice = data[\'choices\'][0]\n'),
    write(S, '            msg = choice.get(\'message\', {})\n\n'),
    write(S, '            content = msg.get(\'content\') or \'\'\n\n'),
    write(S, '            for tc in msg.get(\'tool_calls\', []):\n'),
    write(S, '                if tc.get(\'type\') == \'function\':\n'),
    write(S, '                    func = tc.get(\'function\', {})\n'),
    write(S, '                    try:\n'),
    write(S, '                        arguments = json.loads(func.get(\'arguments\', \'{}\'))\n'),
    write(S, '                    except json.JSONDecodeError:\n'),
    write(S, '                        arguments = {"raw": func.get(\'arguments\', \'\')}\n'),
    write(S, '                    tool_calls.append(ToolCall(\n'),
    write(S, '                        name=func.get(\'name\', \'\'),\n'),
    write(S, '                        arguments=arguments,\n'),
    write(S, '                        id=tc.get(\'id\', \'\')\n                    ))\n'),
    write(S, '                    if on_status:\n'),
    write(S, '                        desc = self._describe_tool_call(\n'),
    write(S, '                            func.get(\'name\', \'?\'), arguments)\n'),
    write(S, '                        on_status(f"[{len(tool_calls)}] {desc}")\n\n'),
    write(S, '        usage = data.get(\'usage\', {})\n        if usage:\n'),
    write(S, '            tokens = {\n                \'input\': usage.get(\'prompt_tokens\', 0),\n'),
    write(S, '                \'output\': usage.get(\'completion_tokens\', 0),\n'),
    write(S, '                \'total\': usage.get(\'total_tokens\', 0),\n            }\n\n'),
    write(S, '        return AgentResponse(\n            content=content,\n'),
    write(S, '            tool_calls=tool_calls,\n            tokens=tokens,\n'),
    write(S, '            raw=data\n        )\n\n'),
    write_py(S, supports_streaming_true, [backend_name='OpenRouter']),
    write(S, '\n'),
    write_py(S, sse_streaming_openrouter),
    write(S, '\n'),
    write_py(S, describe_tool_call_openrouter),
    write(S, '\n'),
    write_py(S, name_property, [display_name='OpenRouter ({self.model})']).

%% =============================================================================
%% Tool Schema Generation (OpenAI function calling format)
%% =============================================================================

%% Generate a single tool schema entry in OpenAI format from tool_spec/2 facts
generate_tool_schema_py(S, ToolName, ToolProps) :-
    member(description(Desc), ToolProps),
    member(parameters(Params), ToolProps),
    write(S, '    {\n'),
    write(S, '        "type": "function",\n'),
    write(S, '        "function": {\n'),
    format(S, '            "name": "~w",~n', [ToolName]),
    format(S, '            "description": "~w",~n', [Desc]),
    write(S, '            "parameters": {\n'),
    write(S, '                "type": "object",\n'),
    write(S, '                "properties": {\n'),
    generate_tool_schema_properties(S, Params),
    write(S, '                },\n'),
    write(S, '                "required": ['),
    generate_tool_schema_required(S, Params),
    write(S, ']\n'),
    write(S, '            }\n'),
    write(S, '        }\n'),
    write(S, '    },\n').

generate_tool_schema_properties(_, []).
generate_tool_schema_properties(S, [param(Name, Type, _, Desc)]) :-
    format(S, '                    "~w": {~n', [Name]),
    format(S, '                        "type": "~w",~n', [Type]),
    format(S, '                        "description": "~w"~n', [Desc]),
    write(S, '                    }\n').
generate_tool_schema_properties(S, [param(Name, Type, _, Desc)|Rest]) :-
    Rest \= [],
    format(S, '                    "~w": {~n', [Name]),
    format(S, '                        "type": "~w",~n', [Type]),
    format(S, '                        "description": "~w"~n', [Desc]),
    write(S, '                    },\n'),
    generate_tool_schema_properties(S, Rest).

generate_tool_schema_required(_, []).
generate_tool_schema_required(S, [param(Name, _, required, _)]) :-
    format(S, '"~w"', [Name]).
generate_tool_schema_required(S, [param(Name, _, required, _)|Rest]) :-
    Rest \= [],
    format(S, '"~w", ', [Name]),
    generate_tool_schema_required(S, Rest).
generate_tool_schema_required(S, [param(_, _, Optional, _)|Rest]) :-
    Optional \= required,
    generate_tool_schema_required(S, Rest).

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
