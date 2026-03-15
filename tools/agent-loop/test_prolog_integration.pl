%% test_prolog_integration.pl — Integration tests for generated Prolog modules
%%
%% Loads each generated .pl module and verifies fact counts, key values,
%% and cross-module consistency.
%%
%% Usage:
%%   cd tools/agent-loop && swipl -l test_prolog_integration.pl -g "run_prolog_tests, halt"

:- module(test_prolog_integration, [run_prolog_tests/0]).

:- use_module(generated/prolog/costs).
:- use_module(generated/prolog/tools).
:- use_module(generated/prolog/commands).
:- use_module(generated/prolog/security).
:- use_module(generated/prolog/backends).
:- use_module(generated/prolog/config).

:- dynamic pl_test_passed/1, pl_test_failed/1.

%% tmp_file_name(+Name, -Path) — get a writable temp file path
tmp_file_name(Name, Path) :-
    (getenv('TMPDIR', TmpDir) -> true
    ; getenv('HOME', Home) -> atom_concat(Home, '/tmp', TmpDir)
    ; TmpDir = '/tmp'),
    (exists_directory(TmpDir) -> true ; make_directory_path(TmpDir)),
    format(atom(Path), "~w/~w", [TmpDir, Name]).

run_prolog_tests :-
    retractall(pl_test_passed(_)),
    retractall(pl_test_failed(_)),
    format("~n=== Prolog Integration Tests ===~n~n"),
    test_costs_module,
    test_tools_module,
    test_commands_module,
    test_security_module,
    test_backends_module,
    test_cost_tracker_runtime,
    test_config_runtime,
    test_plugin_system,
    %% Report
    aggregate_all(count, pl_test_passed(_), Passed),
    aggregate_all(count, pl_test_failed(_), Failed),
    format("~n=== Prolog Integration: ~w passed, ~w failed ===~n", [Passed, Failed]),
    (Failed > 0 -> halt(1) ; true).

%% Assert helper
pl_assert_true(Name, Goal) :-
    (call(Goal) ->
        assert(pl_test_passed(Name)),
        format("  [PASS] ~w~n", [Name])
    ;
        assert(pl_test_failed(Name)),
        format("  [FAIL] ~w~n", [Name])
    ).

pl_assert_eq(Name, Got, Expected) :-
    (Got == Expected ->
        assert(pl_test_passed(Name)),
        format("  [PASS] ~w~n", [Name])
    ;
        assert(pl_test_failed(Name)),
        format("  [FAIL] ~w (expected ~w, got ~w)~n", [Name, Expected, Got])
    ).

%% ============================================================================
%% costs.pl
%% ============================================================================

test_costs_module :-
    format("costs.pl:~n"),
    findall(M, costs:model_pricing(M, _, _), Models),
    length(Models, Count),
    pl_assert_eq('model_pricing count', Count, 16),
    pl_assert_true('opus pricing correct', (
        costs:model_pricing("opus", 15.0, 75.0)
    )),
    pl_assert_true('free model pricing', (
        costs:model_pricing("llama3", 0.0, 0.0)
    )).

%% ============================================================================
%% tools.pl
%% ============================================================================

test_tools_module :-
    format("~ntools.pl:~n"),
    findall(N, tools:tool_spec(N, _), Specs),
    length(Specs, SpecCount),
    pl_assert_eq('tool_spec count', SpecCount, 4),
    findall(N, tools:tool_handler(N, _), Handlers),
    length(Handlers, HCount),
    pl_assert_eq('tool_handler count', HCount, 4),
    findall(N, tools:destructive_tool(N), Destr),
    length(Destr, DCount),
    pl_assert_eq('destructive_tool count', DCount, 3),
    pl_assert_true('bash has timeout 120', (
        tools:tool_spec(bash, Props),
        member(timeout(120), Props)
    )).

%% ============================================================================
%% commands.pl
%% ============================================================================

test_commands_module :-
    format("~ncommands.pl:~n"),
    findall(N, commands:slash_command(N, _, _, _), Cmds),
    length(Cmds, CmdCount),
    pl_assert_eq('slash_command count', CmdCount, 24),
    findall(A, commands:command_alias(A, _), Aliases),
    length(Aliases, ACount),
    pl_assert_eq('command_alias count', ACount, 30),
    findall(G, commands:slash_command_group(G, _), Groups),
    length(Groups, GCount),
    pl_assert_eq('slash_command_group count', GCount, 6).

%% ============================================================================
%% security.pl
%% ============================================================================

test_security_module :-
    format("~nsecurity.pl:~n"),
    findall(N, security:security_profile(N, _), Profiles),
    length(Profiles, PCount),
    pl_assert_eq('security_profile count', PCount, 4),
    findall(P, security:blocked_path(P), Paths),
    length(Paths, PathCount),
    pl_assert_eq('blocked_path count', PathCount, 3),
    findall(R, security:blocked_command_pattern(R, _), Patterns),
    length(Patterns, PatCount),
    pl_assert_eq('blocked_command_pattern count', PatCount, 9),
    pl_assert_true('cautious has path_validation true', (
        security:security_profile(cautious, CProps),
        member(path_validation(true), CProps)
    )).

%% ============================================================================
%% backends.pl
%% ============================================================================

test_backends_module :-
    format("~nbackends.pl:~n"),
    findall(N, backends:agent_backend(N, _), Backends),
    length(Backends, BCount),
    pl_assert_eq('agent_backend count', BCount, 8),
    findall(N, backends:backend_factory(N, _), Factories),
    length(Factories, FCount),
    pl_assert_eq('backend_factory count', FCount, 8),
    pl_assert_true('backend_factory_order has 8', (
        backends:backend_factory_order(Order),
        length(Order, 8)
    )).

%% ============================================================================
%% Runtime tests: cost_tracker
%% ============================================================================

test_cost_tracker_runtime :-
    format("~ncost_tracker runtime:~n"),
    %% Init a test tracker
    costs:cost_tracker_init(test_runtime),
    pl_assert_true('cost_tracker_init creates zero state', (
        costs:cost_tracker_total(test_runtime, tokens(0, 0))
    )),
    %% Add usage — suppress output
    with_output_to(string(_), (
        costs:cost_tracker_add(test_runtime, "opus", 100, 50)
    )),
    pl_assert_true('cost_tracker_add accumulates tokens', (
        costs:cost_tracker_total(test_runtime, tokens(100, 50))
    )),
    %% Add more
    with_output_to(string(_), (
        costs:cost_tracker_add(test_runtime, "haiku", 200, 100)
    )),
    pl_assert_true('cost_tracker_add accumulates across calls', (
        costs:cost_tracker_total(test_runtime, tokens(300, 150))
    )),
    %% Format
    costs:cost_tracker_format(test_runtime, Formatted),
    pl_assert_true('cost_tracker_format produces readable string', (
        sub_string(Formatted, _, _, _, "300 input"),
        sub_string(Formatted, _, _, _, "150 output")
    )),
    %% Clean up
    costs:cost_tracker_init(test_runtime).

%% ============================================================================
%% Runtime tests: config module
%% ============================================================================

test_config_runtime :-
    format("~nconfig runtime:~n"),
    %% resolve_api_key with explicit key
    config:resolve_api_key(claude, 'test-key-123', Key),
    pl_assert_eq('resolve_api_key explicit returns explicit', Key, 'test-key-123'),
    %% resolve_api_key with no key available (nonexistent backend)
    config:resolve_api_key(nonexistent_backend, none, Key2),
    pl_assert_eq('resolve_api_key fallback returns none', Key2, none),
    %% load_config with nonexistent file
    config:load_config('/tmp/nonexistent_agent_config.json', Config),
    pl_assert_true('load_config nonexistent returns empty dict', is_dict(Config)),
    %% cli_argument fact lookup
    pl_assert_true('cli_argument backend has choices', (
        config:cli_argument(backend, Props),
        member(choices(_), Props)
    )),
    %% default_agent_preset lookup
    pl_assert_true('default_agent_preset yolo uses claude-code', (
        config:default_agent_preset(yolo, 'claude-code', _)
    )),
    %% audit_profile_level lookup
    pl_assert_true('audit_profile_level paranoid is forensic', (
        config:audit_profile_level(paranoid, forensic)
    )).

%% ============================================================================
%% Plugin loading and dispatch tests (Phase 15)
%% ============================================================================

test_plugin_system :-
    format("~nplugin system:~n"),
    %% Plugin tool loading from JSON fixture
    tmp_file_name('uwsal_test_plugin.json', TmpFile),
    PluginJson = '{"name":"test-plugin","version":"1.0","tools":[{"name":"greet","description":"Say hello","parameters":[{"name":"who","param_type":"string"}],"command_template":"echo Hello {who}!"}]}',
    open(TmpFile, write, WS),
    write(WS, PluginJson),
    close(WS),
    pl_assert_true('load_plugin_file succeeds', (
        tools:load_plugin_file(TmpFile)
    )),
    pl_assert_true('plugin_tool registered', (
        tools:plugin_tool(greet, _, _)
    )),
    pl_assert_true('execute_plugin_tool renders template', (
        tools:execute_plugin_tool(greet, _{who: 'World'}, Result),
        Result \= error(_)
    )),
    %% Cleanup
    retractall(tools:plugin_tool(greet, _, _)),
    (exists_file(TmpFile) -> delete_file(TmpFile) ; true),
    %% command_action facts
    pl_assert_true('command_action exit exists', (
        commands:command_action(exit, exit)
    )),
    pl_assert_true('command_action clear exists', (
        commands:command_action(clear, clear)
    )),
    pl_assert_true('command_action help exists', (
        commands:command_action(help, help)
    )),
    pl_assert_true('command_action multiline exists', (
        commands:command_action(multiline, multiline)
    )),
    %% handle_slash_command uses command_action
    pl_assert_true('handle_slash_command exit returns exit action', (
        commands:handle_slash_command(exit, "", Action),
        Action == exit
    )),
    pl_assert_true('handle_slash_command help returns help action', (
        commands:handle_slash_command(help, "", Action2),
        Action2 == help
    )).
