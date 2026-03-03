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

:- dynamic pl_test_passed/1, pl_test_failed/1.

run_prolog_tests :-
    retractall(pl_test_passed(_)),
    retractall(pl_test_failed(_)),
    format("~n=== Prolog Integration Tests ===~n~n"),
    test_costs_module,
    test_tools_module,
    test_commands_module,
    test_security_module,
    test_backends_module,
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
    pl_assert_eq('slash_command count', CmdCount, 23),
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
