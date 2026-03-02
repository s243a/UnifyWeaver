%% test_agent_loop.pl — Tests for agent-loop overlay modules
%%
%% All tests are safe read-only registry queries — no file I/O, no network.
%%
%% Usage:
%%   swipl -l test_agent_loop.pl -g "run_tests, halt"

:- module(test_agent_loop, [run_tests/0]).

:- use_module(agent_loop_module).
:- use_module(agent_loop_bindings).
:- use_module(agent_loop_components).
:- use_module('../../src/unifyweaver/core/binding_registry').
:- use_module('../../src/unifyweaver/core/component_registry').

%% ============================================================================
%% Test Runner
%% ============================================================================

:- dynamic test_passed/1, test_failed/1.

run_tests :-
    retractall(test_passed(_)),
    retractall(test_failed(_)),
    format("~n=== Agent-Loop Test Suite ===~n~n"),
    %% Clean state
    clear_all_bindings,
    %% Run tests
    test_binding_registration,
    test_component_registration,
    test_module_exports,
    test_binding_queries,
    test_binding_code_generation,
    test_handler_fragment_count,
    test_cli_override_facts,
    test_streaming_capability,
    test_retry_config,
    test_bindings_summary,
    %% Report
    aggregate_all(count, test_passed(_), Passed),
    aggregate_all(count, test_failed(_), Failed),
    format("~n=== Results: ~w passed, ~w failed ===~n", [Passed, Failed]),
    %% Clean up
    clear_all_bindings,
    (Failed > 0 -> halt(1) ; true).

%% Assert helper
assert_true(Name, Goal) :-
    (call(Goal) ->
        assert(test_passed(Name)),
        format("  [PASS] ~w~n", [Name])
    ;
        assert(test_failed(Name)),
        format("  [FAIL] ~w~n", [Name])
    ).

assert_eq(Name, Got, Expected) :-
    (Got == Expected ->
        assert(test_passed(Name)),
        format("  [PASS] ~w~n", [Name])
    ;
        assert(test_failed(Name)),
        format("  [FAIL] ~w (expected ~w, got ~w)~n", [Name, Expected, Got])
    ).

%% ============================================================================
%% Test 1: Binding Registration
%% ============================================================================

test_binding_registration :-
    format("~nBinding registration:~n"),
    init_agent_loop_bindings,
    bindings_for_target(python, PyBindings),
    bindings_for_target(prolog, PlBindings),
    length(PyBindings, NPy),
    length(PlBindings, NPl),
    assert_eq('Python binding count', NPy, 5),
    assert_eq('Prolog binding count', NPl, 5).

%% ============================================================================
%% Test 2: Component Registration
%% ============================================================================

test_component_registration :-
    format("~nComponent registration:~n"),
    register_agent_loop_components,
    list_components(agent_tools, Tools),
    list_components(agent_commands, Commands),
    list_components(agent_backends, Backends),
    length(Tools, NT),
    length(Commands, NC),
    length(Backends, NB),
    assert_eq('Tool count', NT, 4),
    assert_eq('Command count', NC, 21),
    assert_eq('Backend count', NB, 8).

%% ============================================================================
%% Test 3: Module Exports
%% ============================================================================

test_module_exports :-
    format("~nModule exports:~n"),
    assert_true('tool_handler/2 callable', (
        agent_loop_module:tool_handler(read, _)
    )),
    assert_true('slash_command/4 callable', (
        slash_command(help, _, _, _)
    )),
    assert_true('agent_backend/2 callable', (
        agent_backend(coro, _)
    )),
    assert_true('backend_factory/2 callable', (
        backend_factory(coro, _)
    )),
    assert_true('security_profile/2 callable', (
        agent_loop_module:security_profile(open, _)
    )).

%% ============================================================================
%% Test 4: Binding Queries
%% ============================================================================

test_binding_queries :-
    format("~nBinding queries:~n"),
    assert_true('tool_handler is pure', (
        is_pure_binding(python, tool_handler/2)
    )),
    assert_true('backend_factory has io effect', (
        binding_has_effect(python, backend_factory/2, io)
    )),
    assert_true('audit_profile_level is deterministic', (
        is_deterministic_binding(python, audit_profile_level/2)
    )).

%% ============================================================================
%% Test 5: Binding Code Generation
%% ============================================================================

test_binding_code_generation :-
    format("~nBinding code generation:~n"),
    assert_true('Python tool_handler code', (
        compile_binding_code(python, tool_handler/2, Code),
        atom(Code),
        sub_atom(Code, _, _, _, 'TOOL_HANDLERS')
    )),
    assert_true('Prolog tool_handler code', (
        compile_binding_code(prolog, tool_handler/2, PCode),
        atom(PCode),
        sub_atom(PCode, _, _, _, 'tool_handler')
    )),
    assert_true('Python backend_factory has import', (
        compile_binding_code(python, backend_factory/2, BCode),
        atom(BCode),
        sub_atom(BCode, _, _, _, 'import')
    )).

%% ============================================================================
%% Test 6: Handler Fragment Count
%% ============================================================================

test_handler_fragment_count :-
    format("~nHandler fragments:~n"),
    findall(Name, (
        agent_loop_module:py_fragment(Name, _),
        sub_atom(Name, 0, _, _, handler_)
    ), Handlers),
    length(Handlers, Count),
    assert_eq('Handler fragment count', Count, 17).

%% ============================================================================
%% Test 7: CLI Override Facts
%% ============================================================================

test_cli_override_facts :-
    format("~nCLI overrides:~n"),
    findall(Arg, agent_loop_module:cli_override(Arg, _, _), Overrides),
    length(Overrides, Count),
    assert_eq('CLI override count', Count, 12),
    %% Verify behavior types
    assert_true('backend has backend_special', (
        agent_loop_module:cli_override(backend, _, backend_special)
    )),
    assert_true('auto_tools has set_true', (
        agent_loop_module:cli_override(auto_tools, _, set_true)
    )),
    assert_true('no_tools has clear_list', (
        agent_loop_module:cli_override(no_tools, _, clear_list)
    )).

%% ============================================================================
%% Test 8: Streaming Capability
%% ============================================================================

test_streaming_capability :-
    format("~nStreaming capability:~n"),
    assert_true('api_local is streaming capable', (
        agent_loop_module:streaming_capable(api_local)
    )),
    assert_true('api is streaming capable', (
        agent_loop_module:streaming_capable(api)
    )),
    assert_true('openrouter is streaming capable', (
        agent_loop_module:streaming_capable(openrouter)
    )),
    assert_true('cli is NOT streaming capable', (
        \+ agent_loop_module:streaming_capable(cli)
    )).

%% ============================================================================
%% Test 9: Retry Config (source-level verification)
%% ============================================================================

test_retry_config :-
    format("~nRetry config:~n"),
    %% retry_config/3 is in generated backends.pl, not loaded here.
    %% Verify the source-level streaming_capable facts (which proves generator works)
    assert_true('3 streaming capable types', (
        findall(T, agent_loop_module:streaming_capable(T), Types),
        length(Types, 3)
    )),
    %% Verify backend_factory_order has all 8
    assert_true('backend_factory_order has 8', (
        backend_factory_order(Order),
        length(Order, 8)
    )).

%% ============================================================================
%% Test 10: Bindings Summary
%% ============================================================================

test_bindings_summary :-
    format("~nBindings summary:~n"),
    assert_true('Python summary generated', (
        generate_bindings_summary(python, Summary),
        atom(Summary),
        sub_atom(Summary, _, _, _, 'python bindings (5)')
    )),
    assert_true('Prolog summary generated', (
        generate_bindings_summary(prolog, PlSummary),
        atom(PlSummary),
        sub_atom(PlSummary, _, _, _, 'prolog bindings (5)')
    )).
