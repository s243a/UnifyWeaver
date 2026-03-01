%% agent_loop_bindings.pl — Binding registry integration for agent-loop
%%
%% Maps agent-loop predicates to target-language functions via binding/6.
%% Enables target generators to automatically emit cross-language code.
%%
%% Usage:
%%   swipl -l agent_loop_bindings.pl -g "agent_loop_binding_summary, halt"
%%
%% This module is an optional overlay — generated Prolog files remain standalone.

:- module(agent_loop_bindings, [
    init_agent_loop_bindings/0,
    agent_loop_binding_summary/0
]).

:- reexport('../../src/unifyweaver/core/binding_registry', [
    binding/6,
    bindings_for_target/2,
    bindings_for_predicate/2,
    effect/2,
    is_pure_binding/2,
    is_deterministic_binding/2,
    binding_has_effect/3,
    binding_imports/2
]).

:- use_module('../../src/unifyweaver/core/binding_registry').
:- use_module(agent_loop_module).

%% ============================================================================
%% Effect Declarations
%% ============================================================================

register_agent_loop_effects :-
    %% Pure lookup predicates — no side effects
    declare_effect(tool_handler/2, [deterministic, pure]),
    declare_effect(slash_command/4, [deterministic, pure]),
    declare_effect(backend_factory/2, [deterministic, pure]),
    declare_effect(audit_profile_level/2, [deterministic, pure, total]),
    declare_effect(security_profile/2, [deterministic, pure]).

%% ============================================================================
%% Python Target Bindings
%% ============================================================================

register_python_bindings :-
    %% tool_handler/2 -> TOOL_HANDLERS dict lookup
    declare_binding(python, tool_handler/2,
        'TOOL_HANDLERS[name]',
        [name-atom],
        [handler-callable],
        [pure, deterministic, pattern(dict_lookup)]),

    %% slash_command/4 -> command dispatch
    declare_binding(python, slash_command/4,
        'SLASH_COMMANDS.get(name)',
        [name-atom],
        [match_type-atom, options-list, help_text-string],
        [pure, deterministic, pattern(dict_lookup)]),

    %% backend_factory/2 -> create_backend_from_config
    declare_binding(python, backend_factory/2,
        'create_backend_from_config',
        [name-atom],
        [backend-object],
        [effect(io), nondeterministic, import(agent_loop)]),

    %% audit_profile_level/2 -> audit_levels dict
    declare_binding(python, audit_profile_level/2,
        'audit_levels[profile]',
        [profile-atom],
        [level-atom],
        [pure, deterministic, total, pattern(dict_lookup)]),

    %% security_profile/2 -> SecurityConfig.from_profile
    declare_binding(python, security_profile/2,
        'SecurityConfig.from_profile',
        [name-atom],
        [config-object],
        [effect(io), deterministic, import('security.profiles')]).

%% ============================================================================
%% Prolog Target Bindings
%% ============================================================================

register_prolog_bindings :-
    %% tool_handler/2 -> direct fact access
    declare_binding(prolog, tool_handler/2,
        tool_handler,
        [name-atom],
        [handler-atom],
        [pure, deterministic]),

    %% slash_command/4 -> direct fact access
    declare_binding(prolog, slash_command/4,
        slash_command,
        [name-atom],
        [match_type-atom, options-list, help_text-atom],
        [pure, deterministic]),

    %% backend_factory/2 -> create_backend/3
    declare_binding(prolog, backend_factory/2,
        create_backend,
        [name-atom, options-list],
        [backend-dict],
        [effect(io), nondeterministic]),

    %% audit_profile_level/2 -> direct fact access
    declare_binding(prolog, audit_profile_level/2,
        audit_profile_level,
        [profile-atom],
        [level-atom],
        [pure, deterministic, total]),

    %% security_profile/2 -> direct fact access
    declare_binding(prolog, security_profile/2,
        security_profile,
        [name-atom],
        [config-list],
        [pure, deterministic]).

%% ============================================================================
%% Master Registration
%% ============================================================================

init_agent_loop_bindings :-
    register_agent_loop_effects,
    register_python_bindings,
    register_prolog_bindings.

%% ============================================================================
%% Summary / Diagnostic
%% ============================================================================

agent_loop_binding_summary :-
    init_agent_loop_bindings,
    format("~nAgent-loop binding registry:~n"),
    %% Count by target
    bindings_for_target(python, PyBindings),
    bindings_for_target(prolog, PlBindings),
    length(PyBindings, NPy),
    length(PlBindings, NPl),
    format("  Python bindings: ~w~n", [NPy]),
    format("  Prolog bindings: ~w~n", [NPl]),
    %% List each binding
    format("~n  Python:~n"),
    forall(member(binding(python, Pred, TName, _, _, _), PyBindings),
        format("    ~w -> ~w~n", [Pred, TName])),
    format("~n  Prolog:~n"),
    forall(member(binding(prolog, Pred, TName, _, _, _), PlBindings),
        format("    ~w -> ~w~n", [Pred, TName])),
    %% Show effects
    format("~n  Effects:~n"),
    forall(effect(Pred, Effects),
        format("    ~w: ~w~n", [Pred, Effects])).
