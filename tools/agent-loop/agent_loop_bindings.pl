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
    agent_loop_binding_summary/0,
    compile_binding_code/3,
    generate_bindings_summary/2,
    emit_binding_imports/2,
    emit_binding_dispatch_comment/2
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
    declare_effect(security_profile/2, [deterministic, pure]),
    declare_effect(api_key_env_var/2, [deterministic, pure]),
    declare_effect(api_key_file/2, [deterministic, pure]),
    declare_effect(default_agent_preset/3, [deterministic, pure]).

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
        [effect(io), deterministic, import('security.profiles')]),

    %% api_key_env_var/2 -> env_vars dict lookup
    declare_binding(python, api_key_env_var/2,
        'API_KEY_ENV_VARS[backend]',
        [backend-atom],
        [env_var-string],
        [pure, deterministic, pattern(dict_lookup)]),

    %% api_key_file/2 -> file_locations dict lookup
    declare_binding(python, api_key_file/2,
        'API_KEY_FILE_PATHS[backend]',
        [backend-atom],
        [file_path-string],
        [pure, deterministic, pattern(dict_lookup)]),

    %% default_agent_preset/3 -> preset config lookup
    declare_binding(python, default_agent_preset/3,
        'get_default_config().agents[name]',
        [name-atom],
        [backend-atom, overrides-list],
        [pure, deterministic, pattern(dict_lookup)]).

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
        [pure, deterministic]),

    %% api_key_env_var/2 -> direct fact access
    declare_binding(prolog, api_key_env_var/2,
        api_key_env_var,
        [backend-atom],
        [env_var-atom],
        [pure, deterministic]),

    %% api_key_file/2 -> direct fact access
    declare_binding(prolog, api_key_file/2,
        api_key_file,
        [backend-atom],
        [file_path-atom],
        [pure, deterministic]).

%% ============================================================================
%% Master Registration
%% ============================================================================

init_agent_loop_bindings :-
    register_agent_loop_effects,
    register_python_bindings,
    register_prolog_bindings.

%% ============================================================================
%% Binding Consumers — Code Generation from Bindings
%% ============================================================================

%% compile_binding_code(+Target, +Pred, -Code)
%% Generate a target-language code snippet for a binding
compile_binding_code(Target, Pred, Code) :-
    binding(Target, Pred, TargetName, Inputs, Outputs, Options),
    format_binding_code(Target, Pred, TargetName, Inputs, Outputs, Options, Code).

%% Python code generation
format_binding_code(python, _Pred, TargetName, Inputs, Outputs, Options, Code) :-
    (member(pattern(dict_lookup), Options) ->
        %% Dict lookup pattern: result = DICT[key]
        Inputs = [InputName-_|_],
        Outputs = [OutputName-_|_],
        format(atom(Code), '~w = ~w  # ~w -> ~w', [OutputName, TargetName, InputName, OutputName])
    ; member(import(Mod), Options) ->
        %% Imported function call
        maplist([N-_,N]>>true, Inputs, INames),
        atomic_list_concat(INames, ', ', ArgStr),
        Outputs = [OutputName-_|_],
        format(atom(Code), 'from ~w import ~w~n~w = ~w(~w)', [Mod, TargetName, OutputName, TargetName, ArgStr])
    ;
        %% Default function call
        maplist([N-_,N]>>true, Inputs, INames),
        atomic_list_concat(INames, ', ', ArgStr),
        Outputs = [OutputName-_|_],
        format(atom(Code), '~w = ~w(~w)', [OutputName, TargetName, ArgStr])
    ).

%% Prolog code generation
format_binding_code(prolog, _Pred, TargetName, Inputs, Outputs, _Options, Code) :-
    maplist([N-_,N]>>true, Inputs, INames),
    maplist([N-_,N]>>true, Outputs, ONames),
    append(INames, ONames, AllArgs),
    atomic_list_concat(AllArgs, ', ', ArgStr),
    format(atom(Code), '~w(~w)', [TargetName, ArgStr]).

%% generate_bindings_summary(+Target, -Summary)
%% Generate a formatted summary of all bindings for a target
generate_bindings_summary(Target, Summary) :-
    bindings_for_target(Target, Bindings),
    length(Bindings, Count),
    format(atom(Header), '## ~w bindings (~w)~n', [Target, Count]),
    findall(Line, (
        member(binding(Target, Pred, TName, Inputs, Outputs, Opts), Bindings),
        maplist([N-T,Pair]>>(format(atom(Pair), '~w:~w', [N, T])), Inputs, IPairs),
        maplist([N-T,Pair]>>(format(atom(Pair), '~w:~w', [N, T])), Outputs, OPairs),
        atomic_list_concat(IPairs, ', ', IStr),
        atomic_list_concat(OPairs, ', ', OStr),
        (member(pure, Opts) -> Eff = pure ; Eff = effectful),
        format(atom(Line), '  ~w -> ~w(~w) => (~w) [~w]', [Pred, TName, IStr, OStr, Eff])
    ), Lines),
    atomic_list_concat([Header|Lines], '\n', Summary).

%% ============================================================================
%% Binding Emit Helpers — for Python generator integration
%% ============================================================================

%% emit_binding_imports(+Stream, +Options)
%% Emit Python import lines for bindings that have import(Module) in options.
emit_binding_imports(S, _Options) :-
    init_agent_loop_bindings,
    findall(Mod-TName, (
        binding(python, _Pred, TName, _Ins, _Outs, Opts),
        member(import(Mod), Opts)
    ), ImportPairs),
    forall(member(Mod-TName, ImportPairs), (
        %% Extract base name for dotted targets (e.g. SecurityConfig.from_profile → SecurityConfig)
        (sub_atom(TName, Before, _, _, '.') ->
            sub_atom(TName, 0, Before, _, BaseName)
        ;
            BaseName = TName
        ),
        format(S, 'from ~w import ~w~n', [Mod, BaseName])
    )).

%% emit_binding_dispatch_comment(+Stream, +Options)
%% Emit a comment block documenting binding metadata for the current file.
emit_binding_dispatch_comment(S, _Options) :-
    init_agent_loop_bindings,
    write(S, '# Binding registry metadata:\n'),
    forall(binding(python, Pred, TName, Inputs, _Outputs, Opts), (
        maplist([N-_T,N]>>true, Inputs, INames),
        atomic_list_concat(INames, ', ', IStr),
        (member(pure, Opts) -> Eff = pure ; Eff = effectful),
        format(S, '#   ~w -> ~w(~w) [~w]~n', [Pred, TName, IStr, Eff])
    )),
    write(S, '#\n').

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
