%% agent_loop_bindings.pl — Binding registry integration for agent-loop
%%
%% Maps agent-loop predicates to target-language functions via binding/6.
%% Enables target generators to automatically emit cross-language code.
%%
%% Key predicates:
%%   compile_binding_code/3     — generate target-language code from binding
%%   binding_dict_name/3        — extract base structure name from binding target
%%   binding_pattern/3          — get pattern type (dict_lookup, set_membership, etc.)
%%   emit_binding_metadata_comment/3 — emit per-predicate binding comment
%%   emit_binding_imports/2     — emit Python imports from binding metadata
%%
%% Supported patterns: dict_lookup, set_membership, method_call,
%%   chained_call(Methods), function_call (default)
%%
%% Usage:
%%   swipl -l agent_loop_bindings.pl -g "agent_loop_binding_summary, halt"
%%
%% This module is an optional overlay — generated Prolog files remain standalone.

:- module(agent_loop_bindings, [
    init_agent_loop_bindings/0,
    agent_loop_binding_summary/0,
    compile_binding_code/3,
    binding_dict_name/3,
    binding_pattern/3,
    generate_bindings_summary/2,
    emit_binding_imports/2,
    emit_binding_dispatch_comment/2,
    emit_binding_metadata_comment/3,
    translate_agent_goal/2,
    translate_agent_goal/3,
    translate_agent_goals/3
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
        [pure, deterministic, pattern(dict_lookup)]),

    %% model_pricing/3 -> costs.DEFAULT_PRICING dict lookup
    declare_binding(python, model_pricing/3,
        'DEFAULT_PRICING[model]',
        [model-atom],
        [input_cost-float, output_cost-float],
        [import('costs'), pure, deterministic, pattern(dict_lookup)]),

    %% config_search_path/2 -> config search path list
    declare_binding(python, config_search_path/2,
        'CONFIG_SEARCH_PATHS[path_type]',
        [path_type-atom],
        [path-string],
        [import('config'), pure, deterministic, pattern(dict_lookup)]),

    %% destructive_tool/1 -> tools_generated.DESTRUCTIVE_TOOLS membership
    declare_binding(python, destructive_tool/1,
        'DESTRUCTIVE_TOOLS',
        [tool_name-atom],
        [],
        [import('tools_generated'), pure, deterministic, pattern(set_membership)]).

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
        [pure, deterministic]),

    %% model_pricing/3 -> direct fact access
    declare_binding(prolog, model_pricing/3,
        model_pricing,
        [model-atom],
        [input_cost-float, output_cost-float],
        [pure, deterministic]),

    %% config_search_path/2 -> direct fact access
    declare_binding(prolog, config_search_path/2,
        config_search_path,
        [path_type-atom],
        [path-string],
        [pure, deterministic]),

    %% default_agent_preset/3 -> direct fact access
    declare_binding(prolog, default_agent_preset/3,
        default_agent_preset,
        [name-atom],
        [backend-atom, overrides-list],
        [pure, deterministic]),

    %% destructive_tool/1 -> direct fact access
    declare_binding(prolog, destructive_tool/1,
        destructive_tool,
        [tool_name-atom],
        [],
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

%% binding_dict_name(+Target, +Pred, -DictName)
%% Extract the base structure name from a binding's target.
%% For dict lookups like 'DEFAULT_PRICING[model]', returns 'DEFAULT_PRICING'.
%% For function calls like 'create_backend_from_config', returns the name as-is.
binding_dict_name(Target, Pred, DictName) :-
    binding(Target, Pred, TargetName, _Inputs, _Outputs, _Options),
    (sub_atom(TargetName, Before, _, _, '[') ->
        sub_atom(TargetName, 0, Before, _, DictName)
    ; sub_atom(TargetName, Before, _, _, '.') ->
        sub_atom(TargetName, 0, Before, _, DictName)
    ;
        DictName = TargetName
    ).

%% binding_pattern(+Target, +Pred, -Pattern)
%% Get the binding pattern for a predicate, defaulting to function_call.
binding_pattern(Target, Pred, Pattern) :-
    binding(Target, Pred, _TName, _Inputs, _Outputs, Options),
    (member(pattern(P), Options) -> Pattern = P ; Pattern = function_call).

%% compile_binding_code(+Target, +Pred, -Code)
%% Generate a target-language code snippet for a binding
compile_binding_code(Target, Pred, Code) :-
    binding(Target, Pred, TargetName, Inputs, Outputs, Options),
    format_binding_code(Target, Pred, TargetName, Inputs, Outputs, Options, Code).

%% Python code generation — dispatches on pattern type
format_binding_code(python, _Pred, TargetName, Inputs, Outputs, Options, Code) :-
    (member(pattern(dict_lookup), Options) ->
        %% Dict lookup pattern: result = DICT[key]
        Inputs = [InputName-_|_],
        Outputs = [OutputName-_|_],
        format(atom(Code), '~w = ~w  # ~w -> ~w', [OutputName, TargetName, InputName, OutputName])
    ; member(pattern(set_membership), Options) ->
        %% Set membership test: if key in SET_NAME:
        Inputs = [InputName-_|_],
        format(atom(Code), 'if ~w in ~w:', [InputName, TargetName])
    ; member(pattern(method_call), Options) ->
        %% Method call: result = obj.method(args)
        Inputs = [ObjName-_|RestInputs],
        maplist([N-_,N]>>true, RestInputs, ArgNames),
        (ArgNames = [] ->
            MethodExpr = TargetName
        ;
            atomic_list_concat(ArgNames, ', ', ArgStr),
            format(atom(MethodExpr), '~w(~w)', [TargetName, ArgStr])
        ),
        (Outputs = [OutputName-_|_] ->
            format(atom(Code), '~w = ~w~w', [OutputName, ObjName, MethodExpr])
        ;
            format(atom(Code), '~w~w', [ObjName, MethodExpr])
        )
    ; member(pattern(chained_call(Methods)), Options) ->
        %% Chained method call: result = obj.m1().m2().m3()
        Inputs = [ObjName-_|_],
        format_chained_methods(Methods, ChainStr),
        (Outputs = [OutputName-_|_] ->
            format(atom(Code), '~w = ~w~w', [OutputName, ObjName, ChainStr])
        ;
            format(atom(Code), '~w~w', [ObjName, ChainStr])
        )
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

%% format_chained_methods(+Methods, -ChainStr)
%% Build a method chain string from a list of method specs.
format_chained_methods([], '').
format_chained_methods([Method|Rest], ChainStr) :-
    (Method = method(Name, Args) ->
        (Args = [] ->
            format(atom(Part), '~w()', [Name])
        ;
            atomic_list_concat(Args, ', ', ArgStr),
            format(atom(Part), '~w(~w)', [Name, ArgStr])
        )
    ;
        %% Simple atom method name — no args
        format(atom(Part), '~w()', [Method])
    ),
    format_chained_methods(Rest, RestStr),
    atom_concat(Part, RestStr, ChainStr).

%% ============================================================================
%% Goal Translation — Agent-loop adapter for binding-driven code generation
%% ============================================================================

%% translate_agent_goal(+Goal, -Code)
%% Translate a Prolog goal term into Python code using the binding registry.
%% Mirrors the main compiler's translate_goal/2 (python_target.pl:4304-4327)
%% but works locally with agent-loop bindings.
translate_agent_goal(Goal, Code) :-
    translate_agent_goal(python, Goal, Code).

%% translate_agent_goal(+Target, +Goal, -Code)
%% Translate a Prolog goal term into target-language code.
translate_agent_goal(Target, Goal, Code) :-
    init_agent_loop_bindings,
    functor(Goal, Pred, Arity),
    binding(Target, Pred/Arity, _TargetName, _Inputs, _Outputs, _Options), !,
    compile_binding_code(Target, Pred/Arity, Code).

%% translate_agent_goals(+Goals, +Target, -CodeBlock)
%% Translate a list of Prolog goals into a multi-line code block.
translate_agent_goals([], _Target, '').
translate_agent_goals([Goal|Rest], Target, CodeBlock) :-
    translate_agent_goal(Target, Goal, Code),
    translate_agent_goals(Rest, Target, RestCode),
    (RestCode == '' ->
        CodeBlock = Code
    ;
        atomic_list_concat([Code, '\n', RestCode], CodeBlock)
    ).

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
%% Uses unified emit_module_imports/2 via from() spec terms.
emit_binding_imports(S, _Options) :-
    init_agent_loop_bindings,
    findall(from(Mod, [BaseName]), (
        binding(python, _Pred, TName, _Ins, _Outs, Opts),
        member(import(Mod), Opts),
        %% Extract base name for dotted targets (e.g. SecurityConfig.from_profile → SecurityConfig)
        (sub_atom(TName, Before, _, _, '.') ->
            sub_atom(TName, 0, Before, _, BaseName)
        ;
            BaseName = TName
        )
    ), Specs),
    agent_loop_components:emit_module_imports(S, Specs).

%% emit_binding_dispatch_comment(+Stream, +Options)
%% Emit a comment block documenting binding metadata for the current file.
emit_binding_dispatch_comment(S, _Options) :-
    init_agent_loop_bindings,
    write(S, '# Binding registry metadata:\n'),
    findall(Pred-TName-Inputs-Opts,
        binding(python, Pred, TName, Inputs, _Outputs, Opts), Entries),
    maplist([Pred-TName-Inputs-Opts]>>(
        maplist([N-_T,N]>>true, Inputs, INames),
        atomic_list_concat(INames, ', ', IStr),
        (member(pure, Opts) -> Eff = pure ; Eff = effectful),
        format(S, '#   ~w -> ~w(~w) [~w]~n', [Pred, TName, IStr, Eff])
    ), Entries),
    write(S, '#\n').

%% emit_binding_metadata_comment(+Stream, +Target, +Pred)
%% Emit a comment documenting binding metadata for a specific predicate.
%% Safe no-op if the predicate has no binding for the target.
emit_binding_metadata_comment(S, Target, Pred) :-
    init_agent_loop_bindings,
    (binding(Target, Pred, TName, Inputs, _Outputs, Opts) ->
        maplist([N-_T,N]>>true, Inputs, INames),
        atomic_list_concat(INames, ', ', IStr),
        (member(pattern(P), Opts) -> true ; P = function_call),
        format(S, '# Binding: ~w -> ~w(~w) [~w]~n', [Pred, TName, IStr, P])
    ; true).

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
    maplist([binding(python, Pred, TName, _, _, _)]>>(
        format("    ~w -> ~w~n", [Pred, TName])
    ), PyBindings),
    format("~n  Prolog:~n"),
    maplist([binding(prolog, Pred2, TName2, _, _, _)]>>(
        format("    ~w -> ~w~n", [Pred2, TName2])
    ), PlBindings),
    %% Show effects
    format("~n  Effects:~n"),
    findall(Pred3-Effects, effect(Pred3, Effects), EffPairs),
    maplist([Pred3-Effects]>>(
        format("    ~w: ~w~n", [Pred3, Effects])
    ), EffPairs).
