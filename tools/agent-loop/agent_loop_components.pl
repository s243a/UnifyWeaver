%% agent_loop_components.pl — Component registry integration for agent-loop
%%
%% Bridges agent-loop facts (tool_handler/2, slash_command/4, backend_factory/2)
%% into the UnifyWeaver component registry for cross-system discoverability.
%%
%% Usage:
%%   swipl -l agent_loop_components.pl -g "agent_loop_component_summary, halt"
%%
%% This module is an optional overlay — generated Prolog files remain standalone.

:- module(agent_loop_components, [
    register_agent_loop_components/0,
    agent_loop_component_summary/0,
    emit_predicate_summary/0,
    emit_tool_facts/2,
    emit_command_facts/2,
    emit_backend_facts/2,
    emit_security_facts/2,
    emit_cost_facts/2,
    emit_backend_init_imports/2,
    emit_backend_init_optional/2,
    emit_context_enums/2,
    emit_message_fields/2,
    emit_agent_config_fields/2,
    emit_audit_levels/2,
    emit_cli_overrides/2,
    emit_test_metadata/0,
    emit_prolog_config_facts/2,
    emit_api_key_env_vars_py/2,
    emit_api_key_files_py/2,
    emit_default_presets_py/2,
    emit_security_module_imports/2,
    emit_help_groups/2,
    emit_readme_sections/2,
    emit_backend_module_imports/2,
    emit_streaming_capable_facts/2,
    emit_security_profile_entries/2,
    emit_tool_schemas_py/2,
    emit_tool_dispatch_entries/2,
    emit_cascade_paths/2,
    emit_alias_group_entries/2,
    emit_alias_conditions/2,
    emit_argparse_group_args/2,
    emit_backend_helper_fragments/2,
    emit_module_imports/2,
    emit_module_dependencies/2,
    backend_import_specs/2,
    security_import_specs/1,
    emit_dependency_diagram/2
]).

:- reexport('../../src/unifyweaver/core/component_registry', [
    compile_component/4,
    list_components/2,
    component/4
]).

:- use_module('../../src/unifyweaver/core/component_registry').
:- use_module(agent_loop_module).

%% ============================================================================
%% Inline Type Modules
%% ============================================================================
%%
%% Each type module implements: type_info/1, validate_config/1,
%% init_component/2, compile_component/4
%% These are defined as module-qualified clauses below.

%% --- Tool handler type ---

:- module_transparent agent_tool_type:type_info/1.
:- module_transparent agent_tool_type:validate_config/1.
:- module_transparent agent_tool_type:init_component/2.
:- module_transparent agent_tool_type:compile_component/4.

agent_tool_type:type_info(info{
    name: 'Agent Tool Handler',
    version: '1.0.0',
    description: 'Maps tool names to handler predicates'
}).

agent_tool_type:validate_config(Config) :-
    member(handler(_), Config).

agent_tool_type:init_component(_Name, _Config).

agent_tool_type:compile_component(Name, Config, Options, Code) :-
    (member(target(prolog), Options) ->
        (member(fact_type(FT), Options) ->
            compile_tool_fact(FT, Name, Config, Code)
        ;
            compile_tool_all_facts(Name, Config, Code)
        )
    ; member(target(python), Options) ->
        member(handler(H), Config),
        (member(indent(N), Options) -> true ; N = 4),
        (member(self_prefix(true), Options) ->
            format(atom(HStr), "self.~w", [H])
        ;   atom_string(H, HStr)),
        length(Spaces, N), maplist(=(0' ), Spaces), atom_chars(Indent, Spaces),
        format(atom(Code), "~w'~w': ~w,", [Indent, Name, HStr])
    ;
        member(handler(H), Config),
        format(atom(Code), "tool_handler(~q, ~q).", [Name, H])
    ).

%% compile_tool_fact(+FactType, +Name, +Config, -Code)
%% Generate a single Prolog fact line for a tool component.
compile_tool_fact(tool_spec, Name, Config, Code) :-
    member(tool_spec_props(Props), Config),
    Props \= [],
    with_output_to(atom(Code), (
        format('tool_spec(~q, ', [Name]),
        agent_loop_module:write_prolog_term(current_output, Props),
        write(').')
    )).

compile_tool_fact(tool_handler, Name, Config, Code) :-
    member(handler(H), Config),
    format(atom(Code), "tool_handler(~q, ~q).", [Name, H]).

compile_tool_fact(destructive_tool, Name, Config, Code) :-
    member(destructive(true), Config),
    format(atom(Code), "destructive_tool(~q).", [Name]).

%% compile_tool_all_facts(+Name, +Config, -Code)
%% Generate all Prolog fact lines for a tool, joined with newlines.
compile_tool_all_facts(Name, Config, Code) :-
    findall(Fact, (
        member(FT, [tool_spec, tool_handler, destructive_tool]),
        compile_tool_fact(FT, Name, Config, Fact)
    ), Facts),
    atomic_list_concat(Facts, '\n', Code).

%% --- Slash command type ---

:- module_transparent agent_command_type:type_info/1.
:- module_transparent agent_command_type:validate_config/1.
:- module_transparent agent_command_type:init_component/2.
:- module_transparent agent_command_type:compile_component/4.

agent_command_type:type_info(info{
    name: 'Agent Slash Command',
    version: '1.0.0',
    description: 'Slash command definition for the agent loop'
}).

agent_command_type:validate_config(Config) :-
    member(match_type(_), Config),
    member(help_text(_), Config).

agent_command_type:init_component(_Name, _Config).

agent_command_type:compile_component(Name, Config, Options, Code) :-
    (member(target(prolog), Options) ->
        (member(fact_type(FT), Options) ->
            compile_command_fact(FT, Name, Config, Code)
        ;
            compile_command_fact(slash_command, Name, Config, Code)
        )
    ;
        compile_command_fact(slash_command, Name, Config, Code)
    ).

%% compile_command_fact(+FactType, +Name, +Config, -Code)
compile_command_fact(slash_command, Name, Config, Code) :-
    member(match_type(MT), Config),
    member(help_text(HT), Config),
    (member(options(Opts), Config) -> true ; Opts = []),
    with_output_to(atom(Code), (
        format('slash_command(~q, ~q, ', [Name, MT]),
        agent_loop_module:write_prolog_term(current_output, Opts),
        format(', ~q).', [HT])
    )).

%% --- Backend factory type ---

:- module_transparent agent_backend_type:type_info/1.
:- module_transparent agent_backend_type:validate_config/1.
:- module_transparent agent_backend_type:init_component/2.
:- module_transparent agent_backend_type:compile_component/4.

agent_backend_type:type_info(info{
    name: 'Agent Backend',
    version: '1.0.0',
    description: 'Backend factory for LLM requests'
}).

agent_backend_type:validate_config(Config) :-
    member(resolve_type(_), Config),
    member(class_name(_), Config).

agent_backend_type:init_component(_Name, _Config).

agent_backend_type:compile_component(Name, Config, Options, Code) :-
    (member(target(prolog), Options) ->
        (member(fact_type(FT), Options) ->
            compile_backend_fact(FT, Name, Config, Code)
        ;
            compile_backend_fact(backend_factory, Name, Config, Code)
        )
    ;
        compile_backend_fact(backend_factory, Name, Config, Code)
    ).

%% compile_backend_fact(+FactType, +Name, +Config, -Code)
compile_backend_fact(backend_factory, Name, Config, Code) :-
    with_output_to(atom(Code), (
        format('backend_factory(~q, ', [Name]),
        agent_loop_module:write_prolog_term(current_output, Config),
        write(').')
    )).

%% --- Security profile type ---

:- module_transparent agent_security_type:type_info/1.
:- module_transparent agent_security_type:validate_config/1.
:- module_transparent agent_security_type:init_component/2.
:- module_transparent agent_security_type:compile_component/4.

agent_security_type:type_info(info{
    name: 'Agent Security Profile',
    version: '1.0.0',
    description: 'Security profile with path/command validation rules'
}).

agent_security_type:validate_config(_Config).

agent_security_type:init_component(_Name, _Config).

agent_security_type:compile_component(Name, Config, _Options, Code) :-
    with_output_to(atom(Code), (
        format('security_profile(~q, ', [Name]),
        agent_loop_module:write_prolog_term(current_output, Config),
        write(').')
    )).

%% --- Model cost type ---

:- module_transparent agent_cost_type:type_info/1.
:- module_transparent agent_cost_type:validate_config/1.
:- module_transparent agent_cost_type:init_component/2.
:- module_transparent agent_cost_type:compile_component/4.

agent_cost_type:type_info(info{
    name: 'Agent Model Cost',
    version: '1.0.0',
    description: 'Model pricing for token cost tracking'
}).

agent_cost_type:validate_config(Config) :-
    member(input_price(_), Config),
    member(output_price(_), Config).

agent_cost_type:init_component(_Name, _Config).

agent_cost_type:compile_component(_Name, Config, Options, Code) :-
    member(input_price(In), Config),
    member(output_price(Out), Config),
    member(model_string(Model), Config),
    (member(target(python), Options) ->
        format(atom(Code), '    "~w": {"input": ~w, "output": ~w},', [Model, In, Out])
    ;
        format(atom(Code), "model_pricing(~q, ~w, ~w).", [Model, In, Out])
    ).

%% ============================================================================
%% Category and Type Registration
%% ============================================================================

register_agent_loop_categories :-
    define_category(agent_tools, "Agent-loop tool definitions", [
        requires_compilation(true)
    ]),
    define_category(agent_commands, "Agent-loop slash commands", [
        requires_compilation(true)
    ]),
    define_category(agent_backends, "Agent-loop backend definitions", [
        requires_compilation(true)
    ]),
    define_category(agent_security, "Agent-loop security profiles", [
        requires_compilation(true)
    ]),
    define_category(agent_costs, "Agent-loop model pricing", [
        requires_compilation(true)
    ]).

register_agent_loop_types :-
    register_component_type(agent_tools, tool_handler, agent_tool_type, [
        description("Tool handler mapping")
    ]),
    register_component_type(agent_commands, slash_command, agent_command_type, [
        description("Slash command definition")
    ]),
    register_component_type(agent_backends, backend, agent_backend_type, [
        description("Agent backend factory")
    ]),
    register_component_type(agent_security, security_profile, agent_security_type, [
        description("Security profile definition")
    ]),
    register_component_type(agent_costs, model_pricing, agent_cost_type, [
        description("Model token pricing")
    ]).

%% ============================================================================
%% Instance Population from Existing Facts
%% ============================================================================

%% Populate tool components from tool_handler/2 + tool_spec/2 + destructive_tool/1
register_tool_components :-
    findall(Name-Handler, agent_loop_module:tool_handler(Name, Handler), Pairs),
    maplist([Name-Handler]>>(
        (agent_loop_module:tool_spec(Name, Props) ->
            (member(description(Desc), Props) -> true ; Desc = "")
        ; Props = [], Desc = ""),
        (agent_loop_module:destructive_tool(Name) -> Destr = true ; Destr = false),
        declare_component(agent_tools, Name, tool_handler, [
            handler(Handler),
            description(Desc),
            destructive(Destr),
            tool_spec_props(Props),
            initialization(eager)
        ])
    ), Pairs).

%% Populate command components from slash_command/4
register_command_components :-
    findall(Name-MatchType-Opts-Help, agent_loop_module:slash_command(Name, MatchType, Opts, Help), Entries),
    maplist([Name-MatchType-Opts-Help]>>(
        declare_component(agent_commands, Name, slash_command, [
            match_type(MatchType),
            options(Opts),
            help_text(Help),
            initialization(eager)
        ])
    ), Entries).

%% Populate backend components from backend_factory/2
register_backend_components :-
    findall(Name-Spec, agent_loop_module:backend_factory(Name, Spec), Pairs),
    maplist([Name-Spec]>>(declare_component(agent_backends, Name, backend, Spec)), Pairs).

%% Populate security components from security_profile/2
register_security_components :-
    findall(Name-Props, agent_loop_module:security_profile(Name, Props), Pairs),
    maplist([Name-Props]>>(declare_component(agent_security, Name, security_profile, Props)), Pairs).

%% Populate cost components from model_pricing/3
register_cost_components :-
    findall(Model-In-Out, agent_loop_module:model_pricing(Model, In, Out), Entries),
    maplist([Model-In-Out]>>(
        (atom(Model) -> ModelAtom = Model ; atom_string(ModelAtom, Model)),
        declare_component(agent_costs, ModelAtom, model_pricing, [
            input_price(In),
            output_price(Out),
            model_string(Model)
        ])
    ), Entries).

%% ============================================================================
%% Master Registration
%% ============================================================================

%% register_agent_loop_components/0
%% Registers all categories, types, and populates instances from existing facts.
register_agent_loop_components :-
    register_agent_loop_categories,
    register_agent_loop_types,
    register_tool_components,
    register_command_components,
    register_backend_components,
    register_security_components,
    register_cost_components.

%% ============================================================================
%% Stream Emitters — Write Prolog facts via the component registry
%% ============================================================================

%% emit_tool_facts(+Stream, +Options)
%% Emit tool-related Prolog facts via the component registry.
emit_tool_facts(S, _Options) :-
    %% tool_spec via compile_component
    write(S, '%% tool_spec(+ToolName, +Properties)\n'),
    findall(Name, component(agent_tools, Name, tool_handler, _), ToolNames),
    maplist([Name]>>(
        (compile_component(agent_tools, Name, [target(prolog), fact_type(tool_spec)], Code) ->
            write(S, Code), nl(S)
        ; true)
    ), ToolNames),
    write(S, '\n'),
    %% tool_handler via compile_component
    write(S, '%% tool_handler(+ToolName, +HandlerPredicate)\n'),
    maplist([Name]>>(
        compile_component(agent_tools, Name, [target(prolog), fact_type(tool_handler)], Code),
        write(S, Code), nl(S)
    ), ToolNames),
    write(S, '\n'),
    %% destructive_tool via compile_component
    write(S, '%% destructive_tool(+ToolName)\n'),
    findall(Name, (component(agent_tools, Name, tool_handler, Cfg),
            member(destructive(true), Cfg)), DestrNames),
    maplist([Name]>>(
        compile_component(agent_tools, Name, [target(prolog), fact_type(destructive_tool)], Code),
        write(S, Code), nl(S)
    ), DestrNames),
    write(S, '\n'),
    %% tool_description — stays as raw fact iteration (cross-cutting shape)
    write(S, '%% tool_description(+Backend, +ToolName, +Verb, +ParamKey, +DisplayMode)\n'),
    findall(td(Backend, TN, Verb, PK, DM), agent_loop_module:tool_description(Backend, TN, Verb, PK, DM), TDs),
    maplist([td(Backend, TN, Verb, PK, DM)]>>(
        format(S, 'tool_description(~q, ~q, ~q, ~q, ', [Backend, TN, Verb, PK]),
        agent_loop_module:write_prolog_term(S, DM),
        write(S, ').\n')
    ), TDs),
    write(S, '\n').

%% emit_command_facts(+Stream, +Options)
%% Emit command-related Prolog facts via the component registry.
emit_command_facts(S, _Options) :-
    %% Optimization notes and indexing hints
    findall(_, component(agent_commands, _, slash_command, _), SCs), length(SCs, NSC),
    findall(_, agent_loop_module:command_alias(_, _), CAs), length(CAs, NCA),
    emit_optimization_notes(S, [
        'slash_command/4: first-argument atom-indexed, all distinct',
        'command_alias/2: first-argument string hash-indexed, all distinct'
    ]),
    write(S, '%% Indexing hints (SWI-Prolog auto-indexes first argument):\n'),
    emit_indexing_directive(S, slash_command/4, NSC),
    emit_indexing_directive(S, command_alias/2, NCA),
    write(S, '\n'),
    %% slash_command via compile_component
    write(S, '%% slash_command(+Name, +MatchType, +Options, +HelpText)\n'),
    findall(CName, component(agent_commands, CName, slash_command, _), CmdNames),
    maplist([CName]>>(
        compile_component(agent_commands, CName, [target(prolog), fact_type(slash_command)], Code),
        write(S, Code), nl(S)
    ), CmdNames),
    write(S, '\n'),
    %% command_alias — cross-cutting, stays as raw fact iteration
    write(S, '%% command_alias(+Alias, +CanonicalName)\n'),
    findall(Alias-Canonical, agent_loop_module:command_alias(Alias, Canonical), AliasPairs),
    maplist([Alias-Canonical]>>(format(S, 'command_alias(~q, ~q).~n', [Alias, Canonical])), AliasPairs),
    write(S, '\n'),
    %% slash_command_group — cross-cutting, stays as raw fact iteration
    write(S, '%% slash_command_group(+GroupName, +CommandList)\n'),
    findall(Group-Cmds, agent_loop_module:slash_command_group(Group, Cmds), GroupPairs),
    maplist([Group-Cmds]>>(
        format(S, 'slash_command_group(~q, ', [Group]),
        agent_loop_module:write_prolog_term(S, Cmds),
        write(S, ').\n')
    ), GroupPairs),
    write(S, '\n').

%% emit_backend_facts(+Stream, +Options)
%% Emit backend-related Prolog facts via the component registry.
emit_backend_facts(S, _Options) :-
    %% Optimization notes and indexing hints
    emit_optimization_notes(S, [
        'agent_backend/2: deterministic lookup by backend name',
        'backend_factory/2: deterministic lookup by factory name',
        'streaming_capable/1: 3-clause fact table'
    ]),
    findall(_, agent_loop_module:agent_backend(_, _), ABs), length(ABs, NAB),
    findall(_, (component(agent_backends, _, backend, _)), BFs), length(BFs, NBF),
    findall(_, agent_loop_module:cli_fallbacks(_, _), CFs), length(CFs, NCF),
    write(S, '%% Indexing hints (SWI-Prolog auto-indexes first argument):\n'),
    emit_indexing_directive(S, agent_backend/2, NAB),
    emit_indexing_directive(S, backend_factory/2, NBF),
    emit_indexing_directive(S, cli_fallbacks/2, NCF),
    emit_indexing_directive(S, streaming_capable/1),
    write(S, '\n'),
    %% agent_backend — rich property list not in registry, stays raw
    write(S, '%% agent_backend(+Name, +Properties)\n'),
    findall(Name-Props, agent_loop_module:agent_backend(Name, Props), ABPairs),
    maplist([Name-Props]>>(
        format(S, 'agent_backend(~q, ', [Name]),
        agent_loop_module:write_prolog_term(S, Props),
        write(S, ').\n')
    ), ABPairs),
    write(S, '\n'),
    %% backend_factory via compile_component
    write(S, '%% backend_factory(+Name, +FactorySpec)\n'),
    findall(BFName, component(agent_backends, BFName, backend, _), BFNames),
    maplist([BFName]>>(
        compile_component(agent_backends, BFName, [target(prolog), fact_type(backend_factory)], Code),
        write(S, Code), nl(S)
    ), BFNames),
    write(S, '\n'),
    %% backend_factory_order — singleton, stays raw
    (agent_loop_module:backend_factory_order(Order) ->
        format(S, 'backend_factory_order(', []),
        agent_loop_module:write_prolog_term(S, Order),
        write(S, ').\n\n')
    ; true),
    %% cli_fallbacks — stays raw
    write(S, '%% cli_fallbacks(+BackendName, +FallbackList)\n'),
    findall(CFName-Fallbacks, agent_loop_module:cli_fallbacks(CFName, Fallbacks), CFPairs),
    maplist([CFName-Fallbacks]>>(
        format(S, 'cli_fallbacks(~q, ', [CFName]),
        agent_loop_module:write_prolog_term(S, Fallbacks),
        write(S, ').\n')
    ), CFPairs),
    write(S, '\n').

%% emit_security_facts(+Stream, +Options)
%% Emit security-related facts — dispatches on target(python) vs target(prolog).
emit_security_facts(S, Options) :-
    (member(target(python), Options) ->
        %% Python path: emit blocked_* facts as collection entries
        (member(fact_type(FT), Options) ->
            emit_security_blocked_py(S, FT)
        ;
            emit_security_blocked_py(S, blocked_path),
            emit_security_blocked_py(S, blocked_path_prefix),
            emit_security_blocked_py(S, blocked_home_pattern),
            emit_security_blocked_py(S, blocked_command_pattern)
        )
    ;
        %% Prolog path: security_profile via compile_component + raw enumerations
        write(S, '%% Indexing hints (SWI-Prolog auto-indexes first argument):\n'),
        emit_indexing_directive(S, security_profile/2),
        emit_indexing_directive(S, blocked_path/1),
        emit_indexing_directive(S, blocked_path_prefix/1),
        emit_indexing_directive(S, blocked_home_pattern/1),
        emit_indexing_directive(S, blocked_command_pattern/2),
        write(S, '\n'),
        write(S, '%% security_profile(+Name, +Properties)\n'),
        findall(SPName, component(agent_security, SPName, security_profile, _), SPNames),
        maplist([SPName]>>(
            compile_component(agent_security, SPName, [target(prolog)], Code),
            write(S, Code), nl(S)
        ), SPNames),
        write(S, '\n'),
        write(S, '%% blocked_path(+AbsolutePath)\n'),
        findall(P, agent_loop_module:blocked_path(P), BPs),
        maplist([P]>>(format(S, 'blocked_path(~q).~n', [P])), BPs),
        write(S, '\n'),
        write(S, '%% blocked_path_prefix(+Prefix)\n'),
        findall(P, agent_loop_module:blocked_path_prefix(P), BPPs),
        maplist([P]>>(format(S, 'blocked_path_prefix(~q).~n', [P])), BPPs),
        write(S, '\n'),
        write(S, '%% blocked_home_pattern(+Pattern)\n'),
        findall(P, agent_loop_module:blocked_home_pattern(P), BHPs),
        maplist([P]>>(format(S, 'blocked_home_pattern(~q).~n', [P])), BHPs),
        write(S, '\n'),
        write(S, '%% blocked_command_pattern(+Regex, +Description)\n'),
        findall(Regex-Desc, agent_loop_module:blocked_command_pattern(Regex, Desc), BCPs),
        maplist([Regex-Desc]>>(format(S, 'blocked_command_pattern(~q, ~q).~n', [Regex, Desc])), BCPs),
        write(S, '\n')
    ).

%% emit_security_blocked_py(+Stream, +FactType)
%% Emit blocked_* facts as Python set/tuple/list entries.
emit_security_blocked_py(S, blocked_path) :-
    findall(P, agent_loop_module:blocked_path(P), Ps),
    maplist([P]>>(format(S, '    \'~w\',~n', [P])), Ps).
emit_security_blocked_py(S, blocked_path_prefix) :-
    findall(P, agent_loop_module:blocked_path_prefix(P), Ps),
    maplist([P]>>(format(S, '    \'~w\',~n', [P])), Ps).
emit_security_blocked_py(S, blocked_home_pattern) :-
    findall(P, agent_loop_module:blocked_home_pattern(P), Ps),
    maplist([P]>>(format(S, '    \'~w\',~n', [P])), Ps).
emit_security_blocked_py(S, blocked_command_pattern) :-
    findall(Regex-Desc, agent_loop_module:blocked_command_pattern(Regex, Desc), Ps),
    maplist([Regex-Desc]>>(format(S, '    (r\'~w\', "~w"),~n', [Regex, Desc])), Ps).

%% emit_cost_facts(+Stream, +Options)
%% Emit cost facts — dispatches on target(python) vs target(prolog).
emit_cost_facts(S, Options) :-
    findall(Model, component(agent_costs, Model, model_pricing, _), Models),
    (member(target(python), Options) ->
        maplist([Model]>>(
            compile_component(agent_costs, Model, [target(python)], Code),
            write(S, Code), nl(S)
        ), Models)
    ;
        write(S, '%% Indexing hints (SWI-Prolog auto-indexes first argument):\n'),
        emit_indexing_directive(S, model_pricing/3),
        write(S, '\n'),
        write(S, '%% model_pricing(+Model, +InputPricePerMTok, +OutputPricePerMTok)\n'),
        maplist([Model]>>(
            compile_component(agent_costs, Model, [target(prolog)], Code),
            write(S, Code), nl(S)
        ), Models)
    ).

%% emit_backend_init_imports(+Stream, +Options)
%% Emit Python import lines for non-optional backends.
emit_backend_init_imports(S, _Options) :-
    findall(Name-Props, (agent_loop_module:agent_backend(Name, Props),
            \+ member(optional_import(true), Props)), Pairs),
    maplist([Name-Props]>>(
        agent_loop_module:resolve_class_name(Name, ClassName),
        agent_loop_module:resolve_file_name(Name, FileName),
        format(S, 'from .~w import ~w~n', [FileName, ClassName])
    ), Pairs).

%% emit_backend_init_optional(+Stream, +Options)
%% Emit Python try/except import blocks for optional backends.
emit_backend_init_optional(S, _Options) :-
    findall(Name-Props, (agent_loop_module:agent_backend(Name, Props),
            member(optional_import(true), Props)), Pairs),
    maplist([Name-Props]>>(
        agent_loop_module:resolve_class_name(Name, ClassName),
        agent_loop_module:resolve_file_name(Name, FileName),
        member(description(Desc), Props),
        format(S, '~n# ~w (optional - requires pip package)~n', [Desc]),
        format(S, 'try:~n', []),
        format(S, '    from .~w import ~w~n', [FileName, ClassName]),
        format(S, '    __all__.append(\'~w\')~n', [ClassName]),
        format(S, 'except ImportError:~n', []),
        format(S, '    pass~n', [])
    ), Pairs).

%% ============================================================================
%% Prolog Target Optimization Helpers
%% ============================================================================

%% emit_indexing_directive(+Stream, +Pred/Arity)
%% Emit an indexing hint comment for predicates that benefit from
%% SWI-Prolog's first-argument indexing (auto-applied to fact tables).
emit_indexing_directive(S, Pred/Arity) :-
    format(S, '%%   ~w/~w: first-argument indexed~n', [Pred, Arity]).

%% emit_indexing_directive(+Stream, +Pred/Arity, +ClauseCount)
%% Variant with clause count annotation.
emit_indexing_directive(S, Pred/Arity, Count) :-
    format(S, '%%   ~w/~w: first-argument indexed (~w clauses)~n', [Pred, Arity, Count]).

%% emit_optimization_notes(+Stream, +Notes)
%% Emit optimization documentation as Prolog comments.
emit_optimization_notes(S, Notes) :-
    write(S, '%% Optimization notes:\n'),
    maplist([Note]>>(format(S, '%%   - ~w~n', [Note])), Notes),
    write(S, '\n').

%% ============================================================================
%% Prolog Config Emit — replaces 10 forall loops in generate_prolog_config
%% ============================================================================

%% emit_prolog_config_facts(+Stream, +Options)
%% Emit all fact sections for generated prolog/config.pl.
%% Consolidates 10 forall loops from generate_prolog_config into one predicate.
%% When target is Prolog, emits :- det() directives for deterministic lookups.
emit_prolog_config_facts(S, _Options) :-
    %% Optimization notes and indexing hints
    emit_optimization_notes(S, [
        'api_key_env_var/2, api_key_file/2: deterministic lookup per backend',
        'audit_profile_level/2: deterministic lookup per profile',
        'cli_argument/2: first-argument indexed on atom names (63 clauses)',
        'config_field_json_default/2: deterministic lookup per field'
    ]),
    write(S, '%% Indexing hints (SWI-Prolog auto-indexes first argument):\n'),
    emit_indexing_directive(S, cli_argument/2),
    emit_indexing_directive(S, api_key_env_var/2),
    emit_indexing_directive(S, api_key_file/2),
    emit_indexing_directive(S, audit_profile_level/2),
    emit_indexing_directive(S, config_field_json_default/2),
    write(S, '\n'),
    %% cli_argument facts
    write(S, '%% cli_argument(+Name, +Options)\n'),
    findall(Name-Opts, agent_loop_module:cli_argument(Name, Opts), CLIArgs),
    maplist([Name-Opts]>>(
        format(S, 'cli_argument(~q, ', [Name]),
        agent_loop_module:write_prolog_term(S, Opts),
        write(S, ').\n')
    ), CLIArgs),
    write(S, '\n'),
    %% agent_config_field facts
    write(S, '%% agent_config_field(+Name, +Type, +Default, +Description)\n'),
    findall(acf(Name, Type, Default, Desc), agent_loop_module:agent_config_field(Name, Type, Default, Desc), ACFs),
    maplist([acf(Name, Type, Default, Desc)]>>(
        format(S, 'agent_config_field(~q, ~q, ~q, ~q).~n', [Name, Type, Default, Desc])
    ), ACFs),
    write(S, '\n'),
    %% default_agent_preset facts
    write(S, '%% default_agent_preset(+PresetName, +Backend, +Overrides)\n'),
    findall(dap(Name, Backend, Overrides), agent_loop_module:default_agent_preset(Name, Backend, Overrides), DAPs),
    maplist([dap(Name, Backend, Overrides)]>>(
        format(S, 'default_agent_preset(~q, ~q, ', [Name, Backend]),
        agent_loop_module:write_prolog_term(S, Overrides),
        write(S, ').\n')
    ), DAPs),
    write(S, '\n'),
    %% api_key_env_var facts
    write(S, '%% api_key_env_var(+Backend, +EnvVar)\n'),
    findall(Backend-EnvVar, agent_loop_module:api_key_env_var(Backend, EnvVar), AKEs),
    maplist([Backend-EnvVar]>>(format(S, 'api_key_env_var(~q, ~q).~n', [Backend, EnvVar])), AKEs),
    write(S, '\n'),
    %% api_key_file facts
    write(S, '%% api_key_file(+Backend, +FilePath)\n'),
    findall(Backend-FilePath, agent_loop_module:api_key_file(Backend, FilePath), AKFs),
    maplist([Backend-FilePath]>>(format(S, 'api_key_file(~q, ~q).~n', [Backend, FilePath])), AKFs),
    write(S, '\n'),
    %% example_agent_config facts
    write(S, '%% example_agent_config(+Name, +Backend, +Properties)\n'),
    findall(eac(Name, Backend, Props), agent_loop_module:example_agent_config(Name, Backend, Props), EACs),
    maplist([eac(Name, Backend, Props)]>>(
        format(S, 'example_agent_config(~q, ~q, ', [Name, Backend]),
        agent_loop_module:write_prolog_term(S, Props),
        write(S, ').\n')
    ), EACs),
    write(S, '\n'),
    %% config_search_path facts
    write(S, '%% config_search_path(+Path, +Category)\n'),
    findall(CPath-Cat, agent_loop_module:config_search_path(CPath, Cat), CSPs),
    maplist([CPath-Cat]>>(format(S, 'config_search_path(~q, ~q).~n', [CPath, Cat])), CSPs),
    write(S, '\n'),
    %% config_field_json_default facts
    write(S, '%% config_field_json_default(+FieldName, +JsonDefault)\n'),
    write(S, '%% Maps agent config fields to JSON-safe defaults for code generation.\n'),
    write(S, '%% Special values: positional (name=arg), no_default (data.get with no fallback)\n'),
    findall(FName-FDefault, agent_loop_module:config_field_json_default(FName, FDefault), CFJDs),
    maplist([FName-FDefault]>>(format(S, 'config_field_json_default(~q, ~q).~n', [FName, FDefault])), CFJDs),
    write(S, '\n'),
    %% config_dir_file_name facts
    write(S, '%% config_dir_file_name(+FileName)\n'),
    write(S, '%% Standard config file names searched by load_config_from_dir.\n'),
    findall(FN, agent_loop_module:config_dir_file_name(FN), FNs),
    maplist([FN]>>(format(S, 'config_dir_file_name(~q).~n', [FN])), FNs),
    write(S, '\n'),
    %% audit_profile_level facts
    write(S, '%% audit_profile_level(+Profile, +AuditLevel)\n'),
    write(S, '%% Maps security profile to audit logging level.\n'),
    findall(P-L, agent_loop_module:audit_profile_level(P, L), APLs),
    maplist([P-L]>>(format(S, 'audit_profile_level(~q, ~q).~n', [P, L])), APLs),
    write(S, '\n').

%% ============================================================================
%% Python Emit Predicates — context.py, config.py, agent_loop.py
%% ============================================================================

%% emit_context_enums(+Stream, +Options)
%% Emit Python Enum classes from context_enum/3 facts.
emit_context_enums(S, _Options) :-
    findall(ce(EnumName, DocStr, Values), agent_loop_module:context_enum(EnumName, DocStr, Values), CEs),
    maplist([ce(EnumName, DocStr, Values)]>>(
        format(S, 'class ~w(Enum):~n', [EnumName]),
        format(S, '    """~w"""~n', [DocStr]),
        maplist([value(PyName, StrVal, Comment)]>>(
            format(S, '    ~w = "~w"    # ~w~n', [PyName, StrVal, Comment])
        ), Values),
        write(S, '\n\n')
    ), CEs).

%% emit_message_fields(+Stream, +Options)
%% Emit Python dataclass fields from message_field/3 facts.
emit_message_fields(S, _Options) :-
    findall(mf(Name, Type, Default), agent_loop_module:message_field(Name, Type, Default), MFs),
    maplist([mf(Name, Type, Default)]>>(
        (Default = none ->
            format(S, '    ~w: ~w~n', [Name, Type])
        ;
            format(S, '    ~w: ~w = ~w~n', [Name, Type, Default])
        )
    ), MFs).

%% emit_agent_config_fields(+Stream, +Options)
%% Emit Python dataclass fields from agent_config_field/4 facts.
emit_agent_config_fields(S, _Options) :-
    findall(acf(Name, Type, Default, Comment), agent_loop_module:agent_config_field(Name, Type, Default, Comment), ACFs),
    maplist([acf(Name, Type, Default, Comment)]>>(
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
    ), ACFs).

%% emit_audit_levels(+Stream, +Options)
%% Emit Python dict entries from audit_profile_level/2 facts.
emit_audit_levels(S, _Options) :-
    findall(Profile-Level, agent_loop_module:audit_profile_level(Profile, Level), Pairs),
    maplist([Profile-Level]>>(format(S, '        \'~w\': \'~w\',~n', [Profile, Level])), Pairs).

%% emit_cli_overrides(+Stream, +Options)
%% Emit Python CLI override dispatch from cli_override/3 facts.
emit_cli_overrides(S, _Options) :-
    findall(co(Arg, Field, Behavior), agent_loop_module:cli_override(Arg, Field, Behavior), COs),
    maplist([co(Arg, Field, Behavior)]>>(agent_loop_module:emit_single_override(S, Arg, Field, Behavior)), COs).

%% ============================================================================
%% Python Emit Predicates — config.py data-driven sections
%% ============================================================================

%% emit_api_key_env_vars_py(+Stream, +Options)
%% Emit Python dict entries from api_key_env_var/2 facts.
emit_api_key_env_vars_py(S, _Options) :-
    findall(Backend-Var, agent_loop_module:api_key_env_var(Backend, Var), Pairs),
    maplist([Backend-Var]>>(format(S, '        \'~w\': \'~w\',~n', [Backend, Var])), Pairs).

%% emit_api_key_files_py(+Stream, +Options)
%% Emit Python dict entries from api_key_file/2 facts.
emit_api_key_files_py(S, _Options) :-
    findall(Backend-Path, agent_loop_module:api_key_file(Backend, Path), Pairs),
    maplist([Backend-Path]>>(format(S, '        \'~w\': \'~w\',~n', [Backend, Path])), Pairs).

%% emit_default_presets_py(+Stream, +Options)
%% Emit Python AgentConfig preset entries from default_agent_preset/3 facts.
emit_default_presets_py(S, _Options) :-
    findall(dap(Name, Backend, Props), agent_loop_module:default_agent_preset(Name, Backend, Props), DAPs),
    maplist([dap(Name, Backend, Props)]>>(
        format(S, '    config.agents[\'~w\'] = AgentConfig(~n', [Name]),
        format(S, '        name=\'~w\',~n', [Name]),
        format(S, '        backend=\'~w\',~n', [Backend]),
        maplist([Key=Val]>>(
            (Val = true -> format(S, '        ~w=True,~n', [Key])
            ; Val = false -> format(S, '        ~w=False,~n', [Key])
            ; number(Val) -> format(S, '        ~w=~w,~n', [Key, Val])
            ; format(S, '        ~w=\'~w\',~n', [Key, Val])
            )
        ), Props),
        write(S, '    )\n\n')
    ), DAPs).

%% emit_security_module_imports(+Stream, +Options)
%% Emit Python import lines from security_module/3 facts via unified import system.
emit_security_module_imports(S, _Options) :-
    security_import_specs(Specs),
    emit_module_imports(S, Specs).

%% emit_help_groups(+Stream, +Options)
%% Emit Python help text from slash_command_group/2 facts.
emit_help_groups(S, _Options) :-
    findall(GroupLabel-CmdNames, agent_loop_module:slash_command_group(GroupLabel, CmdNames), Groups),
    maplist([GroupLabel-CmdNames]>>(
        include(agent_loop_module:is_python_command, CmdNames, PyCmds),
        (PyCmds \= [] ->
            format(S, '~w:~n', [GroupLabel]),
            maplist([CmdName]>>(agent_loop_module:format_help_line(S, CmdName)), PyCmds),
            write(S, '\n')
        ; true)
    ), Groups).

%% emit_readme_sections(+Stream, +Options)
%% Emit README markdown for backends and tools.
emit_readme_sections(S, _Options) :-
    write(S, '## Backends\n\n'),
    findall(Name-Desc, (agent_loop_module:agent_backend(Name, Props), member(description(Desc), Props)), BPairs),
    maplist([Name-Desc]>>(format(S, '- **~w**: ~w~n', [Name, Desc])), BPairs),
    write(S, '\n## Tools\n\n'),
    findall(Name-Desc, (agent_loop_module:tool_spec(Name, Props), member(description(Desc), Props)), TPairs),
    maplist([Name-Desc]>>(format(S, '- **~w**: ~w~n', [Name, Desc])), TPairs).

%% emit_backend_module_imports(+Stream, +Options)
%% Emit Python import lines from an imports list in Options via unified import system.
emit_backend_module_imports(S, Options) :-
    (member(imports(Imports), Options) ->
        maplist([Imp]>>(format(S, 'import ~w~n', [Imp])), Imports)
    ; true).

%% ============================================================================
%% Unified Import Infrastructure
%% ============================================================================

%% Import spec terms:
%%   bare(Mod)                -> import Mod
%%   from(Mod, Names)         -> from Mod import N1, N2
%%   from_relative(Mod, Names) -> from .Mod import N1, N2

%% emit_module_imports(+Stream, +Specs)
%% Dispatch on structured import spec terms to emit Python import lines.
emit_module_imports(S, Specs) :-
    maplist([Spec]>>(emit_one_import(S, Spec)), Specs).

emit_one_import(S, bare(Mod)) :-
    format(S, 'import ~w~n', [Mod]).
emit_one_import(S, from(Mod, Names)) :-
    atomic_list_concat(Names, ', ', NStr),
    format(S, 'from ~w import ~w~n', [Mod, NStr]).
emit_one_import(S, from_relative(Mod, Names)) :-
    atomic_list_concat(Names, ', ', NStr),
    format(S, 'from .~w import ~w~n', [Mod, NStr]).

%% backend_import_specs(+Props, -Specs)
%% Convert a backend's module_imports + optional from_imports into spec list.
backend_import_specs(Props, Specs) :-
    (member(module_imports(Mods), Props) -> true ; Mods = []),
    (member(from_imports(FromMap), Props) -> true ; FromMap = []),
    findall(Spec, (
        member(Mod, Mods),
        (member(Mod-Names, FromMap) ->
            Spec = from(Mod, Names)
        ;
            Spec = bare(Mod)
        )
    ), ModSpecs),
    %% Pick up from_imports entries whose module isn't in module_imports
    findall(from(Mod, Names), (
        member(Mod-Names, FromMap),
        \+ member(Mod, Mods)
    ), ExtraFromSpecs),
    append(ModSpecs, ExtraFromSpecs, Specs).

%% security_import_specs(-Specs)
%% Convert security_module/3 facts into from_relative spec list.
security_import_specs(Specs) :-
    findall(from_relative(Mod, [Primary|Extras]),
        agent_loop_module:security_module(Mod, Primary, Extras),
        Specs).

%% ============================================================================
%% Module Dependency Emitter
%% ============================================================================

%% emit_module_dependencies(+Stream, +Options)
%% Emit dependency documentation from module_dependency/3 facts.
%% Options must include module(Mod) to select which module's deps to emit.
emit_module_dependencies(S, Options) :-
    member(module(Mod), Options),
    findall(Dep-Reason, agent_loop_module:module_dependency(Mod, Dep, Reason), Deps),
    (Deps = [] ->
        write(S, '%% Dependencies: none (self-contained)\n')
    ;
        write(S, '%% Dependencies:\n'),
        maplist([Dep-Reason]>>(format(S, '%%   ~w (~w)~n', [Dep, Reason])), Deps)
    ),
    write(S, '\n').

%% ============================================================================
%% Dependency Diagram
%% ============================================================================

%% emit_dependency_diagram(+Stream, +Options)
%% Emit Mermaid dependency diagram from module_dependency/3 facts.
emit_dependency_diagram(S, _Options) :-
    write(S, '## Module Dependencies\n\n'),
    write(S, '```mermaid\ngraph TD\n'),
    %% Collect all modules mentioned in dependencies
    findall(M, (agent_loop_module:module_dependency(M, _, _) ; agent_loop_module:module_dependency(_, M, _)), MsDup),
    sort(MsDup, Modules),
    maplist([M]>>(format(S, '    ~w[~w]~n', [M, M])), Modules),
    write(S, '\n'),
    findall(Src-Tgt, agent_loop_module:module_dependency(Src, Tgt, _), DepEdges),
    maplist([Src-Tgt]>>(format(S, '    ~w --> ~w~n', [Src, Tgt])), DepEdges),
    write(S, '```\n\n').

%% ============================================================================
%% Forall Lift Batch 3 — Prolog backends, security profiles, tool dispatch,
%% config cascade, aliases, argparse, backend helpers
%% ============================================================================

%% emit_streaming_capable_facts(+Stream, +Options)
%% Emit streaming_capable/1 Prolog facts for generate_prolog_backends.
emit_streaming_capable_facts(S, _Options) :-
    write(S, '%% streaming_capable(+ResolveType) — which backends support streaming\n'),
    findall(Type, agent_loop_module:streaming_capable(Type), Types),
    maplist([Type]>>(format(S, 'streaming_capable(~q).~n', [Type])), Types),
    write(S, '\n').

%% emit_security_profile_entries(+Stream, +Options)
%% Emit Python SecurityProfile entries from component registry.
emit_security_profile_entries(S, _Options) :-
    register_agent_loop_components,
    findall(Name-Props, component(agent_security, Name, security_profile, Props), SPEs),
    maplist([Name-Props]>>(agent_loop_module:generate_profile_entry(S, Name, Props)), SPEs).

%% emit_tool_schemas_py(+Stream, +Options)
%% Emit Python tool schema dicts from tool_spec/2 facts.
emit_tool_schemas_py(S, _Options) :-
    findall(ToolName-ToolProps, agent_loop_module:tool_spec(ToolName, ToolProps), TSs),
    maplist([ToolName-ToolProps]>>(agent_loop_module:generate_tool_schema_py(S, ToolName, ToolProps)), TSs).

%% emit_tool_dispatch_entries(+Stream, +Options)
%% Emit self.tools dict entries via compile_component/4.
emit_tool_dispatch_entries(S, _Options) :-
    register_agent_loop_components,
    findall(TN, component(agent_tools, TN, tool_handler, _), TNs),
    maplist([TN]>>(
        compile_component(agent_tools, TN,
            [target(python), self_prefix(true), indent(12)], Code),
        write(S, Code), nl(S)
    ), TNs).

%% emit_cascade_paths(+Stream, +Options)
%% Emit config cascade path entries. Options: path_type(required|fallback), indent(Str).
emit_cascade_paths(S, Options) :-
    (member(path_type(Type), Options) -> true ; Type = required),
    (member(indent(Indent), Options) -> true ; Indent = ''),
    findall(P, agent_loop_module:config_search_path(P, Type), Paths),
    maplist([Path]>>(agent_loop_module:generate_cascade_path_entry(S, Path, Indent)), Paths).

%% emit_alias_group_entries(+Stream, +Options)
%% Emit Python dict entries for an alias category.
emit_alias_group_entries(S, Options) :-
    member(category(Category), Options),
    agent_loop_module:alias_category(Category, Keys),
    maplist([K]>>(
        agent_loop_module:command_alias(K, V),
        format(S, '    "~w": "~w",~n', [K, V])
    ), Keys).

%% emit_alias_conditions(+Stream, +Options)
%% Emit alias match conditions for command dispatch.
%% Options: aliases(List), match_style(exact|prefix_sp).
emit_alias_conditions(S, Options) :-
    member(aliases(Aliases), Options),
    (member(match_style(prefix_sp), Options) ->
        maplist([A]>>(format(S, ' or cmd.startswith(\'~w \')', [A])), Aliases)
    ;
        maplist([A]>>(format(S, ' or cmd == \'~w\'', [A])), Aliases)
    ).

%% emit_argparse_group_args(+Stream, +Options)
%% Emit parser.add_argument() calls for a group of CLI arguments.
emit_argparse_group_args(S, Options) :-
    member(args(ArgNames), Options),
    maplist([ArgName]>>(
        agent_loop_module:cli_argument(ArgName, Props),
        agent_loop_module:generate_add_argument(S, Props)
    ), ArgNames).

%% emit_backend_helper_fragments(+Stream, +Options)
%% Emit trailing helper method fragments for a backend.
emit_backend_helper_fragments(S, Options) :-
    member(backend(BackendName), Options),
    member(fragments(Frags), Options),
    maplist([F]>>(agent_loop_module:emit_helper_fragment(S, BackendName, F)), Frags).

%% ============================================================================
%% Test Metadata Generation
%% ============================================================================

%% emit_test_metadata/0
%% Writes test_metadata.json with component registry data for parameterized tests.
emit_test_metadata :-
    register_agent_loop_components,
    agent_loop_module:output_path(python, 'test_metadata.json', Path),
    open(Path, write, S),
    %% Collect component data
    findall(M, component(agent_costs, M, model_pricing, _), CostModels),
    findall(N, component(agent_tools, N, tool_handler, _), ToolNames),
    findall(P, component(agent_security, P, security_profile, _), SecProfiles),
    findall(B, component(agent_backends, B, backend, _), BackendNames),
    findall(C, component(agent_commands, C, slash_command, _), CmdNames),
    length(CostModels, NC), length(ToolNames, NT), length(SecProfiles, NS),
    length(BackendNames, NB), length(CmdNames, NCm),
    %% Write JSON
    write(S, '{\n'),
    %% costs
    format(S, '  "costs": {"count": ~w, "models": [', [NC]),
    emit_json_string_list(S, CostModels),
    write(S, ']},\n'),
    %% tools
    format(S, '  "tools": {"count": ~w, "names": [', [NT]),
    emit_json_string_list(S, ToolNames),
    write(S, ']},\n'),
    %% security
    format(S, '  "security": {"count": ~w, "profiles": [', [NS]),
    emit_json_string_list(S, SecProfiles),
    write(S, ']},\n'),
    %% backends
    format(S, '  "backends": {"count": ~w, "names": [', [NB]),
    emit_json_string_list(S, BackendNames),
    write(S, ']},\n'),
    %% commands
    format(S, '  "commands": {"count": ~w, "names": [', [NCm]),
    emit_json_string_list(S, CmdNames),
    write(S, ']},\n'),
    %% destructive_tools
    findall(DT, agent_loop_module:destructive_tool(DT), DestructiveTools),
    write(S, '  "destructive_tools": ['),
    emit_json_string_list(S, DestructiveTools),
    write(S, '],\n'),
    %% module_dependencies
    findall(Src-Tgt, agent_loop_module:module_dependency(Src, Tgt, _), DepPairs),
    write(S, '  "module_dependencies": {'),
    emit_dep_map_json(S, DepPairs),
    write(S, '}\n'),
    write(S, '}\n'),
    close(S),
    format('  Generated test_metadata.json~n', []).

%% emit_json_string_list(+Stream, +List)
%% Write a list of atoms as JSON string array entries (no trailing comma).
emit_json_string_list(_, []).
emit_json_string_list(S, [X]) :- !,
    format(S, '"~w"', [X]).
emit_json_string_list(S, [X|Rest]) :-
    format(S, '"~w", ', [X]),
    emit_json_string_list(S, Rest).

%% emit_dep_map_json(+Stream, +Pairs)
%% Write module dependency pairs as JSON: {"src": ["tgt1", "tgt2"], ...}
emit_dep_map_json(_, []).
emit_dep_map_json(S, Pairs) :-
    %% Group by source
    findall(Src, member(Src-_, Pairs), SrcsDup),
    sort(SrcsDup, Srcs),
    emit_dep_map_entries(S, Srcs, Pairs, first).

emit_dep_map_entries(_, [], _, _).
emit_dep_map_entries(S, [Src|Rest], Pairs, Pos) :-
    (Pos = first -> true ; write(S, ', ')),
    findall(Tgt, member(Src-Tgt, Pairs), Tgts),
    format(S, '"~w": [', [Src]),
    emit_json_string_list(S, Tgts),
    write(S, ']'),
    emit_dep_map_entries(S, Rest, Pairs, subsequent).

%% agent_loop_component_summary/0
%% Registers everything and prints a summary.
agent_loop_component_summary :-
    register_agent_loop_components,
    list_components(agent_tools, Tools),
    list_components(agent_commands, Cmds),
    list_components(agent_backends, Backends),
    list_components(agent_security, Sec),
    list_components(agent_costs, Costs),
    length(Tools, NT), length(Cmds, NC), length(Backends, NB),
    length(Sec, NS), length(Costs, NCo),
    format("~nAgent-loop components registered:~n"),
    format("  Tools:    ~w ~w~n", [NT, Tools]),
    format("  Commands: ~w ~w~n", [NC, Cmds]),
    format("  Backends: ~w ~w~n", [NB, Backends]),
    format("  Security: ~w ~w~n", [NS, Sec]),
    format("  Costs:    ~w~n", [NCo]).

%% emit_predicate_summary/0
%% Documents all generator-to-emit-path mappings.
emit_predicate_summary :-
    format("~n=== Generator-to-Emit-Path Summary ===~n~n"),
    format("Verification: pytest integration tests (primary) + prototype diff (optional)~n~n"),
    format("Generators using component/emit paths:~n"),
    format("  costs.py          -> emit_cost_facts/2 [target(python)]~n"),
    format("  tools.py          -> emit_security_facts/2 [blocked_* fact_types]~n"),
    format("  tools.py          -> emit_tool_dispatch_entries/2 (component iteration)~n"),
    format("  backends/__init__ -> emit_backend_init_imports/2 + emit_backend_init_optional/2~n"),
    format("  backends/<name>   -> emit_backend_module_imports/2 [imports(list)]~n"),
    format("  backends/<name>   -> emit_backend_helper_fragments/2~n"),
    format("  security/__init__ -> emit_security_module_imports/2~n"),
    format("  security/profiles -> emit_security_profile_entries/2~n"),
    format("  context.py        -> emit_context_enums/2, emit_message_fields/2~n"),
    format("  config.py         -> emit_agent_config_fields/2, emit_api_key_env_vars_py/2~n"),
    format("  config.py         -> emit_api_key_files_py/2, emit_default_presets_py/2~n"),
    format("  config.py         -> emit_cascade_paths/2 [path_type(required|fallback)]~n"),
    format("  agent_loop.py     -> emit_audit_levels/2, emit_cli_overrides/2~n"),
    format("  agent_loop.py     -> emit_binding_dispatch_comment/2 (8 bindings)~n"),
    format("  agent_loop.py     -> emit_help_groups/2~n"),
    format("  agent_loop.py     -> emit_alias_conditions/2, emit_argparse_group_args/2~n"),
    format("  README.md         -> emit_readme_sections/2~n"),
    format("  Prolog generators -> emit_tool/command/backend/security/cost_facts/2~n"),
    format("  prolog/config.pl  -> emit_prolog_config_facts/2 (with indexing hints)~n"),
    format("  prolog/backends   -> emit_streaming_capable_facts/2~n"),
    format("  prolog/backends   -> emit_backend_facts/2 (with optimization notes)~n"),
    format("  prolog/commands   -> emit_command_facts/2 (with optimization notes)~n"),
    format("  openrouter_api    -> emit_tool_schemas_py/2~n"),
    format("~nUnified import infrastructure:~n"),
    format("  emit_module_imports/2     -> bare/from/from_relative spec dispatch~n"),
    format("  backend_import_specs/2    -> module_imports + from_imports -> specs~n"),
    format("  security_import_specs/1   -> security_module/3 -> from_relative specs~n"),
    format("  emit_module_dependencies/2 -> module_dependency/3 fact-driven comments~n"),
    format("~nGenerators using raw fact iteration (by design):~n"),
    format("  aliases.py        -> emit_alias_group_entries/2 + structural ordering~n"),
    format("  backends/<name>   -> agent_backend/2 + py_fragment/2~n"),
    format("~n").
