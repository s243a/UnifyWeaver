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
    emit_backend_module_imports/2
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
    forall(agent_loop_module:tool_handler(Name, Handler), (
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
    )).

%% Populate command components from slash_command/4
register_command_components :-
    forall(agent_loop_module:slash_command(Name, MatchType, Opts, Help), (
        declare_component(agent_commands, Name, slash_command, [
            match_type(MatchType),
            options(Opts),
            help_text(Help),
            initialization(eager)
        ])
    )).

%% Populate backend components from backend_factory/2
register_backend_components :-
    forall(agent_loop_module:backend_factory(Name, Spec), (
        declare_component(agent_backends, Name, backend, Spec)
    )).

%% Populate security components from security_profile/2
register_security_components :-
    forall(agent_loop_module:security_profile(Name, Props), (
        declare_component(agent_security, Name, security_profile, Props)
    )).

%% Populate cost components from model_pricing/3
register_cost_components :-
    forall(agent_loop_module:model_pricing(Model, In, Out), (
        (atom(Model) -> ModelAtom = Model ; atom_string(ModelAtom, Model)),
        declare_component(agent_costs, ModelAtom, model_pricing, [
            input_price(In),
            output_price(Out),
            model_string(Model)
        ])
    )).

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
    forall(component(agent_tools, Name, tool_handler, _), (
        (compile_component(agent_tools, Name, [target(prolog), fact_type(tool_spec)], Code) ->
            write(S, Code), nl(S)
        ; true)
    )),
    write(S, '\n'),
    %% tool_handler via compile_component
    write(S, '%% tool_handler(+ToolName, +HandlerPredicate)\n'),
    forall(component(agent_tools, Name, tool_handler, _), (
        compile_component(agent_tools, Name, [target(prolog), fact_type(tool_handler)], Code),
        write(S, Code), nl(S)
    )),
    write(S, '\n'),
    %% destructive_tool via compile_component
    write(S, '%% destructive_tool(+ToolName)\n'),
    forall((component(agent_tools, Name, tool_handler, Cfg),
            member(destructive(true), Cfg)), (
        compile_component(agent_tools, Name, [target(prolog), fact_type(destructive_tool)], Code),
        write(S, Code), nl(S)
    )),
    write(S, '\n'),
    %% tool_description — stays as raw fact iteration (cross-cutting shape)
    write(S, '%% tool_description(+Backend, +ToolName, +Verb, +ParamKey, +DisplayMode)\n'),
    forall(agent_loop_module:tool_description(Backend, TN, Verb, PK, DM), (
        format(S, 'tool_description(~q, ~q, ~q, ~q, ', [Backend, TN, Verb, PK]),
        agent_loop_module:write_prolog_term(S, DM),
        write(S, ').\n')
    )),
    write(S, '\n').

%% emit_command_facts(+Stream, +Options)
%% Emit command-related Prolog facts via the component registry.
emit_command_facts(S, _Options) :-
    %% slash_command via compile_component
    write(S, '%% slash_command(+Name, +MatchType, +Options, +HelpText)\n'),
    forall(component(agent_commands, Name, slash_command, _), (
        compile_component(agent_commands, Name, [target(prolog), fact_type(slash_command)], Code),
        write(S, Code), nl(S)
    )),
    write(S, '\n'),
    %% command_alias — cross-cutting, stays as raw fact iteration
    write(S, '%% command_alias(+Alias, +CanonicalName)\n'),
    forall(agent_loop_module:command_alias(Alias, Canonical), (
        format(S, 'command_alias(~q, ~q).~n', [Alias, Canonical])
    )),
    write(S, '\n'),
    %% slash_command_group — cross-cutting, stays as raw fact iteration
    write(S, '%% slash_command_group(+GroupName, +CommandList)\n'),
    forall(agent_loop_module:slash_command_group(Group, Cmds), (
        format(S, 'slash_command_group(~q, ', [Group]),
        agent_loop_module:write_prolog_term(S, Cmds),
        write(S, ').\n')
    )),
    write(S, '\n').

%% emit_backend_facts(+Stream, +Options)
%% Emit backend-related Prolog facts via the component registry.
emit_backend_facts(S, _Options) :-
    %% agent_backend — rich property list not in registry, stays raw
    write(S, '%% agent_backend(+Name, +Properties)\n'),
    forall(agent_loop_module:agent_backend(Name, Props), (
        format(S, 'agent_backend(~q, ', [Name]),
        agent_loop_module:write_prolog_term(S, Props),
        write(S, ').\n')
    )),
    write(S, '\n'),
    %% backend_factory via compile_component
    write(S, '%% backend_factory(+Name, +FactorySpec)\n'),
    forall(component(agent_backends, Name, backend, _), (
        compile_component(agent_backends, Name, [target(prolog), fact_type(backend_factory)], Code),
        write(S, Code), nl(S)
    )),
    write(S, '\n'),
    %% backend_factory_order — singleton, stays raw
    (agent_loop_module:backend_factory_order(Order) ->
        format(S, 'backend_factory_order(', []),
        agent_loop_module:write_prolog_term(S, Order),
        write(S, ').\n\n')
    ; true),
    %% cli_fallbacks — stays raw
    write(S, '%% cli_fallbacks(+BackendName, +FallbackList)\n'),
    forall(agent_loop_module:cli_fallbacks(Name, Fallbacks), (
        format(S, 'cli_fallbacks(~q, ', [Name]),
        agent_loop_module:write_prolog_term(S, Fallbacks),
        write(S, ').\n')
    )),
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
        write(S, '\n'),
        write(S, '%% security_profile(+Name, +Properties)\n'),
        forall(component(agent_security, Name, security_profile, _), (
            compile_component(agent_security, Name, [target(prolog)], Code),
            write(S, Code), nl(S)
        )),
        write(S, '\n'),
        write(S, '%% blocked_path(+AbsolutePath)\n'),
        forall(agent_loop_module:blocked_path(P), (
            format(S, 'blocked_path(~q).~n', [P])
        )),
        write(S, '\n'),
        write(S, '%% blocked_path_prefix(+Prefix)\n'),
        forall(agent_loop_module:blocked_path_prefix(P), (
            format(S, 'blocked_path_prefix(~q).~n', [P])
        )),
        write(S, '\n'),
        write(S, '%% blocked_home_pattern(+Pattern)\n'),
        forall(agent_loop_module:blocked_home_pattern(P), (
            format(S, 'blocked_home_pattern(~q).~n', [P])
        )),
        write(S, '\n'),
        write(S, '%% blocked_command_pattern(+Regex, +Description)\n'),
        forall(agent_loop_module:blocked_command_pattern(Regex, Desc), (
            format(S, 'blocked_command_pattern(~q, ~q).~n', [Regex, Desc])
        )),
        write(S, '\n')
    ).

%% emit_security_blocked_py(+Stream, +FactType)
%% Emit blocked_* facts as Python set/tuple/list entries.
emit_security_blocked_py(S, blocked_path) :-
    forall(agent_loop_module:blocked_path(P),
        format(S, '    \'~w\',~n', [P])).
emit_security_blocked_py(S, blocked_path_prefix) :-
    forall(agent_loop_module:blocked_path_prefix(P),
        format(S, '    \'~w\',~n', [P])).
emit_security_blocked_py(S, blocked_home_pattern) :-
    forall(agent_loop_module:blocked_home_pattern(P),
        format(S, '    \'~w\',~n', [P])).
emit_security_blocked_py(S, blocked_command_pattern) :-
    forall(agent_loop_module:blocked_command_pattern(Regex, Desc),
        format(S, '    (r\'~w\', "~w"),~n', [Regex, Desc])).

%% emit_cost_facts(+Stream, +Options)
%% Emit cost facts — dispatches on target(python) vs target(prolog).
emit_cost_facts(S, Options) :-
    (member(target(python), Options) ->
        forall(component(agent_costs, Model, model_pricing, _), (
            compile_component(agent_costs, Model, [target(python)], Code),
            write(S, Code), nl(S)
        ))
    ;
        write(S, '%% Indexing hints (SWI-Prolog auto-indexes first argument):\n'),
        emit_indexing_directive(S, model_pricing/3),
        write(S, '\n'),
        write(S, '%% model_pricing(+Model, +InputPricePerMTok, +OutputPricePerMTok)\n'),
        forall(component(agent_costs, Model, model_pricing, _), (
            compile_component(agent_costs, Model, [target(prolog)], Code),
            write(S, Code), nl(S)
        ))
    ).

%% emit_backend_init_imports(+Stream, +Options)
%% Emit Python import lines for non-optional backends.
emit_backend_init_imports(S, _Options) :-
    forall((agent_loop_module:agent_backend(Name, Props),
            \+ member(optional_import(true), Props)), (
        agent_loop_module:resolve_class_name(Name, ClassName),
        agent_loop_module:resolve_file_name(Name, FileName),
        format(S, 'from .~w import ~w~n', [FileName, ClassName])
    )).

%% emit_backend_init_optional(+Stream, +Options)
%% Emit Python try/except import blocks for optional backends.
emit_backend_init_optional(S, _Options) :-
    forall((agent_loop_module:agent_backend(Name, Props),
            member(optional_import(true), Props)), (
        agent_loop_module:resolve_class_name(Name, ClassName),
        agent_loop_module:resolve_file_name(Name, FileName),
        member(description(Desc), Props),
        format(S, '~n# ~w (optional - requires pip package)~n', [Desc]),
        format(S, 'try:~n', []),
        format(S, '    from .~w import ~w~n', [FileName, ClassName]),
        format(S, '    __all__.append(\'~w\')~n', [ClassName]),
        format(S, 'except ImportError:~n', []),
        format(S, '    pass~n', [])
    )).

%% ============================================================================
%% Prolog Target Optimization Helpers
%% ============================================================================

%% emit_indexing_directive(+Stream, +Pred/Arity)
%% Emit an indexing hint comment for predicates that benefit from
%% SWI-Prolog's first-argument indexing (auto-applied to fact tables).
emit_indexing_directive(S, Pred/Arity) :-
    format(S, '%%   ~w/~w: first-argument indexed~n', [Pred, Arity]).

%% emit_optimization_notes(+Stream, +Notes)
%% Emit optimization documentation as Prolog comments.
emit_optimization_notes(S, Notes) :-
    write(S, '%% Optimization notes:\n'),
    forall(member(Note, Notes), (
        format(S, '%%   - ~w~n', [Note])
    )),
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
    forall(agent_loop_module:cli_argument(Name, Opts), (
        format(S, 'cli_argument(~q, ', [Name]),
        agent_loop_module:write_prolog_term(S, Opts),
        write(S, ').\n')
    )),
    write(S, '\n'),
    %% agent_config_field facts
    write(S, '%% agent_config_field(+Name, +Type, +Default, +Description)\n'),
    forall(agent_loop_module:agent_config_field(Name, Type, Default, Desc), (
        format(S, 'agent_config_field(~q, ~q, ~q, ~q).~n', [Name, Type, Default, Desc])
    )),
    write(S, '\n'),
    %% default_agent_preset facts
    write(S, '%% default_agent_preset(+PresetName, +Backend, +Overrides)\n'),
    forall(agent_loop_module:default_agent_preset(Name, Backend, Overrides), (
        format(S, 'default_agent_preset(~q, ~q, ', [Name, Backend]),
        agent_loop_module:write_prolog_term(S, Overrides),
        write(S, ').\n')
    )),
    write(S, '\n'),
    %% api_key_env_var facts
    write(S, '%% api_key_env_var(+Backend, +EnvVar)\n'),
    forall(agent_loop_module:api_key_env_var(Backend, EnvVar), (
        format(S, 'api_key_env_var(~q, ~q).~n', [Backend, EnvVar])
    )),
    write(S, '\n'),
    %% api_key_file facts
    write(S, '%% api_key_file(+Backend, +FilePath)\n'),
    forall(agent_loop_module:api_key_file(Backend, FilePath), (
        format(S, 'api_key_file(~q, ~q).~n', [Backend, FilePath])
    )),
    write(S, '\n'),
    %% example_agent_config facts
    write(S, '%% example_agent_config(+Name, +Backend, +Properties)\n'),
    forall(agent_loop_module:example_agent_config(Name, Backend, Props), (
        format(S, 'example_agent_config(~q, ~q, ', [Name, Backend]),
        agent_loop_module:write_prolog_term(S, Props),
        write(S, ').\n')
    )),
    write(S, '\n'),
    %% config_search_path facts
    write(S, '%% config_search_path(+Path, +Category)\n'),
    forall(agent_loop_module:config_search_path(CPath, Cat), (
        format(S, 'config_search_path(~q, ~q).~n', [CPath, Cat])
    )),
    write(S, '\n'),
    %% config_field_json_default facts
    write(S, '%% config_field_json_default(+FieldName, +JsonDefault)\n'),
    write(S, '%% Maps agent config fields to JSON-safe defaults for code generation.\n'),
    write(S, '%% Special values: positional (name=arg), no_default (data.get with no fallback)\n'),
    forall(agent_loop_module:config_field_json_default(FName, FDefault), (
        format(S, 'config_field_json_default(~q, ~q).~n', [FName, FDefault])
    )),
    write(S, '\n'),
    %% config_dir_file_name facts
    write(S, '%% config_dir_file_name(+FileName)\n'),
    write(S, '%% Standard config file names searched by load_config_from_dir.\n'),
    forall(agent_loop_module:config_dir_file_name(FN), (
        format(S, 'config_dir_file_name(~q).~n', [FN])
    )),
    write(S, '\n'),
    %% audit_profile_level facts
    write(S, '%% audit_profile_level(+Profile, +AuditLevel)\n'),
    write(S, '%% Maps security profile to audit logging level.\n'),
    forall(agent_loop_module:audit_profile_level(P, L), (
        format(S, 'audit_profile_level(~q, ~q).~n', [P, L])
    )),
    write(S, '\n').

%% ============================================================================
%% Python Emit Predicates — context.py, config.py, agent_loop.py
%% ============================================================================

%% emit_context_enums(+Stream, +Options)
%% Emit Python Enum classes from context_enum/3 facts.
emit_context_enums(S, _Options) :-
    forall(agent_loop_module:context_enum(EnumName, DocStr, Values), (
        format(S, 'class ~w(Enum):~n', [EnumName]),
        format(S, '    """~w"""~n', [DocStr]),
        forall(member(value(PyName, StrVal, Comment), Values), (
            format(S, '    ~w = "~w"    # ~w~n', [PyName, StrVal, Comment])
        )),
        write(S, '\n\n')
    )).

%% emit_message_fields(+Stream, +Options)
%% Emit Python dataclass fields from message_field/3 facts.
emit_message_fields(S, _Options) :-
    forall(agent_loop_module:message_field(Name, Type, Default), (
        (Default = none ->
            format(S, '    ~w: ~w~n', [Name, Type])
        ;
            format(S, '    ~w: ~w = ~w~n', [Name, Type, Default])
        )
    )).

%% emit_agent_config_fields(+Stream, +Options)
%% Emit Python dataclass fields from agent_config_field/4 facts.
emit_agent_config_fields(S, _Options) :-
    forall(agent_loop_module:agent_config_field(Name, Type, Default, Comment), (
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
    )).

%% emit_audit_levels(+Stream, +Options)
%% Emit Python dict entries from audit_profile_level/2 facts.
emit_audit_levels(S, _Options) :-
    forall(agent_loop_module:audit_profile_level(Profile, Level), (
        format(S, '        \'~w\': \'~w\',~n', [Profile, Level])
    )).

%% emit_cli_overrides(+Stream, +Options)
%% Emit Python CLI override dispatch from cli_override/3 facts.
emit_cli_overrides(S, _Options) :-
    forall(agent_loop_module:cli_override(Arg, Field, Behavior), (
        agent_loop_module:emit_single_override(S, Arg, Field, Behavior)
    )).

%% ============================================================================
%% Python Emit Predicates — config.py data-driven sections
%% ============================================================================

%% emit_api_key_env_vars_py(+Stream, +Options)
%% Emit Python dict entries from api_key_env_var/2 facts.
emit_api_key_env_vars_py(S, _Options) :-
    forall(agent_loop_module:api_key_env_var(Backend, Var), (
        format(S, '        \'~w\': \'~w\',~n', [Backend, Var])
    )).

%% emit_api_key_files_py(+Stream, +Options)
%% Emit Python dict entries from api_key_file/2 facts.
emit_api_key_files_py(S, _Options) :-
    forall(agent_loop_module:api_key_file(Backend, Path), (
        format(S, '        \'~w\': \'~w\',~n', [Backend, Path])
    )).

%% emit_default_presets_py(+Stream, +Options)
%% Emit Python AgentConfig preset entries from default_agent_preset/3 facts.
emit_default_presets_py(S, _Options) :-
    forall(agent_loop_module:default_agent_preset(Name, Backend, Props), (
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
    )).

%% emit_security_module_imports(+Stream, +Options)
%% Emit Python import lines from security_module/3 facts.
emit_security_module_imports(S, _Options) :-
    forall(agent_loop_module:security_module(Mod, Primary, Extras), (
        AllNames = [Primary|Extras],
        atomic_list_concat(AllNames, ', ', NamesStr),
        format(S, 'from .~w import ~w~n', [Mod, NamesStr])
    )).

%% emit_help_groups(+Stream, +Options)
%% Emit Python help text from slash_command_group/2 facts.
emit_help_groups(S, _Options) :-
    forall(agent_loop_module:slash_command_group(GroupLabel, CmdNames), (
        include(agent_loop_module:is_python_command, CmdNames, PyCmds),
        (PyCmds \= [] ->
            format(S, '~w:~n', [GroupLabel]),
            forall(member(CmdName, PyCmds),
                agent_loop_module:format_help_line(S, CmdName)
            ),
            write(S, '\n')
        ; true)
    )).

%% emit_readme_sections(+Stream, +Options)
%% Emit README markdown for backends and tools.
emit_readme_sections(S, _Options) :-
    write(S, '## Backends\n\n'),
    forall(agent_loop_module:agent_backend(Name, Props), (
        member(description(Desc), Props),
        format(S, '- **~w**: ~w~n', [Name, Desc])
    )),
    write(S, '\n## Tools\n\n'),
    forall(agent_loop_module:tool_spec(Name, Props), (
        member(description(Desc), Props),
        format(S, '- **~w**: ~w~n', [Name, Desc])
    )).

%% emit_backend_module_imports(+Stream, +Options)
%% Emit Python import lines from an imports list in Options.
emit_backend_module_imports(S, Options) :-
    (member(imports(Imports), Options) ->
        forall(member(Imp, Imports), format(S, 'import ~w~n', [Imp]))
    ; true).

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
    write(S, ']}\n'),
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
    format("  tools.py          -> generate_tool_dispatch/1 (component iteration)~n"),
    format("  backends/__init__ -> emit_backend_init_imports/2 + emit_backend_init_optional/2~n"),
    format("  backends/<name>   -> emit_backend_module_imports/2 [imports(list)]~n"),
    format("  security/__init__ -> emit_security_module_imports/2~n"),
    format("  security/profiles -> component(agent_security, ...) iteration~n"),
    format("  context.py        -> emit_context_enums/2, emit_message_fields/2~n"),
    format("  config.py         -> emit_agent_config_fields/2, emit_api_key_env_vars_py/2~n"),
    format("  config.py         -> emit_api_key_files_py/2, emit_default_presets_py/2~n"),
    format("  agent_loop.py     -> emit_audit_levels/2, emit_cli_overrides/2~n"),
    format("  agent_loop.py     -> emit_binding_dispatch_comment/2 (8 bindings)~n"),
    format("  agent_loop.py     -> emit_help_groups/2~n"),
    format("  README.md         -> emit_readme_sections/2~n"),
    format("  Prolog generators -> emit_tool/command/backend/security/cost_facts/2~n"),
    format("  prolog/config.pl  -> emit_prolog_config_facts/2 (with indexing hints)~n"),
    format("~nGenerators using raw fact iteration (by design):~n"),
    format("  aliases.py        -> command_alias/2 (structural ordering with comments)~n"),
    format("  backends/<name>   -> agent_backend/2 + py_fragment/2~n"),
    format("~n").
