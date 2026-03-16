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
    emit_dependency_diagram/2,
    write_lines/2,
    emit_from_components/6,
    resolve_collection_name/5,
    compile_options/4,
    emit_open_delimiter/4,
    emit_close_delimiter/3,
    compute_indent/2,
    extract_target/2,
    maybe_emit_prolog_hints/3,
    indent_atom/2,
    generator_import_specs/2,
    emit_import_specs/2,
    emit_py_dict_from_components/5,
    emit_py_set_from_components/5,
    emit_prolog_facts_from_components/4,
    generator_export_specs/2,
    emit_export_specs/2,
    emit_security_check_predicates/2,
    emit_security_profile_fields/2,
    generator_fragments/2,
    derive_fragment_imports/2,
    validate_generator_imports/2,
    emit_config_section/3,
    emit_prolog_module_header/3,
    emit_prolog_declarations/2,
    emit_prolog_module_skeleton/3,
    prolog_fragment_metadata/2,
    generator_prolog_fragments/2,
    prolog_fragment_category/2,
    prolog_fragment_use_modules/2,
    %% Unified fragment/skeleton system
    emit_module_skeleton/4,
    emit_rust_module_skeleton/3,
    fragment_metadata/3,
    rust_fragment_metadata/2,
    %% Rust target emission helpers
    emit_rust_cost_facts/2,
    emit_rust_tool_facts/2,
    emit_rust_command_facts/2,
    emit_rust_security_facts/2,
    emit_rust_config_data/2,
    %% Generalized data table emission
    rust_data_table/5,
    emit_rust_data_table/3,
    emit_rust_data_tables/3
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
        (member(fact_type(tool_spec), Options) ->
            compile_tool_spec_python(Name, Config, Options, Code)
        ; member(fact_type(destructive_tool), Options) ->
            compile_destructive_tool_python(Name, Config, Options, Code)
        ;
            member(handler(H), Config),
            (member(indent(N), Options) -> true ; N = 4),
            (member(self_prefix(true), Options) ->
                format(atom(HStr), "self.~w", [H])
            ;   atom_string(H, HStr)),
            length(Spaces, N), maplist(=(0' ), Spaces), atom_chars(Indent, Spaces),
            format(atom(Code), "~w'~w': ~w,", [Indent, Name, HStr])
        )
    ; member(target(rust), Options) ->
        (member(fact_type(tool_spec), Options) ->
            compile_tool_spec_rust(Name, Config, Options, Code)
        ; member(fact_type(destructive_tool), Options) ->
            compile_destructive_tool_rust(Name, Config, Options, Code)
        ;
            member(handler(H), Config),
            indent_atom(Options, Indent),
            format(atom(Code), '~w("~w", ~w),', [Indent, Name, H])
        )
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

%% compile_tool_spec_python(+Name, +Config, +Options, -Code)
%% Emit a Python dict entry for TOOL_SPECS with description and parameters.
compile_tool_spec_python(Name, Config, Options, Code) :-
    member(tool_spec_props(Props), Config),
    Props \= [],
    member(description(Desc), Props),
    member(parameters(Params), Props),
    (member(indent(N), Options) -> true ; N = 4),
    length(Spaces, N), maplist(=(0' ), Spaces), atom_chars(Indent, Spaces),
    with_output_to(atom(Code), (
        format('~w"~w": {~n', [Indent, Name]),
        format('~w    "description": "~w",~n', [Indent, Desc]),
        format('~w    "parameters": [~n', [Indent]),
        maplist([param(PName, PType, PReq, PDesc)]>>(
            (PReq = required -> PyReq = 'True' ; PyReq = 'False'),
            format('~w        {"name": "~w", "type": "~w", "required": ~w, "description": "~w"},~n',
                   [Indent, PName, PType, PyReq, PDesc])
        ), Params),
        format('~w    ]~n', [Indent]),
        format('~w},', [Indent])
    )).

%% compile_destructive_tool_python(+Name, +Config, +Options, -Code)
%% Emit a Python set entry for DESTRUCTIVE_TOOLS (only if destructive).
compile_destructive_tool_python(Name, Config, Options, Code) :-
    member(destructive(true), Config),
    (member(indent(N), Options) -> true ; N = 4),
    length(Spaces, N), maplist(=(0' ), Spaces), atom_chars(Indent, Spaces),
    format(atom(Code), '~w"~w",', [Indent, Name]).

%% compile_tool_spec_rust(+Name, +Config, +Options, -Code)
%% Emit a Rust ToolSpec struct literal with full parameter schemas.
compile_tool_spec_rust(Name, Config, Options, Code) :-
    member(tool_spec_props(Props), Config),
    Props \= [],
    member(description(Desc), Props),
    indent_atom(Options, Indent),
    %% Extract parameters (or empty list if none)
    (member(parameters(Params), Props) -> true ; Params = []),
    %% Build parameter array entries
    findall(ParamStr, (
        member(param(PName, PType, PReq, PDesc), Params),
        (PReq == required -> ReqStr = 'true' ; ReqStr = 'false'),
        format(atom(ParamStr), 'ToolParam { name: "~w", param_type: "~w", required: ~w, description: "~w" }', [PName, PType, ReqStr, PDesc])
    ), ParamStrs),
    (ParamStrs == [] ->
        ParamArray = '&[]'
    ;
        atomic_list_concat(ParamStrs, ', ', ParamInner),
        format(atom(ParamArray), '&[~w]', [ParamInner])
    ),
    format(atom(Code), '~wToolSpec { name: "~w", description: "~w", parameters: ~w },', [Indent, Name, Desc, ParamArray]).

%% compile_destructive_tool_rust(+Name, +Config, +Options, -Code)
%% Emit a Rust &str entry for destructive tools array.
compile_destructive_tool_rust(Name, Config, Options, Code) :-
    member(destructive(true), Config),
    indent_atom(Options, Indent),
    format(atom(Code), '~w"~w",', [Indent, Name]).

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
    ; member(target(python), Options) ->
        member(match_type(MT), Config),
        member(help_text(HT), Config),
        (member(options(Opts), Config) -> true ; Opts = []),
        (member(indent(N), Options) -> true ; N = 4),
        length(Spaces, N), maplist(=(0' ), Spaces), atom_chars(Indent, Spaces),
        format(atom(Code), "~w'~w': {'match': '~w', 'options': ~q, 'help': ~q},",
               [Indent, Name, MT, Opts, HT])
    ; member(target(rust), Options) ->
        member(match_type(MT), Config),
        member(help_text(HT), Config),
        indent_atom(Options, Indent),
        format(atom(Code), '~w("~w", CommandSpec { match_type: "~w", help: "~w" }),',
               [Indent, Name, MT, HT])
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
    ; member(target(python), Options) ->
        (member(fact_type(all_entry), Options) ->
            %% Emit __all__ list entry: 'ClassName',
            member(class_name(ClassName), Config),
            indent_atom(Options, Indent),
            \+ member(optional_import(true), Config),
            format(atom(Code), "~w'~w',", [Indent, ClassName])
        ;
            member(class_name(ClassName), Config),
            member(resolve_type(ResolveType), Config),
            indent_atom(Options, Indent),
            format(atom(Code), "~w'~w': {'class': '~w', 'resolve': '~w'},",
                   [Indent, Name, ClassName, ResolveType])
        )
    ; member(target(rust), Options) ->
        member(class_name(ClassName), Config),
        member(resolve_type(ResolveType), Config),
        indent_atom(Options, Indent),
        format(atom(Code), '~w("~w", BackendSpec { class_name: "~w", resolve_type: "~w" }),',
               [Indent, Name, ClassName, ResolveType])
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

agent_security_type:compile_component(Name, Config, Options, Code) :-
    (member(target(python), Options) ->
        (member(indent(N), Options) -> true ; N = 4),
        length(Spaces, N), maplist(=(0' ), Spaces), atom_chars(Indent, Spaces),
        (member(path_validation(PV), Config) -> true ; PV = false),
        (member(command_blocklist(CV), Config) -> true ; CV = false),
        format(atom(Code), "~w'~w': {'path_validation': ~w, 'command_validation': ~w},",
               [Indent, Name, PV, CV])
    ; member(target(rust), Options) ->
        (member(path_validation(PV), Config) -> true ; PV = false),
        (member(command_blocklist(CV), Config) -> true ; CV = false),
        (member(proot_isolation(PI), Config) -> true ; PI = false),
        indent_atom(Options, Indent),
        format(atom(Code), '~w("~w", SecurityProfileSpec { path_validation: ~w, command_validation: ~w, proot_isolation: ~w }),',
               [Indent, Name, PV, CV, PI])
    ; member(target(prolog), Options) ->
        with_output_to(atom(Code), (
            format('security_profile(~q, ', [Name]),
            agent_loop_module:write_prolog_term(current_output, Config),
            write(').')
        ))
    ;
        with_output_to(atom(Code), (
            format('security_profile(~q, ', [Name]),
            agent_loop_module:write_prolog_term(current_output, Config),
            write(').')
        ))
    ).

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
    ; member(target(rust), Options) ->
        indent_atom(Options, Indent),
        format(atom(Code), '~w("~w", Pricing { input: ~w, output: ~w }),', [Indent, Model, In, Out])
    ; member(target(prolog), Options) ->
        format(atom(Code), "model_pricing(~q, ~w, ~w).", [Model, In, Out])
    ;
        format(atom(Code), "model_pricing(~q, ~w, ~w).", [Model, In, Out])
    ).

%% ---- Security Rule Type ----

:- module_transparent agent_security_rule_type:type_info/1.
:- module_transparent agent_security_rule_type:validate_config/1.
:- module_transparent agent_security_rule_type:init_component/2.
:- module_transparent agent_security_rule_type:compile_component/4.

agent_security_rule_type:type_info(info{
    name: 'Agent Security Rule',
    version: '1.0.0',
    description: 'Security blocking rule for paths and commands'
}).

agent_security_rule_type:validate_config(Config) :-
    member(rule_type(_), Config).

agent_security_rule_type:init_component(_Name, _Config).

agent_security_rule_type:compile_component(Name, Config, Options, Code) :-
    member(rule_type(RuleType), Config),
    %% Filter by fact_type if specified — only compile matching rule_type
    (member(fact_type(FT), Options) -> RuleType == FT ; true),
    compile_security_rule(RuleType, Name, Config, Options, Code).

%% compile_security_rule(+RuleType, +Name, +Config, +Options, -Code)
%% Target-polymorphic compilation for security blocking rules.
compile_security_rule(blocked_path, Name, _, Options, Code) :-
    (member(target(python), Options) ->
        indent_atom(Options, Indent),
        format(atom(Code), "~w'~w',", [Indent, Name])
    ; member(target(rust), Options) ->
        indent_atom(Options, Indent),
        format(atom(Code), '~w"~w",', [Indent, Name])
    ;
        format(atom(Code), "blocked_path(~q).", [Name])
    ).

compile_security_rule(blocked_path_prefix, Name, _, Options, Code) :-
    (member(target(python), Options) ->
        indent_atom(Options, Indent),
        format(atom(Code), "~w'~w',", [Indent, Name])
    ; member(target(rust), Options) ->
        indent_atom(Options, Indent),
        format(atom(Code), '~w"~w",', [Indent, Name])
    ;
        format(atom(Code), "blocked_path_prefix(~q).", [Name])
    ).

compile_security_rule(blocked_home_pattern, Name, _, Options, Code) :-
    (member(target(python), Options) ->
        indent_atom(Options, Indent),
        format(atom(Code), "~w'~w',", [Indent, Name])
    ; member(target(rust), Options) ->
        indent_atom(Options, Indent),
        format(atom(Code), '~w"~w",', [Indent, Name])
    ;
        format(atom(Code), "blocked_home_pattern(~q).", [Name])
    ).

compile_security_rule(blocked_command_pattern, _, Config, Options, Code) :-
    member(regex(Regex), Config),
    member(description(Desc), Config),
    (member(target(python), Options) ->
        indent_atom(Options, Indent),
        format(atom(Code), "~w(r'~w', \"~w\"),", [Indent, Regex, Desc])
    ; member(target(rust), Options) ->
        indent_atom(Options, Indent),
        format(atom(Code), '~w(r"~w", "~w"),', [Indent, Regex, Desc])
    ;
        format(atom(Code), "blocked_command_pattern(~q, ~q).", [Regex, Desc])
    ).

%% indent_atom(+Options, -Indent)
%% Compute indentation string from indent option (default 4 spaces).
indent_atom(Options, Indent) :-
    (member(indent(N), Options) -> true ; N = 4),
    length(Spaces, N), maplist(=(0' ), Spaces), atom_chars(Indent, Spaces).

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
    ]),
    define_category(agent_security_rules, "Agent-loop security blocking rules", [
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
    ]),
    register_component_type(agent_security_rules, security_rule, agent_security_rule_type, [
        description("Security blocking rule (path, prefix, pattern)")
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
    maplist([Name-Spec]>>(
        %% Merge optional_import from agent_backend into component config
        %% Match by class_name since agent_backend names differ from backend_factory names
        (member(class_name(CN), Spec),
         agent_loop_module:agent_backend(_, ABProps),
         member(class_name(CN), ABProps),
         member(optional_import(true), ABProps) ->
            MergedSpec = [optional_import(true)|Spec]
        ;
            MergedSpec = Spec
        ),
        declare_component(agent_backends, Name, backend, MergedSpec)
    ), Pairs).

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

%% Populate security rule components from blocked_* facts
register_security_rule_components :-
    findall(P, agent_loop_module:blocked_path(P), BPs),
    maplist([P]>>(declare_component(agent_security_rules, P, security_rule,
        [rule_type(blocked_path)])), BPs),
    findall(P, agent_loop_module:blocked_path_prefix(P), BPPs),
    maplist([P]>>(declare_component(agent_security_rules, P, security_rule,
        [rule_type(blocked_path_prefix)])), BPPs),
    findall(P, agent_loop_module:blocked_home_pattern(P), BHPs),
    maplist([P]>>(declare_component(agent_security_rules, P, security_rule,
        [rule_type(blocked_home_pattern)])), BHPs),
    findall(Regex-Desc, agent_loop_module:blocked_command_pattern(Regex, Desc), BCPs),
    maplist([Regex-Desc]>>(
        atom_string(RegexAtom, Regex),
        atom_string(DescAtom, Desc),
        declare_component(agent_security_rules, RegexAtom, security_rule,
            [rule_type(blocked_command_pattern), regex(RegexAtom), description(DescAtom)])
    ), BCPs).

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
    register_cost_components,
    register_security_rule_components.

%% ============================================================================
%% Stream Emitters — Write Prolog facts via the component registry
%% ============================================================================

%% emit_tool_facts(+Stream, +Options)
%% Emit tool-related Prolog facts via the component registry.
emit_tool_facts(S, _Options) :-
    %% tool_spec via unified emit
    write(S, '%% tool_spec(+ToolName, +Properties)\n'),
    emit_from_components(S, agent_tools, tool_handler/2, prolog, facts, [fact_type(tool_spec)]),
    write(S, '\n'),
    %% tool_handler via unified emit
    write(S, '%% tool_handler(+ToolName, +HandlerPredicate)\n'),
    emit_from_components(S, agent_tools, tool_handler/2, prolog, facts, [fact_type(tool_handler)]),
    write(S, '\n'),
    %% destructive_tool via unified emit
    write(S, '%% destructive_tool(+ToolName)\n'),
    emit_from_components(S, agent_tools, tool_handler/2, prolog, facts, [fact_type(destructive_tool)]),
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
    %% slash_command via unified emit
    write(S, '%% slash_command(+Name, +MatchType, +Options, +HelpText)\n'),
    emit_from_components(S, agent_commands, slash_command/4, prolog, facts, [fact_type(slash_command)]),
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
    write(S, '\n'),
    %% command_action — data-driven dispatch for commands without handlers
    write(S, '%% command_action(+Command, +Action) — data-driven dispatch\n'),
    findall(Cmd-Act, agent_loop_module:prolog_command_action(Cmd, Act), ActionPairs),
    maplist([Cmd-Act]>>(format(S, 'command_action(~q, ~q).~n', [Cmd, Act])), ActionPairs),
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
    %% backend_factory via unified emit
    write(S, '%% backend_factory(+Name, +FactorySpec)\n'),
    emit_from_components(S, agent_backends, backend_factory/2, prolog, facts, [fact_type(backend_factory)]),
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
%% Emit security-related facts — target-polymorphic via extract_target.
emit_security_facts(S, Options) :-
    extract_target(Options, Target),
    maybe_emit_prolog_hints(S, Target, [
        security_profile/2, blocked_path/1, blocked_path_prefix/1,
        blocked_home_pattern/1, blocked_command_pattern/2
    ]),
    %% If caller requests a specific fact_type, emit only that
    (member(fact_type(FT), Options) ->
        emit_from_components(S, agent_security_rules, _, Target, entries,
            [fact_type(FT)])
    ;
        %% Prolog path needs security_profile + all rule types with headers
        (Target == prolog ->
            write(S, '%% security_profile(+Name, +Properties)\n'),
            emit_from_components(S, agent_security, security_profile/2, prolog, facts, []),
            write(S, '\n')
        ; true),
        emit_security_rule_group(S, Target, blocked_path, 'blocked_path(+AbsolutePath)'),
        emit_security_rule_group(S, Target, blocked_path_prefix, 'blocked_path_prefix(+Prefix)'),
        emit_security_rule_group(S, Target, blocked_home_pattern, 'blocked_home_pattern(+Pattern)'),
        emit_security_rule_group(S, Target, blocked_command_pattern, 'blocked_command_pattern(+Regex, +Description)')
    ).

%% emit_security_rule_group(+Stream, +Target, +RuleType, +Header)
%% Emit one group of security rules with an optional Prolog header comment.
emit_security_rule_group(S, Target, RuleType, Header) :-
    (Target == prolog ->
        Format = facts,
        format(S, '%% ~w~n', [Header])
    ;
        Format = entries
    ),
    emit_from_components(S, agent_security_rules, _, Target, Format,
        [fact_type(RuleType)]),
    (Target == prolog -> write(S, '\n') ; true).

%% emit_cost_facts(+Stream, +Options)
%% Emit cost facts — target-polymorphic via extract_target + maybe_emit_prolog_hints.
emit_cost_facts(S, Options) :-
    extract_target(Options, Target),
    maybe_emit_prolog_hints(S, Target, [model_pricing/3]),
    (Target == prolog ->
        write(S, '%% model_pricing(+Model, +InputPricePerMTok, +OutputPricePerMTok)\n'),
        emit_from_components(S, agent_costs, model_pricing/3, Target, facts, Options)
    ;
        emit_from_components(S, agent_costs, model_pricing/3, Target, entries, Options)
    ).

%% =============================================================================
%% Rust Target — Declarative Data Table Specifications
%% =============================================================================

%% rust_data_table(+TableName, +Category, +ElementType, +FactType, +Options)
%% Declarative specification for Rust static array tables backed by the
%% component registry. Each fact maps to a `pub static NAME: &[Type] = &[...];`
%% block in generated Rust code.
rust_data_table('PRICING', agent_costs, '(&str, Pricing)', none, []).
rust_data_table('TOOL_SPECS', agent_tools, 'ToolSpec', tool_spec, []).
rust_data_table('DESTRUCTIVE_TOOLS', agent_tools, '&str', destructive_tool, []).
rust_data_table('SLASH_COMMANDS', agent_commands, '(&str, CommandSpec)', none, []).
rust_data_table('SECURITY_PROFILES', agent_security, '(&str, SecurityProfileSpec)', none, []).
rust_data_table('BLOCKED_PATHS', agent_security_rules, '&str', blocked_path, []).
rust_data_table('BLOCKED_PATH_PREFIXES', agent_security_rules, '&str', blocked_path_prefix, []).
rust_data_table('BLOCKED_HOME_PATTERNS', agent_security_rules, '&str', blocked_home_pattern, []).
rust_data_table('BLOCKED_COMMAND_PATTERNS', agent_security_rules, '(&str, &str)', blocked_command_pattern, []).

%% emit_rust_data_table(+Stream, +TableName, +Options)
%% Emit a single Rust static array from its declarative specification.
emit_rust_data_table(S, TableName, Options) :-
    rust_data_table(TableName, Category, ElemType, FactType, TableOpts),
    format(S, 'pub static ~w: &[~w] = &[~n', [TableName, ElemType]),
    (FactType == none ->
        BaseOpts = [target(rust), indent(4)]
    ;
        BaseOpts = [target(rust), indent(4), fact_type(FactType)]
    ),
    append(BaseOpts, TableOpts, CompileOpts0),
    append(CompileOpts0, Options, CompileOpts),
    findall(Name, component(Category, Name, _, _), Names),
    maplist([Name]>>(
        (compile_component(Category, Name, CompileOpts, Code) ->
            write(S, Code), nl(S)
        ; true)
    ), Names),
    write(S, '];\n\n').

%% emit_rust_data_tables(+Stream, +TableNames, +Options)
%% Emit multiple Rust static arrays.
emit_rust_data_tables(S, TableNames, Options) :-
    maplist([TN]>>(emit_rust_data_table(S, TN, Options)), TableNames).

%% =============================================================================
%% Rust Target — Component Emission Helpers (thin wrappers)
%% =============================================================================

%% emit_rust_cost_facts(+Stream, +Options)
emit_rust_cost_facts(S, _Options) :-
    emit_rust_data_table(S, 'PRICING', []).

%% emit_rust_tool_facts(+Stream, +Options)
emit_rust_tool_facts(S, _Options) :-
    emit_rust_data_tables(S, ['TOOL_SPECS', 'DESTRUCTIVE_TOOLS'], []).

%% emit_rust_command_facts(+Stream, +Options)
emit_rust_command_facts(S, _Options) :-
    emit_rust_data_table(S, 'SLASH_COMMANDS', []).

%% emit_rust_security_facts(+Stream, +Options)
emit_rust_security_facts(S, _Options) :-
    emit_rust_data_tables(S, [
        'SECURITY_PROFILES', 'BLOCKED_PATHS', 'BLOCKED_PATH_PREFIXES',
        'BLOCKED_HOME_PATTERNS', 'BLOCKED_COMMAND_PATTERNS'
    ], []).

%% emit_rust_config_data(+Stream, +Options)
%% Emit all config fact tables as Rust static arrays.
%% Delegates to emit_config_section/3 for each section.
emit_rust_config_data(S, Options) :-
    emit_config_section(S, cli_arguments, Options),
    emit_config_section(S, agent_config_fields, Options),
    emit_config_section(S, api_key_env_vars, Options),
    emit_config_section(S, api_key_files, Options),
    emit_config_section(S, config_search_paths, Options),
    emit_config_section(S, config_dir_file_names, Options),
    emit_config_section(S, default_presets, Options),
    emit_config_section(S, audit_levels, Options),
    emit_config_section(S, streaming_capable, Options),
    emit_config_section(S, cli_overrides, Options).

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
    format("  agent_loop.py     -> emit_binding_dispatch_comment/2 (11 bindings)~n"),
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
    format("~nBinding-driven emission:~n"),
    format("  costs.py          -> emit_py_dict_from_components/5 (binding_dict_name for DEFAULT_PRICING)~n"),
    format("  tools_generated   -> binding_dict_name/3 for DESTRUCTIVE_TOOLS~n"),
    format("  agent_loop.py     -> binding_dict_name/3 for audit_levels~n"),
    format("~nData-driven helpers:~n"),
    format("  write_lines/2              -> emit list of strings as lines~n"),
    format("  emit_py_dict_from_components/5 -> binding + component-driven Python dict~n"),
    format("~nGenerators using raw fact iteration (by design):~n"),
    format("  aliases.py        -> emit_alias_group_entries/2 + structural ordering~n"),
    format("  backends/<name>   -> agent_backend/2 + py_fragment/2~n"),
    format("~n").

%% ============================================================================
%% Data-driven Emission Helpers
%% ============================================================================

%% write_lines(+Stream, +Lines)
%% Write a list of strings as lines to Stream.
%% Each element is written followed by a newline.
write_lines(_, []).
write_lines(S, [Line|Rest]) :-
    write(S, Line),
    nl(S),
    write_lines(S, Rest).

%% ============================================================================
%% Declarative Import Specifications
%% ============================================================================

%% generator_import_specs(+Generator, -Specs)
%% Declarative import specs per generator file.
%% Spec types: bare(Module) | from(Module, [Names])
generator_import_specs(tools, [
    bare(json), bare(os), bare(re), bare(subprocess),
    from(dataclasses, [dataclass, field]),
    from(pathlib, ['Path']),
    from('backends.base', ['ToolCall']),
    from('security.audit', ['AuditLogger']),
    from('security.profiles', ['SecurityProfile', get_profile]),
    from('security.proxy', ['CommandProxyManager'])
]).
generator_import_specs(aliases, [
    bare(json),
    from(pathlib, ['Path']),
    from(typing, ['Callable'])
]).
generator_import_specs(backends_base, [
    from(dataclasses, [dataclass, field]),
    from(typing, ['Any']),
    from(abc, ['ABC', abstractmethod])
]).
generator_import_specs(costs, [
    from(dataclasses, [dataclass, field]),
    from(datetime, [datetime]),
    from(typing, ['Any']),
    bare(json), bare(os), bare(sys), bare(time),
    from(pathlib, ['Path']),
    from('urllib.request', [urlopen, 'Request']),
    from('urllib.error', ['URLError'])
]).
generator_import_specs(context, [
    from(dataclasses, [dataclass, field]),
    from(typing, ['Literal']),
    from(enum, ['Enum'])
]).
generator_import_specs(config, [
    bare(os), bare(json),
    from(pathlib, ['Path']),
    from(dataclasses, [dataclass, field]),
    from(typing, ['Any'])
]).
generator_import_specs(security_profiles, [
    from(dataclasses, [dataclass, field])
]).

%% emit_import_specs(+Stream, +Specs)
%% Emit Python import statements from declarative specs.
%% Delegates to emit_module_imports/2 (supports bare, from, from_relative).
emit_import_specs(S, Specs) :-
    emit_module_imports(S, Specs).

%% ============================================================================
%% Declarative Export Specifications
%% ============================================================================

%% generator_export_specs(+Generator, -Exports)
%% Declarative __all__ entries for a generator, data-driven from facts.
generator_export_specs(security_init, Exports) :-
    findall(Export, (
        agent_loop_module:security_module(_, Primary, Extras),
        member(Export, [Primary|Extras])
    ), Exports).

generator_export_specs(backends_init, Exports) :-
    StaticExports = ['AgentBackend', 'AgentResponse', 'ToolCall', 'AsyncApiBackend'],
    findall(ClassName, (
        agent_loop_module:backend_factory(_, Spec),
        member(class_name(ClassName), Spec),
        %% Exclude optional-import backends (they have try/except)
        \+ (agent_loop_module:agent_backend(_, ABProps),
            member(class_name(ClassName), ABProps),
            member(optional_import(true), ABProps))
    ), DynamicExports),
    append(StaticExports, DynamicExports, Exports).

%% emit_export_specs(+Stream, +Exports)
%% Emit a Python __all__ = [...] block from a list of export names.
emit_export_specs(S, Exports) :-
    write(S, '__all__ = [\n'),
    maplist([E]>>(format(S, "    '~w',~n", [E])), Exports),
    write(S, ']\n').

%% ============================================================================
%% Declarative Security Profile Fields
%% ============================================================================

%% emit_security_profile_fields(+Stream, +Options)
%% Target-polymorphic emission of security profile field schema.
%% Python: dataclass fields with type annotations, defaults, and layer comments.
%% Prolog: security_profile_field/4 facts.
emit_security_profile_fields(S, Options) :-
    extract_target(Options, Target),
    findall(field(Name, Type, Default, Props),
        agent_loop_module:security_profile_field(Name, Type, Default, Props),
        Fields),
    (Target == python ->
        emit_profile_fields_loop(S, Fields, none)
    ;
        maplist([field(Name, Type, Default, Props)]>>(
            format(S, "security_profile_field(~q, ~q, ~q, ~q).~n",
                   [Name, Type, Default, Props])
        ), Fields)
    ).

emit_profile_fields_loop(_, [], _).
emit_profile_fields_loop(S, [field(Name, Type, Default, Props)|Rest], PrevLayer) :-
    %% Determine current layer (or 'none' if no layer property)
    (member(layer(L), Props) -> CurLayer = L ; CurLayer = none),
    %% Emit blank line + comment header when entering a new layer group
    (CurLayer \= PrevLayer, member(comment(Comment), Props) ->
        nl(S), format(S, "    # ~w~n", [Comment])
    ; true),
    %% Emit the field line
    (Default == required ->
        format(S, "    ~w: ~w~n", [Name, Type])
    ; member(inline_comment(IC), Props) ->
        format(S, "    ~w: ~w = ~w  # ~w~n", [Name, Type, Default, IC])
    ;
        format(S, "    ~w: ~w = ~w~n", [Name, Type, Default])
    ),
    emit_profile_fields_loop(S, Rest, CurLayer).

%% ============================================================================
%% Fragment Metadata
%% ============================================================================

%% py_fragment_metadata(+FragmentName, +Properties)
%% Metadata annotations for py_fragment/2 facts.
%% Properties: category(Cat), target(Target), imports([ImportSpec, ...])
%% Enables future composition where generators auto-derive imports from fragments.
py_fragment_metadata(tools_handler_class_body, [
    category(tools), target(python),
    imports([from('backends.base', ['ToolCall'])])
]).
py_fragment_metadata(tools_security_config, [
    category(security), target(python),
    imports([from('security.audit', ['AuditLogger'])])
]).
py_fragment_metadata(tools_validate_path, [
    category(security), target(python),
    imports([bare(os)])
]).
py_fragment_metadata(tools_is_command_blocked, [
    category(security), target(python),
    imports([bare(re)])
]).
py_fragment_metadata(context_manager_class, [
    category(context), target(python),
    imports([from(dataclasses, [dataclass, field])])
]).
py_fragment_metadata(context_helpers, [
    category(context), target(python),
    imports([])
]).
py_fragment_metadata(config_resolve_api_key_header, [
    category(config), target(python),
    imports([bare(os)])
]).
py_fragment_metadata(config_load_config, [
    category(config), target(python),
    imports([bare(json), from(pathlib, ['Path'])])
]).
py_fragment_metadata(security_audit_module, [
    category(security), target(python),
    imports([bare(os), bare(json)])
]).
py_fragment_metadata(agent_loop_imports, [
    category(agent_loop), target(python),
    imports([bare(os), bare(sys), bare(json)])
]).
py_fragment_metadata(security_proxy_module, [
    category(security), target(python),
    imports([bare(os), bare(re), bare(shlex), from(dataclasses, [dataclass, field])])
]).
py_fragment_metadata(security_path_proxy_module, [
    category(security), target(python),
    imports([bare(os), bare(stat), from(pathlib, ['Path']), from(dataclasses, [dataclass, field])])
]).
py_fragment_metadata(security_proot_sandbox_module, [
    category(security), target(python),
    imports([bare(os), bare(shlex), bare(shutil), from(dataclasses, [dataclass, field]), from(pathlib, ['Path'])])
]).
py_fragment_metadata(aliases_class_header, [category(aliases), target(python), imports([])]).
py_fragment_metadata(aliases_class_footer, [category(aliases), target(python), imports([])]).
py_fragment_metadata(cost_usage_record, [category(costs), target(python), imports([from(dataclasses, [dataclass])])]).
py_fragment_metadata(cost_tracker_class_def, [category(costs), target(python), imports([from(dataclasses, [dataclass, field])])]).
py_fragment_metadata(cost_tracker_methods, [category(costs), target(python),
    imports([from(datetime, [datetime]), from(pathlib, ['Path']), bare(json)])]).
py_fragment_metadata(cost_openrouter, [category(costs), target(python),
    imports([bare(os), bare(sys), bare(time), bare(json), from(pathlib, ['Path']),
             from('urllib.request', [urlopen, 'Request']), from('urllib.error', ['URLError'])])]).

%% fragment_imports(+FragmentName, -ImportSpecs)
%% Get the import specs for a named fragment.
fragment_imports(Name, Imports) :-
    py_fragment_metadata(Name, Props),
    member(imports(Imports), Props).

%% fragment_category(+FragmentName, -Category)
%% Get the category of a named fragment.
fragment_category(Name, Category) :-
    py_fragment_metadata(Name, Props),
    member(category(Category), Props).

%% ============================================================================
%% Fragment-Driven Import Derivation
%% ============================================================================

%% generator_fragments(+Generator, -FragmentNames)
%% Declares which py_fragment/2 names a generator uses.
generator_fragments(tools, [tools_handler_class_body, tools_security_config,
                            tools_validate_path, tools_is_command_blocked]).
generator_fragments(costs, [cost_usage_record, cost_tracker_class_def,
                            cost_tracker_methods, cost_openrouter]).
generator_fragments(config, [config_resolve_api_key_header, config_load_config]).
generator_fragments(context, [context_manager_class, context_helpers]).
generator_fragments(security_audit, [security_audit_module]).
generator_fragments(security_proxy, [security_proxy_module]).
generator_fragments(security_path_proxy, [security_path_proxy_module]).
generator_fragments(security_proot, [security_proot_sandbox_module]).
generator_fragments(aliases, [aliases_class_header, aliases_class_footer]).
generator_fragments(agent_loop_main, [
    handler_iterations_command, handler_backend_command,
    handler_save_command, handler_load_command,
    handler_sessions_command, handler_format_command,
    handler_export_command, handler_cost_command,
    handler_search_command, handler_stream_command,
    handler_aliases_command, handler_templates_command,
    handler_history_command, handler_undo_command,
    handler_delete_command, handler_edit_command,
    handler_replay_command
]).

%% ============================================================================
%% Prolog Fragment Metadata
%% ============================================================================

%% prolog_fragment_metadata(+FragmentName, +Properties)
%% Metadata annotations for prolog_fragment/2 facts.
%% Properties: category(Cat), target(prolog), use_modules([ModSpec, ...])
%% Parallel to py_fragment_metadata/2 for Python fragments.

prolog_fragment_metadata(cost_tracker_impl, [
    category(costs), target(prolog), use_modules([])
]).
prolog_fragment_metadata(config_parse_cli, [
    category(config), target(prolog), use_modules([library(optparse)])
]).
prolog_fragment_metadata(config_load_config, [
    category(config), target(prolog), use_modules([library(json)])
]).
prolog_fragment_metadata(config_resolve_api_key, [
    category(config), target(prolog), use_modules([])
]).
prolog_fragment_metadata(commands_resolve, [
    category(commands), target(prolog), use_modules([])
]).
prolog_fragment_metadata(commands_handle_slash, [
    category(commands), target(prolog), use_modules([])
]).
prolog_fragment_metadata(tools_execute_dispatch, [
    category(tools), target(prolog),
    use_modules([library(process), library(readutil), library(time)])
]).
prolog_fragment_metadata(tools_schema, [
    category(tools), target(prolog), use_modules([])
]).
prolog_fragment_metadata(tools_describe, [
    category(tools), target(prolog), use_modules([])
]).
prolog_fragment_metadata(tools_confirm, [
    category(tools), target(prolog), use_modules([])
]).
prolog_fragment_metadata(backends_create_backend, [
    category(backends), target(prolog), use_modules([])
]).
prolog_fragment_metadata(backends_retry_config, [
    category(backends), target(prolog), use_modules([])
]).
prolog_fragment_metadata(backends_format_api_error, [
    category(backends), target(prolog), use_modules([])
]).
prolog_fragment_metadata(backends_retry_call, [
    category(backends), target(prolog), use_modules([])
]).
prolog_fragment_metadata(backends_send_request, [
    category(backends), target(prolog), use_modules([])
]).
prolog_fragment_metadata(backends_send_request_raw_api, [
    category(backends), target(prolog),
    use_modules([library(http/http_open), library(json)])
]).
prolog_fragment_metadata(backends_send_request_raw_cli, [
    category(backends), target(prolog), use_modules([library(process)])
]).
prolog_fragment_metadata(backends_extract_response, [
    category(backends), target(prolog), use_modules([])
]).
prolog_fragment_metadata(backends_streaming_dispatch, [
    category(backends), target(prolog), use_modules([])
]).
prolog_fragment_metadata(backends_streaming_ndjson, [
    category(backends), target(prolog),
    use_modules([library(http/http_open), library(json)])
]).
prolog_fragment_metadata(backends_streaming_openai, [
    category(backends), target(prolog),
    use_modules([library(http/http_open), library(json)])
]).
prolog_fragment_metadata(backends_streaming_anthropic, [
    category(backends), target(prolog),
    use_modules([library(http/http_open), library(json)])
]).
prolog_fragment_metadata(backends_tc_delta, [
    category(backends), target(prolog), use_modules([])
]).
prolog_fragment_metadata(backends_sse_parser, [
    category(backends), target(prolog), use_modules([])
]).
prolog_fragment_metadata(agent_loop_init_state, [
    category(agent_loop), target(prolog), use_modules([])
]).
prolog_fragment_metadata(agent_loop_entry, [
    category(agent_loop), target(prolog), use_modules([])
]).
prolog_fragment_metadata(agent_loop_repl_core, [
    category(agent_loop), target(prolog), use_modules([])
]).
prolog_fragment_metadata(agent_loop_process_input, [
    category(agent_loop), target(prolog), use_modules([])
]).
prolog_fragment_metadata(agent_loop_response, [
    category(agent_loop), target(prolog), use_modules([])
]).
prolog_fragment_metadata(agent_loop_actions, [
    category(agent_loop), target(prolog), use_modules([])
]).
prolog_fragment_metadata(agent_loop_helpers, [
    category(agent_loop), target(prolog), use_modules([])
]).
prolog_fragment_metadata(agent_loop_export_search, [
    category(agent_loop), target(prolog),
    use_modules([library(json), library(filesex)])
]).
prolog_fragment_metadata(agent_loop_context, [
    category(agent_loop), target(prolog), use_modules([])
]).

%% rust_fragment_metadata(+FragmentName, +Properties)
%% Metadata annotations for rust_fragment/2 facts.
:- discontiguous rust_fragment_metadata/2.

%% =============================================================================
%% Unified Fragment Metadata — fragment_metadata/3
%% =============================================================================
%% Provides target-polymorphic access to all fragment metadata.

fragment_metadata(prolog, Name, Meta) :- prolog_fragment_metadata(Name, Meta).
fragment_metadata(rust, Name, Meta) :- rust_fragment_metadata(Name, Meta).

%% prolog_fragment_category(+Name, -Category)
prolog_fragment_category(Name, Category) :-
    prolog_fragment_metadata(Name, Props),
    member(category(Category), Props).

%% prolog_fragment_use_modules(+Name, -UseModules)
prolog_fragment_use_modules(Name, UseModules) :-
    prolog_fragment_metadata(Name, Props),
    member(use_modules(UseModules), Props).

%% generator_prolog_fragments(+Generator, -FragmentNames)
%% Declares which prolog_fragment/2 names a Prolog generator uses.
generator_prolog_fragments(costs, [cost_tracker_impl]).
generator_prolog_fragments(config, [config_parse_cli, config_load_config,
                                    config_resolve_api_key]).
generator_prolog_fragments(commands, [commands_resolve, commands_handle_slash]).
generator_prolog_fragments(tools, [tools_execute_dispatch, tools_schema,
                                   tools_describe, tools_confirm]).
generator_prolog_fragments(backends, [backends_create_backend, backends_retry_config,
                                      backends_format_api_error, backends_retry_call,
                                      backends_send_request, backends_send_request_raw_api,
                                      backends_send_request_raw_cli, backends_extract_response,
                                      backends_streaming_dispatch, backends_streaming_ndjson,
                                      backends_streaming_openai, backends_streaming_anthropic,
                                      backends_tc_delta, backends_sse_parser]).
generator_prolog_fragments(agent_loop, [agent_loop_init_state, agent_loop_entry,
                                        agent_loop_repl_core, agent_loop_process_input,
                                        agent_loop_response, agent_loop_actions,
                                        agent_loop_helpers, agent_loop_export_search,
                                        agent_loop_context]).

%% derive_fragment_imports(+Generator, -DerivedImports)
%% Collects all import specs from a generator's fragments and deduplicates.
derive_fragment_imports(Generator, DerivedImports) :-
    generator_fragments(Generator, FragNames),
    findall(Spec, (
        member(FN, FragNames),
        fragment_imports(FN, FragImports),
        member(Spec, FragImports)
    ), AllSpecs),
    sort(AllSpecs, DerivedImports).

%% validate_generator_imports(+Generator, -Warnings)
%% Compares derived imports against declared generator_import_specs.
%% Returns a list of missing(Spec) for specs in fragments but not declared.
validate_generator_imports(Generator, Warnings) :-
    derive_fragment_imports(Generator, Derived),
    (generator_import_specs(Generator, Declared) -> true ; Declared = []),
    findall(missing(Spec), (
        member(Spec, Derived),
        \+ member(Spec, Declared)
    ), Warnings).

%% ============================================================================
%% Target-Polymorphic Config Section Emitters
%% ============================================================================

%% emit_config_section(+Stream, +Section, +Options)
%% Unified config section emission — delegates to Python, Rust, or Prolog target.
emit_config_section(S, api_key_env_vars, Options) :-
    extract_target(Options, Target),
    (Target == python ->
        emit_api_key_env_vars_py(S, Options)
    ; Target == rust ->
        write(S, 'pub static API_KEY_ENV_VARS: &[ApiKeyMapping] = &[\n'),
        findall(B-V, agent_loop_module:api_key_env_var(B, V), Pairs),
        maplist([B-V]>>(
            format(S, '    ApiKeyMapping { backend: "~w", env_var: "~w" },~n', [B, V])
        ), Pairs),
        write(S, '];\n\n')
    ;
        findall(B-V, agent_loop_module:api_key_env_var(B, V), Pairs),
        maplist([B-V]>>(format(S, "api_key_env_var(~q, ~q).~n", [B, V])), Pairs)
    ).
emit_config_section(S, api_key_files, Options) :-
    extract_target(Options, Target),
    (Target == python ->
        emit_api_key_files_py(S, Options)
    ; Target == rust ->
        write(S, 'pub static API_KEY_FILE_PATHS: &[ApiKeyFilePath] = &[\n'),
        findall(B-P, agent_loop_module:api_key_file(B, P), Pairs),
        maplist([B-P]>>(
            format(S, '    ApiKeyFilePath { backend: "~w", file_path: "~w" },~n', [B, P])
        ), Pairs),
        write(S, '];\n\n')
    ;
        findall(B-F, agent_loop_module:api_key_file(B, F), Pairs),
        maplist([B-F]>>(format(S, "api_key_file(~q, ~q).~n", [B, F])), Pairs)
    ).
emit_config_section(S, default_presets, Options) :-
    extract_target(Options, Target),
    (Target == python ->
        emit_default_presets_py(S, Options)
    ; Target == rust ->
        write(S, 'pub static DEFAULT_PRESETS: &[DefaultPreset] = &[\n'),
        findall(dp(N,B,O), agent_loop_module:default_agent_preset(N, B, O), DPs),
        maplist([dp(N,B,O)]>>(
            term_to_atom(O, OStr),
            atom_string(OStr, OString),
            agent_loop_module:replace_all_sub(OString, "\"", "\\\"", OEscaped),
            format(S, '    DefaultPreset { name: "~w", backend: "~w", overrides: "~w" },~n',
                   [N, B, OEscaped])
        ), DPs),
        write(S, '];\n\n')
    ;
        findall(N-B-P, agent_loop_module:default_agent_preset(N, B, P), Presets),
        maplist([N-B-P]>>(format(S, "default_agent_preset(~q, ~q, ~q).~n", [N, B, P])), Presets)
    ).

emit_config_section(S, agent_config_fields, Options) :-
    extract_target(Options, Target),
    (Target == python ->
        emit_agent_config_fields(S, Options)
    ; Target == rust ->
        write(S, 'pub static CONFIG_FIELDS: &[AgentConfigField] = &[\n'),
        findall(acf(N,T,D,C), agent_loop_module:agent_config_field(N,T,D,C), ACFs),
        maplist([acf(N,T,D,C)]>>(
            (D == none -> DStr = ''
            ; atom(D) -> atom_string(D, DS), agent_loop_module:replace_all_sub(DS, "\"", "\\\"", DStr)
            ; term_to_atom(D, DA), atom_string(DA, DS2), agent_loop_module:replace_all_sub(DS2, "\"", "\\\"", DStr)
            ),
            format(S, '    AgentConfigField { name: "~w", type_annotation: "~w", default_value: "~w", comment: "~w" },~n',
                   [N, T, DStr, C])
        ), ACFs),
        write(S, '];\n\n')
    ;
        findall(acf(N,T,D,C), agent_loop_module:agent_config_field(N,T,D,C), ACFs),
        maplist([acf(N,T,D,C)]>>(
            format(S, "agent_config_field(~q, ~q, ~q, ~q).~n", [N,T,D,C])
        ), ACFs)
    ).
emit_config_section(S, audit_levels, Options) :-
    extract_target(Options, Target),
    (Target == python ->
        emit_audit_levels(S, Options)
    ; Target == rust ->
        write(S, 'pub static AUDIT_LEVELS: &[AuditLevel] = &[\n'),
        findall(P-L, agent_loop_module:audit_profile_level(P, L), Pairs),
        maplist([P-L]>>(
            format(S, '    AuditLevel { profile: "~w", level: "~w" },~n', [P, L])
        ), Pairs),
        write(S, '];\n\n')
    ;
        findall(P-L, agent_loop_module:audit_profile_level(P, L), Pairs),
        maplist([P-L]>>(format(S, "audit_profile_level(~q, ~q).~n", [P, L])), Pairs)
    ).
emit_config_section(S, streaming_capable, Options) :-
    extract_target(Options, Target),
    (Target == python ->
        findall(Type, agent_loop_module:streaming_capable(Type), Types),
        maplist([Type]>>(format(S, "    '~w',~n", [Type])), Types)
    ; Target == rust ->
        write(S, 'pub static STREAMING_CAPABLE: &[&str] = &[\n'),
        findall(Type, agent_loop_module:streaming_capable(Type), Types),
        maplist([Type]>>(format(S, '    "~w",~n', [Type])), Types),
        write(S, '];\n\n')
    ;
        emit_streaming_capable_facts(S, Options)
    ).
emit_config_section(S, security_profiles, Options) :-
    extract_target(Options, Target),
    (Target == python ->
        emit_security_profile_entries(S, Options)
    ; Target == rust ->
        emit_rust_security_facts(S, Options)
    ;
        findall(N-P, agent_loop_module:security_profile(N, P), Profiles),
        maplist([N-P]>>(format(S, "security_profile(~q, ~q).~n", [N, P])), Profiles)
    ).
emit_config_section(S, cli_arguments, Options) :-
    extract_target(Options, Target),
    (Target == python ->
        emit_argparse_group_args(S, Options)
    ; Target == rust ->
        write(S, 'pub static CLI_ARGS: &[CliArgument] = &[\n'),
        findall(Name-Opts, agent_loop_module:cli_argument(Name, Opts), CLIArgs),
        maplist([Name-Opts]>>(
            (member(long(Long), Opts) -> true ; Long = ''),
            (member(short(Short), Opts) -> true ; Short = ''),
            (member(default(Def), Opts) ->
                (Def == none -> DefStr = '' ; term_to_atom(Def, DefStr))
            ; DefStr = ''),
            (member(help(Help), Opts) -> true ; Help = ''),
            format(S, '    CliArgument { name: "~w", long_flag: "~w", short_flag: "~w", default_value: "~w", help: "~w" },~n',
                   [Name, Long, Short, DefStr, Help])
        ), CLIArgs),
        write(S, '];\n\n')
    ;
        findall(N-P, agent_loop_module:cli_argument(N, P), Args),
        maplist([N-P]>>(format(S, "cli_argument(~q, ~q).~n", [N, P])), Args)
    ).
emit_config_section(S, config_search_paths, Options) :-
    extract_target(Options, Target),
    (Target == rust ->
        write(S, 'pub static CONFIG_SEARCH_PATHS: &[ConfigSearchPath] = &[\n'),
        findall(P-Prio, agent_loop_module:config_search_path(P, Prio), CSPs),
        maplist([P-Prio]>>(
            format(S, '    ConfigSearchPath { path: "~w", priority: "~w" },~n', [P, Prio])
        ), CSPs),
        write(S, '];\n\n')
    ;
        findall(P-Prio, agent_loop_module:config_search_path(P, Prio), CSPs),
        maplist([P-Prio]>>(format(S, "config_search_path(~q, ~q).~n", [P, Prio])), CSPs)
    ).
emit_config_section(S, config_dir_file_names, Options) :-
    extract_target(Options, Target),
    (Target == rust ->
        write(S, 'pub static CONFIG_DIR_FILE_NAMES: &[&str] = &[\n'),
        findall(F, agent_loop_module:config_dir_file_name(F), Fs),
        maplist([F]>>(format(S, '    "~w",~n', [F])), Fs),
        write(S, '];\n\n')
    ;
        findall(F, agent_loop_module:config_dir_file_name(F), Fs),
        maplist([F]>>(format(S, "config_dir_file_name(~q).~n", [F])), Fs)
    ).
emit_config_section(S, cli_overrides, Options) :-
    extract_target(Options, Target),
    (Target == rust ->
        write(S, '/// CLI override rule: (cli_flag, config_field, behavior)\n'),
        write(S, '#[derive(Debug, Clone)]\n'),
        write(S, 'pub struct CliOverride {\n'),
        write(S, '    pub cli_flag: &\'static str,\n'),
        write(S, '    pub config_field: &\'static str,\n'),
        write(S, '    pub behavior: &\'static str,\n'),
        write(S, '}\n\n'),
        write(S, 'pub static CLI_OVERRIDES: &[CliOverride] = &[\n'),
        findall(co(F,C,B), agent_loop_module:cli_override(F, C, B), COs),
        maplist([co(F,C,B)]>>(
            format(S, '    CliOverride { cli_flag: "~w", config_field: "~w", behavior: "~w" },~n',
                   [F, C, B])
        ), COs),
        write(S, '];\n\n')
    ;
        findall(co(F,C,B), agent_loop_module:cli_override(F, C, B), COs),
        maplist([co(F,C,B)]>>(format(S, "cli_override(~q, ~q, ~q).~n", [F, C, B])), COs)
    ).

%% ============================================================================
%% Prolog Module Header Emission
%% ============================================================================

%% emit_prolog_module_header(+Stream, +Module, +Options)
%% Emit a Prolog module declaration with exports and use_module directives.
%% Options: exports(List), use_modules(List)
emit_prolog_module_header(S, Module, Options) :-
    (member(exports(Exports), Options) ->
        format(S, ':- module(~w, [~n', [Module]),
        length(Exports, Len),
        emit_module_export_list(S, Exports, 1, Len),
        write(S, ']).\n\n')
    ; true),
    (member(use_modules(UseMods), Options) ->
        maplist([UM]>>(format(S, ':- use_module(~w).~n', [UM])), UseMods),
        nl(S)
    ; true).

emit_module_export_list(_, [], _, _).
emit_module_export_list(S, [E], I, I) :-
    format(S, '    ~w~n', [E]).
emit_module_export_list(S, [E|Rest], Pos, Len) :-
    Rest \= [],
    format(S, '    ~w,~n', [E]),
    Pos1 is Pos + 1,
    emit_module_export_list(S, Rest, Pos1, Len).

%% emit_prolog_declarations(+Stream, +Options)
%% Emit Prolog declaration directives (det, dynamic).
%% Options: det(List), dynamic(List)
emit_prolog_declarations(S, Options) :-
    (member(det(Dets), Options) ->
        maplist([D]>>(format(S, ':- det(~w).~n', [D])), Dets),
        nl(S)
    ; true),
    (member(dynamic(Dyns), Options) ->
        maplist([D]>>(format(S, ':- dynamic ~w.~n', [D])), Dyns)
    ; true).

%% emit_prolog_module_skeleton(+Stream, +Module, +Directives)
%% =============================================================================
%% Unified Module Skeleton — emit_module_skeleton/4
%% =============================================================================
%% Target-dispatching wrapper over target-specific skeleton emitters.

emit_module_skeleton(S, prolog, Module, Directives) :- !,
    emit_prolog_module_skeleton(S, Module, Directives).
emit_module_skeleton(S, rust, Module, Directives) :- !,
    emit_rust_module_skeleton(S, Module, Directives).
emit_module_skeleton(_, Target, _, _) :-
    format(atom(Msg), 'No skeleton emitter for target: ~w', [Target]),
    throw(error(domain_error(target, Target), context(emit_module_skeleton/4, Msg))).

%% --- Prolog skeleton ---

%% One-call module setup from an ordered list of directives.
%% The order of directives in the list determines the output order.
%% Supported directives:
%%   exports(List)         — :- module(Name, [exports...]).
%%   use_modules(List)     — :- use_module(...) with trailing blank line
%%   use_modules_compact(List) — :- use_module(...) WITHOUT trailing blank line
%%   det(List)             — :- det(...) with trailing blank line
%%   dynamic(List)         — :- dynamic ... (no trailing blank)
%%   dependencies(DepOpts) — %% Dependencies comment
%%   discontiguous(List)   — :- discontiguous ... with trailing blank line
%%   table(List)           — :- table ... with trailing blank line
%%   table(List, Comment)  — %% Comment then :- table ... with trailing blank line
emit_prolog_module_skeleton(S, Module, Directives) :-
    emit_skeleton_directives(S, Module, Directives).

emit_skeleton_directives(_, _, []).
emit_skeleton_directives(S, Module, [D|Rest]) :-
    emit_one_skeleton_directive(S, Module, D),
    emit_skeleton_directives(S, Module, Rest).

emit_one_skeleton_directive(S, Module, exports(Exports)) :- !,
    format(S, ':- module(~w, [~n', [Module]),
    length(Exports, Len),
    emit_module_export_list(S, Exports, 1, Len),
    write(S, ']).\n\n').
emit_one_skeleton_directive(S, _, use_modules(UMs)) :- !,
    maplist([UM]>>(format(S, ':- use_module(~w).~n', [UM])), UMs),
    nl(S).
emit_one_skeleton_directive(S, _, use_modules_compact(UMs)) :- !,
    maplist([UM]>>(format(S, ':- use_module(~w).~n', [UM])), UMs).
emit_one_skeleton_directive(S, _, det(Dets)) :- !,
    maplist([D]>>(format(S, ':- det(~w).~n', [D])), Dets),
    nl(S).
emit_one_skeleton_directive(S, _, dynamic(Dyns)) :- !,
    maplist([D]>>(format(S, ':- dynamic ~w.~n', [D])), Dyns).
emit_one_skeleton_directive(S, _, dependencies(DepOpts)) :- !,
    emit_module_dependencies(S, DepOpts).
emit_one_skeleton_directive(S, _, discontiguous(Ds)) :- !,
    maplist([D]>>(format(S, ':- discontiguous ~w.~n', [D])), Ds),
    nl(S).
emit_one_skeleton_directive(S, _, table(Ts)) :- !,
    maplist([T]>>(format(S, ':- table ~w.~n', [T])), Ts),
    nl(S).
emit_one_skeleton_directive(S, _, table(Ts, Comment)) :- !,
    format(S, '%% ~w~n', [Comment]),
    maplist([T]>>(format(S, ':- table ~w.~n', [T])), Ts),
    nl(S).
emit_one_skeleton_directive(S, _, comment(Text)) :- !,
    format(S, '%% ~w~n~n', [Text]).

%% --- Rust skeleton ---
%%
%% Emits Rust module header from the same directive DSL.
%% Supported directives:
%%   exports(List)           — // Public API: ... doc comment
%%   use_modules(List)       — use crate::module; per item
%%   use_external(List)      — use dep::{...}; per item
%%   dynamic(List)           — // Mutable state: ... comment
%%   derives(List)           — #[derive(D1, D2, ...)]
%%   comment(Text)           — /// Text
%%   dependencies(DepOpts)   — // Dependencies: ... comment

emit_rust_module_skeleton(S, Module, Directives) :-
    format(S, '// Module: ~w~n', [Module]),
    format(S, '// Auto-generated by UnifyWeaver — do not edit manually.~n~n', []),
    emit_rust_skeleton_directives(S, Module, Directives).

emit_rust_skeleton_directives(_, _, []).
emit_rust_skeleton_directives(S, Module, [D|Rest]) :-
    emit_one_rust_directive(S, Module, D),
    emit_rust_skeleton_directives(S, Module, Rest).

emit_one_rust_directive(S, _, exports(Exports)) :- !,
    write(S, '// Public API:\n'),
    maplist([E]>>(
        (E = Name/_ -> true ; Name = E),
        format(S, '//   - ~w~n', [Name])
    ), Exports),
    nl(S).
emit_one_rust_directive(S, _, use_modules(Mods)) :- !,
    maplist([M]>>(format(S, 'use crate::~w;~n', [M])), Mods),
    nl(S).
emit_one_rust_directive(S, _, use_external(Exts)) :- !,
    maplist([Ext]>>(
        (Ext = Crate-Items ->
            atomic_list_concat(Items, ', ', ItemsStr),
            format(S, 'use ~w::{~w};~n', [Crate, ItemsStr])
        ;
            format(S, 'use ~w;~n', [Ext])
        )
    ), Exts),
    nl(S).
emit_one_rust_directive(S, _, derives(Ds)) :- !,
    atomic_list_concat(Ds, ', ', DsStr),
    format(S, '#[derive(~w)]~n', [DsStr]).
emit_one_rust_directive(S, _, dynamic(Dyns)) :- !,
    write(S, '// Mutable state (requires Mutex/RwLock):\n'),
    maplist([D]>>(
        (D = Name/_ -> true ; Name = D),
        format(S, '//   - ~w~n', [Name])
    ), Dyns),
    nl(S).
emit_one_rust_directive(S, _, comment(Text)) :- !,
    format(S, '/// ~w~n', [Text]).
emit_one_rust_directive(S, _, dependencies(DepOpts)) :- !,
    write(S, '// Dependencies:\n'),
    (member(module(M), DepOpts) ->
        format(S, '//   module: ~w~n', [M])
    ; true),
    nl(S).
%% Ignore directives not applicable to Rust (det, discontiguous, table, etc.)
emit_one_rust_directive(_, _, _).

%% ============================================================================
%% Unified Component Emission
%% ============================================================================

%% emit_from_components(+Stream, +Category, +Pred, +Target, +Format, +Options)
%% Generic component emission — single predicate for all output formats.
%% Format: dict | set | facts | list
%%   dict  — Python dict with binding-derived or overridden name
%%   set   — Python set with binding-derived or overridden name
%%   list  — Python list with binding-derived or overridden name
%%   facts — Prolog facts (no collection wrapper, just bare facts)
%% Options:
%%   dict_name(Name)    — override binding-derived collection name
%%   outer_indent(N)    — indent the wrapper delimiters by N spaces
%%   indent(N)          — indent each entry by N spaces (passed to compile_component)
%%   fact_type(FT)      — select specific fact type for compilation
emit_from_components(S, Category, Pred, Target, Format, Options) :-
    agent_loop_bindings:init_agent_loop_bindings,
    resolve_collection_name(Format, Target, Pred, Options, CollName),
    compute_indent(Options, Indent),
    emit_open_delimiter(S, Format, Indent, CollName),
    findall(Name, component(Category, Name, _, _), Names),
    compile_options(Format, Target, Options, CompileOpts),
    maplist([Name]>>(
        (compile_component(Category, Name, CompileOpts, Code) ->
            write(S, Code), nl(S)
        ; true)
    ), Names),
    emit_close_delimiter(S, Format, Indent).

%% resolve_collection_name(+Format, +Target, +Pred, +Options, -Name)
%% Resolve the collection variable name from options or binding metadata.
resolve_collection_name(entries, _, _, _, '') :- !.
resolve_collection_name(facts, _, _, _, '') :- !.
resolve_collection_name(_, _Target, _Pred, Options, Name) :-
    member(dict_name(Name), Options), !.
resolve_collection_name(_, Target, Pred, _, Name) :-
    agent_loop_bindings:binding_dict_name(Target, Pred, Name).

%% compile_options(+Format, +Target, +Options, -CompileOpts)
%% Build the options list passed to compile_component.
compile_options(entries, Target, Options, [target(Target)|Options]) :- !.
compile_options(facts, _Target, Options, [target(prolog)|Options]) :- !.
compile_options(rust_entries, _Target, Options, [target(rust)|Options]) :- !.
compile_options(rust_array, _Target, Options, [target(rust)|Options]) :- !.
compile_options(_, Target, Options, [target(Target)|Options]).

%% emit_open_delimiter(+Stream, +Format, +Indent, +Name)
%% Write the opening delimiter for a collection.
emit_open_delimiter(_, entries, _, _) :- !.
emit_open_delimiter(_, facts, _, _) :- !.
emit_open_delimiter(_, rust_entries, _, _) :- !.
emit_open_delimiter(S, dict, Indent, Name) :- format(S, '~w~w = {~n', [Indent, Name]).
emit_open_delimiter(S, set, Indent, Name) :- format(S, '~w~w = {~n', [Indent, Name]).
emit_open_delimiter(S, list, Indent, Name) :- format(S, '~w~w = [~n', [Indent, Name]).
emit_open_delimiter(S, rust_array, Indent, _) :- format(S, '~w&[~n', [Indent]).

%% emit_close_delimiter(+Stream, +Format, +Indent)
%% Write the closing delimiter for a collection.
emit_close_delimiter(_, entries, _) :- !.
emit_close_delimiter(_, facts, _) :- !.
emit_close_delimiter(_, rust_entries, _) :- !.
emit_close_delimiter(S, dict, Indent) :- format(S, '~w}~n', [Indent]).
emit_close_delimiter(S, set, Indent) :- format(S, '~w}~n', [Indent]).
emit_close_delimiter(S, list, Indent) :- format(S, '~w]~n', [Indent]).
emit_close_delimiter(S, rust_array, Indent) :- format(S, '~w];~n', [Indent]).

%% compute_indent(+Options, -Indent)
%% Compute the indentation string from outer_indent option.
compute_indent(Options, Indent) :-
    (member(outer_indent(OI), Options) -> true ; OI = 0),
    length(Spaces, OI), maplist(=(0' ), Spaces), atom_chars(Indent, Spaces).

%% extract_target(+Options, -Target)
%% Extract target from options, defaulting to prolog.
extract_target(Options, Target) :-
    (member(target(T), Options) -> Target = T ; Target = prolog).

%% maybe_emit_prolog_hints(+Stream, +Target, +Predicates)
%% Emit SWI-Prolog indexing hints for Prolog target, no-op for Python.
maybe_emit_prolog_hints(_, python, _) :- !.
maybe_emit_prolog_hints(S, prolog, Predicates) :-
    write(S, '%% Indexing hints (SWI-Prolog auto-indexes first argument):\n'),
    maplist([Pred]>>(emit_indexing_directive(S, Pred)), Predicates),
    write(S, '\n').

%% ---- Backward-compatible wrappers ----

%% emit_py_dict_from_components(+Stream, +Category, +Pred, +Target, +Options)
%% Backward-compatible wrapper — delegates to emit_from_components/6 with dict format.
emit_py_dict_from_components(S, Cat, Pred, Target, Opts) :-
    emit_from_components(S, Cat, Pred, Target, dict, Opts).

%% emit_py_set_from_components(+Stream, +Category, +Pred, +Target, +Options)
%% Backward-compatible wrapper — delegates to emit_from_components/6 with set format.
emit_py_set_from_components(S, Cat, Pred, Target, Opts) :-
    emit_from_components(S, Cat, Pred, Target, set, Opts).

%% emit_prolog_facts_from_components(+Stream, +Category, +FactType, +Options)
%% Backward-compatible wrapper — delegates to emit_from_components/6 with facts format.
emit_prolog_facts_from_components(S, Cat, FactType, Opts) :-
    emit_from_components(S, Cat, _Pred, prolog, facts, [fact_type(FactType)|Opts]).

%% ============================================================================
%% Declarative Security Validation Rules
%% ============================================================================
%%
%% These facts declare how security checks map to data categories.
%% The compile predicates emit target-specific validation logic from them.

%% path_check_rule(+RuleName, +CheckType, +Properties)
%% Declarative security path validation rules.
path_check_rule(exact_match,   blocked_path,        [comment("Exact blocked path match")]).
path_check_rule(prefix_match,  blocked_path_prefix, [comment("Blocked path prefix match")]).
path_check_rule(home_pattern,  blocked_home_pattern, [comment("Home directory pattern match")]).

%% command_check_rule(+RuleName, +CheckType, +Properties)
%% Declarative security command validation rules.
command_check_rule(regex_match, blocked_command_pattern, [comment("Command regex pattern match")]).

%% compile_path_check_rules(+Stream, +Target, +Options)
%% Emit path validation logic for the given target, driven by path_check_rule/3 facts.
compile_path_check_rules(S, python, _Options) :-
    findall(CheckType-Props, path_check_rule(_, CheckType, Props), Rules),
    maplist([CT-Ps]>>(
        member(comment(Comment), Ps),
        format(S, "    # ~w~n", [Comment]),
        compile_path_check_python(S, CT)
    ), Rules).
compile_path_check_rules(S, prolog, _Options) :-
    findall(RN-CT-Ps, path_check_rule(RN, CT, Ps), Rules),
    maplist([RuleName-CheckType-Props]>>(
        member(comment(Comment), Props),
        format(S, "%% ~w: ~w~n", [RuleName, Comment]),
        compile_path_check_prolog(S, CheckType)
    ), Rules).

compile_path_check_python(S, blocked_path) :-
    write(S, '    if abs_path in _BLOCKED_PATHS:\n'),
    write(S, '        return True\n').
compile_path_check_python(S, blocked_path_prefix) :-
    write(S, '    for prefix in _BLOCKED_PREFIXES:\n'),
    write(S, '        if abs_path.startswith(prefix):\n'),
    write(S, '            return True\n').
compile_path_check_python(S, blocked_home_pattern) :-
    write(S, '    home = os.path.expanduser("~")\n'),
    write(S, '    for pattern in _BLOCKED_HOME_PATTERNS:\n'),
    write(S, '        if abs_path.startswith(os.path.join(home, pattern)):\n'),
    write(S, '            return True\n').

compile_path_check_prolog(S, blocked_path) :-
    write(S, 'is_path_blocked(Path) :- blocked_path(Path), !.\n').
compile_path_check_prolog(S, blocked_path_prefix) :-
    write(S, 'is_path_blocked(Path) :- blocked_path_prefix(Prefix), atom_concat(Prefix, _, Path), !.\n').
compile_path_check_prolog(S, blocked_home_pattern) :-
    write(S, 'is_path_blocked(Path) :- blocked_home_pattern(Pat), expand_home(Pat, Full), atom_concat(Full, _, Path), !.\n').

%% compile_command_check_rules(+Stream, +Target, +Options)
%% Emit command validation logic for the given target, driven by command_check_rule/3 facts.
compile_command_check_rules(S, python, _Options) :-
    findall(Props, command_check_rule(_, _, Props), Rules),
    maplist([Ps]>>(
        member(comment(Comment), Ps),
        format(S, "    # ~w~n", [Comment]),
        write(S, '    for pattern, desc in _BLOCKED_COMMAND_PATTERNS:\n'),
        write(S, '        if re.search(pattern, command):\n'),
        write(S, '            return True, desc\n')
    ), Rules).
compile_command_check_rules(S, prolog, _Options) :-
    findall(RN-Ps, command_check_rule(RN, _, Ps), Rules),
    maplist([RuleName-Props]>>(
        member(comment(Comment), Props),
        format(S, "%% ~w: ~w~n", [RuleName, Comment]),
        write(S, 'is_command_blocked(Cmd, Desc) :- blocked_command_pattern(Regex, Desc), re_match(Regex, Cmd), !.\n')
    ), Rules).

%% ============================================================================
%% Composed Security Check Predicates (Prolog target)
%% ============================================================================

%% emit_security_check_predicates(+Stream, +Options)
%% Emits composed Prolog security check predicates:
%%   1. is_path_blocked/1 (from compile_path_check_rules)
%%   2. is_command_blocked/2 (from compile_command_check_rules)
%%   3. check_path_allowed/2 (thin wrapper with profile dispatch)
%%   4. check_command_allowed/2 (thin wrapper with profile dispatch)
%%   5. set_security_profile/1 (profile switching)
emit_security_check_predicates(S, _Options) :-
    %% Building blocks from compiled rules
    compile_path_check_rules(S, prolog, []),
    nl(S),
    compile_command_check_rules(S, prolog, []),
    nl(S),
    %% Thin wrapper: check_path_allowed/2
    write(S, '%% Check if a file path is allowed under current profile\n'),
    write(S, 'check_path_allowed(Path, Result) :-\n'),
    write(S, '    current_security_profile(Profile),\n'),
    write(S, '    (Profile = open -> Result = allowed\n'),
    write(S, '    ; is_path_blocked(Path) -> Result = blocked("Blocked path")\n'),
    write(S, '    ; Result = allowed).\n\n'),
    %% Thin wrapper: check_command_allowed/2
    write(S, '%% Check if a command is allowed under current profile\n'),
    write(S, 'check_command_allowed(Command, Result) :-\n'),
    write(S, '    current_security_profile(Profile),\n'),
    write(S, '    (Profile = open -> Result = allowed\n'),
    write(S, '    ; is_command_blocked(Command, Desc) -> Result = blocked(Desc)\n'),
    write(S, '    ; Result = allowed).\n\n'),
    %% set_security_profile/1
    write(S, '%% Set the active security profile\n'),
    write(S, 'set_security_profile(Profile) :-\n'),
    write(S, '    (security_profile(Profile, _) ->\n'),
    write(S, '        retractall(current_security_profile(_)),\n'),
    write(S, '        assert(current_security_profile(Profile)),\n'),
    write(S, '        format("Security profile set to: ~w~n", [Profile])\n'),
    write(S, '    ; format("Unknown profile: ~w~n", [Profile])).\n').
