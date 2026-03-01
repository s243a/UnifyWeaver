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
    agent_loop_component_summary/0
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
    member(handler(H), Config),
    (member(target(prolog), Options) ->
        format(atom(Code), "tool_handler(~q, ~q).", [Name, H])
    ; member(target(python), Options) ->
        format(atom(Code), "    '~w': ~w,", [Name, H])
    ;
        format(atom(Code), "tool_handler(~q, ~q).", [Name, H])
    ).

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
    member(match_type(MT), Config),
    member(help_text(HT), Config),
    (member(options(Opts), Config) -> true ; Opts = []),
    (member(target(prolog), Options) ->
        format(atom(Code), "slash_command(~q, ~q, ~q, ~q).", [Name, MT, Opts, HT])
    ;
        format(atom(Code), "slash_command(~q, ~q, ~q, ~q).", [Name, MT, Opts, HT])
    ).

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
        format(atom(Code), "backend_factory(~q, ~q).", [Name, Config])
    ;
        format(atom(Code), "backend_factory(~q, ~q).", [Name, Config])
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
    ]).

%% ============================================================================
%% Instance Population from Existing Facts
%% ============================================================================

%% Populate tool components from tool_handler/2 + tool_spec/2 + destructive_tool/1
register_tool_components :-
    forall(agent_loop_module:tool_handler(Name, Handler), (
        (agent_loop_module:tool_spec(Name, Props) ->
            (member(description(Desc), Props) -> true ; Desc = "")
        ; Desc = ""),
        (agent_loop_module:destructive_tool(Name) -> Destr = true ; Destr = false),
        declare_component(agent_tools, Name, tool_handler, [
            handler(Handler),
            description(Desc),
            destructive(Destr),
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
    register_backend_components.

%% agent_loop_component_summary/0
%% Registers everything and prints a summary.
agent_loop_component_summary :-
    register_agent_loop_components,
    list_components(agent_tools, Tools),
    list_components(agent_commands, Cmds),
    list_components(agent_backends, Backends),
    length(Tools, NT), length(Cmds, NC), length(Backends, NB),
    format("~nAgent-loop components registered:~n"),
    format("  Tools:    ~w ~w~n", [NT, Tools]),
    format("  Commands: ~w ~w~n", [NC, Cmds]),
    format("  Backends: ~w ~w~n", [NB, Backends]).
