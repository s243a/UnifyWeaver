% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% mindmap_glue.pl - Cross-Target Glue Integration for Mind Map DSL
%
% Integrates the mindmap DSL with UnifyWeaver's cross-target glue system,
% enabling seamless communication between Prolog (DSL) and Python (computation).
%
% Key concepts:
% - Prolog handles DSL definition and orchestration
% - Python handles heavy computation (layout, optimization)
% - Data flows via JSON Lines over pipes
%
% Usage:
%   ?- declare_mindmap_target(force_layout, python).
%   ?- execute_mindmap_pipeline(my_map, Positions).

:- module(mindmap_glue, [
    % Target declarations
    declare_mindmap_target/2,       % declare_mindmap_target(+Predicate, +Target)
    declare_mindmap_target/3,       % declare_mindmap_target(+Predicate, +Target, +Options)
    get_mindmap_target/2,           % get_mindmap_target(+Predicate, -Target)

    % Pipeline execution
    execute_mindmap_pipeline/2,     % execute_mindmap_pipeline(+MapId, -Result)
    execute_mindmap_pipeline/3,     % execute_mindmap_pipeline(+MapId, +Options, -Result)

    % Stage execution
    execute_layout_stage/4,         % execute_layout_stage(+Graph, +Algorithm, +Options, -Positions)
    execute_optimize_stage/4,       % execute_optimize_stage(+Positions, +Optimizer, +Options, -NewPositions)

    % Glue code generation
    generate_mindmap_glue/3,        % generate_mindmap_glue(+Stage, +Target, -GlueCode)
    generate_pipeline_glue/3,       % generate_pipeline_glue(+Pipeline, +Options, -Script)

    % Protocol handling
    create_mindmap_pipe/3,          % create_mindmap_pipe(+Direction, +Protocol, -Pipe)
    send_mindmap_data/3,            % send_mindmap_data(+Pipe, +Data, +Protocol)
    receive_mindmap_data/3,         % receive_mindmap_data(+Pipe, -Data, +Protocol)

    % Testing
    test_mindmap_glue/0
]).

:- use_module(library(lists)).
:- use_module(library(process)).

% Load serializer
:- use_module('../serialization/mindmap_serializer').

% ============================================================================
% TARGET DECLARATIONS
% ============================================================================

:- dynamic mindmap_target/3.  % mindmap_target(Predicate, Target, Options)

% Default target mappings - Python for heavy computation
mindmap_target(force_layout, python, [
    module('unifyweaver.mindmap.layout.force_directed'),
    function(compute),
    location(local_process),
    transport(pipe),
    format(jsonl)
]).

mindmap_target(radial_layout, python, [
    module('unifyweaver.mindmap.layout.radial'),
    function(compute),
    location(local_process),
    transport(pipe),
    format(jsonl)
]).

mindmap_target(hierarchical_layout, python, [
    module('unifyweaver.mindmap.layout.hierarchical'),
    function(compute),
    location(local_process),
    transport(pipe),
    format(jsonl)
]).

mindmap_target(overlap_removal, python, [
    module('unifyweaver.mindmap.optimize.overlap_removal'),
    function(compute),
    location(local_process),
    transport(pipe),
    format(jsonl)
]).

mindmap_target(crossing_minimization, python, [
    module('unifyweaver.mindmap.optimize.crossing_minimization'),
    function(compute),
    location(local_process),
    transport(pipe),
    format(jsonl)
]).

mindmap_target(spacing, python, [
    module('unifyweaver.mindmap.optimize.spacing'),
    function(compute),
    location(local_process),
    transport(pipe),
    format(jsonl)
]).

mindmap_target(centering, python, [
    module('unifyweaver.mindmap.optimize.centering'),
    function(compute),
    location(local_process),
    transport(pipe),
    format(jsonl)
]).

%% declare_mindmap_target(+Predicate, +Target)
declare_mindmap_target(Predicate, Target) :-
    declare_mindmap_target(Predicate, Target, []).

%% declare_mindmap_target(+Predicate, +Target, +Options)
declare_mindmap_target(Predicate, Target, Options) :-
    retractall(mindmap_target(Predicate, _, _)),
    assertz(mindmap_target(Predicate, Target, Options)).

%% get_mindmap_target(+Predicate, -TargetInfo)
get_mindmap_target(Predicate, target(Target, Options)) :-
    mindmap_target(Predicate, Target, Options).

% ============================================================================
% PIPELINE EXECUTION
% ============================================================================

%% execute_mindmap_pipeline(+MapId, -Result)
%  Execute the default pipeline for a mindmap.
%
execute_mindmap_pipeline(MapId, Result) :-
    execute_mindmap_pipeline(MapId, [], Result).

%% execute_mindmap_pipeline(+MapId, +Options, -Result)
execute_mindmap_pipeline(MapId, Options, Result) :-
    % Get graph data
    get_mindmap_graph(MapId, Graph),

    % Determine pipeline stages
    (   member(pipeline(Stages), Options)
    ->  true
    ;   Stages = [force_layout, overlap_removal, centering]  % Default pipeline
    ),

    % Execute stages
    execute_stages(Graph, Stages, Options, Result).

execute_stages(Graph, [], _Options, graph_result(Graph, [])).
execute_stages(Graph, [Stage | Rest], Options, Result) :-
    execute_stage(Graph, Stage, Options, StageResult),
    (   StageResult = positions(Positions)
    ->  execute_stages_with_positions(Graph, Rest, Positions, Options, Result)
    ;   execute_stages(Graph, Rest, Options, Result)
    ).

execute_stages_with_positions(Graph, [], Positions, _Options, graph_result(Graph, Positions)).
execute_stages_with_positions(Graph, [Stage | Rest], Positions, Options, Result) :-
    execute_optimize_stage(Positions, Stage, Options, NewPositions),
    execute_stages_with_positions(Graph, Rest, NewPositions, Options, Result).

%% execute_stage(+Graph, +Stage, +Options, -Result)
execute_stage(Graph, Stage, Options, Result) :-
    get_mindmap_target(Stage, target(Target, TargetOpts)),
    (   Target == python
    ->  execute_python_stage(Graph, Stage, TargetOpts, Options, Result)
    ;   Target == prolog
    ->  execute_prolog_stage(Graph, Stage, Options, Result)
    ;   format(user_error, 'Unknown target: ~w~n', [Target]),
        fail
    ).

% ============================================================================
% STAGE EXECUTION
% ============================================================================

%% execute_layout_stage(+Graph, +Algorithm, +Options, -Positions)
execute_layout_stage(Graph, Algorithm, Options, Positions) :-
    get_mindmap_target(Algorithm, target(Target, TargetOpts)),
    (   Target == python
    ->  execute_python_layout(Graph, Algorithm, TargetOpts, Options, Positions)
    ;   execute_prolog_layout(Graph, Algorithm, Options, Positions)
    ).

%% execute_optimize_stage(+Positions, +Optimizer, +Options, -NewPositions)
execute_optimize_stage(Positions, Optimizer, Options, NewPositions) :-
    get_mindmap_target(Optimizer, target(Target, TargetOpts)),
    (   Target == python
    ->  execute_python_optimizer(Positions, Optimizer, TargetOpts, Options, NewPositions)
    ;   execute_prolog_optimizer(Positions, Optimizer, Options, NewPositions)
    ).

% ============================================================================
% PYTHON STAGE EXECUTION
% ============================================================================

%% execute_python_stage(+Graph, +Stage, +TargetOpts, +Options, -Result)
execute_python_stage(Graph, Stage, TargetOpts, Options, Result) :-
    % Generate Python command
    member(module(Module), TargetOpts),
    member(function(Function), TargetOpts),

    % Serialize graph to JSON Lines
    serialize_graph_for_python(Graph, Options, InputData),

    % Build Python command
    format(atom(PythonCode),
        'import sys; import json; from ~w import ~w; data = json.loads(sys.stdin.read()); result = ~w(data); print(json.dumps(result))',
        [Module, Function, Function]),

    % Execute via pipe
    process_create(path(python3), ['-c', PythonCode],
        [stdin(pipe(In)), stdout(pipe(Out)), stderr(null)]),

    % Send input
    format(In, '~w~n', [InputData]),
    close(In),

    % Read output
    read_string(Out, _, OutputStr),
    close(Out),

    % Parse result
    atom_json_dict(OutputStr, ResultDict, []),
    parse_python_result(Stage, ResultDict, Result).

execute_python_layout(Graph, _Algorithm, TargetOpts, Options, Positions) :-
    member(module(Module), TargetOpts),

    % Prepare graph data
    Graph = graph(_Id, Nodes, Edges, _),
    serialize_graph_dict(Nodes, Edges, Options, GraphDict),

    % Build Python script
    format(atom(Script),
'import sys, json
from ~w import compute
data = json.loads(sys.stdin.read())
positions = compute(data["nodes"], data["edges"], data.get("options", {}))
result = [{"type": "position", "id": k, "x": v[0], "y": v[1]} for k, v in positions.items()]
for r in result:
    print(json.dumps(r))
', [Module]),

    % Execute
    execute_python_script(Script, GraphDict, OutputLines),

    % Parse positions
    parse_position_lines(OutputLines, Positions).

execute_python_optimizer(Positions, _Optimizer, TargetOpts, Options, NewPositions) :-
    member(module(Module), TargetOpts),

    % Prepare positions dict
    positions_to_dict(Positions, PosDict),
    merge_options(Options, PosDict, InputDict),

    % Build Python script
    format(atom(Script),
'import sys, json
from ~w import compute
data = json.loads(sys.stdin.read())
positions = {k: tuple(v) for k, v in data["positions"].items()}
options = {k: v for k, v in data.items() if k != "positions"}
new_positions = compute(positions, **options)
result = [{"type": "position", "id": k, "x": v[0], "y": v[1]} for k, v in new_positions.items()]
for r in result:
    print(json.dumps(r))
', [Module]),

    % Execute
    execute_python_script(Script, InputDict, OutputLines),

    % Parse positions
    parse_position_lines(OutputLines, NewPositions).

execute_python_script(Script, InputDict, OutputLines) :-
    % Convert dict to JSON
    with_output_to(string(InputJson), json_write(current_output, InputDict, [width(0)])),

    % Execute Python
    process_create(path(python3), ['-c', Script],
        [stdin(pipe(In)), stdout(pipe(Out)), stderr(null)]),

    format(In, '~w', [InputJson]),
    close(In),

    read_string(Out, _, OutputStr),
    close(Out),

    split_string(OutputStr, "\n", "\r\t ", OutputLines0),
    exclude(=(""), OutputLines0, OutputLines).

% ============================================================================
% PROLOG STAGE EXECUTION (FALLBACK)
% ============================================================================

execute_prolog_stage(Graph, Stage, Options, Result) :-
    format(user_error, 'Prolog fallback for ~w not implemented~n', [Stage]),
    Result = error(not_implemented, Stage).

execute_prolog_layout(_Graph, Algorithm, _Options, []) :-
    format(user_error, 'Prolog layout ~w not implemented~n', [Algorithm]).

execute_prolog_optimizer(Positions, Optimizer, _Options, Positions) :-
    format(user_error, 'Prolog optimizer ~w not implemented~n', [Optimizer]).

% ============================================================================
% GLUE CODE GENERATION
% ============================================================================

%% generate_mindmap_glue(+Stage, +Target, -GlueCode)
%  Generate glue code for a mindmap pipeline stage.
%
generate_mindmap_glue(layout(Algorithm), python, GlueCode) :-
    get_mindmap_target(Algorithm, target(python, TargetOpts)),
    member(module(Module), TargetOpts),
    format(atom(GlueCode),
'#!/usr/bin/env python3
"""Mindmap layout glue - ~w"""
import sys
import json
from ~w import compute

def main():
    data = json.loads(sys.stdin.read())
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    options = data.get("options", {})

    # Convert nodes to graph structure
    graph = {}
    for n in nodes:
        graph[n["id"]] = n

    # Convert edges to adjacency
    adj = {}
    for e in edges:
        adj.setdefault(e["from"], []).append(e["to"])

    # Compute layout
    positions = compute(graph, adj, options)

    # Output as JSON Lines
    for node_id, (x, y) in positions.items():
        print(json.dumps({"type": "position", "id": node_id, "x": x, "y": y}))

if __name__ == "__main__":
    main()
', [Algorithm, Module]).

generate_mindmap_glue(optimize(Optimizer), python, GlueCode) :-
    get_mindmap_target(Optimizer, target(python, TargetOpts)),
    member(module(Module), TargetOpts),
    format(atom(GlueCode),
'#!/usr/bin/env python3
"""Mindmap optimizer glue - ~w"""
import sys
import json
from ~w import compute

def main():
    # Read positions from stdin (JSON Lines)
    positions = {}
    for line in sys.stdin:
        obj = json.loads(line.strip())
        if obj.get("type") == "position":
            positions[obj["id"]] = (obj["x"], obj["y"])

    # Run optimizer
    new_positions = compute(positions)

    # Output as JSON Lines
    for node_id, (x, y) in new_positions.items():
        print(json.dumps({"type": "position", "id": node_id, "x": x, "y": y}))

if __name__ == "__main__":
    main()
', [Optimizer, Module]).

%% generate_pipeline_glue(+Pipeline, +Options, -Script)
%  Generate a complete pipeline script.
%
generate_pipeline_glue(Pipeline, Options, Script) :-
    findall(StageScript,
        (   member(Stage, Pipeline),
            generate_stage_command(Stage, Options, StageScript)
        ),
        StageScripts),
    atomic_list_concat(StageScripts, ' | ', PipelineCmd),
    format(atom(Script),
'#!/bin/bash
# Mindmap pipeline: ~w
# Generated by UnifyWeaver mindmap_glue

~w
', [Pipeline, PipelineCmd]).

generate_stage_command(layout(Algorithm), _Options, Cmd) :-
    get_mindmap_target(Algorithm, target(python, TargetOpts)),
    member(module(Module), TargetOpts),
    format(atom(Cmd), 'python3 -c "from ~w import compute; import sys,json; d=json.load(sys.stdin); print(json.dumps(compute(d)))"', [Module]).

generate_stage_command(optimize(Optimizer), _Options, Cmd) :-
    get_mindmap_target(Optimizer, target(python, TargetOpts)),
    member(module(Module), TargetOpts),
    format(atom(Cmd), 'python3 -c "from ~w import compute; import sys,json; d=json.load(sys.stdin); print(json.dumps(compute(d)))"', [Module]).

% ============================================================================
% PROTOCOL HANDLING
% ============================================================================

%% create_mindmap_pipe(+Direction, +Protocol, -Pipe)
create_mindmap_pipe(input, jsonl, pipe(stdin, jsonl, [])).
create_mindmap_pipe(output, jsonl, pipe(stdout, jsonl, [])).

%% send_mindmap_data(+Pipe, +Data, +Protocol)
send_mindmap_data(pipe(Stream, jsonl, _), Data, jsonl) :-
    (   is_list(Data)
    ->  forall(member(Item, Data),
            (   serialize_object(Item, Dict),
                dict_to_json_line(Dict, Line),
                writeln(Stream, Line)
            ))
    ;   serialize_object(Data, Dict),
        dict_to_json_line(Dict, Line),
        writeln(Stream, Line)
    ).

%% receive_mindmap_data(+Pipe, -Data, +Protocol)
receive_mindmap_data(pipe(Stream, jsonl, _), Data, jsonl) :-
    read_mindmap_jsonl(Stream, Data).

% ============================================================================
% UTILITIES
% ============================================================================

get_mindmap_graph(MapId, graph(MapId, Nodes, Edges, [])) :-
    % This would normally query the mindmap DSL
    % For now, just return empty if not found
    findall(node(Id, Props), mindmap_dsl:mindmap_node(Id, Props), Nodes),
    findall(edge(F, T, P), mindmap_dsl:mindmap_edge(F, T, P), Edges).

serialize_graph_for_python(graph(_Id, Nodes, Edges, _), _Options, JsonStr) :-
    serialize_graph_dict(Nodes, Edges, [], Dict),
    with_output_to(string(JsonStr), json_write(current_output, Dict, [width(0)])).

serialize_graph_dict(Nodes, Edges, Options, Dict) :-
    maplist(node_to_dict, Nodes, NodeDicts),
    maplist(edge_to_dict, Edges, EdgeDicts),
    Dict = json{nodes: NodeDicts, edges: EdgeDicts, options: Options}.

node_to_dict(node(Id, Props), json{id: Id, props: PropsDict}) :-
    props_to_dict(Props, PropsDict).

edge_to_dict(edge(From, To, Props), json{from: From, to: To, props: PropsDict}) :-
    props_to_dict(Props, PropsDict).

positions_to_dict(Positions, json{positions: PosDict}) :-
    findall(Id-[X, Y], member(position(Id, X, Y), Positions), Pairs),
    dict_create(PosDict, json, Pairs).

merge_options(Options, BaseDict, MergedDict) :-
    findall(Key-Value, member(Key=Value, Options), Pairs),
    dict_create(OptsDict, json, Pairs),
    put_dict(OptsDict, BaseDict, MergedDict).

parse_position_lines(Lines, Positions) :-
    findall(position(Id, X, Y),
        (   member(Line, Lines),
            Line \== "",
            atom_json_dict(Line, Dict, []),
            Dict.type == "position",
            Id = Dict.id,
            X = Dict.x,
            Y = Dict.y
        ),
        Positions).

parse_python_result(Stage, ResultDict, positions(Positions)) :-
    is_list(ResultDict),
    !,
    findall(position(Id, X, Y),
        (   member(P, ResultDict),
            P.type == "position",
            Id = P.id,
            X = P.x,
            Y = P.y
        ),
        Positions).
parse_python_result(Stage, ResultDict, error(Stage, ResultDict)).

% ============================================================================
% TESTING
% ============================================================================

test_mindmap_glue :-
    format('~n=== Mind Map Glue Tests ===~n~n'),

    % Test 1: Target declarations exist
    format('Test 1: Default target declarations...~n'),
    (   get_mindmap_target(force_layout, target(python, Opts)),
        member(module(_), Opts)
    ->  format('  PASS: force_layout target declared~n')
    ;   format('  FAIL: force_layout target missing~n')
    ),

    % Test 2: Custom target declaration
    format('~nTest 2: Custom target declaration...~n'),
    declare_mindmap_target(custom_layout, python, [module('custom.layout'), function(run)]),
    (   get_mindmap_target(custom_layout, target(python, CustomOpts)),
        member(module('custom.layout'), CustomOpts)
    ->  format('  PASS: Custom target declared~n')
    ;   format('  FAIL: Custom target declaration failed~n')
    ),

    % Test 3: Generate layout glue code
    format('~nTest 3: Generate layout glue code...~n'),
    generate_mindmap_glue(layout(force_layout), python, GlueCode),
    (   sub_atom(GlueCode, _, _, _, 'force_directed'),
        sub_atom(GlueCode, _, _, _, 'json.dumps')
    ->  format('  PASS: Glue code generated~n')
    ;   format('  FAIL: Glue code generation failed~n')
    ),

    % Test 4: Generate optimizer glue code
    format('~nTest 4: Generate optimizer glue code...~n'),
    generate_mindmap_glue(optimize(overlap_removal), python, OptGlue),
    (   sub_atom(OptGlue, _, _, _, 'overlap_removal'),
        sub_atom(OptGlue, _, _, _, 'compute')
    ->  format('  PASS: Optimizer glue generated~n')
    ;   format('  FAIL: Optimizer glue generation failed~n')
    ),

    % Test 5: Serialization utilities
    format('~nTest 5: Serialization utilities...~n'),
    TestNodes = [node(a, [label("A")]), node(b, [label("B")])],
    TestEdges = [edge(a, b, [])],
    serialize_graph_dict(TestNodes, TestEdges, [], Dict),
    (   is_dict(Dict),
        length(Dict.nodes, 2)
    ->  format('  PASS: Graph serialization works~n')
    ;   format('  FAIL: Graph serialization failed~n')
    ),

    % Cleanup
    retractall(mindmap_target(custom_layout, _, _)),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% INITIALIZATION
% ============================================================================

:- initialization((
    format('Mind map glue module loaded~n', [])
), now).
