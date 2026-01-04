% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% mindmap_glue.pl - Cross-Target Glue Integration for Mind Map DSL
%
% Integrates with the UnifyWeaver glue system for cross-target execution:
%
% 1. Target is Python → Generate pure Python code
% 2. Target is Prolog (SWI) → Use janus_glue for in-process Python calls
% 3. Target is Prolog (GNU/other) → Use pipe_glue with jsonL serialization
%
% This module delegates to the existing glue infrastructure:
% - janus_glue.pl: In-process Python↔Prolog via Janus
% - pipe_glue.pl: Subprocess communication via pipes
% - shell_glue.pl: Shell script execution
%
% Usage:
%   ?- mindmap_janus_available.  % Check if Janus can be used for mindmaps
%   ?- execute_mindmap_pipeline(my_map, Positions).
%   ?- generate_mindmap_python(my_map, [layout(force)], PythonCode).

:- module(mindmap_glue, [
    % Janus availability (delegated to janus_glue)
    mindmap_janus_available/0,      % mindmap_janus_available - true if Janus is usable
    mindmap_janus_available/1,      % mindmap_janus_available(-Info)
    check_mindmap_packages/1,       % check_mindmap_packages(-Available)

    % Target declarations
    declare_mindmap_target/2,       % declare_mindmap_target(+Predicate, +Target)
    declare_mindmap_target/3,       % declare_mindmap_target(+Predicate, +Target, +Options)
    get_mindmap_target/2,           % get_mindmap_target(+Predicate, -Target)

    % Transport selection
    select_transport/3,             % select_transport(+Predicate, +Options, -Transport)

    % Pipeline execution (Prolog runtime)
    execute_mindmap_pipeline/2,     % execute_mindmap_pipeline(+MapId, -Result)
    execute_mindmap_pipeline/3,     % execute_mindmap_pipeline(+MapId, +Options, -Result)

    % Stage execution
    execute_layout_stage/4,         % execute_layout_stage(+Graph, +Algorithm, +Options, -Positions)
    execute_optimize_stage/4,       % execute_optimize_stage(+Positions, +Optimizer, +Options, -NewPositions)

    % Python code generation (when target is Python)
    generate_mindmap_python/3,      % generate_mindmap_python(+MapId, +Options, -PythonCode)
    generate_layout_python/3,       % generate_layout_python(+Algorithm, +Options, -Code)
    generate_optimizer_python/3,    % generate_optimizer_python(+Optimizer, +Options, -Code)
    generate_pipeline_python/3,     % generate_pipeline_python(+Pipeline, +Options, -Code)

    % Glue code generation (for cross-target)
    generate_mindmap_glue/3,        % generate_mindmap_glue(+Stage, +Target, -GlueCode)
    generate_pipeline_glue/3,       % generate_pipeline_glue(+Pipeline, +Options, -Script)

    % Testing
    test_mindmap_glue/0
]).

:- use_module(library(lists)).

% Conditionally load process library (not available in all Prologs)
:- catch(use_module(library(process)), _, true).

% Load existing glue infrastructure
:- catch(use_module('../../glue/janus_glue'), _, true).
:- catch(use_module('../../glue/pipe_glue'), _, true).

% Load serializer
:- use_module('../serialization/mindmap_serializer').

% ============================================================================
% JANUS AVAILABILITY (DELEGATED TO JANUS_GLUE)
% ============================================================================

%% mindmap_janus_available
%  True if Janus is available for mindmap operations.
%  Delegates to janus_glue:janus_available/0.
%
mindmap_janus_available :-
    catch(janus_glue:janus_available, _, fail).

%% mindmap_janus_available(-Info)
%  Check Janus availability and return info.
%  Delegates to janus_glue:janus_available/1.
%
mindmap_janus_available(Info) :-
    catch(janus_glue:janus_available(Info), _, fail).

%% check_mindmap_packages(-Available)
%  Check if required mindmap Python packages are available via Janus.
%  Only meaningful when Janus is available.
%
check_mindmap_packages(true) :-
    mindmap_janus_available,
    check_package_via_janus('unifyweaver.mindmap.layout'),
    check_package_via_janus('unifyweaver.mindmap.optimize'),
    !.
check_mindmap_packages(false).

%% check_package_via_janus(+Package)
%  Check if a Python package is importable via Janus.
%
check_package_via_janus(Package) :-
    catch(
        janus_glue:janus_import_module(Package, _),
        _,
        fail
    ).

% ============================================================================
% TARGET DECLARATIONS
% ============================================================================

:- dynamic mindmap_target/3.  % mindmap_target(Predicate, Target, Options)

% Default target mappings - Python for heavy computation
% Transport will be determined at runtime based on Janus availability
mindmap_target(force_layout, python, [
    module('unifyweaver.mindmap.layout.force_directed'),
    function(compute)
]).

mindmap_target(radial_layout, python, [
    module('unifyweaver.mindmap.layout.radial'),
    function(compute)
]).

mindmap_target(hierarchical_layout, python, [
    module('unifyweaver.mindmap.layout.hierarchical'),
    function(compute)
]).

mindmap_target(grid_layout, python, [
    module('unifyweaver.mindmap.layout.grid'),
    function(compute)
]).

mindmap_target(circular_layout, python, [
    module('unifyweaver.mindmap.layout.circular'),
    function(compute)
]).

mindmap_target(overlap_removal, python, [
    module('unifyweaver.mindmap.optimize.overlap_removal'),
    function(compute)
]).

mindmap_target(crossing_minimization, python, [
    module('unifyweaver.mindmap.optimize.crossing_minimization'),
    function(compute)
]).

mindmap_target(spacing, python, [
    module('unifyweaver.mindmap.optimize.spacing'),
    function(compute)
]).

mindmap_target(centering, python, [
    module('unifyweaver.mindmap.optimize.centering'),
    function(compute)
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
% TRANSPORT SELECTION
% ============================================================================

%% select_transport(+Predicate, +Options, -Transport)
%  Select the best transport mechanism for calling Python from Prolog.
%
%  Transport options:
%  - janus: In-process via Janus (SWI-Prolog only, fastest)
%  - pipe: Subprocess via Unix pipes with jsonL
%  - socket: Network socket (for remote/distributed)
%
%  Selection is delegated to the glue infrastructure where possible.
%  Janus is preferred when:
%  - Running on SWI-Prolog (janus_glue reports available)
%  - Required mindmap Python packages are importable
%
select_transport(_Predicate, Options, Transport) :-
    % Check if transport is explicitly specified
    member(transport(Transport), Options),
    !.
select_transport(Predicate, _Options, Transport) :-
    % Auto-select based on runtime
    auto_select_transport(Predicate, Transport).

%% auto_select_transport(+Predicate, -Transport)
%  Auto-select transport based on runtime capabilities.
%
auto_select_transport(_Predicate, janus) :-
    % Use Janus if available and mindmap packages are accessible
    mindmap_janus_available,
    check_mindmap_packages(true),
    !.
auto_select_transport(_Predicate, pipe) :-
    % Fallback to pipe for any Prolog implementation
    !.

% ============================================================================
% PIPELINE EXECUTION (PROLOG RUNTIME)
% ============================================================================

%% execute_mindmap_pipeline(+MapId, -Result)
execute_mindmap_pipeline(MapId, Result) :-
    execute_mindmap_pipeline(MapId, [], Result).

%% execute_mindmap_pipeline(+MapId, +Options, -Result)
execute_mindmap_pipeline(MapId, Options, Result) :-
    % Get graph data
    get_mindmap_graph(MapId, Graph),

    % Determine pipeline stages
    (   member(pipeline(Stages), Options)
    ->  true
    ;   Stages = [force_layout, overlap_removal, centering]
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
    get_mindmap_target(Stage, target(python, TargetOpts)),
    select_transport(Stage, Options, Transport),
    (   Transport == janus
    ->  execute_janus_stage(Graph, Stage, TargetOpts, Options, Result)
    ;   Transport == pipe
    ->  execute_pipe_stage(Graph, Stage, TargetOpts, Options, Result)
    ;   format(user_error, 'Unknown transport: ~w~n', [Transport]),
        fail
    ).

% ============================================================================
% JANUS EXECUTION (IN-PROCESS VIA JANUS_GLUE)
% ============================================================================
%
% Uses janus_glue:janus_call_python/4 for in-process Python calls.
% This provides:
% - Zero serialization overhead for compatible types
% - Direct function calls without process spawning
% - Shared memory for large data structures
%

execute_janus_stage(Graph, _Stage, TargetOpts, Options, positions(Positions)) :-
    member(module(Module), TargetOpts),
    member(function(Function), TargetOpts),

    % Convert graph to arguments list
    graph_to_janus_args(Graph, Options, Args),

    % Call Python via janus_glue
    catch(
        (   janus_glue:janus_call_python(Module, Function, Args, PyResult),
            py_result_to_positions(PyResult, Positions)
        ),
        Error,
        (   format(user_error, 'Janus error: ~w~n', [Error]),
            Positions = []
        )
    ).

execute_janus_layout(Graph, _Algorithm, TargetOpts, Options, Positions) :-
    member(module(Module), TargetOpts),

    Graph = graph(_Id, Nodes, Edges, _),
    nodes_to_py_dict(Nodes, NodesDict),
    edges_to_py_list(Edges, EdgesList),
    options_to_py_dict(Options, OptsDict),

    % Call module.compute(nodes, edges, options) via janus_glue
    catch(
        (   janus_glue:janus_call_python(Module, compute, [NodesDict, EdgesList, OptsDict], PyResult),
            py_result_to_positions(PyResult, Positions)
        ),
        _,
        Positions = []
    ).

execute_janus_optimizer(Positions, _Optimizer, TargetOpts, Options, NewPositions) :-
    member(module(Module), TargetOpts),

    positions_to_py_dict(Positions, PosDict),
    options_to_py_dict(Options, OptsDict),

    % Call module.compute(positions, options) via janus_glue
    catch(
        (   janus_glue:janus_call_python(Module, compute, [PosDict, OptsDict], PyResult),
            py_result_to_positions(PyResult, NewPositions)
        ),
        _,
        NewPositions = Positions
    ).

%% graph_to_janus_args(+Graph, +Options, -Args)
%  Convert graph and options to Janus-compatible argument list.
%
graph_to_janus_args(Graph, Options, [NodesDict, EdgesList, OptsDict]) :-
    Graph = graph(_Id, Nodes, Edges, _),
    nodes_to_py_dict(Nodes, NodesDict),
    edges_to_py_list(Edges, EdgesList),
    options_to_py_dict(Options, OptsDict).

% ============================================================================
% PIPE EXECUTION (SUBPROCESS)
% ============================================================================

execute_pipe_stage(Graph, Stage, TargetOpts, Options, positions(Positions)) :-
    member(module(Module), TargetOpts),

    % Serialize graph to JSON
    serialize_graph_for_python(Graph, Options, InputJson),

    % Build Python script
    format(atom(Script),
'import sys, json
from ~w import compute
data = json.loads(sys.stdin.read())
nodes = {n["id"]: n.get("props", {}) for n in data.get("nodes", [])}
edges = [(e["from"], e["to"]) for e in data.get("edges", [])]
options = data.get("options", {})
result = compute(nodes, edges, options) if "nodes" in data else compute(data.get("positions", {}), **options)
for node_id, pos in result.items():
    print(json.dumps({"type": "position", "id": node_id, "x": pos[0], "y": pos[1]}))', [Module]),

    % Execute via pipe
    execute_python_script(Script, InputJson, OutputLines),

    % Parse positions
    parse_position_lines(OutputLines, Positions).

execute_python_script(Script, InputJson, OutputLines) :-
    process_create(path(python3), ['-c', Script],
        [stdin(pipe(In)), stdout(pipe(Out)), stderr(null)]),

    format(In, '~w', [InputJson]),
    close(In),

    read_string(Out, _, OutputStr),
    close(Out),

    split_string(OutputStr, "\n", "\r\t ", OutputLines0),
    exclude(=(""), OutputLines0, OutputLines).

% ============================================================================
% STAGE EXECUTION DISPATCH
% ============================================================================

%% execute_layout_stage(+Graph, +Algorithm, +Options, -Positions)
execute_layout_stage(Graph, Algorithm, Options, Positions) :-
    get_mindmap_target(Algorithm, target(python, TargetOpts)),
    select_transport(Algorithm, Options, Transport),
    (   Transport == janus
    ->  execute_janus_layout(Graph, Algorithm, TargetOpts, Options, Positions)
    ;   execute_pipe_layout(Graph, Algorithm, TargetOpts, Options, Positions)
    ).

execute_pipe_layout(Graph, _Algorithm, TargetOpts, Options, Positions) :-
    member(module(Module), TargetOpts),

    Graph = graph(_Id, Nodes, Edges, _),
    serialize_graph_dict(Nodes, Edges, Options, GraphDict),
    with_output_to(string(InputJson), json_write(current_output, GraphDict, [width(0)])),

    format(atom(Script),
'import sys, json
from ~w import compute
data = json.loads(sys.stdin.read())
nodes = {n["id"]: n.get("props", {}) for n in data.get("nodes", [])}
edges = [(e["from"], e["to"]) for e in data.get("edges", [])]
options = data.get("options", {})
result = compute(nodes, edges, options)
for node_id, pos in result.items():
    print(json.dumps({"type": "position", "id": node_id, "x": pos[0], "y": pos[1]}))', [Module]),

    execute_python_script(Script, InputJson, OutputLines),
    parse_position_lines(OutputLines, Positions).

%% execute_optimize_stage(+Positions, +Optimizer, +Options, -NewPositions)
execute_optimize_stage(Positions, Optimizer, Options, NewPositions) :-
    get_mindmap_target(Optimizer, target(python, TargetOpts)),
    select_transport(Optimizer, Options, Transport),
    (   Transport == janus
    ->  execute_janus_optimizer(Positions, Optimizer, TargetOpts, Options, NewPositions)
    ;   execute_pipe_optimizer(Positions, Optimizer, TargetOpts, Options, NewPositions)
    ).

execute_pipe_optimizer(Positions, _Optimizer, TargetOpts, Options, NewPositions) :-
    member(module(Module), TargetOpts),

    positions_to_dict(Positions, PosDict),
    merge_options(Options, PosDict, InputDict),
    with_output_to(string(InputJson), json_write(current_output, InputDict, [width(0)])),

    format(atom(Script),
'import sys, json
from ~w import compute
data = json.loads(sys.stdin.read())
positions = {k: tuple(v) for k, v in data["positions"].items()}
options = {k: v for k, v in data.items() if k != "positions"}
result = compute(positions, **options)
for node_id, pos in result.items():
    print(json.dumps({"type": "position", "id": node_id, "x": pos[0], "y": pos[1]}))', [Module]),

    execute_python_script(Script, InputJson, OutputLines),
    parse_position_lines(OutputLines, NewPositions).

% ============================================================================
% PYTHON CODE GENERATION (WHEN TARGET IS PYTHON)
% ============================================================================

%% generate_mindmap_python(+MapId, +Options, -PythonCode)
%  Generate standalone Python code for mindmap processing.
%  Used when the overall target is Python (not Prolog calling Python).
%
generate_mindmap_python(MapId, Options, PythonCode) :-
    % Determine pipeline
    (   member(pipeline(Stages), Options)
    ->  true
    ;   Stages = [force_layout, overlap_removal, centering]
    ),

    % Generate imports
    generate_imports(Stages, Imports),

    % Generate pipeline code
    generate_pipeline_python(Stages, Options, PipelineCode),

    % Assemble
    format(atom(PythonCode),
'#!/usr/bin/env python3
"""
Mind Map Pipeline: ~w
Generated by UnifyWeaver mindmap_glue
"""

~w

from unifyweaver.mindmap.io import read_graph, write_positions

def process_mindmap(graph):
    """Process a mindmap graph through the pipeline."""
    nodes = {n.id: {"label": n.label, "type": n.node_type} for n in graph.nodes}
    edges = [(e.from_id, e.to_id) for e in graph.edges]

~w

    return positions

def main():
    import sys
    graph = read_graph(sys.stdin)
    if graph:
        positions = process_mindmap(graph)
        write_positions(positions, sys.stdout)

if __name__ == "__main__":
    main()
', [MapId, Imports, PipelineCode]).

generate_imports(Stages, Imports) :-
    findall(Import,
        (   member(Stage, Stages),
            get_mindmap_target(Stage, target(python, Opts)),
            member(module(Mod), Opts),
            format(atom(Import), 'from ~w import compute as ~w_compute', [Mod, Stage])
        ),
        ImportList),
    atomic_list_concat(ImportList, '\n', Imports).

%% generate_layout_python(+Algorithm, +Options, -Code)
generate_layout_python(Algorithm, _Options, Code) :-
    get_mindmap_target(Algorithm, target(python, Opts)),
    member(module(Module), Opts),
    format(atom(Code),
'from ~w import compute as layout_compute

def compute_layout(nodes, edges, options=None):
    """Compute layout positions for nodes."""
    return layout_compute(nodes, edges, options or {})
', [Module]).

%% generate_optimizer_python(+Optimizer, +Options, -Code)
generate_optimizer_python(Optimizer, _Options, Code) :-
    get_mindmap_target(Optimizer, target(python, Opts)),
    member(module(Module), Opts),
    format(atom(Code),
'from ~w import compute as optimize_compute

def optimize_positions(positions, options=None):
    """Optimize node positions."""
    return optimize_compute(positions, **(options or {}))
', [Module]).

%% generate_pipeline_python(+Pipeline, +Options, -Code)
generate_pipeline_python(Pipeline, _Options, Code) :-
    generate_pipeline_steps(Pipeline, 1, StepsCode),
    format(atom(Code),
'    # Pipeline execution
~w', [StepsCode]).

generate_pipeline_steps([], _, '').
generate_pipeline_steps([Stage | Rest], N, Code) :-
    (   N == 1
    ->  % First stage is layout
        format(atom(StepCode),
'    positions = ~w_compute(nodes, edges, {})~n', [Stage])
    ;   % Subsequent stages are optimizers
        format(atom(StepCode),
'    positions = ~w_compute(positions)~n', [Stage])
    ),
    N1 is N + 1,
    generate_pipeline_steps(Rest, N1, RestCode),
    atom_concat(StepCode, RestCode, Code).

% ============================================================================
% GLUE CODE GENERATION (FOR SHELL SCRIPTS)
% ============================================================================

%% generate_mindmap_glue(+Stage, +Target, -GlueCode)
generate_mindmap_glue(layout(Algorithm), python, GlueCode) :-
    get_mindmap_target(Algorithm, target(python, TargetOpts)),
    member(module(Module), TargetOpts),
    format(atom(GlueCode),
'#!/usr/bin/env python3
"""Mindmap layout glue - ~w"""
import sys
import json
from ~w import compute
from unifyweaver.mindmap.io import read_graph, write_positions

def main():
    graph = read_graph(sys.stdin)
    if not graph:
        data = json.loads(sys.stdin.read())
        nodes = {n["id"]: n.get("props", {}) for n in data.get("nodes", [])}
        edges = [(e["from"], e["to"]) for e in data.get("edges", [])]
    else:
        nodes = {n.id: {"label": n.label} for n in graph.nodes}
        edges = [(e.from_id, e.to_id) for e in graph.edges]

    positions = compute(nodes, edges, {})
    write_positions(positions)

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
from ~w import compute
from unifyweaver.mindmap.io import read_positions_dict, write_positions

def main():
    positions = read_positions_dict(sys.stdin)
    new_positions = compute(positions)
    write_positions(new_positions)

if __name__ == "__main__":
    main()
', [Optimizer, Module]).

%% generate_pipeline_glue(+Pipeline, +Options, -Script)
generate_pipeline_glue(Pipeline, _Options, Script) :-
    findall(Cmd,
        (   member(Stage, Pipeline),
            get_mindmap_target(Stage, target(python, Opts)),
            member(module(Mod), Opts),
            format(atom(Cmd),
                'python3 -c "from ~w import compute; from unifyweaver.mindmap.io import *; import sys,json; exec(open(\\'glue_~w.py\\').read())"',
                [Mod, Stage])
        ),
        Cmds),
    atomic_list_concat(Cmds, ' | ', PipelineCmd),
    format(atom(Script),
'#!/bin/bash
# Mindmap pipeline: ~w
# Generated by UnifyWeaver mindmap_glue

~w
', [Pipeline, PipelineCmd]).

% ============================================================================
% UTILITIES
% ============================================================================

get_mindmap_graph(MapId, graph(MapId, Nodes, Edges, [])) :-
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

% Janus-specific conversions
graph_to_py_dict(graph(_Id, Nodes, Edges, _), _Options, PyDict) :-
    nodes_to_py_dict(Nodes, NodesDict),
    edges_to_py_list(Edges, EdgesList),
    PyDict = py{nodes: NodesDict, edges: EdgesList}.

nodes_to_py_dict(Nodes, Dict) :-
    findall(Id-Props,
        (member(node(Id, PropList), Nodes), props_to_py(PropList, Props)),
        Pairs),
    dict_create(Dict, py, Pairs).

edges_to_py_list(Edges, List) :-
    findall([From, To], member(edge(From, To, _), Edges), List).

options_to_py_dict(Options, Dict) :-
    findall(Key-Value, member(Key=Value, Options), Pairs),
    dict_create(Dict, py, Pairs).

positions_to_py_dict(Positions, Dict) :-
    findall(Id-[X, Y], member(position(Id, X, Y), Positions), Pairs),
    dict_create(Dict, py, Pairs).

props_to_py(PropList, Props) :-
    findall(Key-Value, (member(Prop, PropList), Prop =.. [Key, Value]), Pairs),
    dict_create(Props, py, Pairs).

py_result_to_positions(PyResult, Positions) :-
    dict_pairs(PyResult, _, Pairs),
    findall(position(Id, X, Y),
        (member(Id-[X, Y], Pairs)),
        Positions).

% ============================================================================
% TESTING
% ============================================================================

test_mindmap_glue :-
    format('~n=== Mind Map Glue Tests ===~n~n'),

    % Test 1: Janus availability check (delegated to janus_glue)
    format('Test 1: Janus availability (via janus_glue)...~n'),
    (   mindmap_janus_available(Info)
    ->  format('  Janus available: ~w~n', [Info])
    ;   format('  Janus not available (will use pipe transport)~n')
    ),

    % Test 2: Mindmap packages check
    format('~nTest 2: Mindmap packages availability...~n'),
    check_mindmap_packages(PkgAvail),
    format('  Mindmap packages available: ~w~n', [PkgAvail]),

    % Test 3: Transport selection
    format('~nTest 3: Transport selection...~n'),
    select_transport(force_layout, [], Transport),
    format('  Selected transport: ~w~n', [Transport]),
    (   member(Transport, [janus, pipe, socket])
    ->  format('  PASS: Valid transport selected~n')
    ;   format('  FAIL: Invalid transport~n')
    ),

    % Test 4: Target declarations
    format('~nTest 4: Target declarations...~n'),
    (   get_mindmap_target(force_layout, target(python, Opts)),
        member(module(_), Opts)
    ->  format('  PASS: force_layout target exists~n')
    ;   format('  FAIL: force_layout target missing~n')
    ),

    % Test 5: Python code generation
    format('~nTest 5: Python code generation...~n'),
    generate_layout_python(force_layout, [], PyCode),
    (   sub_atom(PyCode, _, _, _, 'force_directed'),
        sub_atom(PyCode, _, _, _, 'compute')
    ->  format('  PASS: Python code generated~n')
    ;   format('  FAIL: Python code generation failed~n')
    ),

    % Test 6: Pipeline Python generation
    format('~nTest 6: Pipeline Python generation...~n'),
    generate_mindmap_python(test_map, [pipeline([force_layout, overlap_removal])], FullPyCode),
    (   sub_atom(FullPyCode, _, _, _, 'def process_mindmap'),
        sub_atom(FullPyCode, _, _, _, 'force_layout_compute')
    ->  format('  PASS: Full pipeline code generated~n')
    ;   format('  FAIL: Pipeline code generation failed~n')
    ),

    % Test 7: Glue code generation
    format('~nTest 7: Glue code generation...~n'),
    generate_mindmap_glue(layout(force_layout), python, GlueCode),
    (   sub_atom(GlueCode, _, _, _, 'read_graph'),
        sub_atom(GlueCode, _, _, _, 'write_positions')
    ->  format('  PASS: Glue code uses io module~n')
    ;   format('  FAIL: Glue code missing io module~n')
    ),

    % Test 8: Integration with janus_glue
    format('~nTest 8: Integration with janus_glue...~n'),
    (   catch(current_module(janus_glue), _, fail)
    ->  format('  PASS: janus_glue module accessible~n')
    ;   format('  INFO: janus_glue module not loaded (normal if not SWI-Prolog)~n')
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% INITIALIZATION
% ============================================================================

:- initialization((
    (   mindmap_janus_available(Info)
    ->  format('Mind map glue loaded (Janus: ~w)~n', [Info])
    ;   format('Mind map glue loaded (using pipe transport)~n')
    )
), now).
