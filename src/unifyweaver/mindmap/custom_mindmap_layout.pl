% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% custom_mindmap_layout.pl - Custom Mind Map Layout Component Type
%
% Allows defining custom layout algorithms for mind maps.
% Can delegate to target-specific custom components (Python, Go, etc.)
% or implement pure Prolog layout logic.
%
% Example:
%   declare_component(mindmap_layout, my_custom_layout, custom_mindmap_layout, [
%       algorithm(force_directed),
%       delegate(python, "my_layout.compute"),
%       options([iterations(100), spring_k(0.1)])
%   ]).

:- module(custom_mindmap_layout, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4,

    % Layout-specific predicates
    layout_algorithms/1,
    apply_layout/4,

    % Testing
    test_custom_layout/0
]).

:- use_module('../core/component_registry').
:- use_module(library(lists)).

% ============================================================================
% COMPONENT TYPE INTERFACE
% ============================================================================

%% type_info(-Info)
%
%  Metadata about the custom_mindmap_layout component type.
%
type_info(info(
    name('Custom Mind Map Layout'),
    version('1.0.0'),
    description('Defines custom layout algorithms for mind map visualization'),
    category(mindmap_layout)
)).

%% validate_config(+Config)
%
%  Validate the component configuration.
%  Either 'algorithm' or 'delegate' must be specified.
%
validate_config(Config) :-
    (   member(algorithm(Algo), Config),
        valid_algorithm(Algo)
    ->  true
    ;   member(delegate(Target, _), Config),
        valid_delegate_target(Target)
    ->  true
    ;   member(code(_), Config)
    ->  true
    ;   throw(error(invalid_layout_config(must_specify_algorithm_delegate_or_code)))
    ).

%% init_component(+Name, +Config)
%
%  Initialize the layout component.
%
init_component(Name, Config) :-
    (   member(algorithm(Algo), Config)
    ->  format('Initialized layout component ~w with algorithm ~w~n', [Name, Algo])
    ;   member(delegate(Target, Func), Config)
    ->  format('Initialized layout component ~w delegating to ~w:~w~n', [Name, Target, Func])
    ;   format('Initialized layout component ~w with custom code~n', [Name])
    ).

%% invoke_component(+Name, +Config, +Input, -Output)
%
%  Invoke the layout component at runtime.
%  Input: graph(Nodes, Edges, Options)
%  Output: positions([(NodeId, X, Y), ...])
%
invoke_component(Name, Config, graph(Nodes, Edges, Options), Output) :-
    (   member(algorithm(Algo), Config)
    ->  apply_builtin_algorithm(Algo, Nodes, Edges, Options, Output)
    ;   member(delegate(Target, _Func), Config)
    ->  % Delegation requires target-specific runtime
        format(user_error, 'Runtime delegation to ~w not supported in Prolog~n', [Target]),
        throw(error(runtime_delegation_not_supported(Name, Target)))
    ;   member(code(Code), Config)
    ->  % Custom Prolog code - call it directly
        call(Code, Nodes, Edges, Options, Output)
    ).

%% compile_component(+Name, +Config, +Options, -Code)
%
%  Compile the layout component to target code.
%
compile_component(Name, Config, Options, Code) :-
    (   member(target(python), Options)
    ->  compile_to_python(Name, Config, Options, Code)
    ;   member(target(go), Options)
    ->  compile_to_go(Name, Config, Options, Code)
    ;   member(target(typescript), Options)
    ->  compile_to_typescript(Name, Config, Options, Code)
    ;   % Default to Python
        compile_to_python(Name, Config, Options, Code)
    ).

% ============================================================================
% BUILT-IN ALGORITHMS
% ============================================================================

%% layout_algorithms(-Algorithms)
%
%  List available built-in layout algorithms.
%
layout_algorithms([
    force_directed,
    radial,
    hierarchical,
    grid,
    circular
]).

valid_algorithm(Algo) :-
    layout_algorithms(Algos),
    member(Algo, Algos).

valid_delegate_target(python).
valid_delegate_target(go).
valid_delegate_target(typescript).
valid_delegate_target(javascript).

%% apply_builtin_algorithm(+Algorithm, +Nodes, +Edges, +Options, -Positions)
apply_builtin_algorithm(force_directed, Nodes, Edges, Options, Positions) :-
    apply_force_directed(Nodes, Edges, Options, Positions).
apply_builtin_algorithm(radial, Nodes, Edges, Options, Positions) :-
    apply_radial(Nodes, Edges, Options, Positions).
apply_builtin_algorithm(hierarchical, Nodes, Edges, Options, Positions) :-
    apply_hierarchical(Nodes, Edges, Options, Positions).
apply_builtin_algorithm(grid, Nodes, _Edges, Options, Positions) :-
    apply_grid(Nodes, Options, Positions).
apply_builtin_algorithm(circular, Nodes, _Edges, Options, Positions) :-
    apply_circular(Nodes, Options, Positions).

%% apply_force_directed(+Nodes, +Edges, +Options, -Positions)
%
%  Simple force-directed layout (pure Prolog, for small graphs).
%
apply_force_directed(Nodes, Edges, Options, positions(Positions)) :-
    option_value(Options, iterations, 50, Iterations),
    option_value(Options, spring_k, 0.1, SpringK),
    option_value(Options, repulsion, 1000, Repulsion),

    % Initialize random positions
    length(Nodes, N),
    init_random_positions(Nodes, N, InitPositions),

    % Iterate
    force_directed_iterate(Iterations, InitPositions, Edges, SpringK, Repulsion, Positions).

init_random_positions([], _, []).
init_random_positions([node(Id, _) | Rest], N, [(Id, X, Y) | RestPos]) :-
    % Spread nodes in a grid initially
    length(Rest, Remaining),
    Index is N - Remaining - 1,
    Cols is ceiling(sqrt(N)),
    Row is Index // Cols,
    Col is Index mod Cols,
    X is Col * 100 + 50,
    Y is Row * 100 + 50,
    init_random_positions(Rest, N, RestPos).

force_directed_iterate(0, Positions, _, _, _, Positions) :- !.
force_directed_iterate(N, Positions, Edges, SpringK, Repulsion, FinalPositions) :-
    N > 0,
    apply_forces(Positions, Edges, SpringK, Repulsion, NewPositions),
    N1 is N - 1,
    force_directed_iterate(N1, NewPositions, Edges, SpringK, Repulsion, FinalPositions).

apply_forces(Positions, Edges, SpringK, Repulsion, NewPositions) :-
    maplist(compute_node_forces(Positions, Edges, SpringK, Repulsion), Positions, NewPositions).

compute_node_forces(AllPositions, Edges, SpringK, Repulsion, (Id, X, Y), (Id, NewX, NewY)) :-
    % Compute repulsion from all other nodes
    compute_repulsion(Id, X, Y, AllPositions, Repulsion, RepX, RepY),
    % Compute spring attraction from connected nodes
    compute_attraction(Id, X, Y, AllPositions, Edges, SpringK, AttrX, AttrY),
    % Apply forces with damping
    Damping = 0.85,
    DX is (RepX + AttrX) * Damping,
    DY is (RepY + AttrY) * Damping,
    NewX is X + DX,
    NewY is Y + DY.

compute_repulsion(_, _, _, [], _, 0, 0).
compute_repulsion(Id, X, Y, [(Id, _, _) | Rest], Repulsion, RX, RY) :-
    !,
    compute_repulsion(Id, X, Y, Rest, Repulsion, RX, RY).
compute_repulsion(Id, X, Y, [(_, OX, OY) | Rest], Repulsion, RX, RY) :-
    DX is X - OX,
    DY is Y - OY,
    Dist is max(1, sqrt(DX * DX + DY * DY)),
    Force is Repulsion / (Dist * Dist),
    RXPart is (DX / Dist) * Force,
    RYPart is (DY / Dist) * Force,
    compute_repulsion(Id, X, Y, Rest, Repulsion, RestRX, RestRY),
    RX is RXPart + RestRX,
    RY is RYPart + RestRY.

compute_attraction(_, _, _, _, [], _, 0, 0).
compute_attraction(Id, X, Y, Positions, [edge(From, To, _) | Rest], SpringK, AX, AY) :-
    (   (From == Id ; To == Id)
    ->  (From == Id -> Other = To ; Other = From),
        (   member((Other, OX, OY), Positions)
        ->  DX is OX - X,
            DY is OY - Y,
            Dist is sqrt(DX * DX + DY * DY),
            Force is SpringK * Dist,
            AXPart is (DX / max(1, Dist)) * Force,
            AYPart is (DY / max(1, Dist)) * Force
        ;   AXPart = 0, AYPart = 0
        )
    ;   AXPart = 0, AYPart = 0
    ),
    compute_attraction(Id, X, Y, Positions, Rest, SpringK, RestAX, RestAY),
    AX is AXPart + RestAX,
    AY is AYPart + RestAY.

%% apply_radial(+Nodes, +Edges, +Options, -Positions)
%
%  Radial layout around a central root node.
%
apply_radial(Nodes, Edges, Options, positions(Positions)) :-
    option_value(Options, center_x, 400, CenterX),
    option_value(Options, center_y, 300, CenterY),
    option_value(Options, radius_step, 100, RadiusStep),

    % Find root node
    find_root_node(Nodes, Edges, RootId),

    % Build level assignments
    assign_levels(RootId, Nodes, Edges, Levels),

    % Position nodes by level
    position_radially(Levels, CenterX, CenterY, RadiusStep, Positions).

find_root_node(Nodes, Edges, RootId) :-
    % Root is a node with no incoming edges
    member(node(RootId, Props), Nodes),
    (   member(type(root), Props)
    ->  true
    ;   \+ (member(edge(_, RootId, _), Edges))
    ),
    !.
find_root_node([node(RootId, _) | _], _, RootId).

assign_levels(RootId, Nodes, Edges, Levels) :-
    assign_levels_bfs([(RootId, 0)], Nodes, Edges, [], Levels).

assign_levels_bfs([], _, _, Acc, Acc).
assign_levels_bfs([(Id, Level) | Queue], Nodes, Edges, Acc, Levels) :-
    (   member((Id, _), Acc)
    ->  % Already assigned
        assign_levels_bfs(Queue, Nodes, Edges, Acc, Levels)
    ;   % Find children
        findall((ChildId, NextLevel),
                (member(edge(Id, ChildId, _), Edges),
                 \+ member((ChildId, _), Acc),
                 NextLevel is Level + 1),
                Children),
        append(Queue, Children, NewQueue),
        assign_levels_bfs(NewQueue, Nodes, Edges, [(Id, Level) | Acc], Levels)
    ).

position_radially(Levels, CenterX, CenterY, RadiusStep, Positions) :-
    % Group by level
    findall(L, member((_, L), Levels), AllLevels),
    sort(AllLevels, UniqueLevels),
    position_by_level(UniqueLevels, Levels, CenterX, CenterY, RadiusStep, Positions).

position_by_level([], _, _, _, _, []).
position_by_level([Level | RestLevels], AllLevels, CX, CY, RadiusStep, Positions) :-
    findall(Id, member((Id, Level), AllLevels), NodesAtLevel),
    length(NodesAtLevel, N),
    Radius is Level * RadiusStep,
    position_nodes_on_circle(NodesAtLevel, N, 0, CX, CY, Radius, LevelPositions),
    position_by_level(RestLevels, AllLevels, CX, CY, RadiusStep, RestPositions),
    append(LevelPositions, RestPositions, Positions).

position_nodes_on_circle([], _, _, _, _, _, []).
position_nodes_on_circle([Id | Rest], Total, Index, CX, CY, Radius, [(Id, X, Y) | RestPos]) :-
    (   Radius =:= 0
    ->  X = CX, Y = CY
    ;   Angle is 2 * pi * Index / Total,
        X is CX + Radius * cos(Angle),
        Y is CY + Radius * sin(Angle)
    ),
    Index1 is Index + 1,
    position_nodes_on_circle(Rest, Total, Index1, CX, CY, Radius, RestPos).

%% apply_hierarchical(+Nodes, +Edges, +Options, -Positions)
apply_hierarchical(Nodes, Edges, Options, positions(Positions)) :-
    option_value(Options, level_height, 80, LevelHeight),
    option_value(Options, node_spacing, 100, NodeSpacing),

    find_root_node(Nodes, Edges, RootId),
    assign_levels(RootId, Nodes, Edges, Levels),
    position_hierarchically(Levels, LevelHeight, NodeSpacing, Positions).

position_hierarchically(Levels, LevelHeight, NodeSpacing, Positions) :-
    findall(L, member((_, L), Levels), AllLevels),
    sort(AllLevels, UniqueLevels),
    position_hier_by_level(UniqueLevels, Levels, LevelHeight, NodeSpacing, Positions).

position_hier_by_level([], _, _, _, []).
position_hier_by_level([Level | RestLevels], AllLevels, LevelHeight, NodeSpacing, Positions) :-
    findall(Id, member((Id, Level), AllLevels), NodesAtLevel),
    length(NodesAtLevel, N),
    Y is Level * LevelHeight + 50,
    StartX is -(N - 1) * NodeSpacing / 2 + 400,
    position_nodes_horizontal(NodesAtLevel, StartX, Y, NodeSpacing, LevelPositions),
    position_hier_by_level(RestLevels, AllLevels, LevelHeight, NodeSpacing, RestPositions),
    append(LevelPositions, RestPositions, Positions).

position_nodes_horizontal([], _, _, _, []).
position_nodes_horizontal([Id | Rest], X, Y, Spacing, [(Id, X, Y) | RestPos]) :-
    NextX is X + Spacing,
    position_nodes_horizontal(Rest, NextX, Y, Spacing, RestPos).

%% apply_grid(+Nodes, +Options, -Positions)
apply_grid(Nodes, Options, positions(Positions)) :-
    option_value(Options, cell_width, 120, CellW),
    option_value(Options, cell_height, 80, CellH),
    option_value(Options, columns, 0, ColsOpt),

    length(Nodes, N),
    (   ColsOpt > 0
    ->  Cols = ColsOpt
    ;   Cols is ceiling(sqrt(N))
    ),
    position_in_grid(Nodes, 0, Cols, CellW, CellH, Positions).

position_in_grid([], _, _, _, _, []).
position_in_grid([node(Id, _) | Rest], Index, Cols, CellW, CellH, [(Id, X, Y) | RestPos]) :-
    Row is Index // Cols,
    Col is Index mod Cols,
    X is Col * CellW + CellW / 2,
    Y is Row * CellH + CellH / 2,
    Index1 is Index + 1,
    position_in_grid(Rest, Index1, Cols, CellW, CellH, RestPos).

%% apply_circular(+Nodes, +Options, -Positions)
apply_circular(Nodes, Options, positions(Positions)) :-
    option_value(Options, center_x, 400, CX),
    option_value(Options, center_y, 300, CY),
    option_value(Options, radius, 200, Radius),

    length(Nodes, N),
    position_circular(Nodes, 0, N, CX, CY, Radius, Positions).

position_circular([], _, _, _, _, _, []).
position_circular([node(Id, _) | Rest], Index, Total, CX, CY, Radius, [(Id, X, Y) | RestPos]) :-
    Angle is 2 * pi * Index / Total,
    X is CX + Radius * cos(Angle),
    Y is CY + Radius * sin(Angle),
    Index1 is Index + 1,
    position_circular(Rest, Index1, Total, CX, CY, Radius, RestPos).

%% apply_layout(+Algorithm, +Nodes, +Edges, -Positions)
%
%  Public API to apply a layout algorithm.
%
apply_layout(Algorithm, Nodes, Edges, Positions) :-
    apply_builtin_algorithm(Algorithm, Nodes, Edges, [], Positions).

% ============================================================================
% CODE GENERATION
% ============================================================================

compile_to_python(Name, Config, _Options, Code) :-
    atom_string(Name, NameStr),
    (   member(algorithm(Algo), Config)
    ->  atom_string(Algo, AlgoStr),
        format(string(Code),
"# Custom Layout Component: ~w
from unifyweaver.mindmap.layout import ~w

class Layout_~w:
    \"\"\"Custom layout component using ~w algorithm.\"\"\"

    def __init__(self, options=None):
        self.options = options or {}

    def compute(self, nodes, edges):
        \"\"\"Compute node positions.\"\"\"
        return ~w(nodes, edges, **self.options)
", [NameStr, AlgoStr, NameStr, AlgoStr, AlgoStr])
    ;   member(delegate(python, Func), Config)
    ->  format(string(Code),
"# Custom Layout Component: ~w (delegated)
from ~w import compute as delegate_compute

class Layout_~w:
    \"\"\"Custom layout component delegating to ~w.\"\"\"

    def __init__(self, options=None):
        self.options = options or {}

    def compute(self, nodes, edges):
        \"\"\"Compute node positions.\"\"\"
        return delegate_compute(nodes, edges, **self.options)
", [NameStr, Func, NameStr, Func])
    ;   member(code(CustomCode), Config)
    ->  format(string(Code),
"# Custom Layout Component: ~w (inline code)

class Layout_~w:
    \"\"\"Custom layout component with inline code.\"\"\"

    def __init__(self, options=None):
        self.options = options or {}

    def compute(self, nodes, edges):
        \"\"\"Compute node positions.\"\"\"
~w
", [NameStr, NameStr, CustomCode])
    ;   Code = "# Invalid configuration"
    ).

compile_to_go(Name, Config, _Options, Code) :-
    atom_string(Name, NameStr),
    (   member(algorithm(Algo), Config)
    ->  atom_string(Algo, AlgoStr),
        format(string(Code),
"// Custom Layout Component: ~w
package layout

import \"github.com/unifyweaver/mindmap/layout\"

type Layout~w struct {
    Options map[string]interface{}
}

func (l *Layout~w) Compute(nodes []Node, edges []Edge) []Position {
    return layout.~w(nodes, edges, l.Options)
}
", [NameStr, NameStr, NameStr, AlgoStr])
    ;   Code = "// Invalid configuration"
    ).

compile_to_typescript(Name, Config, _Options, Code) :-
    atom_string(Name, NameStr),
    (   member(algorithm(Algo), Config)
    ->  atom_string(Algo, AlgoStr),
        format(string(Code),
"// Custom Layout Component: ~w
import { ~w } from '@unifyweaver/mindmap/layout';
import { Node, Edge, Position, LayoutOptions } from '@unifyweaver/mindmap/types';

export class Layout~w {
    constructor(private options: LayoutOptions = {}) {}

    compute(nodes: Node[], edges: Edge[]): Position[] {
        return ~w(nodes, edges, this.options);
    }
}
", [NameStr, AlgoStr, NameStr, AlgoStr])
    ;   Code = "// Invalid configuration"
    ).

% ============================================================================
% UTILITIES
% ============================================================================

option_value(Options, Key, Default, Value) :-
    Opt =.. [Key, Value],
    (   member(Opt, Options)
    ->  true
    ;   Value = Default
    ).

% ============================================================================
% TESTING
% ============================================================================

test_custom_layout :-
    format('~n=== Custom Mind Map Layout Tests ===~n~n'),

    % Test 1: Validate config
    format('Test 1: Validate config...~n'),
    (   validate_config([algorithm(force_directed)])
    ->  format('  PASS: Algorithm config valid~n')
    ;   format('  FAIL: Algorithm config should be valid~n')
    ),

    % Test 2: Grid layout
    format('~nTest 2: Grid layout...~n'),
    TestNodes = [node(a, []), node(b, []), node(c, []), node(d, [])],
    apply_grid(TestNodes, [columns(2)], positions(GridPos)),
    (   length(GridPos, 4)
    ->  format('  PASS: Grid positioned 4 nodes~n')
    ;   format('  FAIL: Grid layout incorrect~n')
    ),

    % Test 3: Circular layout
    format('~nTest 3: Circular layout...~n'),
    apply_circular(TestNodes, [radius(100)], positions(CircPos)),
    (   length(CircPos, 4)
    ->  format('  PASS: Circular positioned 4 nodes~n')
    ;   format('  FAIL: Circular layout incorrect~n')
    ),

    % Test 4: Force-directed layout
    format('~nTest 4: Force-directed layout...~n'),
    TestEdges = [edge(a, b, []), edge(b, c, []), edge(c, d, [])],
    apply_force_directed(TestNodes, TestEdges, [iterations(10)], positions(ForcePos)),
    (   length(ForcePos, 4)
    ->  format('  PASS: Force-directed positioned 4 nodes~n')
    ;   format('  FAIL: Force-directed layout incorrect~n')
    ),

    % Test 5: Compile to Python
    format('~nTest 5: Compile to Python...~n'),
    compile_to_python(test_layout, [algorithm(radial)], [], PyCode),
    (   sub_string(PyCode, _, _, _, "class Layout_test_layout")
    ->  format('  PASS: Python code generated~n')
    ;   format('  FAIL: Python generation incorrect~n')
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% REGISTRATION
% ============================================================================

:- initialization((
    register_component_type(mindmap_layout, custom_mindmap_layout, custom_mindmap_layout, [
        description("Custom Mind Map Layout Algorithm")
    ]),
    format('Custom mind map layout component type registered~n', [])
), now).
