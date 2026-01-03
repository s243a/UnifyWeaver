% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% custom_mindmap_optimizer.pl - Custom Mind Map Optimizer Component Type
%
% Allows defining custom optimization passes for mind map layouts.
% Optimizers refine positions to improve visual quality.
%
% Example:
%   declare_component(mindmap_optimizer, my_optimizer, custom_mindmap_optimizer, [
%       optimizer(overlap_removal),
%       options([min_distance(20), iterations(100)])
%   ]).

:- module(custom_mindmap_optimizer, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4,

    % Optimizer-specific predicates
    optimizer_passes/1,
    apply_optimizer/4,

    % Testing
    test_custom_optimizer/0
]).

:- use_module('../core/component_registry').
:- use_module(library(lists)).

% ============================================================================
% COMPONENT TYPE INTERFACE
% ============================================================================

%% type_info(-Info)
%
%  Metadata about the custom_mindmap_optimizer component type.
%
type_info(info(
    name('Custom Mind Map Optimizer'),
    version('1.0.0'),
    description('Defines custom optimization passes for mind map layouts'),
    category(mindmap_optimizer)
)).

%% validate_config(+Config)
%
%  Validate the component configuration.
%
validate_config(Config) :-
    (   member(optimizer(Opt), Config),
        valid_optimizer(Opt)
    ->  true
    ;   member(delegate(Target, _), Config),
        valid_delegate_target(Target)
    ->  true
    ;   member(code(_), Config)
    ->  true
    ;   throw(error(invalid_optimizer_config))
    ).

%% init_component(+Name, +Config)
init_component(Name, Config) :-
    (   member(optimizer(Opt), Config)
    ->  format('Initialized optimizer ~w with pass ~w~n', [Name, Opt])
    ;   format('Initialized optimizer ~w~n', [Name])
    ).

%% invoke_component(+Name, +Config, +Input, -Output)
%
%  Input: positions([(NodeId, X, Y), ...], NodeData, Options)
%  Output: positions([(NodeId, X, Y), ...])
%
invoke_component(_Name, Config, positions(Positions, NodeData, Options), Output) :-
    (   member(optimizer(Opt), Config)
    ->  apply_builtin_optimizer(Opt, Positions, NodeData, Options, Output)
    ;   member(code(Code), Config)
    ->  call(Code, Positions, NodeData, Options, Output)
    ).

%% compile_component(+Name, +Config, +Options, -Code)
compile_component(Name, Config, Options, Code) :-
    (   member(target(python), Options)
    ->  compile_to_python(Name, Config, Options, Code)
    ;   member(target(go), Options)
    ->  compile_to_go(Name, Config, Options, Code)
    ;   compile_to_python(Name, Config, Options, Code)
    ).

% ============================================================================
% BUILT-IN OPTIMIZERS
% ============================================================================

%% optimizer_passes(-Passes)
%
%  List available built-in optimizer passes.
%
optimizer_passes([
    overlap_removal,
    crossing_minimization,
    spacing_adjustment,
    edge_straightening,
    centering
]).

valid_optimizer(Opt) :-
    optimizer_passes(Passes),
    member(Opt, Passes).

valid_delegate_target(python).
valid_delegate_target(go).
valid_delegate_target(typescript).

%% apply_builtin_optimizer(+Optimizer, +Positions, +NodeData, +Options, -NewPositions)
apply_builtin_optimizer(overlap_removal, Positions, NodeData, Options, NewPositions) :-
    remove_overlaps(Positions, NodeData, Options, NewPositions).
apply_builtin_optimizer(crossing_minimization, Positions, _NodeData, Options, NewPositions) :-
    minimize_crossings(Positions, Options, NewPositions).
apply_builtin_optimizer(spacing_adjustment, Positions, _NodeData, Options, NewPositions) :-
    adjust_spacing(Positions, Options, NewPositions).
apply_builtin_optimizer(edge_straightening, Positions, _NodeData, Options, NewPositions) :-
    straighten_edges(Positions, Options, NewPositions).
apply_builtin_optimizer(centering, Positions, _NodeData, Options, NewPositions) :-
    center_graph(Positions, Options, NewPositions).

%% remove_overlaps(+Positions, +NodeData, +Options, -NewPositions)
%
%  Remove overlapping nodes by pushing them apart.
%
remove_overlaps(Positions, NodeData, Options, positions(NewPositions)) :-
    option_value(Options, min_distance, 50, MinDist),
    option_value(Options, iterations, 50, MaxIter),
    overlap_iterate(MaxIter, Positions, NodeData, MinDist, NewPositions).

overlap_iterate(0, Positions, _, _, Positions) :- !.
overlap_iterate(N, Positions, NodeData, MinDist, FinalPositions) :-
    N > 0,
    push_apart(Positions, NodeData, MinDist, NewPositions),
    (   Positions == NewPositions
    ->  FinalPositions = NewPositions  % Converged
    ;   N1 is N - 1,
        overlap_iterate(N1, NewPositions, NodeData, MinDist, FinalPositions)
    ).

push_apart(Positions, NodeData, MinDist, NewPositions) :-
    maplist(push_node_apart(Positions, NodeData, MinDist), Positions, NewPositions).

push_node_apart(AllPositions, NodeData, MinDist, (Id, X, Y), (Id, NewX, NewY)) :-
    % Get node dimensions
    get_node_dimensions(Id, NodeData, W, H),

    % Find overlapping nodes and compute push vector
    findall((DX, DY),
            (member((OtherId, OX, OY), AllPositions),
             OtherId \== Id,
             get_node_dimensions(OtherId, NodeData, OW, OH),
             nodes_overlap(X, Y, W, H, OX, OY, OW, OH, MinDist),
             compute_push_vector(X, Y, OX, OY, MinDist, DX, DY)),
            PushVectors),

    % Sum push vectors
    sum_vectors(PushVectors, TotalDX, TotalDY),
    Damping = 0.5,
    NewX is X + TotalDX * Damping,
    NewY is Y + TotalDY * Damping.

get_node_dimensions(Id, NodeData, W, H) :-
    (   member((Id, Props), NodeData),
        member(width(W), Props),
        member(height(H), Props)
    ->  true
    ;   W = 80, H = 40  % Default dimensions
    ).

nodes_overlap(X1, Y1, W1, H1, X2, Y2, W2, H2, MinDist) :-
    HalfW1 is W1 / 2, HalfH1 is H1 / 2,
    HalfW2 is W2 / 2, HalfH2 is H2 / 2,
    Left1 is X1 - HalfW1 - MinDist / 2,
    Right1 is X1 + HalfW1 + MinDist / 2,
    Top1 is Y1 - HalfH1 - MinDist / 2,
    Bottom1 is Y1 + HalfH1 + MinDist / 2,
    Left2 is X2 - HalfW2,
    Right2 is X2 + HalfW2,
    Top2 is Y2 - HalfH2,
    Bottom2 is Y2 + HalfH2,
    Left1 < Right2,
    Right1 > Left2,
    Top1 < Bottom2,
    Bottom1 > Top2.

compute_push_vector(X1, Y1, X2, Y2, MinDist, DX, DY) :-
    VX is X1 - X2,
    VY is Y1 - Y2,
    Dist is max(1, sqrt(VX * VX + VY * VY)),
    Force is MinDist / Dist,
    DX is (VX / Dist) * Force,
    DY is (VY / Dist) * Force.

sum_vectors([], 0, 0).
sum_vectors([(DX, DY) | Rest], TotalX, TotalY) :-
    sum_vectors(Rest, RestX, RestY),
    TotalX is DX + RestX,
    TotalY is DY + RestY.

%% minimize_crossings(+Positions, +Options, -NewPositions)
%
%  Minimize edge crossings by angular adjustment.
%  (Simplified implementation)
%
minimize_crossings(Positions, _Options, positions(Positions)).
% TODO: Implement crossing minimization

%% adjust_spacing(+Positions, +Options, -NewPositions)
%
%  Adjust node spacing for visual balance.
%
adjust_spacing(Positions, Options, positions(NewPositions)) :-
    option_value(Options, target_spacing, 100, TargetSpacing),
    adjust_spacing_pass(Positions, TargetSpacing, NewPositions).

adjust_spacing_pass(Positions, TargetSpacing, NewPositions) :-
    maplist(adjust_node_spacing(Positions, TargetSpacing), Positions, NewPositions).

adjust_node_spacing(AllPositions, TargetSpacing, (Id, X, Y), (Id, NewX, NewY)) :-
    % Find nearest neighbor
    findall(Dist-Other,
            (member((Other, OX, OY), AllPositions),
             Other \== Id,
             Dist is sqrt((X - OX)^2 + (Y - OY)^2)),
            Distances),
    (   Distances \== [],
        sort(Distances, [NearestDist-NearestId | _]),
        member((NearestId, NX, NY), AllPositions)
    ->  % Adjust if too close or too far
        Diff is TargetSpacing - NearestDist,
        (   abs(Diff) > 10
        ->  VX is X - NX,
            VY is Y - NY,
            Dist is max(1, sqrt(VX * VX + VY * VY)),
            Factor is Diff * 0.1 / Dist,
            NewX is X + VX * Factor,
            NewY is Y + VY * Factor
        ;   NewX = X, NewY = Y
        )
    ;   NewX = X, NewY = Y
    ).

%% straighten_edges(+Positions, +Options, -NewPositions)
%
%  Attempt to straighten edges where possible.
%  (Placeholder implementation)
%
straighten_edges(Positions, _Options, positions(Positions)).
% TODO: Implement edge straightening

%% center_graph(+Positions, +Options, -NewPositions)
%
%  Center the graph in the viewport.
%
center_graph(Positions, Options, positions(NewPositions)) :-
    option_value(Options, viewport_width, 800, VW),
    option_value(Options, viewport_height, 600, VH),

    % Find bounding box
    findall(X, member((_, X, _), Positions), Xs),
    findall(Y, member((_, _, Y), Positions), Ys),
    min_list(Xs, MinX), max_list(Xs, MaxX),
    min_list(Ys, MinY), max_list(Ys, MaxY),

    % Compute center offset
    CenterX is (MinX + MaxX) / 2,
    CenterY is (MinY + MaxY) / 2,
    TargetX is VW / 2,
    TargetY is VH / 2,
    OffsetX is TargetX - CenterX,
    OffsetY is TargetY - CenterY,

    % Apply offset
    maplist(offset_position(OffsetX, OffsetY), Positions, NewPositions).

offset_position(OffsetX, OffsetY, (Id, X, Y), (Id, NewX, NewY)) :-
    NewX is X + OffsetX,
    NewY is Y + OffsetY.

%% apply_optimizer(+Optimizer, +Positions, +NodeData, -NewPositions)
%
%  Public API to apply an optimizer.
%
apply_optimizer(Optimizer, Positions, NodeData, NewPositions) :-
    apply_builtin_optimizer(Optimizer, Positions, NodeData, [], NewPositions).

% ============================================================================
% CODE GENERATION
% ============================================================================

compile_to_python(Name, Config, _Options, Code) :-
    atom_string(Name, NameStr),
    (   member(optimizer(Opt), Config)
    ->  atom_string(Opt, OptStr),
        format(string(Code),
"# Custom Optimizer Component: ~w
from unifyweaver.mindmap.optimize import ~w

class Optimizer_~w:
    \"\"\"Custom optimizer using ~w pass.\"\"\"

    def __init__(self, options=None):
        self.options = options or {}

    def optimize(self, positions, node_data):
        \"\"\"Optimize node positions.\"\"\"
        return ~w(positions, node_data, **self.options)
", [NameStr, OptStr, NameStr, OptStr, OptStr])
    ;   Code = "# Invalid configuration"
    ).

compile_to_go(Name, Config, _Options, Code) :-
    atom_string(Name, NameStr),
    (   member(optimizer(Opt), Config)
    ->  atom_string(Opt, OptStr),
        format(string(Code),
"// Custom Optimizer Component: ~w
package optimize

import \"github.com/unifyweaver/mindmap/optimize\"

type Optimizer~w struct {
    Options map[string]interface{}
}

func (o *Optimizer~w) Optimize(positions []Position, nodeData []NodeData) []Position {
    return optimize.~w(positions, nodeData, o.Options)
}
", [NameStr, NameStr, NameStr, OptStr])
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

test_custom_optimizer :-
    format('~n=== Custom Mind Map Optimizer Tests ===~n~n'),

    % Test 1: Validate config
    format('Test 1: Validate config...~n'),
    (   validate_config([optimizer(overlap_removal)])
    ->  format('  PASS: Optimizer config valid~n')
    ;   format('  FAIL: Optimizer config should be valid~n')
    ),

    % Test 2: Overlap removal
    format('~nTest 2: Overlap removal...~n'),
    TestPositions = [(a, 100, 100), (b, 110, 105), (c, 300, 300)],
    NodeData = [(a, [width(80), height(40)]),
                (b, [width(80), height(40)]),
                (c, [width(80), height(40)])],
    remove_overlaps(TestPositions, NodeData, [iterations(20)], positions(NewPos)),
    (   length(NewPos, 3)
    ->  format('  PASS: Overlap removal completed~n')
    ;   format('  FAIL: Overlap removal failed~n')
    ),

    % Test 3: Centering
    format('~nTest 3: Centering...~n'),
    OffsetPositions = [(a, 0, 0), (b, 100, 0), (c, 50, 100)],
    center_graph(OffsetPositions, [viewport_width(800), viewport_height(600)], positions(CenteredPos)),
    (   member((a, CX, _), CenteredPos), CX > 300
    ->  format('  PASS: Graph centered~n')
    ;   format('  FAIL: Centering incorrect~n')
    ),

    % Test 4: Compile to Python
    format('~nTest 4: Compile to Python...~n'),
    compile_to_python(test_opt, [optimizer(overlap_removal)], [], PyCode),
    (   sub_string(PyCode, _, _, _, "class Optimizer_test_opt")
    ->  format('  PASS: Python code generated~n')
    ;   format('  FAIL: Python generation incorrect~n')
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% REGISTRATION
% ============================================================================

:- initialization((
    register_component_type(mindmap_optimizer, custom_mindmap_optimizer, custom_mindmap_optimizer, [
        description("Custom Mind Map Optimizer Pass")
    ]),
    format('Custom mind map optimizer component type registered~n', [])
), now).
