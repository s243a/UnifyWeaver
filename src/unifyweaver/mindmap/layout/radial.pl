% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% radial.pl - Radial Layout Algorithm for Mind Maps
%
% Implements a radial/circular layout where the root node is at the center
% and child nodes are placed in concentric circles around it.
%
% The algorithm:
% 1. Place root at center
% 2. For each level, calculate radius needed to maintain minimum spacing
% 3. Position children in their parent's angular sector
%
% Based on algorithms from generate_simplemind_map.py

:- module(mindmap_layout_radial, [
    % Component interface
    type_info/1,
    validate_config/1,
    init_component/2,
    compute_layout/3,

    % Direct API
    radial_layout/4,                % radial_layout(+IR, +Options, -Positions, -Stats)
    count_nodes_per_level/3,        % count_nodes_per_level(+IR, +RootId, -LevelCounts)
    compute_level_radii/3           % compute_level_radii(+LevelCounts, +Options, -Radii)
]).

:- use_module(library(lists)).

% ============================================================================
% COMPONENT INTERFACE
% ============================================================================

%% type_info(-Info)
%
%  Component type information.
%
type_info(info{
    name: radial,
    category: mindmap_layout,
    description: "Radial layout with root at center",
    version: "1.0.0",
    parameters: [
        center_x - "X coordinate of center (default 500)",
        center_y - "Y coordinate of center (default 500)",
        base_radius - "Base radius for first level (default 150)",
        min_spacing - "Minimum spacing between nodes (default 80)",
        level_growth - "How radius grows per level: linear, sqrt, log (default linear)"
    ]
}).

%% validate_config(+Config)
%
%  Validate layout configuration.
%
validate_config(Config) :-
    is_list(Config),
    (   member(center_x(X), Config)
    ->  number(X)
    ;   true
    ),
    (   member(center_y(Y), Config)
    ->  number(Y)
    ;   true
    ),
    (   member(base_radius(R), Config)
    ->  number(R), R > 0
    ;   true
    ),
    (   member(min_spacing(S), Config)
    ->  number(S), S > 0
    ;   true
    ).

%% init_component(+Name, +Config)
%
%  Initialize the component (no-op for radial layout).
%
init_component(_Name, _Config).

%% compute_layout(+Graph, +Options, -Positions)
%
%  Main entry point for component invocation.
%
%  @param Graph     term - graph(Nodes, Edges, RootId)
%  @param Options   list - layout options
%  @param Positions list - list of position(Id, X, Y) terms
%
compute_layout(graph(Nodes, Edges, RootId), Options, Positions) :-
    % Build minimal IR for internal use
    IR = ir(Nodes, Edges, RootId),
    radial_layout(IR, Options, Positions, _Stats).

% ============================================================================
% RADIAL LAYOUT ALGORITHM
% ============================================================================

%% radial_layout(+IR, +Options, -Positions, -Stats)
%
%  Compute radial layout positions.
%
%  @param IR        term - ir(Nodes, Edges, RootId) or mindmap_ir term
%  @param Options   list - layout options
%  @param Positions list - list of position(Id, X, Y)
%  @param Stats     dict - layout statistics
%
radial_layout(IR, Options, Positions, Stats) :-
    % Extract graph data
    extract_ir_data(IR, Nodes, Edges, RootId),

    % Get options with defaults
    option_or_default(center_x, Options, 500, CenterX),
    option_or_default(center_y, Options, 500, CenterY),
    option_or_default(base_radius, Options, 150, BaseRadius),
    option_or_default(min_spacing, Options, 80, MinSpacing),

    % Build adjacency for children lookup
    build_children_map(Edges, ChildrenMap),

    % Count nodes at each level
    count_nodes_per_level_impl(RootId, ChildrenMap, 0, [], LevelCounts),

    % Compute radius for each level
    compute_level_radii(LevelCounts, [base_radius(BaseRadius), min_spacing(MinSpacing)], LevelRadii),

    % Position nodes recursively
    position_nodes(RootId, ChildrenMap, CenterX, CenterY, LevelRadii,
                   0, 0, 2*pi, [], Positions),

    % Compute statistics
    length(Nodes, NumNodes),
    length(Positions, NumPositioned),
    (   LevelRadii = [_ | _]
    ->  last(LevelRadii, _-MaxRadius)
    ;   MaxRadius = 0
    ),
    Stats = stats{
        nodes_total: NumNodes,
        nodes_positioned: NumPositioned,
        max_radius: MaxRadius,
        levels: LevelCounts
    }.

%% extract_ir_data(+IR, -Nodes, -Edges, -RootId)
%
%  Extract nodes, edges, and root from IR structure.
%
extract_ir_data(ir(Nodes, Edges, RootId), Nodes, Edges, RootId) :- !.
extract_ir_data(mindmap_ir(_, graph(Nodes, Edges, RootId), _, _, _), Nodes, Edges, RootId) :- !.
extract_ir_data(graph(Nodes, Edges, RootId), Nodes, Edges, RootId).

%% build_children_map(+Edges, -ChildrenMap)
%
%  Build a map from parent ID to list of children IDs.
%
build_children_map(Edges, ChildrenMap) :-
    findall(Parent-Children,
            (setof(Child, member(edge(Parent, Child, _), Edges), Children)),
            Pairs),
    ChildrenMap = Pairs.

get_children(Id, ChildrenMap, Children) :-
    (   member(Id-Children, ChildrenMap)
    ->  true
    ;   Children = []
    ).

%% count_nodes_per_level(+IR, +RootId, -LevelCounts)
%
%  Count how many nodes are at each depth level.
%
%  @param IR          term - IR structure
%  @param RootId      atom - root node ID
%  @param LevelCounts list - list of Level-Count pairs
%
count_nodes_per_level(IR, RootId, LevelCounts) :-
    extract_ir_data(IR, _Nodes, Edges, _),
    build_children_map(Edges, ChildrenMap),
    count_nodes_per_level_impl(RootId, ChildrenMap, 0, [], LevelCounts).

count_nodes_per_level_impl(NodeId, ChildrenMap, Level, AccIn, AccOut) :-
    % Increment count for this level
    (   select(Level-Count, AccIn, AccRest)
    ->  NewCount is Count + 1,
        Acc1 = [Level-NewCount | AccRest]
    ;   Acc1 = [Level-1 | AccIn]
    ),
    % Process children
    get_children(NodeId, ChildrenMap, Children),
    ChildLevel is Level + 1,
    foldl(count_child(ChildrenMap, ChildLevel), Children, Acc1, AccOut).

count_child(ChildrenMap, Level, ChildId, AccIn, AccOut) :-
    count_nodes_per_level_impl(ChildId, ChildrenMap, Level, AccIn, AccOut).

%% compute_level_radii(+LevelCounts, +Options, -Radii)
%
%  Compute the radius for each level to maintain minimum spacing.
%
%  Formula: radius = max(base_radius, n_nodes * min_spacing / (2 * pi))
%
%  @param LevelCounts list - list of Level-Count pairs
%  @param Options     list - options including base_radius, min_spacing
%  @param Radii       list - list of Level-Radius pairs
%
compute_level_radii(LevelCounts, Options, Radii) :-
    option_or_default(base_radius, Options, 150, BaseRadius),
    option_or_default(min_spacing, Options, 80, MinSpacing),

    % Sort by level
    msort(LevelCounts, SortedCounts),

    % Compute cumulative radius
    compute_radii_impl(SortedCounts, BaseRadius, MinSpacing, 0, [], Radii).

compute_radii_impl([], _, _, _, Acc, Radii) :-
    reverse(Acc, Radii).
compute_radii_impl([Level-Count | Rest], BaseRadius, MinSpacing, CumulativeR, Acc, Radii) :-
    (   Level =:= 0
    ->  % Root is at center
        LevelRadius = 0,
        NewCumulative = 0
    ;   % Calculate radius needed for this level's circumference
        % circumference = 2 * pi * r, spacing = circumference / n_nodes
        % So r = n_nodes * spacing / (2 * pi)
        Pi is 3.14159265358979,
        NeededRadius is max(BaseRadius, Count * MinSpacing / (2 * Pi)),
        NewCumulative is CumulativeR + NeededRadius,
        LevelRadius = NewCumulative
    ),
    compute_radii_impl(Rest, BaseRadius, MinSpacing, NewCumulative, [Level-LevelRadius | Acc], Radii).

%% position_nodes(+NodeId, +ChildrenMap, +CX, +CY, +Radii, +Level, +AngleStart, +AngleSpan, +AccIn, -AccOut)
%
%  Recursively position nodes.
%
position_nodes(NodeId, ChildrenMap, CX, CY, Radii, Level, AngleStart, AngleSpan, AccIn, AccOut) :-
    % Get radius for this level
    (   member(Level-Radius, Radii)
    ->  true
    ;   Radius = 0
    ),

    % Compute position
    (   Level =:= 0
    ->  X = CX, Y = CY
    ;   % Position at this level's radius within the angular sector
        AngleMid is AngleStart + AngleSpan / 2,
        X is CX + Radius * cos(AngleMid),
        Y is CY + Radius * sin(AngleMid)
    ),

    Acc1 = [position(NodeId, X, Y) | AccIn],

    % Position children
    get_children(NodeId, ChildrenMap, Children),
    length(Children, NumChildren),
    (   NumChildren > 0
    ->  ChildLevel is Level + 1,
        ChildAngleSpan is AngleSpan / NumChildren,
        position_children(Children, ChildrenMap, CX, CY, Radii, ChildLevel,
                         AngleStart, ChildAngleSpan, Acc1, AccOut)
    ;   AccOut = Acc1
    ).

position_children([], _, _, _, _, _, _, _, Acc, Acc).
position_children([Child | Rest], ChildrenMap, CX, CY, Radii, Level, AngleStart, AngleSpan, AccIn, AccOut) :-
    position_nodes(Child, ChildrenMap, CX, CY, Radii, Level, AngleStart, AngleSpan, AccIn, Acc1),
    NextAngleStart is AngleStart + AngleSpan,
    position_children(Rest, ChildrenMap, CX, CY, Radii, Level, NextAngleStart, AngleSpan, Acc1, AccOut).

% ============================================================================
% UTILITIES
% ============================================================================

%% option_or_default(+Key, +Options, +Default, -Value)
%
%  Get option value or default.
%
option_or_default(Key, Options, Default, Value) :-
    Term =.. [Key, Value],
    (   member(Term, Options)
    ->  true
    ;   Value = Default
    ).

% Math constants
pi is 3.14159265358979.

% ============================================================================
% TESTING
% ============================================================================

test_radial_layout :-
    format('~n=== Radial Layout Tests ===~n~n'),

    % Test data
    Nodes = [
        node(root, [label("Root")]),
        node(a, [label("A")]),
        node(b, [label("B")]),
        node(c, [label("C")]),
        node(d, [label("D")])
    ],
    Edges = [
        edge(root, a, []),
        edge(root, b, []),
        edge(a, c, []),
        edge(a, d, [])
    ],

    % Test 1: Compute layout
    format('Test 1: Compute radial layout...~n'),
    compute_layout(graph(Nodes, Edges, root), [], Positions),
    length(Positions, NumPos),
    (   NumPos =:= 5
    ->  format('  PASS: Generated ~w positions~n', [NumPos])
    ;   format('  FAIL: Expected 5 positions, got ~w~n', [NumPos])
    ),

    % Test 2: Root at center
    format('~nTest 2: Root at center...~n'),
    member(position(root, RX, RY), Positions),
    (   RX =:= 500, RY =:= 500
    ->  format('  PASS: Root at (500, 500)~n')
    ;   format('  FAIL: Root at (~w, ~w)~n', [RX, RY])
    ),

    % Test 3: Children positioned
    format('~nTest 3: Children positioned around root...~n'),
    member(position(a, AX, AY), Positions),
    member(position(b, BX, BY), Positions),
    DistA is sqrt((AX-500)^2 + (AY-500)^2),
    DistB is sqrt((BX-500)^2 + (BY-500)^2),
    (   DistA > 100, DistB > 100
    ->  format('  PASS: Children at radius ~2f, ~2f~n', [DistA, DistB])
    ;   format('  FAIL: Children too close to center~n')
    ),

    format('~n=== Tests Complete ===~n').

:- initialization((
    format('Radial layout module loaded~n', [])
), now).
