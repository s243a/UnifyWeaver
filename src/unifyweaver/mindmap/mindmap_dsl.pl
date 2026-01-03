% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% mindmap_dsl.pl - Declarative Mind Map Layout DSL
%
% This module provides a declarative domain-specific language for specifying
% mind map structure, layout, and styling that compiles to multiple targets.
%
% Usage:
%   % Define nodes
%   mindmap_node(root, [label("Central Topic"), type(root)]).
%   mindmap_node(child1, [label("Branch A"), parent(root)]).
%
%   % Define layout
%   declare_mindmap_layout(my_map, force_directed, [
%       iterations(300),
%       min_distance(50)
%   ]).
%
%   % Generate output
%   ?- generate_mindmap_svg(my_map, SVG).
%   ?- generate_mindmap_positions(my_map, Positions).

:- module(mindmap_dsl, [
    % Node and edge declaration
    mindmap_node/2,                 % mindmap_node(+Id, +Properties)
    mindmap_edge/3,                 % mindmap_edge(+From, +To, +Properties)

    % Specification predicates
    mindmap_spec/2,                 % mindmap_spec(+Name, +Options)

    % Constraint and preference predicates
    mindmap_constraint/2,           % mindmap_constraint(+Type, +Options)
    mindmap_preference/2,           % mindmap_preference(+Type, +Options)

    % Layout predicates
    mindmap_layout/2,               % mindmap_layout(+Name, +Options)
    declare_mindmap_layout/3,       % declare_mindmap_layout(+Name, +Algorithm, +Options)

    % Pipeline predicates
    mindmap_pipeline/2,             % mindmap_pipeline(+Name, +Stages)

    % Style and theme predicates
    mindmap_style/2,                % mindmap_style(+Selector, +Properties)
    mindmap_theme/2,                % mindmap_theme(+Name, +Properties)

    % Management predicates
    declare_mindmap_node/2,         % declare_mindmap_node(+Id, +Properties)
    declare_mindmap_edge/3,         % declare_mindmap_edge(+From, +To, +Properties)
    declare_mindmap_spec/2,         % declare_mindmap_spec(+Name, +Options)
    clear_mindmap/0,                % clear_mindmap - clears all mindmap data
    clear_mindmap/1,                % clear_mindmap(+Name) - clears specific mindmap

    % Query predicates
    has_mindmap_layout/1,           % has_mindmap_layout(+Name)
    get_mindmap_nodes/2,            % get_mindmap_nodes(+SpecName, -Nodes)
    get_mindmap_edges/2,            % get_mindmap_edges(+SpecName, -Edges)
    get_mindmap_root/2,             % get_mindmap_root(+SpecName, -RootId)

    % Generation predicates (high-level)
    generate_mindmap_positions/2,   % generate_mindmap_positions(+Name, -Positions)
    generate_mindmap_svg/2,         % generate_mindmap_svg(+Name, -SVG)
    generate_mindmap_svg/3,         % generate_mindmap_svg(+Name, +Options, -SVG)
    compile_mindmap/3,              % compile_mindmap(+Name, +Target, -Code)

    % Testing
    test_mindmap_dsl/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic mindmap_node/2.          % mindmap_node(Id, Properties)
:- dynamic mindmap_edge/3.          % mindmap_edge(From, To, Properties)
:- dynamic mindmap_spec/2.          % mindmap_spec(Name, Options)
:- dynamic mindmap_constraint/2.    % mindmap_constraint(Type, Options)
:- dynamic mindmap_preference/2.    % mindmap_preference(Type, Options)
:- dynamic mindmap_layout/2.        % mindmap_layout(Name, Options)
:- dynamic mindmap_pipeline/2.      % mindmap_pipeline(Name, Stages)
:- dynamic mindmap_style/2.         % mindmap_style(Selector, Properties)
:- dynamic mindmap_theme/2.         % mindmap_theme(Name, Properties)

% Allow discontiguous definitions for user convenience
:- discontiguous mindmap_node/2.
:- discontiguous mindmap_edge/3.
:- discontiguous mindmap_spec/2.
:- discontiguous mindmap_constraint/2.
:- discontiguous mindmap_preference/2.
:- discontiguous mindmap_layout/2.
:- discontiguous mindmap_style/2.
:- discontiguous mindmap_theme/2.

% ============================================================================
% MANAGEMENT PREDICATES
% ============================================================================

%% declare_mindmap_node(+Id, +Properties)
%
%  Declare a mind map node with given properties.
%
%  Properties:
%    - label(Text)      - Display text for the node
%    - parent(ParentId) - Parent node (creates implicit edge)
%    - type(Type)       - Node type: root, branch, leaf, hub
%    - link(URL)        - Hyperlink associated with node
%    - style(Style)     - Node style reference
%    - cluster(Name)    - Cluster grouping
%    - importance(Level)- Importance: high, medium, low
%    - position(X, Y)   - Fixed position (optional)
%
%  @param Id         atom - unique node identifier
%  @param Properties list - node properties
%
declare_mindmap_node(Id, Properties) :-
    atom(Id),
    is_list(Properties),
    % Remove existing node if any
    retractall(mindmap_node(Id, _)),
    % Add node
    assertz(mindmap_node(Id, Properties)),
    % Create implicit edge from parent if specified
    (   member(parent(ParentId), Properties)
    ->  declare_mindmap_edge(ParentId, Id, [implicit(true)])
    ;   true
    ).

%% declare_mindmap_edge(+From, +To, +Properties)
%
%  Declare an edge between two nodes.
%
%  Properties:
%    - weight(W)      - Edge weight for layout
%    - style(Style)   - Edge style: straight, bezier, orthogonal
%    - label(Text)    - Edge label
%    - implicit(Bool) - Whether edge was auto-created from parent()
%
%  @param From       atom - source node id
%  @param To         atom - target node id
%  @param Properties list - edge properties
%
declare_mindmap_edge(From, To, Properties) :-
    atom(From),
    atom(To),
    is_list(Properties),
    % Remove existing edge if any (prevent duplicates)
    retractall(mindmap_edge(From, To, _)),
    assertz(mindmap_edge(From, To, Properties)).

%% declare_mindmap_spec(+Name, +Options)
%
%  Declare a complete mind map specification.
%
%  Options:
%    - nodes(List)          - List of node ids or inline node specs
%    - edges(List)          - List of From-To pairs
%    - layout(Algorithm)    - Layout algorithm to use
%    - theme(ThemeName)     - Theme to apply
%    - constraints(List)    - Layout constraints
%    - preferences(List)    - Layout preferences
%    - pipeline(List)       - Processing pipeline stages
%
%  @param Name    atom - specification name
%  @param Options list - specification options
%
declare_mindmap_spec(Name, Options) :-
    atom(Name),
    is_list(Options),
    retractall(mindmap_spec(Name, _)),
    assertz(mindmap_spec(Name, Options)),
    % Process inline node/edge definitions if present
    process_spec_nodes(Name, Options),
    process_spec_edges(Name, Options).

%% declare_mindmap_layout(+Name, +Algorithm, +Options)
%
%  Declare a layout configuration.
%
%  Algorithms:
%    - radial           - Radial/circular layout from center
%    - force_directed   - Force-directed physics simulation
%    - hierarchical     - Tree-like hierarchical layout
%    - spiral           - Spiral pattern from center
%
%  Options vary by algorithm but common ones include:
%    - iterations(N)    - Number of iterations for iterative layouts
%    - min_distance(D)  - Minimum distance between nodes
%    - center(X, Y)     - Center point
%    - theme(Theme)     - Visual theme
%
%  @param Name      atom - layout name
%  @param Algorithm atom - layout algorithm
%  @param Options   list - algorithm-specific options
%
declare_mindmap_layout(Name, Algorithm, Options) :-
    atom(Name),
    atom(Algorithm),
    is_list(Options),
    FullOptions = [algorithm(Algorithm) | Options],
    retractall(mindmap_layout(Name, _)),
    assertz(mindmap_layout(Name, FullOptions)).

%% clear_mindmap
%
%  Clear all mind map data from dynamic storage.
%
clear_mindmap :-
    retractall(mindmap_node(_, _)),
    retractall(mindmap_edge(_, _, _)),
    retractall(mindmap_spec(_, _)),
    retractall(mindmap_constraint(_, _)),
    retractall(mindmap_preference(_, _)),
    retractall(mindmap_layout(_, _)),
    retractall(mindmap_pipeline(_, _)),
    retractall(mindmap_style(_, _)),
    retractall(mindmap_theme(_, _)).

%% clear_mindmap(+Name)
%
%  Clear a specific mind map specification and its associated data.
%
clear_mindmap(Name) :-
    atom(Name),
    retractall(mindmap_spec(Name, _)),
    retractall(mindmap_layout(Name, _)),
    retractall(mindmap_pipeline(Name, _)).

% ============================================================================
% QUERY PREDICATES
% ============================================================================

%% has_mindmap_layout(+Name)
%
%  Check if a mind map layout exists.
%
has_mindmap_layout(Name) :-
    mindmap_layout(Name, _).

%% get_mindmap_nodes(+SpecName, -Nodes)
%
%  Get all nodes for a mind map specification.
%  Returns list of node(Id, Properties) terms.
%
get_mindmap_nodes(SpecName, Nodes) :-
    mindmap_spec(SpecName, Options),
    (   member(nodes(NodeIds), Options)
    ->  findall(node(Id, Props),
                (member(Id, NodeIds), mindmap_node(Id, Props)),
                Nodes)
    ;   % If no nodes specified, get all declared nodes
        findall(node(Id, Props), mindmap_node(Id, Props), Nodes)
    ).

%% get_mindmap_edges(+SpecName, -Edges)
%
%  Get all edges for a mind map specification.
%  Returns list of edge(From, To, Properties) terms.
%
get_mindmap_edges(SpecName, Edges) :-
    (   mindmap_spec(SpecName, Options),
        member(edges(EdgeList), Options)
    ->  findall(edge(F, T, Props),
                (member(F-T, EdgeList), mindmap_edge(F, T, Props)),
                Edges)
    ;   % If no edges specified, get all declared edges
        findall(edge(F, T, Props), mindmap_edge(F, T, Props), Edges)
    ).

%% get_mindmap_root(+SpecName, -RootId)
%
%  Find the root node of a mind map.
%  Root is defined as a node with type(root) or no parent.
%
get_mindmap_root(SpecName, RootId) :-
    get_mindmap_nodes(SpecName, Nodes),
    (   % First try to find explicit root
        member(node(RootId, Props), Nodes),
        member(type(root), Props)
    ->  true
    ;   % Otherwise find node with no incoming edges
        member(node(RootId, _), Nodes),
        \+ mindmap_edge(_, RootId, _)
    ).

% ============================================================================
% HELPER PREDICATES
% ============================================================================

%% process_spec_nodes(+Name, +Options)
%
%  Process inline node definitions from a spec.
%
process_spec_nodes(_Name, Options) :-
    (   member(nodes(NodeList), Options)
    ->  maplist(ensure_node_exists, NodeList)
    ;   true
    ).

ensure_node_exists(Id) :-
    atom(Id),
    !,
    (   mindmap_node(Id, _)
    ->  true
    ;   declare_mindmap_node(Id, [])
    ).
ensure_node_exists(node(Id, Props)) :-
    declare_mindmap_node(Id, Props).

%% process_spec_edges(+Name, +Options)
%
%  Process inline edge definitions from a spec.
%
process_spec_edges(_Name, Options) :-
    (   member(edges(EdgeList), Options)
    ->  maplist(ensure_edge_exists, EdgeList)
    ;   true
    ).

ensure_edge_exists(From-To) :-
    (   mindmap_edge(From, To, _)
    ->  true
    ;   declare_mindmap_edge(From, To, [])
    ).

% ============================================================================
% GENERATION PREDICATES (STUBS - implemented in render modules)
% ============================================================================

%% generate_mindmap_positions(+Name, -Positions)
%
%  Generate node positions using the configured layout algorithm.
%  Returns list of position(Id, X, Y) terms.
%
%  This is a high-level predicate that:
%  1. Retrieves the layout configuration
%  2. Builds the intermediate representation
%  3. Executes the layout algorithm
%  4. Returns positioned nodes
%
generate_mindmap_positions(Name, Positions) :-
    (   mindmap_layout(Name, LayoutOptions)
    ->  true
    ;   LayoutOptions = [algorithm(radial)]  % Default layout
    ),
    get_mindmap_nodes(Name, Nodes),
    get_mindmap_edges(Name, Edges),
    member(algorithm(Algorithm), LayoutOptions),
    % Call appropriate layout algorithm
    compute_layout(Algorithm, Nodes, Edges, LayoutOptions, Positions).

%% compute_layout(+Algorithm, +Nodes, +Edges, +Options, -Positions)
%
%  Dispatch to appropriate layout algorithm.
%  Default implementation provides basic radial layout.
%
compute_layout(radial, Nodes, _Edges, Options, Positions) :-
    !,
    compute_radial_layout(Nodes, Options, Positions).
compute_layout(force_directed, Nodes, Edges, Options, Positions) :-
    !,
    compute_force_directed_layout(Nodes, Edges, Options, Positions).
compute_layout(hierarchical, Nodes, Edges, Options, Positions) :-
    !,
    compute_hierarchical_layout(Nodes, Edges, Options, Positions).
compute_layout(Algorithm, _Nodes, _Edges, _Options, []) :-
    format('Warning: Unknown layout algorithm: ~w~n', [Algorithm]).

%% compute_radial_layout(+Nodes, +Options, -Positions)
%
%  Basic radial layout: root at center, children in circles.
%
compute_radial_layout(Nodes, Options, Positions) :-
    % Get center point
    (member(center(CX, CY), Options) -> true ; CX = 500, CY = 500),
    (member(base_radius(R), Options) -> true ; R = 150),
    length(Nodes, N),
    (   N =:= 0
    ->  Positions = []
    ;   N =:= 1
    ->  Nodes = [node(Id, _)],
        Positions = [position(Id, CX, CY)]
    ;   % Place first node at center, others in circle
        Nodes = [node(RootId, _) | RestNodes],
        length(RestNodes, NumRest),
        AngleStep is 2 * pi / max(1, NumRest),
        compute_circle_positions(RestNodes, CX, CY, R, 0, AngleStep, RestPositions),
        Positions = [position(RootId, CX, CY) | RestPositions]
    ).

compute_circle_positions([], _, _, _, _, _, []).
compute_circle_positions([node(Id, _) | Rest], CX, CY, R, Angle, Step, [position(Id, X, Y) | RestPos]) :-
    X is CX + R * cos(Angle),
    Y is CY + R * sin(Angle),
    NextAngle is Angle + Step,
    compute_circle_positions(Rest, CX, CY, R, NextAngle, Step, RestPos).

%% compute_force_directed_layout(+Nodes, +Edges, +Options, -Positions)
%
%  Force-directed layout stub - full implementation in layout/force_directed.pl
%
compute_force_directed_layout(Nodes, _Edges, Options, Positions) :-
    % Fallback to radial for now
    compute_radial_layout(Nodes, Options, Positions).

%% compute_hierarchical_layout(+Nodes, +Edges, +Options, -Positions)
%
%  Hierarchical tree layout stub - full implementation in layout/hierarchical.pl
%
compute_hierarchical_layout(Nodes, _Edges, Options, Positions) :-
    % Fallback to radial for now
    compute_radial_layout(Nodes, Options, Positions).

%% generate_mindmap_svg(+Name, -SVG)
%
%  Generate SVG output for a mind map.
%
generate_mindmap_svg(Name, SVG) :-
    generate_mindmap_svg(Name, [], SVG).

%% generate_mindmap_svg(+Name, +Options, -SVG)
%
%  Generate SVG output with options.
%
%  Options:
%    - width(W)        - SVG width
%    - height(H)       - SVG height
%    - background(C)   - Background color
%    - node_shape(S)   - Default node shape: ellipse, rectangle, diamond
%    - edge_style(S)   - Default edge style: straight, bezier
%
generate_mindmap_svg(Name, Options, SVG) :-
    generate_mindmap_positions(Name, Positions),
    get_mindmap_nodes(Name, Nodes),
    get_mindmap_edges(Name, Edges),
    (member(width(W), Options) -> true ; W = 1000),
    (member(height(H), Options) -> true ; H = 800),
    generate_svg_document(W, H, Nodes, Edges, Positions, Options, SVG).

%% generate_svg_document(+W, +H, +Nodes, +Edges, +Positions, +Options, -SVG)
%
%  Generate complete SVG document.
%
generate_svg_document(W, H, Nodes, Edges, Positions, Options, SVG) :-
    % Build position lookup
    findall(Id-pos(X,Y), member(position(Id, X, Y), Positions), PosLookup),
    % Generate edges first (underneath nodes)
    generate_svg_edges(Edges, PosLookup, EdgesSVG),
    % Generate nodes
    generate_svg_nodes(Nodes, PosLookup, Options, NodesSVG),
    % Combine into document
    (member(background(BG), Options) -> true ; BG = '#ffffff'),
    format(atom(SVG),
        '<svg xmlns="http://www.w3.org/2000/svg" width="~w" height="~w" viewBox="0 0 ~w ~w">~n  <rect width="100%" height="100%" fill="~w"/>~n  <g class="edges">~n~w  </g>~n  <g class="nodes">~n~w  </g>~n</svg>',
        [W, H, W, H, BG, EdgesSVG, NodesSVG]).

generate_svg_edges([], _, '').
generate_svg_edges([edge(From, To, _Props) | Rest], PosLookup, SVG) :-
    (   member(From-pos(X1, Y1), PosLookup),
        member(To-pos(X2, Y2), PosLookup)
    ->  format(atom(EdgeSVG), '    <line x1="~2f" y1="~2f" x2="~2f" y2="~2f" stroke="#666" stroke-width="2"/>~n', [X1, Y1, X2, Y2])
    ;   EdgeSVG = ''
    ),
    generate_svg_edges(Rest, PosLookup, RestSVG),
    atom_concat(EdgeSVG, RestSVG, SVG).

generate_svg_nodes([], _, _, '').
generate_svg_nodes([node(Id, Props) | Rest], PosLookup, Options, SVG) :-
    (   member(Id-pos(X, Y), PosLookup)
    ->  (member(label(Label), Props) -> true ; atom_string(Id, Label)),
        (member(node_radius(R), Options) -> true ; R = 40),
        format(atom(NodeSVG), '    <g class="node" data-id="~w">~n      <ellipse cx="~2f" cy="~2f" rx="~w" ry="~w" fill="#4a90d9" stroke="#2c5a8c" stroke-width="2"/>~n      <text x="~2f" y="~2f" text-anchor="middle" dominant-baseline="middle" fill="white" font-size="12">~w</text>~n    </g>~n', [Id, X, Y, R, R, X, Y, Label])
    ;   NodeSVG = ''
    ),
    generate_svg_nodes(Rest, PosLookup, Options, RestSVG),
    atom_concat(NodeSVG, RestSVG, SVG).

%% compile_mindmap(+Name, +Target, -Code)
%
%  Compile a mind map to target format.
%
%  Targets:
%    - svg              - Static SVG
%    - smmx             - .smmx format
%    - mm               - .mm format
%    - graph_interactive - Interactive graph component
%    - graphviz         - DOT format
%    - positions        - Just node positions
%
compile_mindmap(Name, svg, Code) :-
    !,
    generate_mindmap_svg(Name, Code).
compile_mindmap(Name, positions, Code) :-
    !,
    generate_mindmap_positions(Name, Code).
compile_mindmap(Name, Target, _Code) :-
    format('Error: Target ~w not yet implemented for mind map ~w~n', [Target, Name]),
    fail.

% ============================================================================
% TESTING
% ============================================================================

test_mindmap_dsl :-
    format('~n=== Mind Map DSL Tests ===~n~n'),

    % Setup
    clear_mindmap,

    % Test 1: Node declaration
    format('Test 1: Node declaration...~n'),
    declare_mindmap_node(root, [label("Central Topic"), type(root)]),
    declare_mindmap_node(child1, [label("Branch A"), parent(root)]),
    declare_mindmap_node(child2, [label("Branch B"), parent(root)]),
    (   mindmap_node(root, RootProps),
        member(label("Central Topic"), RootProps)
    ->  format('  PASS: Root node declared~n')
    ;   format('  FAIL: Root node not found~n')
    ),

    % Test 2: Implicit edge creation
    format('~nTest 2: Implicit edge creation...~n'),
    (   mindmap_edge(root, child1, _)
    ->  format('  PASS: Implicit edge created~n')
    ;   format('  FAIL: Implicit edge not found~n')
    ),

    % Test 3: Spec declaration
    format('~nTest 3: Spec declaration...~n'),
    declare_mindmap_spec(test_map, [
        nodes([root, child1, child2]),
        layout(radial)
    ]),
    (   mindmap_spec(test_map, _)
    ->  format('  PASS: Spec declared~n')
    ;   format('  FAIL: Spec not found~n')
    ),

    % Test 4: Layout declaration
    format('~nTest 4: Layout declaration...~n'),
    declare_mindmap_layout(test_map, force_directed, [
        iterations(100),
        min_distance(50)
    ]),
    (   has_mindmap_layout(test_map)
    ->  format('  PASS: Layout declared~n')
    ;   format('  FAIL: Layout not found~n')
    ),

    % Test 5: Position generation
    format('~nTest 5: Position generation...~n'),
    generate_mindmap_positions(test_map, Positions),
    length(Positions, NumPos),
    (   NumPos > 0
    ->  format('  PASS: Generated ~w positions~n', [NumPos])
    ;   format('  FAIL: No positions generated~n')
    ),

    % Test 6: SVG generation
    format('~nTest 6: SVG generation...~n'),
    generate_mindmap_svg(test_map, SVG),
    (   sub_atom(SVG, _, _, _, '<svg')
    ->  format('  PASS: SVG generated~n')
    ;   format('  FAIL: Invalid SVG~n')
    ),

    % Test 7: Get root node
    format('~nTest 7: Get root node...~n'),
    (   get_mindmap_root(test_map, RootId),
        RootId == root
    ->  format('  PASS: Root is ~w~n', [RootId])
    ;   format('  FAIL: Could not find root~n')
    ),

    % Cleanup
    clear_mindmap,

    format('~n=== Tests Complete ===~n').

% ============================================================================
% MODULE INITIALIZATION
% ============================================================================

:- initialization((
    format('Mind Map DSL module loaded~n', [])
), now).
