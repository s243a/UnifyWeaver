% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% force_directed.pl - Force-Directed Layout Algorithm for Mind Maps
%
% Implements a force-directed physics simulation for laying out mind maps.
% Nodes repel each other while edges act as springs pulling connected nodes together.
%
% Forces:
% - Repulsion: Nodes push apart (inverse square law)
% - Attraction: Connected nodes pull together (spring force)
% - Damping: Velocities decay over time
%
% Based on algorithms from generate_simplemind_map.py

:- module(mindmap_layout_force, [
    % Component interface
    type_info/1,
    validate_config/1,
    init_component/2,
    compute_layout/3,

    % Direct API
    force_directed_layout/4,        % force_directed_layout(+IR, +Options, -Positions, -Stats)
    force_directed_optimize/4,      % force_directed_optimize(+InitialPos, +Edges, +Options, -Positions)
    compute_forces/5                % compute_forces(+Positions, +Edges, +Options, -Forces, -MaxForce)
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
    name: force_directed,
    category: mindmap_layout,
    description: "Force-directed physics simulation layout",
    version: "1.0.0",
    parameters: [
        iterations - "Number of simulation iterations (default 300)",
        repulsion - "Repulsion force strength (default 100000)",
        attraction - "Attraction force to connected nodes (default 0.001)",
        min_distance - "Minimum distance between nodes (default 120)",
        damping - "Velocity damping factor 0-1 (default 0.8)",
        max_velocity - "Maximum velocity per step (default 150)",
        convergence_threshold - "Stop when max movement below this (default 0.5)",
        initial_layout - "Initial layout: radial, random, grid (default radial)"
    ]
}).

%% validate_config(+Config)
%
%  Validate layout configuration.
%
validate_config(Config) :-
    is_list(Config),
    (   member(iterations(I), Config)
    ->  integer(I), I > 0
    ;   true
    ),
    (   member(repulsion(R), Config)
    ->  number(R), R > 0
    ;   true
    ),
    (   member(damping(D), Config)
    ->  number(D), D >= 0, D =< 1
    ;   true
    ).

%% init_component(+Name, +Config)
%
%  Initialize the component (no-op for force layout).
%
init_component(_Name, _Config).

%% compute_layout(+Graph, +Options, -Positions)
%
%  Main entry point for component invocation.
%
compute_layout(graph(Nodes, Edges, RootId), Options, Positions) :-
    IR = ir(Nodes, Edges, RootId),
    force_directed_layout(IR, Options, Positions, _Stats).

% ============================================================================
% FORCE-DIRECTED LAYOUT ALGORITHM
% ============================================================================

%% force_directed_layout(+IR, +Options, -Positions, -Stats)
%
%  Compute force-directed layout.
%
%  @param IR        term - ir(Nodes, Edges, RootId)
%  @param Options   list - layout options
%  @param Positions list - list of position(Id, X, Y)
%  @param Stats     dict - layout statistics
%
force_directed_layout(IR, Options, Positions, Stats) :-
    extract_ir_data(IR, Nodes, Edges, RootId),

    % Get options
    option_or_default(iterations, Options, 300, MaxIterations),
    option_or_default(initial_layout, Options, radial, InitialLayout),

    % Generate initial positions
    generate_initial_positions(Nodes, RootId, InitialLayout, Options, InitialPositions),

    % Build edge list for physics
    build_edge_pairs(Edges, EdgePairs),

    % Run force-directed optimization
    force_directed_optimize(InitialPositions, EdgePairs, Options, Positions, IterationsUsed, Converged),

    % Statistics
    length(Nodes, NumNodes),
    Stats = stats{
        nodes: NumNodes,
        iterations_used: IterationsUsed,
        iterations_max: MaxIterations,
        converged: Converged
    }.

%% extract_ir_data(+IR, -Nodes, -Edges, -RootId)
extract_ir_data(ir(Nodes, Edges, RootId), Nodes, Edges, RootId) :- !.
extract_ir_data(mindmap_ir(_, graph(Nodes, Edges, RootId), _, _, _), Nodes, Edges, RootId) :- !.
extract_ir_data(graph(Nodes, Edges, RootId), Nodes, Edges, RootId).

%% generate_initial_positions(+Nodes, +RootId, +Layout, +Options, -Positions)
%
%  Generate initial node positions.
%
generate_initial_positions(Nodes, RootId, radial, Options, Positions) :-
    !,
    option_or_default(center_x, Options, 500, CX),
    option_or_default(center_y, Options, 500, CY),
    length(Nodes, N),
    (   N =< 1
    ->  Nodes = [node(Id, _)],
        Positions = [pos(Id, CX, CY, 0, 0)]
    ;   % Place root at center, others in circle
        Radius = 200,
        N1 is N - 1,
        AngleStep is 2 * pi / max(1, N1),
        generate_circle_positions(Nodes, RootId, CX, CY, Radius, 0, AngleStep, Positions)
    ).

generate_initial_positions(Nodes, _RootId, random, Options, Positions) :-
    !,
    option_or_default(center_x, Options, 500, CX),
    option_or_default(center_y, Options, 500, CY),
    option_or_default(spread, Options, 400, Spread),
    generate_random_positions(Nodes, CX, CY, Spread, Positions).

generate_initial_positions(Nodes, _RootId, grid, Options, Positions) :-
    !,
    option_or_default(center_x, Options, 500, CX),
    option_or_default(center_y, Options, 500, CY),
    option_or_default(grid_spacing, Options, 100, Spacing),
    generate_grid_positions(Nodes, CX, CY, Spacing, Positions).

generate_initial_positions(Nodes, RootId, _, Options, Positions) :-
    % Default to radial
    generate_initial_positions(Nodes, RootId, radial, Options, Positions).

generate_circle_positions([], _, _, _, _, _, _, []).
generate_circle_positions([node(Id, _) | Rest], RootId, CX, CY, R, Angle, Step, [pos(Id, X, Y, 0, 0) | RestPos]) :-
    (   Id == RootId
    ->  X = CX, Y = CY, NextAngle = Angle
    ;   X is CX + R * cos(Angle),
        Y is CY + R * sin(Angle),
        NextAngle is Angle + Step
    ),
    generate_circle_positions(Rest, RootId, CX, CY, R, NextAngle, Step, RestPos).

generate_random_positions([], _, _, _, []).
generate_random_positions([node(Id, _) | Rest], CX, CY, Spread, [pos(Id, X, Y, 0, 0) | RestPos]) :-
    random(R1), random(R2),
    X is CX + (R1 - 0.5) * Spread,
    Y is CY + (R2 - 0.5) * Spread,
    generate_random_positions(Rest, CX, CY, Spread, RestPos).

generate_grid_positions(Nodes, CX, CY, Spacing, Positions) :-
    length(Nodes, N),
    GridSize is ceiling(sqrt(N)),
    Offset is (GridSize - 1) * Spacing / 2,
    generate_grid_impl(Nodes, CX, CY, Spacing, Offset, 0, 0, GridSize, Positions).

generate_grid_impl([], _, _, _, _, _, _, _, []).
generate_grid_impl([node(Id, _) | Rest], CX, CY, Spacing, Offset, Row, Col, GridSize, [pos(Id, X, Y, 0, 0) | RestPos]) :-
    X is CX + Col * Spacing - Offset,
    Y is CY + Row * Spacing - Offset,
    NextCol is (Col + 1) mod GridSize,
    (   NextCol =:= 0
    ->  NextRow is Row + 1
    ;   NextRow = Row
    ),
    generate_grid_impl(Rest, CX, CY, Spacing, Offset, NextRow, NextCol, GridSize, RestPos).

%% build_edge_pairs(+Edges, -Pairs)
%
%  Extract From-To pairs from edges.
%
build_edge_pairs(Edges, Pairs) :-
    findall(From-To, member(edge(From, To, _), Edges), Pairs).

%% force_directed_optimize(+InitialPos, +EdgePairs, +Options, -Positions, -IterationsUsed, -Converged)
%
%  Run force-directed optimization.
%
force_directed_optimize(InitialPos, EdgePairs, Options, FinalPositions, IterationsUsed, Converged) :-
    option_or_default(iterations, Options, 300, MaxIterations),
    option_or_default(convergence_threshold, Options, 0.5, Threshold),

    % Build connected pairs set for fast lookup
    findall(P, (member(A-B, EdgePairs), (P = A-B ; P = B-A)), AllPairs),
    list_to_set(AllPairs, ConnectedSet),

    % Run simulation
    simulate_loop(InitialPos, EdgePairs, ConnectedSet, Options, MaxIterations, Threshold,
                  0, FinalPosWithVel, IterationsUsed, Converged),

    % Strip velocities from result
    findall(position(Id, X, Y),
            member(pos(Id, X, Y, _, _), FinalPosWithVel),
            FinalPositions).

simulate_loop(Positions, _, _, _, MaxIter, _, Iter, Positions, Iter, false) :-
    Iter >= MaxIter,
    !.
simulate_loop(Positions, EdgePairs, ConnectedSet, Options, MaxIter, Threshold, Iter, FinalPos, IterUsed, Converged) :-
    % Compute forces
    compute_forces(Positions, EdgePairs, ConnectedSet, Options, Forces),

    % Update positions
    update_positions(Positions, Forces, Options, NewPositions, MaxMovement),

    % Check convergence
    NextIter is Iter + 1,
    (   MaxMovement < Threshold
    ->  FinalPos = NewPositions,
        IterUsed = NextIter,
        Converged = true
    ;   simulate_loop(NewPositions, EdgePairs, ConnectedSet, Options, MaxIter, Threshold,
                      NextIter, FinalPos, IterUsed, Converged)
    ).

%% compute_forces(+Positions, +EdgePairs, +ConnectedSet, +Options, -Forces)
%
%  Compute forces for all nodes.
%
%  @param Positions    list - list of pos(Id, X, Y, VX, VY)
%  @param EdgePairs    list - list of From-To pairs
%  @param ConnectedSet set  - set of connected pairs for fast lookup
%  @param Options      list - force parameters
%  @param Forces       list - list of force(Id, FX, FY)
%
compute_forces(Positions, EdgePairs, ConnectedSet, Options, Forces) :-
    option_or_default(repulsion, Options, 100000, Repulsion),
    option_or_default(attraction, Options, 0.001, Attraction),
    option_or_default(min_distance, Options, 120, MinDist),

    % Compute repulsion between all pairs
    findall(force(Id, FX, FY),
            (member(pos(Id, _, _, _, _), Positions),
             compute_node_forces(Id, Positions, EdgePairs, ConnectedSet,
                                Repulsion, Attraction, MinDist, FX, FY)),
            Forces).

compute_node_forces(Id, Positions, EdgePairs, ConnectedSet, Repulsion, Attraction, MinDist, FX, FY) :-
    member(pos(Id, X, Y, _, _), Positions),

    % Repulsion from all other nodes
    findall(fx(RFX, RFY),
            (member(pos(OtherId, OX, OY, _, _), Positions),
             OtherId \== Id,
             compute_repulsion(X, Y, OX, OY, Id, OtherId, ConnectedSet, Repulsion, MinDist, RFX, RFY)),
            RepulsionForces),
    sum_forces(RepulsionForces, RepFX, RepFY),

    % Attraction to connected nodes (parents/children)
    findall(fx(AFX, AFY),
            (member(From-To, EdgePairs),
             (   (From == Id, member(pos(To, OX, OY, _, _), Positions))
             ;   (To == Id, member(pos(From, OX, OY, _, _), Positions))
             ),
             compute_attraction(X, Y, OX, OY, Attraction, AFX, AFY)),
            AttractionForces),
    sum_forces(AttractionForces, AttrFX, AttrFY),

    FX is RepFX + AttrFX,
    FY is RepFY + AttrFY.

compute_repulsion(X1, Y1, X2, Y2, Id1, Id2, ConnectedSet, Repulsion, MinDist, FX, FY) :-
    DX is X2 - X1,
    DY is Y2 - Y1,
    DistSq is DX*DX + DY*DY + 1,  % +1 to avoid division by zero
    Dist is sqrt(DistSq),

    % Only apply repulsion within range
    (   Dist < MinDist * 5
    ->  % Connected nodes repel less
        (   (member(Id1-Id2, ConnectedSet) ; member(Id2-Id1, ConnectedSet))
        ->  ConnectionFactor = 0.3
        ;   ConnectionFactor = 1.5
        ),
        % Inverse square repulsion
        ForceMag is (Repulsion * ConnectionFactor) / (DistSq + 100),
        % Extra boost when overlapping
        (   Dist < MinDist
        ->  ForceMag2 is ForceMag * 3
        ;   ForceMag2 = ForceMag
        ),
        (   Dist > 0.1
        ->  FX is -(DX / Dist) * ForceMag2,
            FY is -(DY / Dist) * ForceMag2
        ;   FX = 0, FY = 0
        )
    ;   FX = 0, FY = 0
    ).

compute_attraction(X1, Y1, X2, Y2, Attraction, FX, FY) :-
    DX is X2 - X1,
    DY is Y2 - Y1,
    Dist is sqrt(DX*DX + DY*DY),

    % Only attract when far apart
    IdealDist = 100,
    (   Dist > IdealDist
    ->  ForceMag is (Dist - IdealDist) * Attraction,
        (   Dist > IdealDist * 2
        ->  ForceMag2 is ForceMag * 2
        ;   ForceMag2 = ForceMag
        ),
        (   Dist > 0.1
        ->  FX is (DX / Dist) * ForceMag2,
            FY is (DY / Dist) * ForceMag2
        ;   FX = 0, FY = 0
        )
    ;   FX = 0, FY = 0
    ).

sum_forces([], 0, 0).
sum_forces([fx(FX, FY) | Rest], TotalFX, TotalFY) :-
    sum_forces(Rest, RestFX, RestFY),
    TotalFX is FX + RestFX,
    TotalFY is FY + RestFY.

%% update_positions(+Positions, +Forces, +Options, -NewPositions, -MaxMovement)
%
%  Update positions based on forces and velocities.
%
update_positions(Positions, Forces, Options, NewPositions, MaxMovement) :-
    option_or_default(damping, Options, 0.8, Damping),
    option_or_default(max_velocity, Options, 150, MaxVel),

    findall(Movement,
            (member(pos(Id, _, _, _, _), Positions),
             member(force(Id, _, _), Forces),
             update_single_position(Id, Positions, Forces, Damping, MaxVel, _, Movement)),
            Movements),

    findall(pos(Id, NX, NY, NVX, NVY),
            (member(pos(Id, _, _, _, _), Positions),
             update_single_position(Id, Positions, Forces, Damping, MaxVel,
                                   pos(Id, NX, NY, NVX, NVY), _)),
            NewPositions),

    (   Movements = []
    ->  MaxMovement = 0
    ;   max_list(Movements, MaxMovement)
    ).

update_single_position(Id, Positions, Forces, Damping, MaxVel, pos(Id, NX, NY, NVX, NVY), Movement) :-
    member(pos(Id, X, Y, VX, VY), Positions),
    member(force(Id, FX, FY), Forces),

    % Update velocity with damping
    VX1 is (VX + FX) * Damping,
    VY1 is (VY + FY) * Damping,

    % Limit velocity
    VelMag is sqrt(VX1*VX1 + VY1*VY1),
    (   VelMag > MaxVel
    ->  NVX is VX1 * MaxVel / VelMag,
        NVY is VY1 * MaxVel / VelMag
    ;   NVX = VX1, NVY = VY1
    ),

    % Update position
    NX is X + NVX,
    NY is Y + NVY,

    Movement is max(abs(NVX), abs(NVY)).

% ============================================================================
% UTILITIES
% ============================================================================

option_or_default(Key, Options, Default, Value) :-
    Term =.. [Key, Value],
    (   member(Term, Options)
    ->  true
    ;   Value = Default
    ).

pi is 3.14159265358979.

% ============================================================================
% TESTING
% ============================================================================

test_force_directed :-
    format('~n=== Force-Directed Layout Tests ===~n~n'),

    % Test data
    Nodes = [
        node(root, [label("Root")]),
        node(a, [label("A")]),
        node(b, [label("B")]),
        node(c, [label("C")])
    ],
    Edges = [
        edge(root, a, []),
        edge(root, b, []),
        edge(a, c, [])
    ],

    % Test 1: Compute layout
    format('Test 1: Compute force-directed layout...~n'),
    compute_layout(graph(Nodes, Edges, root), [iterations(50)], Positions),
    length(Positions, NumPos),
    (   NumPos =:= 4
    ->  format('  PASS: Generated ~w positions~n', [NumPos])
    ;   format('  FAIL: Expected 4 positions, got ~w~n', [NumPos])
    ),

    % Test 2: All nodes have valid positions
    format('~nTest 2: All nodes positioned...~n'),
    (   forall(member(node(Id, _), Nodes), member(position(Id, _, _), Positions))
    ->  format('  PASS: All nodes have positions~n')
    ;   format('  FAIL: Some nodes missing positions~n')
    ),

    % Test 3: No overlapping (approximate check)
    format('~nTest 3: Checking node separation...~n'),
    findall(Dist,
            (member(position(Id1, X1, Y1), Positions),
             member(position(Id2, X2, Y2), Positions),
             Id1 @< Id2,
             Dist is sqrt((X2-X1)^2 + (Y2-Y1)^2)),
            Distances),
    min_list(Distances, MinDist),
    (   MinDist > 50
    ->  format('  PASS: Minimum distance ~2f > 50~n', [MinDist])
    ;   format('  WARN: Minimum distance ~2f may indicate overlap~n', [MinDist])
    ),

    format('~n=== Tests Complete ===~n').

:- initialization((
    format('Force-directed layout module loaded~n', [])
), now).
