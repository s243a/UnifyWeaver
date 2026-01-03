% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% overlap_removal.pl - Overlap Removal Optimizer for Mind Maps
%
% Removes overlapping nodes by pushing them apart while preserving
% the overall layout structure.
%
% Algorithm:
% 1. Detect overlapping node pairs
% 2. Compute push vectors to separate them
% 3. Apply forces iteratively until no overlaps remain

:- module(mindmap_opt_overlap, [
    % Component interface
    type_info/1,
    validate_config/1,
    init_component/2,
    optimize/3,

    % Direct API
    remove_overlaps/4,              % remove_overlaps(+Positions, +Options, -NewPositions, -Stats)
    detect_overlaps/3,              % detect_overlaps(+Positions, +MinDist, -Overlaps)
    compute_separation_vector/5     % compute_separation_vector(+P1, +P2, +MinDist, -DX, -DY)
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
    name: overlap_removal,
    category: mindmap_optimizer,
    description: "Remove overlapping nodes by pushing them apart",
    version: "1.0.0",
    parameters: [
        min_distance - "Minimum distance between node centers (default 50)",
        iterations - "Maximum iterations (default 100)",
        step_size - "How much to move per iteration (default 0.5)",
        preserve_root - "Keep root node fixed (default true)"
    ]
}).

%% validate_config(+Config)
%
%  Validate optimizer configuration.
%
validate_config(Config) :-
    is_list(Config),
    (   member(min_distance(D), Config)
    ->  number(D), D > 0
    ;   true
    ),
    (   member(iterations(I), Config)
    ->  integer(I), I > 0
    ;   true
    ).

%% init_component(+Name, +Config)
%
%  Initialize the component.
%
init_component(_Name, _Config).

%% optimize(+Positions, +Options, -OptimizedPositions)
%
%  Main entry point for component invocation.
%
optimize(Positions, Options, OptimizedPositions) :-
    remove_overlaps(Positions, Options, OptimizedPositions, _Stats).

% ============================================================================
% OVERLAP REMOVAL ALGORITHM
% ============================================================================

%% remove_overlaps(+Positions, +Options, -NewPositions, -Stats)
%
%  Remove overlaps from node positions.
%
%  @param Positions    list - list of position(Id, X, Y)
%  @param Options      list - optimizer options
%  @param NewPositions list - positions with overlaps removed
%  @param Stats        dict - optimization statistics
%
remove_overlaps(Positions, Options, NewPositions, Stats) :-
    option_or_default(min_distance, Options, 50, MinDist),
    option_or_default(iterations, Options, 100, MaxIterations),
    option_or_default(step_size, Options, 0.5, StepSize),
    option_or_default(preserve_root, Options, true, PreserveRoot),

    % Find root if we need to preserve it
    (   PreserveRoot == true,
        Positions = [position(RootId, _, _) | _]
    ->  FixedNodes = [RootId]
    ;   FixedNodes = []
    ),

    % Run iterative overlap removal
    overlap_removal_loop(Positions, MinDist, StepSize, FixedNodes, MaxIterations, 0,
                         NewPositions, IterationsUsed, OverlapsRemoved),

    Stats = stats{
        iterations_used: IterationsUsed,
        iterations_max: MaxIterations,
        overlaps_removed: OverlapsRemoved,
        min_distance: MinDist
    }.

overlap_removal_loop(Positions, _, _, _, MaxIter, Iter, Positions, Iter, 0) :-
    Iter >= MaxIter,
    !.
overlap_removal_loop(Positions, MinDist, StepSize, FixedNodes, MaxIter, Iter, FinalPositions, IterUsed, TotalRemoved) :-
    % Detect current overlaps
    detect_overlaps(Positions, MinDist, Overlaps),

    (   Overlaps = []
    ->  % No overlaps, we're done
        FinalPositions = Positions,
        IterUsed = Iter,
        TotalRemoved = 0
    ;   % Apply separation
        length(Overlaps, NumOverlaps),
        apply_separations(Positions, Overlaps, MinDist, StepSize, FixedNodes, NewPositions),
        NextIter is Iter + 1,
        overlap_removal_loop(NewPositions, MinDist, StepSize, FixedNodes, MaxIter, NextIter,
                            FinalPositions, IterUsed, RestRemoved),
        TotalRemoved is NumOverlaps + RestRemoved
    ).

%% detect_overlaps(+Positions, +MinDist, -Overlaps)
%
%  Find all overlapping node pairs.
%
%  @param Positions list - list of position(Id, X, Y)
%  @param MinDist   number - minimum distance threshold
%  @param Overlaps  list - list of overlap(Id1, Id2, Dist, DX, DY)
%
detect_overlaps(Positions, MinDist, Overlaps) :-
    findall(overlap(Id1, Id2, Dist, DX, DY),
            (member(position(Id1, X1, Y1), Positions),
             member(position(Id2, X2, Y2), Positions),
             Id1 @< Id2,  % Avoid duplicates
             DX is X2 - X1,
             DY is Y2 - Y1,
             Dist is sqrt(DX*DX + DY*DY),
             Dist < MinDist),
            Overlaps).

%% compute_separation_vector(+X1, +Y1, +X2, +Y2, +MinDist, -SepX, -SepY)
%
%  Compute vector to separate two overlapping nodes.
%
compute_separation_vector(X1, Y1, X2, Y2, MinDist, SepX, SepY) :-
    DX is X2 - X1,
    DY is Y2 - Y1,
    Dist is sqrt(DX*DX + DY*DY),

    (   Dist < 0.1
    ->  % Nodes at same position - push in random direction
        random(R),
        Angle is R * 2 * pi,
        SepX is MinDist * cos(Angle),
        SepY is MinDist * sin(Angle)
    ;   % Push apart along connecting line
        Overlap is MinDist - Dist,
        SepX is (DX / Dist) * Overlap,
        SepY is (DY / Dist) * Overlap
    ).

%% apply_separations(+Positions, +Overlaps, +MinDist, +StepSize, +FixedNodes, -NewPositions)
%
%  Apply separation forces for all overlaps.
%
apply_separations(Positions, Overlaps, MinDist, StepSize, FixedNodes, NewPositions) :-
    % Accumulate displacement for each node
    findall(Id-disp(0, 0), member(position(Id, _, _), Positions), InitDisp),
    accumulate_displacements(Overlaps, MinDist, StepSize, InitDisp, FinalDisp),

    % Apply displacements
    findall(position(Id, NX, NY),
            (member(position(Id, X, Y), Positions),
             (   member(Id, FixedNodes)
             ->  NX = X, NY = Y  % Don't move fixed nodes
             ;   member(Id-disp(DispX, DispY), FinalDisp),
                 NX is X + DispX,
                 NY is Y + DispY
             )),
            NewPositions).

accumulate_displacements([], _, _, Disp, Disp).
accumulate_displacements([overlap(Id1, Id2, Dist, DX, DY) | Rest], MinDist, StepSize, DispIn, DispOut) :-
    % Compute separation amount
    (   Dist < 0.1
    ->  % At same position - use random direction
        random(R),
        Angle is R * 2 * pi,
        SepX is MinDist * StepSize * cos(Angle),
        SepY is MinDist * StepSize * sin(Angle)
    ;   Overlap is MinDist - Dist,
        SepX is (DX / Dist) * Overlap * StepSize,
        SepY is (DY / Dist) * Overlap * StepSize
    ),

    % Each node moves half the separation
    HalfSepX is SepX / 2,
    HalfSepY is SepY / 2,

    % Update displacements for both nodes
    update_displacement(Id1, -HalfSepX, -HalfSepY, DispIn, Disp1),
    update_displacement(Id2, HalfSepX, HalfSepY, Disp1, Disp2),

    accumulate_displacements(Rest, MinDist, StepSize, Disp2, DispOut).

update_displacement(Id, AddX, AddY, DispIn, DispOut) :-
    (   select(Id-disp(OldX, OldY), DispIn, Rest)
    ->  NewX is OldX + AddX,
        NewY is OldY + AddY,
        DispOut = [Id-disp(NewX, NewY) | Rest]
    ;   DispOut = [Id-disp(AddX, AddY) | DispIn]
    ).

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

test_overlap_removal :-
    format('~n=== Overlap Removal Tests ===~n~n'),

    % Test 1: Detect overlaps
    format('Test 1: Detect overlaps...~n'),
    TestPositions = [
        position(a, 100, 100),
        position(b, 120, 100),  % 20 units from a
        position(c, 200, 200),  % Far from others
        position(d, 105, 105)   % ~7 units from a
    ],
    detect_overlaps(TestPositions, 50, Overlaps),
    length(Overlaps, NumOverlaps),
    (   NumOverlaps > 0
    ->  format('  PASS: Detected ~w overlaps~n', [NumOverlaps])
    ;   format('  FAIL: No overlaps detected~n')
    ),

    % Test 2: Remove overlaps
    format('~nTest 2: Remove overlaps...~n'),
    remove_overlaps(TestPositions, [min_distance(50)], NewPositions, Stats),
    detect_overlaps(NewPositions, 50, RemainingOverlaps),
    (   RemainingOverlaps = []
    ->  format('  PASS: All overlaps removed~n')
    ;   length(RemainingOverlaps, Remaining),
        format('  FAIL: ~w overlaps remain~n', [Remaining])
    ),
    format('  Stats: ~w~n', [Stats]),

    % Test 3: Minimum distances maintained
    format('~nTest 3: Minimum distances...~n'),
    findall(Dist,
            (member(position(Id1, X1, Y1), NewPositions),
             member(position(Id2, X2, Y2), NewPositions),
             Id1 @< Id2,
             Dist is sqrt((X2-X1)^2 + (Y2-Y1)^2)),
            Distances),
    min_list(Distances, MinActual),
    (   MinActual >= 49.9  % Allow small floating point error
    ->  format('  PASS: Minimum distance ~2f >= 50~n', [MinActual])
    ;   format('  FAIL: Minimum distance ~2f < 50~n', [MinActual])
    ),

    format('~n=== Tests Complete ===~n').

:- initialization((
    format('Overlap removal optimizer loaded~n', [])
), now).
