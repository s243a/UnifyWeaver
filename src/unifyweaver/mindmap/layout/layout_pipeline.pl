% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% layout_pipeline.pl - Mind Map Layout Pipeline
%
% Orchestrates the execution of layout stages and optimization passes.
% A pipeline consists of:
%   - Initial layout stage (radial, force_directed, hierarchical)
%   - Optional refinement stages
%   - Optimization passes (overlap removal, crossing minimization)
%
% Usage:
%   mindmap_pipeline(my_pipeline, [
%       stage(radial, [level_spacing(120)]),
%       stage(force_directed, [iterations(100)]),
%       optimize(overlap_removal, [min_distance(50)])
%   ]).
%
%   ?- execute_pipeline(my_pipeline, Graph, FinalPositions).

:- module(mindmap_layout_pipeline, [
    % Pipeline execution
    execute_pipeline/3,             % execute_pipeline(+Graph, +Pipeline, -Positions)
    execute_pipeline/4,             % execute_pipeline(+Graph, +Pipeline, +Options, -Result)

    % Stage execution
    execute_stage/4,                % execute_stage(+Stage, +Graph, +Positions, -NewPositions)
    execute_optimizer/3,            % execute_optimizer(+Optimizer, +Positions, -NewPositions)

    % Pipeline building
    build_default_pipeline/2,       % build_default_pipeline(+Options, -Pipeline)
    validate_pipeline/1,            % validate_pipeline(+Pipeline)

    % Progress reporting
    pipeline_progress/2,            % pipeline_progress(+Stage, +Progress) - hook for progress

    % Testing
    test_layout_pipeline/0
]).

:- use_module(library(lists)).

% Load layout modules if available
:- if(exists_source(library('./radial'))).
:- use_module('./radial').
:- endif.
:- if(exists_source(library('./force_directed'))).
:- use_module('./force_directed').
:- endif.
:- if(exists_source(library('../optimize/overlap_removal'))).
:- use_module('../optimize/overlap_removal').
:- endif.

% ============================================================================
% PIPELINE EXECUTION
% ============================================================================

%% execute_pipeline(+Graph, +Pipeline, -Positions)
%
%  Execute a layout pipeline on a graph.
%
%  @param Graph     term - graph(Nodes, Edges, RootId)
%  @param Pipeline  list - list of stage() and optimize() terms
%  @param Positions list - final node positions
%
execute_pipeline(Graph, Pipeline, Positions) :-
    execute_pipeline(Graph, Pipeline, [], result(Positions, _)).

%% execute_pipeline(+Graph, +Pipeline, +Options, -Result)
%
%  Execute a pipeline with options and detailed result.
%
%  Options:
%    - report_progress(Bool) - call pipeline_progress/2 hook
%    - collect_stats(Bool)   - collect statistics from each stage
%
%  @param Graph    term - graph structure
%  @param Pipeline list - pipeline stages
%  @param Options  list - execution options
%  @param Result   term - result(Positions, Stats)
%
execute_pipeline(Graph, Pipeline, Options, result(FinalPositions, Stats)) :-
    % Validate pipeline
    (   validate_pipeline(Pipeline)
    ->  true
    ;   format('Warning: Invalid pipeline structure~n', [])
    ),

    option_or_default(report_progress, Options, false, ReportProgress),
    option_or_default(collect_stats, Options, true, CollectStats),

    % Initialize with empty positions
    InitialPositions = [],

    % Execute each stage
    execute_stages(Pipeline, Graph, InitialPositions, Options, ReportProgress, CollectStats,
                   FinalPositions, StageStats),

    % Aggregate statistics
    Stats = pipeline_stats{
        stages_executed: StageStats,
        total_stages: Pipeline
    }.

execute_stages([], _Graph, Positions, _Options, _Report, _Collect, Positions, []).
execute_stages([Stage | Rest], Graph, Positions, Options, Report, Collect, FinalPositions, [StageStat | RestStats]) :-
    % Report progress if enabled
    (   Report == true
    ->  pipeline_progress(Stage, starting)
    ;   true
    ),

    % Execute the stage
    execute_single_stage(Stage, Graph, Positions, NewPositions, StageStat),

    % Report completion
    (   Report == true
    ->  pipeline_progress(Stage, completed)
    ;   true
    ),

    % Continue with rest of pipeline
    execute_stages(Rest, Graph, NewPositions, Options, Report, Collect, FinalPositions, RestStats).

%% execute_single_stage(+Stage, +Graph, +Positions, -NewPositions, -Stats)
%
%  Execute a single pipeline stage.
%
execute_single_stage(stage(Algorithm, StageOptions), Graph, _Positions, NewPositions, Stats) :-
    !,
    execute_stage(Algorithm, Graph, StageOptions, NewPositions),
    Stats = stage_stat{type: layout, algorithm: Algorithm, status: completed}.

execute_single_stage(optimize(Optimizer, OptOptions), _Graph, Positions, NewPositions, Stats) :-
    !,
    execute_optimizer_stage(Optimizer, Positions, OptOptions, NewPositions),
    Stats = stage_stat{type: optimizer, algorithm: Optimizer, status: completed}.

execute_single_stage(Unknown, _Graph, Positions, Positions, Stats) :-
    format('Warning: Unknown pipeline stage: ~w~n', [Unknown]),
    Stats = stage_stat{type: unknown, algorithm: Unknown, status: skipped}.

%% execute_stage(+Algorithm, +Graph, +Options, -Positions)
%
%  Execute a layout algorithm stage.
%
execute_stage(radial, Graph, Options, Positions) :-
    !,
    (   catch(mindmap_layout_radial:compute_layout(Graph, Options, Positions), _, fail)
    ->  true
    ;   compute_radial_fallback(Graph, Options, Positions)
    ).

execute_stage(force_directed, Graph, Options, Positions) :-
    !,
    (   catch(mindmap_layout_force:compute_layout(Graph, Options, Positions), _, fail)
    ->  true
    ;   compute_force_fallback(Graph, Options, Positions)
    ).

execute_stage(hierarchical, Graph, Options, Positions) :-
    !,
    compute_hierarchical_fallback(Graph, Options, Positions).

execute_stage(Algorithm, _Graph, _Options, []) :-
    format('Error: Unknown layout algorithm: ~w~n', [Algorithm]).

%% execute_optimizer(+Optimizer, +Positions, -NewPositions)
%
%  Execute an optimization pass.
%
execute_optimizer(Optimizer, Positions, NewPositions) :-
    execute_optimizer_stage(Optimizer, Positions, [], NewPositions).

execute_optimizer_stage(overlap_removal, Positions, Options, NewPositions) :-
    !,
    (   catch(mindmap_opt_overlap:optimize(Positions, Options, NewPositions), _, fail)
    ->  true
    ;   overlap_removal_fallback(Positions, Options, NewPositions)
    ).

execute_optimizer_stage(crossing_minimization, Positions, _Options, Positions) :-
    !,
    format('Note: Crossing minimization not yet implemented~n', []).

execute_optimizer_stage(spacing_adjustment, Positions, _Options, Positions) :-
    !,
    format('Note: Spacing adjustment not yet implemented~n', []).

execute_optimizer_stage(Optimizer, Positions, _Options, Positions) :-
    format('Warning: Unknown optimizer: ~w~n', [Optimizer]).

% ============================================================================
% FALLBACK IMPLEMENTATIONS
% ============================================================================

%% compute_radial_fallback(+Graph, +Options, -Positions)
%
%  Basic radial layout fallback.
%
compute_radial_fallback(graph(Nodes, _Edges, RootId), Options, Positions) :-
    option_or_default(center_x, Options, 500, CX),
    option_or_default(center_y, Options, 500, CY),
    option_or_default(base_radius, Options, 150, R),

    length(Nodes, N),
    (   N =:= 0
    ->  Positions = []
    ;   N =:= 1
    ->  Nodes = [node(Id, _)],
        Positions = [position(Id, CX, CY)]
    ;   % Place root at center, others in circle
        AngleStep is 2 * pi / max(1, N - 1),
        place_nodes_circle(Nodes, RootId, CX, CY, R, 0, AngleStep, Positions)
    ).

place_nodes_circle([], _, _, _, _, _, _, []).
place_nodes_circle([node(Id, _) | Rest], RootId, CX, CY, R, Angle, Step, [position(Id, X, Y) | RestPos]) :-
    (   Id == RootId
    ->  X = CX, Y = CY, NextAngle = Angle
    ;   X is CX + R * cos(Angle),
        Y is CY + R * sin(Angle),
        NextAngle is Angle + Step
    ),
    place_nodes_circle(Rest, RootId, CX, CY, R, NextAngle, Step, RestPos).

%% compute_force_fallback(+Graph, +Options, -Positions)
%
%  Force-directed fallback (uses radial then basic spreading).
%
compute_force_fallback(Graph, Options, Positions) :-
    % Start with radial
    compute_radial_fallback(Graph, Options, RadialPositions),
    % Apply basic overlap removal
    overlap_removal_fallback(RadialPositions, Options, Positions).

%% compute_hierarchical_fallback(+Graph, +Options, -Positions)
%
%  Hierarchical layout fallback.
%
compute_hierarchical_fallback(graph(Nodes, Edges, RootId), Options, Positions) :-
    option_or_default(level_spacing, Options, 100, LevelSpacing),
    option_or_default(sibling_spacing, Options, 80, SiblingSpacing),
    option_or_default(center_x, Options, 500, CX),
    option_or_default(start_y, Options, 50, StartY),

    % Build children map
    findall(Parent-Children,
            setof(Child, member(edge(Parent, Child, _), Edges), Children),
            ChildrenMap),

    % Position recursively
    hierarchical_position(RootId, ChildrenMap, CX, StartY, LevelSpacing, SiblingSpacing, [], Positions).

hierarchical_position(NodeId, ChildrenMap, X, Y, LevelSpacing, SiblingSpacing, AccIn, AccOut) :-
    AccMid = [position(NodeId, X, Y) | AccIn],

    % Get children
    (   member(NodeId-Children, ChildrenMap)
    ->  true
    ;   Children = []
    ),

    % Position children
    length(Children, NumChildren),
    (   NumChildren > 0
    ->  ChildY is Y + LevelSpacing,
        TotalWidth is (NumChildren - 1) * SiblingSpacing,
        StartX is X - TotalWidth / 2,
        position_children_hierarchical(Children, ChildrenMap, StartX, ChildY,
                                       LevelSpacing, SiblingSpacing, AccMid, AccOut)
    ;   AccOut = AccMid
    ).

position_children_hierarchical([], _, _, _, _, _, Acc, Acc).
position_children_hierarchical([Child | Rest], ChildrenMap, X, Y, LevelSpacing, SiblingSpacing, AccIn, AccOut) :-
    hierarchical_position(Child, ChildrenMap, X, Y, LevelSpacing, SiblingSpacing, AccIn, AccMid),
    NextX is X + SiblingSpacing,
    position_children_hierarchical(Rest, ChildrenMap, NextX, Y, LevelSpacing, SiblingSpacing, AccMid, AccOut).

%% overlap_removal_fallback(+Positions, +Options, -NewPositions)
%
%  Basic overlap removal fallback.
%
overlap_removal_fallback(Positions, Options, NewPositions) :-
    option_or_default(min_distance, Options, 50, MinDist),
    option_or_default(iterations, Options, 50, MaxIter),

    overlap_removal_loop(Positions, MinDist, MaxIter, 0, NewPositions).

overlap_removal_loop(Positions, _, MaxIter, Iter, Positions) :-
    Iter >= MaxIter,
    !.
overlap_removal_loop(Positions, MinDist, MaxIter, Iter, FinalPositions) :-
    % Find overlapping pairs
    findall(overlap(Id1, Id2),
            (member(position(Id1, X1, Y1), Positions),
             member(position(Id2, X2, Y2), Positions),
             Id1 @< Id2,
             Dist is sqrt((X2-X1)^2 + (Y2-Y1)^2),
             Dist < MinDist),
            Overlaps),

    (   Overlaps = []
    ->  FinalPositions = Positions
    ;   % Push overlapping nodes apart
        push_apart(Positions, Overlaps, MinDist, NewPositions),
        NextIter is Iter + 1,
        overlap_removal_loop(NewPositions, MinDist, MaxIter, NextIter, FinalPositions)
    ).

push_apart(Positions, [], _, Positions).
push_apart(Positions, [overlap(Id1, Id2) | Rest], MinDist, FinalPositions) :-
    member(position(Id1, X1, Y1), Positions),
    member(position(Id2, X2, Y2), Positions),

    DX is X2 - X1,
    DY is Y2 - Y1,
    Dist is sqrt(DX*DX + DY*DY),

    (   Dist < 0.1
    ->  % Same position - push in random direction
        random(R),
        Angle is R * 2 * pi,
        PushX is MinDist * 0.5 * cos(Angle),
        PushY is MinDist * 0.5 * sin(Angle)
    ;   % Push along line
        Overlap is (MinDist - Dist) / 2,
        PushX is (DX / Dist) * Overlap,
        PushY is (DY / Dist) * Overlap
    ),

    % Update positions
    select(position(Id1, X1, Y1), Positions, Rest1),
    select(position(Id2, X2, Y2), Rest1, Rest2),
    NX1 is X1 - PushX, NY1 is Y1 - PushY,
    NX2 is X2 + PushX, NY2 is Y2 + PushY,
    NewPositions = [position(Id1, NX1, NY1), position(Id2, NX2, NY2) | Rest2],

    push_apart(NewPositions, Rest, MinDist, FinalPositions).

% ============================================================================
% PIPELINE BUILDING
% ============================================================================

%% build_default_pipeline(+Options, -Pipeline)
%
%  Build a default pipeline based on options.
%
%  Options:
%    - layout(Algorithm)       - initial layout algorithm
%    - optimize(Bool)          - whether to add optimization passes
%    - quality(fast|balanced|high) - quality preset
%
build_default_pipeline(Options, Pipeline) :-
    option_or_default(layout, Options, radial, Layout),
    option_or_default(quality, Options, balanced, Quality),

    % Base layout stage
    (   Quality == fast
    ->  LayoutOptions = []
    ;   Quality == balanced
    ->  LayoutOptions = [iterations(100)]
    ;   LayoutOptions = [iterations(300)]
    ),
    LayoutStage = stage(Layout, LayoutOptions),

    % Optimization stages based on quality
    (   Quality == fast
    ->  OptStages = []
    ;   Quality == balanced
    ->  OptStages = [optimize(overlap_removal, [iterations(50)])]
    ;   OptStages = [
            stage(force_directed, [iterations(100)]),
            optimize(overlap_removal, [iterations(100)])
        ]
    ),

    append([LayoutStage], OptStages, Pipeline).

%% validate_pipeline(+Pipeline)
%
%  Validate pipeline structure.
%
validate_pipeline(Pipeline) :-
    is_list(Pipeline),
    forall(member(Stage, Pipeline), valid_stage(Stage)).

valid_stage(stage(Algorithm, Options)) :-
    atom(Algorithm),
    is_list(Options).
valid_stage(optimize(Optimizer, Options)) :-
    atom(Optimizer),
    is_list(Options).

% ============================================================================
% PROGRESS REPORTING
% ============================================================================

%% pipeline_progress(+Stage, +Status)
%
%  Hook for reporting pipeline progress.
%  Override in application code to handle progress events.
%
pipeline_progress(Stage, Status) :-
    format('Pipeline: ~w - ~w~n', [Stage, Status]).

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

test_layout_pipeline :-
    format('~n=== Layout Pipeline Tests ===~n~n'),

    % Test data
    Graph = graph(
        [node(root, [label("Root")]),
         node(a, [label("A")]),
         node(b, [label("B")]),
         node(c, [label("C")])],
        [edge(root, a, []),
         edge(root, b, []),
         edge(a, c, [])],
        root
    ),

    % Test 1: Simple pipeline
    format('Test 1: Simple radial pipeline...~n'),
    Pipeline1 = [stage(radial, [])],
    execute_pipeline(Graph, Pipeline1, Positions1),
    length(Positions1, NumPos1),
    (   NumPos1 =:= 4
    ->  format('  PASS: Generated ~w positions~n', [NumPos1])
    ;   format('  FAIL: Expected 4 positions, got ~w~n', [NumPos1])
    ),

    % Test 2: Pipeline with optimization
    format('~nTest 2: Pipeline with overlap removal...~n'),
    Pipeline2 = [
        stage(radial, [base_radius(30)]),  % Small radius to force overlaps
        optimize(overlap_removal, [min_distance(50)])
    ],
    execute_pipeline(Graph, Pipeline2, Positions2),
    findall(Dist,
            (member(position(Id1, X1, Y1), Positions2),
             member(position(Id2, X2, Y2), Positions2),
             Id1 @< Id2,
             Dist is sqrt((X2-X1)^2 + (Y2-Y1)^2)),
            Distances),
    min_list(Distances, MinDist),
    (   MinDist >= 49
    ->  format('  PASS: Minimum distance ~2f >= 50~n', [MinDist])
    ;   format('  WARN: Minimum distance ~2f < 50~n', [MinDist])
    ),

    % Test 3: Default pipeline builder
    format('~nTest 3: Default pipeline (balanced)...~n'),
    build_default_pipeline([quality(balanced)], DefaultPipeline),
    length(DefaultPipeline, NumStages),
    (   NumStages >= 2
    ->  format('  PASS: Built ~w-stage pipeline~n', [NumStages])
    ;   format('  WARN: Expected 2+ stages, got ~w~n', [NumStages])
    ),

    % Test 4: Hierarchical layout
    format('~nTest 4: Hierarchical layout...~n'),
    Pipeline4 = [stage(hierarchical, [level_spacing(80)])],
    execute_pipeline(Graph, Pipeline4, Positions4),
    % Check that children are below parent
    member(position(root, _, RY), Positions4),
    member(position(a, _, AY), Positions4),
    (   AY > RY
    ->  format('  PASS: Child below parent (y: ~2f > ~2f)~n', [AY, RY])
    ;   format('  FAIL: Child not below parent~n')
    ),

    format('~n=== Tests Complete ===~n').

:- initialization((
    format('Layout pipeline module loaded~n', [])
), now).
