%% smoothing_policy.pl
%% Declarative policy for selecting smoothing techniques based on tree structure.
%%
%% The FFT approach creates a natural tree (MST) over clusters. This module
%% defines rules for when to apply which smoothing technique at each node.
%%
%% Usage:
%%   ?- smoothing_plan(RootNode, Plan).
%%   Plan = [apply(fft, root), apply(basis, segment_1), ...]

:- module(smoothing_policy, [
    smoothing_plan/2,
    recommended_technique/2,
    refinement_needed/1,
    sufficient_data/2,
    load_tree_from_json/1,
    export_plan_to_json/2,
    recursive_smoothing_plan/2
]).

%% Load JSON support
:- use_module(library(http/json)).

%% =============================================================================
%% Node Properties (populated from Python via bridge)
%% =============================================================================

%% node(Id, ClusterCount, TotalPairs, Depth, AvgPairsPerCluster)
:- dynamic node/5.

%% parent(ParentId, ChildId)
:- dynamic parent/2.

%% similarity_score(NodeId, Score) - how coherent the node's clusters are
:- dynamic similarity_score/2.

%% =============================================================================
%% Technique Characteristics
%% =============================================================================

%% technique(Name, MinClusters, MaxClusters, Complexity, Accuracy)
technique(fft,          10,  100000, low,    high).
technique(basis_k4,     5,   500,    medium, medium).
technique(basis_k8,     10,  200,    high,   high).
technique(basis_k16,    20,  100,    very_high, high).
technique(hierarchical, 20,  1000,   high,   medium).
technique(baseline,     1,   100000, very_low, medium).

%% complexity_order(Complexity, NumericOrder)
complexity_order(very_low, 1).
complexity_order(low, 2).
complexity_order(medium, 3).
complexity_order(high, 4).
complexity_order(very_high, 5).

%% =============================================================================
%% Core Decision Rules
%% =============================================================================

%% sufficient_data(NodeId, Technique)
%% True if node has enough data for the technique to be meaningful
sufficient_data(NodeId, Technique) :-
    node(NodeId, ClusterCount, _TotalPairs, _, AvgPairs),
    technique(Technique, MinC, MaxC, _, _),
    ClusterCount >= MinC,
    ClusterCount =< MaxC,
    AvgPairs >= 1.  % At least 1 pair per cluster on average

%% refinement_needed(NodeId)
%% True if this node would benefit from further refinement.
%%
%% Key insight: Refine based on cluster DISTINGUISHABILITY, not just size.
%% If clusters within a segment are already well-separated after projection,
%% no need to refine further. Only refine where clusters are still confusable.
%%
%% The similarity_score represents intra-segment cluster separation:
%% - High score (>0.7) = clusters similar/confusable → may need refinement
%% - Low score (<0.7) = clusters distinct → stop refinement
refinement_needed(NodeId) :-
    node(NodeId, ClusterCount, _, Depth, _),
    ClusterCount > 10,      % Enough clusters to potentially confuse
    Depth < 4,              % Reasonable depth limit
    similarity_score(NodeId, Score),
    Score > 0.7.            % Clusters still too similar → refine

%% distinguish_threshold(Threshold)
%% Minimum separation score to consider clusters distinguishable
distinguish_threshold(0.3).

%% clusters_distinguishable(NodeId)
%% True if clusters within this node are well-separated after projection
clusters_distinguishable(NodeId) :-
    similarity_score(NodeId, Score),
    distinguish_threshold(Threshold),
    Score < Threshold.  % Lower similarity = more distinguishable

%% =============================================================================
%% Technique Selection Rules (with depth-based transitions)
%% =============================================================================

%% Thresholds for technique selection
%% FFT is efficient at scale, but basis becomes preferable at lower cluster counts
fft_threshold(30).          % Minimum clusters for FFT to be worthwhile
basis_sweet_spot(10, 50).   % Range where basis methods excel

%% recommended_technique(NodeId, Technique)
%% Recommend a technique based on node properties AND depth.
%%
%% Key insight: FFT's O(N log N) advantage diminishes at lower scales.
%% At some point transitioning to basis methods gives better accuracy
%% without the FFT overhead of MST construction and DFS ordering.

%% Rule 1: Large clusters (>30) at shallow depths -> FFT
%% FFT is the best choice here: fast and effective at scale
recommended_technique(NodeId, fft) :-
    node(NodeId, ClusterCount, _, Depth, _),
    fft_threshold(Threshold),
    ClusterCount >= Threshold,
    Depth < 3.  % FFT makes sense at top levels

%% Rule 2: Medium clusters at any depth -> consider basis methods
%% The "sweet spot" where basis methods work well
recommended_technique(NodeId, basis_k8) :-
    node(NodeId, ClusterCount, _, Depth, AvgPairs),
    basis_sweet_spot(MinC, MaxC),
    ClusterCount >= MinC,
    ClusterCount =< MaxC,
    Depth >= 1,             % Not at root (FFT handles that)
    AvgPairs >= 2.          % Need decent data per cluster

%% Rule 3: Deeper levels with moderate clusters -> basis_k4
%% Simpler basis is faster and still effective
recommended_technique(NodeId, basis_k4) :-
    node(NodeId, ClusterCount, _, Depth, AvgPairs),
    ClusterCount >= 5,
    ClusterCount < 20,
    Depth >= 2,             % Deep enough that FFT overhead isn't worth it
    AvgPairs >= 2.

%% Rule 4: Very small clusters -> baseline (no smoothing)
%% Not enough data to benefit from smoothing
recommended_technique(NodeId, baseline) :-
    node(NodeId, ClusterCount, _, _, _),
    ClusterCount < 5.

%% Rule 5: Large clusters at deep levels -> still FFT but note the tradeoff
%% Even at depth, if cluster count is high, FFT's efficiency helps
recommended_technique(NodeId, fft) :-
    node(NodeId, ClusterCount, _, Depth, _),
    ClusterCount >= 50,
    Depth >= 3.  % Deep but large enough to justify FFT

%% Rule 6: Fallback for edge cases
recommended_technique(NodeId, basis_k4) :-
    node(NodeId, ClusterCount, _, _, _),
    ClusterCount >= 5,
    \+ recommended_technique(NodeId, fft),
    \+ recommended_technique(NodeId, basis_k8).

%% =============================================================================
%% Recursive Refinement Rules
%% =============================================================================

%% can_subdivide(NodeId)
%% True if node is large enough to create meaningful sub-segments
can_subdivide(NodeId) :-
    node(NodeId, ClusterCount, _, Depth, _),
    fft_threshold(Threshold),
    ClusterCount >= Threshold,
    Depth < 5.  % Limit recursion depth

%% should_subdivide(NodeId)
%% True if subdivision would likely improve results
should_subdivide(NodeId) :-
    can_subdivide(NodeId),
    similarity_score(NodeId, Score),
    Score < 0.8.  % Not already highly coherent

%% max_recursion_depth(Depth)
max_recursion_depth(4).

%% =============================================================================
%% Recursive Plan Generation
%% =============================================================================

%% recursive_smoothing_plan(NodeId, Plan)
%% Generate plan with recursive FFT on large subtrees
recursive_smoothing_plan(NodeId, Plan) :-
    recursive_plan_acc(NodeId, [], PlanRev),
    reverse(PlanRev, Plan).

recursive_plan_acc(NodeId, Acc, Plan) :-
    node(NodeId, _, _, Depth, _),
    max_recursion_depth(MaxDepth),
    Depth =< MaxDepth,
    recommended_technique(NodeId, Technique),
    Action = apply(Technique, NodeId),

    % Check if we should recurse into children
    (   should_subdivide(NodeId)
    ->  % Get children and recursively plan
        findall(ChildId, parent(NodeId, ChildId), Children),
        foldl(recursive_plan_child, Children, [Action|Acc], Plan)
    ;   % No subdivision, just this action
        Plan = [Action|Acc]
    ).

%% Base case: node too deep, just apply recommended technique
recursive_plan_acc(NodeId, Acc, [Action|Acc]) :-
    node(NodeId, _, _, Depth, _),
    max_recursion_depth(MaxDepth),
    Depth > MaxDepth,
    recommended_technique(NodeId, Technique),
    Action = apply(Technique, NodeId).

recursive_plan_child(ChildId, AccIn, AccOut) :-
    recursive_plan_acc(ChildId, AccIn, AccOut).

%% =============================================================================
%% Plan Analysis
%% =============================================================================

%% count_techniques(Plan, TechniqueCounts)
%% Count how many times each technique appears
count_techniques(Plan, Counts) :-
    findall(Tech, member(apply(Tech, _), Plan), Techs),
    msort(Techs, SortedTechs),
    count_occurrences(SortedTechs, Counts).

count_occurrences([], []).
count_occurrences([H|T], [H-Count|Rest]) :-
    count_same(H, [H|T], Count, Remaining),
    count_occurrences(Remaining, Rest).

count_same(_, [], 0, []).
count_same(X, [X|T], N, Rest) :-
    count_same(X, T, N1, Rest),
    N is N1 + 1.
count_same(X, [Y|T], 0, [Y|T]) :-
    X \= Y.

%% plan_summary(Plan, Summary)
%% Get a summary of the plan
plan_summary(Plan, Summary) :-
    length(Plan, NumActions),
    count_techniques(Plan, TechCounts),
    total_plan_cost(Plan, TotalCost),
    Summary = summary{
        num_actions: NumActions,
        technique_counts: TechCounts,
        estimated_cost_ms: TotalCost
    }.

%% =============================================================================
%% Plan Generation
%% =============================================================================

%% smoothing_plan(NodeId, Plan)
%% Generate a complete smoothing plan for a subtree
smoothing_plan(NodeId, Plan) :-
    smoothing_plan_acc(NodeId, [], Plan).

smoothing_plan_acc(NodeId, Acc, Plan) :-
    % Get technique for this node
    recommended_technique(NodeId, Technique),

    % Build action for this node
    Action = apply(Technique, NodeId),

    % Check if refinement needed
    (   refinement_needed(NodeId)
    ->  % Recursively plan children
        findall(ChildId, parent(NodeId, ChildId), Children),
        foldl(smoothing_plan_child, Children, [Action|Acc], Plan)
    ;   % No refinement, just this action
        Plan = [Action|Acc]
    ).

smoothing_plan_child(ChildId, AccIn, AccOut) :-
    smoothing_plan_acc(ChildId, AccIn, AccOut).

%% =============================================================================
%% Cost Estimation
%% =============================================================================

%% estimated_cost(NodeId, Technique, CostMs)
%% Estimate training cost in milliseconds
estimated_cost(NodeId, fft, Cost) :-
    node(NodeId, C, _, _, _),
    Cost is C * 0.4.  % ~0.4ms per cluster for FFT

estimated_cost(NodeId, basis_k4, Cost) :-
    node(NodeId, C, _, _, _),
    Cost is C * 10.   % ~10ms per cluster for basis K=4

estimated_cost(NodeId, basis_k8, Cost) :-
    node(NodeId, C, _, _, _),
    Cost is C * 15.   % ~15ms per cluster for basis K=8

estimated_cost(NodeId, baseline, Cost) :-
    node(NodeId, C, _, _, _),
    Cost is C * 0.02. % ~0.02ms per cluster for baseline

%% total_plan_cost(Plan, TotalCost)
total_plan_cost([], 0).
total_plan_cost([apply(Tech, NodeId)|Rest], Total) :-
    estimated_cost(NodeId, Tech, Cost),
    total_plan_cost(Rest, RestCost),
    Total is Cost + RestCost.

%% =============================================================================
%% Optimization: Find best plan under budget
%% =============================================================================

%% optimized_plan(NodeId, MaxCostMs, Plan)
%% Find a plan that fits within budget while maximizing accuracy
optimized_plan(NodeId, MaxCost, Plan) :-
    findall(P-C, (
        smoothing_plan(NodeId, P),
        total_plan_cost(P, C),
        C =< MaxCost
    ), PlansWithCosts),

    % Sort by cost descending (higher cost = more thorough)
    sort(2, @>=, PlansWithCosts, Sorted),

    % Take the most thorough plan within budget
    Sorted = [Plan-_|_].

%% =============================================================================
%% Example Queries
%% =============================================================================

%% Example usage (after loading node data from Python):
%%
%% % What technique for the root?
%% ?- recommended_technique(root, T).
%% T = fft.
%%
%% % Generate full plan
%% ?- smoothing_plan(root, Plan).
%% Plan = [apply(fft, root), apply(basis_k8, seg1), apply(basis_k4, seg2), ...].
%%
%% % Estimate cost
%% ?- smoothing_plan(root, Plan), total_plan_cost(Plan, Cost).
%% Cost = 1523.5.
%%
%% % Get plan under 500ms budget
%% ?- optimized_plan(root, 500, Plan).
%% Plan = [apply(fft, root), apply(baseline, seg1), ...].

%% =============================================================================
%% Federation Integration: Materialized Similarity Paths
%% =============================================================================

%% The MST+DFS ordering creates a materialized path through cluster space.
%% This path can augment federation with similarity-based links:
%%
%% Traditional federation links:
%%   - By topic/domain
%%   - By data source
%%   - By access pattern
%%
%% Similarity path links (from FFT ordering):
%%   - prev_similar(ClusterA, ClusterB)  - adjacent in path
%%   - same_segment(ClusterA, ClusterB)  - in same coherent region
%%   - segment_boundary(ClusterA, ClusterB) - bridge between segments

%% path_position(ClusterId, Position)
:- dynamic path_position/2.

%% segment_membership(ClusterId, SegmentId)
:- dynamic segment_membership/2.

%% load_path_from_ordering(OrderingList)
%% Load the FFT ordering as path positions
load_path_from_ordering(Ordering) :-
    retractall(path_position(_, _)),
    load_positions(Ordering, 0).

load_positions([], _).
load_positions([ClusterId|Rest], Pos) :-
    assertz(path_position(ClusterId, Pos)),
    NextPos is Pos + 1,
    load_positions(Rest, NextPos).

%% prev_similar(ClusterA, ClusterB)
%% True if A comes immediately before B in similarity path
prev_similar(A, B) :-
    path_position(A, PosA),
    path_position(B, PosB),
    PosB is PosA + 1.

%% next_similar(ClusterA, ClusterB)
%% True if B comes immediately after A in similarity path
next_similar(A, B) :-
    prev_similar(A, B).

%% nearby_similar(ClusterA, ClusterB, Distance)
%% True if A and B are within Distance steps in path
nearby_similar(A, B, MaxDist) :-
    path_position(A, PosA),
    path_position(B, PosB),
    A \= B,
    Dist is abs(PosA - PosB),
    Dist =< MaxDist.

%% same_segment(ClusterA, ClusterB)
%% True if A and B belong to the same segment
same_segment(A, B) :-
    segment_membership(A, Seg),
    segment_membership(B, Seg),
    A \= B.

%% segment_boundary(ClusterA, ClusterB, SegA, SegB)
%% True if A and B are adjacent in path but in different segments
segment_boundary(A, B, SegA, SegB) :-
    (prev_similar(A, B) ; prev_similar(B, A)),
    segment_membership(A, SegA),
    segment_membership(B, SegB),
    SegA \= SegB.

%% federation_links(ClusterId, Links)
%% Get all federation-relevant links for a cluster
federation_links(ClusterId, Links) :-
    findall(
        link(Type, Target),
        cluster_link(ClusterId, Type, Target),
        Links
    ).

cluster_link(C, prev_similar, Target) :-
    prev_similar(Target, C).
cluster_link(C, next_similar, Target) :-
    next_similar(C, Target).
cluster_link(C, same_segment, Target) :-
    same_segment(C, Target).
cluster_link(C, segment_bridge, Target) :-
    segment_boundary(C, Target, _, _).

%% export_federation_graph(JsonPath)
%% Export similarity graph for federation use
export_federation_graph(JsonPath) :-
    % Collect all nodes with positions
    findall(
        _{id: Id, position: Pos, segment: Seg},
        (path_position(Id, Pos), segment_membership(Id, Seg)),
        Nodes
    ),

    % Collect path edges (prev/next links)
    findall(
        _{from: A, to: B, type: path_adjacent},
        prev_similar(A, B),
        PathEdges
    ),

    % Collect segment boundary edges
    findall(
        _{from: A, to: B, type: segment_bridge, from_seg: SegA, to_seg: SegB},
        segment_boundary(A, B, SegA, SegB),
        BridgeEdges
    ),

    append(PathEdges, BridgeEdges, AllEdges),

    % Write JSON
    open(JsonPath, write, Stream),
    json_write_dict(Stream, _{
        nodes: Nodes,
        edges: AllEdges,
        metadata: _{
            type: "similarity_path",
            description: "MST+DFS ordering for federation"
        }
    }),
    close(Stream).

%% =============================================================================
%% Bridge Interface (called from Python)
%% =============================================================================

%% load_tree_from_json(JsonPath)
%% Load node structure from JSON file generated by Python
load_tree_from_json(JsonPath) :-
    % Clear existing data
    retractall(node(_, _, _, _, _)),
    retractall(parent(_, _)),
    retractall(similarity_score(_, _)),

    % Load JSON (using SWI-Prolog's http/json library)
    open(JsonPath, read, Stream),
    json_read_dict(Stream, Data),
    close(Stream),

    % Assert nodes
    forall(
        member(Node, Data.nodes),
        assertz(node(Node.id, Node.cluster_count, Node.total_pairs,
                     Node.depth, Node.avg_pairs))
    ),

    % Assert parent relationships
    forall(
        member(Edge, Data.edges),
        assertz(parent(Edge.parent, Edge.child))
    ),

    % Assert similarity scores
    forall(
        member(Sim, Data.similarities),
        assertz(similarity_score(Sim.id, Sim.score))
    ).

%% export_plan_to_json(Plan, JsonPath)
%% Export plan to JSON for Python to execute
export_plan_to_json(Plan, JsonPath) :-
    findall(_{technique: Tech, node: Node},
            member(apply(Tech, Node), Plan),
            Actions),
    open(JsonPath, write, Stream),
    json_write_dict(Stream, _{actions: Actions}),
    close(Stream).
