% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025-2026 John William Creighton (@s243a)
%
% min_semantic_distance.pl — Minimum semantic distance specification
%
% Finds the shortest weighted path between nodes where edge weights
% are semantic distances (1 - cosine_similarity between embeddings).
%
% The shortest path is typically close to the greedy route (follow
% minimum semantic error at each step), but backtracking handles
% cases where greedy leads to dead ends or suboptimal paths.
%
% This Prolog specification is designed for transpilation by UnifyWeaver.
% The transpiler recognizes the weighted_shortest_path pattern and
% lowers it to Dijkstra's algorithm with a priority queue in the
% target language.

:- module(min_semantic_distance, [
    min_semantic_dist/3,        % +Start, +Target, -MinDist
    semantic_path_cost/4,       % +Start, +Target, +Visited, -Cost
    edge_weight/3               % +From, +To, -Weight (dynamic)
]).

:- use_module(library(lists)).

% ============================================================================
% EDGE WEIGHTS (precomputed semantic distances)
% ============================================================================
%
% edge_weight(From, To, Weight) where Weight = 1 - cosine_sim(emb(From), emb(To))
% Lower weight = more semantically similar.
%
% These are asserted dynamically from precomputed embeddings, or loaded
% from a file. Example:
%
%   edge_weight(cat_machine_learning, cat_artificial_intelligence, 0.15).
%   edge_weight(cat_artificial_intelligence, cat_computer_science, 0.22).
%   edge_weight(cat_computer_science, cat_science, 0.35).

:- dynamic edge_weight/3.

% ============================================================================
% CORE SPECIFICATION
% ============================================================================

%% min_semantic_dist(+Start, +Target, -MinDist)
%
%  Find the minimum-cost path from Start to Target where cost is the
%  sum of semantic edge weights along the path.
%
%  This is the top-level predicate that UnifyWeaver transpiles.
%  The transpiler recognizes the aggregate_all(min(...)) + recursive
%  path cost pattern and lowers it to Dijkstra's algorithm.
%
min_semantic_dist(Start, Target, MinDist) :-
    aggregate_all(min(Cost),
        semantic_path_cost(Start, Target, [Start], Cost),
        MinDist).

%% semantic_path_cost(+Start, +Target, +Visited, -Cost)
%
%  Compute the cost of a specific path from Start to Target,
%  avoiding cycles via the Visited list.
%
%  Base case: direct edge exists.
%  Recursive case: follow an edge, accumulate weight, recurse.
%
%  The greedy heuristic emerges naturally: paths through
%  semantically close nodes have lower total cost and are
%  found first by the optimizer.
%
semantic_path_cost(X, Y, _Visited, W) :-
    edge_weight(X, Y, W).

semantic_path_cost(X, Y, Visited, Cost) :-
    edge_weight(X, Z, W),
    \+ member(Z, Visited),
    semantic_path_cost(Z, Y, [Z|Visited], RestCost),
    Cost is W + RestCost.

% ============================================================================
% DEMO DATA
% ============================================================================
%
% Small category hierarchy with semantic edge weights.
% Weights represent semantic distance (1 - cosine_similarity).

edge_weight(ml, ai, 0.12).
edge_weight(ai, cs, 0.18).
edge_weight(cs, science, 0.30).
edge_weight(ml, statistics, 0.20).
edge_weight(statistics, math, 0.15).
edge_weight(math, science, 0.25).
edge_weight(ai, math, 0.28).
edge_weight(ml, cs, 0.35).          % direct but semantically further

% ============================================================================
% TEST
% ============================================================================

test_min_semantic_distance :-
    format('=== Min Semantic Distance Tests ===~n'),

    % ml → science: two main paths
    %   ml → ai → cs → science = 0.12 + 0.18 + 0.30 = 0.60
    %   ml → statistics → math → science = 0.20 + 0.15 + 0.25 = 0.60
    %   ml → ai → math → science = 0.12 + 0.28 + 0.25 = 0.65
    %   ml → cs → science = 0.35 + 0.30 = 0.65
    % Min should be 0.60
    min_semantic_dist(ml, science, D1),
    format('  ml → science: ~4f (expect 0.60)~n', [D1]),
    (abs(D1 - 0.60) < 0.001 -> format('  PASS~n') ; format('  FAIL~n')),

    % ml → ai: direct edge
    min_semantic_dist(ml, ai, D2),
    format('  ml → ai: ~4f (expect 0.12)~n', [D2]),
    (abs(D2 - 0.12) < 0.001 -> format('  PASS~n') ; format('  FAIL~n')),

    % ml → math: ml→statistics→math (0.35) vs ml→ai→math (0.40)
    min_semantic_dist(ml, math, D3),
    format('  ml → math: ~4f (expect 0.35)~n', [D3]),
    (abs(D3 - 0.35) < 0.001 -> format('  PASS~n') ; format('  FAIL~n')),

    format('=== Tests Complete ===~n').
