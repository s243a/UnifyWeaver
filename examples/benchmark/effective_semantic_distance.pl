% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025-2026 John William Creighton (@s243a)
%
% effective_semantic_distance.pl — Effective distance with semantic weights
%
% Combines the power-mean effective distance formula:
%     d_eff = (Σ dᵢ^(-N))^(-1/N)
% with semantic edge weights (1 - cosine_similarity) instead of hop counts.
%
% Each path's cost dᵢ is the sum of semantic edge weights along that path.
% The power-mean then aggregates across ALL paths, heavily weighting
% the shortest (most semantically coherent) paths.
%
% This is the full generalization:
%   - Hop-count effective distance: dᵢ = hop count, N=5
%   - Min semantic distance: equivalent to N→∞ (only shortest path matters)
%   - Effective semantic distance: dᵢ = weighted path cost, N configurable

:- module(effective_semantic_distance, [
    effective_semantic_dist/4,    % +Start, +Target, +N, -Deff
    effective_semantic_dist/3,    % +Start, +Target, -Deff (N=5)
    semantic_path_cost/4,         % +Start, +Target, +Visited, -Cost
    edge_weight/3,                % +From, +To, -Weight (dynamic)
    dimension_n/1                 % -N (configurable)
]).

:- use_module(library(lists)).
:- use_module(library(aggregate)).

:- dynamic edge_weight/3.

%% dimension_n(-N)
%  Power-mean exponent. Default 5 (Wikipedia graph dimensionality).
%  Override by retracting and asserting a new value.
:- dynamic dimension_n/1.
dimension_n(5).

% ============================================================================
% CORE SPECIFICATION
% ============================================================================

%% effective_semantic_dist(+Start, +Target, -Deff)
%  Effective semantic distance with default N from dimension_n/1.
effective_semantic_dist(Start, Target, Deff) :-
    dimension_n(N),
    effective_semantic_dist(Start, Target, N, Deff).

%% effective_semantic_dist(+Start, +Target, +N, -Deff)
%
%  Compute effective semantic distance using power-mean over all paths.
%
%    d_eff = (Σ dᵢ^(-N))^(-1/N)
%
%  where dᵢ = sum of semantic edge weights along path i.
%
%  The formula heavily weights shorter paths:
%    N=1  → harmonic mean (mild weighting)
%    N=5  → strong weighting toward shortest paths
%    N→∞  → converges to min (only shortest path matters)
%
%  Transpiler recognizes this pattern:
%    aggregate_all(sum(W), (path_enum, W is Cost^NegN), WeightSum)
%  and can optimize the path enumeration.
%
effective_semantic_dist(Start, Target, N, Deff) :-
    NegN is -N,
    aggregate_all(sum(W),
        ( semantic_path_cost(Start, Target, [Start], Cost),
          W is Cost ** NegN ),
        WeightSum),
    WeightSum > 0,
    InvN is -1 / N,
    Deff is WeightSum ** InvN.

%% semantic_path_cost(+Start, +Target, +Visited, -Cost)
%  Enumerate all paths from Start to Target with cycle detection.
%  Cost = sum of edge weights along the path.
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

edge_weight(ml, ai, 0.12).
edge_weight(ai, cs, 0.18).
edge_weight(cs, science, 0.30).
edge_weight(ml, statistics, 0.20).
edge_weight(statistics, math, 0.15).
edge_weight(math, science, 0.25).
edge_weight(ai, math, 0.28).
edge_weight(ml, cs, 0.35).

% ============================================================================
% TESTS
% ============================================================================

test_effective_semantic_distance :-
    format('=== Effective Semantic Distance Tests ===~n'),

    % ml → science: 4 paths with costs [0.60, 0.60, 0.65, 0.65]
    % d_eff(N=5) = (Σ dᵢ^-5)^(-1/5) = 0.4714
    % Multiple paths make effective distance LESS than min path cost,
    % because more paths = more "ways to get there" = effectively closer.
    effective_semantic_dist(ml, science, 5, D1),
    format('  ml -> science (N=5): ~6f~n', [D1]),
    (abs(D1 - 0.4714) < 0.01 -> format('  PASS~n') ; format('  FAIL~n')),

    % N=1 (harmonic mean) — even more paths contribute, smaller result
    effective_semantic_dist(ml, science, 1, D2),
    format('  ml -> science (N=1): ~6f~n', [D2]),
    (D2 < D1 -> format('  PASS: N=1 < N=5 (more paths contribute)~n') ; format('  FAIL~n')),

    % N=20 — converges toward min distance (0.60)
    effective_semantic_dist(ml, science, 20, D3),
    format('  ml -> science (N=20): ~6f~n', [D3]),
    (D3 > D1 -> format('  PASS: N=20 > N=5 (approaches min)~n') ; format('  FAIL~n')),

    % ml → ai: single path, d_eff = path cost regardless of N
    effective_semantic_dist(ml, ai, 5, D4),
    format('  ml -> ai (N=5): ~6f (expect 0.12)~n', [D4]),
    (abs(D4 - 0.12) < 0.001 -> format('  PASS~n') ; format('  FAIL~n')),

    % Default N (uses dimension_n/1)
    effective_semantic_dist(ml, science, D5),
    format('  ml -> science (default N): ~6f~n', [D5]),
    (abs(D5 - D1) < 0.001 -> format('  PASS: matches N=5~n') ; format('  FAIL~n')),

    format('=== Tests Complete ===~n').
