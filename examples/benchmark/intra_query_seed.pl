%% ==========================================================================
%% intra_query_seed.pl
%%
%% Phase 4.0 workload for the WAM-Haskell intra-query parallelism
%% implementation plan. The shape mirrors effective-distance's
%% category_ancestor/4 aggregation, but the generator that wraps this
%% file disables FFI (`no_kernels(true)`) so the WAM *interpreter* is
%% in the hot path, and runs with only a handful of seeds. In that
%% regime, seed-level parallelism (PR #1377) gives at most N sparks for
%% N seeds — pinning most cores idle. Any speedup here would have to
%% come from forking *within* a query, which is exactly what Phase 4.1+
%% will deliver.
%%
%% Notes:
%%   - No `:- parallel(...)` directive yet — that's a Phase 4.1 addition
%%     to wam_target.pl. Phase 4.0 just establishes the workload and the
%%     sequential timing baseline.
%%   - max_depth is intentionally lower than the effective-distance
%%     default (6 vs 10) so the workload finishes in seconds per seed,
%%     not minutes. Larger values make the gap between seq and par even
%%     more dramatic but cost runtime on CI.
%%   - `power_sum_bound/4` is the aggregate the benchmark driver calls
%%     per seed. It's copied verbatim from effective_distance.pl so
%%     results are directly comparable.
%%
%% Related: docs/design/WAM_HASKELL_INTRA_QUERY_IMPLEMENTATION_PLAN.md §4.0
%% ==========================================================================

:- discontiguous category_parent/2.

:- dynamic max_depth/1.
max_depth(6).

dimension_n(5).

%% category_ancestor(+Cat, -Ancestor, -Hops, +Visited)
%  Transitive closure over category_parent/2 with cycle detection.
category_ancestor(Cat, Parent, 1, Visited) :-
    category_parent(Cat, Parent),
    \+ member(Parent, Visited).

category_ancestor(Cat, Ancestor, Hops, Visited) :-
    max_depth(MaxD),
    length(Visited, Depth),
    Depth < MaxD, !,
    category_parent(Cat, Mid),
    \+ member(Mid, Visited),
    category_ancestor(Mid, Ancestor, H1, [Mid|Visited]),
    Hops is H1 + 1.

%% power_sum_bound(+Cat, -Root, +NegN, -WeightSum)
%  WeightSum = Σ (Hops+1)^NegN across all ancestor paths.
%  Compiled to WAM begin_aggregate/end_aggregate instructions.
%  With Root unbound, aggregates over every reachable ancestor —
%  each is a choice point in the inner goal.
power_sum_bound(Cat, Root, NegN, WeightSum) :-
    aggregate_all(sum(W),
        (category_ancestor(Cat, Root, Hops, [Cat]),
         H is Hops + 1,
         W is H ** NegN),
        WeightSum).
