:- encoding(utf8).
%% ========================================================================
%% Demand Guard Benchmark
%%
%% Compares effective_distance with and without demand-driven pruning.
%% The demand set prunes branches that can never reach a root category,
%% eliminating ~80% of DFS exploration at typical scales.
%%
%% Usage:
%%   swipl -q -s examples/benchmark/benchmark_demand_guards.pl \
%%         -- data/benchmark/1k
%%
%%   swipl -q -s examples/benchmark/benchmark_demand_guards.pl \
%%         -- data/benchmark/10k
%% ========================================================================

:- initialization(main, main).

:- use_module('../../src/unifyweaver/core/demand_analysis').
:- use_module(library(lists)).

%% ========================================================================
%% Load the effective_distance workload
%% ========================================================================

:- dynamic article_category/2.
:- dynamic category_parent/2.
:- dynamic root_category/1.

dimension_n(5).
:- dynamic max_depth/1.
max_depth(10).

%% Mode declaration — required for demand analysis
:- assert(user:mode(category_ancestor(+, +, -, +))).

%% ========================================================================
%% Original category_ancestor/4 (unguarded)
%% ========================================================================

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

%% ========================================================================
%% Demand-guarded category_ancestor/4
%% ========================================================================

:- dynamic can_reach_via_category_parent/1.

init_demand_set :-
    retractall(can_reach_via_category_parent(_)),
    % Seed: all root categories
    forall(root_category(R), assert(can_reach_via_category_parent(R))),
    % Fixpoint: backward BFS over category_parent
    % edge(Child, Parent) — if Parent is reachable, Child is too
    repeat,
    (   category_parent(Child, Parent),
        can_reach_via_category_parent(Parent),
        \+ can_reach_via_category_parent(Child)
    ->  assert(can_reach_via_category_parent(Child)), fail
    ;   !
    ).

category_ancestor_guarded(Cat, Parent, 1, Visited) :-
    category_parent(Cat, Parent),
    can_reach_via_category_parent(Parent),
    \+ member(Parent, Visited).

category_ancestor_guarded(Cat, Ancestor, Hops, Visited) :-
    max_depth(MaxD),
    length(Visited, Depth),
    Depth < MaxD, !,
    category_parent(Cat, Mid),
    can_reach_via_category_parent(Mid),
    \+ member(Mid, Visited),
    category_ancestor_guarded(Mid, Ancestor, H1, [Mid|Visited]),
    Hops is H1 + 1.

%% ========================================================================
%% Benchmark infrastructure
%% ========================================================================

%% Load TSV fact files
load_facts(Dir) :-
    directory_file_path(Dir, 'category_parent.tsv', CpPath),
    directory_file_path(Dir, 'article_category.tsv', AcPath),
    directory_file_path(Dir, 'root_categories.tsv', RcPath),
    load_tsv2(CpPath, category_parent),
    load_tsv2(AcPath, article_category),
    load_tsv1(RcPath, root_category).

load_tsv2(Path, Pred) :-
    setup_call_cleanup(
        open(Path, read, In),
        (   read_line_to_string(In, _Header),  % skip header
            read_tsv2_lines(In, Pred)
        ),
        close(In)
    ).

read_tsv2_lines(In, Pred) :-
    read_line_to_string(In, Line),
    (   Line == end_of_file
    ->  true
    ;   split_string(Line, "\t", "", Parts),
        (   Parts = [A, B|_]
        ->  atom_string(AA, A), atom_string(AB, B),
            Fact =.. [Pred, AA, AB],
            assert(Fact)
        ;   true
        ),
        read_tsv2_lines(In, Pred)
    ).

load_tsv1(Path, Pred) :-
    setup_call_cleanup(
        open(Path, read, In),
        (   read_line_to_string(In, _Header),
            read_tsv1_lines(In, Pred)
        ),
        close(In)
    ).

read_tsv1_lines(In, Pred) :-
    read_line_to_string(In, Line),
    (   Line == end_of_file
    ->  true
    ;   atom_string(A, Line),
        (   A \= ''
        ->  Fact =.. [Pred, A], assert(Fact)
        ;   true
        ),
        read_tsv1_lines(In, Pred)
    ).

%% Collect seed categories
seed_categories(Seeds) :-
    findall(Cat, article_category(_, Cat), Cats0),
    sort(Cats0, Seeds).

%% Run one seed and collect weight sum
run_seed_original(Cat, Root, WeightSum) :-
    dimension_n(N), NegN is -N,
    aggregate_all(sum(W),
        (category_ancestor(Cat, Root, Hops, [Cat]),
         W is (Hops + 1) ** NegN),
        WeightSum).

run_seed_guarded(Cat, Root, WeightSum) :-
    dimension_n(N), NegN is -N,
    aggregate_all(sum(W),
        (category_ancestor_guarded(Cat, Root, Hops, [Cat]),
         W is (Hops + 1) ** NegN),
        WeightSum).

%% ========================================================================
%% Main
%% ========================================================================

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [FactsDir|_] -> true ; FactsDir = 'data/benchmark/1k' ),

    format(user_error, "Loading facts from ~w...~n", [FactsDir]),
    load_facts(FactsDir),

    aggregate_all(count, category_parent(_, _), NumEdges),
    aggregate_all(count, article_category(_, _), NumArticles),
    aggregate_all(count, root_category(_), NumRoots),
    format(user_error, "  edges=~w articles=~w roots=~w~n", [NumEdges, NumArticles, NumRoots]),

    seed_categories(Seeds),
    length(Seeds, NumSeeds),
    root_category(Root),

    %% --- Phase 1: Run demand analysis ---
    format(user_error, "~nPhase 1: Demand analysis detection...~n", []),
    CA_Clauses = [
        (category_ancestor(Cat1, Parent1, 1, Visited1) :-
            (category_parent(Cat1, Parent1), \+ member(Parent1, Visited1))),
        (category_ancestor(Cat2, Ancestor2, Hops2, Visited2) :-
            (max_depth(MaxD2), length(Visited2, Depth2), Depth2 < MaxD2, !,
             category_parent(Cat2, Mid2), \+ member(Mid2, Visited2),
             category_ancestor(Mid2, Ancestor2, H12, [Mid2|Visited2]),
             Hops2 is H12 + 1))
    ],
    findall(H-B, member((H :- B), CA_Clauses), Pairs),
    (   detect_demand_eligible(category_ancestor, 4, Pairs, Spec)
    ->  demand_spec_edge_pred(Spec, EP),
        demand_spec_target_arg(Spec, TA),
        demand_spec_guard_points(Spec, GPs),
        length(GPs, NumGPs),
        format(user_error, "  eligible=yes edge_pred=~w target_arg=~w guard_points=~w~n",
               [EP, TA, NumGPs])
    ;   format(user_error, "  eligible=no~n", []),
        halt(1)
    ),

    %% --- Phase 2: Compute demand set ---
    format(user_error, "~nPhase 2: Computing demand set...~n", []),
    get_time(T0),
    init_demand_set,
    get_time(T1),
    aggregate_all(count, can_reach_via_category_parent(_), DemandSize),
    DemandMs is (T1 - T0) * 1000,
    findall(C, (category_parent(C, _) ; category_parent(_, C)), AllCats0),
    sort(AllCats0, AllCats),
    length(AllCats, TotalNodes),
    NodePruneRatio is 1.0 - (DemandSize / max(TotalNodes, 1)),
    format(user_error, "  demand_set_size=~w  total_nodes=~w  demand_ms=~1f~n",
           [DemandSize, TotalNodes, DemandMs]),
    format(user_error, "  node_prune_ratio=~2f (fraction of nodes pruned)~n", [NodePruneRatio]),

    %% --- Phase 3: Benchmark original (unguarded) ---
    format(user_error, "~nPhase 3: Running original (unguarded) on ~w seeds...~n", [NumSeeds]),
    get_time(T2),
    findall(Cat-WS,
        (member(Cat, Seeds), run_seed_original(Cat, Root, WS), WS > 0),
        OrigResults),
    get_time(T3),
    length(OrigResults, OrigCount),
    OrigMs is (T3 - T2) * 1000,
    format(user_error, "  original: tuple_count=~w query_ms=~0f~n", [OrigCount, OrigMs]),

    %% --- Phase 4: Benchmark guarded ---
    format(user_error, "~nPhase 4: Running demand-guarded on ~w seeds...~n", [NumSeeds]),
    get_time(T4),
    findall(Cat-WS,
        (member(Cat, Seeds), run_seed_guarded(Cat, Root, WS), WS > 0),
        GuardedResults),
    get_time(T5),
    length(GuardedResults, GuardedCount),
    GuardedMs is (T5 - T4) * 1000,
    format(user_error, "  guarded:  tuple_count=~w query_ms=~0f~n", [GuardedCount, GuardedMs]),

    %% --- Phase 5: Verify correctness ---
    format(user_error, "~nPhase 5: Correctness check...~n", []),
    sort(OrigResults, OrigSorted),
    sort(GuardedResults, GuardedSorted),
    (   OrigSorted == GuardedSorted
    ->  format(user_error, "  PASS: identical results~n", [])
    ;   length(OrigSorted, OL), length(GuardedSorted, GL),
        format(user_error, "  FAIL: original=~w guarded=~w results differ~n", [OL, GL])
    ),

    %% --- Summary ---
    (   OrigMs > 0
    ->  Speedup is OrigMs / GuardedMs
    ;   Speedup = 0.0
    ),
    format(user_error, "~n========================================~n", []),
    format(user_error, "edges=~w seeds=~w demand_set=~w total_nodes=~w~n",
           [NumEdges, NumSeeds, DemandSize, TotalNodes]),
    format(user_error, "node_prune_ratio=~2f~n", [NodePruneRatio]),
    format(user_error, "demand_init_ms=~1f~n", [DemandMs]),
    format(user_error, "original_query_ms=~0f~n", [OrigMs]),
    format(user_error, "guarded_query_ms=~0f~n", [GuardedMs]),
    format(user_error, "speedup=~2fx~n", [Speedup]),
    format(user_error, "========================================~n", []),

    halt(0).

main :- halt(1).
