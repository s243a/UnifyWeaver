:- initialization(main, main).

:- use_module('../../src/unifyweaver/targets/wam_haskell_target').
:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module(library(option)).
:- use_module(library(lists)).

%% Generate a WAM Haskell benchmark project that queries the enwiki
%% LMDB directly using int32 IDs as atoms (no string interning).
%%
%% Usage:
%%   swipl -q -s generate_wam_haskell_enwiki_lmdb_benchmark.pl -- \
%%       <facts.pl> <output-dir> [--cache | --cache-l1]
%%
%% facts.pl is the seeded benchmark workload (effective_distance.pl shape
%% with dimension_n/1, max_depth/1, category_ancestor/4, power_sum_bound/4).
%% category_parent/2 facts come from LMDB at runtime; the predicate is
%% in the lmdb_backed_facts allow-list and skipped from WAM compilation
%% via the external_source path.
%%
%% Optional cache flags:
%%   --cache      adds lmdb_cache_mode(memoize) — shared IntMap cache
%%                across all parMap worker threads.  Helps on workloads
%%                with subgraph overlap; can hurt on random shallow
%%                seeds where the atomicModifyIORef' contention exceeds
%%                the FFI savings.
%%   --cache-l1   adds lmdb_cache_mode(per_hec) — per-thread L1 cache.
%%                Zero CAS contention on the hot path; cache hits don't
%%                share across threads.  Right call for most DFS-style
%%                workloads where intra-thread reuse dominates.
%%
%% The generated project expects two extra files alongside the LMDB at
%% the runtime factsDir:
%%   seed_ids.txt  — newline-separated int32 seed category IDs
%%   root_ids.txt  — newline-separated int32 root category IDs

benchmark_workload_path(Path) :-
    source_file(benchmark_workload_path(_), ThisFile),
    file_directory_name(ThisFile, Here),
    directory_file_path(Here, 'effective_distance.pl', Path).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [FactsPath, OutputDir]
    ->  CacheMode = no
    ;   Argv = [FactsPath, OutputDir, '--cache']
    ->  CacheMode = memoize
    ;   Argv = [FactsPath, OutputDir, '--cache-l1']
    ->  CacheMode = per_hec
    ;   format(user_error,
               'Usage: ... -- <facts.pl> <output-dir> [--cache | --cache-l1]~n', []),
        halt(1)
    ),
    generate(FactsPath, OutputDir, CacheMode),
    halt(0).

main :-
    format(user_error, 'Error: enwiki LMDB benchmark generation failed~n', []),
    halt(1).

%% power_sum_bound(+Cat, +Root, +NegN, -WeightSum)
power_sum_bound(Cat, Root, NegN, WeightSum) :-
    aggregate_all(sum(W),
        (category_ancestor(Cat, Root, Hops, [Cat]),
         H is Hops + 1,
         W is H ** NegN),
        WeightSum).

generate(FactsPath, OutputDir, CacheMode) :-
    benchmark_workload_path(WorkloadPath),
    load_files(WorkloadPath, [silent(true)]),
    load_files(FactsPath, [silent(true)]),
    retractall(user:mode(category_ancestor(_, _, _, _))),
    assertz(user:mode(category_ancestor(-, +, -, +))),

    % Predicates compiled through WAM. category_parent/2 is in the
    % default lmdb_backed_facts allow-list, so it short-circuits via
    % external_source (no per-fact compilation).
    Predicates = [
        user:dimension_n/1,
        user:max_depth/1,
        user:category_parent/2,
        user:category_ancestor/4,
        user:power_sum_bound/4
    ],

    BaseOptions = [
        module_name('wam-haskell-enwiki'),
        use_lmdb(true),
        % Streaming-pipeline ingest format: named "main" subdb with
        % MDB_DUPSORT, one entry per (key, parent) edge.  Reader uses
        % cursor MDB_NEXT_DUP iteration.
        lmdb_layout(dupsort),
        int_atom_seeds(true),
        % Demand filter relies on the in-memory parentsIndexInterned
        % which is empty in int-atom-seeds mode (edges live in LMDB).
        % Disable so the FFI kernel falls back to LMDB lookups directly.
        demand_filter(false)
    ],
    (   CacheMode == memoize
    ->  Options = [lmdb_cache_mode(memoize) | BaseOptions]
    ;   CacheMode == per_hec
    ->  Options = [lmdb_cache_mode(per_hec) | BaseOptions]
    ;   Options = BaseOptions
    ),

    write_wam_haskell_project(Predicates, Options, OutputDir),
    (   CacheMode == memoize
    ->  format(user_error,
               '[WAM-Haskell] enwiki int-atom benchmark (memoize cache) generated at ~w~n',
               [OutputDir])
    ;   CacheMode == per_hec
    ->  format(user_error,
               '[WAM-Haskell] enwiki int-atom benchmark (L1 per-HEC cache) generated at ~w~n',
               [OutputDir])
    ;   format(user_error,
               '[WAM-Haskell] enwiki int-atom benchmark generated at ~w~n',
               [OutputDir])
    ).
