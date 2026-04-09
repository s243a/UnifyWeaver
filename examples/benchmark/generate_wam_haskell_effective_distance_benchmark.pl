:- initialization(main, main).

:- use_module('../../src/unifyweaver/targets/wam_haskell_target').
:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module(library(option)).
:- use_module(library(lists)).

%% Generates a Haskell WAM benchmark for the effective-distance workload.
%%
%% Usage:
%%   swipl -q -s generate_wam_haskell_effective_distance_benchmark.pl -- <facts.pl> <output-dir>

benchmark_workload_path(Path) :-
    source_file(benchmark_workload_path(_), ThisFile),
    file_directory_name(ThisFile, Here),
    directory_file_path(Here, 'effective_distance.pl', Path).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [_FactsPath, OutputDir]
    ->  true
    ;   format(user_error, 'Usage: ... -- <facts.pl> <output-dir>~n', []),
        halt(1)
    ),
    generate_wam_haskell_benchmark(OutputDir),
    halt(0).

main :-
    format(user_error, 'Error: generation failed~n', []),
    halt(1).

%% power_sum_bound(+Cat, +Root, +NegN, -WeightSum)
%  Computes WeightSum = Σ (Hops+1)^NegN over all category_ancestor paths.
%  NegN should be negative (e.g., -5 for dimension n=5).
%  Compiled to WAM begin_aggregate/end_aggregate instructions.
power_sum_bound(Cat, Root, NegN, WeightSum) :-
    aggregate_all(sum(W),
        (category_ancestor(Cat, Root, Hops, [Cat]),
         H is Hops + 1,
         W is H ** NegN),
        WeightSum).

generate_wam_haskell_benchmark(OutputDir) :-
    benchmark_workload_path(WorkloadPath),
    load_files(WorkloadPath, [silent(true)]),
    retractall(user:mode(category_ancestor(_, _, _, _))),
    assertz(user:mode(category_ancestor(-, +, -, +))),

    Predicates = [
        user:dimension_n/1,
        user:max_depth/1,
        user:category_ancestor/4,
        user:power_sum_bound/4
    ],
    Options = [module_name('wam-haskell-bench')],

    write_wam_haskell_project(Predicates, Options, OutputDir),
    format(user_error, '[WAM-Haskell] Benchmark project generated.~n', []).
