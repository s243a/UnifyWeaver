:- initialization(main, main).

:- use_module('../../src/unifyweaver/targets/wam_haskell_target').
:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module(library(option)).
:- use_module(library(lists)).

%% Generates Haskell WAM benchmarks for comparing fact layouts.
%%
%% Usage:
%%   swipl -q -s generate_wam_haskell_fact_layout_benchmark.pl -- <facts.pl> <output-dir> <compiled|inline>
%%
%% The 'compiled' variant uses compiled_only layout (all facts as WAM instructions).
%% The 'inline' variant uses fact_count_threshold(10) to trigger inline_data
%% for fact predicates with >10 clauses, using CallFactStream + literal lists.
%%
%% Both variants include category_parent/2 in the WAM predicate list (not just
%% TSV runtime loading) so the fact layout choice matters for the WAM interpreter.

benchmark_workload_path(Path) :-
    source_file(benchmark_workload_path(_), ThisFile),
    file_directory_name(ThisFile, Here),
    directory_file_path(Here, 'effective_distance.pl', Path).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [FactsPath, OutputDir, LayoutAtom]
    ->  true
    ;   format(user_error,
            'Usage: ... -- <facts.pl> <output-dir> <compiled|inline>~n', []),
        halt(1)
    ),
    parse_layout(LayoutAtom, LayoutOptions),
    generate_fact_layout_benchmark(FactsPath, OutputDir, LayoutOptions),
    halt(0).

main :-
    format(user_error, 'Error: generation failed~n', []),
    halt(1).

parse_layout('compiled', [fact_layout_policy(compiled_only)]) :- !.
parse_layout('inline', [fact_count_threshold(10)]) :- !.
parse_layout(X, _) :-
    format(user_error, 'Unknown layout: ~w (use compiled or inline)~n', [X]),
    halt(1).

%% power_sum_bound(+Cat, +Root, +NegN, -WeightSum)
power_sum_bound(Cat, Root, NegN, WeightSum) :-
    aggregate_all(sum(W),
        (category_ancestor(Cat, Root, Hops, [Cat]),
         H is Hops + 1,
         W is H ** NegN),
        WeightSum).

generate_fact_layout_benchmark(FactsPath, OutputDir, LayoutOptions) :-
    benchmark_workload_path(WorkloadPath),
    load_files(WorkloadPath, [silent(true)]),
    load_files(FactsPath, [silent(true)]),
    retractall(user:mode(category_ancestor(_, _, _, _))),
    assertz(user:mode(category_ancestor(-, +, -, +))),

    % Include category_parent/2 in the WAM predicate list so the
    % fact layout choice (compiled vs inline_data) is exercised.
    Predicates = [
        user:dimension_n/1,
        user:max_depth/1,
        user:category_parent/2,
        user:category_ancestor/4,
        user:power_sum_bound/4
    ],
    append([module_name('wam-haskell-bench'), no_kernels(true)],
           LayoutOptions, Options),

    write_wam_haskell_project(Predicates, Options, OutputDir),
    format(user_error, '[WAM-Haskell] Fact layout benchmark generated (~w).~n',
           [LayoutOptions]).
