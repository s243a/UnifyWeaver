:- initialization(main, main).

:- use_module('../../src/unifyweaver/targets/wam_fsharp_target').
:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module('../../src/unifyweaver/targets/prolog_target').
:- use_module(library(option)).
:- use_module(library(lists)).

%% generate_wam_fsharp_optimized_benchmark.pl
%%
%% Generates an F# WAM benchmark from OPTIMIZED Prolog.
%%
%% Pipeline:
%%   1. Load the effective-distance Prolog workload + facts
%%   2. Generate optimized predicates via prolog_target (seeded accumulation)
%%   3. Load the generated script to assert optimized helper predicates
%%   4. Compile ALL predicates (base + generated) to WAM -> F#
%%
%% This bridges the Prolog optimization passes with the WAM F# target,
%% ensuring the WAM compiler sees the same optimized predicates that make
%% the SWI-Prolog benchmarks fast.
%%
%% Usage:
%%   swipl -q -s generate_wam_fsharp_optimized_benchmark.pl -- \
%%       <facts.pl> <output-dir> [accumulated|seeded]
%%
%% Variants:
%%   seeded      - base category_ancestor + power_sum_bound (no helpers)
%%   accumulated - full seeded accumulation with effective_distance_sum helpers

benchmark_workload_path(Path) :-
    source_file(benchmark_workload_path(_), ThisFile),
    file_directory_name(ThisFile, Here),
    directory_file_path(Here, 'effective_distance.pl', Path).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [_FactsPath, OutputDir, VariantAtom]
    ->  true
    ;   Argv = [_FactsPath, OutputDir]
    ->  VariantAtom = accumulated  % default
    ;   format(user_error,
            'Usage: ... -- <facts.pl> <output-dir> [accumulated|seeded]~n', []),
        halt(1)
    ),
    generate(VariantAtom, OutputDir),
    halt(0).

main :-
    format(user_error, 'Error: generation failed~n', []),
    halt(1).

generate(VariantAtom, OutputDir) :-
    % Step 1: Load the base workload
    benchmark_workload_path(WorkloadPath),
    load_files(WorkloadPath, [silent(true)]),
    retractall(user:mode(category_ancestor(_, _, _, _))),
    assertz(user:mode(category_ancestor(-, +, -, +))),

    % Step 2: Generate optimized Prolog via prolog_target
    parse_variant(VariantAtom, OptimizationOptions),
    BasePreds = [dimension_n/1, max_depth/1, category_ancestor/4],
    prolog_target:generate_prolog_script(BasePreds, OptimizationOptions, ScriptCode),

    % Step 3: Write to temp file and load it to assert the generated predicates
    tmp_file_stream(text, TmpPath, TmpStream),
    write(TmpStream, ScriptCode),
    close(TmpStream),
    load_files(TmpPath, [silent(true)]),
    delete_file(TmpPath),

    % Step 4: Collect all relevant predicates (base + generated helpers)
    collect_wam_predicates(VariantAtom, Predicates),
    format(user_error, '[WAM-FSharp-Optimized] variant=~w predicates=~w~n',
           [VariantAtom, Predicates]),

    % Step 5: Generate F# WAM project with lowered emitter and parallel support
    Options = [module_name('wam-fsharp-optimized-bench'),
               emit_mode(functions),
               parallel(true)],
    write_wam_fsharp_project(Predicates, Options, OutputDir),
    format(user_error, '[WAM-FSharp-Optimized] Generated project at ~w~n', [OutputDir]).

parse_variant(seeded, [
    dialect(swi),
    branch_pruning(false),
    min_closure(false)
]).
parse_variant(accumulated, [
    dialect(swi),
    branch_pruning(false),
    min_closure(false),
    seeded_accumulation(auto)
]).

%% collect_wam_predicates(+Variant, -Predicates)
%  Collect the predicate list to compile through WAM, including
%  generated helper predicates from the optimization pass.
collect_wam_predicates(seeded, [
    user:dimension_n/1,
    user:max_depth/1,
    user:category_ancestor/4,
    user:power_sum_bound/4
]).
collect_wam_predicates(accumulated, [
    user:dimension_n/1,
    user:max_depth/1,
    user:category_ancestor/4,
    user:'category_ancestor$power_sum_bound'/3,
    user:'category_ancestor$power_sum_selected'/3,
    user:'category_ancestor$power_sum_grouped'/3,
    user:'category_ancestor$power_sum_grouped$grouped'/2,
    user:'category_ancestor$power_sum_grouped$grouped$sum_pairs'/2,
    user:'category_ancestor$effective_distance_sum_selected'/3,
    user:'category_ancestor$effective_distance_sum_bound'/3
]).

%% power_sum_bound/4 - needed for seeded variant
power_sum_bound(Cat, Root, NegN, WeightSum) :-
    aggregate_all(sum(W),
        (category_ancestor(Cat, Root, Hops, [Cat]),
         H is Hops + 1,
         W is H ** NegN),
        WeightSum).
