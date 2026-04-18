:- module(generate_wam_python_effective_distance_benchmark, [main/0, generate/3]).
:- initialization(main, main).

:- use_module('../../src/unifyweaver/targets/wam_python_target').
:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module('../../src/unifyweaver/targets/prolog_target').
:- use_module(library(option)).
:- use_module(library(lists)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).

%% generate_wam_python_effective_distance_benchmark.pl
%%
%% Generates a Python hybrid-WAM benchmark for the effective-distance workload.
%%
%% Pipeline:
%%   1. Load the effective-distance workload and the benchmark facts.
%%   2. Generate optimized predicates via prolog_target (accumulated variant).
%%   3. Force the selected predicates through the shared Python WAM path.
%%   4. Emit a Python benchmark driver that queries the compiled VM directly.
%%
%% This is the non-optimized baseline — mirrors
%% generate_wam_go_effective_distance_benchmark.pl for Python.
%%
%% Usage:
%%   swipl -q -s generate_wam_python_effective_distance_benchmark.pl -- \
%%       <facts.pl> <output-dir> [accumulated] [kernels_on|kernels_off]

benchmark_workload_path(Path) :-
    source_file(benchmark_workload_path(_), ThisFile),
    file_directory_name(ThisFile, Here),
    directory_file_path(Here, 'effective_distance.pl', Path).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [FactsPath, OutputDir, KernelModeAtom]
    ->  true
    ;   Argv = [FactsPath, OutputDir]
    ->  KernelModeAtom = kernels_on
    ;   format(user_error,
            'Usage: ... -- <facts.pl> <output-dir> [kernels_on|kernels_off]~n',
            []),
        halt(1)
    ),
    generate(FactsPath, OutputDir, KernelModeAtom),
    halt(0).

main :-
    format(user_error, 'Error: generation failed~n', []),
    halt(1).

generate(FactsPath, OutputDir, KernelModeAtom) :-
    % Step 1: Load the base workload
    benchmark_workload_path(WorkloadPath),
    load_files(WorkloadPath, [silent(true)]),
    retractall(user:mode(category_ancestor(_, _, _, _))),
    assertz(user:mode(category_ancestor(-, +, -, +))),

    % Step 2: Generate optimized Prolog via prolog_target (accumulated variant)
    OptimizationOptions = [
        dialect(swi),
        branch_pruning(false),
        min_closure(false),
        seeded_accumulation(auto)
    ],
    BasePreds = [dimension_n/1, max_depth/1, category_ancestor/4],
    prolog_target:generate_prolog_script(BasePreds, OptimizationOptions, ScriptCode),

    % Step 3: Load generated predicates
    tmp_file_stream(text, TmpPath, TmpStream),
    write(TmpStream, ScriptCode),
    close(TmpStream),
    load_files(TmpPath, [silent(true)]),
    delete_file(TmpPath),

    % Step 4: Load benchmark facts
    load_files(FactsPath, [silent(true)]),

    % Step 5: Collect predicates based on kernel mode
    parse_kernel_mode(KernelModeAtom, KernelOptions),
    collect_wam_predicates(KernelModeAtom, Predicates),

    % Step 6: Generate Python WAM project
    append([
        [module_name('wam-python-effective-distance-bench'),
         prefer_wam(true),
         wam_fallback(true),
         emit_mode(functions),
         parallel(true)],
        KernelOptions
    ], Options),
    write_wam_python_project(Predicates, Options, OutputDir),
    format(user_error,
           '[WAM-Python-EffectiveDistance] kernels=~w output=~w~n',
           [KernelModeAtom, OutputDir]).

parse_kernel_mode(kernels_on, []).
parse_kernel_mode(kernels_off, [no_kernels(true)]).

collect_wam_predicates(kernels_on, [
    user:dimension_n/1,
    user:max_depth/1,
    user:category_ancestor/4,
    user:'category_ancestor$power_sum_bound'/3,
    user:'category_ancestor$power_sum_selected'/3,
    user:'category_ancestor$effective_distance_sum_selected'/3,
    user:'category_ancestor$effective_distance_sum_bound'/3
]).

collect_wam_predicates(kernels_off, [
    user:dimension_n/1,
    user:max_depth/1,
    user:category_parent/2,
    user:category_ancestor/4,
    user:'category_ancestor$power_sum_bound'/3,
    user:'category_ancestor$power_sum_selected'/3,
    user:'category_ancestor$effective_distance_sum_selected'/3,
    user:'category_ancestor$effective_distance_sum_bound'/3
]).
