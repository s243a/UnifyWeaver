:- initialization(main, main).

:- use_module('../../src/unifyweaver/targets/wam_haskell_target').
:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module('../../src/unifyweaver/targets/prolog_target').

%% generate_wam_haskell_matrix_benchmark.pl
%%
%% Generates an effective-distance Haskell benchmark from OPTIMIZED Prolog,
%% with explicit control over the execution mode:
%%
%%   - interpreter + no_kernels(true)  => pure interpreter baseline
%%   - interpreter + kernels enabled   => hybrid WAM + FFI
%%   - functions   + no_kernels(true)  => lowered-only
%%   - functions   + kernels enabled   => lowered with WAM fallback + FFI
%%
%% This makes the benchmark path explicit:
%%   workload Prolog
%%     -> prolog_target optimization
%%     -> optimized Prolog predicates
%%     -> WAM/Haskell project generation
%%
%% Usage:
%%   swipl -q -s generate_wam_haskell_matrix_benchmark.pl -- \
%%       <facts.pl> <output-dir> <seeded|accumulated> <interpreter|functions> <kernels_on|kernels_off>

benchmark_workload_path(Path) :-
    source_file(benchmark_workload_path(_), ThisFile),
    file_directory_name(ThisFile, Here),
    directory_file_path(Here, 'effective_distance.pl', Path).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [_FactsPath, OutputDir, VariantAtom, EmitModeAtom, KernelModeAtom]
    ->  true
    ;   format(user_error,
            'Usage: ... -- <facts.pl> <output-dir> <seeded|accumulated> <interpreter|functions> <kernels_on|kernels_off>~n',
            []),
        halt(1)
    ),
    generate(VariantAtom, EmitModeAtom, KernelModeAtom, OutputDir),
    halt(0).

main :-
    format(user_error, 'Error: generation failed~n', []),
    halt(1).

generate(VariantAtom, EmitModeAtom, KernelModeAtom, OutputDir) :-
    benchmark_workload_path(WorkloadPath),
    load_files(WorkloadPath, [silent(true)]),
    retractall(user:mode(category_ancestor(_, _, _, _))),
    assertz(user:mode(category_ancestor(-, +, -, +))),
    parse_variant(VariantAtom, OptimizationOptions),
    parse_emit_mode(EmitModeAtom, EmitMode),
    parse_kernel_mode(KernelModeAtom, KernelOptions),
    BasePreds = [dimension_n/1, max_depth/1, category_ancestor/4],
    prolog_target:generate_prolog_script(BasePreds, OptimizationOptions, ScriptCode),
    tmp_file_stream(text, TmpPath, TmpStream),
    write(TmpStream, ScriptCode),
    close(TmpStream),
    load_files(TmpPath, [silent(true)]),
    delete_file(TmpPath),
    collect_wam_predicates(VariantAtom, Predicates),
    query_pred_for_variant(VariantAtom, QueryPredOpts),
    append([[module_name('wam-haskell-matrix-bench'), emit_mode(EmitMode)], KernelOptions, QueryPredOpts], Options),
    write_wam_haskell_project(Predicates, Options, OutputDir),
    format(user_error,
           '[WAM-Haskell-Matrix] variant=~w emit_mode=~w kernels=~w output=~w~n',
           [VariantAtom, EmitMode, KernelModeAtom, OutputDir]).

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

parse_emit_mode(interpreter, interpreter).
parse_emit_mode(functions, functions).

parse_kernel_mode(kernels_on, []).
parse_kernel_mode(kernels_off, [no_kernels(true)]).

query_pred_for_variant(seeded, []).
query_pred_for_variant(accumulated, [
    query_pred('category_ancestor$effective_distance_sum_selected/3')
]).

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
    user:'category_ancestor$effective_distance_sum_selected'/3,
    user:'category_ancestor$effective_distance_sum_bound'/3
]).

power_sum_bound(Cat, Root, NegN, WeightSum) :-
    aggregate_all(sum(W),
        (category_ancestor(Cat, Root, Hops, [Cat]),
         H is Hops + 1,
         W is H ** NegN),
        WeightSum).
