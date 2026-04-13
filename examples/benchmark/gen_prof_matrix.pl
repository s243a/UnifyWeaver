:- initialization(main, main).

:- use_module('../../src/unifyweaver/targets/wam_haskell_target').
:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module('../../src/unifyweaver/targets/prolog_target').
:- use_module(library(option)).
:- use_module(library(lists)).

%% gen_prof_matrix.pl
%%
%% Generates 4 Haskell WAM profiling configurations from optimized Prolog:
%%
%%   A: pure-interp    — emit_mode(interpreter), no_kernels(true)
%%   B: interp-ffi     — emit_mode(interpreter), kernels enabled
%%   C: lowered-only   — emit_mode(functions),   no_kernels(true)
%%   D: lowered-ffi    — emit_mode(functions),   kernels enabled
%%
%% All configurations use GHC profiling (-prof -fprof-auto -rtsopts)
%% so you can run with +RTS -p to get cost-centre profiling output.
%%
%% Usage:
%%   swipl -q -s gen_prof_matrix.pl -- <facts.pl> <output-base-dir>
%%
%% Generates:
%%   <output-base-dir>/A-pure-interp/
%%   <output-base-dir>/B-interp-ffi/
%%   <output-base-dir>/C-lowered-only/
%%   <output-base-dir>/D-lowered-ffi/

benchmark_workload_path(Path) :-
    source_file(benchmark_workload_path(_), ThisFile),
    file_directory_name(ThisFile, Here),
    directory_file_path(Here, 'effective_distance.pl', Path).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [_FactsPath, OutputBase]
    ->  true
    ;   format(user_error,
            'Usage: swipl -q -s gen_prof_matrix.pl -- <facts.pl> <output-base-dir>~n', []),
        halt(1)
    ),
    generate_all(OutputBase),
    halt(0).

main :- halt(1).

generate_all(OutputBase) :-
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

    % Step 4: Generate all 4 configurations
    Predicates = [
        user:dimension_n/1,
        user:max_depth/1,
        user:category_ancestor/4,
        user:'category_ancestor$power_sum_bound'/3,
        user:'category_ancestor$power_sum_selected'/3,
        user:'category_ancestor$effective_distance_sum_selected'/3,
        user:'category_ancestor$effective_distance_sum_bound'/3
    ],
    QueryPredOpts = [query_pred('category_ancestor$effective_distance_sum_selected/3')],

    configs(Configs),
    forall(
        member(config(Label, EmitMode, NoKernels), Configs),
        generate_config(OutputBase, Label, EmitMode, NoKernels, Predicates, QueryPredOpts)
    ),
    format(user_error, '~n[prof-matrix] All 4 configurations generated under ~w~n', [OutputBase]).

configs([
    config('A-pure-interp',  interpreter, true),
    config('B-interp-ffi',   interpreter, false),
    config('C-lowered-only', functions,   true),
    config('D-lowered-ffi',  functions,   false)
]).

generate_config(OutputBase, Label, EmitMode, NoKernels, Predicates, QueryPredOpts) :-
    format(atom(OutputDir), '~w/~w', [OutputBase, Label]),
    format(atom(ModName), 'wam-prof-~w', [Label]),
    BaseOpts = [
        module_name(ModName),
        emit_mode(EmitMode),
        profiling(true)
    ],
    (   NoKernels == true
    ->  KernelOpts = [no_kernels(true)]
    ;   KernelOpts = []
    ),
    append([BaseOpts, KernelOpts, QueryPredOpts], Options),
    format(user_error, '[prof-matrix] Generating ~w (emit_mode=~w, no_kernels=~w)~n',
           [Label, EmitMode, NoKernels]),
    write_wam_haskell_project(Predicates, Options, OutputDir),
    format(user_error, '[prof-matrix] ~w -> ~w~n', [Label, OutputDir]).
