:- initialization(main, main).

:- use_module('../../src/unifyweaver/targets/wam_haskell_target').
:- use_module('../../src/unifyweaver/targets/wam_go_target').
:- use_module('../../src/unifyweaver/targets/wam_python_target').
:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module('../../src/unifyweaver/targets/prolog_target').
:- use_module(library(option)).
:- use_module(library(lists)).

%% gen_prof_matrix.pl
%%
%% Generates 4 Haskell + 4 Go + 4 Python WAM profiling configurations
%% from optimized Prolog:
%%
%%   Haskell:
%%     A: pure-interp    — emit_mode(interpreter), no_kernels(true)
%%     B: interp-ffi     — emit_mode(interpreter), kernels enabled
%%     C: lowered-only   — emit_mode(functions),   no_kernels(true)
%%     D: lowered-ffi    — emit_mode(functions),   kernels enabled
%%
%%   Go:
%%     E: go-pure-interp  — emit_mode(interpreter), no_kernels(true)
%%     F: go-interp-ffi   — emit_mode(interpreter), kernels enabled
%%     G: go-lowered-only — emit_mode(functions),   no_kernels(true)
%%     H: go-lowered-ffi  — emit_mode(functions),   kernels enabled
%%
%%   Python:
%%     I: py-pure-interp  — emit_mode(interpreter), no_kernels(true)
%%     J: py-interp-ffi   — emit_mode(interpreter), kernels enabled
%%     K: py-lowered-only — emit_mode(functions),   no_kernels(true)
%%     L: py-lowered-ffi  — emit_mode(functions),   kernels enabled
%%
%% Haskell configurations use GHC profiling (-prof -fprof-auto -rtsopts).
%% Go configurations are profiled via `go tool pprof` (CPU profiling).
%% Python configurations are profiled via cProfile / py-spy.
%%
%% Usage:
%%   swipl -q -s gen_prof_matrix.pl -- <facts.pl> <output-base-dir>
%%
%% Generates:
%%   <output-base-dir>/A-pure-interp/       (Haskell)
%%   <output-base-dir>/B-interp-ffi/        (Haskell)
%%   <output-base-dir>/C-lowered-only/      (Haskell)
%%   <output-base-dir>/D-lowered-ffi/       (Haskell)
%%   <output-base-dir>/E-go-pure-interp/    (Go)
%%   <output-base-dir>/F-go-interp-ffi/     (Go)
%%   <output-base-dir>/G-go-lowered-only/   (Go)
%%   <output-base-dir>/H-go-lowered-ffi/    (Go)
%%   <output-base-dir>/I-py-pure-interp/    (Python)
%%   <output-base-dir>/J-py-interp-ffi/     (Python)
%%   <output-base-dir>/K-py-lowered-only/   (Python)
%%   <output-base-dir>/L-py-lowered-ffi/    (Python)

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

    % Step 4: Generate all 12 configurations (4 Haskell + 4 Go + 4 Python)
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

    haskell_configs(HaskellConfigs),
    forall(
        member(config(Label, EmitMode, NoKernels), HaskellConfigs),
        generate_haskell_config(OutputBase, Label, EmitMode, NoKernels, Predicates, QueryPredOpts)
    ),

    go_configs(GoConfigs),
    forall(
        member(config(Label, EmitMode, NoKernels), GoConfigs),
        generate_go_config(OutputBase, Label, EmitMode, NoKernels, Predicates)
    ),

    python_configs(PythonConfigs),
    forall(
        member(config(Label, EmitMode, NoKernels), PythonConfigs),
        generate_python_config(OutputBase, Label, EmitMode, NoKernels, Predicates)
    ),
    format(user_error, '~n[prof-matrix] All 12 configurations generated under ~w~n', [OutputBase]).

haskell_configs([
    config('A-pure-interp',  interpreter, true),
    config('B-interp-ffi',   interpreter, false),
    config('C-lowered-only', functions,   true),
    config('D-lowered-ffi',  functions,   false)
]).

go_configs([
    config('E-go-pure-interp',  interpreter, true),
    config('F-go-interp-ffi',   interpreter, false),
    config('G-go-lowered-only', functions,   true),
    config('H-go-lowered-ffi',  functions,   false)
]).

generate_haskell_config(OutputBase, Label, EmitMode, NoKernels, Predicates, QueryPredOpts) :-
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
    format(user_error, '[prof-matrix] Generating Haskell ~w (emit_mode=~w, no_kernels=~w)~n',
           [Label, EmitMode, NoKernels]),
    write_wam_haskell_project(Predicates, Options, OutputDir),
    format(user_error, '[prof-matrix] ~w -> ~w~n', [Label, OutputDir]).

generate_go_config(OutputBase, Label, EmitMode, NoKernels, Predicates) :-
    format(atom(OutputDir), '~w/~w', [OutputBase, Label]),
    format(atom(ModName), 'wam-prof-~w', [Label]),
    BaseOpts = [
        module_name(ModName),
        package_name(main),
        prefer_wam(true),
        wam_fallback(true),
        foreign_lowering(true),
        emit_mode(EmitMode),
        parallel(true)
    ],
    (   NoKernels == true
    ->  KernelOpts = [no_kernels(true)]
    ;   KernelOpts = []
    ),
    append(BaseOpts, KernelOpts, Options),
    format(user_error, '[prof-matrix] Generating Go ~w (emit_mode=~w, no_kernels=~w)~n',
           [Label, EmitMode, NoKernels]),
    write_wam_go_project(Predicates, Options, OutputDir),
    format(user_error, '[prof-matrix] ~w -> ~w~n', [Label, OutputDir]).

python_configs([
    config('I-py-pure-interp',  interpreter, true),
    config('J-py-interp-ffi',   interpreter, false),
    config('K-py-lowered-only', functions,   true),
    config('L-py-lowered-ffi',  functions,   false)
]).

generate_python_config(OutputBase, Label, EmitMode, NoKernels, Predicates) :-
    format(atom(OutputDir), '~w/~w', [OutputBase, Label]),
    format(atom(ModName), 'wam-prof-~w', [Label]),
    BaseOpts = [
        module_name(ModName),
        prefer_wam(true),
        wam_fallback(true),
        emit_mode(EmitMode),
        parallel(true)
    ],
    (   NoKernels == true
    ->  KernelOpts = [no_kernels(true)]
    ;   KernelOpts = []
    ),
    append(BaseOpts, KernelOpts, Options),
    format(user_error, '[prof-matrix] Generating Python ~w (emit_mode=~w, no_kernels=~w)~n',
           [Label, EmitMode, NoKernels]),
    write_wam_python_project(Predicates, Options, OutputDir),
    format(user_error, '[prof-matrix] ~w -> ~w~n', [Label, OutputDir]).
