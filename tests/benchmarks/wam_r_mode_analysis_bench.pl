:- encoding(utf8).
%% Microbenchmark for WAM-R mode-analysis/lowered-emitter work.
%%
%% The benchmark builds small generated WAM-R projects and drives their
%% `Rscript --bench N` loop, so R startup is paid once per measurement.
%%
%% Cases:
%%   recursive_pn
%%     `pn(0). pn(N) :- N > 0, N1 is N - 1, pn(N1).`
%%     Compares emit_mode(interpreter) with emit_mode(functions). This
%%     measures the current lowered multi-clause + inline is/2 path as a
%%     whole, not one individual phase in isolation.
%%
%%   get_value_same
%%     `same(X, X)` under `:- mode(same(+,+))`.
%%     Compares emit_mode(functions) with and without mode_specialise(off),
%%     isolating the repeated-bound-head get_value fast path.
%%
%% Usage:
%%   swipl -g main -t halt tests/benchmarks/wam_r_mode_analysis_bench.pl
%%   swipl -g main -t halt tests/benchmarks/wam_r_mode_analysis_bench.pl -- --inner 1000 --depth 2000
%%
%% Defaults: --inner 500, --depth 500. Honours WAM_R_BENCH_KEEP=1.

:- use_module('../../src/unifyweaver/targets/wam_r_target').
:- use_module(library(filesex), [directory_file_path/3,
                                  delete_directory_and_contents/1]).
:- use_module(library(process)).
:- use_module(library(readutil)).

rscript_available :-
    catch(
        (   process_create(path('Rscript'), ['--version'],
                [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0))
        ),
        _, fail).

%% ========================================================================
%% Benchmark programs
%% ========================================================================

setup_recursive_pn :-
    retractall(user:mode(pn(_))),
    retractall(user:pn(_)),
    assertz(user:mode(pn(+))),
    assertz(user:pn(0)),
    assertz((user:pn(N) :-
        N > 0,
        N1 is N - 1,
        pn(N1))).

setup_get_value_same :-
    retractall(user:mode(same(_, _))),
    retractall(user:same(_, _)),
    assertz(user:mode(same(+, +))),
    assertz((user:same(X, X) :- true)).

cleanup_programs :-
    retractall(user:mode(pn(_))),
    retractall(user:pn(_)),
    retractall(user:mode(same(_, _))),
    retractall(user:same(_, _)).

variant_options(interpreter, [emit_mode(interpreter), fact_table_layout(off)]).
variant_options(functions, [emit_mode(functions), fact_table_layout(off)]).
variant_options(functions_no_specialise,
                [emit_mode(functions), fact_table_layout(off),
                 mode_specialise(off)]).

variant_label(interpreter, "interpreter").
variant_label(functions, "functions").
variant_label(functions_no_specialise, "functions_no_specialise").

%% ========================================================================
%% Rscript helpers
%% ========================================================================

run_rscript_bench(RDir, N, PredKey, Args, BenchSec) :-
    atom_string(N, NStr),
    append(['generated_program.R', '--bench', NStr, PredKey], Args, ProcArgs),
    process_create(path('Rscript'), ProcArgs,
                   [cwd(RDir), stdout(pipe(O)), stderr(pipe(E)),
                    process(Pid)]),
    read_string(O, _, OutStr),
    read_string(E, _, ErrStr),
    close(O),
    close(E),
    process_wait(Pid, exit(EC)),
    (   (EC =:= 0 ; EC =:= 1)
    ->  parse_bench_line(OutStr, BenchSec)
    ;   throw(error(rscript_bench_failed(EC, PredKey, Args, ErrStr), _))
    ).

parse_bench_line(Str, Seconds) :-
    split_string(Str, "\n", "", Lines),
    member(Line, Lines),
    sub_string(Line, _, _, _, "BENCH"),
    sub_string(Line, B, _, _, "elapsed="),
    Bp is B + 8,
    sub_string(Line, Bp, _, 0, Tail),
    split_string(Tail, " \n", "", [SecStr | _]),
    number_string(Seconds, SecStr), !.

%% ========================================================================
%% Project and timing helpers
%% ========================================================================

bench_proj_path(Case, Variant, P) :-
    format(atom(Rel), '_tmp_wam_r_mode_analysis_~w_~w', [Case, Variant]),
    absolute_file_name(Rel, P).

cleanup_after(ProjectDir) :-
    (   getenv('WAM_R_BENCH_KEEP', "1")
    ->  true
    ;   catch(delete_directory_and_contents(ProjectDir), _, true)
    ).

time_variant(Case, Variant, Setup, Preds, PredKey, Args, InnerN, Sec) :-
    bench_proj_path(Case, Variant, ProjectDir),
    catch(delete_directory_and_contents(ProjectDir), _, true),
    call(Setup),
    variant_options(Variant, Opts),
    write_wam_r_project(Preds, Opts, ProjectDir),
    directory_file_path(ProjectDir, 'R', RDir),
    run_rscript_bench(RDir, InnerN, PredKey, Args, Sec),
    cleanup_after(ProjectDir),
    cleanup_programs.

report_variant(Case, Variant, Sec, InnerN) :-
    variant_label(Variant, Label),
    PerIterUs is (Sec / InnerN) * 1_000_000,
    format("RESULT case=~w variant=~s inner_total=~6f per_iter_us=~3f~n",
           [Case, Label, Sec, PerIterUs]).

report_ratio(Case, NumerName, NumerSec, DenomName, DenomSec) :-
    Ratio is NumerSec / DenomSec,
    format("RATIO case=~w ~w_over_~w=~3f~n",
           [Case, NumerName, DenomName, Ratio]).

run_recursive_pn(InnerN, Depth) :-
    atom_string(Depth, DepthArg),
    time_variant(recursive_pn, interpreter, setup_recursive_pn,
                 [user:pn/1], 'pn/1', [DepthArg], InnerN, InterpSec),
    time_variant(recursive_pn, functions, setup_recursive_pn,
                 [user:pn/1], 'pn/1', [DepthArg], InnerN, FuncSec),
    report_variant(recursive_pn, interpreter, InterpSec, InnerN),
    report_variant(recursive_pn, functions, FuncSec, InnerN),
    report_ratio(recursive_pn, functions, FuncSec,
                 interpreter, InterpSec).

run_get_value_same(InnerN) :-
    time_variant(get_value_same, functions_no_specialise,
                 setup_get_value_same, [user:same/2], 'same/2',
                 ['alice', 'alice'], InnerN, OptOutSec),
    time_variant(get_value_same, functions, setup_get_value_same,
                 [user:same/2], 'same/2', ['alice', 'alice'],
                 InnerN, FuncSec),
    report_variant(get_value_same, functions_no_specialise,
                   OptOutSec, InnerN),
    report_variant(get_value_same, functions, FuncSec, InnerN),
    report_ratio(get_value_same, functions, FuncSec,
                 functions_no_specialise, OptOutSec).

%% ========================================================================
%% Entry point
%% ========================================================================

inner_iterations(I) :-
    current_prolog_flag(argv, Argv),
    append(_, ['--inner', IStr | _], Argv),
    atom_number(IStr, I),
    integer(I), I > 0, !.
inner_iterations(500).

recursion_depth(D) :-
    current_prolog_flag(argv, Argv),
    append(_, ['--depth', DStr | _], Argv),
    atom_number(DStr, D),
    integer(D), D >= 0, !.
recursion_depth(500).

main :-
    (   rscript_available
    ->  inner_iterations(InnerN),
        recursion_depth(Depth),
        format("[INFO] WAM-R mode-analysis bench -- inner=~w depth=~w~n",
               [InnerN, Depth]),
        run_recursive_pn(InnerN, Depth),
        run_get_value_same(InnerN)
    ;   format("[SKIP] Rscript not on PATH; skipping bench.~n")
    ).
