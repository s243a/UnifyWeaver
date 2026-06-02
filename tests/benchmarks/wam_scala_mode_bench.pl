:- encoding(utf8).
%% wam_scala_mode_bench.pl
%%
%% Microbenchmark comparing the three WAM Scala execution modes on the
%% same graph workload, to quantify the speedup from the per-predicate
%% lowered emitter (emit_mode(functions)) and the native graph kernels
%% (kernel_dispatch(true)) over the baseline step-loop interpreter.
%%
%%   * interp  — emit_mode default; every predicate runs in the step loop.
%%   * lowered — emit_mode(functions); deterministic clause-1 fast paths.
%%   * kernel  — kernel_dispatch(true); tc/2 lowered to a native BFS handler.
%%
%% Workload: transitive_closure tc/2 over a chain c0 -> c1 -> ... -> cN.
%%   tc(X,Y) :- edge(X,Y).
%%   tc(X,Y) :- edge(X,Z), tc(Z,Y).
%% Query: tc(c0, cN) — deep reachability across the whole chain.
%%
%% For each mode the project is generated, compiled once with scalac, then
%% run with `--bench INNER` so JVM startup is amortised and the reported
%% elapsed is dominated by per-query runtime cost. Prints a table of
%% per-iteration time and speedup vs the interpreter.
%%
%% Usage:
%%   swipl -g main -t halt tests/benchmarks/wam_scala_mode_bench.pl
%%   swipl -g main -t halt tests/benchmarks/wam_scala_mode_bench.pl -- 100 2000
%%     (chain size N = 100, inner iterations = 2000)
%%
%% Skips cleanly (exit 0) if scalac/scala aren't on PATH. Honours
%% SCALA_BENCH_KEEP=1 to retain the generated projects.

:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module('../../src/unifyweaver/targets/wam_scala_target').
:- use_module(library(filesex), [directory_file_path/3, make_directory_path/1,
                                  delete_directory_and_contents/1,
                                  directory_member/3]).
:- use_module(library(process)).
:- use_module(library(readutil)).

:- dynamic user:edge/2.
:- dynamic user:tc/2.

% ========================================================================
% Toolchain detection
% ========================================================================

tool_runs(Path, Args) :-
    catch((process_create(path(Path), Args,
              [stdout(null), stderr(null), process(Pid)]),
           process_wait(Pid, exit(0))), _, fail).

scalac_available :- tool_runs(scalac, ['--version']).
scala_available  :- tool_runs(scala,  ['--version']).

% ========================================================================
% Workload
% ========================================================================

%% setup_chain(+N) — assert edge(c0,c1) .. edge(c(N-1),cN) and the tc rules.
setup_chain(N) :-
    retractall(user:edge(_, _)),
    retractall(user:tc(_, _)),
    N1 is N - 1,
    forall(between(0, N1, I),
           ( J is I + 1,
             atom_concat(c, I, Ci),
             atom_concat(c, J, Cj),
             assertz(user:edge(Ci, Cj)) )),
    assertz((user:tc(X, Y) :- user:edge(X, Y))),
    assertz((user:tc(X, Y) :- user:edge(X, Z), user:tc(Z, Y))).

pkg('gen.modebench.core').

mode_opts(interp, Opts) :-
    pkg(P), Opts = [package(P), runtime_package(P), module_name('modebench')].
mode_opts(lowered, Opts) :-
    pkg(P), Opts = [package(P), runtime_package(P), module_name('modebench'),
                    emit_mode(functions)].
mode_opts(kernel, Opts) :-
    pkg(P), Opts = [package(P), runtime_package(P), module_name('modebench'),
                    kernel_dispatch(true)].

mode_dir(Mode, Dir) :-
    format(atom(Rel), '_tmp_modebench_~w', [Mode]),
    absolute_file_name(Rel, Dir).

% ========================================================================
% Compile / run
% ========================================================================

compile_project(Dir) :-
    absolute_file_name(Dir, Abs),
    directory_file_path(Abs, 'classes', ClassDir),
    make_directory_path(ClassDir),
    find_scala_sources(Abs, Sources),
    Sources \= [],
    process_create(path(scalac), ['-d', ClassDir | Sources],
        [cwd(Abs), stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, _), read_string(E, _, Err), close(O), close(E),
    process_wait(Pid, exit(EC)),
    ( EC =:= 0 -> true ; throw(error(scala_compile_failed(EC, Err), _)) ).

find_scala_sources(Abs, Sources) :-
    directory_file_path(Abs, 'src', SrcDir),
    findall(F,
        ( directory_member(SrcDir, RelF, [extensions([scala]), recursive(true)]),
          directory_file_path(SrcDir, RelF, F) ),
        Sources).

%% run_bench(+Dir, +Inner, +PredKey, +Args, -ElapsedSec)
run_bench(Dir, Inner, PredKey, Args, Sec) :-
    absolute_file_name(Dir, Abs),
    directory_file_path(Abs, 'classes', ClassDir),
    pkg(P), atom_concat(P, '.GeneratedProgram', Main),
    atom_string(PredKey, PStr), atom_string(Inner, InnerStr),
    maplist([A, S]>>atom_string(A, S), Args, AStrs),
    append(['-classpath', ClassDir, Main, '--bench', InnerStr, PStr],
           AStrs, ProcArgs),
    process_create(path(scala), ProcArgs,
        [cwd(Abs), stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, Out), read_string(E, _, Err), close(O), close(E),
    process_wait(Pid, exit(EC)),
    ( EC =:= 0 -> parse_bench_line(Out, Sec)
    ; throw(error(scala_bench_failed(EC, Err), _)) ).

parse_bench_line(Str, Seconds) :-
    split_string(Str, "\n", "", Lines),
    member(Line, Lines),
    sub_string(Line, _, _, _, "BENCH"),
    sub_string(Line, B, _, _, "elapsed="),
    Bp is B + 8,
    sub_string(Line, Bp, _, 0, Tail),
    split_string(Tail, " \n", "", [SecStr | _]),
    number_string(Seconds, SecStr), !.

% ========================================================================
% Driver
% ========================================================================

bench_mode(Mode, _N, Inner, Queries, PerIters) :-
    mode_opts(Mode, Opts),
    mode_dir(Mode, Dir),
    catch(delete_directory_and_contents(Dir), _, true),
    write_wam_scala_project([user:tc/2, user:edge/2], Opts, Dir),
    compile_project(Dir),
    findall(PerIter,
            ( member(Args, Queries),
              run_bench(Dir, Inner, 'tc/2', Args, Sec),
              PerIter is Sec / Inner ),
            PerIters),
    ( getenv('SCALA_BENCH_KEEP', "1") -> true
    ; catch(delete_directory_and_contents(Dir), _, true) ).

main :-
    ( scalac_available, scala_available
    -> true
    ; format("SKIP: scalac/scala not on PATH; mode benchmark skipped.~n", []),
      halt(0) ),
    current_prolog_flag(argv, Argv),
    ( Argv = [NA, IA | _], atom_number(NA, N), atom_number(IA, Inner)
    -> true ; N = 100, Inner = 2000 ),
    setup_chain(N),
    atom_concat(c, N, CN),
    % Two queries: a deep recursive reachability (graph-algorithm shape,
    % where the kernel wins and the recursion-heavy lowered path falls
    % back), and a base-case direct edge (which the lowered fast path
    % handles without falling back).
    Queries = [ ['c0', CN], ['c0', 'c1'] ],
    format("WAM Scala mode benchmark~n", []),
    format("  workload: transitive_closure tc/2 over a chain of ~w nodes~n", [N]),
    format("  inner iterations per query: ~w~n", [Inner]),
    format("  queries: deep = tc(c0, c~w)   base = tc(c0, c1)~n~n", [N]),
    bench_mode(interp,  N, Inner, Queries, [DiInterp, BaInterp]),
    bench_mode(lowered, N, Inner, Queries, [DiLowered, BaLowered]),
    bench_mode(kernel,  N, Inner, Queries, [DiKernel, BaKernel]),
    format("query: deep = tc(c0, c~w)~n", [N]),
    report_row('mode', 'per_iter_sec', 'speedup'),
    report_data('interp',  DiInterp,  DiInterp),
    report_data('lowered', DiLowered, DiInterp),
    report_data('kernel',  DiKernel,  DiInterp),
    nl,
    format("query: base = tc(c0, c1)~n", []),
    report_row('mode', 'per_iter_sec', 'speedup'),
    report_data('interp',  BaInterp,  BaInterp),
    report_data('lowered', BaLowered, BaInterp),
    report_data('kernel',  BaKernel,  BaInterp).

report_row(A, B, C) :-
    format("  ~w~t~14|~w~t~32|~w~n", [A, B, C]).
report_data(Mode, PerIter, Baseline) :-
    Speed is Baseline / PerIter,
    format("  ~w~t~14|~6f~t~32|~2fx~n", [Mode, PerIter, Speed]).

