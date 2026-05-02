:- encoding(utf8).
%% Microbenchmark for the WAM Scala hybrid target's three fact-source
%% backends:
%%
%%   * wam     — facts compiled to WAM bytecode (asserted as Prolog
%%               clauses, then run through compile_predicate_to_wam/3).
%%   * inline  — facts passed declaratively as scala_fact_sources(...)
%%               with inline tuple lists; codegen synthesises a
%%               ForeignHandler that returns ForeignMulti.
%%   * file    — same as inline, but tuples are written to a CSV file
%%               and read at runtime by the generated handler.
%%
%% For each backend the benchmark times three pipeline phases:
%%   project generation  (write_wam_scala_project/3)
%%   scalac compilation  (process_create scalac ...)
%%   first query run     (process_create scala ...)
%%
%% The query is a single-row lookup category_parent(c0, P) and is
%% intentionally cheap so JVM startup + classpath + project compile
%% dominate the run-phase number — these are the costs a real user
%% pays per project, and they show how the three backends compare on
%% the cold-start path.
%%
%% Usage:
%%   swipl -g main -t halt tests/benchmarks/wam_scala_fact_source_bench.pl
%%   swipl -g main -t halt tests/benchmarks/wam_scala_fact_source_bench.pl -- 50
%%
%% N defaults to 25 (number of category_parent rows). The fact set is
%% c0->c1, c1->c2, ..., c(N-1)->cN — a simple chain.
%%
%% Skip behaviour:
%%   If scalac/scala aren't on PATH, prints a diagnostic and exits
%%   cleanly. Honours SCALA_BENCH_KEEP=1 to retain generated projects.

:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module('../../src/unifyweaver/targets/wam_scala_target').
:- use_module(library(filesex), [directory_file_path/3, make_directory_path/1,
                                  delete_directory_and_contents/1]).
:- use_module(library(process)).
:- use_module(library(readutil)).

:- dynamic user:category_parent/2.

%% ========================================================================
%% Toolchain detection
%% ========================================================================

tool_runs(Path, Args) :-
    catch(
        (   process_create(path(Path), Args,
                [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0))
        ),
        _, fail).

scalac_available :- tool_runs(scalac, ['--version']).
scala_available  :- tool_runs(scala,  ['--version']).

%% ========================================================================
%% Data generation
%% ========================================================================

%% chain_tuples(+N, -Tuples)
%  Builds N rows: [[c0,c1], [c1,c2], ..., [c(N-1), cN]].
chain_tuples(N, Tuples) :-
    numlist(0, N, Ns0),
    Ns0 = [_|Tail],
    pairs_(Ns0, Tail, Tuples).

pairs_([], _, []).
pairs_([_|_], [], []).
pairs_([X|Xs], [Y|Ys], [[Cx, Cy] | Rest]) :-
    format(atom(Cx), 'c~w', [X]),
    format(atom(Cy), 'c~w', [Y]),
    pairs_(Xs, Ys, Rest).

%% all_atoms(+Tuples, -Atoms)
%  Flatten tuple lists and dedup so we can pass them via intern_atoms.
all_atoms(Tuples, Atoms) :-
    findall(A, (member(T, Tuples), member(A, T)), Bag),
    sort(Bag, Atoms).

%% ========================================================================
%% Backend setup
%% ========================================================================

%% setup_backend(+Backend, +Tuples, +Atoms, +CsvPath, -Predicates, -Options)
%
%  For each backend, prepares the world:
%    - retracts any prior category_parent/2 clauses;
%    - for `wam`, asserts the tuples as Prolog clauses;
%    - for `inline` / `file`, asserts a single stub clause and provides
%      scala_fact_sources entries.
setup_backend(wam, Tuples, _Atoms, _Csv,
              [user:category_parent/2],
              [ package('generated.wam_scala_bench.core'),
                runtime_package('generated.wam_scala_bench.core'),
                module_name('wam-scala-bench')
              ]) :-
    retractall(user:category_parent(_, _)),
    forall(member([X, Y], Tuples), assertz(user:category_parent(X, Y))).

setup_backend(inline, Tuples, Atoms, _Csv,
              [user:category_parent/2],
              [ package('generated.wam_scala_bench.core'),
                runtime_package('generated.wam_scala_bench.core'),
                module_name('wam-scala-bench'),
                intern_atoms(Atoms),
                scala_fact_sources([source(category_parent/2, Tuples)])
              ]) :-
    retractall(user:category_parent(_, _)),
    assertz((user:category_parent(_, _))).

setup_backend(file, Tuples, Atoms, Csv,
              [user:category_parent/2],
              [ package('generated.wam_scala_bench.core'),
                runtime_package('generated.wam_scala_bench.core'),
                module_name('wam-scala-bench'),
                intern_atoms(Atoms),
                scala_fact_sources([source(category_parent/2, file(Csv))])
              ]) :-
    write_facts_csv(Csv, Tuples),
    retractall(user:category_parent(_, _)),
    assertz((user:category_parent(_, _))).

write_facts_csv(Path, Tuples) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        forall(member(Row, Tuples),
               ( atomic_list_concat(Row, ',', Line),
                 format(Stream, '~w~n', [Line]) )),
        close(Stream)).

%% ========================================================================
%% Compile + run
%% ========================================================================

compile_scala_project(ProjectDir) :-
    absolute_file_name(ProjectDir, Abs),
    directory_file_path(Abs, 'classes', ClassDir),
    make_directory_path(ClassDir),
    find_scala_sources(Abs, Sources),
    Sources \= [],
    process_create(path(scalac),
                   ['-d', ClassDir | Sources],
                   [cwd(Abs), stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, _), read_string(E, _, ErrStr),
    close(O), close(E),
    process_wait(Pid, exit(EC)),
    (   EC =:= 0
    ->  true
    ;   throw(error(scala_compile_failed(EC, ErrStr), _))
    ).

find_scala_sources(Abs, Sources) :-
    directory_file_path(Abs, 'src', Src),
    findall(F,
        ( directory_member(Src, R, [extensions([scala]), recursive(true)]),
          directory_file_path(Src, R, F)
        ),
        Sources).

run_scala_query(ProjectDir, PredKey, Args, Output) :-
    absolute_file_name(ProjectDir, Abs),
    directory_file_path(Abs, 'classes', ClassDir),
    atom_string(PredKey, P),
    maplist([A, S]>>atom_string(A, S), Args, AStrs),
    append(['-classpath', ClassDir,
            'generated.wam_scala_bench.core.GeneratedProgram',
            P], AStrs, ProcArgs),
    process_create(path(scala), ProcArgs,
                   [cwd(Abs), stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, OutStr), read_string(E, _, ErrStr),
    close(O), close(E),
    process_wait(Pid, exit(EC)),
    normalize_space(string(Output), OutStr),
    (   EC =:= 0
    ->  true
    ;   throw(error(scala_run_failed(EC, PredKey, Args, ErrStr), _))
    ).

%% run_scala_bench(+ProjectDir, +Iterations, +PredKey, +Args, -BenchSec)
%  Runs the generated program with `--bench N` so the JVM amortises
%  startup over N iterations of the same query. Parses the
%  "BENCH n=<N> elapsed=<sec>" line from stdout to extract elapsed.
run_scala_bench(ProjectDir, N, PredKey, Args, BenchSec) :-
    absolute_file_name(ProjectDir, Abs),
    directory_file_path(Abs, 'classes', ClassDir),
    atom_string(PredKey, P),
    atom_string(N, NStr),
    maplist([A, S]>>atom_string(A, S), Args, AStrs),
    append(['-classpath', ClassDir,
            'generated.wam_scala_bench.core.GeneratedProgram',
            '--bench', NStr, P], AStrs, ProcArgs),
    process_create(path(scala), ProcArgs,
                   [cwd(Abs), stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, OutStr), read_string(E, _, ErrStr),
    close(O), close(E),
    process_wait(Pid, exit(EC)),
    (   EC =:= 0
    ->  parse_bench_line(OutStr, BenchSec)
    ;   throw(error(scala_bench_failed(EC, PredKey, Args, ErrStr), _))
    ).

%% parse_bench_line(+String, -Seconds)
%  Pulls the elapsed= field out of a "BENCH n=N elapsed=X.XXX last=..." line.
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
%% Bench driver
%% ========================================================================

%% time_bench(+Backend, +Tuples, +Atoms, +InnerN,
%%            -GenSec, -CompileSec, -ColdRunSec, -InnerLoopSec)
%  Builds the project once and times:
%    GenSec       — write_wam_scala_project/3
%    CompileSec   — scalac compile
%    ColdRunSec   — single-shot scala invocation (cold start)
%    InnerLoopSec — `--bench InnerN` invocation; elapsed reported by the
%                   generated program itself, so JVM startup is excluded.
%                   The per-iteration cost is InnerLoopSec / InnerN.
time_bench(Backend, Tuples, Atoms, InnerN,
           GenSec, CompileSec, ColdRunSec, InnerLoopSec) :-
    bench_csv_path(Csv),
    bench_proj_path(Backend, ProjectDir),
    catch(delete_directory_and_contents(ProjectDir), _, true),
    setup_backend(Backend, Tuples, Atoms, Csv, Preds, Opts),
    get_time(T0),
    write_wam_scala_project(Preds, Opts, ProjectDir),
    get_time(T1),
    compile_scala_project(ProjectDir),
    get_time(T2),
    run_scala_query(ProjectDir, 'category_parent/2', ['c0', 'c1'], Out),
    get_time(T3),
    run_scala_bench(ProjectDir, InnerN, 'category_parent/2',
                    ['c0', 'c1'], InnerLoopSec),
    GenSec     is T1 - T0,
    CompileSec is T2 - T1,
    ColdRunSec is T3 - T2,
    (   Out == "true"
    ->  true
    ;   format(user_error,
               "[FAIL] backend=~w expected \"true\" got ~q~n",
               [Backend, Out])
    ),
    cleanup_after(Backend, ProjectDir, Csv).

bench_csv_path(P) :-
    absolute_file_name('_tmp_bench_facts.csv', P).
bench_proj_path(Backend, P) :-
    format(atom(Rel), '_tmp_bench_proj_~w', [Backend]),
    absolute_file_name(Rel, P).

cleanup_after(_, ProjectDir, Csv) :-
    (   getenv('SCALA_BENCH_KEEP', "1")
    ->  true
    ;   catch(delete_directory_and_contents(ProjectDir), _, true),
        catch(delete_file(Csv), _, true)
    ).

run_one_size(N, InnerN) :-
    chain_tuples(N, Tuples),
    all_atoms(Tuples, Atoms),
    format("[INFO] N=~w rows, inner-loop=~w iterations per backend~n",
           [N, InnerN]),
    forall(
        member(Backend, [wam, inline, file]),
        ( time_bench(Backend, Tuples, Atoms, InnerN, G, C, R, I),
          PerIter is I / InnerN,
          format("RESULT n=~w backend=~w gen=~3f compile=~3f run=~3f inner_total=~6f per_iter=~9f~n",
                 [N, Backend, G, C, R, I, PerIter])
        )).

%% ========================================================================
%% Entry point
%% ========================================================================

bench_sizes([N]) :-
    current_prolog_flag(argv, Argv),
    append(_, ['--', NStr], Argv),
    atom_number(NStr, N),
    integer(N), N > 0, !.
bench_sizes([10, 50, 100]).

inner_iterations(I) :-
    current_prolog_flag(argv, Argv),
    append(_, ['--inner', IStr], Argv),
    atom_number(IStr, I),
    integer(I), I > 0, !.
inner_iterations(2000).

main :-
    (   scalac_available, scala_available
    ->  bench_sizes(Sizes),
        inner_iterations(InnerN),
        format("[INFO] WAM Scala fact-source bench — sizes ~w, inner=~w~n",
               [Sizes, InnerN]),
        forall(member(N, Sizes), run_one_size(N, InnerN))
    ;   format("[SKIP] scalac/scala not on PATH; skipping bench.~n")
    ).
