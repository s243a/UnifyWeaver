:- encoding(utf8).
%% Microbenchmark for the WAM R hybrid target. Compares two
%% fact-source backends:
%%
%%   * wam     -- facts asserted as Prolog clauses, compiled to WAM
%%                bytecode that runs through the R-side stepping
%%                engine.
%%   * foreign -- facts encoded into a generated R foreign handler
%%                that does a hash-lookup at call time.
%%
%% For each backend the benchmark times:
%%   gen          write_wam_r_project/3
%%   run          cold-start single-shot `Rscript` invocation
%%   inner_total  `--bench M` invocation; R startup paid once
%%                (per-iter = inner_total / M)
%%
%% The query is a single-row lookup `cp(c0, X)` against a chain of
%% N rows c0 -> c1 -> ... -> cN, which is intentionally cheap so
%% R startup dominates the run-phase number. The inner-loop number
%% reveals the real per-query cost difference between backends.
%%
%% Usage:
%%   swipl -g main -t halt tests/benchmarks/wam_r_fact_source_bench.pl
%%   swipl -g main -t halt tests/benchmarks/wam_r_fact_source_bench.pl -- 50
%%   swipl -g main -t halt tests/benchmarks/wam_r_fact_source_bench.pl -- 50 --inner 1000
%%
%% N defaults to [10, 50, 100]. Inner iterations default to 500.
%%
%% Skip behaviour: if Rscript isn't on PATH, prints a diagnostic and
%% exits cleanly. Honours WAM_R_BENCH_KEEP=1 to retain projects.

:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module('../../src/unifyweaver/targets/wam_r_target').
:- use_module(library(filesex), [directory_file_path/3, make_directory_path/1,
                                  delete_directory_and_contents/1]).
:- use_module(library(process)).
:- use_module(library(readutil)).

:- dynamic user:cp/2.

%% ========================================================================
%% Toolchain detection
%% ========================================================================

rscript_available :-
    catch(
        (   process_create(path('Rscript'), ['--version'],
                [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0))
        ),
        _, fail).

%% ========================================================================
%% Data generation
%% ========================================================================

%% chain_pairs(+N, -Pairs)
%  Builds N edges: c0->c1, c1->c2, ..., c(N-1)->cN.
chain_pairs(N, Pairs) :-
    numlist(0, N, Ns0),
    Ns0 = [_|Tail],
    chain_pairs_(Ns0, Tail, Pairs).

chain_pairs_([], _, []).
chain_pairs_([_|_], [], []).
chain_pairs_([X|Xs], [Y|Ys], [Cx-Cy | Rest]) :-
    format(atom(Cx), 'c~w', [X]),
    format(atom(Cy), 'c~w', [Y]),
    chain_pairs_(Xs, Ys, Rest).

all_atoms(Pairs, Atoms) :-
    findall(A, (member(X-Y, Pairs), member(A, [X, Y])), Bag),
    sort(Bag, Atoms).

%% ========================================================================
%% Backend setup
%% ========================================================================

setup_backend(wam, Pairs, _Atoms,
              [user:cp/2],
              [module_name('wam.r.bench')]) :-
    retractall(user:cp(_, _)),
    forall(member(X-Y, Pairs), assertz(user:cp(X, Y))).

setup_backend(foreign, Pairs, Atoms,
              [user:cp/2],
              [ module_name('wam.r.bench'),
                intern_atoms(Atoms),
                foreign_predicates([cp/2]),
                r_foreign_handlers([handler(cp/2, HandlerSrc)])
              ]) :-
    retractall(user:cp(_, _)),
    assertz((user:cp(_, _))),    % stub clause -- body replaced by foreign call
    foreign_handler_source(Pairs, HandlerSrc).

%% foreign_handler_source(+Pairs, -RSourceString)
%  Emit an R foreign handler whose body holds a static lookup table
%  built from Pairs. On call, derefs the first arg as an atom, looks
%  the name up in the table, and binds the second arg to the
%  successor's atom.
foreign_handler_source(Pairs, Src) :-
    pairs_to_r_assoc(Pairs, AssocStr),
    format(string(Src),
        'function(state, args, table) {\n  lookup <- list(~w)\n  v <- WamRuntime$deref(state, args[[1]])\n  if (is.null(v) || is.null(v$tag) || v$tag != "atom") return(list(ok = FALSE))\n  name <- WamRuntime$string_of(table, v$id)\n  if (!(name %in% names(lookup))) return(list(ok = FALSE))\n  list(ok = TRUE, bindings = list(list(idx = 2L, val = Atom(WamRuntime$intern(table, lookup[[name]])))))\n}',
        [AssocStr]).

pairs_to_r_assoc(Pairs, AssocStr) :-
    maplist(pair_to_r_entry, Pairs, Entries),
    atomic_list_concat(Entries, ', ', AssocStr).

pair_to_r_entry(X-Y, Entry) :-
    format(string(Entry), '"~w" = "~w"', [X, Y]).

%% ========================================================================
%% Run helpers
%% ========================================================================

run_rscript_query(RDir, PredKey, Args, Output) :-
    append(['generated_program.R', PredKey], Args, ProcArgs),
    process_create(path('Rscript'), ProcArgs,
                   [cwd(RDir), stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, OutStr), read_string(E, _, ErrStr),
    close(O), close(E),
    process_wait(Pid, exit(EC)),
    normalize_space(string(Output), OutStr),
    (   (EC =:= 0 ; EC =:= 1)
    ->  true
    ;   throw(error(rscript_run_failed(EC, PredKey, Args, ErrStr), _))
    ).

%% run_rscript_bench(+RDir, +N, +PredKey, +Args, -BenchSec)
%  Drives `Rscript --bench N pred/<arity> ...` and parses the
%  "BENCH n=<N> elapsed=<sec> last=<true|false>" line from stdout.
run_rscript_bench(RDir, N, PredKey, Args, BenchSec) :-
    atom_string(N, NStr),
    append(['generated_program.R', '--bench', NStr, PredKey], Args, ProcArgs),
    process_create(path('Rscript'), ProcArgs,
                   [cwd(RDir), stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, OutStr), read_string(E, _, ErrStr),
    close(O), close(E),
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
%% Bench driver
%% ========================================================================

time_bench(Backend, Pairs, Atoms, InnerN,
           GenSec, ColdRunSec, InnerLoopSec) :-
    bench_proj_path(Backend, ProjectDir),
    catch(delete_directory_and_contents(ProjectDir), _, true),
    setup_backend(Backend, Pairs, Atoms, Preds, Opts),
    get_time(T0),
    write_wam_r_project(Preds, Opts, ProjectDir),
    get_time(T1),
    directory_file_path(ProjectDir, 'R', RDir),
    run_rscript_query(RDir, 'cp/2', ['c0', 'c1'], Out),
    get_time(T2),
    run_rscript_bench(RDir, InnerN, 'cp/2', ['c0', 'c1'], InnerLoopSec),
    GenSec     is T1 - T0,
    ColdRunSec is T2 - T1,
    (   Out == "true"
    ->  true
    ;   format(user_error, "[FAIL] backend=~w expected \"true\" got ~q~n",
               [Backend, Out])
    ),
    cleanup_after(ProjectDir).

bench_proj_path(Backend, P) :-
    format(atom(Rel), '_tmp_wam_r_bench_~w', [Backend]),
    absolute_file_name(Rel, P).

cleanup_after(ProjectDir) :-
    (   getenv('WAM_R_BENCH_KEEP', "1")
    ->  true
    ;   catch(delete_directory_and_contents(ProjectDir), _, true)
    ).

run_one_size(N, InnerN) :-
    chain_pairs(N, Pairs),
    all_atoms(Pairs, Atoms),
    format("[INFO] N=~w rows, inner-loop=~w iterations per backend~n",
           [N, InnerN]),
    forall(
        member(Backend, [wam, foreign]),
        ( time_bench(Backend, Pairs, Atoms, InnerN, G, R, I),
          PerIter is I / InnerN,
          format("RESULT n=~w backend=~w gen=~3f run=~3f inner_total=~6f per_iter=~9f~n",
                 [N, Backend, G, R, I, PerIter])
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
inner_iterations(500).

main :-
    (   rscript_available
    ->  bench_sizes(Sizes),
        inner_iterations(InnerN),
        format("[INFO] WAM R fact-source bench -- sizes ~w, inner=~w~n",
               [Sizes, InnerN]),
        forall(member(N, Sizes), run_one_size(N, InnerN))
    ;   format("[SKIP] Rscript not on PATH; skipping bench.~n")
    ).
