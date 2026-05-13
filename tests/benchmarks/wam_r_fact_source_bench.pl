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

%% run_rscript_profile(+RDir, +N, +PredKey, +Args, -ElapsedSec, -ProfileText)
%  Drives a small profile_runner.R (written into RDir on demand) that
%  source()s generated_program.R, runs the predicate N times under
%  Rprof, then prints summaryRprof()$by.self for the hottest fns plus
%  a single PROFILE elapsed=<sec> line. Returns the elapsed seconds
%  and the full multi-line profile digest.
run_rscript_profile(RDir, N, PredKey, Args, ElapsedSec, ProfileText) :-
    write_profile_runner(RDir),
    atom_string(N, NStr),
    ProcArgs = ['profile_runner.R', NStr, PredKey | Args],
    process_create(path('Rscript'), ProcArgs,
                   [cwd(RDir), stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, OutStr), read_string(E, _, ErrStr),
    close(O), close(E),
    process_wait(Pid, exit(EC)),
    (   (EC =:= 0 ; EC =:= 1)
    ->  parse_profile_output(OutStr, ElapsedSec, ProfileText)
    ;   throw(error(rscript_profile_failed(EC, PredKey, Args, ErrStr), _))
    ).

parse_profile_output(Str, Seconds, ProfileText) :-
    split_string(Str, "\n", "", Lines),
    (   member(Line, Lines),
        sub_string(Line, _, _, _, "PROFILE"),
        sub_string(Line, B, _, _, "elapsed="),
        Bp is B + 8,
        sub_string(Line, Bp, _, 0, Tail),
        split_string(Tail, " \n", "", [SecStr | _]),
        number_string(Seconds, SecStr)
    ->  true
    ;   Seconds = 0.0
    ),
    ProfileText = Str.

%% write_profile_runner(+RDir)
%  Writes profile_runner.R into RDir (idempotent). The runner sources
%  generated_program.R (whose main entrypoint is gated by
%  sys.nframe() == 0L, so source-from-script is a no-op), parses CLI
%  args the same way the main entrypoint does, then wraps a fixed-N
%  loop with Rprof() + summaryRprof().
write_profile_runner(RDir) :-
    directory_file_path(RDir, 'profile_runner.R', RunnerPath),
    (   exists_file(RunnerPath)
    ->  true
    ;   profile_runner_source(Src),
        setup_call_cleanup(
            open(RunnerPath, write, Stream),
            write(Stream, Src),
            close(Stream))
    ).

profile_runner_source(Src) :-
    Src =
"# Auto-generated by wam_r_fact_source_bench.pl. Source the generated
# program (its main block is gated by sys.nframe() == 0L so source-
# from-script is a no-op), resolve the requested predicate, then loop
# N times under Rprof and print a digest of the top self-time
# hotspots.
local({
  argv <- commandArgs(trailingOnly = TRUE)
  if (length(argv) < 2L) {
    cat('usage: Rscript profile_runner.R <N> <pred>/<arity> [args...]\\n')
    quit(status = 1L)
  }
  N <- suppressWarnings(as.integer(argv[1L]))
  if (is.na(N) || N <= 0L) {
    cat('invalid N\\n'); quit(status = 1L)
  }
  pred_arity <- argv[2L]
  raw_args <- if (length(argv) > 2L) argv[-(1:2)] else character(0)
  source('generated_program.R', local = TRUE)
  start_pc <- shared_labels[[pred_arity]]
  if (is.null(start_pc)) {
    cat('unknown predicate: ', pred_arity, '\\n', sep = '')
    quit(status = 1L)
  }
  parse_state <- WamRuntime$new_state()
  WamRuntime$promote_regs(parse_state)
  parse_vars <- new.env(parent = emptyenv())
  parsed <- lapply(raw_args, function(a) {
    toks <- WamRuntime$tokenize_term(a)
    if (!is.null(toks) && length(toks) > 0L) {
      p <- new.env(parent = emptyenv())
      p$tokens <- toks
      p$pos    <- 1L
      p$table  <- intern_table
      p$state  <- parse_state
      p$vars   <- parse_vars
      term <- WamRuntime$wam_parse_expr(p, 1200L)
      if (!is.null(term) && p$pos > length(toks)) return(term)
    }
    n <- suppressWarnings(as.integer(a))
    if (!is.na(n) && as.character(n) == a) IntTerm(n)
    else Atom(WamRuntime$intern(intern_table, a))
  })
  warmup <- min(50L, max(1L, as.integer(N %/% 20L)))
  for (i in seq_len(warmup)) {
    WamRuntime$run_predicate(shared_program, start_pc, parsed)
  }
  prof_file <- tempfile(fileext = '.Rprof')
  Rprof(prof_file, interval = 0.005, line.profiling = FALSE)
  t0 <- proc.time()[['elapsed']]
  for (i in seq_len(N)) {
    WamRuntime$run_predicate(shared_program, start_pc, parsed)
  }
  elapsed <- proc.time()[['elapsed']] - t0
  Rprof(NULL)
  cat(sprintf('PROFILE elapsed=%.6f n=%d\\n', elapsed, N))
  s <- summaryRprof(prof_file)
  by_self <- s$by.self
  if (!is.null(by_self) && nrow(by_self) > 0L) {
    keep <- min(nrow(by_self), 25L)
    cat('TOP_BY_SELF\\n')
    cat(sprintf('  %-44s %8s %6s %8s %6s\\n',
                'function', 'self.s', '%self', 'total.s', '%tot'))
    for (i in seq_len(keep)) {
      nm <- rownames(by_self)[i]
      r <- by_self[i, ]
      cat(sprintf('  %-44s %8.3f %5.1f%% %8.3f %5.1f%%\\n',
                  substr(nm, 1L, 44L),
                  r$self.time, r$self.pct,
                  r$total.time, r$total.pct))
    }
  }
  unlink(prof_file)
})
".

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
    ->  inner_iterations(InnerN),
        (   profile_mode_enabled(N)
        ->  format("[INFO] WAM R fact-source profile -- N=~w rows, inner=~w~n",
                   [N, InnerN]),
            run_profile(N, InnerN)
        ;   bench_sizes(Sizes),
            format("[INFO] WAM R fact-source bench -- sizes ~w, inner=~w~n",
                   [Sizes, InnerN]),
            forall(member(N, Sizes), run_one_size(N, InnerN))
        )
    ;   format("[SKIP] Rscript not on PATH; skipping bench.~n")
    ).

%% --profile [N]: profile the WAM backend at chain size N (default
%  100) under Rprof. Builds a single project, drives it via the
%  profile_runner.R, and emits the elapsed time plus the top-25
%  hotspots by self-time. The foreign backend isn't profiled here
%  (the per-call cost is already a single hash lookup; the
%  interesting hot path is in the WAM stepping engine).
profile_mode_enabled(N) :-
    current_prolog_flag(argv, Argv),
    member('--profile', Argv),
    (   append(_, ['--profile', NStr | _], Argv),
        atom_number(NStr, N0),
        integer(N0), N0 > 0
    ->  N = N0
    ;   N = 100
    ).

run_profile(N, InnerN) :-
    chain_pairs(N, Pairs),
    all_atoms(Pairs, Atoms),
    bench_proj_path(profile, ProjectDir),
    catch(delete_directory_and_contents(ProjectDir), _, true),
    setup_backend(wam, Pairs, Atoms, Preds, Opts),
    write_wam_r_project(Preds, Opts, ProjectDir),
    directory_file_path(ProjectDir, 'R', RDir),
    run_rscript_profile(RDir, InnerN, 'cp/2', ['c0', 'c1'],
                        ElapsedSec, ProfileText),
    format("~n", []),
    format("PROFILE n=~w inner=~w elapsed=~6f~n",
           [N, InnerN, ElapsedSec]),
    format("~n", []),
    write(ProfileText), nl,
    cleanup_after(ProjectDir).
