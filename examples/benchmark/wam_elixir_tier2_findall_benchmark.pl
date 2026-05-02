:- encoding(utf8).
% WAM-Elixir Tier-2 findall: parallel-vs-sequential benchmark.
%
% Goal: measure the wall-clock crossover where Tier-2 parallel
% findall starts to beat sequential. Phase 4 §6 risks #2 (snapshot
% cost) and #8 (determinism for measurement) and §10 ("performance
% tuning waits on measurements") all flag this as needed.
%
% Design:
%   - 8-clause Tier-2-eligible predicate `bench_p/1` whose clause
%     bodies each call a recursive countdown helper `iterate/1`.
%     The countdown depth is the per-branch workload knob.
%   - 8 branches > the Tier-2 forkMinBranches threshold of 3.
%   - Two Elixir projects compiled from the same source: one with
%     `intra_query_parallel(false)` (sequential), one without
%     (default = parallel; super-wrapper fans out via
%     Task.async_stream).
%   - Each project's driver runs `findall(X, bench_p(X), L)` K=10
%     times back-to-back, captures wall-clock per iteration via
%     :timer.tc/1, prints the median.
%
% Crossover signal: parallel wins when the per-branch cost
% amortises Task.async_stream's process-spawn + send-receive cost.
% At very small workloads parallel is slower (overhead dominates);
% at large enough workloads parallel approaches a max speedup of
% ~min(n_branches, n_schedulers).
%
% Output: BENCH lines on stdout, one per (mode, work, run-index).
% A summary row prints median(parallel) / median(sequential) and
% the speedup ratio.
%
% Usage:
%   swipl -g run_benchmark -t halt \
%     examples/benchmark/wam_elixir_tier2_findall_benchmark.pl
%
% Skipped gracefully when `elixir` is not on PATH.

:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module(library(option)).
:- use_module(library(lists)).
:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module('../../src/unifyweaver/targets/wam_elixir_target',
              [write_wam_elixir_project/3]).
:- use_module('../../src/unifyweaver/core/clause_body_analysis').

%% Benchmark workload sizes (per-branch iteration depth).
%  0 = no work beyond head match (overhead-only test).
%  Larger values exercise per-branch compute that should amortise
%  Task.async_stream overhead.
%% K = number of inner_p facts. Per-branch work ~ K dispatch cycles.
bench_workloads([1, 10, 50, 200, 1000]).

%% Number of timed runs per (mode, workload). Median is reported.
bench_runs(10).

%% PATH guard.
elixir_available :-
    catch(
        process_create(path(elixir), ['--version'],
                       [stdout(pipe(Out)), stderr(null), process(PID)]),
        _,
        fail),
    read_string(Out, _, _),
    close(Out),
    process_wait(PID, exit(0)).

%% Test fixture: 8-clause Tier-2-eligible bench_p, where each
%  clause body does a nested sequential findall over a K-solution
%  inner_p. Per-branch work scales linearly with K. We use atom-
%  only constants — no arithmetic — to dodge a separate WAM-Elixir
%  compiler issue where integer literals in clause heads are
%  emitted as Elixir strings ("0") instead of integers, breaking
%  numeric head-match.
%
%  Why this shape: the inner findall runs sequentially per branch
%  (the parallel_depth > 0 gate fires; this is the same pattern
%  Phase 4d covers). Outer parallel fans 8 branches; each branch's
%  inner findall enumerates K solutions. So per-branch cost ~ K
%  WamDispatcher.call cycles + K aggregate_collect ops.

:- dynamic user:bench_p/1.
:- dynamic user:inner_p/1.

:- dynamic user:bench/1.
user:bench(L) :- findall(X, bench_p(X), L).

bench_predicates([
    user:bench_p/1,
    user:inner_p/1,
    user:bench/1
]).

%% Tier-2 purity: bench_p (8 clauses, fans out) and inner_p (K
%  clauses, nested findall inside branches stays sequential due
%  to the parallel_depth gate, but inner_p still needs purity
%  for the static gate to consider it).
bench_purity_decls([
    user:bench_p/1,
    user:inner_p/1
]).

%% Generate K inner_p facts using K-character atom names (a, b, ...).
%  K small (say 20-200) keeps codegen fast while giving meaningful
%  per-branch work.
install_workload(K) :-
    retractall(user:bench_p(_)),
    retractall(user:inner_p(_)),
    forall(between(1, K, I), (
        format(atom(A), 'i_~w', [I]),
        assertz(user:inner_p(A))
    )),
    forall(member(C, ['a','b','c','d','e','f','g','h']),
           assertz((user:bench_p(C) :- findall(_, user:inner_p(_), _)))).

%% tmp_root — same fallback chain as Phase 4c.
tmp_root_candidate(Root) :-
    member(Env, ['TMPDIR', 'TMP', 'TEMP']),
    getenv(Env, Root),
    Root \== ''.
tmp_root_candidate(Root) :-
    getenv('PREFIX', Prefix),
    Prefix \== '',
    directory_file_path(Prefix, tmp, Root).
tmp_root_candidate('output').

writable_tmp_root(Root) :-
    tmp_root_candidate(Root),
    catch(make_directory_path(Root), _, fail),
    access_file(Root, write),
    !.

unique_project_dir(Mode, Work, Dir) :-
    writable_tmp_root(Root),
    get_time(T),
    Stamp is floor(T * 1000),
    format(atom(Name), 'uw_elixir_tier2_bench_~w_w~w_~w', [Mode, Work, Stamp]),
    directory_file_path(Root, Name, Dir).

%% Generate one project for one (mode, work) combination.
%  Mode: parallel | sequential.
write_bench_project(Mode, Work, ProjectDir) :-
    install_workload(Work),
    bench_predicates(Predicates),
    bench_purity_decls(PurityDecls),
    forall(member(Pred, PurityDecls),
           assertz(clause_body_analysis:order_independent(Pred))),
    findall(P/A-WamCode, (
        member(M:P/A, Predicates),
        wam_target:compile_predicate_to_wam(M:P/A, [], WamCode)
    ), PredWamPairs),
    base_options([
        module_name('tier2_bench'),
        emit_mode(lowered)
    ], Mode, Options),
    write_wam_elixir_project(PredWamPairs, Options, ProjectDir),
    write_bench_driver(ProjectDir, Mode, Work),
    cleanup_purity_decls.

base_options(Base, parallel,   Base) :- !.
base_options(Base, sequential, [intra_query_parallel(false) | Base]).

cleanup_purity_decls :-
    bench_purity_decls(PurityDecls),
    forall(member(Pred, PurityDecls),
           ignore(retract(clause_body_analysis:order_independent(Pred)))).

%% Driver: K timed runs, one BENCH line per run.
write_bench_driver(ProjectDir, Mode, Work) :-
    bench_runs(Runs),
    directory_file_path(ProjectDir, 'smoke_driver.exs', DriverPath),
    bench_predicates(Predicates),
    open(DriverPath, write, S),
    format(S, 'Code.require_file("lib/wam_runtime.ex", __DIR__)~n', []),
    format(S, 'Code.require_file("lib/wam_dispatcher.ex", __DIR__)~n', []),
    forall(member(_:P/_, Predicates), (
        atom_string(P, PStr),
        format(S, 'Code.require_file("lib/~w.ex", __DIR__)~n', [PStr])
    )),
    format(S, '~n', []),
    % Warmup: BEAM JIT + first-time module loads.
    format(S, '_warmup = Tier2Bench.Bench.run([{:unbound, make_ref()}])~n', []),
    format(S, 'for i <- 1..~w do~n', [Runs]),
    format(S, '  {us, _} = :timer.tc(fn ->~n', []),
    format(S, '    Tier2Bench.Bench.run([{:unbound, make_ref()}])~n', []),
    format(S, '  end)~n', []),
    format(S, '  IO.puts("BENCH mode=~w work=~w run=#{i} elapsed_us=#{us}")~n',
           [Mode, Work]),
    format(S, 'end~n', []),
    close(S).

%% Run `elixir smoke_driver.exs`, capture stdout.
run_bench_driver(ProjectDir, StdOut, StdErr, ExitCode) :-
    process_create(path(elixir),
                   ['smoke_driver.exs'],
                   [cwd(ProjectDir),
                    stdout(pipe(OutStream)),
                    stderr(pipe(ErrStream)),
                    process(PID)]),
    read_string(OutStream, _, StdOut),
    read_string(ErrStream, _, StdErr),
    close(OutStream),
    close(ErrStream),
    process_wait(PID, exit(ExitCode)).

%% Parse BENCH lines from stdout — return list of microsecond timings
%  for the given (Mode, Work) tuple.
parse_bench_us(StdOut, Mode, Work, Timings) :-
    split_string(StdOut, "\n", "\n", Lines),
    format(string(Prefix), "BENCH mode=~w work=~w ", [Mode, Work]),
    findall(Us,
            (member(Line, Lines),
             string_concat(Prefix, Tail, Line),
             sub_string(Tail, _, _, 0, Tail2),
             sub_string(Tail2, BeforeUs, _, _, "elapsed_us="),
             UsStart is BeforeUs + 11,
             string_length(Tail2, TailLen),
             UsLen is TailLen - UsStart,
             sub_string(Tail2, UsStart, UsLen, 0, UsStr),
             number_string(Us, UsStr)),
            Timings).

%% Median of a non-empty list of numbers.
median(Xs, Med) :-
    msort(Xs, Sorted),
    length(Sorted, N),
    N > 0,
    Mid is N // 2,
    (   N mod 2 =:= 1
    ->  nth0(Mid, Sorted, Med)
    ;   Lo is Mid - 1,
        nth0(Lo, Sorted, A),
        nth0(Mid, Sorted, B),
        Med is (A + B) / 2
    ).

%% One (mode, work) cell.
run_one(Mode, Work, MedUs) :-
    unique_project_dir(Mode, Work, Dir),
    setup_call_cleanup(
        write_bench_project(Mode, Work, Dir),
        (   run_bench_driver(Dir, StdOut, StdErr, ExitCode),
            (   ExitCode == 0
            ->  parse_bench_us(StdOut, Mode, Work, Timings),
                (   Timings == []
                ->  format(user_error, 'no BENCH lines for ~w/~w; stderr=~w~n',
                           [Mode, Work, StdErr]),
                    fail
                ;   median(Timings, MedUs),
                    format('~w  work=~6|~w~17|  median_us=~w~n', [Mode, Work, MedUs])
                )
            ;   format(user_error, 'driver exit=~w; stderr=~w~n',
                       [ExitCode, StdErr]),
                fail
            )
        ),
        catch(delete_directory_and_contents(Dir), _, true)
    ).

%% Run the full grid and print summary.
run_benchmark :-
    format('~n=== WAM-Elixir Tier-2 findall: parallel vs sequential ===~n~n'),
    (   elixir_available
    ->  bench_workloads(Works),
        forall(member(W, Works), (
            run_one(parallel,   W, _),
            run_one(sequential, W, _)
        )),
        format('~n=== Done. Inspect rows above; speedup = seq_us / par_us. ===~n~n')
    ;   format('elixir not on PATH — benchmark skipped.~n')
    ).
