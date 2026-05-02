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

%% Benchmark workload sizes per shape.
%  - nested_findall: K = number of inner_p facts. Per-branch cost
%    ~ K dispatch cycles + K aggregate_collect ops.
%  - arith: N = depth of recursive countdown via `iterate/1`. Per-
%    branch cost ~ N WAM-instruction dispatches + N is/2 + N >/2.
%    Each iteration costs more than a fact-only dispatch in the
%    nested_findall shape, so the crossover should be lower.
bench_workloads(nested_findall, [1, 10, 50, 200, 1000]).
bench_workloads(arith,          [1, 10, 50, 200, 1000]).

%% Shapes to run. Both run by default; comment out to skip.
bench_shapes([nested_findall, arith]).

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

%% Test fixtures: 8-clause Tier-2-eligible bench_p with two shape
%  variants — nested_findall (atom-only inner findall per branch)
%  and arith (recursive integer countdown per branch).

:- dynamic user:bench_p/1.
:- dynamic user:inner_p/1.
:- dynamic user:iterate/1.

:- dynamic user:bench/1.
user:bench(L) :- findall(X, bench_p(X), L).

%% iterate/1 stays the same across arith runs; only bench_p changes.
%  Pre-installed once at consult time.
user:iterate(0).
user:iterate(N) :- N > 0, N1 is N - 1, iterate(N1).

bench_predicates(nested_findall, [
    user:bench_p/1,
    user:inner_p/1,
    user:bench/1
]).
bench_predicates(arith, [
    user:bench_p/1,
    user:iterate/1,
    user:bench/1
]).

%% Tier-2 purity:
%  - nested_findall: bench_p (fans out) + inner_p (parallel_depth
%    gate keeps it sequential per branch, but it's still a Tier-2
%    candidate the static gate considers).
%  - arith: bench_p only. iterate/1 is order-DEPENDENT (recursive
%    countdown), so it must NOT be declared pure — it'd run
%    sequentially in any case (called from within a branch where
%    parallel_depth=1 already gates the gate).
bench_purity_decls(nested_findall, [user:bench_p/1, user:inner_p/1]).
bench_purity_decls(arith,          [user:bench_p/1]).

%% install_workload(Shape, Size) — set up the dynamic predicates.
install_workload(nested_findall, K) :-
    retractall(user:bench_p(_)),
    retractall(user:inner_p(_)),
    forall(between(1, K, I), (
        format(atom(A), 'i_~w', [I]),
        assertz(user:inner_p(A))
    )),
    forall(member(C, ['a','b','c','d','e','f','g','h']),
           assertz((user:bench_p(C) :- findall(_, user:inner_p(_), _)))).
install_workload(arith, N) :-
    retractall(user:bench_p(_)),
    forall(member(C, ['a','b','c','d','e','f','g','h']),
           assertz((user:bench_p(C) :- user:iterate(N)))).

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

unique_project_dir(Shape, Mode, Work, Dir) :-
    writable_tmp_root(Root),
    get_time(T),
    Stamp is floor(T * 1000),
    format(atom(Name), 'uw_elixir_tier2_bench_~w_~w_w~w_~w',
           [Shape, Mode, Work, Stamp]),
    directory_file_path(Root, Name, Dir).

%% Generate one project for one (shape, mode, work) combination.
write_bench_project(Shape, Mode, Work, ProjectDir) :-
    install_workload(Shape, Work),
    bench_predicates(Shape, Predicates),
    bench_purity_decls(Shape, PurityDecls),
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
    write_bench_driver(ProjectDir, Shape, Mode, Work),
    cleanup_purity_decls(Shape).

base_options(Base, parallel,   Base) :- !.
base_options(Base, sequential, [intra_query_parallel(false) | Base]).

cleanup_purity_decls(Shape) :-
    bench_purity_decls(Shape, PurityDecls),
    forall(member(Pred, PurityDecls),
           ignore(retract(clause_body_analysis:order_independent(Pred)))).

%% Driver: K timed runs, one BENCH line per run. The shape tag is
%  emitted into each line so the parser can group by it.
write_bench_driver(ProjectDir, Shape, Mode, Work) :-
    bench_runs(Runs),
    directory_file_path(ProjectDir, 'smoke_driver.exs', DriverPath),
    bench_predicates(Shape, Predicates),
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
    format(S, '  IO.puts("BENCH shape=~w mode=~w work=~w run=#{i} elapsed_us=#{us}")~n',
           [Shape, Mode, Work]),
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
%  for the given (Shape, Mode, Work) tuple.
parse_bench_us(StdOut, Shape, Mode, Work, Timings) :-
    split_string(StdOut, "\n", "\n", Lines),
    format(string(Prefix), "BENCH shape=~w mode=~w work=~w ",
           [Shape, Mode, Work]),
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

%% One (shape, mode, work) cell.
run_one(Shape, Mode, Work, MedUs) :-
    unique_project_dir(Shape, Mode, Work, Dir),
    setup_call_cleanup(
        write_bench_project(Shape, Mode, Work, Dir),
        (   run_bench_driver(Dir, StdOut, StdErr, ExitCode),
            (   ExitCode == 0
            ->  parse_bench_us(StdOut, Shape, Mode, Work, Timings),
                (   Timings == []
                ->  format(user_error,
                           'no BENCH lines for ~w/~w/~w; stderr=~w~n',
                           [Shape, Mode, Work, StdErr]),
                    fail
                ;   median(Timings, MedUs),
                    format('~w/~w  work=~12|~w~22|  median_us=~w~n',
                           [Shape, Mode, Work, MedUs])
                )
            ;   format(user_error, 'driver exit=~w; stderr=~w~n',
                       [ExitCode, StdErr]),
                fail
            )
        ),
        catch(delete_directory_and_contents(Dir), _, true)
    ).

%% Run the full grid (all shapes × workloads × modes) and print summary.
run_benchmark :-
    format('~n=== WAM-Elixir Tier-2 findall: parallel vs sequential ===~n~n'),
    (   elixir_available
    ->  bench_shapes(Shapes),
        forall(member(Shape, Shapes), (
            format('--- shape=~w ---~n', [Shape]),
            bench_workloads(Shape, Works),
            forall(member(W, Works), (
                run_one(Shape, parallel,   W, _),
                run_one(Shape, sequential, W, _)
            ))
        )),
        format('~n=== Done. Inspect rows above; speedup = seq_us / par_us. ===~n~n')
    ;   format('elixir not on PATH — benchmark skipped.~n')
    ).
