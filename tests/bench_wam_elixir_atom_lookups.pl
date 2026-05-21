:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% bench_wam_elixir_atom_lookups.pl
%
% Runtime baseline bench for the WAM-Elixir target. Mirrors the
% shape of the Rust category_ancestor workload that proved out the
% 7.9x atom-interning win
% (docs/design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md).
%
% Purpose: capture a NUMERIC baseline for atom-keyed fact lookups
% on Elixir BEFORE deciding whether atom interning is worth
% implementing. BEAM has small-binary optimization that may give
% us most of the win for free — only the bench can tell.
%
% Workload: chained parent/2 facts (parent(a0,a1), parent(a1,a2),
% ...) at multiple sizes, plus ancestor/2 recursive query.
% Reports wallclock per-invocation in microseconds.
%
% Usage:
%   swipl -g run_bench -t halt tests/bench_wam_elixir_atom_lookups.pl
%
% Gated on `elixir` being on PATH.

:- module(bench_wam_elixir_atom_lookups, [run_bench/0, run_eprof/0]).

:- use_module(library(lists)).
:- use_module(library(process)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_elixir_target').

% ============================================================
% Top-level
% ============================================================

run_bench :-
    (   elixir_on_path
    ->  true
    ;   format("SKIP: elixir not on PATH~n"), halt(0)
    ),
    format("~n================================================~n"),
    format("  WAM-Elixir atom-lookup baseline bench~n"),
    format("================================================~n"),
    % Small sizes first — verify the lowered emitter handles the
    % clause count before scaling up. If N=100 works, try larger.
    Sizes = [50, 100, 250, 500, 1000],
    Reps = 3,
    forall(member(N, Sizes), run_one(N, Reps)),
    format("~n================================================~n"),
    format("  Done~n"),
    format("================================================~n~n").

elixir_on_path :-
    catch(
        process_create(path(elixir), ['--version'],
                       [stdout(null), stderr(null), process(Pid)]),
        _, fail),
    process_wait(Pid, exit(0)).

% ============================================================
% Per-size run
% ============================================================

run_one(N, Reps) :-
    format("~n=== N=~w chain depth, ~w repetitions ===~n", [N, Reps]),
    setup_facts(N),
    catch(
        ( time_compile_and_run(N, Reps, MeanUs),
          format("  mean_us=~w~n", [MeanUs])
        ),
        Err,
        format("  ERROR: ~w~n", [Err])
    ),
    teardown_facts.

setup_facts(N) :-
    catch(abolish(user:parent/2), _, true),
    catch(abolish(user:ancestor/2), _, true),
    N1 is N - 1,
    forall(
        between(0, N1, I), (
            I1 is I + 1,
            atom_concat(a, I, From),
            atom_concat(a, I1, To),
            assertz(user:parent(From, To))
        )
    ),
    % Two-clause ancestor: direct parent + transitive.
    user:assertz((ancestor(X, Y) :- parent(X, Y))),
    user:assertz((ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y))).

teardown_facts :-
    catch(abolish(user:parent/2), _, true),
    catch(abolish(user:ancestor/2), _, true).

% ============================================================
% Compile, run, time
% ============================================================

time_compile_and_run(N, Reps, MeanUs) :-
    unique_tmp_dir('tmp_elixir_bench', TmpDir),
    BaseOpts = [ module_name('wam_elixir_bench'),
                 emit_mode(lowered),
                 source_module(user) ],
    Preds = [user:parent/2, user:ancestor/2],
    compile_preds(Preds, [], PredWamPairs),
    write_wam_elixir_project(PredWamPairs, BaseOpts, TmpDir),
    bench_driver_path(DriverSrc),
    directory_file_path(TmpDir, 'run_bench.exs', DriverDst),
    copy_file(DriverSrc, DriverDst),
    % Query: ancestor(a0, a<N>) — succeeds, walks the whole chain.
    atom_concat(a, N, LastAtom),
    atom_string(LastAtom, LastAtomStr),
    setup_call_cleanup(
        true,
        run_bench_driver(TmpDir, 'ancestor/2', Reps,
                         ['a0', LastAtomStr], MeanUs),
        delete_directory_and_contents(TmpDir)).

compile_preds([], _Opts, []).
compile_preds([Mod:Name/Arity | Rest], Opts,
              [Name/Arity-WamCode | RestPairs]) :-
    (   wam_target:compile_predicate_to_wam(Name/Arity, Opts, WamCode) -> true
    ;   wam_target:compile_predicate_to_wam(Mod:Name/Arity, Opts, WamCode) -> true
    ;   throw(error(wam_compile_failed(Mod:Name/Arity), _))
    ),
    compile_preds(Rest, Opts, RestPairs).

run_bench_driver(ProjectDir, PredKey, Reps, Args, MeanUs) :-
    absolute_file_name(ProjectDir, AbsProjectDir),
    atom_string(PredKey, PredStr),
    number_string(Reps, RepsStr),
    maplist([A, S]>>atom_string(A, S), Args, ArgStrs),
    append(['run_bench.exs', "WamElixirBench", PredStr, RepsStr],
           ArgStrs, ProcArgs),
    process_create(path(elixir), ProcArgs,
                   [cwd(AbsProjectDir),
                    stdout(pipe(Out)), stderr(pipe(Err)),
                    process(Pid)]),
    read_string(Out, _, OutStr),
    read_string(Err, _, ErrStr),
    close(Out),
    close(Err),
    process_wait(Pid, exit(ExitCode)),
    (   ExitCode =:= 0
    ->  parse_mean_us(OutStr, MeanUs)
    ;   throw(error(elixir_bench_failed(ExitCode, ErrStr), _))
    ).

parse_mean_us(Out, MeanUs) :-
    split_string(Out, "\n", " \t", Lines),
    member(Line, Lines),
    split_string(Line, " ", "", Tokens),
    member(Tok, Tokens),
    string_concat("mean_us=", ValStr, Tok),
    !,
    number_string(MeanUs, ValStr).

unique_tmp_dir(Prefix, TmpDir) :-
    get_time(T),
    Stamp is floor(T * 1000),
    (   getenv('TMPDIR', Base) -> true
    ;   Base = '/tmp'
    ),
    format(atom(TmpDir), '~w/~w_~w', [Base, Prefix, Stamp]).

:- (   prolog_load_context(directory, Dir),
       directory_file_path(Dir, 'elixir_e2e/run_bench.exs', P),
       assertz(bench_driver_path_fact(P))
   ;   true
   ).
bench_driver_path(P) :- bench_driver_path_fact(P).

:- (   prolog_load_context(directory, Dir2),
       directory_file_path(Dir2, 'elixir_e2e/run_eprof.exs', P2),
       assertz(eprof_driver_path_fact(P2))
   ;   true
   ).
eprof_driver_path(P) :- eprof_driver_path_fact(P).

% ============================================================
% Profiling entry point — captures :eprof per-function breakdown
% for a single query at fixed N.
% ============================================================

run_eprof :-
    (   elixir_on_path
    ->  true
    ;   format("SKIP: elixir not on PATH~n"), halt(0)
    ),
    N = 500,
    format("~n================================================~n"),
    format("  WAM-Elixir :eprof profile, N=~w~n", [N]),
    format("================================================~n~n"),
    setup_facts(N),
    catch(
        profile_query(N),
        Err,
        format("ERROR: ~w~n", [Err])
    ),
    teardown_facts.

profile_query(N) :-
    unique_tmp_dir('tmp_elixir_eprof', TmpDir),
    BaseOpts = [ module_name('wam_elixir_bench'),
                 emit_mode(lowered),
                 source_module(user) ],
    Preds = [user:parent/2, user:ancestor/2],
    compile_preds(Preds, [], PredWamPairs),
    write_wam_elixir_project(PredWamPairs, BaseOpts, TmpDir),
    eprof_driver_path(DriverSrc),
    directory_file_path(TmpDir, 'run_eprof.exs', DriverDst),
    copy_file(DriverSrc, DriverDst),
    atom_concat(a, N, LastAtom),
    atom_string(LastAtom, LastAtomStr),
    setup_call_cleanup(
        true,
        run_eprof_driver(TmpDir, 'ancestor/2', ['a0', LastAtomStr]),
        delete_directory_and_contents(TmpDir)).

run_eprof_driver(ProjectDir, PredKey, Args) :-
    absolute_file_name(ProjectDir, AbsProjectDir),
    atom_string(PredKey, PredStr),
    maplist([A, S]>>atom_string(A, S), Args, ArgStrs),
    append(['run_eprof.exs', "WamElixirBench", PredStr], ArgStrs, ProcArgs),
    process_create(path(elixir), ProcArgs,
                   [cwd(AbsProjectDir),
                    stdout(pipe(Out)), stderr(pipe(Err)),
                    process(Pid)]),
    read_string(Out, _, OutStr),
    read_string(Err, _, ErrStr),
    close(Out),
    close(Err),
    process_wait(Pid, exit(ExitCode)),
    format("~w~n", [OutStr]),
    (   ExitCode =:= 0
    ->  true
    ;   format("stderr: ~w~n", [ErrStr]),
        throw(error(elixir_eprof_failed(ExitCode), _))
    ).
