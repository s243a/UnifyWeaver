:- encoding(utf8).
%% Microbenchmark for the \+ member(X, L) lowering.
%%
%% Generates one Haskell project with two predicates that have
%% IDENTICAL Prolog source — `bench_notmember_lowered/2` (with
%% `:- mode/1`) and `bench_notmember_unlowered/2` (no mode).
%% The mode declaration triggers the NotMemberList lowering;
%% the unannotated copy compiles to put_structure member/2 +
%% builtin_call \+/1.
%%
%% A custom Bench.hs (in tests/fixtures/wam_not_member_bench/)
%% imports the generated runtime and times each predicate across N
%% iterations on a 50-element visited list, alternating order to
%% reduce cache-warming bias.
%%
%% Usage:
%%   swipl -g main -t halt tests/benchmarks/wam_not_member_bench.pl [-- N]
%%   N defaults to 50000.
%%
%% Skip behaviour: same as wam_term_construction_bench.pl.
%% Honours WAM_NOT_MEMBER_BENCH_KEEP=1.

:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module('../../src/unifyweaver/targets/wam_haskell_target').
:- use_module(library(filesex), [directory_file_path/3, make_directory_path/1, copy_file/2]).
:- use_module(library(process)).
:- use_module(library(readutil)).

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

ghc_available   :- tool_runs(ghc,   ['--version']).
cabal_available :- tool_runs(cabal, ['--version']).

%% ========================================================================
%% Paths
%% ========================================================================

repo_root(Root) :-
    source_file(repo_root(_), This),
    file_directory_name(This, BenchDir),
    file_directory_name(BenchDir, TestsDir),
    file_directory_name(TestsDir, Root).

bench_fixture_path(Path) :-
    repo_root(Root),
    directory_file_path(Root,
        'tests/fixtures/wam_not_member_bench/Bench.hs', Path).

tmp_root(Root) :-
    (   getenv('TMPDIR', R0), R0 \== ''
    ->  Root = R0
    ;   Root = '/tmp'
    ).

build_dir(Dir) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_not_member_bench', Dir).

%% ========================================================================
%% Fixture
%% ========================================================================

:- dynamic user:bench_notmember_lowered/2.
:- dynamic user:bench_notmember_unlowered/2.
:- dynamic user:mode/1.

setup_fixture :-
    retractall(user:bench_notmember_lowered(_,_)),
    retractall(user:bench_notmember_unlowered(_,_)),
    retractall(user:mode(bench_notmember_lowered(_,_))),
    assert(user:mode(bench_notmember_lowered(+, +))),
    assert(user:(bench_notmember_lowered(X, V) :-
        \+ member(X, V))),
    assert(user:(bench_notmember_unlowered(X, V) :-
        \+ member(X, V))).

teardown_fixture :-
    retractall(user:bench_notmember_lowered(_,_)),
    retractall(user:bench_notmember_unlowered(_,_)),
    retractall(user:mode(bench_notmember_lowered(_,_))).

ensure_dir(D) :-
    (   exists_directory(D) -> true ; make_directory_path(D) ).

generate_project(Dir) :-
    ensure_dir(Dir),
    setup_fixture,
    catch(
        write_wam_haskell_project(
            [user:bench_notmember_lowered/2, user:bench_notmember_unlowered/2],
            [module_name('uw-not-member-bench'), use_hashmap(false)],
            Dir),
        E,
        (teardown_fixture, throw(E))),
    teardown_fixture.

write_bench_cabal(Dir) :-
    directory_file_path(Dir, 'uw-not-member-bench.cabal', CabalPath),
    Cabal = "cabal-version: 2.4\nname:          uw-not-member-bench\nversion:       0.1.0.0\nbuild-type:    Simple\n\nexecutable uw-not-member-bench\n  main-is:          Bench.hs\n  hs-source-dirs:   src\n  other-modules:    WamTypes, WamRuntime, Predicates\n  build-depends:    base >= 4.12, containers >= 0.6, array, time >= 1.8, deepseq >= 1.4, parallel >= 3.2, async >= 2.2\n  default-language: Haskell2010\n  ghc-options:      -O2 -Wno-overlapping-patterns\n",
    setup_call_cleanup(
        open(CabalPath, write, S, [encoding(utf8)]),
        write(S, Cabal),
        close(S)).

drop_bench_hs(Dir) :-
    bench_fixture_path(Src),
    directory_file_path(Dir, 'src/Bench.hs', Dst),
    copy_file(Src, Dst).

%% ========================================================================
%% Cabal build / run
%% ========================================================================

run_cabal(Args, Dir, ExitCode, Output) :-
    catch(
        setup_call_cleanup(
            process_create(path(cabal), Args,
                [cwd(Dir),
                 stdout(pipe(Out)),
                 stderr(pipe(Err)),
                 process(Pid)]),
            (   read_string(Out, _, OutStr),
                read_string(Err, _, ErrStr),
                process_wait(Pid, exit(ExitCode)),
                format(string(Output), "~w~n~w", [OutStr, ErrStr])
            ),
            (catch(close(Out), _, true), catch(close(Err), _, true))
        ),
        Err,
        (ExitCode = -1, format(string(Output), "~w", [Err]))).

%% ========================================================================
%% Main
%% ========================================================================

parse_n(N) :-
    current_prolog_flag(argv, Argv),
    (   Argv = [NS|_], atom_number(NS, NN), integer(NN), NN > 0
    ->  N = NN
    ;   N = 50000
    ).

main :-
    parse_n(N),
    format("~n========================================~n"),
    format("WAM \\+ member(X, L) lowering benchmark~n"),
    format("========================================~n~n"),
    format("[INFO] iterations per case: ~w~n", [N]),
    (   \+ ghc_available
    ->  format("[SKIP] ghc not on PATH~n"), halt(0)
    ;   \+ cabal_available
    ->  format("[SKIP] cabal not on PATH~n"), halt(0)
    ;   true
    ),
    build_dir(Dir),
    catch(cleanup_dir(Dir), _, true),
    generate_project(Dir),
    write_bench_cabal(Dir),
    drop_bench_hs(Dir),
    format("[INFO] cabal v2-build at ~w~n", [Dir]),
    run_cabal(['v2-build', 'uw-not-member-bench'], Dir, BuildEC, BuildOut),
    (   BuildEC \== 0
    ->  format("[FAIL] cabal build failed (exit ~w)~n--- output ---~n~w~n--- end ---~n",
              [BuildEC, BuildOut]),
        keep_or_clean(Dir),
        halt(1)
    ;   true
    ),
    format("[INFO] cabal v2-run uw-not-member-bench -- ~w~n", [N]),
    atom_number(NA, N),
    run_cabal(['v2-run', '-v0', 'uw-not-member-bench', '--', NA], Dir, RunEC, RunOut),
    (   RunEC \== 0
    ->  format("[FAIL] benchmark run failed (exit ~w)~n--- output ---~n~w~n--- end ---~n",
              [RunEC, RunOut]),
        keep_or_clean(Dir),
        halt(1)
    ;   true
    ),
    format("~n========================================~n"),
    format("Benchmark output~n"),
    format("========================================~n"),
    format("~w~n", [RunOut]),
    keep_or_clean(Dir),
    halt(0).

keep_or_clean(Dir) :-
    (   getenv('WAM_NOT_MEMBER_BENCH_KEEP', V), V \== ''
    ->  format("[INFO] keeping build dir ~w (WAM_NOT_MEMBER_BENCH_KEEP set)~n", [Dir])
    ;   catch(cleanup_dir(Dir), _, true)
    ).

cleanup_dir(Dir) :-
    (   exists_directory(Dir)
    ->  process_create(path(rm), ['-rf', Dir], [process(Pid)]),
        process_wait(Pid, _)
    ;   true
    ).

:- initialization(main, main).
