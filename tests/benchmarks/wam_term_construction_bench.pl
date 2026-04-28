:- encoding(utf8).
%% Microbenchmark for the =../2 compose-mode lowering.
%%
%% Generates a Haskell project containing two predicates with
%% IDENTICAL Prolog source — `bench_lowered/3` (with `:- mode/1`)
%% and `bench_unlowered/3` (no mode declaration). The mode
%% declaration triggers the PutStructureDyn lowering for
%% bench_lowered/3; bench_unlowered/3 falls through to the
%% list-build + BuiltinCall "=../2" runtime path.
%%
%% A custom Bench.hs harness (in tests/fixtures/) imports
%% the generated WamRuntime + Predicates and times each predicate
%% across N iterations, alternating the run order to reduce
%% cache-warming bias.
%%
%% Usage:
%%   swipl -g main -t halt tests/benchmarks/wam_term_construction_bench.pl [-- N]
%%   N defaults to 100000.
%%
%% Skip behaviour:
%%   If `cabal --version` or `ghc --version` fail, OR if cabal
%%   cannot resolve the project's dependencies (no Hackage access),
%%   print a diagnostic and exit cleanly without failure.
%%
%% Honours WAM_TERM_BENCH_KEEP=1 to retain the build dir.

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
        'tests/fixtures/wam_term_construction_bench/Bench.hs', Path).

tmp_root(Root) :-
    (   getenv('TMPDIR', R0), R0 \== ''
    ->  Root = R0
    ;   Root = '/tmp'
    ).

build_dir(Dir) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_term_construction_bench', Dir).

%% ========================================================================
%% Fixture
%% ========================================================================

:- dynamic user:bench_lowered/3.
:- dynamic user:bench_unlowered/3.
:- dynamic user:mode/1.

setup_fixture :-
    retractall(user:bench_lowered(_,_,_)),
    retractall(user:bench_unlowered(_,_,_)),
    retractall(user:mode(bench_lowered(_,_,_))),
    %% Mode declaration on bench_lowered triggers the lowering.
    assert(user:mode(bench_lowered(+, +, -))),
    assert(user:(bench_lowered(Name, Arg, T) :-
        T =.. [Name, Arg])),
    %% Same body, no mode → fallthrough.
    assert(user:(bench_unlowered(Name, Arg, T) :-
        T =.. [Name, Arg])).

teardown_fixture :-
    retractall(user:bench_lowered(_,_,_)),
    retractall(user:bench_unlowered(_,_,_)),
    retractall(user:mode(bench_lowered(_,_,_))).

ensure_dir(D) :-
    (   exists_directory(D) -> true ; make_directory_path(D) ).

generate_project(Dir) :-
    ensure_dir(Dir),
    setup_fixture,
    catch(
        write_wam_haskell_project(
            [user:bench_lowered/3, user:bench_unlowered/3],
            [module_name('uw-term-bench'), use_hashmap(false)],
            Dir),
        E,
        (teardown_fixture, throw(E))),
    teardown_fixture.

%% Drop our custom benchmark harness next to the generated sources and
%% override the cabal file so it builds Bench.hs as the executable
%% (Main.hs / Lowered.hs are not on the bench path).
write_bench_cabal(Dir) :-
    directory_file_path(Dir, 'uw-term-bench.cabal', CabalPath),
    Cabal = "cabal-version: 2.4\nname:          uw-term-bench\nversion:       0.1.0.0\nbuild-type:    Simple\n\nexecutable uw-term-bench\n  main-is:          Bench.hs\n  hs-source-dirs:   src\n  other-modules:    WamTypes, WamRuntime, Predicates\n  build-depends:    base >= 4.12, containers >= 0.6, array, time >= 1.8, deepseq >= 1.4, parallel >= 3.2, async >= 2.2\n  default-language: Haskell2010\n  ghc-options:      -O2 -Wno-overlapping-patterns\n",
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
    ;   N = 100000
    ).

main :-
    parse_n(N),
    format("~n========================================~n"),
    format("WAM term-construction lowering benchmark~n"),
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
    run_cabal(['v2-build', 'uw-term-bench'], Dir, BuildEC, BuildOut),
    (   BuildEC \== 0
    ->  format("[FAIL] cabal build failed (exit ~w)~n--- output ---~n~w~n--- end ---~n",
              [BuildEC, BuildOut]),
        keep_or_clean(Dir),
        halt(1)
    ;   true
    ),
    format("[INFO] cabal v2-run uw-term-bench -- ~w~n", [N]),
    atom_number(NA, N),
    run_cabal(['v2-run', '-v0', 'uw-term-bench', '--', NA], Dir, RunEC, RunOut),
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
    (   getenv('WAM_TERM_BENCH_KEEP', V), V \== ''
    ->  format("[INFO] keeping build dir ~w (WAM_TERM_BENCH_KEEP set)~n", [Dir])
    ;   catch(cleanup_dir(Dir), _, true)
    ).

cleanup_dir(Dir) :-
    (   exists_directory(Dir)
    ->  process_create(path(rm), ['-rf', Dir], [process(Pid)]),
        process_wait(Pid, _)
    ;   true
    ).

:- initialization(main, main).
