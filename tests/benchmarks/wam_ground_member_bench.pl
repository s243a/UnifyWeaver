:- encoding(utf8).
%% Microbenchmark for the ground-list `\+ member(X, [a, b, c])`
%% lowering — i.e. NotMemberConstAtoms.
%%
%% The same Prolog source predicate is compiled into two separate
%% Haskell projects:
%%
%%   bench_ground(X) :- \+ member(X, [a, b, c, d, e, f, g, h]).
%%
%%   * lowered:   default codegen — emits a single NotMemberConstAtoms
%%                instruction (atom IDs baked in).
%%   * unlowered: codegen with `wam_target:lowering_disabled(ground_member)`
%%                asserted — falls through to put_structure cons cells +
%%                builtin_call \\+/1 + member walk (~ N+5 instructions
%%                for an N-element list).
%%
%% A shared Bench.hs (in tests/fixtures/wam_ground_member_bench/) calls
%% bench_ground(Integer k) N times per case. The probe value is an
%% Integer that varies per iteration to defeat constant-folding; it is
%% never one of the atoms so the check always succeeds.
%%
%% Skip behaviour: same as wam_not_member_bench.pl. Honours
%% WAM_GROUND_MEMBER_BENCH_KEEP=1 to retain build dirs after the run.
%%
%% Usage:
%%   swipl -g main -t halt tests/benchmarks/wam_ground_member_bench.pl [-- N]
%%   N defaults to 200000.

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
        'tests/fixtures/wam_ground_member_bench/Bench.hs', Path).

tmp_root(Root) :-
    (   getenv('TMPDIR', R0), R0 \== ''
    ->  Root = R0
    ;   Root = '/tmp'
    ).

build_dir_for(Variant, Dir) :-
    tmp_root(Root),
    atom_concat('uw_ground_member_bench_', Variant, Sub),
    directory_file_path(Root, Sub, Dir).

%% ========================================================================
%% Fixture
%% ========================================================================

:- dynamic user:bench_ground/1.

setup_fixture :-
    retractall(user:bench_ground(_)),
    assert(user:(bench_ground(X) :-
        \+ member(X, [a, b, c, d, e, f, g, h]))).

teardown_fixture :-
    retractall(user:bench_ground(_)).

ensure_dir(D) :-
    (   exists_directory(D) -> true ; make_directory_path(D) ).

generate_project(Variant, Dir) :-
    ensure_dir(Dir),
    setup_fixture,
    %% Toggle the per-process disable flag based on variant. The
    %% lowering is keyed off `wam_target:lowering_disabled(ground_member)`.
    retractall(wam_target:lowering_disabled(ground_member)),
    (   Variant == unlowered
    ->  assertz(wam_target:lowering_disabled(ground_member))
    ;   true
    ),
    catch(
        write_wam_haskell_project(
            [user:bench_ground/1],
            [module_name('uw-ground-member-bench'), use_hashmap(false)],
            Dir),
        E,
        (teardown_fixture,
         retractall(wam_target:lowering_disabled(ground_member)),
         throw(E))),
    teardown_fixture,
    retractall(wam_target:lowering_disabled(ground_member)).

write_bench_cabal(Dir) :-
    directory_file_path(Dir, 'uw-ground-member-bench.cabal', CabalPath),
    Cabal = "cabal-version: 2.4\nname:          uw-ground-member-bench\nversion:       0.1.0.0\nbuild-type:    Simple\n\nexecutable uw-ground-member-bench\n  main-is:          Bench.hs\n  hs-source-dirs:   src\n  other-modules:    WamTypes, WamRuntime, Predicates\n  build-depends:    base >= 4.12, containers >= 0.6, array, time >= 1.8, deepseq >= 1.4, parallel >= 3.2, async >= 2.2\n  default-language: Haskell2010\n  ghc-options:      -O2 -Wno-overlapping-patterns\n",
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
%% Per-variant build + run
%% ========================================================================

build_and_run(Variant, N, Output) :-
    build_dir_for(Variant, Dir),
    catch(cleanup_dir(Dir), _, true),
    format("[INFO] ~w: generating project at ~w~n", [Variant, Dir]),
    generate_project(Variant, Dir),
    write_bench_cabal(Dir),
    drop_bench_hs(Dir),
    format("[INFO] ~w: cabal v2-build~n", [Variant]),
    run_cabal(['v2-build', 'uw-ground-member-bench'], Dir, BuildEC, BuildOut),
    (   BuildEC \== 0
    ->  format("[FAIL] ~w build (exit ~w)~n--- output ---~n~w~n",
              [Variant, BuildEC, BuildOut]),
        Output = ""
    ;   format("[INFO] ~w: cabal v2-run -- ~w~n", [Variant, N]),
        atom_number(NA, N),
        run_cabal(['v2-run', '-v0', 'uw-ground-member-bench', '--', NA],
                  Dir, RunEC, RunOut),
        (   RunEC \== 0
        ->  format("[FAIL] ~w run (exit ~w)~n--- output ---~n~w~n",
                  [Variant, RunEC, RunOut]),
            Output = ""
        ;   Output = RunOut
        )
    ).

%% ========================================================================
%% Main
%% ========================================================================

parse_n(N) :-
    current_prolog_flag(argv, Argv),
    (   Argv = [NS|_], atom_number(NS, NN), integer(NN), NN > 0
    ->  N = NN
    ;   N = 200000
    ).

main :-
    parse_n(N),
    format("~n========================================~n"),
    format("WAM ground-list \\+ member lowering benchmark~n"),
    format("========================================~n~n"),
    format("[INFO] iterations per case: ~w~n", [N]),
    (   \+ ghc_available
    ->  format("[SKIP] ghc not on PATH~n"), halt(0)
    ;   \+ cabal_available
    ->  format("[SKIP] cabal not on PATH~n"), halt(0)
    ;   true
    ),
    build_and_run(lowered,   N, OutL),
    build_and_run(unlowered, N, OutU),
    format("~n========================================~n"),
    format("Lowered (NotMemberConstAtoms)~n"),
    format("========================================~n~w~n", [OutL]),
    format("========================================~n"),
    format("Unlowered (builtin \\+/1 + member walk)~n"),
    format("========================================~n~w~n", [OutU]),
    keep_or_clean,
    halt(0).

keep_or_clean :-
    (   getenv('WAM_GROUND_MEMBER_BENCH_KEEP', V), V \== ''
    ->  format("[INFO] keeping build dirs (WAM_GROUND_MEMBER_BENCH_KEEP set)~n", [])
    ;   build_dir_for(lowered, DL),
        build_dir_for(unlowered, DU),
        catch(cleanup_dir(DL), _, true),
        catch(cleanup_dir(DU), _, true)
    ).

cleanup_dir(Dir) :-
    (   exists_directory(Dir)
    ->  process_create(path(rm), ['-rf', Dir], [process(Pid)]),
        process_wait(Pid, _)
    ;   true
    ).

:- initialization(main, main).
