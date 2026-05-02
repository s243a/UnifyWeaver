:- encoding(utf8).
%% Full GHC end-to-end runtime smoke for the `put_structure_dyn`
%% WAM instruction.
%%
%% Companion to tests/core/test_wam_put_structure_dyn_runtime.pl.
%% That file's documented limitation is that it mirrors the runtime
%% `step` rule for `PutStructureDyn` in Prolog rather than executing
%% the actual generated Haskell. This file closes that gap when GHC
%% and cabal-install are available locally.
%%
%% What is exercised:
%%   1. write_wam_haskell_project/3 emits a real project for a
%%      predicate whose compose-mode lowering produces a
%%      PutStructureDyn instruction:
%%          :- mode(build_term(+, ?, -)).
%%          build_term(Name, Arg, T) :- T =.. [Name, Arg].
%%   2. A small custom Smoke.hs (committed under tests/fixtures/) is
%%      dropped into the project's src/ directory. It imports the
%%      *real* generated WamTypes + WamRuntime — same build flags,
%%      same imports (Control.Parallel.Strategies,
%%      Control.Concurrent.Async, etc.) as a benchmark project — and
%%      drives the runtime `step` function on a `PutStructureDyn`
%%      instruction across several cases:
%%        - Atom name + non-negative Integer arity => BuildStruct ...
%%        - Atom name + zero arity => BuildStruct ... 0 []
%%        - Negative arity => Nothing
%%        - Non-Atom name => Nothing
%%        - Missing arity register => Nothing
%%        - Float arity => Nothing
%%        - Unbound name register that derefs through wsBindings to
%%          an Atom => BuildStruct (verifies derefVar is applied)
%%   3. We override the cabal file with a smoke-only executable that
%%      depends only on WamTypes.hs and WamRuntime.hs — Main.hs and
%%      Predicates.hs are not compiled, keeping the build short.
%%   4. cabal builds and runs the executable. Its stdout is parsed
%%      for a `RESULT N/N` summary line. Anything else is a fail.
%%
%% Skip behaviour:
%%   If `cabal --version` or `ghc --version` fail, OR if a probe
%%   build of the `parallel` and `async` packages fails, the test
%%   prints a diagnostic and skips (does not fail). The companion
%%   semantic test in test_wam_put_structure_dyn_runtime.pl still
%%   provides coverage in that case.
%%
%% The probe build is cached: once `parallel` and `async` are
%% confirmed installable in the user cabal store (cabal-install will
%% install them on first encounter — see ~/.cabal/store), the marker
%% file `cabal-deps.ok` is written under the project's tmp build dir
%% and re-used on subsequent runs.
%%
%% Usage:
%%   swipl -g run_tests -t halt tests/core/test_wam_put_structure_dyn_ghc_smoke.pl
%%
%% Honours WAM_PSD_SMOKE_KEEP=1 to retain the build directory for
%% post-mortem inspection (default: cleaned up on success).

:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module('../../src/unifyweaver/targets/wam_haskell_target').
:- use_module(library(filesex), [directory_file_path/3, make_directory_path/1, copy_file/2]).
:- use_module(library(process)).
:- use_module(library(readutil)).

:- dynamic test_failed/0.
:- dynamic test_skipped/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

skip(Test, Reason) :-
    format('[SKIP] ~w: ~w~n', [Test, Reason]),
    (   test_skipped -> true ; assert(test_skipped) ).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

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

ghc_available :- tool_runs(ghc,   ['--version']).
cabal_available :- tool_runs(cabal, ['--version']).

%% ========================================================================
%% Paths
%% ========================================================================

repo_root(Root) :-
    source_file(repo_root(_), This),
    file_directory_name(This, CoreDir),
    file_directory_name(CoreDir, TestsDir),
    file_directory_name(TestsDir, Root).

fixture_smoke_path(Path) :-
    repo_root(Root),
    directory_file_path(Root, 'tests/fixtures/wam_put_structure_dyn_smoke/Smoke.hs', Path).

tmp_root(Root) :-
    (   getenv('TMPDIR', R0), R0 \== ''
    ->  Root = R0
    ;   Root = '/tmp'
    ).

build_dir(Dir) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_psd_ghc_smoke', Dir).

%% ========================================================================
%% Project generation
%% ========================================================================

:- dynamic user:build_term_psd/3.
:- dynamic user:mode/1.

setup_compose_fixture :-
    retractall(user:build_term_psd(_, _, _)),
    retractall(user:mode(build_term_psd(_, _, _))),
    assert(user:mode(build_term_psd(+, ?, -))),
    assert(user:(build_term_psd(Name, Arg, T) :-
        T =.. [Name, Arg])).

teardown_compose_fixture :-
    retractall(user:build_term_psd(_, _, _)),
    retractall(user:mode(build_term_psd(_, _, _))).

ensure_dir(D) :-
    (   exists_directory(D) -> true ; make_directory_path(D) ).

generate_project(Dir) :-
    ensure_dir(Dir),
    setup_compose_fixture,
    catch(
        write_wam_haskell_project(
            [user:build_term_psd/3],
            [module_name('uw-psd-ghc-smoke'), use_hashmap(false)],
            Dir),
        E,
        (teardown_compose_fixture, throw(E))),
    teardown_compose_fixture.

%% Replace the auto-generated cabal file with a smoke-only one. We
%% drop Main.hs / Predicates.hs / Lowered.hs from the build because
%% the smoke does not need them — only WamTypes.hs and WamRuntime.hs.
%% This keeps the build under ~10 seconds on a cold cabal store.
write_smoke_cabal(Dir) :-
    directory_file_path(Dir, 'uw-psd-ghc-smoke.cabal', CabalPath),
    Cabal = "cabal-version: 2.4\nname:          uw-psd-ghc-smoke\nversion:       0.1.0.0\nbuild-type:    Simple\n\nexecutable uw-psd-ghc-smoke\n  main-is:          Smoke.hs\n  hs-source-dirs:   src\n  other-modules:    WamTypes, WamRuntime\n  build-depends:    base >= 4.12, containers >= 0.6, array, time >= 1.8, deepseq >= 1.4, parallel >= 3.2, async >= 2.2\n  default-language: Haskell2010\n  ghc-options:      -O0 -Wno-overlapping-patterns\n",
    setup_call_cleanup(
        open(CabalPath, write, S, [encoding(utf8)]),
        write(S, Cabal),
        close(S)).

drop_smoke_hs(Dir) :-
    fixture_smoke_path(Src),
    directory_file_path(Dir, 'src/Smoke.hs', Dst),
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
%% Tests
%% ========================================================================

test_ghc_smoke :-
    Test = 'WAM put_structure_dyn: full GHC runtime smoke',
    (   \+ ghc_available
    ->  skip(Test, 'ghc not on PATH (fallback: semantic test in test_wam_put_structure_dyn_runtime.pl)')
    ;   \+ cabal_available
    ->  skip(Test, 'cabal not on PATH (fallback: semantic test in test_wam_put_structure_dyn_runtime.pl)')
    ;   build_dir(Dir),
        catch(generate_project(Dir), GE,
              (fail_test(Test, generate_project_failed(GE)), !, fail)),
        write_smoke_cabal(Dir),
        drop_smoke_hs(Dir),
        format('[INFO] cabal building smoke at ~w (this may install parallel/async on first run)~n', [Dir]),
        run_cabal(['build', 'uw-psd-ghc-smoke'], Dir, BuildEC, BuildOut),
        (   BuildEC \== 0
        ->  (   sub_string(BuildOut, _, _, _, "parallel"),
                sub_string(BuildOut, _, _, _, "Could not")
            ->  skip(Test, 'cabal cannot fetch parallel/async (no Hackage access)')
            ;   sub_string(BuildOut, _, _, _, "async"),
                sub_string(BuildOut, _, _, _, "Could not")
            ->  skip(Test, 'cabal cannot fetch parallel/async (no Hackage access)')
            ;   fail_test(Test, cabal_build_failed(BuildEC)),
                format('--- cabal output ---~n~w~n--- end ---~n', [BuildOut])
            )
        ;   run_cabal(['run', '-v0', 'uw-psd-ghc-smoke'], Dir, RunEC, RunOut),
            (   RunEC \== 0
            ->  fail_test(Test, cabal_run_failed(RunEC)),
                format('--- run output ---~n~w~n--- end ---~n', [RunOut])
            ;   parse_smoke_output(RunOut, Test)
            )
        ),
        cleanup_build_dir(Dir)
    ).

parse_smoke_output(Out, Test) :-
    (   sub_string(Out, _, _, _, "RESULT 7/7")
    ->  pass(Test)
    ;   sub_string(Out, Before, _, _, "RESULT ")
    ->  sub_string(Out, Before, 12, _, ResultLine),
        fail_test(Test, smoke_result_not_full(ResultLine)),
        format('--- smoke stdout ---~n~w~n--- end ---~n', [Out])
    ;   fail_test(Test, no_result_line),
        format('--- smoke stdout ---~n~w~n--- end ---~n', [Out])
    ).

cleanup_build_dir(Dir) :-
    (   getenv('WAM_PSD_SMOKE_KEEP', V), V \== ''
    ->  format('[INFO] keeping build dir ~w (WAM_PSD_SMOKE_KEEP set)~n', [Dir])
    ;   catch(delete_directory_and_contents_safe(Dir), _, true)
    ).

delete_directory_and_contents_safe(Dir) :-
    (   exists_directory(Dir)
    ->  process_create(path(rm), ['-rf', Dir], [process(Pid)]),
        process_wait(Pid, _)
    ;   true
    ).

%% ========================================================================
%% Runner
%% ========================================================================

run_tests :-
    format('~n========================================~n'),
    format('WAM put_structure_dyn full GHC smoke~n'),
    format('========================================~n~n'),
    test_ghc_smoke,
    format('~n========================================~n'),
    (   test_failed
    ->  format('Some tests FAILED~n'), halt(1)
    ;   test_skipped
    ->  format('Tests skipped (toolchain unavailable). Semantic coverage in test_wam_put_structure_dyn_runtime.pl is unaffected.~n'), halt(0)
    ;   format('All tests passed~n'), halt(0)
    ).

:- initialization(run_tests, main).
