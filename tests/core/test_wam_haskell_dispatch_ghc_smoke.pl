:- encoding(utf8).
%% Full GHC end-to-end runtime smoke for the dispatch fixes in
%% wam_haskell_target.pl (PR #2356 — mirrors F# Bug A from
%% PRs #2351 / #2352).
%%
%% Companion to tests/test_wam_haskell_target.pl, which only asserts
%% the generated Haskell *source* contains the expected case-expression
%% branches.  This file closes the runtime gap by cabal-building the
%% real WamTypes + WamRuntime modules and driving `step` directly
%% against crafted instructions.
%%
%% Cases exercised (see tests/fixtures/wam_haskell_dispatch_smoke/Smoke.hs):
%%   * TrustMe / RetryMeElse / RetryMeElsePc with empty wsCPs:
%%     must advance PC (Bug A.2 — previously Nothing).
%%   * TrustMe / RetryMeElsePc with one CP: sanity that the classical
%%     behaviour still holds (pop / mutate cpNextPC respectively).
%%   * SwitchOnConstantPc with Atom miss, Integer miss, Float in A1:
%%     must fall through to PC+1 (Bug A.1 — previously Nothing).
%%   * SwitchOnConstantPc with Atom hit + Integer hit: sanity that
%%     hits still jump to the mapped PC.
%%   * SwitchOnConstant (Map-keyed Value variant): same miss vs. hit
%%     coverage.
%%
%% Skip behaviour:
%%   If `ghc --version` or `cabal --version` fail, OR if cabal cannot
%%   fetch dependencies (no Hackage access), the test prints a
%%   diagnostic and skips (does not fail).  Source-level coverage in
%%   tests/test_wam_haskell_target.pl is unaffected in that case.
%%
%% Honours WAM_HS_DISPATCH_SMOKE_KEEP=1 to retain the build dir.
%%
%% Cross-platform tmp directory
%% ----------------------------
%% Build directories live under a writable tmp root resolved with the
%% following precedence (first existing+writable wins):
%%   1. UW_SMOKE_TMPDIR        explicit override for this test
%%   2. TMPDIR / TMP / TEMP    standard env vars (Unix / Windows / Cygwin)
%%   3. $PREFIX/tmp            Termux convention
%%   4. /data/data/com.termux/files/usr/tmp   Termux default path
%%   5. /tmp                   Unix default
%%   6. ./tmp                  cwd-relative last resort
%% Cleanup uses library(filesex):delete_directory_and_contents/1 so
%% the test does not depend on `rm` being on PATH (works on Windows).

:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module('../../src/unifyweaver/targets/wam_haskell_target').
:- use_module('../helpers/smoke_paths', [tmp_root/1, clean_dir/1]).
:- use_module(library(filesex), [directory_file_path/3,
                                  make_directory_path/1,
                                  copy_file/2]).
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
    directory_file_path(Root, 'tests/fixtures/wam_haskell_dispatch_smoke/Smoke.hs', Path).

%% tmp_root/1 and clean_dir/1 imported from helpers/smoke_paths.

build_dir(Dir) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_hs_dispatch_ghc_smoke', Dir).

%% ========================================================================
%% Project generation
%%
%% We only need WamTypes.hs + WamRuntime.hs from the generated project.
%% Use a trivial single-fact predicate so codegen succeeds; the smoke
%% does not exercise Main.hs / Predicates.hs / Lowered.hs.
%% ========================================================================

:- dynamic user:hs_dispatch_smoke_fact/1.
:- dynamic user:mode/1.

setup_fixture :-
    retractall(user:hs_dispatch_smoke_fact(_)),
    retractall(user:mode(hs_dispatch_smoke_fact(_))),
    assert(user:mode(hs_dispatch_smoke_fact(+))),
    assert(user:hs_dispatch_smoke_fact(seed)).

teardown_fixture :-
    retractall(user:hs_dispatch_smoke_fact(_)),
    retractall(user:mode(hs_dispatch_smoke_fact(_))).

ensure_dir(D) :-
    (   exists_directory(D) -> true ; make_directory_path(D) ).

generate_project(Dir) :-
    ensure_dir(Dir),
    setup_fixture,
    catch(
        write_wam_haskell_project(
            [user:hs_dispatch_smoke_fact/1],
            [module_name('uw-hs-dispatch-ghc-smoke'), use_hashmap(false)],
            Dir),
        E,
        (teardown_fixture, throw(E))),
    teardown_fixture.

%% Smoke-only cabal: drops Main.hs / Predicates.hs / Lowered.hs to
%% keep the build short (~10s cold cabal store).
write_smoke_cabal(Dir) :-
    directory_file_path(Dir, 'uw-hs-dispatch-ghc-smoke.cabal', CabalPath),
    Cabal = "cabal-version: 2.4\nname:          uw-hs-dispatch-ghc-smoke\nversion:       0.1.0.0\nbuild-type:    Simple\n\nexecutable uw-hs-dispatch-ghc-smoke\n  main-is:          Smoke.hs\n  hs-source-dirs:   src\n  other-modules:    WamTypes, WamRuntime\n  build-depends:    base >= 4.12, containers >= 0.6, array, time >= 1.8, deepseq >= 1.4, parallel >= 3.2, async >= 2.2\n  default-language: Haskell2010\n  ghc-options:      -O0 -Wno-overlapping-patterns\n",
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
    Test = 'WAM-Haskell dispatch (PR #2356): full GHC runtime smoke',
    (   \+ ghc_available
    ->  skip(Test, 'ghc not on PATH (fallback: source-level test in tests/test_wam_haskell_target.pl)')
    ;   \+ cabal_available
    ->  skip(Test, 'cabal not on PATH (fallback: source-level test in tests/test_wam_haskell_target.pl)')
    ;   build_dir(Dir),
        catch(generate_project(Dir), GE,
              (fail_test(Test, generate_project_failed(GE)), !, fail)),
        write_smoke_cabal(Dir),
        drop_smoke_hs(Dir),
        format('[INFO] cabal building smoke at ~w (this may install parallel/async on first run)~n', [Dir]),
        run_cabal(['build', 'uw-hs-dispatch-ghc-smoke'], Dir, BuildEC, BuildOut),
        (   BuildEC \== 0
        ->  (   sub_string(BuildOut, _, _, _, "Could not")
            ->  skip(Test, 'cabal cannot fetch deps (no Hackage access)')
            ;   fail_test(Test, cabal_build_failed(BuildEC)),
                format('--- cabal output ---~n~w~n--- end ---~n', [BuildOut])
            )
        ;   run_cabal(['run', '-v0', 'uw-hs-dispatch-ghc-smoke'], Dir, RunEC, RunOut),
            (   RunEC \== 0
            ->  fail_test(Test, cabal_run_failed(RunEC)),
                format('--- run output ---~n~w~n--- end ---~n', [RunOut])
            ;   parse_smoke_output(RunOut, Test)
            )
        ),
        cleanup_build_dir(Dir)
    ).

parse_smoke_output(Out, Test) :-
    (   sub_string(Out, _, _, _, "RESULT 12/12")
    ->  pass(Test)
    ;   sub_string(Out, Before, _, _, "RESULT ")
    ->  sub_string(Out, Before, 14, _, ResultLine),
        fail_test(Test, smoke_result_not_full(ResultLine)),
        format('--- smoke stdout ---~n~w~n--- end ---~n', [Out])
    ;   fail_test(Test, no_result_line),
        format('--- smoke stdout ---~n~w~n--- end ---~n', [Out])
    ).

cleanup_build_dir(Dir) :-
    (   getenv('WAM_HS_DISPATCH_SMOKE_KEEP', V), V \== ''
    ->  format('[INFO] keeping build dir ~w (WAM_HS_DISPATCH_SMOKE_KEEP set)~n', [Dir])
    ;   clean_dir(Dir)
    ).

%% ========================================================================
%% Runner
%% ========================================================================

run_tests :-
    format('~n========================================~n'),
    format('WAM-Haskell dispatch full GHC smoke~n'),
    format('========================================~n~n'),
    test_ghc_smoke,
    format('~n========================================~n'),
    (   test_failed
    ->  format('Some tests FAILED~n'), halt(1)
    ;   test_skipped
    ->  format('Tests skipped (toolchain unavailable). Source-level coverage in tests/test_wam_haskell_target.pl is unaffected.~n'), halt(0)
    ;   format('All tests passed~n'), halt(0)
    ).

:- initialization(run_tests, main).
