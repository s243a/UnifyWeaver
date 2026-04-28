:- encoding(utf8).
%% Phase M7 of the mode-analysis plan: full Haskell-cabal end-to-end
%% smoke for the term-construction lowerings (=../2 and functor/3).
%%
%% What this test validates that the codegen unit tests do not:
%%
%%   1. write_wam_haskell_project/3 produces a real project on disk
%%      whose generated Predicates.hs contains `PutStructureDyn` for
%%      mode-annotated compose-mode predicates, AND `BuiltinCall
%%      "=../2"` (resp. `BuiltinCall "functor/3"`) for unannotated
%%      ones in the same project.
%%   2. The full project (Main.hs + Predicates.hs + Lowered.hs +
%%      WamTypes.hs + WamRuntime.hs) compiles end-to-end with cabal.
%%      A successful build proves wam_instr_to_haskell/2's
%%      put_structure_dyn → PutStructureDyn translation produces
%%      well-typed Haskell when emitted from a real Prolog source.
%%
%% Skip behaviour:
%%   If `cabal --version` or `ghc --version` fail, OR if cabal cannot
%%   resolve the project's dependencies (no Hackage access, etc.),
%%   the test prints a diagnostic and skips. Codegen unit tests in
%%   test_wam_univ_lowering.pl and test_wam_functor3_lowering.pl
%%   still cover the WAM-text level.
%%
%% Honours WAM_TERM_E2E_KEEP=1 to retain the build dir.
%%
%% Usage:
%%   swipl -g run_tests -t halt tests/core/test_wam_term_construction_e2e.pl

:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module('../../src/unifyweaver/targets/wam_haskell_target').
:- use_module(library(filesex), [directory_file_path/3, make_directory_path/1]).
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

tmp_root(Root) :-
    (   getenv('TMPDIR', R0), R0 \== ''
    ->  Root = R0
    ;   Root = '/tmp'
    ).

build_dir(Dir) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_term_construction_e2e', Dir).

%% ========================================================================
%% Fixture
%% ========================================================================
%%
%% Four predicates exercising both lowerings and both fall-through paths:
%%
%%   build_term_compose/3 (=../2, mode +,?,-)  → expect PutStructureDyn
%%   build_term_nomode/3 (=../2, no mode)      → expect BuiltinCall "=../2"
%%   make_pair_compose/2 (functor/3, mode +,-) → expect PutStructureDyn
%%   make_pair_nomode/2  (functor/3, no mode)  → expect BuiltinCall "functor/3"

:- dynamic user:build_term_compose/3.
:- dynamic user:build_term_nomode/3.
:- dynamic user:make_pair_compose/2.
:- dynamic user:make_pair_nomode/2.
:- dynamic user:mode/1.

setup_fixture :-
    retractall(user:build_term_compose(_,_,_)),
    retractall(user:build_term_nomode(_,_,_)),
    retractall(user:make_pair_compose(_,_)),
    retractall(user:make_pair_nomode(_,_)),
    retractall(user:mode(build_term_compose(_,_,_))),
    retractall(user:mode(make_pair_compose(_,_))),
    assert(user:mode(build_term_compose(+, ?, -))),
    assert(user:mode(make_pair_compose(+, -))),
    assert(user:(build_term_compose(Name, Arg, T) :-
        T =.. [Name, Arg])),
    assert(user:(build_term_nomode(Name, Arg, T) :-
        T =.. [Name, Arg])),
    assert(user:(make_pair_compose(Name, T) :-
        functor(T, Name, 2))),
    assert(user:(make_pair_nomode(Name, T) :-
        functor(T, Name, 2))).

teardown_fixture :-
    retractall(user:build_term_compose(_,_,_)),
    retractall(user:build_term_nomode(_,_,_)),
    retractall(user:make_pair_compose(_,_)),
    retractall(user:make_pair_nomode(_,_)),
    retractall(user:mode(build_term_compose(_,_,_))),
    retractall(user:mode(make_pair_compose(_,_))).

ensure_dir(D) :-
    (   exists_directory(D) -> true ; make_directory_path(D) ).

generate_project(Dir) :-
    ensure_dir(Dir),
    setup_fixture,
    catch(
        write_wam_haskell_project(
            [user:build_term_compose/3,
             user:build_term_nomode/3,
             user:make_pair_compose/2,
             user:make_pair_nomode/2],
            [module_name('uw-term-construction-e2e'), use_hashmap(false)],
            Dir),
        E,
        (teardown_fixture, throw(E))),
    teardown_fixture.

%% ========================================================================
%% Cabal build
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

cabal_dependency_problem(BuildOut) :-
    sub_string(BuildOut, _, _, _, "Could not"),
    (   sub_string(BuildOut, _, _, _, "parallel")
    ;   sub_string(BuildOut, _, _, _, "async")
    ;   sub_string(BuildOut, _, _, _, "containers")
    ;   sub_string(BuildOut, _, _, _, "Hackage")
    ).

%% ========================================================================
%% Predicates.hs grep
%% ========================================================================

predicates_hs_path(Dir, Path) :-
    directory_file_path(Dir, 'src/Predicates.hs', Path).

read_predicates_hs(Dir, Content) :-
    predicates_hs_path(Dir, Path),
    read_file_to_string(Path, Content, []).

%% Count occurrences of a needle substring in a haystack.
count_substr(Hay, Needle, N) :-
    string_length(Needle, NeedleLen),
    findall(P, sub_string(Hay, P, NeedleLen, _, Needle), Ps),
    length(Ps, N).

%% ========================================================================
%% Tests
%% ========================================================================

test_full_pipeline :-
    Test = 'WAM term construction: full project compiles via cabal',
    (   \+ ghc_available
    ->  skip(Test, 'ghc not on PATH')
    ;   \+ cabal_available
    ->  skip(Test, 'cabal not on PATH')
    ;   build_dir(Dir),
        catch(cleanup_build_dir(Dir), _, true),
        catch(generate_project(Dir), GE,
              (fail_test(Test, generate_project_failed(GE)), !, fail)),
        format('[INFO] cabal v2-build full project at ~w~n', [Dir]),
        run_cabal(['v2-build', 'all'], Dir, BuildEC, BuildOut),
        (   BuildEC \== 0
        ->  (   cabal_dependency_problem(BuildOut)
            ->  skip(Test, 'cabal cannot resolve dependencies (no Hackage access)')
            ;   fail_test(Test, cabal_build_failed(BuildEC)),
                format('--- cabal output ---~n~w~n--- end ---~n', [BuildOut])
            )
        ;   pass(Test)
        ),
        keep_or_clean(Dir)
    ).

test_compose_lowerings_in_predicates_hs :-
    Test = 'WAM term construction: PutStructureDyn count and fallthrough builtins',
    build_dir(Dir),
    (   exists_directory(Dir)
    ->  true
    ;   skip(Test, 'project dir missing (full pipeline test must run first)'),
        !, fail
    ),
    catch(read_predicates_hs(Dir, Hs), RE,
          (fail_test(Test, read_predicates_hs_failed(RE)), !, fail)),
    %% allCode is one merged instruction list per project. Two compose
    %% predicates ⇒ two PutStructureDyn occurrences. Two non-mode
    %% predicates ⇒ exactly one BuiltinCall "=../2" and one BuiltinCall
    %% "functor/3" each.
    count_substr(Hs, "PutStructureDyn", DynCount),
    count_substr(Hs, "BuiltinCall \"=../2\"", UnivBuiltinCount),
    count_substr(Hs, "BuiltinCall \"functor/3\"", FunctorBuiltinCount),
    (   DynCount =:= 2,
        UnivBuiltinCount =:= 1,
        FunctorBuiltinCount =:= 1
    ->  pass(Test)
    ;   fail_test(Test,
            counts(put_structure_dyn=DynCount,
                   builtin_univ=UnivBuiltinCount,
                   builtin_functor=FunctorBuiltinCount)),
        format('--- Predicates.hs head ---~n', []),
        sub_string(Hs, 0, 2048, _, Head),
        format('~w~n--- end ---~n', [Head])
    ).

%% ========================================================================
%% Cleanup
%% ========================================================================

keep_or_clean(Dir) :-
    (   getenv('WAM_TERM_E2E_KEEP', V), V \== ''
    ->  format('[INFO] keeping build dir ~w (WAM_TERM_E2E_KEEP set)~n', [Dir])
    ;   true   % leave for the second test to read; cleanup at the end
    ).

cleanup_build_dir(Dir) :-
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
    format('WAM term construction full E2E (=../2 + functor/3)~n'),
    format('========================================~n~n'),
    test_full_pipeline,
    test_compose_lowerings_in_predicates_hs,
    cleanup_at_end,
    format('~n========================================~n'),
    (   test_failed
    ->  format('Some tests FAILED~n'), halt(1)
    ;   test_skipped
    ->  format('Tests skipped (toolchain unavailable). Codegen coverage in test_wam_univ_lowering.pl + test_wam_functor3_lowering.pl is unaffected.~n'), halt(0)
    ;   format('All tests passed~n'), halt(0)
    ).

cleanup_at_end :-
    build_dir(Dir),
    (   getenv('WAM_TERM_E2E_KEEP', V), V \== ''
    ->  format('[INFO] keeping build dir ~w (WAM_TERM_E2E_KEEP set)~n', [Dir])
    ;   catch(cleanup_build_dir(Dir), _, true)
    ).

:- initialization(run_tests, main).
