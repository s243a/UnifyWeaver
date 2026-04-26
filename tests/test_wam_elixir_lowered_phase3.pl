:- encoding(utf8).
% Phase 3 runtime smoke test for the WAM-lowered Elixir target.
%
% Tests end-to-end: compile a findall/3-using predicate through the
% WAM compiler → lower to Elixir → write the project → compile + run
% via `elixir` → assert the result list.
%
% This is the first test that actually executes the lowered Elixir
% runtime, validating the full Phase 1 (substrate) + Phase 2
% (instruction lowering) chain end-to-end. See
% docs/proposals/WAM_ELIXIR_TIER2_FINDALL.md §7 (Phase 3 plan).
%
% Usage: swipl -g run_tests -t halt tests/test_wam_elixir_lowered_phase3.pl
%
% Skipped gracefully when `elixir` is not on PATH (matches the
% Clojure / Haskell phase-3 test conventions). Opt-in from the
% top-level test runner — only invoke from CI nodes that have
% Elixir provisioned.
%
% ============================================================================
% Phase 3 findings surfaced by this smoke (worth following up):
%
% 1. WAM-compiler bug for module-qualified inner goals.
%    `findall(X, user:p(X), L)` compiles the inner call as a `:/2`
%    builtin that overwrites A1 with the module name string ("user")
%    before end_aggregate reads value_reg=A1 — captures "user" instead
%    of the Template binding. Affects all targets; root cause is in
%    wam_target.pl's compile_aggregate_all/5 which defaults
%    ValueReg=A1 when Template is a bare variable but doesn't ensure
%    A1 still holds the binding by end_aggregate time. Workaround:
%    don't module-qualify the inner goal.
%
% 2. Tier-2 super-wrapper × findall interaction breaks finalise.
%    When the findall's inner goal is Tier-2 eligible (≥3 clauses,
%    declared/inferred pure), the super-wrapper's parallel arm fires
%    because in_forkable_aggregate_frame? returns true (the parent
%    findall pushed the frame). The parallel arm calls
%    merge_into_aggregate and RETURNS — doesn't throw fail. The
%    caller's post-call code then runs aggregate_collect on a stale
%    A1 and throws fail, which finalises an accumulator that has
%    BOTH the parallel-merged results AND a spurious aggregate_collect
%    entry — incorrect.
%    This materialises proposal §6 risk #2 / Haskell precedent
%    finding (wam_haskell_target.pl:1502-1516). The proposal was
%    correct that nested-parallel needs a branch-local-accum protocol;
%    this is the runtime confirmation. Workaround for Phase 3:
%    intra_query_parallel(false) kill-switches the super-wrapper.
%    Phase 4 needs to import Haskell's branch-local pattern.
% ============================================================================

:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module(library(option)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_elixir_target',
              [write_wam_elixir_project/3]).

:- dynamic test_failed/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

%% PATH guard — skip the suite cleanly if `elixir` is unavailable.
%  Matches the convention of other subprocess-execution test files
%  (test_wam_clojure_runtime_smoke.pl, test_wam_haskell_lowered_phase3.pl)
%  that don't run on hosts without their respective toolchains.
elixir_available :-
    catch(
        process_create(path(elixir), ['--version'],
                       [stdout(pipe(Out)), stderr(null), process(PID)]),
        _,
        fail),
    read_string(Out, _, _),
    close(Out),
    process_wait(PID, exit(0)).

%% Test fixtures — declared at the top level so the WAM compiler can
%% find them via user:Pred/Arity. The findall body in
%% phase3_smoke_findall/1 is what exercises begin_aggregate /
%% end_aggregate at runtime.

:- dynamic user:phase3_smoke_p/1.
:- dynamic user:phase3_smoke_findall/1.

user:phase3_smoke_p('a').
user:phase3_smoke_p('b').
user:phase3_smoke_p('c').
% Inner goal is NOT module-qualified — `findall(X, user:phase3_smoke_p(X), L)`
% triggers a known bug where compile_aggregate_all/5 emits the inner
% call as a `:/2` builtin that overwrites A1 with the module name
% string before end_aggregate reads it (value_reg defaults to A1
% when Template is a bare variable). The bug is in wam_target.pl's
% compile_findall — out of scope for Phase 3 of the Elixir track,
% noted as a follow-up finding for the WAM compiler.
user:phase3_smoke_findall(L) :- findall(X, phase3_smoke_p(X), L).

%% tmp_root — try TMPDIR / TMP / TEMP / $PREFIX/tmp / ./output in
%% order. Same fallback chain as the existing benchmark harness.
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

unique_project_dir(Dir) :-
    writable_tmp_root(Root),
    get_time(T),
    Stamp is floor(T * 1000),
    format(atom(Name), 'uw_elixir_findall_smoke_~w', [Stamp]),
    directory_file_path(Root, Name, Dir).

%% Generate the test project and the driver .exs script. Predicates
%% must include both the findall caller and the inner predicate
%% (the dispatcher needs to route phase3_smoke_p/1 calls).
write_smoke_project(ProjectDir) :-
    Predicates = [
        user:phase3_smoke_p/1,
        user:phase3_smoke_findall/1
    ],
    findall(P/A-WamCode, (
        member(M:P/A, Predicates),
        wam_target:compile_predicate_to_wam(M:P/A, [], WamCode)
    ), PredWamPairs),
    % intra_query_parallel(false) kill-switches the Tier-2 super-wrapper.
    % Without it, the 3-clause phase3_smoke_p/1 gets Tier-2-eligible, its
    % super-wrapper fires (because the parent findall pushed an agg
    % frame, so in_forkable_aggregate_frame? returns true), and the
    % parallel arm returns from merge_into_aggregate instead of throwing
    % fail to drive finalise — materialising proposal §6 risk #2.
    % Phase 3 validates sequential first; the parallel path needs the
    % branch-local-accum protocol from Haskell's wam_haskell_target.pl
    % :1502-1516, deferred to Phase 4.
    Options = [
        module_name('phase3_smoke'),
        emit_mode(lowered),
        intra_query_parallel(false)
    ],
    write_wam_elixir_project(PredWamPairs, Options, ProjectDir),
    write_smoke_driver(ProjectDir).

%% The driver loads the runtime + dispatcher + predicate files and
%% calls the findall predicate with one unbound argument. IO.inspect
%% the result so we can parse stdout from the swipl side.
write_smoke_driver(ProjectDir) :-
    directory_file_path(ProjectDir, 'smoke_driver.exs', DriverPath),
    open(DriverPath, write, S),
    format(S, 'Code.require_file("lib/wam_runtime.ex", __DIR__)~n', []),
    format(S, 'Code.require_file("lib/wam_dispatcher.ex", __DIR__)~n', []),
    format(S, 'Code.require_file("lib/phase3_smoke_p.ex", __DIR__)~n', []),
    format(S, 'Code.require_file("lib/phase3_smoke_findall.ex", __DIR__)~n', []),
    format(S, '~n', []),
    % Pass an unbound var so run/1 returns the materialised binding.
    format(S, 'unbound_l = {:unbound, make_ref()}~n', []),
    format(S, 'result = Phase3Smoke.Phase3SmokeFindall.run([unbound_l])~n', []),
    format(S, 'IO.inspect(result, label: "FINDALL_RESULT", charlists: :as_lists)~n', []),
    close(S).

%% Run `elixir smoke_driver.exs` and capture stdout/stderr.
run_smoke_driver(ProjectDir, StdOut, StdErr, ExitCode) :-
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

%% Assert the driver output contains the three values [a, b, c]
%% (in any order, since fail-driven enumeration order is well-defined
%% but we don't care about it for the smoke — only that all three
%% solutions are present).
assert_findall_result(StdOut) :-
    sub_string(StdOut, _, _, _, "FINDALL_RESULT"),
    sub_string(StdOut, _, _, _, "\"a\""),
    sub_string(StdOut, _, _, _, "\"b\""),
    sub_string(StdOut, _, _, _, "\"c\"").

test_findall_smoke_three_clauses :-
    Test = 'Phase 3 smoke: findall(X, phase3_smoke_p(X), L) → L contains [a, b, c]',
    unique_project_dir(ProjectDir),
    setup_call_cleanup(
        write_smoke_project(ProjectDir),
        (   run_smoke_driver(ProjectDir, StdOut, StdErr, ExitCode),
            (   ExitCode == 0,
                assert_findall_result(StdOut)
            ->  pass(Test)
            ;   format(atom(Reason),
                       'exit=~w stdout=~w stderr=~w',
                       [ExitCode, StdOut, StdErr]),
                fail_test(Test, Reason)
            )
        ),
        catch(delete_directory_and_contents(ProjectDir), _, true)
    ).

%% Test runner

run_tests :-
    format('~n=== WAM-Elixir Lowered Phase 3 Runtime Smoke ===~n~n'),
    (   elixir_available
    ->  test_findall_smoke_three_clauses
    ;   format('% elixir not on PATH — skipping Phase 3 runtime tests~n'),
        format('~n=== Phase 3 Runtime Smoke Skipped ===~n')
    ),
    (   test_failed
    ->  halt(1)
    ;   format('~n=== Phase 3 Runtime Smoke Complete ===~n')
    ).
