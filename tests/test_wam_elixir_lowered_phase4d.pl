:- encoding(utf8).
% Phase 4d runtime smoke: nested-fork suppression via parallel_depth gate.
%
% Phase 4 proposal §4.6 (WAM_ELIXIR_TIER2_FINDALL_PHASE4.md:175–177)
% claims the `parallel_depth > 0` gate in the Tier-2 super-wrapper
% short-circuits nested forks to sequential. The gate is at
% wam_elixir_lowered_emitter.pl:1194 — when an outer Tier-2 fan-out
% is in progress (state.parallel_depth = 1), any inner Tier-2-eligible
% predicate's super-wrapper falls back to its sequential `_impl`
% chain instead of fanning out a second time.
%
% Section 6 risk #4 (line 205) flagged this as needing an explicit
% test fixture. This file is that fixture.
%
% Setup:
%   - phase4d_inner/1: 3-clause pure predicate (Tier-2 eligible).
%   - phase4d_outer_p/1: 3-clause pure predicate. Each clause body
%     invokes findall(_, phase4d_inner(_), _) — the inner predicate's
%     super-wrapper is on the call path.
%   - phase4d_nested/1: top-level, single clause, calls findall over
%     outer_p. Not Tier-2-eligible itself (1 clause), but its findall
%     routes to outer_p's super-wrapper with parallel_depth = 0.
%
% Runtime sequence:
%   1. Top-level findall over outer_p → outer's super-wrapper sees
%      parallel_depth = 0, fans out into 3 Task.async_stream branches.
%   2. Each branch executes one outer_p clause body, which invokes
%      findall over phase4d_inner. The inner findall hits inner's
%      super-wrapper with parallel_depth = 1 — gate fires — inner
%      runs sequentially.
%   3. Each branch's outer_p head binds 'a' / 'b' / 'c'; the parent
%      collects all three.
%
% Failure modes prevented by the gate:
%   - BEAM scheduler exhaustion from unbounded fork nesting.
%   - Sub-branch result corruption (each inner fan-out would need
%     its own merge protocol; sequential avoids this entirely).
%
% Assertion: result is the set {'a', 'b', 'c'} — same as the non-
% nested Phase 4c findall. Correctness here is the primary signal;
% the secondary signal is that `elixir smoke_driver.exs` returns
% with exit 0 (no fork bomb / scheduler timeout).
%
% Usage: swipl -g run_tests -t halt tests/test_wam_elixir_lowered_phase4d.pl
%
% Skipped gracefully when `elixir` is not on PATH.

:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module(library(option)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_elixir_target',
              [write_wam_elixir_project/3]).
:- use_module('../src/unifyweaver/core/clause_body_analysis').

:- dynamic test_failed/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

%% PATH guard — skip the suite cleanly if `elixir` is unavailable.
elixir_available :-
    catch(
        process_create(path(elixir), ['--version'],
                       [stdout(pipe(Out)), stderr(null), process(PID)]),
        _,
        fail),
    read_string(Out, _, _),
    close(Out),
    process_wait(PID, exit(0)).

%% Test fixtures.

:- dynamic user:phase4d_inner/1.
:- dynamic user:phase4d_outer_p/1.
:- dynamic user:phase4d_nested/1.

% Inner: 3-clause Tier-2-eligible predicate.
user:phase4d_inner(1).
user:phase4d_inner(2).
user:phase4d_inner(3).

% Outer: 3-clause Tier-2-eligible predicate. Each clause body invokes
% findall over phase4d_inner — that findall's compiled call routes
% through inner's super-wrapper at runtime, where the parallel_depth
% gate fires (depth = 1 from outer's fan-out).
%
% This direct-body shape originally surfaced a WAM compiler bug
% (multi-clause + body-level findall = no env frame = caller cp lost
% across finalise_aggregate). The fix in compile_clauses_fragments/8
% now forces an env frame whenever the body contains a findall or
% aggregate_all. The original-shape fixture below is the runtime
% regression check for that fix.
user:phase4d_outer_p('a') :- findall(_, phase4d_inner(_), _).
user:phase4d_outer_p('b') :- findall(_, phase4d_inner(_), _).
user:phase4d_outer_p('c') :- findall(_, phase4d_inner(_), _).

% Top-level: not Tier-2-eligible (1 clause). Its findall routes to
% outer_p's super-wrapper with parallel_depth = 0 — outer fans out.
user:phase4d_nested(L) :- findall(X, phase4d_outer_p(X), L).

phase4d_predicates([
    user:phase4d_inner/1,
    user:phase4d_outer_p/1,
    user:phase4d_nested/1
]).

phase4d_purity_decls([
    user:phase4d_inner/1,
    user:phase4d_outer_p/1
]).

phase4d_driver_invocations([
    'r1 = Phase4dSmoke.Phase4dNested.run([{:unbound, make_ref()}])',
    'IO.inspect(r1, label: "SCENARIO_NESTED_FORK_SUPPRESSION", charlists: :as_lists)'
]).

%% tmp_root — same fallback chain as Phase 3 / 4c.
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
    format(atom(Name), 'uw_elixir_phase4d_smoke_~w', [Stamp]),
    directory_file_path(Root, Name, Dir).

%% Generate the test project. Same parallel-enabled setup as Phase 4c:
%  purity decls asserted before compile, intra_query_parallel(false)
%  omitted from Options.
write_smoke_project(ProjectDir) :-
    phase4d_predicates(Predicates),
    phase4d_purity_decls(PurityDecls),
    forall(member(Pred, PurityDecls),
           assertz(clause_body_analysis:order_independent(Pred))),
    findall(P/A-WamCode, (
        member(M:P/A, Predicates),
        wam_target:compile_predicate_to_wam(M:P/A, [], WamCode)
    ), PredWamPairs),
    Options = [
        module_name('phase4d_smoke'),
        emit_mode(lowered)
    ],
    write_wam_elixir_project(PredWamPairs, Options, ProjectDir),
    write_smoke_driver(ProjectDir).

cleanup_purity_decls :-
    phase4d_purity_decls(PurityDecls),
    forall(member(Pred, PurityDecls),
           ignore(retract(clause_body_analysis:order_independent(Pred)))).

write_smoke_driver(ProjectDir) :-
    directory_file_path(ProjectDir, 'smoke_driver.exs', DriverPath),
    phase4d_predicates(Predicates),
    phase4d_driver_invocations(Invocations),
    open(DriverPath, write, S),
    format(S, 'Code.require_file("lib/wam_runtime.ex", __DIR__)~n', []),
    format(S, 'Code.require_file("lib/wam_dispatcher.ex", __DIR__)~n', []),
    forall(member(_:P/_, Predicates), (
        atom_string(P, PStr),
        format(S, 'Code.require_file("lib/~w.ex", __DIR__)~n', [PStr])
    )),
    format(S, '~n', []),
    forall(member(Line, Invocations),
           format(S, '~w~n', [Line])),
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

%% Scope after label so substring matches don't cross-contaminate.
scope_after_label(StdOut, Label, Window) :-
    sub_string(StdOut, LabelOffset, _, _, Label),
    AfterLabel is LabelOffset + 1,
    string_length(StdOut, TotalLen),
    Remaining is TotalLen - AfterLabel,
    sub_string(StdOut, AfterLabel, Remaining, 0, Tail),
    (   sub_string(Tail, NextLabel, _, _, "SCENARIO_")
    ->  sub_string(Tail, 0, NextLabel, _, Window)
    ;   Window = Tail
    ).

%% Set-equality assertion: 'a', 'b', 'c' must all appear. Order is
%  non-deterministic because the outer fans out; the inner runs
%  sequentially per branch but branch completion order varies.
assert_scenario_nested(StdOut) :-
    scope_after_label(StdOut, "SCENARIO_NESTED_FORK_SUPPRESSION", W),
    sub_string(W, _, _, _, "\"a\""),
    sub_string(W, _, _, _, "\"b\""),
    sub_string(W, _, _, _, "\"c\"").

run_scenario(Test, Assertion, StdOut, StdErr, ExitCode) :-
    (   ExitCode == 0,
        call(Assertion, StdOut)
    ->  pass(Test)
    ;   format(atom(Reason),
               'exit=~w assertion failed; stderr=~w',
               [ExitCode, StdErr]),
        fail_test(Test, Reason)
    ).

run_all_scenarios(StdOut, StdErr, ExitCode) :-
    run_scenario('Phase 4d: nested-fork suppression — outer parallel, inner sequential → set {a, b, c}',
                 assert_scenario_nested,
                 StdOut, StdErr, ExitCode).

run_tests :-
    format('~n=== WAM-Elixir Lowered Phase 4d Nested-Fork Suppression Smoke ===~n~n'),
    (   elixir_available
    ->  unique_project_dir(ProjectDir),
        setup_call_cleanup(
            write_smoke_project(ProjectDir),
            (   run_smoke_driver(ProjectDir, StdOut, StdErr, ExitCode),
                run_all_scenarios(StdOut, StdErr, ExitCode)
            ),
            (   catch(delete_directory_and_contents(ProjectDir), _, true),
                cleanup_purity_decls
            )
        )
    ;   format('% elixir not on PATH — skipping Phase 4d nested-fork tests~n'),
        format('~n=== Phase 4d Nested-Fork Suppression Smoke Skipped ===~n')
    ),
    (   test_failed
    ->  halt(1)
    ;   format('~n=== Phase 4d Nested-Fork Suppression Smoke Complete ===~n')
    ).
