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

% Inner goals are NOT module-qualified — `findall(X, user:p(X), L)`
% triggers a known WAM-compiler bug (Finding 1 from #1647): the
% inner call compiles as a `:/2` builtin that overwrites A1 with the
% module-name string before end_aggregate reads value_reg=A1. Out of
% scope for Phase 3 of the Elixir track.
%
% Scenario 1: 3-clause sequential findall (the original Phase 3a smoke).
user:phase3_smoke_p('a').
user:phase3_smoke_p('b').
user:phase3_smoke_p('c').
user:phase3_smoke_findall(L) :- findall(X, phase3_smoke_p(X), L).

% Scenario 2: empty-result findall.
%   findall(X, fail, L) → L = []. fail/0 hits the execute_builtin
%   catch-all (returns :fail), the lowering throws fail, backtrack
%   pops the agg frame and finalises with accum=[] → empty list.
user:phase3_findall_fail(L) :- findall(X, fail, L).

% Scenario 3: aggregate_all(count, ..., N) → integer length of accum.
user:phase3_pn(1).
user:phase3_pn(2).
user:phase3_pn(3).
user:phase3_count(N) :- aggregate_all(count, phase3_pn(X), N).

% Scenario 4: aggregate_all(max(X), fail, M) → predicate fails.
%   accum is empty when finalise_aggregate hits the :max branch;
%   per the substrate's empty-aggregator semantics (PR #1627 review
%   round), :max throws {:fail, state} which propagates up to run/1's
%   outer catch, which returns :fail. Validates the deviation from
%   proposal §6 risk #6 documented in WAM_ELIXIR_TIER2_FINDALL.md §0.
user:phase3_max_empty(M) :- aggregate_all(max(X), fail, M).

% Scenarios 5–8: aggregator coverage on non-empty accum. Each uses
% the existing 3-clause phase3_smoke_p/1 fixture — no new fact
% predicates needed. Together these cover the four
% finalise_aggregate branches that hadn't run end-to-end yet:
% :bag (collect-with-duplicates), :set (Enum.uniq), :max non-empty
% (Enum.max), :min non-empty (Enum.min).
user:phase3_bag(B) :- aggregate_all(bag(X), phase3_smoke_p(X), B).
user:phase3_set(S) :- aggregate_all(set(X), phase3_smoke_p(X), S).
user:phase3_max(M) :- aggregate_all(max(X), phase3_smoke_p(X), M).
user:phase3_min(M) :- aggregate_all(min(X), phase3_smoke_p(X), M).

% Scenario 9: module-qualified findall, end-to-end — closes Finding 1
% from #1647. The static-module-qualifier unwrap in compile_goal_call/4
% (#1658) emits a regular `call phase3_smoke_p/1` for the inner
% goal instead of routing through the `:/2` builtin path, so the
% Elixir runtime never sees `:/2` and never hits its catch-all
% `_ -> :fail`. Functionally equivalent to scenario 1 once the
% unwrap fires. Dynamic module qualifiers (Module = m, Module:p(X))
% still hit `:/2` and remain a known limitation — see
% WAM_ELIXIR_TIER2_FINDALL.md for the forward reference.
user:phase3_findall_qualified(L) :- findall(X, user:phase3_smoke_p(X), L).

% Scenario 10: single-clause findall — degenerate case. Inner predicate
% has exactly one solution (no try_me_else / retry_me_else CPs).
% Validates the path where the inner enumeration exhausts immediately
% after one solution: aggregate_collect captures the value, throw fail
% drives backtrack, the topmost CP IS the agg frame (no inner CPs to
% pop first), and finalise binds the 1-element list.
user:phase3_one_solution(only).
user:phase3_findall_one(L) :- findall(X, phase3_one_solution(X), L).

% Scenario 11: nested findall — proposal §6 risk #7. Deferred from
% the Phase 3c trail-bind PR (#1659) and now activated by this
% PR's finalise-pops-env-frame fix (option (a) from that PR's
% deferred-scenario comment block).
%
% Outer findall iterates inner-findall results. The inner
% phase3_nested_inner/1 is single-clause and deterministically
% returns [a, b, c], so the outer collects exactly one solution
% whose value is that inner list. Validates the full chain:
%   - inner finalise binds L through the trail (PR #1659)
%   - inner finalise pops inner_env from state.stack (this PR), so
%     the saved cp (= outer's post-call k1) runs with the OUTER's
%     Y-regs active
%   - outer's end_aggregate Y1 reads the now-correctly-bound L
%   - outer aggregate_collect captures it
%   - outer finalise binds LL through the trail
%   - outer finalise pops outer_env, tail-calls terminal_cp
user:phase3_nested_inner(L) :- findall(X, phase3_smoke_p(X), L).
user:phase3_nested_outer(LL) :- findall(L, phase3_nested_inner(L), LL).

% Scenario 12: compound Template findall — proposal §6 risk #1 probe
% (deep-copy of compound captured values). Surfaced and resolved in
% this PR via two changes:
%
%   1. compile_aggregate_all/5 now detects compound Templates
%      (`Template = compound(...)`), allocates Y-regs for variable
%      args via the new compile_compound_template/5 helper, and
%      emits put_structure + set_value/set_constant code AFTER the
%      inner-goal call so each iteration constructs a fresh heap
%      structure. Without this the WAM emitted no construction
%      instructions and end_aggregate captured only A1 (the first
%      Template arg), losing the other args entirely.
%
%   2. WamRuntime.aggregate_collect/2 now deep-copies the captured
%      value via deep_copy_value/2 before adding to accum.
%      For atomic captured values (the prior cases, scenarios 1–11)
%      deep-copy is a no-op pass-through; for compound `{:ref, addr}`
%      values it walks the heap structure and builds a self-contained
%      {:struct, "name/arity", [args]} tuple. Without deep-copy all
%      accum entries would point to the same final iterations heap
%      addresses (heap_len rewinds during backtrack so put_structure
%      reuses the same addrs).
user:phase3_q(a, '1').
user:phase3_q(b, '2').
user:phase3_q(c, '3').
user:phase3_findall_compound(L) :- findall(p(X, Y), phase3_q(X, Y), L).

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

%% phase3_predicates(-List)
%  All predicates compiled into the smoke project. Each is a
%  Module:Pred/Arity term. The order matters only for dispatcher
%  registration — the runtime resolves by name, not position.
phase3_predicates([
    user:phase3_smoke_p/1,
    user:phase3_smoke_findall/1,
    user:phase3_findall_fail/1,
    user:phase3_pn/1,
    user:phase3_count/1,
    user:phase3_max_empty/1,
    user:phase3_bag/1,
    user:phase3_set/1,
    user:phase3_max/1,
    user:phase3_min/1,
    user:phase3_findall_qualified/1,
    user:phase3_one_solution/1,
    user:phase3_findall_one/1,
    user:phase3_nested_inner/1,
    user:phase3_nested_outer/1,
    user:phase3_q/2,
    user:phase3_findall_compound/1
]).

%% phase3_driver_invocations(-Lines)
%  Each scenario gets one line that calls the predicate with an
%  unbound argument and one line that IO.inspect's the result with
%  a unique label. The label is the parsing key on the swipl side.
phase3_driver_invocations([
    'r1 = Phase3Smoke.Phase3SmokeFindall.run([{:unbound, make_ref()}])',
    'IO.inspect(r1, label: "SCENARIO_FINDALL_THREE", charlists: :as_lists)',
    'r2 = Phase3Smoke.Phase3FindallFail.run([{:unbound, make_ref()}])',
    'IO.inspect(r2, label: "SCENARIO_FINDALL_EMPTY", charlists: :as_lists)',
    'r3 = Phase3Smoke.Phase3Count.run([{:unbound, make_ref()}])',
    'IO.inspect(r3, label: "SCENARIO_AGG_COUNT", charlists: :as_lists)',
    'r4 = Phase3Smoke.Phase3MaxEmpty.run([{:unbound, make_ref()}])',
    'IO.inspect(r4, label: "SCENARIO_AGG_MAX_EMPTY", charlists: :as_lists)',
    'r5 = Phase3Smoke.Phase3Bag.run([{:unbound, make_ref()}])',
    'IO.inspect(r5, label: "SCENARIO_AGG_BAG", charlists: :as_lists)',
    'r6 = Phase3Smoke.Phase3Set.run([{:unbound, make_ref()}])',
    'IO.inspect(r6, label: "SCENARIO_AGG_SET", charlists: :as_lists)',
    'r7 = Phase3Smoke.Phase3Max.run([{:unbound, make_ref()}])',
    'IO.inspect(r7, label: "SCENARIO_AGG_MAX", charlists: :as_lists)',
    'r8 = Phase3Smoke.Phase3Min.run([{:unbound, make_ref()}])',
    'IO.inspect(r8, label: "SCENARIO_AGG_MIN", charlists: :as_lists)',
    'r9 = Phase3Smoke.Phase3FindallQualified.run([{:unbound, make_ref()}])',
    'IO.inspect(r9, label: "SCENARIO_FINDALL_QUALIFIED", charlists: :as_lists)',
    'r10 = Phase3Smoke.Phase3FindallOne.run([{:unbound, make_ref()}])',
    'IO.inspect(r10, label: "SCENARIO_FINDALL_ONE_CLAUSE", charlists: :as_lists)',
    'r11 = Phase3Smoke.Phase3NestedOuter.run([{:unbound, make_ref()}])',
    'IO.inspect(r11, label: "SCENARIO_FINDALL_NESTED", charlists: :as_lists)',
    'r12 = Phase3Smoke.Phase3FindallCompound.run([{:unbound, make_ref()}])',
    'IO.inspect(r12, label: "SCENARIO_FINDALL_COMPOUND", charlists: :as_lists)'
]).

%% Generate the test project. Predicates and driver invocations are
%% read from phase3_predicates/1 and phase3_driver_invocations/1 so
%% adding a scenario is one fixture + two driver lines + one assertion.
write_smoke_project(ProjectDir) :-
    phase3_predicates(Predicates),
    findall(P/A-WamCode, (
        member(M:P/A, Predicates),
        wam_target:compile_predicate_to_wam(M:P/A, [], WamCode)
    ), PredWamPairs),
    % intra_query_parallel(false) kill-switches the Tier-2 super-wrapper.
    % Without it, the 3-clause phase3_smoke_p/1 gets Tier-2-eligible, its
    % super-wrapper fires (because the parent findall pushed an agg
    % frame, so in_forkable_aggregate_frame? returns true), and the
    % parallel arm returns from merge_into_aggregate instead of throwing
    % fail to drive finalise — materialising proposal §6 risk #2 (Finding
    % 2 from #1647). Phase 3 validates sequential first; the parallel
    % path needs the branch-local-accum protocol from Haskell's
    % wam_haskell_target.pl:1502-1516, deferred to Phase 4.
    Options = [
        module_name('phase3_smoke'),
        emit_mode(lowered),
        intra_query_parallel(false)
    ],
    write_wam_elixir_project(PredWamPairs, Options, ProjectDir),
    write_smoke_driver(ProjectDir).

%% The driver loads the runtime + dispatcher + every predicate file
%% and runs every scenario from phase3_driver_invocations/1.
write_smoke_driver(ProjectDir) :-
    directory_file_path(ProjectDir, 'smoke_driver.exs', DriverPath),
    phase3_predicates(Predicates),
    phase3_driver_invocations(Invocations),
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

%% Per-scenario assertions. Each takes the full StdOut from one
%% elixir invocation and checks for its scenario's label + expected
%% shape. Scenarios share the project + the elixir process for
%% performance — one ~3s elixir cold-start covers all scenarios.

%% scope_after_label(+StdOut, +Label, -Window)
%  Returns a substring of StdOut starting at Label and ending at the
%  next SCENARIO_ label (or end of stream). Avoids a scenario's
%  assertion accidentally matching another scenario's output.
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

assert_scenario_findall_three(StdOut) :-
    scope_after_label(StdOut, "SCENARIO_FINDALL_THREE", W),
    sub_string(W, _, _, _, "\"a\""),
    sub_string(W, _, _, _, "\"b\""),
    sub_string(W, _, _, _, "\"c\"").

assert_scenario_findall_empty(StdOut) :-
    scope_after_label(StdOut, "SCENARIO_FINDALL_EMPTY", W),
    % Empty list materialises as `{:ok, [[]]}` — outer list wraps
    % the run/1 args, inner [] is the empty findall result.
    sub_string(W, _, _, _, "[]").

assert_scenario_agg_count(StdOut) :-
    scope_after_label(StdOut, "SCENARIO_AGG_COUNT", W),
    % Count of 3 phase3_pn/1 facts.
    sub_string(W, _, _, _, "3").

assert_scenario_agg_max_empty(StdOut) :-
    scope_after_label(StdOut, "SCENARIO_AGG_MAX_EMPTY", W),
    % :max on empty accum throws fail → run/1's outer catch
    % returns :fail. The IO.inspect output is bare `:fail`.
    sub_string(W, _, _, _, ":fail").

assert_scenario_agg_bag(StdOut) :-
    scope_after_label(StdOut, "SCENARIO_AGG_BAG", W),
    % :bag returns the reversed accumulator — same as :findall.
    % All three values present in the scenario's stdout window.
    sub_string(W, _, _, _, "\"a\""),
    sub_string(W, _, _, _, "\"b\""),
    sub_string(W, _, _, _, "\"c\"").

assert_scenario_agg_set(StdOut) :-
    scope_after_label(StdOut, "SCENARIO_AGG_SET", W),
    % :set applies Enum.uniq. Three distinct values in, three out.
    sub_string(W, _, _, _, "\"a\""),
    sub_string(W, _, _, _, "\"b\""),
    sub_string(W, _, _, _, "\"c\"").

assert_scenario_agg_max(StdOut) :-
    scope_after_label(StdOut, "SCENARIO_AGG_MAX", W),
    % Enum.max on string list is lex order — max("a","b","c") = "c".
    sub_string(W, _, _, _, "\"c\"").

assert_scenario_agg_min(StdOut) :-
    scope_after_label(StdOut, "SCENARIO_AGG_MIN", W),
    % Enum.min on string list — min("a","b","c") = "a".
    sub_string(W, _, _, _, "\"a\"").

assert_scenario_findall_qualified(StdOut) :-
    scope_after_label(StdOut, "SCENARIO_FINDALL_QUALIFIED", W),
    % Module-qualified findall now works end-to-end after the static
    % unwrap in compile_goal_call/4. Same expected shape as scenario 1
    % since the unwrap makes the WAM byte-code identical to the
    % unqualified case — three solution values present.
    sub_string(W, _, _, _, "\"a\""),
    sub_string(W, _, _, _, "\"b\""),
    sub_string(W, _, _, _, "\"c\"").

assert_scenario_findall_one_clause(StdOut) :-
    scope_after_label(StdOut, "SCENARIO_FINDALL_ONE_CLAUSE", W),
    % One-element list — the single fact's value `only` is captured
    % once. No try_me_else CPs to backtrack through; finalise is
    % reached on the very first throw-fail after aggregate_collect.
    sub_string(W, _, _, _, "\"only\"").

assert_scenario_findall_nested(StdOut) :-
    scope_after_label(StdOut, "SCENARIO_FINDALL_NESTED", W),
    % Nested findall: outer iterates inner-findall results. The
    % inner phase3_nested_inner/1 deterministically returns
    % [a, b, c]; outer collects exactly one solution whose value
    % is that inner list. After the finalise-pops-env-frame fix
    % in this PR, all three values are correctly bound through
    % the trail across both aggregate frames and the parameter
    % slot resolves to the nested list.
    sub_string(W, _, _, _, "\"a\""),
    sub_string(W, _, _, _, "\"b\""),
    sub_string(W, _, _, _, "\"c\"").

assert_scenario_findall_compound(StdOut) :-
    scope_after_label(StdOut, "SCENARIO_FINDALL_COMPOUND", W),
    % Compound Template findall: each iteration constructs a fresh
    % p(X, Y) on the heap and aggregate_collect deep-copies it into
    % a self-contained {:struct, "p/2", [X, Y]} tuple. After all
    % three iterations, L should contain three distinct {:struct,
    % ...} tuples — sub-term identity preserved per Perplexity's
    % "sharper probe" recommendation. We assert all six sub-term
    % values are present in the scenario's stdout window. If
    % deep-copy was missing, all three list elements would point
    % to the same heap region (the final iteration's p(c, "3"))
    % so "a", "b", "1", "2" would be absent.
    sub_string(W, _, _, _, ":struct"),
    sub_string(W, _, _, _, "p/2"),
    sub_string(W, _, _, _, "\"a\""),
    sub_string(W, _, _, _, "\"b\""),
    sub_string(W, _, _, _, "\"c\""),
    sub_string(W, _, _, _, "\"1\""),
    sub_string(W, _, _, _, "\"2\""),
    sub_string(W, _, _, _, "\"3\"").

%% Per-scenario test wrappers that share captured StdOut/StdErr/ExitCode.

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
    run_scenario('Phase 3: findall(X, phase3_smoke_p(X), L) → [a, b, c]',
                 assert_scenario_findall_three,
                 StdOut, StdErr, ExitCode),
    run_scenario('Phase 3: findall(X, fail, L) → []',
                 assert_scenario_findall_empty,
                 StdOut, StdErr, ExitCode),
    run_scenario('Phase 3: aggregate_all(count, phase3_pn(X), N) → N = 3',
                 assert_scenario_agg_count,
                 StdOut, StdErr, ExitCode),
    run_scenario('Phase 3: aggregate_all(max(X), fail, M) → :fail (empty bag, no identity)',
                 assert_scenario_agg_max_empty,
                 StdOut, StdErr, ExitCode),
    run_scenario('Phase 3: aggregate_all(bag(X), phase3_smoke_p(X), B) → [a, b, c]',
                 assert_scenario_agg_bag,
                 StdOut, StdErr, ExitCode),
    run_scenario('Phase 3: aggregate_all(set(X), phase3_smoke_p(X), S) → [a, b, c] (uniq)',
                 assert_scenario_agg_set,
                 StdOut, StdErr, ExitCode),
    run_scenario('Phase 3: aggregate_all(max(X), phase3_smoke_p(X), M) → "c" (lex max)',
                 assert_scenario_agg_max,
                 StdOut, StdErr, ExitCode),
    run_scenario('Phase 3: aggregate_all(min(X), phase3_smoke_p(X), M) → "a" (lex min)',
                 assert_scenario_agg_min,
                 StdOut, StdErr, ExitCode),
    run_scenario('Phase 3: findall(X, user:phase3_smoke_p(X), L) → [a, b, c] (module-qualified, closes #1647 Finding 1)',
                 assert_scenario_findall_qualified,
                 StdOut, StdErr, ExitCode),
    run_scenario('Phase 3c: findall(X, phase3_one_solution(X), L) → ["only"] (single-clause, no try_me_else CPs)',
                 assert_scenario_findall_one_clause,
                 StdOut, StdErr, ExitCode),
    run_scenario('Phase 3c: nested findall — outer iterates inner-findall results (proposal §6 risk #7)',
                 assert_scenario_findall_nested,
                 StdOut, StdErr, ExitCode),
    run_scenario('Phase 3c: findall(p(X,Y), q(X,Y), L) — compound Template + deep-copy (proposal §6 risk #1)',
                 assert_scenario_findall_compound,
                 StdOut, StdErr, ExitCode).

%% Test runner — single project, single elixir invocation, all
%% assertions on the captured stdout.

run_tests :-
    format('~n=== WAM-Elixir Lowered Phase 3 Runtime Smoke ===~n~n'),
    (   elixir_available
    ->  unique_project_dir(ProjectDir),
        setup_call_cleanup(
            write_smoke_project(ProjectDir),
            (   run_smoke_driver(ProjectDir, StdOut, StdErr, ExitCode),
                run_all_scenarios(StdOut, StdErr, ExitCode)
            ),
            catch(delete_directory_and_contents(ProjectDir), _, true)
        )
    ;   format('% elixir not on PATH — skipping Phase 3 runtime tests~n'),
        format('~n=== Phase 3 Runtime Smoke Skipped ===~n')
    ),
    (   test_failed
    ->  halt(1)
    ;   format('~n=== Phase 3 Runtime Smoke Complete ===~n')
    ).
