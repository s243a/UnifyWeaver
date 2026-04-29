:- encoding(utf8).
% Phase 4c runtime smoke: parallel findall via Tier-2 super-wrapper.
%
% Mirrors the Phase 3 sequential harness (tests/test_wam_elixir_lowered
% _phase3.pl) but flips two switches:
%
%   1. Project options DROP `intra_query_parallel(false)` — let par_wrap
%      _segment/4's gate fire when its three static gates pass (purity
%      certificate, ≥3 clauses, kill-switch absent). The super-wrapper
%      from #1706 (with #1708's _branch variants) emits and runs.
%
%   2. Tier-2-eligible predicates are declared pure via
%      clause_body_analysis:order_independent/1 so the purity gate
%      accepts them.
%
% Assertions use SET-equality (presence of expected values, ignoring
% order) because Task.async_stream(ordered: false) returns branch
% results in non-deterministic completion order. Tier-2's purity gate
% restricts forkable aggregators to order-independent ones (findall,
% bag, set, count, sum, max, min — all commutative/associative or
% multi-set semantics), so order independence is part of the contract.
%
% Usage: swipl -g run_tests -t halt tests/test_wam_elixir_lowered_phase4c.pl
%
% Skipped gracefully when `elixir` is not on PATH (matches the
% Phase 3 / Clojure / Haskell phase-3 conventions).

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

%% Test fixtures — declared pure so the Tier-2 purity gate accepts.
%  Note: order_independent/1 is asserted at module load time; the
%  test setup ensures it is in place when write_wam_elixir_project/3
%  runs par_wrap_segment/4's purity check.

:- dynamic user:phase4c_p/1.
:- dynamic user:phase4c_pn/1.
:- dynamic user:phase4c_findall/1.
:- dynamic user:phase4c_count/1.
:- dynamic user:phase4c_max/1.

% Scenario 1: parallel findall over a 3-clause pure predicate.
% Each clause fans out as a Task.async_stream branch; results are
% concatenated by the parent's merge_into_aggregate call.
user:phase4c_p('a').
user:phase4c_p('b').
user:phase4c_p('c').
user:phase4c_findall(L) :- findall(X, phase4c_p(X), L).

% Scenario 2: parallel aggregate_all(count, ...) — N is order-
% independent so the parallel result must equal the sequential
% one (3).
user:phase4c_pn(1).
user:phase4c_pn(2).
user:phase4c_pn(3).
user:phase4c_count(N) :- aggregate_all(count, phase4c_pn(X), N).

% Scenario 3: parallel aggregate_all(max(X), ...) — max picks the
% lex-greatest from the merged accum. Branch order doesn't matter;
% Enum.max in finalise_aggregate sees the same multiset of values
% regardless of which branch returned which value.
user:phase4c_max(M) :- aggregate_all(max(X), phase4c_p(X), M).

phase4c_predicates([
    user:phase4c_p/1,
    user:phase4c_pn/1,
    user:phase4c_findall/1,
    user:phase4c_count/1,
    user:phase4c_max/1
]).

phase4c_purity_decls([
    user:phase4c_p/1,
    user:phase4c_pn/1
]).

phase4c_driver_invocations([
    'r1 = Phase4cSmoke.Phase4cFindall.run([{:unbound, make_ref()}])',
    'IO.inspect(r1, label: "SCENARIO_PARALLEL_FINDALL", charlists: :as_lists)',
    'r2 = Phase4cSmoke.Phase4cCount.run([{:unbound, make_ref()}])',
    'IO.inspect(r2, label: "SCENARIO_PARALLEL_COUNT", charlists: :as_lists)',
    'r3 = Phase4cSmoke.Phase4cMax.run([{:unbound, make_ref()}])',
    'IO.inspect(r3, label: "SCENARIO_PARALLEL_MAX", charlists: :as_lists)'
]).

%% tmp_root — same fallback chain as Phase 3.
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
    format(atom(Name), 'uw_elixir_phase4c_smoke_~w', [Stamp]),
    directory_file_path(Root, Name, Dir).

%% Generate the test project. KEY DIFFERENCE from Phase 3:
%  - Asserts purity declarations BEFORE compile so the gate fires.
%  - Project Options OMIT `intra_query_parallel(false)` — defaults
%    to parallel-enabled. par_wrap_segment/4 will emit the super-
%    wrapper for any Tier-2-eligible predicate.
write_smoke_project(ProjectDir) :-
    phase4c_predicates(Predicates),
    phase4c_purity_decls(PurityDecls),
    forall(member(Pred, PurityDecls),
           assertz(clause_body_analysis:order_independent(Pred))),
    findall(P/A-WamCode, (
        member(M:P/A, Predicates),
        wam_target:compile_predicate_to_wam(M:P/A, [], WamCode)
    ), PredWamPairs),
    Options = [
        module_name('phase4c_smoke'),
        emit_mode(lowered)
        % NOTE: intra_query_parallel(false) intentionally omitted.
    ],
    write_wam_elixir_project(PredWamPairs, Options, ProjectDir),
    write_smoke_driver(ProjectDir).

cleanup_purity_decls :-
    phase4c_purity_decls(PurityDecls),
    forall(member(Pred, PurityDecls),
           ignore(retract(clause_body_analysis:order_independent(Pred)))).

%% Driver follows the Phase 3 shape — Code.require_file the runtime,
%% dispatcher, and per-predicate modules; emit one IO.inspect per
%% scenario with a unique label.
write_smoke_driver(ProjectDir) :-
    directory_file_path(ProjectDir, 'smoke_driver.exs', DriverPath),
    phase4c_predicates(Predicates),
    phase4c_driver_invocations(Invocations),
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

%% scope_after_label — same as Phase 3, narrows to one scenario's
%% stdout window so substring matches don't cross-contaminate.
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

%% Per-scenario assertions — set-equality semantics. Each expected
%% value must appear; order is non-deterministic.
assert_scenario_parallel_findall(StdOut) :-
    scope_after_label(StdOut, "SCENARIO_PARALLEL_FINDALL", W),
    sub_string(W, _, _, _, "\"a\""),
    sub_string(W, _, _, _, "\"b\""),
    sub_string(W, _, _, _, "\"c\"").

assert_scenario_parallel_count(StdOut) :-
    scope_after_label(StdOut, "SCENARIO_PARALLEL_COUNT", W),
    % count is order-independent; expected value 3.
    sub_string(W, _, _, _, "3").

assert_scenario_parallel_max(StdOut) :-
    scope_after_label(StdOut, "SCENARIO_PARALLEL_MAX", W),
    % Lex max of {"a", "b", "c"} = "c". Order-independent.
    sub_string(W, _, _, _, "\"c\"").

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
    run_scenario('Phase 4c: parallel findall(X, phase4c_p(X), L) → set {a, b, c}',
                 assert_scenario_parallel_findall,
                 StdOut, StdErr, ExitCode),
    run_scenario('Phase 4c: parallel aggregate_all(count, phase4c_pn(X), N) → 3',
                 assert_scenario_parallel_count,
                 StdOut, StdErr, ExitCode),
    run_scenario('Phase 4c: parallel aggregate_all(max(X), phase4c_p(X), M) → "c"',
                 assert_scenario_parallel_max,
                 StdOut, StdErr, ExitCode).

%% Test runner — single project, single elixir invocation.

run_tests :-
    format('~n=== WAM-Elixir Lowered Phase 4c Parallel Runtime Smoke ===~n~n'),
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
    ;   format('% elixir not on PATH — skipping Phase 4c parallel runtime tests~n'),
        format('~n=== Phase 4c Parallel Runtime Smoke Skipped ===~n')
    ),
    (   test_failed
    ->  halt(1)
    ;   format('~n=== Phase 4c Parallel Runtime Smoke Complete ===~n')
    ).
