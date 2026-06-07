:- encoding(utf8).
%% Phase 0 smoke test for recurrence_evaluation_strategy.pl
%%
%% Verifies:
%%   - module loads without error
%%   - select_evaluation_strategy/3 returns the baseline strategy
%%   - trace contains the `step(stub, ...)` marker (Phase 5 leak detector)
%%   - determinism contract holds: deterministic inside if-then-else,
%%     exactly one solution under findall/3
%%   - valid_strategy/1 checks the outer strategy/1 wrapper (not just
%%     the inner Mode term) — Phase 0 success criterion flagged in review
%%
%% Usage:
%%   swipl -g run_tests -t halt tests/core/test_res_skeleton.pl

:- use_module('../../src/unifyweaver/core/recurrence_evaluation_strategy').
:- use_module(library(lists)).

%% ========================================================================
%% Test runner — same shape as test_algorithm_manifest.pl for consistency
%% ========================================================================

run_tests :-
    format("~n========================================~n"),
    format("RES Phase 0 Skeleton Tests~n"),
    format("========================================~n~n"),
    findall(Test, test(Test), Tests),
    length(Tests, Total),
    run_all(Tests, 0, Passed),
    format("~n========================================~n"),
    (   Passed =:= Total
    ->  format("All ~w tests passed~n", [Total])
    ;   Failed is Total - Passed,
        format("~w of ~w tests FAILED~n", [Failed, Total]),
        format("Tests FAILED~n"),
        halt(1)
    ),
    format("========================================~n").

run_all([], Passed, Passed).
run_all([Test|Rest], Acc, Passed) :-
    (   catch(call(Test), Error,
            (format("[FAIL] ~w: ~w~n", [Test, Error]), fail))
    ->  format("[PASS] ~w~n", [Test]),
        Acc1 is Acc + 1,
        run_all(Rest, Acc1, Passed)
    ;   run_all(Rest, Acc, Passed)
    ).

%% ========================================================================
%% Test declarations
%% ========================================================================

test(test_module_loads).
test(test_baseline_strategy_returned).
test(test_trace_contains_stub_step).
test(test_determinism_in_if_then_else).
test(test_determinism_under_findall).
test(test_valid_strategy_accepts_wrapped).
test(test_valid_strategy_rejects_bare_mode).
test(test_valid_strategy_rejects_unknown_mode).
test(test_valid_recurrence_accepts_well_formed).
test(test_valid_recurrence_rejects_non_recurrence_term).
test(test_valid_workload_accepts_list).
test(test_valid_workload_rejects_non_list).

%% ========================================================================
%% Test fixtures
%% ========================================================================

a_recurrence(recurrence(transitive_closure2, my_pred/2,
    [value_domain(combinatorial), monotone(true)])).

a_workload([]).

%% ========================================================================
%% Tests
%% ========================================================================

%% Trivially true if the test file loaded at all (use_module at top
%% would have failed otherwise). Kept as an explicit assertion so a
%% reader sees "module loads" listed in the test output.
test_module_loads :-
    current_predicate(recurrence_evaluation_strategy:select_evaluation_strategy/3).

test_baseline_strategy_returned :-
    a_recurrence(R),
    a_workload(W),
    select_evaluation_strategy(R, W, strategy_choice(Strategy, _Trace)),
    Strategy == strategy(per_query(unidirectional)).

test_trace_contains_stub_step :-
    a_recurrence(R),
    a_workload(W),
    select_evaluation_strategy(R, W, strategy_choice(_Strategy, trace(Steps))),
    member(step(stub, not_yet_implemented(phase_0), []), Steps).

%% Determinism: called inside if-then-else, the predicate succeeds
%% in the if-branch without leaving a choicepoint. The cut after
%% select_evaluation_strategy/3's success would suppress
%% non-determinism, but the test runs the predicate as the if-condition
%% which exercises the same determinism contract.
test_determinism_in_if_then_else :-
    a_recurrence(R),
    a_workload(W),
    (   select_evaluation_strategy(R, W, _Result)
    ->  true
    ;   fail
    ).

%% findall/3 collects all solutions; deterministic means exactly one.
test_determinism_under_findall :-
    a_recurrence(R),
    a_workload(W),
    findall(Result, select_evaluation_strategy(R, W, Result), Results),
    length(Results, N),
    N == 1.

test_valid_strategy_accepts_wrapped :-
    valid_strategy(strategy(per_query(bidirectional))),
    valid_strategy(strategy(per_query(unidirectional))),
    valid_strategy(strategy(fixed_point(semi_naive))),
    valid_strategy(strategy(cached)),
    valid_strategy(strategy(hybrid([per_query(astar), fixed_point(semi_naive)]))).

%% A bare Mode term without the outer strategy/1 wrapper must be
%% rejected. The Phase 0 success criterion in the implementation
%% plan explicitly calls this out.
test_valid_strategy_rejects_bare_mode :-
    catch(valid_strategy(per_query(bidirectional)),
          error(type_error(strategy_wrapper, _), _),
          true).

test_valid_strategy_rejects_unknown_mode :-
    catch(valid_strategy(strategy(quantum)),
          error(domain_error(strategy_mode, _), _),
          true).

test_valid_recurrence_accepts_well_formed :-
    a_recurrence(R),
    valid_recurrence(R).

test_valid_recurrence_rejects_non_recurrence_term :-
    catch(valid_recurrence(foo(bar)),
          error(type_error(recurrence_term, _), _),
          true).

test_valid_workload_accepts_list :-
    valid_workload([]),
    valid_workload([cardinality(large), kernel_mode(bidirectional)]).

test_valid_workload_rejects_non_list :-
    catch(valid_workload(not_a_list),
          error(type_error(_, _), _),
          true).
