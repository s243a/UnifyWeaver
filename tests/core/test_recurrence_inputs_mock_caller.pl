:- encoding(utf8).
%% Mock-second-caller test for recurrence_inputs.pl.
%%
%% Exercises build_recurrence_term/3 and build_workload_signals/2
%% with workload signals that have no F# semantics. Acts as a
%% stand-in for the eventual Haskell-WAM or C-WAM target integration:
%% if the helpers had been silently shaped by F# WAM as the only
%% first-iteration caller, this mock would catch the
%% F#-shape-calcification.
%%
%% Per the SPEC's concrete signal list: cardinality(large),
%% query_frequency(high), query_pattern(single_pair), b_eff(15.0),
%% branching_d(4.5), contraction_r(0.04), heuristic_predicate_available(true),
%% heuristic_predicate(my_heuristic/2). Critically excludes csr_path
%% (most-commonly F#-WAM-specific per SPEC's signal table).
%%
%% Usage:
%%   swipl -g run_tests -t halt tests/core/test_recurrence_inputs_mock_caller.pl

:- use_module('../../src/unifyweaver/core/recurrence_inputs').
:- use_module('../../src/unifyweaver/core/recurrence_evaluation_strategy',
              [select_evaluation_strategy/3]).
:- use_module(library(lists)).

%% ========================================================================
%% Test runner
%% ========================================================================

run_tests :-
    format("~n========================================~n"),
    format("recurrence_inputs Mock-Second-Caller Test~n"),
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
%% Test fixtures — non-F# scenario shaping
%% ========================================================================

%% A Haskell-WAM-style detected kernel (uses transitive_closure2,
%% the kernel kind any WAM target would consume).
mock_detected_kernel(recursive_kernel(transitive_closure2, my_pred/2, [])).

%% Non-F# workload options: NO csr_path (most-commonly F#-WAM-
%% specific). Uses generic vocabulary the SPEC covers.
non_fsharp_options([
    cardinality(large),
    query_frequency(high),
    query_pattern(single_pair),
    b_eff(15.0),
    branching_d(4.5),
    contraction_r(0.04),
    heuristic_predicate_available(true),
    heuristic_predicate(my_heuristic/2)
]).

%% ========================================================================
%% Test declarations
%% ========================================================================

test(test_build_recurrence_term_with_mock_kernel).
test(test_build_workload_signals_no_fsharp_specific).
test(test_workload_contains_no_csr_path).
test(test_workload_contains_all_expected_signals).
test(test_helpers_dont_import_target_modules).
test(test_end_to_end_with_mock_caller_workload).

%% ========================================================================
%% Tests
%% ========================================================================

%% Helper builds a well-formed Recurrence from the mock detected
%% kernel without erroring. Doesn't assume F#-specific config.
test_build_recurrence_term_with_mock_kernel :-
    mock_detected_kernel(K),
    build_recurrence_term(K, [], Recurrence),
    Recurrence = recurrence(transitive_closure2, my_pred/2, Props),
    is_list(Props),
    member(value_domain(combinatorial), Props),
    member(monotone(true), Props).

%% Helper builds a well-formed Workload from the non-F# options
%% without erroring or implicitly adding F#-specific signals.
test_build_workload_signals_no_fsharp_specific :-
    non_fsharp_options(Options),
    build_workload_signals(Options, Workload),
    is_list(Workload),
    Workload \== [].

%% Critical: csr_path is most-commonly F#-WAM-specific (per SPEC's
%% signal table). The mock options don't include it; the workload
%% should not contain it either.
test_workload_contains_no_csr_path :-
    non_fsharp_options(Options),
    build_workload_signals(Options, Workload),
    \+ member(csr_path(_), Workload),
    \+ member(csr_available(_), Workload).  % csr_path implies csr_available

%% All expected vocabulary signals from non_fsharp_options should
%% appear in the workload.
test_workload_contains_all_expected_signals :-
    non_fsharp_options(Options),
    build_workload_signals(Options, Workload),
    member(cardinality(large), Workload),
    member(query_frequency(high), Workload),
    member(query_pattern(single_pair), Workload),
    member(b_eff(15.0), Workload),
    member(branching_d(4.5), Workload),
    member(contraction_r(0.04), Workload),
    member(heuristic_predicate_available(true), Workload),
    member(heuristic_predicate(my_heuristic/2), Workload).

%% Verify recurrence_inputs.pl doesn't import target-specific
%% modules. Reads the :- use_module/1 directives via the module's
%% predicate_property/2 mechanism.
test_helpers_dont_import_target_modules :-
    %% This test relies on the module being already loaded by the
    %% use_module directive at the top of this file.
    findall(M,
            ( predicate_property(recurrence_inputs:_, imported_from(M))
            ),
            Imports),
    sort(Imports, UniqueImports),
    forall(member(I, UniqueImports),
           ( atom(I),
             atom_string(I, IStr),
             \+ sub_string(IStr, _, _, _, "fsharp"),
             \+ sub_string(IStr, _, _, _, "haskell"),
             \+ sub_string(IStr, _, _, _, "c_target"),
             \+ sub_string(IStr, _, _, _, "csharp")
           )).

%% End-to-end: mock caller exercises the full pipeline (build
%% inputs → call selector → get strategy). Verifies the helpers
%% interoperate with the selector for a non-F# workload.
test_end_to_end_with_mock_caller_workload :-
    mock_detected_kernel(K),
    non_fsharp_options(Options),
    build_recurrence_term(K, [], Recurrence),
    build_workload_signals(Options, Workload),
    select_evaluation_strategy(Recurrence, Workload,
                               strategy_choice(Strategy, trace(_Steps))),
    %% Strategy should be SOME valid per_query or fixed_point variant.
    %% Don't assert a specific value — the assertion that the call
    %% succeeded without erroring is the point.
    Strategy = strategy(_Mode).
