:- encoding(utf8).
% Phase 1 tests for the WAM-lowered Haskell code-generation path.
%
% Phase 1 is plumbing only: the emit_mode/1 selector, the
% user:wam_haskell_emit_mode/1 dynamic fallback, the stub
% wam_haskell_lowerable/3, and the partition helper. The real lowering
% emitter lands in Phase 3+.
%
% These tests assert only that the plumbing resolves correctly and
% that every mode partitions predicates identically (all interpreted)
% because the stub fails for every predicate.
%
% Usage: swipl -g run_tests -t halt tests/test_wam_haskell_lowered_phase1.pl

:- use_module('../src/unifyweaver/targets/wam_haskell_target')  .

:- dynamic test_failed/0.
:- dynamic user:wam_haskell_emit_mode/1.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

%% ---------------------------------------------------------------------
%% Selector resolution
%% ---------------------------------------------------------------------

test_default_mode_is_interpreter :-
    Test = 'WAM-Haskell-Lowered Phase 1: default mode is interpreter',
    retractall(user:wam_haskell_emit_mode(_)),
    wam_haskell_resolve_emit_mode([], Mode),
    (   Mode == interpreter
    ->  pass(Test)
    ;   fail_test(Test, ['expected interpreter, got ', Mode])
    ).

test_option_overrides_default :-
    Test = 'WAM-Haskell-Lowered Phase 1: emit_mode option overrides default',
    retractall(user:wam_haskell_emit_mode(_)),
    wam_haskell_resolve_emit_mode([emit_mode(functions)], Mode),
    (   Mode == functions
    ->  pass(Test)
    ;   fail_test(Test, ['expected functions, got ', Mode])
    ).

test_user_fact_fallback :-
    Test = 'WAM-Haskell-Lowered Phase 1: user:wam_haskell_emit_mode fallback',
    retractall(user:wam_haskell_emit_mode(_)),
    assertz(user:wam_haskell_emit_mode(functions)),
    wam_haskell_resolve_emit_mode([], Mode),
    retractall(user:wam_haskell_emit_mode(_)),
    (   Mode == functions
    ->  pass(Test)
    ;   fail_test(Test, ['expected functions from user fact, got ', Mode])
    ).

test_option_beats_user_fact :-
    Test = 'WAM-Haskell-Lowered Phase 1: option beats user fact',
    retractall(user:wam_haskell_emit_mode(_)),
    assertz(user:wam_haskell_emit_mode(functions)),
    wam_haskell_resolve_emit_mode([emit_mode(interpreter)], Mode),
    retractall(user:wam_haskell_emit_mode(_)),
    (   Mode == interpreter
    ->  pass(Test)
    ;   fail_test(Test, ['expected interpreter from option, got ', Mode])
    ).

test_mixed_mode_accepted :-
    Test = 'WAM-Haskell-Lowered Phase 1: mixed(List) mode accepted',
    retractall(user:wam_haskell_emit_mode(_)),
    wam_haskell_resolve_emit_mode([emit_mode(mixed([cat_anc/4, pow_sum/4]))], Mode),
    (   Mode == mixed([cat_anc/4, pow_sum/4])
    ->  pass(Test)
    ;   fail_test(Test, ['expected mixed(...), got ', Mode])
    ).

test_unknown_mode_throws :-
    Test = 'WAM-Haskell-Lowered Phase 1: unknown mode throws domain_error',
    retractall(user:wam_haskell_emit_mode(_)),
    catch(
        wam_haskell_resolve_emit_mode([emit_mode(nonsense)], _),
        error(domain_error(wam_haskell_emit_mode, nonsense), _),
        ThrewCorrectly = true
    ),
    (   ThrewCorrectly == true
    ->  pass(Test)
    ;   fail_test(Test, 'unknown mode did not throw domain_error')
    ).

test_mixed_non_list_rejected :-
    Test = 'WAM-Haskell-Lowered Phase 1: mixed(non-list) rejected',
    retractall(user:wam_haskell_emit_mode(_)),
    catch(
        wam_haskell_resolve_emit_mode([emit_mode(mixed(not_a_list))], _),
        error(domain_error(wam_haskell_emit_mode, mixed(not_a_list)), _),
        ThrewCorrectly = true
    ),
    (   ThrewCorrectly == true
    ->  pass(Test)
    ;   fail_test(Test, 'mixed(non-list) did not throw')
    ).

%% ---------------------------------------------------------------------
%% Lowerability stub
%% ---------------------------------------------------------------------

test_lowerable_rejects_unsupported :-
    Test = 'WAM-Haskell-Lowered Phase 1: wam_haskell_lowerable/3 rejects unsupported instructions',
    % Phase 3 whitelist only covers get_constant+proceed. A WAM text
    % containing a Call instruction must fail lowerability.
    WamText = "foo/1:\n    call bar/0, 0\n    proceed\n",
    (   \+ wam_haskell_lowerable(foo/1, WamText, _Reason)
    ->  pass(Test)
    ;   fail_test(Test, 'unsupported Call instruction was accepted')
    ).

%% ---------------------------------------------------------------------
%% Partition
%% ---------------------------------------------------------------------

test_partition_interpreter_mode :-
    Test = 'WAM-Haskell-Lowered Phase 1: partition in interpreter mode is identity',
    Preds = [a/1, b/2, c/3],
    wam_haskell_partition_predicates(interpreter, Preds, Interp, Lower),
    (   Interp == Preds, Lower == []
    ->  pass(Test)
    ;   fail_test(Test, ['unexpected partition ', Interp, Lower])
    ).

% functions and mixed modes attempt lowering, which requires calling
% wam_target:compile_predicate_to_wam/3 on each listed predicate. Those
% calls exercise a wider stack than the pure selector tests above, so
% rather than fabricate a WAM body we ship a trivial predicate to the
% user module and partition against it.

:- dynamic user:phase1_probe/1.
user:phase1_probe(a).
user:phase1_probe(b).

test_partition_functions_mode :-
    Test = 'WAM-Haskell-Lowered Phase 1: partition in functions mode routes all to interpreter',
    wam_haskell_partition_predicates(functions, [user:phase1_probe/1], Interp, Lower),
    (   Interp == [user:phase1_probe/1], Lower == []
    ->  pass(Test)
    ;   fail_test(Test, ['unexpected partition ', Interp, Lower])
    ).

test_partition_mixed_mode_hot_and_cold :-
    Test = 'WAM-Haskell-Lowered Phase 1: partition in mixed mode — hot attempts lowering, cold straight to interpreter',
    % user:phase1_probe/1 is in HotPreds: attempts lowering (stub fails), goes to Interp.
    % user:phase1_cold/1 is NOT in HotPreds: goes straight to Interp without a WAM compile.
    % We synthesize a second indicator that does not have clauses, which is
    % fine because the cold branch never calls wam_target:compile_predicate_to_wam/3.
    wam_haskell_partition_predicates(
        mixed([phase1_probe/1]),
        [user:phase1_probe/1, user:phase1_cold/1],
        Interp,
        Lower),
    (   Interp == [user:phase1_probe/1, user:phase1_cold/1], Lower == []
    ->  pass(Test)
    ;   fail_test(Test, ['unexpected partition ', Interp, Lower])
    ).

%% ---------------------------------------------------------------------
%% Runner
%% ---------------------------------------------------------------------

run_tests :-
    retractall(test_failed),
    format('~n=== WAM-Haskell-Lowered Phase 1 tests ===~n', []),
    test_default_mode_is_interpreter,
    test_option_overrides_default,
    test_user_fact_fallback,
    test_option_beats_user_fact,
    test_mixed_mode_accepted,
    test_unknown_mode_throws,
    test_mixed_non_list_rejected,
    test_lowerable_rejects_unsupported,
    test_partition_interpreter_mode,
    test_partition_functions_mode,
    test_partition_mixed_mode_hot_and_cold,
    format('~n', []),
    (   test_failed
    ->  format('=== FAILED ===~n', []), halt(1)
    ;   format('=== All Phase 1 tests passed ===~n', [])
    ).
