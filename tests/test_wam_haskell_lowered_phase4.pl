:- encoding(utf8).
% Phase 4 smoke tests for the WAM-lowered Haskell path.
%
% Phase 4 adds F#-parity features to the lowered emitter:
%   - Inline optimizations (get_constant, get_value, put_structure, put_list,
%     set_variable, set_value, set_constant, deallocate, cut)
%   - Phase I specialized instructions (PutStructureDyn, Arg, NotMemberList,
%     NotMemberConstAtoms, BuildEmptySet, SetInsert, NotMemberSet)
%   - New lowerable instructions (call_foreign, retry_me_else, fail)
%   - LabelMap-aware ITE continuation tracking
%   - Fixed fresh_sv shadowing bug
%   - Quote handling in val_hs
%   - Double-quote escaping in escape_dq
%   - switch_on_constant_a2 prefix stripping
%   - Defensive error handling
%
% Usage: swipl -g run_tests -t halt tests/test_wam_haskell_lowered_phase4.pl

:- use_module('../src/unifyweaver/targets/wam_haskell_target').
:- use_module('../src/unifyweaver/targets/wam_haskell_lowered_emitter').
:- use_module('../src/unifyweaver/targets/wam_target').

:- dynamic test_failed/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

%% =====================================================================
%% Lowerability: new instructions accepted
%% =====================================================================

test_lowerable_accepts_set_variable :-
    Test = 'Phase 4: lowerable accepts set_variable',
    WamText = "foo/1:\n    put_structure bar/2, A1\n    set_variable X1\n    set_value A2\n    proceed\n",
    (   wam_haskell_lowerable(foo/1, WamText, _)
    ->  pass(Test)
    ;   fail_test(Test, 'set_variable was rejected')
    ).

test_lowerable_accepts_call_foreign :-
    Test = 'Phase 4: lowerable accepts call_foreign',
    WamText = "foo/1:\n    call_foreign bar, 1\n    proceed\n",
    (   wam_haskell_lowerable(foo/1, WamText, _)
    ->  pass(Test)
    ;   fail_test(Test, 'call_foreign was rejected')
    ).

test_lowerable_accepts_fail :-
    Test = 'Phase 4: lowerable accepts fail',
    WamText = "foo/0:\n    fail\n",
    (   wam_haskell_lowerable(foo/0, WamText, _)
    ->  pass(Test)
    ;   fail_test(Test, 'fail was rejected')
    ).

test_lowerable_accepts_retry_me_else :-
    Test = 'Phase 4: lowerable accepts retry_me_else',
    WamText = "foo/1:\n    retry_me_else L1\n    get_constant a, A1\n    proceed\n",
    (   wam_haskell_lowerable(foo/1, WamText, _)
    ->  pass(Test)
    ;   fail_test(Test, 'retry_me_else was rejected')
    ).

%% Phase I instructions
test_lowerable_accepts_put_structure_dyn :-
    Test = 'Phase 4: lowerable accepts put_structure_dyn',
    WamText = "foo/3:\n    put_structure_dyn A1, A2, A3\n    proceed\n",
    (   wam_haskell_lowerable(foo/3, WamText, _)
    ->  pass(Test)
    ;   fail_test(Test, 'put_structure_dyn was rejected')
    ).

test_lowerable_accepts_not_member_list :-
    Test = 'Phase 4: lowerable accepts not_member_list',
    WamText = "foo/2:\n    not_member_list A1, A2\n    proceed\n",
    (   wam_haskell_lowerable(foo/2, WamText, _)
    ->  pass(Test)
    ;   fail_test(Test, 'not_member_list was rejected')
    ).

test_lowerable_accepts_build_empty_set :-
    Test = 'Phase 4: lowerable accepts build_empty_set',
    WamText = "foo/1:\n    build_empty_set A1\n    proceed\n",
    (   wam_haskell_lowerable(foo/1, WamText, _)
    ->  pass(Test)
    ;   fail_test(Test, 'build_empty_set was rejected')
    ).

test_lowerable_accepts_set_insert :-
    Test = 'Phase 4: lowerable accepts set_insert',
    WamText = "foo/3:\n    set_insert A1, A2, A3\n    proceed\n",
    (   wam_haskell_lowerable(foo/3, WamText, _)
    ->  pass(Test)
    ;   fail_test(Test, 'set_insert was rejected')
    ).

test_lowerable_accepts_not_member_set :-
    Test = 'Phase 4: lowerable accepts not_member_set',
    WamText = "foo/2:\n    not_member_set A1, A2\n    proceed\n",
    (   wam_haskell_lowerable(foo/2, WamText, _)
    ->  pass(Test)
    ;   fail_test(Test, 'not_member_set was rejected')
    ).

test_lowerable_accepts_get_structure :-
    Test = 'Phase 4: lowerable now accepts get_structure (added to Instruction type)',
    WamText = "foo/1:\n    get_structure f/2, A1\n    proceed\n",
    (   wam_haskell_lowerable(foo/1, WamText, _)
    ->  pass(Test)
    ;   fail_test(Test, 'get_structure was rejected but should now be accepted')
    ).

%% =====================================================================
%% Emission: inline optimizations
%% =====================================================================

test_emit_get_constant_inline :-
    Test = 'Phase 4: get_constant emits inline case (not step delegation)',
    WamText = "foo/1:\n    get_constant 42, A1\n    proceed\n",
    lower_predicate_to_haskell(foo/1, WamText, [], lowered(_, _, Code)),
    (   sub_string(Code, _, _, _, "case val_"),
        sub_string(Code, _, _, _, "Just (Unbound vid)"),
        sub_string(Code, _, _, _, "TrailEntry vid"),
        \+ sub_string(Code, _, _, _, "step ctx")
    ->  pass(Test)
    ;   fail_test(Test, 'expected inline case, got step delegation')
    ).

test_emit_get_value_inline :-
    Test = 'Phase 4: get_value emits inline case (symmetric)',
    WamText = "foo/2:\n    get_value X1, A1\n    proceed\n",
    lower_predicate_to_haskell(foo/2, WamText, [], lowered(_, _, Code)),
    (   sub_string(Code, _, _, _, "(Just a, Just x) | a == x"),
        sub_string(Code, _, _, _, "(Just a, Just (Unbound vid))")
    ->  pass(Test)
    ;   fail_test(Test, 'expected symmetric inline get_value')
    ).

test_emit_put_structure_inline :-
    Test = 'Phase 4: put_structure emits inline let (not step)',
    WamText = "foo/1:\n    put_structure bar/2, A1\n    set_variable X1\n    set_value A2\n    proceed\n",
    lower_predicate_to_haskell(foo/1, WamText, [], lowered(_, _, Code)),
    (   sub_string(Code, _, _, _, "wsBuilder = BuildStruct"),
        sub_string(Code, _, _, _, "let ")
    ->  pass(Test)
    ;   fail_test(Test, 'expected inline put_structure')
    ).

test_emit_put_list_inline :-
    Test = 'Phase 4: put_list emits inline let (not step)',
    WamText = "foo/1:\n    put_list A1\n    set_value X1\n    set_value X2\n    proceed\n",
    lower_predicate_to_haskell(foo/1, WamText, [], lowered(_, _, Code)),
    (   sub_string(Code, _, _, _, "wsBuilder = BuildList")
    ->  pass(Test)
    ;   fail_test(Test, 'expected inline put_list')
    ).

test_emit_set_variable_inline :-
    Test = 'Phase 4: set_variable emits inline with addToBuilder',
    WamText = "foo/1:\n    put_structure bar/1, A1\n    set_variable X1\n    proceed\n",
    lower_predicate_to_haskell(foo/1, WamText, [], lowered(_, _, Code)),
    (   sub_string(Code, _, _, _, "addToBuilder var_")
    ->  pass(Test)
    ;   fail_test(Test, 'expected inline set_variable with addToBuilder')
    ).

test_emit_set_value_inline :-
    Test = 'Phase 4: set_value emits inline with addToBuilder',
    WamText = "foo/2:\n    put_structure bar/1, A1\n    set_value A2\n    proceed\n",
    lower_predicate_to_haskell(foo/2, WamText, [], lowered(_, _, Code)),
    (   sub_string(Code, _, _, _, "addToBuilder val")
    ->  pass(Test)
    ;   fail_test(Test, 'expected inline set_value with addToBuilder')
    ).

test_emit_set_constant_inline :-
    Test = 'Phase 4: set_constant emits inline addToBuilder',
    WamText = "foo/1:\n    put_structure bar/1, A1\n    set_constant hello\n    proceed\n",
    lower_predicate_to_haskell(foo/1, WamText, [], lowered(_, _, Code)),
    (   sub_string(Code, _, _, _, "addToBuilder (Atom")
    ->  pass(Test)
    ;   fail_test(Test, 'expected inline set_constant')
    ).

test_emit_deallocate_inline :-
    Test = 'Phase 4: deallocate emits inline let (not step)',
    WamText = "foo/1:\n    allocate\n    get_constant 1, A1\n    deallocate\n    proceed\n",
    lower_predicate_to_haskell(foo/1, WamText, [], lowered(_, _, Code)),
    (   sub_string(Code, _, _, _, "EnvFrame oldCP"),
        \+ sub_string(Code, _, _, _, "step ctx")
    ->  pass(Test)
    ;   fail_test(Test, 'expected inline deallocate')
    ).

test_emit_cut_inline :-
    Test = 'Phase 4: builtin_call !/0 emits inline cut (not step)',
    WamText = "foo/1:\n    allocate\n    builtin_call !/0, 0\n    deallocate\n    proceed\n",
    lower_predicate_to_haskell(foo/1, WamText, [], lowered(_, _, Code)),
    (   sub_string(Code, _, _, _, "wsCPsLen"),
        sub_string(Code, _, _, _, "wsCutBar"),
        sub_string(Code, _, _, _, "drop drop_")
    ->  pass(Test)
    ;   fail_test(Test, 'expected inline cut')
    ).

%% =====================================================================
%% Emission: new instructions
%% =====================================================================

test_emit_fail_terminal :-
    Test = 'Phase 4: fail emits Nothing',
    WamText = "foo/0:\n    fail\n",
    lower_predicate_to_haskell(foo/0, WamText, [], lowered(_, _, Code)),
    (   sub_string(Code, _, _, _, "Nothing")
    ->  pass(Test)
    ;   fail_test(Test, 'expected Nothing for fail')
    ).

test_emit_call_foreign :-
    Test = 'Phase 4: call_foreign emits step delegation',
    WamText = "foo/1:\n    call_foreign bar, 1\n    proceed\n",
    lower_predicate_to_haskell(foo/1, WamText, [], lowered(_, _, Code)),
    (   sub_string(Code, _, _, _, "CallForeign")
    ->  pass(Test)
    ;   fail_test(Test, 'expected CallForeign in output')
    ).

test_emit_retry_me_else :-
    Test = 'Phase 4: retry_me_else emits step delegation',
    WamText = "foo/1:\n    retry_me_else L1\n    get_constant a, A1\n    proceed\n",
    lower_predicate_to_haskell(foo/1, WamText, [], lowered(_, _, Code)),
    (   sub_string(Code, _, _, _, "RetryMeElse")
    ->  pass(Test)
    ;   fail_test(Test, 'expected RetryMeElse in output')
    ).

%% Phase I instruction emission
test_emit_put_structure_dyn :-
    Test = 'Phase 4: put_structure_dyn emits PutStructureDyn',
    WamText = "foo/3:\n    put_structure_dyn A1, A2, A3\n    proceed\n",
    lower_predicate_to_haskell(foo/3, WamText, [], lowered(_, _, Code)),
    (   sub_string(Code, _, _, _, "PutStructureDyn")
    ->  pass(Test)
    ;   fail_test(Test, 'expected PutStructureDyn in output')
    ).

test_emit_not_member_list :-
    Test = 'Phase 4: not_member_list emits NotMemberList',
    WamText = "foo/2:\n    not_member_list A1, A2\n    proceed\n",
    lower_predicate_to_haskell(foo/2, WamText, [], lowered(_, _, Code)),
    (   sub_string(Code, _, _, _, "NotMemberList")
    ->  pass(Test)
    ;   fail_test(Test, 'expected NotMemberList in output')
    ).

test_emit_build_empty_set :-
    Test = 'Phase 4: build_empty_set emits BuildEmptySet',
    WamText = "foo/1:\n    build_empty_set A1\n    proceed\n",
    lower_predicate_to_haskell(foo/1, WamText, [], lowered(_, _, Code)),
    (   sub_string(Code, _, _, _, "BuildEmptySet")
    ->  pass(Test)
    ;   fail_test(Test, 'expected BuildEmptySet in output')
    ).

test_emit_set_insert :-
    Test = 'Phase 4: set_insert emits SetInsert',
    WamText = "foo/3:\n    set_insert A1, A2, A3\n    proceed\n",
    lower_predicate_to_haskell(foo/3, WamText, [], lowered(_, _, Code)),
    (   sub_string(Code, _, _, _, "SetInsert")
    ->  pass(Test)
    ;   fail_test(Test, 'expected SetInsert in output')
    ).

test_emit_not_member_set :-
    Test = 'Phase 4: not_member_set emits NotMemberSet',
    WamText = "foo/2:\n    not_member_set A1, A2\n    proceed\n",
    lower_predicate_to_haskell(foo/2, WamText, [], lowered(_, _, Code)),
    (   sub_string(Code, _, _, _, "NotMemberSet")
    ->  pass(Test)
    ;   fail_test(Test, 'expected NotMemberSet in output')
    ).

%% =====================================================================
%% Helpers: bug fixes
%% =====================================================================

test_fresh_sv_no_shadowing :-
    Test = 'Phase 4: fresh_sv does not shadow s_0 (bug fix)',
    wam_haskell_lowered_emitter:fresh_sv(s_init, S0),
    wam_haskell_lowered_emitter:fresh_sv(S0, S1),
    wam_haskell_lowered_emitter:fresh_sv(S1, S2),
    (   S0 == s_0, S1 == s_1, S2 == s_2
    ->  pass(Test)
    ;   fail_test(Test, ['expected s_0,s_1,s_2 got ', S0, S1, S2])
    ).

test_escape_dq_handles_both :-
    Test = 'Phase 4: escape_dq escapes both backslash and double-quote',
    wam_haskell_lowered_emitter:escape_dq("foo\\bar\"baz", Esc),
    (   Esc == "foo\\\\bar\\\"baz"
    ->  pass(Test)
    ;   fail_test(Test, ['expected foo\\\\bar\\\"baz got ', Esc])
    ).

%% =====================================================================
%% Defensive error handling
%% =====================================================================

test_get_variable_defensive :-
    Test = 'Phase 4: get_variable uses error instead of silent Atom atomEmpty',
    WamText = "foo/2:\n    get_variable X1, A1\n    proceed\n",
    lower_predicate_to_haskell(foo/2, WamText, [], lowered(_, _, Code)),
    (   sub_string(Code, _, _, _, "error \"GetVariable: source register not bound\"")
    ->  pass(Test)
    ;   fail_test(Test, 'expected defensive error in GetVariable')
    ).

test_put_value_defensive :-
    Test = 'Phase 4: put_value uses error instead of silent Atom atomEmpty',
    WamText = "foo/2:\n    put_value X1, A1\n    proceed\n",
    lower_predicate_to_haskell(foo/2, WamText, [], lowered(_, _, Code)),
    (   sub_string(Code, _, _, _, "error \"PutValue: source register not bound\"")
    ->  pass(Test)
    ;   fail_test(Test, 'expected defensive error in PutValue')
    ).

test_deallocate_defensive :-
    Test = 'Phase 4: deallocate uses error for empty stack',
    WamText = "foo/1:\n    allocate\n    deallocate\n    proceed\n",
    lower_predicate_to_haskell(foo/1, WamText, [], lowered(_, _, Code)),
    (   sub_string(Code, _, _, _, "error \"Deallocate: empty WsStack\"")
    ->  pass(Test)
    ;   fail_test(Test, 'expected defensive error in Deallocate')
    ).

%% =====================================================================
%% Runner
%% =====================================================================

run_tests :-
    retractall(test_failed),
    format('~n=== WAM-Haskell-Lowered Phase 4 tests ===~n', []),
    % Lowerability
    test_lowerable_accepts_set_variable,
    test_lowerable_accepts_call_foreign,
    test_lowerable_accepts_fail,
    test_lowerable_accepts_retry_me_else,
    test_lowerable_accepts_put_structure_dyn,
    test_lowerable_accepts_not_member_list,
    test_lowerable_accepts_build_empty_set,
    test_lowerable_accepts_set_insert,
    test_lowerable_accepts_not_member_set,
    test_lowerable_accepts_get_structure,
    % Inline optimizations
    test_emit_get_constant_inline,
    test_emit_get_value_inline,
    test_emit_put_structure_inline,
    test_emit_put_list_inline,
    test_emit_set_variable_inline,
    test_emit_set_value_inline,
    test_emit_set_constant_inline,
    test_emit_deallocate_inline,
    test_emit_cut_inline,
    % New instructions
    test_emit_fail_terminal,
    test_emit_call_foreign,
    test_emit_retry_me_else,
    test_emit_put_structure_dyn,
    test_emit_not_member_list,
    test_emit_build_empty_set,
    test_emit_set_insert,
    test_emit_not_member_set,
    % Helpers
    test_fresh_sv_no_shadowing,
    test_escape_dq_handles_both,
    % Defensive errors
    test_get_variable_defensive,
    test_put_value_defensive,
    test_deallocate_defensive,
    format('~n', []),
    (   test_failed
    ->  format('=== FAILED ===~n', []), halt(1)
    ;   format('=== All Phase 4 tests passed ===~n', [])
    ).
