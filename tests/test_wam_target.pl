:- encoding(utf8).
% Test suite for WAM target
% Usage: swipl -g run_tests -t halt tests/test_wam_target.pl

:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_text_parser', [wam_text_to_items/2]).

%% Test data (facts) - MUST BE DYNAMIC for clause/2 to work across modules
:- dynamic test_parent/2, test_grandparent/2, test_ancestor/2, test_wrap/2,
           test_alias/2, test_len/2, test_list_check/1, test_list_wrap/1,
           test_linear_fact/1, test_linear_three/1, test_linear_a/1,
           test_linear_b/1, test_linear_c/1.

test_parent(alice, bob).
test_parent(bob, charlie).

%% Grandparent rule
test_grandparent(X, Z) :- test_parent(X, Y), test_parent(Y, Z).

%% Simple forwarding rule — exercises native items for a one-goal tail call.
test_alias(X, Y) :- test_parent(X, Y).

%% Simple builtin rule — exercises native items for generic builtin_call.
test_len(L, N) :- length(L, N).

%% Non-indexed multi-clause shapes — exercises native items for linear
%  try/retry/trust chains without going through the WAM text parser.
test_linear_fact(_).
test_linear_fact(_).

test_linear_a(_).
test_linear_b(_).
test_linear_c(_).
test_linear_three(X) :- test_linear_a(X).
test_linear_three(X) :- test_linear_b(X).
test_linear_three(X) :- test_linear_c(X).

%% Recursive ancestor rule
test_ancestor(X, Y) :- test_parent(X, Y).
test_ancestor(X, Y) :- test_parent(X, Z), test_ancestor(Z, Y).

%% Rule with compound body argument — exercises put_structure
%  The body goal `test_check(pair(X, done))` has a compound argument.
:- dynamic test_check/1.
test_check(pair(_, _)).
test_wrap(X) :- test_check(pair(X, done)).

%% Rule with list body argument — exercises put_list + nested [|]/2 emit.
test_list_check([_, _]).
test_list_wrap(X) :- test_list_check([X, done]).

%% Compound head — exercises get_structure + unify_constant
:- dynamic test_color/2.
test_color(rgb(255, 0, 0), red).

%% Multi-clause body containing a findall — regression for the bug
%  where compile_clauses_fragments only emitted allocate/deallocate
%  when goal-count > 1, missing the case where a single surface goal
%  is itself a findall/aggregate that internally produces a Call.
%  Without an env frame, the post-end_aggregate continuation has no
%  saved caller cp to return to; finalise_aggregate's cp chain loops
%  back into k2 forever. See compile_single_clause_wam/3 for the
%  same logic that was previously only applied to single-clause.
:- dynamic test_multi_findall/1, test_multi_inner/1.
test_multi_inner(1).
test_multi_inner(2).
test_multi_findall('a') :- findall(_, test_multi_inner(_), _).
test_multi_findall('b') :- findall(_, test_multi_inner(_), _).
test_multi_findall('c') :- findall(_, test_multi_inner(_), _).

%% Nested compound body arg — exercises recursive put_structure
:- dynamic test_nested_check/1.
test_nested_check(box(inner(_, _))).
:- dynamic test_nested_wrap/1.
test_nested_wrap(X) :- test_nested_check(box(inner(X, done))).

:- dynamic test_failed/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

%% Tests
test_wam_facts :-
    Test = 'WAM: compile_facts',
    (   wam_target:compile_facts_to_wam(user:test_parent, 2, Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'test_parent/2:'),
        sub_string(S, _, _, _, 'get_constant alice, A1'),
        sub_string(S, _, _, _, 'try_me_else'),
        sub_string(S, _, _, _, 'trust_me')
    ->  pass(Test)
    ;   fail_test(Test, 'Incorrect WAM output for facts')
    ).

test_wam_single_clause :-
    Test = 'WAM: single clause rule',
    (   wam_target:compile_predicate_to_wam(user:test_grandparent/2, [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'test_grandparent/2:'),
        sub_string(S, _, _, _, 'allocate'),
        sub_string(S, _, _, _, 'get_variable'),
        sub_string(S, _, _, _, 'put_value'),
        sub_string(S, _, _, _, 'call test_parent/2'),
        sub_string(S, _, _, _, 'deallocate'),
        sub_string(S, _, _, _, 'execute test_parent/2')
    ->  pass(Test)
    ;   wam_target:compile_predicate_to_wam(user:test_grandparent/2, [], Code2),
        format(user_error, 'DEBUG: single clause output:~n~w~n', [Code2]),
        fail_test(Test, 'Incorrect WAM output for single clause')
    ).

test_wam_recursion :-
    Test = 'WAM: recursive rule (ancestor)',
    (   wam_target:compile_predicate_to_wam(user:test_ancestor/2, [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'test_ancestor/2:'),
        sub_string(S, _, _, _, 'try_me_else'),
        sub_string(S, _, _, _, 'trust_me'),
        sub_string(S, _, _, _, 'execute test_parent/2'),
        sub_string(S, _, _, _, 'allocate'),
        sub_string(S, _, _, _, 'put_value'),
        sub_string(S, _, _, _, 'call test_parent/2'),
        sub_string(S, _, _, _, 'deallocate'),
        sub_string(S, _, _, _, 'execute test_ancestor/2')
    ->  pass(Test)
    ;   wam_target:compile_predicate_to_wam(user:test_ancestor/2, [], Code2),
        format(user_error, 'DEBUG: recursion output:~n~w~n', [Code2]),
        fail_test(Test, 'Incorrect WAM output for recursion')
    ).

test_wam_module :-
    Test = 'WAM: compile_wam_module (template)',
    (   wam_target:compile_wam_module([user:test_parent/2, user:test_grandparent/2], [module_name('FamilyModule')], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'WAM Module: FamilyModule'),
        sub_string(S, _, _, _, 'test_parent/2:'),
        sub_string(S, _, _, _, 'test_grandparent/2:')
    ->  pass(Test)
    ;   wam_target:compile_wam_module([user:test_parent/2, user:test_grandparent/2], [module_name('FamilyModule')], Code2),
        format(user_error, 'DEBUG: Failed Code:~n~w~n', [Code2]),
        fail_test(Test, 'Incorrect WAM module output from template')
    ).

test_wam_put_structure :-
    Test = 'WAM: put_structure for compound body args',
    (   wam_target:compile_predicate_to_wam(user:test_wrap/1, [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'put_structure pair/2')
    ->  pass(Test)
    ;   (   wam_target:compile_predicate_to_wam(user:test_wrap/1, [], Code2)
        ->  format(user_error, 'DEBUG: put_structure output:~n~w~n', [Code2])
        ;   format(user_error, 'DEBUG: compile_predicate_to_wam failed~n', [])
        ),
        fail_test(Test, 'Missing put_structure in compound body arg output')
    ).

test_wam_compound_head :-
    Test = 'WAM: compound head (get_structure)',
    (   wam_target:compile_predicate_to_wam(user:test_color/2, [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'get_structure rgb/3'),
        sub_string(S, _, _, _, 'unify_constant 255'),
        sub_string(S, _, _, _, 'get_constant red')
    ->  pass(Test)
    ;   (   wam_target:compile_predicate_to_wam(user:test_color/2, [], Code2)
        ->  format(user_error, 'DEBUG: compound head output:~n~w~n', [Code2])
        ;   format(user_error, 'DEBUG: compile failed~n', [])
        ),
        fail_test(Test, 'Missing get_structure in compound head output')
    ).

test_wam_nested_put_structure :-
    Test = 'WAM: nested put_structure for compound body args',
    (   wam_target:compile_predicate_to_wam(user:test_nested_wrap/1, [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'put_structure box/1'),
        sub_string(S, _, _, _, 'put_structure inner/2')
    ->  pass(Test)
    ;   (   wam_target:compile_predicate_to_wam(user:test_nested_wrap/1, [], Code2)
        ->  format(user_error, 'DEBUG: nested put_structure output:~n~w~n', [Code2])
        ;   format(user_error, 'DEBUG: compile failed~n', [])
        ),
        fail_test(Test, 'Missing nested put_structure in output')
    ).

%% Multi-clause findall regression: every clause must have an
%  allocate/deallocate pair so the inner findall's post-end_aggregate
%  continuation can retrieve the caller cp from the env frame.
test_wam_multi_clause_findall_emits_allocate :-
    Test = 'WAM: multi-clause body with findall emits allocate/deallocate per clause',
    (   wam_target:compile_predicate_to_wam(user:test_multi_findall/1, [], Code),
        atom_string(Code, S),
        % Each of the three clause bodies should contain begin_aggregate
        % surrounded by allocate ... deallocate. Three pairs total.
        aggsubs_count(S, 'allocate', AllocCount),
        aggsubs_count(S, 'deallocate', DeallocCount),
        aggsubs_count(S, 'begin_aggregate', AggCount),
        AllocCount >= 3,
        DeallocCount >= 3,
        AggCount =:= 3
    ->  pass(Test)
    ;   wam_target:compile_predicate_to_wam(user:test_multi_findall/1, [], Code2),
        format(user_error, 'DEBUG: multi-clause findall output:~n~w~n', [Code2]),
        fail_test(Test, 'Multi-clause findall body missing allocate/deallocate per clause')
    ).

%% Count non-overlapping occurrences of Sub in S.
aggsubs_count(S, Sub, N) :-
    findall(_, sub_string(S, _, _, _, Sub), Occurrences),
    length(Occurrences, N).

%% A2 indexing: when A1 is variable in every clause, the compiler
%% falls back to A2-based indexing. Three flavours, mirroring A1:
%%   all constants  → switch_on_constant_a2
%%   all compounds  → switch_on_structure_a2 (no lists)
%%   mixed types    → switch_on_term_a2
:- dynamic test_a2_const/2, test_a2_struct/2, test_a2_term/2.
test_a2_const(_, alpha).
test_a2_const(_, beta).
test_a2_const(_, gamma).

test_a2_struct(_, foo(_)).
test_a2_struct(_, bar(_, _)).
test_a2_struct(_, baz(_, _, _)).

test_a2_term(_, t).
test_a2_term(_, t(_, _)).
test_a2_term(_, []).
test_a2_term(_, [_|_]).

test_wam_a2_indexing :-
    Test = 'WAM: A2-arg indexing emits constant/structure/term variants',
    (   wam_target:compile_predicate_to_wam(user:test_a2_const/2, [], C1),
        wam_target:compile_predicate_to_wam(user:test_a2_struct/2, [], C2),
        wam_target:compile_predicate_to_wam(user:test_a2_term/2, [], C3),
        atom_string(C1, S1), atom_string(C2, S2), atom_string(C3, S3),
        sub_string(S1, _, _, _, 'switch_on_constant_a2'),
        sub_string(S2, _, _, _, 'switch_on_structure_a2'),
        sub_string(S3, _, _, _, 'switch_on_term_a2'),
        % Term-form should not be a degenerate structure/constant emit.
        \+ sub_string(S3, _, _, _, 'switch_on_structure_a2'),
        \+ sub_string(S3, _, _, _, 'switch_on_constant_a2 ')
    ->  pass(Test)
    ;   fail_test(Test, 'A2 indexing did not emit the expected pseudo-instruction')
    ).

%% Mixed-mode A1 indexing: predicates with a variable A1 clause
%% should now get a switch_on_constant_fallthrough for their indexed
%% prefix, instead of no A1 indexing.
:- dynamic test_mma_trailing/2, test_mma_middle/2, test_mma_first/2,
           test_mma_none/2.
test_mma_trailing(a, 1).
test_mma_trailing(b, 2).
test_mma_trailing(c, 3).
test_mma_trailing(_, 0).

test_mma_middle(a, 1).
test_mma_middle(_, 99).
test_mma_middle(b, 2).

test_mma_first(_, 0).
test_mma_first(a, 1).
test_mma_first(b, 2).

test_mma_none(a, 1).
test_mma_none(b, 2).
test_mma_none(c, 3).

%% Mixed-mode A2 indexing — mirror of the A1 case. Predicates whose
%% A1 is variable in every clause should now get
%% switch_on_constant_a2_fallthrough for their indexed A2 prefix
%% instead of dropping A2 indexing entirely.
:- dynamic test_mma2_trailing/3, test_mma2_middle/3, test_mma2_first/3,
           test_mma2_none/3.
test_mma2_trailing(_, error, red).
test_mma2_trailing(_, warn,  yellow).
test_mma2_trailing(_, ok,    green).
test_mma2_trailing(_, _,     gray).

test_mma2_middle(_, a, 1).
test_mma2_middle(_, _, 99).
test_mma2_middle(_, b, 2).

test_mma2_first(_, _, 0).
test_mma2_first(_, a, 1).
test_mma2_first(_, b, 2).

test_mma2_none(_, a, 1).
test_mma2_none(_, b, 2).
test_mma2_none(_, c, 3).

test_wam_mixed_mode_a2_indexing :-
    Test = 'WAM: mixed-mode A2 indexing emits switch_on_constant_a2_fallthrough',
    (   wam_target:compile_predicate_to_wam(user:test_mma2_trailing/3, [], C1),
        wam_target:compile_predicate_to_wam(user:test_mma2_middle/3, [], C2),
        wam_target:compile_predicate_to_wam(user:test_mma2_first/3, [], C3),
        wam_target:compile_predicate_to_wam(user:test_mma2_none/3, [], C4),
        atom_string(C1, S1), atom_string(C2, S2),
        atom_string(C3, S3), atom_string(C4, S4),
        % Trailing var A2: indexed prefix = error,warn,ok.
        sub_string(S1, _, _, _, 'switch_on_constant_a2_fallthrough'),
        % Middle var A2: indexed prefix = just `a`.
        sub_string(S2, _, _, _, 'switch_on_constant_a2_fallthrough'),
        % Var A2 first: no A2 indexing.
        \+ sub_string(S3, _, _, _, 'switch_on_constant_a2_fallthrough'),
        % No var A2: plain switch_on_constant_a2 (NOT fallthrough).
        sub_string(S4, _, _, _, 'switch_on_constant_a2 '),
        \+ sub_string(S4, _, _, _, 'switch_on_constant_a2_fallthrough')
    ->  pass(Test)
    ;   fail_test(Test,
                  'Mixed-mode A2 indexing did not emit the expected pseudo-instruction')
    ).

test_wam_mixed_mode_a1_indexing :-
    Test = 'WAM: mixed-mode A1 indexing emits switch_on_constant_fallthrough',
    (   wam_target:compile_predicate_to_wam(user:test_mma_trailing/2, [], C1),
        wam_target:compile_predicate_to_wam(user:test_mma_middle/2, [], C2),
        wam_target:compile_predicate_to_wam(user:test_mma_first/2, [], C3),
        wam_target:compile_predicate_to_wam(user:test_mma_none/2, [], C4),
        atom_string(C1, S1), atom_string(C2, S2),
        atom_string(C3, S3), atom_string(C4, S4),
        % Trailing var: indexed prefix is a,b,c → fallthrough form.
        sub_string(S1, _, _, _, 'switch_on_constant_fallthrough'),
        % Middle var: indexed prefix is just `a` → fallthrough form.
        sub_string(S2, _, _, _, 'switch_on_constant_fallthrough'),
        % Var first: no A1 indexing (falls back to A2).
        \+ sub_string(S3, _, _, _, 'switch_on_constant_fallthrough'),
        % No var clause: plain switch_on_constant (NOT the fallthrough form).
        sub_string(S4, _, _, _, 'switch_on_constant '),
        \+ sub_string(S4, _, _, _, 'switch_on_constant_fallthrough')
    ->  pass(Test)
    ;   fail_test(Test,
                  'Mixed-mode A1 indexing did not emit the expected pseudo-instruction')
    ).

test_wam_items_native_single_fact :-
    Test = 'WAM: native items API for single fact',
    ExpectedItems = [
        label("test_color/2"),
        get_structure("rgb/3", "A1"),
        unify_constant("255"),
        unify_constant("0"),
        unify_constant("0"),
        get_constant("red", "A2"),
        proceed
    ],
    (   wam_target:compile_predicate_to_wam_text(user:test_color/2, [], TextCode),
        wam_text_to_items(TextCode, BridgeItems),
        wam_target:compile_predicate_to_wam_items(user:test_color/2, [], Items),
        Items == ExpectedItems,
        Items == BridgeItems
    ->  pass(Test)
    ;   fail_test(Test, 'Native fact items do not match canonical WAM text shape')
    ).

test_wam_items_native_simple_tail_call :-
    Test = 'WAM: native items API for simple tail call',
    ExpectedItems = [
        label("test_alias/2"),
        allocate,
        get_variable("X1", "A1"),
        get_variable("X2", "A2"),
        put_value("X1", "A1"),
        put_value("X2", "A2"),
        deallocate,
        execute("test_parent/2")
    ],
    (   wam_target:compile_predicate_to_wam_text(user:test_alias/2, [], TextCode),
        wam_text_to_items(TextCode, BridgeItems),
        wam_target:compile_predicate_to_wam_items(user:test_alias/2, [], Items),
        Items == ExpectedItems,
        Items == BridgeItems
    ->  pass(Test)
    ;   fail_test(Test, 'Native simple-tail-call items do not match canonical WAM text shape')
    ).

test_wam_items_native_multi_goal_rule :-
    Test = 'WAM: native items API for simple multi-goal rule',
    ExpectedItems = [
        label("test_grandparent/2"),
        allocate,
        get_variable("X3", "A1"),
        get_variable("Y2", "A2"),
        put_value("X3", "A1"),
        put_variable("Y1", "A2"),
        call("test_parent/2", "2"),
        put_value("Y1", "A1"),
        put_value("Y2", "A2"),
        deallocate,
        execute("test_parent/2")
    ],
    (   wam_target:compile_predicate_to_wam(user:test_grandparent/2, [], LegacyCode),
        wam_target:compile_predicate_to_wam_text(user:test_grandparent/2, [], TextCode),
        wam_target:compile_predicate_to_wam_items(user:test_grandparent/2, [], Items),
        LegacyCode == TextCode,
        Items == ExpectedItems,
        wam_text_to_items(TextCode, Items),
        member(label("test_grandparent/2"), Items),
        member(allocate, Items),
        member(call("test_parent/2", "2"), Items),
        member(execute("test_parent/2"), Items)
    ->  pass(Test)
    ;   fail_test(Test, 'Explicit text/items APIs do not match legacy WAM output')
    ).

test_wam_items_native_simple_builtin :-
    Test = 'WAM: native items API for simple builtin call',
    ExpectedItems = [
        label("test_len/2"),
        get_variable("X1", "A1"),
        get_variable("X2", "A2"),
        put_value("X1", "A1"),
        put_value("X2", "A2"),
        builtin_call("length/2", "2"),
        proceed
    ],
    (   wam_target:compile_predicate_to_wam_text(user:test_len/2, [], TextCode),
        wam_text_to_items(TextCode, BridgeItems),
        wam_target:compile_predicate_to_wam_items(user:test_len/2, [], Items),
        Items == ExpectedItems,
        Items == BridgeItems
    ->  pass(Test)
    ;   fail_test(Test, 'Native simple-builtin items do not match canonical WAM text shape')
    ).

test_wam_items_native_compound_body_arg :-
    Test = 'WAM: native items API for compound body arg',
    ExpectedItems = [
        label("test_wrap/1"),
        allocate,
        get_variable("X1", "A1"),
        put_structure("pair/2", "A1"),
        set_value("X1"),
        set_constant("done"),
        deallocate,
        execute("test_check/1")
    ],
    (   wam_target:compile_predicate_to_wam_text(user:test_wrap/1, [], TextCode),
        wam_text_to_items(TextCode, BridgeItems),
        wam_target:compile_predicate_to_wam_items(user:test_wrap/1, [], Items),
        Items == ExpectedItems,
        Items == BridgeItems,
        member(put_structure("pair/2", "A1"), Items)
    ->  pass(Test)
    ;   fail_test(Test, 'Native compound-body-arg items do not match canonical WAM text shape')
    ).

test_wam_items_native_nested_compound_body_arg :-
    Test = 'WAM: native items API for nested compound body arg',
    ExpectedItems = [
        label("test_nested_wrap/1"),
        allocate,
        get_variable("X1", "A1"),
        put_structure("box/1", "A1"),
        set_variable("X3"),
        put_structure("inner/2", "X3"),
        set_value("X1"),
        set_constant("done"),
        deallocate,
        execute("test_nested_check/1")
    ],
    (   wam_target:compile_predicate_to_wam_text(user:test_nested_wrap/1, [], TextCode),
        wam_text_to_items(TextCode, BridgeItems),
        wam_target:compile_predicate_to_wam_items(user:test_nested_wrap/1, [], Items),
        Items == ExpectedItems,
        Items == BridgeItems,
        member(put_structure("box/1", "A1"), Items),
        member(put_structure("inner/2", "X3"), Items)
    ->  pass(Test)
    ;   fail_test(Test, 'Native nested-compound-body-arg items do not match canonical WAM text shape')
    ).

test_wam_items_native_list_body_arg :-
    Test = 'WAM: native items API for list body arg',
    ExpectedItems = [
        label("test_list_wrap/1"),
        allocate,
        get_variable("X1", "A1"),
        put_list("A1"),
        set_value("X1"),
        set_variable("X3"),
        put_structure("[|]/2", "X3"),
        set_constant("done"),
        set_constant("[]"),
        deallocate,
        execute("test_list_check/1")
    ],
    (   wam_target:compile_predicate_to_wam_text(user:test_list_wrap/1, [], TextCode),
        wam_text_to_items(TextCode, BridgeItems),
        wam_target:compile_predicate_to_wam_items(user:test_list_wrap/1, [], Items),
        Items == ExpectedItems,
        Items == BridgeItems,
        member(put_list("A1"), Items),
        member(put_structure("[|]/2", "X3"), Items)
    ->  pass(Test)
    ;   fail_test(Test, 'Native list-body-arg items do not match canonical WAM text shape')
    ).

test_wam_items_compound_arg_legacy_order_fallback :-
    Test = 'WAM: items API compound arg legacy-order fallback',
    Options = [args_first_emission(false)],
    (   wam_target:compile_predicate_to_wam_text(user:test_nested_wrap/1, Options, TextCode),
        wam_text_to_items(TextCode, BridgeItems),
        wam_target:compile_predicate_to_wam_items(user:test_nested_wrap/1, Options, Items),
        Items == BridgeItems,
        member(put_structure("box/1", "A1"), Items),
        member(put_structure("inner/2", "X3"), Items)
    ->  pass(Test)
    ;   fail_test(Test, 'Compound body arg legacy-order fallback does not match canonical WAM text shape')
    ).

test_wam_items_native_linear_fact_clauses :-
    Test = 'WAM: native items API for linear multi-clause facts',
    ExpectedItems = [
        label("test_linear_fact/1"),
        try_me_else("L_test_linear_fact_1_2"),
        get_variable("X1", "A1"),
        proceed,
        label("L_test_linear_fact_1_2"),
        trust_me,
        label("L_test_linear_fact_1_2_body"),
        get_variable("X1", "A1"),
        proceed
    ],
    (   wam_target:compile_predicate_to_wam_text(user:test_linear_fact/1, [], TextCode),
        wam_text_to_items(TextCode, BridgeItems),
        wam_target:compile_predicate_to_wam_items(user:test_linear_fact/1, [], Items),
        Items == ExpectedItems,
        Items == BridgeItems,
        \+ member(switch_on_constant(_), Items),
        member(try_me_else("L_test_linear_fact_1_2"), Items),
        member(trust_me, Items)
    ->  pass(Test)
    ;   fail_test(Test, 'Native linear fact items do not match canonical WAM text shape')
    ).

test_wam_items_native_linear_rule_clauses :-
    Test = 'WAM: native items API for linear multi-clause rules',
    ExpectedItems = [
        label("test_linear_three/1"),
        try_me_else("L_test_linear_three_1_2"),
        allocate,
        get_variable("X1", "A1"),
        put_value("X1", "A1"),
        deallocate,
        execute("test_linear_a/1"),
        label("L_test_linear_three_1_2"),
        retry_me_else("L_test_linear_three_1_3"),
        label("L_test_linear_three_1_2_body"),
        allocate,
        get_variable("X1", "A1"),
        put_value("X1", "A1"),
        deallocate,
        execute("test_linear_b/1"),
        label("L_test_linear_three_1_3"),
        trust_me,
        label("L_test_linear_three_1_3_body"),
        allocate,
        deallocate,
        execute("test_linear_c/1")
    ],
    (   wam_target:compile_predicate_to_wam_text(user:test_linear_three/1, [], TextCode),
        wam_text_to_items(TextCode, BridgeItems),
        wam_target:compile_predicate_to_wam_items(user:test_linear_three/1, [], Items),
        Items == ExpectedItems,
        Items == BridgeItems,
        \+ member(switch_on_constant(_), Items),
        member(retry_me_else("L_test_linear_three_1_3"), Items),
        member(execute("test_linear_c/1"), Items)
    ->  pass(Test)
    ;   fail_test(Test, 'Native linear rule items do not match canonical WAM text shape')
    ).

test_wam_items_native_switch_on_constant_index :-
    Test = 'WAM: native items API for switch_on_constant indexed clauses',
    ExpectedItems = [
        label("test_parent/2"),
        switch_on_constant(["alice:default", "bob:L_test_parent_2_2_body"]),
        try_me_else("L_test_parent_2_2"),
        get_constant("alice", "A1"),
        get_constant("bob", "A2"),
        proceed,
        label("L_test_parent_2_2"),
        trust_me,
        label("L_test_parent_2_2_body"),
        get_constant("bob", "A1"),
        get_constant("charlie", "A2"),
        proceed
    ],
    (   wam_target:compile_predicate_to_wam_text(user:test_parent/2, [], TextCode),
        wam_text_to_items(TextCode, BridgeItems),
        wam_target:compile_predicate_to_wam_items(user:test_parent/2, [], Items),
        wam_target:wam_predicate_clauses(user:test_parent/2, [], Pred, Arity, Clauses),
        wam_target:compile_clauses_to_wam_items_native(Pred, Arity, Clauses, [], NativeItems),
        NativeItems == ExpectedItems,
        Items == ExpectedItems,
        Items == BridgeItems,
        member(switch_on_constant(["alice:default", "bob:L_test_parent_2_2_body"]), Items),
        member(label("test_parent/2"), Items)
    ->  pass(Test)
    ;   fail_test(Test, 'Native switch_on_constant items do not match canonical WAM text shape')
    ).

test_wam_items_native_switch_on_constant_fallthrough_index :-
    Test = 'WAM: native items API for switch_on_constant_fallthrough indexed clauses',
    ExpectedItems = [
        label("test_mma_middle/2"),
        switch_on_constant_fallthrough(["a:default"]),
        try_me_else("L_test_mma_middle_2_2"),
        get_constant("a", "A1"),
        get_constant("1", "A2"),
        proceed,
        label("L_test_mma_middle_2_2"),
        retry_me_else("L_test_mma_middle_2_3"),
        label("L_test_mma_middle_2_2_body"),
        get_variable("X1", "A1"),
        get_constant("99", "A2"),
        proceed,
        label("L_test_mma_middle_2_3"),
        trust_me,
        label("L_test_mma_middle_2_3_body"),
        get_constant("b", "A1"),
        get_constant("2", "A2"),
        proceed
    ],
    (   wam_target:compile_predicate_to_wam_text(user:test_mma_middle/2, [], TextCode),
        wam_text_to_items(TextCode, BridgeItems),
        wam_target:compile_predicate_to_wam_items(user:test_mma_middle/2, [], Items),
        wam_target:wam_predicate_clauses(user:test_mma_middle/2, [], Pred, Arity, Clauses),
        wam_target:compile_clauses_to_wam_items_native(Pred, Arity, Clauses, [], NativeItems),
        NativeItems == ExpectedItems,
        Items == ExpectedItems,
        Items == BridgeItems,
        member(switch_on_constant_fallthrough(["a:default"]), Items),
        member(label("test_mma_middle/2"), Items)
    ->  pass(Test)
    ;   fail_test(Test, 'Native switch_on_constant_fallthrough items do not match canonical WAM text shape')
    ).

test_wam_items_native_switch_on_constant_a2_index :-
    Test = 'WAM: native items API for switch_on_constant_a2 indexed clauses',
    ExpectedItems = [
        label("test_a2_const/2"),
        switch_on_constant_a2(["alpha:default", "beta:L_test_a2_const_2_2", "gamma:L_test_a2_const_2_3"]),
        try_me_else("L_test_a2_const_2_2"),
        get_variable("X1", "A1"),
        get_constant("alpha", "A2"),
        proceed,
        label("L_test_a2_const_2_2"),
        retry_me_else("L_test_a2_const_2_3"),
        label("L_test_a2_const_2_2_body"),
        get_variable("X1", "A1"),
        get_constant("beta", "A2"),
        proceed,
        label("L_test_a2_const_2_3"),
        trust_me,
        label("L_test_a2_const_2_3_body"),
        get_variable("X1", "A1"),
        get_constant("gamma", "A2"),
        proceed
    ],
    (   wam_target:compile_predicate_to_wam_text(user:test_a2_const/2, [], TextCode),
        wam_text_to_items(TextCode, BridgeItems),
        wam_target:compile_predicate_to_wam_items(user:test_a2_const/2, [], Items),
        wam_target:wam_predicate_clauses(user:test_a2_const/2, [], Pred, Arity, Clauses),
        wam_target:compile_clauses_to_wam_items_native(Pred, Arity, Clauses, [], NativeItems),
        NativeItems == ExpectedItems,
        Items == ExpectedItems,
        Items == BridgeItems,
        member(switch_on_constant_a2(_), Items),
        member(label("test_a2_const/2"), Items)
    ->  pass(Test)
    ;   fail_test(Test, 'Native switch_on_constant_a2 items do not match canonical WAM text shape')
    ).

test_wam_items_native_switch_on_constant_a2_fallthrough_index :-
    Test = 'WAM: native items API for switch_on_constant_a2_fallthrough indexed clauses',
    ExpectedItems = [
        label("test_mma2_trailing/3"),
        switch_on_constant_a2_fallthrough(["error:default", "warn:L_test_mma2_trailing_3_2", "ok:L_test_mma2_trailing_3_3"]),
        try_me_else("L_test_mma2_trailing_3_2"),
        get_variable("X1", "A1"),
        get_constant("error", "A2"),
        get_constant("red", "A3"),
        proceed,
        label("L_test_mma2_trailing_3_2"),
        retry_me_else("L_test_mma2_trailing_3_3"),
        label("L_test_mma2_trailing_3_2_body"),
        get_variable("X1", "A1"),
        get_constant("warn", "A2"),
        get_constant("yellow", "A3"),
        proceed,
        label("L_test_mma2_trailing_3_3"),
        retry_me_else("L_test_mma2_trailing_3_4"),
        label("L_test_mma2_trailing_3_3_body"),
        get_variable("X1", "A1"),
        get_constant("ok", "A2"),
        get_constant("green", "A3"),
        proceed,
        label("L_test_mma2_trailing_3_4"),
        trust_me,
        label("L_test_mma2_trailing_3_4_body"),
        get_variable("X1", "A1"),
        get_variable("X2", "A2"),
        get_constant("gray", "A3"),
        proceed
    ],
    (   wam_target:compile_predicate_to_wam_text(user:test_mma2_trailing/3, [], TextCode),
        wam_text_to_items(TextCode, BridgeItems),
        wam_target:compile_predicate_to_wam_items(user:test_mma2_trailing/3, [], Items),
        wam_target:wam_predicate_clauses(user:test_mma2_trailing/3, [], Pred, Arity, Clauses),
        wam_target:compile_clauses_to_wam_items_native(Pred, Arity, Clauses, [], NativeItems),
        NativeItems == ExpectedItems,
        Items == ExpectedItems,
        Items == BridgeItems,
        member(switch_on_constant_a2_fallthrough(_), Items),
        member(label("test_mma2_trailing/3"), Items)
    ->  pass(Test)
    ;   fail_test(Test, 'Native switch_on_constant_a2_fallthrough items do not match canonical WAM text shape')
    ).

test_wam_items_a2_structure_term_indexes_still_bridge :-
    Test = 'WAM: A2 structure/term indexed multi-clause items still use bridge',
    (   wam_target:wam_predicate_clauses(user:test_a2_struct/2, [], Pred1, Arity1, Clauses1),
        \+ wam_target:compile_clauses_to_wam_items_native(Pred1, Arity1, Clauses1, [], _NativeItems1),
        wam_target:compile_predicate_to_wam_text(user:test_a2_struct/2, [], TextCode1),
        wam_text_to_items(TextCode1, BridgeItems1),
        wam_target:compile_predicate_to_wam_items(user:test_a2_struct/2, [], Items1),
        Items1 == BridgeItems1,
        member(switch_on_structure_a2(_), Items1),
        wam_target:wam_predicate_clauses(user:test_a2_term/2, [], Pred2, Arity2, Clauses2),
        \+ wam_target:compile_clauses_to_wam_items_native(Pred2, Arity2, Clauses2, [], _NativeItems2),
        wam_target:compile_predicate_to_wam_text(user:test_a2_term/2, [], TextCode2),
        wam_text_to_items(TextCode2, BridgeItems2),
        wam_target:compile_predicate_to_wam_items(user:test_a2_term/2, [], Items2),
        Items2 == BridgeItems2,
        member(switch_on_term_a2(_), Items2)
    ->  pass(Test)
    ;   fail_test(Test, 'A2 structure/term indexed predicates did not remain on bridge')
    ).

%% Run all tests
run_tests :-
    format('~n========================================~n'),
    format('WAM Target Test Suite~n'),
    format('========================================~n~n'),
    
    test_wam_facts,
    test_wam_single_clause,
    test_wam_recursion,
    test_wam_put_structure,
    test_wam_nested_put_structure,
    test_wam_compound_head,
    test_wam_module,
    test_wam_items_native_single_fact,
    test_wam_items_native_simple_tail_call,
    test_wam_items_native_multi_goal_rule,
    test_wam_items_native_simple_builtin,
    test_wam_items_native_compound_body_arg,
    test_wam_items_native_nested_compound_body_arg,
    test_wam_items_native_list_body_arg,
    test_wam_items_compound_arg_legacy_order_fallback,
    test_wam_items_native_linear_fact_clauses,
    test_wam_items_native_linear_rule_clauses,
    test_wam_items_native_switch_on_constant_index,
    test_wam_items_native_switch_on_constant_fallthrough_index,
    test_wam_items_native_switch_on_constant_a2_index,
    test_wam_items_native_switch_on_constant_a2_fallthrough_index,
    test_wam_items_a2_structure_term_indexes_still_bridge,
    test_wam_multi_clause_findall_emits_allocate,
    test_wam_a2_indexing,
    test_wam_mixed_mode_a1_indexing,
    test_wam_mixed_mode_a2_indexing,

    format('~n========================================~n'),
    (   test_failed
    ->  format('Some tests FAILED~n'),
        format('========================================~n'),
        halt(1)
    ;   format('All tests passed~n'),
        format('========================================~n')
    ).

:- initialization(run_tests, main).
