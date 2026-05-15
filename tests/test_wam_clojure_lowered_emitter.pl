:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module('../src/unifyweaver/targets/wam_clojure_lowered_emitter').

:- begin_tests(wam_clojure_lowered_emitter).

has(Code, Substr) :-
    once(sub_string(Code, _, _, _, Substr)).

simple_fact_wam("test_fact/1:\nget_constant a, A1\nproceed\n").

multi_clause_wam("test_choice/1:\ntry_me_else test_choice_alt\nget_constant a, A1\nproceed\ntest_choice_alt:\ntrust_me\nget_constant b, A1\nproceed\n").

unsupported_wam("test_switch/1:\nswitch_on_constant a:test_a, default:test_default\nproceed\n").

test(deterministic_predicate_is_lowerable) :-
    simple_fact_wam(WamCode),
    wam_clojure_lowerable(test_fact/1, WamCode, Reason),
    assertion(Reason == deterministic).

test(multi_clause_clause1_is_lowerable) :-
    multi_clause_wam(WamCode),
    wam_clojure_lowerable(test_choice/1, WamCode, Reason),
    assertion(Reason == multi_clause_1).

test(multi_clause_emits_empty_lowered_prefix) :-
    once((
        WamCode = "test_choice/1:\nallocate\ndeallocate\nexecute test_fact/1\ntry_me_else test_choice_alt\nallocate\ndeallocate\nexecute test_fact/1\ntest_choice_alt:\ntrust_me\nget_constant z, A1\nproceed\n",
        wam_clojure_lowerable(test_choice/1, WamCode, multi_clause_1),
        lower_predicate_to_clojure(test_choice/1, WamCode, [], Code),
        has(Code, "defn lowered-test-choice-1 [state]"),
        assertion(\+ has(Code, "update :env-stack conj {}")),
        assertion(\+ has(Code, "\"test_fact/1\"")),
        assertion(\+ has(Code, ":pc target-pc")),
        assertion(\+ has(Code, "runtime/step"))
    )).

test(switch_on_constant_not_yet_lowerable, [fail]) :-
    unsupported_wam(WamCode),
    wam_clojure_lowerable(test_switch/1, WamCode, _).

test(determinism_detection) :-
    simple_fact_wam(Simple),
    multi_clause_wam(Multi),
    wam_clojure_lowerable(test_fact/1, Simple, _),
    atom_string(Simple, _),
    atom_string(Multi, _),
    wam_clojure_lowered_emitter:is_deterministic_pred_clojure([get_constant(a, 'A1'), proceed]),
    \+ wam_clojure_lowered_emitter:is_deterministic_pred_clojure([try_me_else(foo), get_constant(a, 'A1'), proceed]).

test(function_name_generation) :-
    clojure_lowered_func_name('my_pred'/3, Name),
    assertion(Name == 'lowered-my-pred-3').

test(lower_predicate_to_clojure_emits_function) :-
    once((
        simple_fact_wam(WamCode),
        lower_predicate_to_clojure(test_fact/1, WamCode, [], Code),
        has(Code, "defn lowered-test-fact-1 [state]"),
        has(Code, "get-constant a, A1"),
        has(Code, "runtime/normalize-literal-term"),
        has(Code, "runtime/reg-get-raw"),
        has(Code, "runtime/interned-equal?")
    )).

test(lowered_multi_clause_stays_runtime_mediated) :-
    once((
        multi_clause_wam(WamCode),
        lower_predicate_to_clojure(test_choice/1, WamCode, [], Code),
        has(Code, "defn lowered-test-choice-1 [state]"),
        assertion(\+ has(Code, "get-constant a, A1")),
        assertion(\+ has(Code, "get-constant b, A1")),
        assertion(\+ has(Code, "runtime/step"))
    )).

test(call_foreign_is_part_of_scaffold) :-
    WamCode = "test_foreign/2:\ncall_foreign category_parent/2, 2\nproceed\n",
    wam_clojure_lowerable(test_foreign/2, WamCode, deterministic),
    lower_predicate_to_clojure(test_foreign/2, WamCode, [], Code),
    has(Code, "defn lowered-test-foreign-2 [state]"),
    has(Code, "state").

test(simple_register_ops_are_direct_lowered) :-
    once((
        WamCode = "test_regs/2:\nget_variable X1, A1\nput_variable X2, A2\nget_value X1, A2\nput_value X1, A1\nproceed\n",
        wam_clojure_lowerable(test_regs/2, WamCode, deterministic),
        lower_predicate_to_clojure(test_regs/2, WamCode, [], Code),
        has(Code, "runtime/fresh-var"),
        has(Code, "runtime/unify-values"),
        has(Code, "runtime/deref-value"),
        has(Code, "runtime/reg-set-raw")
    )).

test(structure_read_entry_is_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_read_struct/1:\nget_structure f/2, A1\nunify_constant a\nunify_variable X1\nproceed\n",
        wam_clojure_lowerable(test_read_struct/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_read_struct/1, WamCode, [], Code),
        has(Code, "runtime/enter-unify-mode"),
        has(Code, "runtime/pop-unify-item"),
        has(Code, "runtime/structure-term?"),
        has(Code, "runtime/interned-equal?"),
        has(Code, "runtime/unify-values")
    )).

test(list_read_entry_is_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_read_list/1:\nget_list A1\nunify_constant head\nunify_constant []\nproceed\n",
        wam_clojure_lowerable(test_read_list/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_read_list/1, WamCode, [], Code),
        has(Code, "runtime/enter-unify-mode"),
        has(Code, "runtime/list-functor-term"),
        has(Code, "runtime/structure-term?"),
        has(Code, "runtime/pop-unify-item"),
        has(Code, "runtime/unify-values")
    )).

test(unify_value_is_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_unify_value/2:\nget_variable X1, A2\nget_structure f/1, A1\nunify_value X1\nproceed\n",
        wam_clojure_lowerable(test_unify_value/2, WamCode, deterministic),
        lower_predicate_to_clojure(test_unify_value/2, WamCode, [], Code),
        has(Code, "runtime/pop-unify-item"),
        has(Code, "runtime/reg-get-raw"),
        has(Code, "runtime/unify-values")
    )).

test(simple_builtin_equality_is_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_eq/2:\nbuiltin_call =/2, 2\nproceed\n",
        wam_clojure_lowerable(test_eq/2, WamCode, deterministic),
        lower_predicate_to_clojure(test_eq/2, WamCode, [], Code),
        has(Code, "runtime/reg-get-raw"),
        has(Code, "\"A1\""),
        has(Code, "\"A2\""),
        has(Code, "runtime/unify-values")
    )).

test(simple_builtin_not_unify_is_direct_lowered_in_prefix) :-
    once((
        lower_predicate_to_clojure(test_not_unify/2, [builtin_call('\\=/2', 2), proceed], [], Code),
        has(Code, "runtime/reg-get-raw"),
        has(Code, "\"A1\""),
        has(Code, "\"A2\""),
        has(Code, "runtime/unifiable?"),
        has(Code, "runtime/backtrack")
    )).

test(simple_builtin_identity_is_direct_lowered_in_prefix) :-
    once((
        lower_predicate_to_clojure(test_identity/2, [builtin_call('==/2', 2), proceed], [], Code),
        has(Code, "runtime/reg-get-raw"),
        has(Code, "\"A1\""),
        has(Code, "\"A2\""),
        has(Code, "runtime/term-identical?"),
        has(Code, "runtime/backtrack")
    )).

test(simple_builtin_not_identity_is_direct_lowered_in_prefix) :-
    once((
        lower_predicate_to_clojure(test_not_identity/2, [builtin_call('\\==/2', 2), proceed], [], Code),
        has(Code, "runtime/reg-get-raw"),
        has(Code, "\"A1\""),
        has(Code, "\"A2\""),
        has(Code, "runtime/term-identical?"),
        has(Code, "runtime/backtrack")
    )).

test(simple_builtin_term_less_is_direct_lowered_in_prefix) :-
    once((
        lower_predicate_to_clojure(test_term_less/2, [builtin_call('@</2', 2), proceed], [], Code),
        has(Code, "runtime/reg-get-raw"),
        has(Code, "\"A1\""),
        has(Code, "\"A2\""),
        has(Code, "runtime/term-less?"),
        has(Code, "runtime/backtrack")
    )).

test(simple_builtin_term_less_or_equal_is_direct_lowered_in_prefix) :-
    once((
        lower_predicate_to_clojure(test_term_less_or_equal/2, [builtin_call('@=</2', 2), proceed], [], Code),
        has(Code, "runtime/reg-get-raw"),
        has(Code, "\"A1\""),
        has(Code, "\"A2\""),
        has(Code, "runtime/term-less-or-equal?"),
        has(Code, "runtime/backtrack")
    )).

test(simple_builtin_term_greater_is_direct_lowered_in_prefix) :-
    once((
        lower_predicate_to_clojure(test_term_greater/2, [builtin_call('@>/2', 2), proceed], [], Code),
        has(Code, "runtime/reg-get-raw"),
        has(Code, "\"A1\""),
        has(Code, "\"A2\""),
        has(Code, "runtime/term-greater?"),
        has(Code, "runtime/backtrack")
    )).

test(simple_builtin_term_greater_or_equal_is_direct_lowered_in_prefix) :-
    once((
        lower_predicate_to_clojure(test_term_greater_or_equal/2, [builtin_call('@>=/2', 2), proceed], [], Code),
        has(Code, "runtime/reg-get-raw"),
        has(Code, "\"A1\""),
        has(Code, "\"A2\""),
        has(Code, "runtime/term-greater-or-equal?"),
        has(Code, "runtime/backtrack")
    )).

test(simple_builtin_compare_is_direct_lowered_in_prefix) :-
    once((
        lower_predicate_to_clojure(test_compare/3, [builtin_call('compare/3', 3), proceed], [], Code),
        has(Code, "runtime/apply-compare-solution"),
        \+ has(Code, "runtime/step")
    )).

test(simple_compare_call_is_direct_lowered_in_prefix) :-
    once((
        lower_predicate_to_clojure(test_compare_call/0, [call('compare/3', 3), proceed], [], Code),
        has(Code, "runtime/apply-compare-solution"),
        \+ has(Code, "target-pc")
    )).

test(simple_builtin_arithmetic_equal_is_direct_lowered_in_prefix) :-
    once((
        lower_predicate_to_clojure(test_arith_eq/2, [builtin_call('=:=/2', 2), proceed], [], Code),
        has(Code, "runtime/reg-get-raw"),
        has(Code, "\"A1\""),
        has(Code, "\"A2\""),
        has(Code, "runtime/arithmetic-equal?"),
        has(Code, "runtime/backtrack")
    )).

test(simple_builtin_arithmetic_not_equal_is_direct_lowered_in_prefix) :-
    once((
        lower_predicate_to_clojure(test_arith_neq/2, [builtin_call('=\\=/2', 2), proceed], [], Code),
        has(Code, "runtime/reg-get-raw"),
        has(Code, "\"A1\""),
        has(Code, "\"A2\""),
        has(Code, "runtime/arithmetic-not-equal?"),
        has(Code, "runtime/backtrack")
    )).

test(simple_builtin_is_is_direct_lowered_in_prefix) :-
    once((
        lower_predicate_to_clojure(test_is/2, [builtin_call('is/2', 2), proceed], [], Code),
        has(Code, "runtime/reg-get-raw"),
        has(Code, "\"A1\""),
        has(Code, "\"A2\""),
        has(Code, "runtime/eval-arithmetic-term"),
        has(Code, "runtime/unify-values"),
        has(Code, "runtime/backtrack")
    )).

test(simple_builtin_arithmetic_less_is_direct_lowered_in_prefix) :-
    once((
        lower_predicate_to_clojure(test_arith_lt/2, [builtin_call('</2', 2), proceed], [], Code),
        has(Code, "runtime/reg-get-raw"),
        has(Code, "\"A1\""),
        has(Code, "\"A2\""),
        has(Code, "runtime/arithmetic-less?"),
        has(Code, "runtime/backtrack")
    )).

test(simple_builtin_arithmetic_greater_is_direct_lowered_in_prefix) :-
    once((
        lower_predicate_to_clojure(test_arith_gt/2, [builtin_call('>/2', 2), proceed], [], Code),
        has(Code, "runtime/reg-get-raw"),
        has(Code, "\"A1\""),
        has(Code, "\"A2\""),
        has(Code, "runtime/arithmetic-greater?"),
        has(Code, "runtime/backtrack")
    )).

test(simple_builtin_arithmetic_less_or_equal_is_direct_lowered_in_prefix) :-
    once((
        lower_predicate_to_clojure(test_arith_le/2, [builtin_call('=</2', 2), proceed], [], Code),
        has(Code, "runtime/reg-get-raw"),
        has(Code, "\"A1\""),
        has(Code, "\"A2\""),
        has(Code, "runtime/arithmetic-less-or-equal?"),
        has(Code, "runtime/backtrack")
    )).

test(simple_builtin_arithmetic_greater_or_equal_is_direct_lowered_in_prefix) :-
    once((
        lower_predicate_to_clojure(test_arith_ge/2, [builtin_call('>=/2', 2), proceed], [], Code),
        has(Code, "runtime/reg-get-raw"),
        has(Code, "\"A1\""),
        has(Code, "\"A2\""),
        has(Code, "runtime/arithmetic-greater-or-equal?"),
        has(Code, "runtime/backtrack")
    )).

test(simple_builtin_true_is_direct_lowered_in_prefix) :-
    once((
        lower_predicate_to_clojure(test_true/0, [builtin_call('true/0', 0), proceed], [], Code),
        has(Code, "runtime/advance"),
        has(Code, "runtime/succeed-state")
    )).

test(simple_builtin_fail_is_direct_lowered_in_prefix) :-
    once((
        lower_predicate_to_clojure(test_fail/0, [builtin_call('fail/0', 0), proceed], [], Code),
        has(Code, "runtime/backtrack"),
        assertion(\+ has(Code, "runtime/succeed-state")),
        assertion(\+ has(Code, "runtime/step"))
    )).

test(cut_builtin_is_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_cut/0:\nallocate\nbuiltin_call !/0, 0\ndeallocate\nproceed\n",
        wam_clojure_lowerable(test_cut/0, WamCode, deterministic),
        lower_predicate_to_clojure(test_cut/0, WamCode, [], Code),
        has(Code, "update :choice-points"),
        has(Code, "take (:cut-bar"),
        has(Code, "runtime/advance"),
        has(Code, "runtime/succeed-state")
    )).

test(simple_builtin_atom_is_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_atom/1:\nbuiltin_call atom/1, 1\nproceed\n",
        wam_clojure_lowerable(test_atom/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_atom/1, WamCode, [], Code),
        has(Code, "runtime/deref-value"),
        has(Code, "runtime/atom-term?"),
        has(Code, "runtime/advance"),
        has(Code, "runtime/backtrack")
    )).

test(simple_builtin_integer_is_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_integer/1:\nbuiltin_call integer/1, 1\nproceed\n",
        wam_clojure_lowerable(test_integer/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_integer/1, WamCode, [], Code),
        has(Code, "runtime/deref-value"),
        has(Code, "integer? value"),
        has(Code, "runtime/advance"),
        has(Code, "runtime/backtrack")
    )).

test(simple_builtin_number_is_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_number/1:\nbuiltin_call number/1, 1\nproceed\n",
        wam_clojure_lowerable(test_number/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_number/1, WamCode, [], Code),
        has(Code, "runtime/deref-value"),
        has(Code, "number? value"),
        has(Code, "runtime/advance"),
        has(Code, "runtime/backtrack")
    )).

test(simple_builtin_atomic_is_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_atomic/1:\nbuiltin_call atomic/1, 1\nproceed\n",
        wam_clojure_lowerable(test_atomic/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_atomic/1, WamCode, [], Code),
        has(Code, "runtime/deref-value"),
        has(Code, "runtime/atom-term? value"),
        has(Code, "number? value"),
        has(Code, "runtime/advance"),
        has(Code, "runtime/backtrack")
    )).

test(terminal_execute_atomic_is_direct_lowered_as_succeeding_builtin) :-
    once((
        WamCode = "test_execute_atomic/1:\nallocate\ndeallocate\nexecute atomic/1\n",
        wam_clojure_lowerable(test_execute_atomic/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_execute_atomic/1, WamCode, [], Code),
        has(Code, "runtime/atom-term? value"),
        has(Code, "number? value"),
        has(Code, "runtime/succeed-state next-state"),
        assertion(\+ has(Code, "\"atomic/1\"")),
        assertion(\+ has(Code, ":pc target-pc"))
    )).

test(simple_builtin_nonvar_is_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_nonvar/1:\nbuiltin_call nonvar/1, 1\nproceed\n",
        wam_clojure_lowerable(test_nonvar/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_nonvar/1, WamCode, [], Code),
        has(Code, "runtime/deref-value"),
        has(Code, "not= value ::lowered-unbound"),
        has(Code, "not (runtime/logic-var? value)"),
        has(Code, "runtime/advance"),
        has(Code, "runtime/backtrack")
    )).

test(terminal_execute_nonvar_is_direct_lowered_as_succeeding_builtin) :-
    once((
        WamCode = "test_execute_nonvar/1:\nallocate\ndeallocate\nexecute nonvar/1\n",
        wam_clojure_lowerable(test_execute_nonvar/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_execute_nonvar/1, WamCode, [], Code),
        has(Code, "not= value ::lowered-unbound"),
        has(Code, "not (runtime/logic-var? value)"),
        has(Code, "runtime/succeed-state next-state"),
        assertion(\+ has(Code, "\"nonvar/1\"")),
        assertion(\+ has(Code, ":pc target-pc"))
    )).

test(simple_builtin_var_is_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_var/1:\nbuiltin_call var/1, 1\nproceed\n",
        wam_clojure_lowerable(test_var/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_var/1, WamCode, [], Code),
        has(Code, "runtime/deref-value"),
        has(Code, "= value ::lowered-unbound"),
        has(Code, "runtime/logic-var? value"),
        has(Code, "runtime/advance"),
        has(Code, "runtime/backtrack")
    )).

test(terminal_execute_var_is_direct_lowered_as_succeeding_builtin) :-
    once((
        WamCode = "test_execute_var/1:\nallocate\ndeallocate\nexecute var/1\n",
        wam_clojure_lowerable(test_execute_var/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_execute_var/1, WamCode, [], Code),
        has(Code, "= value ::lowered-unbound"),
        has(Code, "runtime/logic-var? value"),
        has(Code, "runtime/succeed-state next-state"),
        assertion(\+ has(Code, "\"var/1\"")),
        assertion(\+ has(Code, ":pc target-pc"))
    )).

test(simple_builtin_compound_is_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_compound/1:\nbuiltin_call compound/1, 1\nproceed\n",
        wam_clojure_lowerable(test_compound/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_compound/1, WamCode, [], Code),
        has(Code, "runtime/deref-value"),
        has(Code, "runtime/structure-term? value"),
        has(Code, "runtime/advance"),
        has(Code, "runtime/backtrack")
    )).

test(terminal_execute_compound_is_direct_lowered_as_succeeding_builtin) :-
    once((
        WamCode = "test_execute_compound/1:\nallocate\ndeallocate\nexecute compound/1\n",
        wam_clojure_lowerable(test_execute_compound/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_execute_compound/1, WamCode, [], Code),
        has(Code, "runtime/structure-term? value"),
        has(Code, "runtime/succeed-state next-state"),
        assertion(\+ has(Code, "\"compound/1\"")),
        assertion(\+ has(Code, ":pc target-pc"))
    )).

test(simple_builtin_callable_is_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_callable/1:\nbuiltin_call callable/1, 1\nproceed\n",
        wam_clojure_lowerable(test_callable/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_callable/1, WamCode, [], Code),
        has(Code, "runtime/deref-value"),
        has(Code, "runtime/atom-term? value"),
        has(Code, "runtime/structure-term? value"),
        has(Code, "runtime/advance"),
        has(Code, "runtime/backtrack")
    )).

test(terminal_execute_callable_is_direct_lowered_as_succeeding_builtin) :-
    once((
        WamCode = "test_execute_callable/1:\nallocate\ndeallocate\nexecute callable/1\n",
        wam_clojure_lowerable(test_execute_callable/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_execute_callable/1, WamCode, [], Code),
        has(Code, "runtime/atom-term? value"),
        has(Code, "runtime/structure-term? value"),
        has(Code, "runtime/succeed-state next-state"),
        assertion(\+ has(Code, "\"callable/1\"")),
        assertion(\+ has(Code, ":pc target-pc"))
    )).

test(simple_builtin_float_is_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_float/1:\nbuiltin_call float/1, 1\nproceed\n",
        wam_clojure_lowerable(test_float/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_float/1, WamCode, [], Code),
        has(Code, "runtime/deref-value"),
        has(Code, "float? value"),
        has(Code, "runtime/advance"),
        has(Code, "runtime/backtrack")
    )).

test(terminal_execute_float_is_direct_lowered_as_succeeding_builtin) :-
    once((
        WamCode = "test_execute_float/1:\nallocate\ndeallocate\nexecute float/1\n",
        wam_clojure_lowerable(test_execute_float/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_execute_float/1, WamCode, [], Code),
        has(Code, "float? value"),
        has(Code, "runtime/succeed-state next-state"),
        assertion(\+ has(Code, "\"float/1\"")),
        assertion(\+ has(Code, ":pc target-pc"))
    )).

test(simple_builtin_is_list_is_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_is_list/1:\nbuiltin_call is_list/1, 1\nproceed\n",
        wam_clojure_lowerable(test_is_list/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_is_list/1, WamCode, [], Code),
        has(Code, "runtime/deref-value"),
        has(Code, "runtime/proper-list-term?"),
        has(Code, "runtime/advance"),
        has(Code, "runtime/backtrack")
    )).

test(terminal_execute_is_list_is_direct_lowered_as_succeeding_builtin) :-
    once((
        WamCode = "test_execute_is_list/1:\nallocate\ndeallocate\nexecute is_list/1\n",
        wam_clojure_lowerable(test_execute_is_list/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_execute_is_list/1, WamCode, [], Code),
        has(Code, "runtime/proper-list-term?"),
        has(Code, "runtime/succeed-state next-state"),
        assertion(\+ has(Code, "\"is_list/1\"")),
        assertion(\+ has(Code, ":pc target-pc"))
    )).

test(simple_builtin_length_is_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_length/2:\nbuiltin_call length/2, 2\nproceed\n",
        wam_clojure_lowerable(test_length/2, WamCode, deterministic),
        lower_predicate_to_clojure(test_length/2, WamCode, [], Code),
        has(Code, "runtime/deref-value"),
        has(Code, "runtime/proper-list-length"),
        has(Code, "runtime/unify-values"),
        has(Code, "runtime/advance"),
        has(Code, "runtime/backtrack")
    )).

test(simple_builtin_member_is_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_member/2:\nbuiltin_call member/2, 2\nproceed\n",
        wam_clojure_lowerable(test_member/2, WamCode, deterministic),
        lower_predicate_to_clojure(test_member/2, WamCode, [], Code),
        has(Code, "runtime/apply-member-solution"),
        has(Code, "\"A1\""),
        has(Code, "\"A2\""),
        has(Code, "inc (:pc"),
        has(Code, "runtime/deref-value"),
        assertion(\+ has(Code, "runtime/step"))
    )).

test(simple_builtin_append_is_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_append/3:\nbuiltin_call append/3, 3\nproceed\n",
        wam_clojure_lowerable(test_append/3, WamCode, deterministic),
        lower_predicate_to_clojure(test_append/3, WamCode, [], Code),
        has(Code, "runtime/apply-append-solution"),
        has(Code, "\"A1\""),
        has(Code, "\"A2\""),
        has(Code, "\"A3\""),
        has(Code, "inc (:pc"),
        assertion(\+ has(Code, "runtime/step"))
    )).

test(simple_builtin_sort_is_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_sort/2:\nbuiltin_call sort/2, 2\nproceed\n",
        wam_clojure_lowerable(test_sort/2, WamCode, deterministic),
        lower_predicate_to_clojure(test_sort/2, WamCode, [], Code),
        has(Code, "runtime/apply-sort-solution"),
        assertion(\+ has(Code, "runtime/step"))
    )).

test(simple_builtin_msort_is_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_msort/2:\nbuiltin_call msort/2, 2\nproceed\n",
        wam_clojure_lowerable(test_msort/2, WamCode, deterministic),
        lower_predicate_to_clojure(test_msort/2, WamCode, [], Code),
        has(Code, "runtime/apply-msort-solution"),
        assertion(\+ has(Code, "runtime/step"))
    )).

test(simple_builtin_copy_term_is_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_copy_term/2:\nbuiltin_call copy_term/2, 2\nproceed\n",
        wam_clojure_lowerable(test_copy_term/2, WamCode, deterministic),
        lower_predicate_to_clojure(test_copy_term/2, WamCode, [], Code),
        has(Code, "runtime/apply-copy-term-solution"),
        assertion(\+ has(Code, "runtime/step"))
    )).

test(simple_builtin_functor_is_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_functor/3:\nbuiltin_call functor/3, 3\nproceed\n",
        wam_clojure_lowerable(test_functor/3, WamCode, deterministic),
        lower_predicate_to_clojure(test_functor/3, WamCode, [], Code),
        has(Code, "runtime/apply-functor-solution"),
        assertion(\+ has(Code, "runtime/step"))
    )).

test(simple_builtin_arg_is_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_arg/3:\nbuiltin_call arg/3, 3\nproceed\n",
        wam_clojure_lowerable(test_arg/3, WamCode, deterministic),
        lower_predicate_to_clojure(test_arg/3, WamCode, [], Code),
        has(Code, "runtime/apply-arg-solution"),
        assertion(\+ has(Code, "runtime/step"))
    )).

test(simple_builtin_univ_is_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_univ/2:\nbuiltin_call =../2, 2\nproceed\n",
        wam_clojure_lowerable(test_univ/2, WamCode, deterministic),
        lower_predicate_to_clojure(test_univ/2, WamCode, [], Code),
        has(Code, "runtime/apply-univ-solution"),
        assertion(\+ has(Code, "runtime/step"))
    )).

test(simple_builtin_ground_is_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_ground/1:\nbuiltin_call ground/1, 1\nproceed\n",
        wam_clojure_lowerable(test_ground/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_ground/1, WamCode, [], Code),
        has(Code, "runtime/deref-value"),
        has(Code, "runtime/ground-term?"),
        has(Code, "runtime/advance"),
        has(Code, "runtime/backtrack")
    )).

test(terminal_execute_ground_is_direct_lowered_as_succeeding_builtin) :-
    once((
        WamCode = "test_execute_ground/1:\nallocate\ndeallocate\nexecute ground/1\n",
        wam_clojure_lowerable(test_execute_ground/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_execute_ground/1, WamCode, [], Code),
        has(Code, "runtime/ground-term?"),
        has(Code, "runtime/succeed-state next-state"),
        assertion(\+ has(Code, "\"ground/1\"")),
        assertion(\+ has(Code, ":pc target-pc"))
    )).

test(env_framed_equality_reaches_direct_builtin_prefix) :-
    once((
        WamCode = "test_env_eq/2:\nallocate\nput_value X1, A1\nput_value X2, A2\nbuiltin_call =/2, 2\ndeallocate\nproceed\n",
        wam_clojure_lowerable(test_env_eq/2, WamCode, deterministic),
        lower_predicate_to_clojure(test_env_eq/2, WamCode, [], Code),
        has(Code, "update :env-stack conj {}"),
        has(Code, "update :env-stack #(if (seq %) (pop %) %)"),
        has(Code, "runtime/unify-values"),
        has(Code, "runtime/succeed-state")
    )).

test(execute_is_direct_lowered_after_lowered_prefix) :-
    once((
        WamCode = "test_exec/1:\nallocate\ndeallocate\nexecute test_fact/1\n",
        wam_clojure_lowerable(test_exec/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_exec/1, WamCode, [], Code),
        has(Code, "update :env-stack conj {}"),
        has(Code, "update :env-stack #(if (seq %) (pop %) %)"),
        has(Code, "if-let [target-pc"),
        has(Code, "(get (:labels"),
        has(Code, "\"test_fact/1\""),
        has(Code, "assoc"),
        has(Code, ":pc target-pc"),
        has(Code, "runtime/backtrack"),
        assertion(\+ has(Code, "runtime/step"))
    )).

test(execute_terminal_stops_lowered_suffix) :-
    once((
        WamCode = "test_exec_suffix/1:\nexecute test_fact/1\ndeallocate\nproceed\n",
        wam_clojure_lowerable(test_exec_suffix/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_exec_suffix/1, WamCode, [], Code),
        has(Code, "\"test_fact/1\""),
        assertion(\+ has(Code, "update :env-stack #(if (seq %) (pop %) %)")),
        assertion(\+ has(Code, "runtime/succeed-state")),
        assertion(\+ has(Code, "runtime/step"))
    )).

test(multi_clause_execute_stays_runtime_mediated) :-
    once((
        WamCode = "test_choice_or_z/1:\ntry_me_else test_choice_or_z_alt\nexecute test_fact/1\ntest_choice_or_z_alt:\ntrust_me\nget_constant z, A1\nproceed\n",
        wam_clojure_lowerable(test_choice_or_z/1, WamCode, multi_clause_1),
        lower_predicate_to_clojure(test_choice_or_z/1, WamCode, [], Code),
        assertion(\+ has(Code, "\"test_fact/1\"")),
        assertion(\+ has(Code, ":pc target-pc")),
        assertion(\+ has(Code, "runtime/step"))
    )).

test(call_is_direct_lowered_after_lowered_prefix) :-
    once((
        WamCode = "test_call/1:\nallocate\nget_variable Y1, A1\nput_value Y1, A1\ncall test_fact/1, 1\ndeallocate\nproceed\n",
        wam_clojure_lowerable(test_call/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_call/1, WamCode, [], Code),
        has(Code, "update :env-stack conj {}"),
        has(Code, "get-variable Y1, A1"),
        has(Code, "put-value Y1, A1"),
        has(Code, "if-let [target-pc"),
        has(Code, "(get (:labels"),
        has(Code, "\"test_fact/1\""),
        has(Code, "update :stack conj (inc (:pc"),
        has(Code, ":pc target-pc"),
        has(Code, "runtime/backtrack"),
        assertion(\+ has(Code, "update :env-stack #(if (seq %) (pop %) %)")),
        assertion(\+ has(Code, "runtime/succeed-state")),
        assertion(\+ has(Code, "runtime/step"))
    )).

test(multi_clause_call_stays_runtime_mediated) :-
    once((
        WamCode = "test_choice_call/1:\ntry_me_else test_choice_call_alt\ncall test_fact/1, 1\nproceed\ntest_choice_call_alt:\ntrust_me\nget_constant z, A1\nproceed\n",
        wam_clojure_lowerable(test_choice_call/1, WamCode, multi_clause_1),
        lower_predicate_to_clojure(test_choice_call/1, WamCode, [], Code),
        assertion(\+ has(Code, "\"test_fact/1\"")),
        assertion(\+ has(Code, ":pc target-pc")),
        assertion(\+ has(Code, "update :stack conj")),
        assertion(\+ has(Code, "runtime/step"))
    )).

test(jump_is_direct_lowered_after_lowered_prefix) :-
    once((
        WamCode = "test_jump/1:\nallocate\njump test_jump_done\ntest_jump_done:\ndeallocate\nproceed\n",
        wam_clojure_lowerable(test_jump/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_jump/1, WamCode, [], Code),
        has(Code, "update :env-stack conj {}"),
        has(Code, "if-let [target-pc"),
        has(Code, "(get (:labels"),
        has(Code, "\"test_jump_done\""),
        has(Code, ":pc target-pc"),
        has(Code, "runtime/backtrack"),
        assertion(\+ has(Code, "update :env-stack #(if (seq %) (pop %) %)")),
        assertion(\+ has(Code, "runtime/succeed-state")),
        assertion(\+ has(Code, "runtime/step"))
    )).

test(jump_terminal_stops_lowered_suffix) :-
    once((
        WamCode = "test_jump_suffix/1:\njump test_jump_done\ndeallocate\nproceed\ntest_jump_done:\nproceed\n",
        wam_clojure_lowerable(test_jump_suffix/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_jump_suffix/1, WamCode, [], Code),
        has(Code, "\"test_jump_done\""),
        assertion(\+ has(Code, "update :env-stack #(if (seq %) (pop %) %)")),
        assertion(\+ has(Code, "runtime/succeed-state")),
        assertion(\+ has(Code, "runtime/step"))
    )).

test(multi_clause_jump_stays_runtime_mediated) :-
    once((
        WamCode = "test_choice_jump/1:\ntry_me_else test_choice_jump_alt\njump test_choice_jump_done\ntest_choice_jump_done:\nproceed\ntest_choice_jump_alt:\ntrust_me\nget_constant z, A1\nproceed\n",
        wam_clojure_lowerable(test_choice_jump/1, WamCode, multi_clause_1),
        lower_predicate_to_clojure(test_choice_jump/1, WamCode, [], Code),
        assertion(\+ has(Code, "\"test_choice_jump_done\"")),
        assertion(\+ has(Code, ":pc target-pc")),
        assertion(\+ has(Code, "runtime/step"))
    )).

test(structure_build_ops_are_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_build/1:\nput_structure f/2, A1\nset_constant a\nset_variable X1\nset_value X1\nproceed\n",
        wam_clojure_lowerable(test_build/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_build/1, WamCode, [], Code),
        has(Code, "runtime/push-build-frame"),
        has(Code, "runtime/functor-arity"),
        has(Code, "runtime/append-build-arg"),
        has(Code, "runtime/finalize-complete-builds")
    )).

test(list_build_ops_are_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_list_build/1:\nput_list A1\nset_constant head\nset_constant []\nproceed\n",
        wam_clojure_lowerable(test_list_build/1, WamCode, deterministic),
        lower_predicate_to_clojure(test_list_build/1, WamCode, [], Code),
        has(Code, "runtime/push-build-frame"),
        has(Code, "runtime/list-functor-term"),
        has(Code, "runtime/append-build-arg"),
        has(Code, "runtime/finalize-complete-builds")
    )).

test(build_arg_ops_are_direct_lowered_in_prefix) :-
    once((
        WamCode = "test_build_prefix/0:\nset_constant a\nset_variable X1\nset_value X1\nproceed\n",
        wam_clojure_lowerable(test_build_prefix/0, WamCode, deterministic),
        lower_predicate_to_clojure(test_build_prefix/0, WamCode, [], Code),
        has(Code, "runtime/append-build-arg"),
        has(Code, "runtime/finalize-complete-builds")
    )).

:- end_tests(wam_clojure_lowered_emitter).
