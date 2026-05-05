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
