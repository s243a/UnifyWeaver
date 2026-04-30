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

test(lowered_multi_clause_keeps_clause1_shape) :-
    once((
        multi_clause_wam(WamCode),
        lower_predicate_to_clojure(test_choice/1, WamCode, [], Code),
        has(Code, "defn lowered-test-choice-1 [state]"),
        has(Code, "get-constant a, A1"),
        \+ has(Code, "get-constant b, A1")
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

:- end_tests(wam_clojure_lowered_emitter).
