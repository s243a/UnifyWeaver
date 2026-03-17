:- module(test_r_target, [run_all_tests/0]).

:- use_module(library(plunit)).
:- use_module('../src/unifyweaver/targets/type_declarations').
:- use_module('../src/unifyweaver/targets/r_target').

run_all_tests :-
    run_tests([r_target]).

setup_r_test :-
    clear_type_declarations,
    init_r_target.

cleanup_r_test :-
    clear_type_declarations,
    retractall(user:choose_value(_, _)),
    retractall(user:lower_number(_)).

:- begin_tests(r_target, [
    setup(setup_r_test),
    cleanup(cleanup_r_test)
]).

test(return_type_constraints_filter_incompatible_clauses) :-
    clear_type_declarations,
    init_r_target,
    assertz(user:(choose_value(number, Value) :- to_numeric(7, Value))),
    assertz(user:(choose_value(text, Value) :- string_lower('HI', Value))),
    assertz(type_declarations:uw_return_type(choose_value/2, number)),
    once(compile_predicate_to_r(choose_value/2, [], Code)),
    once(sub_string(Code, _, _, _, 'as.numeric(7)')),
    \+ sub_string(Code, _, _, _, 'tolower("HI")'),
    once(sub_string(Code, _, _, _, 'numeric()')).

test(return_type_constraints_can_be_disabled) :-
    clear_type_declarations,
    init_r_target,
    assertz(user:(choose_value(number, Value) :- to_numeric(7, Value))),
    assertz(user:(choose_value(text, Value) :- string_lower('HI', Value))),
    assertz(type_declarations:uw_return_type(choose_value/2, number)),
    once(compile_predicate_to_r(choose_value/2, [type_constraints(false)], Code)),
    once(sub_string(Code, _, _, _, 'as.numeric(7)')),
    once(sub_string(Code, _, _, _, 'tolower("HI")')),
    once(sub_string(Code, _, _, _, 'stop("No matching clause for choose_value")')).

test(single_clause_uses_typed_fallback_on_incompatible_return) :-
    clear_type_declarations,
    init_r_target,
    assertz(user:(lower_number(Value) :- string_lower('HI', Value))),
    assertz(type_declarations:uw_return_type(lower_number/1, number)),
    once(compile_predicate_to_r(lower_number/1, [], Code)),
    \+ sub_string(Code, _, _, _, 'tolower("HI")'),
    once(sub_string(Code, _, _, _, 'numeric()')).

:- end_tests(r_target).
