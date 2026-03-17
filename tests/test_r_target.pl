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
    retractall(user:lower_number(_)),
    retractall(user:choose_with_guard(_, _)),
    retractall(user:guarded_lower(_)),
    retractall(user:report_lower(_)).

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

test(conjunction_with_trailing_true_uses_last_meaningful_return_type) :-
    clear_type_declarations,
    init_r_target,
    assertz(user:(choose_with_guard(ok, Value) :- string_lower('HI', Value), true)),
    assertz(user:(choose_with_guard(num, Value) :- to_numeric(7, Value), true)),
    assertz(type_declarations:uw_return_type(choose_with_guard/2, number)),
    once(compile_predicate_to_r(choose_with_guard/2, [], Code)),
    once(sub_string(Code, _, _, _, 'as.numeric(7)')),
    \+ sub_string(Code, _, _, _, 'tolower("HI")').

test(type_diagnostics_error_throws_on_incompatible_single_clause) :-
    clear_type_declarations,
    init_r_target,
    assertz(user:(guarded_lower(Value) :- string_lower('HI', Value), true)),
    assertz(type_declarations:uw_return_type(guarded_lower/1, number)),
    catch(
        compile_predicate_to_r(guarded_lower/1, [type_diagnostics(error)], _),
        error(r_type_constraint_violation(guarded_lower/1, single_clause_fallback, number, string, _), _),
        Caught = true
    ),
    assertion(Caught == true).

test(type_diagnostics_warn_preserves_fallback_behavior) :-
    clear_type_declarations,
    init_r_target,
    assertz(user:(guarded_lower(Value) :- string_lower('HI', Value), true)),
    assertz(type_declarations:uw_return_type(guarded_lower/1, number)),
    once(compile_predicate_to_r(guarded_lower/1, [type_diagnostics(warn)], Code)),
    \+ sub_string(Code, _, _, _, 'tolower("HI")'),
    once(sub_string(Code, _, _, _, 'numeric()')).

test(type_diagnostics_report_collects_structured_entries) :-
    clear_type_declarations,
    init_r_target,
    assertz(user:(report_lower(Value) :- string_lower('HI', Value), true)),
    assertz(type_declarations:uw_return_type(report_lower/1, number)),
    once(compile_predicate_to_r(report_lower/1, [type_diagnostics_report(Report)], _Code)),
    Report = [Diagnostic],
    get_dict(target, Diagnostic, Target),
    get_dict(predicate, Diagnostic, PredSpec),
    get_dict(action, Diagnostic, Action),
    get_dict(expected, Diagnostic, Expected),
    get_dict(inferred, Diagnostic, Inferred),
    assertion(Target == r),
    assertion(PredSpec == report_lower/1),
    assertion(Action == single_clause_fallback),
    assertion(Expected == number),
    assertion(Inferred == string).

:- end_tests(r_target).
