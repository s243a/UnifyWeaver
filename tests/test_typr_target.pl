:- module(test_typr_target, [run_all_tests/0]).

:- use_module(library(plunit)).
:- use_module('../src/unifyweaver/targets/type_declarations').
:- use_module('../src/unifyweaver/targets/typr_target').
:- use_module('../src/unifyweaver/core/target_registry').
:- use_module('../src/unifyweaver/core/recursive_compiler').

run_all_tests :-
    run_tests([typr_target]).

setup_typr_test :-
    clear_type_declarations.

cleanup_typr_test :-
    clear_type_declarations,
    retractall(user:edge(_, _)),
    retractall(user:tc(_, _)),
    retractall(user:simple_fact(_)),
    retractall(user:lower_name(_, _)),
    retractall(user:classify_name(_, _)),
    retractall(user:inferred_lower(_, _)),
    retractall(user:classify_name_guarded(_, _)),
    retractall(user:guarded_name(_, _)),
    retractall(user:mid_guard(_, _)),
    retractall(user:multi_guard_chain(_, _)),
    retractall(user:multi_clause_chain(_, _)),
    retractall(user:comparison_guard_chain(_, _)),
    retractall(user:comparison_clause_chain(_, _)),
    retractall(user:fanout_chain(_, _)),
    retractall(user:fanout_clause_chain(_, _)),
    retractall(user:split_recombine_chain(_, _)),
    retractall(user:split_recombine_clause_chain(_, _)),
    retractall(user:if_then_else_choice(_, _)),
    retractall(user:if_then_else_choice_clause(_, _)),
    retractall(user:if_then_else_multi_result(_, _)),
    retractall(user:if_then_else_multi_result_clause(_, _)),
    retractall(user:if_then_choice(_, _)),
    retractall(user:if_then_choice_clause(_, _)),
    retractall(user:if_then_multi_result(_, _)),
    retractall(user:if_then_multi_result_clause(_, _)),
    retractall(user:if_then_guard_continuation(_, _)),
    retractall(user:if_then_guard_continuation_clause(_, _)),
    retractall(user:if_then_else_guard_continuation(_, _)),
    retractall(user:if_then_else_guard_continuation_clause(_, _)),
    retractall(user:if_then_else_asymmetric_rejoin(_, _)),
    retractall(user:if_then_else_asymmetric_rejoin_clause(_, _)),
    retractall(user:factorial_acc(_, _, _)),
    retractall(user:factorial_linear(_, _)),
    retractall(user:list_length(_, _)),
    retractall(user:power(_, _, _)),
    retractall(user:power_if(_, _, _)),
    retractall(user:power_multistate(_, _, _)),
    retractall(user:count_occ(_, _, _)),
    retractall(user:count_weighted(_, _, _)),
    retractall(user:list_length_from(_, _, _)),
    retractall(user:alternative_assign_chain(_, _)),
    retractall(user:alternative_assign_clause_chain(_, _)),
    retractall(user:direct_output_choice(_, _)),
    retractall(user:direct_output_choice_clause(_, _)),
    retractall(user:branch_local_multi_result(_, _)),
    retractall(user:branch_local_multi_result_clause(_, _)),
    retractall(user:multi_multi_rejoin(_, _)),
    retractall(user:multi_multi_rejoin_clause(_, _)),
    retractall(user:asymmetric_partial_rejoin(_, _)),
    retractall(user:asymmetric_partial_rejoin_clause(_, _)),
    retractall(user:multi_result_choice(_, _)),
    retractall(user:multi_result_choice_clause(_, _)),
    retractall(user:three_result_choice(_, _)),
    retractall(user:three_result_choice_clause(_, _)),
    retractall(user:nested_guarded_choice(_, _)),
    retractall(user:nested_guarded_choice_clause(_, _)),
    retractall(user:nested_multi_result_choice(_, _)),
    retractall(user:nested_multi_result_choice_clause(_, _)),
    retractall(user:two_level_nested_multi_result_choice(_, _)),
    retractall(user:two_level_nested_multi_result_choice_clause(_, _)),
    retractall(user:sort_rows(_, _)),
    retractall(user:filter_rows(_, _)),
    retractall(user:group_rows(_, _)),
    retractall(user:numeric_input(_)).

:- begin_tests(typr_target, [
    setup(setup_typr_test),
    cleanup(cleanup_typr_test)
]).

test(explicit_mode_emits_declared_scalar_annotations) :-
    clear_type_declarations,
    assertz(type_declarations:uw_type(edge/2, 1, atom)),
    assertz(type_declarations:uw_type(edge/2, 2, atom)),
    once(compile_predicate_to_typr(tc/2, [base_pred(edge), typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let tc_all <- fn(start: char): [#N, char]")),
    once(sub_string(Code, _, _, _, "let tc_check <- fn(start: char, target: char): bool")).

test(infer_mode_omits_scalar_parameter_annotations) :-
    clear_type_declarations,
    assertz(type_declarations:uw_type(edge/2, 1, atom)),
    assertz(type_declarations:uw_type(edge/2, 2, atom)),
    once(compile_predicate_to_typr(tc/2, [base_pred(edge), typed_mode(infer)], Code)),
    \+ sub_string(Code, _, _, _, "start: char"),
    once(sub_string(Code, _, _, _, "let tc_all <- fn(start)")).

test(explicit_any_is_preserved_in_infer_mode) :-
    clear_type_declarations,
    assertz(type_declarations:uw_type(edge/2, 1, any)),
    assertz(type_declarations:uw_type(edge/2, 2, any)),
    once(compile_predicate_to_typr(tc/2, [base_pred(edge), typed_mode(infer)], Code)),
    once(sub_string(Code, _, _, _, "start: Any")),
    once(sub_string(Code, _, _, _, "target: Any")).

test(per_predicate_typed_mode_overrides_call_option) :-
    clear_type_declarations,
    assertz(type_declarations:uw_type(edge/2, 1, atom)),
    assertz(type_declarations:uw_type(edge/2, 2, atom)),
    assertz(type_declarations:uw_typed_mode(tc/2, off)),
    once(compile_predicate_to_typr(tc/2, [base_pred(edge), typed_mode(explicit)], Code)),
    \+ sub_string(Code, _, _, _, ": char"),
    \+ sub_string(Code, _, _, _, ": [#N, char]").

test(target_registry_registers_typr_family) :-
    clear_type_declarations,
    target_exists(typr),
    target_family(typr, r).

test(target_registry_dispatches_typr) :-
    clear_type_declarations,
    assertz(type_declarations:uw_type(edge/2, 1, atom)),
    assertz(type_declarations:uw_type(edge/2, 2, atom)),
    once(compile_to_target(typr, tc/2, [base_pred(edge), typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "Generated by UnifyWeaver TypR Target")).

test(generic_predicates_receive_typed_signature) :-
    clear_type_declarations,
    assertz(user:simple_fact(hello)),
    assertz(type_declarations:uw_type(simple_fact/1, 1, atom)),
    once(compile_predicate_to_typr(simple_fact/1, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let simple_fact <- fn(arg1: char): bool")),
    once(sub_string(Code, _, _, _, "identical(arg1, \"hello\")")),
    generated_typr_is_valid(Code, exit(0)).

test(recursive_compiler_supports_typr_non_recursive_path) :-
    clear_type_declarations,
    assertz(user:simple_fact(hello)),
    assertz(type_declarations:uw_type(simple_fact/1, 1, atom)),
    once(recursive_compiler:compile_recursive(simple_fact/1, [target(typr), typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let simple_fact <- fn(arg1: char): bool")),
    generated_typr_is_valid(Code, exit(0)).

test(recursive_compiler_supports_typr_tail_recursion_path) :-
    clear_type_declarations,
    assertz(user:factorial_acc(0, Acc, Acc)),
    assertz(user:(factorial_acc(N, Acc, Result) :-
        N > 0,
        N1 is N - 1,
        Acc1 is Acc * N,
        factorial_acc(N1, Acc1, Result)
    )),
    assertz(type_declarations:uw_type(factorial_acc/3, 1, integer)),
    assertz(type_declarations:uw_type(factorial_acc/3, 2, integer)),
    assertz(type_declarations:uw_type(factorial_acc/3, 3, integer)),
    assertz(type_declarations:uw_return_type(factorial_acc/3, integer)),
    once(recursive_compiler:compile_recursive(factorial_acc/3, [target(typr), typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let factorial_acc <- fn(arg1: int, arg2: int, arg3: int): int")),
    once(sub_string(Code, _, _, _, "while (!identical(current_input, 0))")),
    once(sub_string(Code, _, _, _, "current_acc = step_2;")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(recursive_compiler_supports_typr_linear_recursion_path) :-
    clear_type_declarations,
    assertz(user:factorial_linear(0, 1)),
    assertz(user:(factorial_linear(N, Result) :-
        N > 0,
        N1 is N - 1,
        factorial_linear(N1, Prev),
        Result is N * Prev
    )),
    assertz(type_declarations:uw_type(factorial_linear/2, 1, integer)),
    assertz(type_declarations:uw_type(factorial_linear/2, 2, integer)),
    assertz(type_declarations:uw_return_type(factorial_linear/2, integer)),
    once(recursive_compiler:compile_recursive(factorial_linear/2, [target(typr), typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let factorial_linear <- fn(arg1: int, arg2: int): int")),
    once(sub_string(Code, _, _, _, "for (current in seq(current_input, 1))")),
    once(sub_string(Code, _, _, _, "acc = (current * acc);")),
    once(sub_string(Code, _, _, _, "stop(\"No matching recursive clause for factorial_linear\")")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(recursive_compiler_supports_typr_list_linear_recursion_path) :-
    clear_type_declarations,
    assertz(user:list_length([], 0)),
    assertz(user:(list_length([_|T], N) :-
        list_length(T, N1),
        N is N1 + 1
    )),
    assertz(type_declarations:uw_type(list_length/2, 1, list(any))),
    assertz(type_declarations:uw_type(list_length/2, 2, integer)),
    assertz(type_declarations:uw_return_type(list_length/2, integer)),
    once(recursive_compiler:compile_recursive(list_length/2, [target(typr), typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let list_length <- fn(arg1: [#N, Any], arg2: int): int")),
    once(sub_string(Code, _, _, _, "if (length(current_input) == 0)")),
    once(sub_string(Code, _, _, _, "for (current in rev(current_input))")),
    once(sub_string(Code, _, _, _, "acc = (acc + 1);")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(recursive_compiler_supports_typr_nary_linear_recursion_path) :-
    clear_type_declarations,
    assertz(user:power(_Base, 0, 1)),
    assertz(user:(power(Base, N, Result) :-
        N > 0,
        N1 is N - 1,
        power(Base, N1, Prev),
        Result is Base * Prev
    )),
    assertz(type_declarations:uw_type(power/3, 1, integer)),
    assertz(type_declarations:uw_type(power/3, 2, integer)),
    assertz(type_declarations:uw_type(power/3, 3, integer)),
    assertz(type_declarations:uw_return_type(power/3, integer)),
    once(recursive_compiler:compile_recursive(power/3, [target(typr), typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let power <- fn(arg1: int, arg2: int, arg3: int): int")),
    once(sub_string(Code, _, _, _, "current_input = arg2;")),
    once(sub_string(Code, _, _, _, "for (current in seq(current_input, 1))")),
    once(sub_string(Code, _, _, _, "acc = (arg1 * acc);")),
    once(sub_string(Code, _, _, _, "arg3 <- v4;")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(recursive_compiler_supports_typr_nary_list_linear_recursion_path) :-
    clear_type_declarations,
    assertz(user:list_length_from(Base, [], Base)),
    assertz(user:(list_length_from(Base, [_|T], N) :-
        list_length_from(Base, T, N1),
        N is N1 + 1
    )),
    assertz(type_declarations:uw_type(list_length_from/3, 1, integer)),
    assertz(type_declarations:uw_type(list_length_from/3, 2, list(any))),
    assertz(type_declarations:uw_type(list_length_from/3, 3, integer)),
    assertz(type_declarations:uw_return_type(list_length_from/3, integer)),
    once(recursive_compiler:compile_recursive(list_length_from/3, [target(typr), typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let list_length_from <- fn(arg1: int, arg2: [#N, Any], arg3: int): int")),
    once(sub_string(Code, _, _, _, "current_input = arg2;")),
    once(sub_string(Code, _, _, _, "if (length(current_input) == 0)")),
    once(sub_string(Code, _, _, _, "acc = (acc + 1);")),
    once(sub_string(Code, _, _, _, "arg3 <- v4;")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(recursive_compiler_supports_typr_guarded_nary_linear_recursion_path) :-
    clear_type_declarations,
    assertz(user:power_if(_Base, 0, 1)),
    assertz(user:(power_if(Base, N, Result) :-
        N > 0,
        N1 is N - 1,
        power_if(Base, N1, Prev),
        ( Base > 1 -> Result is Base * Prev ; Result is Prev )
    )),
    assertz(type_declarations:uw_type(power_if/3, 1, integer)),
    assertz(type_declarations:uw_type(power_if/3, 2, integer)),
    assertz(type_declarations:uw_type(power_if/3, 3, integer)),
    assertz(type_declarations:uw_return_type(power_if/3, integer)),
    once(recursive_compiler:compile_recursive(power_if/3, [target(typr), typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let power_if <- fn(arg1: int, arg2: int, arg3: int): int")),
    once(sub_string(Code, _, _, _, "current_input = arg2;")),
    once(sub_string(Code, _, _, _, "acc = if (@{ arg1 > 1 }@) { (arg1 * acc) } else { acc };")),
    once(sub_string(Code, _, _, _, "arg3 <- v4;")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(recursive_compiler_supports_typr_guarded_nary_list_linear_recursion_path) :-
    clear_type_declarations,
    assertz(user:count_occ(_, [], 0)),
    assertz(user:(count_occ(X, [Y|T], N) :-
        count_occ(X, T, N1),
        ( X == Y -> N is N1 + 1 ; N is N1 )
    )),
    assertz(type_declarations:uw_type(count_occ/3, 1, integer)),
    assertz(type_declarations:uw_type(count_occ/3, 2, list(integer))),
    assertz(type_declarations:uw_type(count_occ/3, 3, integer)),
    assertz(type_declarations:uw_return_type(count_occ/3, integer)),
    once(recursive_compiler:compile_recursive(count_occ/3, [target(typr), typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let count_occ <- fn(arg1: int, arg2: [#N, int], arg3: int): int")),
    once(sub_string(Code, _, _, _, "current_input = arg2;")),
    once(sub_string(Code, _, _, _, "acc = if (@{ arg1 == current }@) { (acc + 1) } else { acc };")),
    once(sub_string(Code, _, _, _, "arg3 <- v4;")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(recursive_compiler_supports_typr_multistate_nary_linear_recursion_path) :-
    clear_type_declarations,
    assertz(user:power_multistate(_Base, 0, 1)),
    assertz(user:(power_multistate(Base, N, Result) :-
        N > 0,
        N1 is N - 1,
        power_multistate(Base, N1, Prev),
        ( Base > 1 ->
            Step is Base * Prev,
            Offset is Step + 1
        ;   Step is Prev,
            Offset is Step + 2
        ),
        Result is Offset + Step
    )),
    assertz(type_declarations:uw_type(power_multistate/3, 1, integer)),
    assertz(type_declarations:uw_type(power_multistate/3, 2, integer)),
    assertz(type_declarations:uw_type(power_multistate/3, 3, integer)),
    assertz(type_declarations:uw_return_type(power_multistate/3, integer)),
    once(recursive_compiler:compile_recursive(power_multistate/3, [target(typr), typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let power_multistate <- fn(arg1: int, arg2: int, arg3: int): int")),
    once(sub_string(Code, _, _, _, "current_input = arg2;")),
    once(sub_string(Code, _, _, _, "if (@{ arg1 > 1 }@) { (arg1 * acc) } else { acc }")),
    once(sub_string(Code, _, _, _, "if (@{ arg1 > 1 }@) { ((arg1 * acc) + 1) } else { (acc + 2) }")),
    once(sub_string(Code, _, _, _, "arg3 <- v4;")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(recursive_compiler_supports_typr_multistate_nary_list_linear_recursion_path) :-
    clear_type_declarations,
    assertz(user:count_weighted(_, [], 0)),
    assertz(user:(count_weighted(X, [Y|T], N) :-
        count_weighted(X, T, N1),
        ( X == Y ->
            Delta is N1 + 1,
            Adjust is Delta + 1
        ;   Delta is N1,
            Adjust is Delta + 2
        ),
        N is Adjust + Delta
    )),
    assertz(type_declarations:uw_type(count_weighted/3, 1, integer)),
    assertz(type_declarations:uw_type(count_weighted/3, 2, list(integer))),
    assertz(type_declarations:uw_type(count_weighted/3, 3, integer)),
    assertz(type_declarations:uw_return_type(count_weighted/3, integer)),
    once(recursive_compiler:compile_recursive(count_weighted/3, [target(typr), typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let count_weighted <- fn(arg1: int, arg2: [#N, int], arg3: int): int")),
    once(sub_string(Code, _, _, _, "current_input = arg2;")),
    once(sub_string(Code, _, _, _, "if (@{ arg1 == current }@) { (acc + 1) } else { acc }")),
    once(sub_string(Code, _, _, _, "if (@{ arg1 == current }@) { ((acc + 1) + 1) } else { (acc + 2) }")),
    once(sub_string(Code, _, _, _, "arg3 <- v4;")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(generic_body_predicates_reuse_r_backend) :-
    clear_type_declarations,
    assertz(user:(lower_name(Name, Lower) :- string_lower(Name, Lower))),
    assertz(type_declarations:uw_type(lower_name/2, 1, atom)),
    assertz(type_declarations:uw_type(lower_name/2, 2, atom)),
    assertz(type_declarations:uw_return_type(lower_name/2, atom)),
    once(compile_predicate_to_typr(lower_name/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let lower_name <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "tolower(arg1)")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(generic_multi_clause_predicates_reuse_r_backend) :-
    clear_type_declarations,
    assertz(user:(classify_name(short, Lower) :- string_lower('HI', Lower))),
    assertz(user:(classify_name(long, Lower) :- string_lower('BYE', Lower))),
    assertz(type_declarations:uw_type(classify_name/2, 1, atom)),
    assertz(type_declarations:uw_type(classify_name/2, 2, atom)),
    once(compile_predicate_to_typr(classify_name/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let classify_name <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "else if")),
    once(sub_string(Code, _, _, _, "tolower(\"HI\")")),
    once(sub_string(Code, _, _, _, "tolower(\"BYE\")")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(generic_body_predicates_infer_return_type_without_declaration) :-
    clear_type_declarations,
    assertz(user:(inferred_lower(Name, Lower) :- string_lower(Name, Lower), true)),
    assertz(type_declarations:uw_type(inferred_lower/2, 1, atom)),
    assertz(type_declarations:uw_type(inferred_lower/2, 2, atom)),
    once(compile_predicate_to_typr(inferred_lower/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let inferred_lower <- fn(arg1: char, arg2: char): char")),
    generated_typr_is_valid(Code, exit(0)).

test(generic_multi_clause_predicates_infer_guarded_return_type) :-
    clear_type_declarations,
    assertz(user:(classify_name_guarded(short, Lower) :- string_lower('HI', Lower), true)),
    assertz(user:(classify_name_guarded(long, Lower) :- string_lower('BYE', Lower), true)),
    assertz(type_declarations:uw_type(classify_name_guarded/2, 1, atom)),
    assertz(type_declarations:uw_type(classify_name_guarded/2, 2, atom)),
    once(compile_predicate_to_typr(classify_name_guarded/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let classify_name_guarded <- fn(arg1: char, arg2: char): char")),
    generated_typr_is_valid(Code, exit(0)).

test(type_diagnostics_report_defaults_to_empty_for_native_typr_lowering) :-
    clear_type_declarations,
    assertz(user:(lower_name(Name, Lower) :- string_lower(Name, Lower))),
    assertz(type_declarations:uw_type(lower_name/2, 1, atom)),
    assertz(type_declarations:uw_type(lower_name/2, 2, atom)),
    once(compile_predicate_to_typr(lower_name/2, [typed_mode(explicit), type_diagnostics_report(Report)], _Code)),
    assertion(Report == []).

test(guard_bindings_expand_native_clause_conditions) :-
    clear_type_declarations,
    assertz(user:(guarded_name(Name, Lower) :- is_character(Name), string_lower(Name, Lower))),
    assertz(type_declarations:uw_type(guarded_name/2, 1, atom)),
    assertz(type_declarations:uw_type(guarded_name/2, 2, atom)),
    once(compile_predicate_to_typr(guarded_name/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "is.character(arg1)")),
    once(sub_string(Code, _, _, _, "tolower(arg1)")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(sequential_guards_after_output_lower_natively) :-
    clear_type_declarations,
    assertz(user:(mid_guard(Name, Len) :- string_lower(Name, Lower), is_character(Lower), string_length(Lower, Len))),
    assertz(type_declarations:uw_type(mid_guard/2, 1, atom)),
    assertz(type_declarations:uw_type(mid_guard/2, 2, integer)),
    once(compile_predicate_to_typr(mid_guard/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "tolower(arg1)")),
    once(sub_string(Code, _, _, _, "is.character(v3)")),
    once(sub_string(Code, _, _, _, "nchar(v3)")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(multi_decision_guard_chains_use_let_for_new_intermediates) :-
    clear_type_declarations,
    assertz(user:(multi_guard_chain(Name, Out) :- string_lower(Name, Lower), is_character(Lower), string_length(Lower, Len), is_numeric(Len), string_upper(Lower, Upper), is_character(Upper), string_concat(Upper, '!', Out))),
    assertz(type_declarations:uw_type(multi_guard_chain/2, 1, atom)),
    assertz(type_declarations:uw_type(multi_guard_chain/2, 2, atom)),
    once(compile_predicate_to_typr(multi_guard_chain/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let v3 <- @{ tolower(arg1) }@;")),
    once(sub_string(Code, _, _, _, "let v4 <- if (@{ is.character(v3) }@)")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ is.numeric(v4) }@)")),
    once(sub_string(Code, _, _, _, "arg2 <- if (@{ is.character(v5) }@)")),
    \+ sub_string(Code, _, _, _, "local({"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(multi_clause_decision_chains_stay_native_in_branch_bodies) :-
    clear_type_declarations,
    assertz(user:(multi_clause_chain(short, Out) :- string_lower('HI', Lower), is_character(Lower), string_length(Lower, Len), is_numeric(Len), string_upper(Lower, Out))),
    assertz(user:(multi_clause_chain(long, Out) :- string_lower('BYE', Lower), is_character(Lower), string_length(Lower, Len), is_numeric(Len), string_upper(Lower, Out))),
    assertz(type_declarations:uw_type(multi_clause_chain/2, 1, atom)),
    assertz(type_declarations:uw_type(multi_clause_chain/2, 2, atom)),
    once(compile_predicate_to_typr(multi_clause_chain/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "else if")),
    once(sub_string(Code, _, _, _, "let v3 <- @{ tolower(\"HI\") }@;")),
    once(sub_string(Code, _, _, _, "let v4 <- if (@{ is.character(v3) }@)")),
    once(sub_string(Code, _, _, _, "arg2 <- if (@{ is.numeric(v4) }@)")),
    \+ sub_string(Code, _, _, _, "local({"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(comparison_guard_chains_lower_natively) :-
    clear_type_declarations,
    assertz(user:(comparison_guard_chain(Name, Out) :- string_lower(Name, Lower), string_length(Lower, Len), >(Len, 1), string_upper(Lower, Upper), string_length(Upper, UpperLen), <(UpperLen, 10), string_concat(Upper, '!', Out))),
    assertz(type_declarations:uw_type(comparison_guard_chain/2, 1, atom)),
    assertz(type_declarations:uw_type(comparison_guard_chain/2, 2, atom)),
    once(compile_predicate_to_typr(comparison_guard_chain/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let v4 <- @{ nchar(v3) }@;")),
    once(sub_string(Code, _, _, _, "if (@{ v4 > 1 }@)")),
    once(sub_string(Code, _, _, _, "let v6 <- @{ nchar(v5) }@;")),
    once(sub_string(Code, _, _, _, "if (@{ v6 < 10 }@)")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(comparison_guard_branch_bodies_stay_native) :-
    clear_type_declarations,
    assertz(user:(comparison_clause_chain(short, Out) :- string_lower('HI', Lower), string_length(Lower, Len), >(Len, 1), string_upper(Lower, Out))),
    assertz(user:(comparison_clause_chain(long, Out) :- string_lower('BYE', Lower), string_length(Lower, Len), >(Len, 2), string_upper(Lower, Out))),
    assertz(type_declarations:uw_type(comparison_clause_chain/2, 1, atom)),
    assertz(type_declarations:uw_type(comparison_clause_chain/2, 2, atom)),
    once(compile_predicate_to_typr(comparison_clause_chain/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "else if")),
    once(sub_string(Code, _, _, _, "let v4 <- @{ nchar(v3) }@;")),
    once(sub_string(Code, _, _, _, "if (@{ v4 > 1 }@)")),
    \+ sub_string(Code, _, _, _, "local({"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(fanout_chains_reuse_one_intermediate_across_multiple_later_outputs) :-
    clear_type_declarations,
    assertz(user:(fanout_chain(Name, Out) :- string_lower(Name, Lower), string_length(Lower, Len), >(Len, 1), string_upper(Lower, Upper), string_concat(Lower, '?', Tagged), string_length(Tagged, TaggedLen), <(TaggedLen, 20), string_concat(Upper, Tagged, Out))),
    assertz(type_declarations:uw_type(fanout_chain/2, 1, atom)),
    assertz(type_declarations:uw_type(fanout_chain/2, 2, atom)),
    once(compile_predicate_to_typr(fanout_chain/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let v3 <- @{ tolower(arg1) }@;")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 1 }@)")),
    once(sub_string(Code, _, _, _, "let v6 <- @{ paste0(v3, \"?\") }@;")),
    once(sub_string(Code, _, _, _, "arg2 <- if (@{ v7 < 20 }@)")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(fanout_branch_bodies_stay_native) :-
    clear_type_declarations,
    assertz(user:(fanout_clause_chain(short, Out) :- string_lower('HI', Lower), string_length(Lower, Len), >(Len, 1), string_upper(Lower, Upper), string_concat(Lower, '?', Tagged), string_concat(Upper, Tagged, Out))),
    assertz(user:(fanout_clause_chain(long, Out) :- string_lower('BYE', Lower), string_length(Lower, Len), >(Len, 2), string_upper(Lower, Upper), string_concat(Lower, '?', Tagged), string_concat(Upper, Tagged, Out))),
    assertz(type_declarations:uw_type(fanout_clause_chain/2, 1, atom)),
    assertz(type_declarations:uw_type(fanout_clause_chain/2, 2, atom)),
    once(compile_predicate_to_typr(fanout_clause_chain/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "else if")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 1 }@)")),
    once(sub_string(Code, _, _, _, "let v6 <- @{ paste0(v3, \"?\") }@;")),
    once(sub_string(Code, _, _, _, "let arg2 <- @{ paste0(v5, v6) }@;")),
    \+ sub_string(Code, _, _, _, "local({"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(split_recombine_chains_stay_native) :-
    clear_type_declarations,
    assertz(user:(split_recombine_chain(Name, Out) :- string_lower(Name, Lower), string_length(Lower, LowerLen), >(LowerLen, 1), string_upper(Lower, Upper), string_length(Upper, UpperLen), <(UpperLen, 10), string_concat(Lower, '?', Tagged), string_length(Tagged, TaggedLen), <(TaggedLen, 20), string_concat(Upper, Tagged, Out))),
    assertz(type_declarations:uw_type(split_recombine_chain/2, 1, atom)),
    assertz(type_declarations:uw_type(split_recombine_chain/2, 2, atom)),
    once(compile_predicate_to_typr(split_recombine_chain/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 1 }@)")),
    once(sub_string(Code, _, _, _, "let v7 <- if (@{ v6 < 10 }@)")),
    once(sub_string(Code, _, _, _, "arg2 <- if (@{ v8 < 20 }@)")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(split_recombine_branch_bodies_stay_native) :-
    clear_type_declarations,
    assertz(user:(split_recombine_clause_chain(short, Out) :- string_lower('HI', Lower), string_length(Lower, LowerLen), >(LowerLen, 1), string_upper(Lower, Upper), string_length(Upper, UpperLen), <(UpperLen, 10), string_concat(Lower, '?', Tagged), string_length(Tagged, TaggedLen), <(TaggedLen, 20), string_concat(Upper, Tagged, Out))),
    assertz(user:(split_recombine_clause_chain(long, Out) :- string_lower('BYE', Lower), string_length(Lower, LowerLen), >(LowerLen, 2), string_upper(Lower, Upper), string_length(Upper, UpperLen), <(UpperLen, 10), string_concat(Lower, '?', Tagged), string_length(Tagged, TaggedLen), <(TaggedLen, 20), string_concat(Upper, Tagged, Out))),
    assertz(type_declarations:uw_type(split_recombine_clause_chain/2, 1, atom)),
    assertz(type_declarations:uw_type(split_recombine_clause_chain/2, 2, atom)),
    once(compile_predicate_to_typr(split_recombine_clause_chain/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "else if")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 1 }@)")),
    once(sub_string(Code, _, _, _, "let v7 <- if (@{ v6 < 10 }@)")),
    once(sub_string(Code, _, _, _, "arg2 <- if (@{ v8 < 20 }@)")),
    \+ sub_string(Code, _, _, _, "local({"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(if_then_else_single_result_stays_native) :-
    clear_type_declarations,
    assertz(user:(if_then_else_choice(Name, Out) :- string_lower(Name, Lower), string_length(Lower, Len), ((>(Len, 3) -> string_upper(Lower, Mid) ; string_concat(Lower, '?', Mid))), string_concat(Mid, '#', Out))),
    assertz(type_declarations:uw_type(if_then_else_choice/2, 1, atom)),
    assertz(type_declarations:uw_type(if_then_else_choice/2, 2, atom)),
    assertz(type_declarations:uw_return_type(if_then_else_choice/2, atom)),
    once(compile_predicate_to_typr(if_then_else_choice/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let if_then_else_choice <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "else if (TRUE)")),
    once(sub_string(Code, _, _, _, "arg2 <- @{ paste0(v5, \"#\") }@;")),
    \+ sub_string(Code, _, _, _, "Unknown predicate"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(if_then_else_single_result_stays_native_in_branch_bodies) :-
    clear_type_declarations,
    assertz(user:(if_then_else_choice_clause(long, Out) :- string_lower('Hello', Lower), string_length(Lower, Len), ((>(Len, 3) -> string_upper(Lower, Mid) ; string_concat(Lower, '?', Mid))), string_concat(Mid, '#', Out))),
    assertz(user:(if_then_else_choice_clause(short, Out) :- string_lower('Yo', Lower), string_length(Lower, Len), ((>(Len, 3) -> string_upper(Lower, Mid) ; string_concat(Lower, '?', Mid))), string_concat(Mid, '#', Out))),
    assertz(type_declarations:uw_type(if_then_else_choice_clause/2, 1, atom)),
    assertz(type_declarations:uw_type(if_then_else_choice_clause/2, 2, atom)),
    assertz(type_declarations:uw_return_type(if_then_else_choice_clause/2, atom)),
    once(compile_predicate_to_typr(if_then_else_choice_clause/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let if_then_else_choice_clause <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "else if")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "else if (TRUE)")),
    once(sub_string(Code, _, _, _, "let arg2 <- @{ paste0(v5, \"#\") }@;")),
    \+ sub_string(Code, _, _, _, "Unknown predicate"),
    \+ sub_string(Code, _, _, _, "local({"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(if_then_else_multi_result_stays_native) :-
    clear_type_declarations,
    assertz(user:(if_then_else_multi_result(Name, Out) :- string_lower(Name, Lower), string_length(Lower, Len), ((>(Len, 3) -> string_upper(Lower, Mid), string_concat('LONG', ':', Tag) ; string_concat(Lower, '?', Mid), string_concat('SHORT', ':', Tag))), string_concat(Mid, Tag, Out))),
    assertz(type_declarations:uw_type(if_then_else_multi_result/2, 1, atom)),
    assertz(type_declarations:uw_type(if_then_else_multi_result/2, 2, atom)),
    assertz(type_declarations:uw_return_type(if_then_else_multi_result/2, atom)),
    once(compile_predicate_to_typr(if_then_else_multi_result/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let if_then_else_multi_result <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "@{ list(v5, v6) }@")),
    once(sub_string(Code, _, _, _, "else if (TRUE)")),
    once(sub_string(Code, _, _, _, "let v6 <- @{ .subset2(v5, 1) }@;")),
    once(sub_string(Code, _, _, _, "let v7 <- @{ .subset2(v5, 2) }@;")),
    once(sub_string(Code, _, _, _, "arg2 <- @{ paste0(v6, v7) }@;")),
    \+ sub_string(Code, _, _, _, "Unknown predicate"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(if_then_else_multi_result_stays_native_in_branch_bodies) :-
    clear_type_declarations,
    assertz(user:(if_then_else_multi_result_clause(long, Out) :- string_lower('Hello', Lower), string_length(Lower, Len), ((>(Len, 3) -> string_upper(Lower, Mid), string_concat('LONG', ':', Tag) ; string_concat(Lower, '?', Mid), string_concat('SHORT', ':', Tag))), string_concat(Mid, Tag, Out))),
    assertz(user:(if_then_else_multi_result_clause(short, Out) :- string_lower('Yo', Lower), string_length(Lower, Len), ((>(Len, 3) -> string_upper(Lower, Mid), string_concat('LONG', ':', Tag) ; string_concat(Lower, '?', Mid), string_concat('SHORT', ':', Tag))), string_concat(Mid, Tag, Out))),
    assertz(type_declarations:uw_type(if_then_else_multi_result_clause/2, 1, atom)),
    assertz(type_declarations:uw_type(if_then_else_multi_result_clause/2, 2, atom)),
    assertz(type_declarations:uw_return_type(if_then_else_multi_result_clause/2, atom)),
    once(compile_predicate_to_typr(if_then_else_multi_result_clause/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let if_then_else_multi_result_clause <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "else if")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "@{ list(v5, v6) }@")),
    once(sub_string(Code, _, _, _, "else if (TRUE)")),
    once(sub_string(Code, _, _, _, "let v6 <- @{ .subset2(v5, 1) }@;")),
    once(sub_string(Code, _, _, _, "let v7 <- @{ .subset2(v5, 2) }@;")),
    once(sub_string(Code, _, _, _, "let arg2 <- @{ paste0(v6, v7) }@;")),
    \+ sub_string(Code, _, _, _, "Unknown predicate"),
    \+ sub_string(Code, _, _, _, "local({"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(if_then_single_result_stays_native) :-
    clear_type_declarations,
    assertz(user:(if_then_choice(Name, Out) :- string_lower(Name, Lower), string_length(Lower, Len), (>(Len, 3) -> string_upper(Lower, Mid)), string_concat(Mid, '#', Out))),
    assertz(type_declarations:uw_type(if_then_choice/2, 1, atom)),
    assertz(type_declarations:uw_type(if_then_choice/2, 2, atom)),
    assertz(type_declarations:uw_return_type(if_then_choice/2, atom)),
    once(compile_predicate_to_typr(if_then_choice/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let if_then_choice <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "stop(\"No matching clause for if_then_choice\")")),
    once(sub_string(Code, _, _, _, "arg2 <- @{ paste0(v5, \"#\") }@;")),
    \+ sub_string(Code, _, _, _, "Unknown predicate"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(if_then_single_result_stays_native_in_branch_bodies) :-
    clear_type_declarations,
    assertz(user:(if_then_choice_clause(long, Out) :- string_lower('Hello', Lower), string_length(Lower, Len), (>(Len, 3) -> string_upper(Lower, Mid)), string_concat(Mid, '#', Out))),
    assertz(user:(if_then_choice_clause(short, Out) :- string_lower('Okay', Lower), string_length(Lower, Len), (>(Len, 3) -> string_upper(Lower, Mid)), string_concat(Mid, '#', Out))),
    assertz(type_declarations:uw_type(if_then_choice_clause/2, 1, atom)),
    assertz(type_declarations:uw_type(if_then_choice_clause/2, 2, atom)),
    assertz(type_declarations:uw_return_type(if_then_choice_clause/2, atom)),
    once(compile_predicate_to_typr(if_then_choice_clause/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let if_then_choice_clause <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "else if")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "stop(\"No matching clause for if_then_choice_clause\")")),
    once(sub_string(Code, _, _, _, "let arg2 <- @{ paste0(v5, \"#\") }@;")),
    \+ sub_string(Code, _, _, _, "Unknown predicate"),
    \+ sub_string(Code, _, _, _, "local({"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(if_then_multi_result_stays_native) :-
    clear_type_declarations,
    assertz(user:(if_then_multi_result(Name, Out) :- string_lower(Name, Lower), string_length(Lower, Len), (>(Len, 3) -> string_upper(Lower, Mid), string_concat('LONG', ':', Tag)), string_concat(Mid, Tag, Out))),
    assertz(type_declarations:uw_type(if_then_multi_result/2, 1, atom)),
    assertz(type_declarations:uw_type(if_then_multi_result/2, 2, atom)),
    assertz(type_declarations:uw_return_type(if_then_multi_result/2, atom)),
    once(compile_predicate_to_typr(if_then_multi_result/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let if_then_multi_result <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "@{ list(v5, v6) }@")),
    once(sub_string(Code, _, _, _, "stop(\"No matching clause for if_then_multi_result\")")),
    once(sub_string(Code, _, _, _, "let v6 <- @{ .subset2(v5, 1) }@;")),
    once(sub_string(Code, _, _, _, "let v7 <- @{ .subset2(v5, 2) }@;")),
    once(sub_string(Code, _, _, _, "arg2 <- @{ paste0(v6, v7) }@;")),
    \+ sub_string(Code, _, _, _, "Unknown predicate"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(if_then_multi_result_stays_native_in_branch_bodies) :-
    clear_type_declarations,
    assertz(user:(if_then_multi_result_clause(long, Out) :- string_lower('Hello', Lower), string_length(Lower, Len), (>(Len, 3) -> string_upper(Lower, Mid), string_concat('LONG', ':', Tag)), string_concat(Mid, Tag, Out))),
    assertz(user:(if_then_multi_result_clause(short, Out) :- string_lower('Okay', Lower), string_length(Lower, Len), (>(Len, 3) -> string_upper(Lower, Mid), string_concat('LONG', ':', Tag)), string_concat(Mid, Tag, Out))),
    assertz(type_declarations:uw_type(if_then_multi_result_clause/2, 1, atom)),
    assertz(type_declarations:uw_type(if_then_multi_result_clause/2, 2, atom)),
    assertz(type_declarations:uw_return_type(if_then_multi_result_clause/2, atom)),
    once(compile_predicate_to_typr(if_then_multi_result_clause/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let if_then_multi_result_clause <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "else if")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "@{ list(v5, v6) }@")),
    once(sub_string(Code, _, _, _, "stop(\"No matching clause for if_then_multi_result_clause\")")),
    once(sub_string(Code, _, _, _, "let v6 <- @{ .subset2(v5, 1) }@;")),
    once(sub_string(Code, _, _, _, "let v7 <- @{ .subset2(v5, 2) }@;")),
    once(sub_string(Code, _, _, _, "let arg2 <- @{ paste0(v6, v7) }@;")),
    \+ sub_string(Code, _, _, _, "Unknown predicate"),
    \+ sub_string(Code, _, _, _, "local({"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(if_then_guard_only_continuation_stays_native) :-
    clear_type_declarations,
    assertz(user:(if_then_guard_continuation(Name, Out) :- string_lower(Name, Lower), string_length(Lower, Len), (>(Len, 3) -> is_character(Lower)), string_upper(Lower, Upper), string_concat(Upper, '#', Out))),
    assertz(type_declarations:uw_type(if_then_guard_continuation/2, 1, atom)),
    assertz(type_declarations:uw_type(if_then_guard_continuation/2, 2, atom)),
    assertz(type_declarations:uw_return_type(if_then_guard_continuation/2, atom)),
    once(compile_predicate_to_typr(if_then_guard_continuation/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let if_then_guard_continuation <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 && is.character(v3) }@)")),
    once(sub_string(Code, _, _, _, "arg2 <- @{ paste0(v5, \"#\") }@;")),
    \+ sub_string(Code, _, _, _, "Unknown predicate"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(if_then_guard_only_continuation_stays_native_in_branch_bodies) :-
    clear_type_declarations,
    assertz(user:(if_then_guard_continuation_clause(long, Out) :- string_lower('Hello', Lower), string_length(Lower, Len), (>(Len, 3) -> is_character(Lower)), string_upper(Lower, Upper), string_concat(Upper, '#', Out))),
    assertz(user:(if_then_guard_continuation_clause(short, Out) :- string_lower('Okay', Lower), string_length(Lower, Len), (>(Len, 3) -> is_character(Lower)), string_upper(Lower, Upper), string_concat(Upper, '#', Out))),
    assertz(type_declarations:uw_type(if_then_guard_continuation_clause/2, 1, atom)),
    assertz(type_declarations:uw_type(if_then_guard_continuation_clause/2, 2, atom)),
    assertz(type_declarations:uw_return_type(if_then_guard_continuation_clause/2, atom)),
    once(compile_predicate_to_typr(if_then_guard_continuation_clause/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let if_then_guard_continuation_clause <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "else if")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 && is.character(v3) }@)")),
    once(sub_string(Code, _, _, _, "let arg2 <- @{ paste0(v5, \"#\") }@;")),
    \+ sub_string(Code, _, _, _, "Unknown predicate"),
    \+ sub_string(Code, _, _, _, "local({"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(if_then_else_guard_only_continuation_stays_native) :-
    clear_type_declarations,
    assertz(user:(if_then_else_guard_continuation(Name, Out) :- string_lower(Name, Lower), string_length(Lower, Len), (>(Len, 3) -> is_character(Lower) ; =<(Len, 3), is_character(Lower)), string_upper(Lower, Upper), string_concat(Upper, '#', Out))),
    assertz(type_declarations:uw_type(if_then_else_guard_continuation/2, 1, atom)),
    assertz(type_declarations:uw_type(if_then_else_guard_continuation/2, 2, atom)),
    assertz(type_declarations:uw_return_type(if_then_else_guard_continuation/2, atom)),
    once(compile_predicate_to_typr(if_then_else_guard_continuation/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let if_then_else_guard_continuation <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ ifelse(v4 > 3, is.character(v3), v4 <= 3 && is.character(v3)) }@)")),
    once(sub_string(Code, _, _, _, "arg2 <- @{ paste0(v5, \"#\") }@;")),
    \+ sub_string(Code, _, _, _, "Unknown predicate"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(if_then_else_guard_only_continuation_stays_native_in_branch_bodies) :-
    clear_type_declarations,
    assertz(user:(if_then_else_guard_continuation_clause(long, Out) :- string_lower('Hello', Lower), string_length(Lower, Len), (>(Len, 3) -> is_character(Lower) ; =<(Len, 3), is_character(Lower)), string_upper(Lower, Upper), string_concat(Upper, '#', Out))),
    assertz(user:(if_then_else_guard_continuation_clause(short, Out) :- string_lower('Okay', Lower), string_length(Lower, Len), (>(Len, 3) -> is_character(Lower) ; =<(Len, 3), is_character(Lower)), string_upper(Lower, Upper), string_concat(Upper, '#', Out))),
    assertz(type_declarations:uw_type(if_then_else_guard_continuation_clause/2, 1, atom)),
    assertz(type_declarations:uw_type(if_then_else_guard_continuation_clause/2, 2, atom)),
    assertz(type_declarations:uw_return_type(if_then_else_guard_continuation_clause/2, atom)),
    once(compile_predicate_to_typr(if_then_else_guard_continuation_clause/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let if_then_else_guard_continuation_clause <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "else if")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ ifelse(v4 > 3, is.character(v3), v4 <= 3 && is.character(v3)) }@)")),
    once(sub_string(Code, _, _, _, "let arg2 <- @{ paste0(v5, \"#\") }@;")),
    \+ sub_string(Code, _, _, _, "Unknown predicate"),
    \+ sub_string(Code, _, _, _, "local({"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(if_then_else_asymmetric_multi_result_rejoin_stays_native) :-
    clear_type_declarations,
    assertz(user:(if_then_else_asymmetric_rejoin(Name, Out) :- string_lower(Name, Lower), string_length(Lower, Len), (>(Len, 3) -> string_upper(Lower, Upper), string_concat(Upper, '!', Mid), string_concat('LONG', ':', Tag) ; =<(Len, 3), string_concat(Lower, '?', Tagged), string_concat(Tagged, '!', Mid), string_concat('SHORT', ':', Tag)), string_length(Mid, MidLen), (>(MidLen, 6) -> string_concat(Tag, 'A', FinalTag) ; =<(MidLen, 6), string_concat(Tag, 'B', FinalTag)), string_concat(Mid, FinalTag, Out))),
    assertz(type_declarations:uw_type(if_then_else_asymmetric_rejoin/2, 1, atom)),
    assertz(type_declarations:uw_type(if_then_else_asymmetric_rejoin/2, 2, atom)),
    assertz(type_declarations:uw_return_type(if_then_else_asymmetric_rejoin/2, atom)),
    once(compile_predicate_to_typr(if_then_else_asymmetric_rejoin/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let if_then_else_asymmetric_rejoin <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "@{ list(v6, v7) }@")),
    once(sub_string(Code, _, _, _, "let v6 <- @{ .subset2(v5, 1) }@;")),
    once(sub_string(Code, _, _, _, "let v7 <- @{ .subset2(v5, 2) }@;")),
    once(sub_string(Code, _, _, _, "let v9 <- if (@{ v8 > 6 }@)")),
    once(sub_string(Code, _, _, _, "arg2 <- @{ paste0(v6, v9) }@;")),
    \+ sub_string(Code, _, _, _, "Unknown predicate"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(if_then_else_asymmetric_multi_result_rejoin_stays_native_in_branch_bodies) :-
    clear_type_declarations,
    assertz(user:(if_then_else_asymmetric_rejoin_clause(long, Out) :- string_lower('HelloThere', Lower), string_length(Lower, Len), (>(Len, 3) -> string_upper(Lower, Upper), string_concat(Upper, '!', Mid), string_concat('LONG', ':', Tag) ; =<(Len, 3), string_concat(Lower, '?', Tagged), string_concat(Tagged, '!', Mid), string_concat('SHORT', ':', Tag)), string_length(Mid, MidLen), (>(MidLen, 6) -> string_concat(Tag, 'A', FinalTag) ; =<(MidLen, 6), string_concat(Tag, 'B', FinalTag)), string_concat(Mid, FinalTag, Out))),
    assertz(user:(if_then_else_asymmetric_rejoin_clause(short, Out) :- string_lower('Yo', Lower), string_length(Lower, Len), (>(Len, 3) -> string_upper(Lower, Upper), string_concat(Upper, '!', Mid), string_concat('LONG', ':', Tag) ; =<(Len, 3), string_concat(Lower, '?', Tagged), string_concat(Tagged, '!', Mid), string_concat('SHORT', ':', Tag)), string_length(Mid, MidLen), (>(MidLen, 6) -> string_concat(Tag, 'A', FinalTag) ; =<(MidLen, 6), string_concat(Tag, 'B', FinalTag)), string_concat(Mid, FinalTag, Out))),
    assertz(type_declarations:uw_type(if_then_else_asymmetric_rejoin_clause/2, 1, atom)),
    assertz(type_declarations:uw_type(if_then_else_asymmetric_rejoin_clause/2, 2, atom)),
    assertz(type_declarations:uw_return_type(if_then_else_asymmetric_rejoin_clause/2, atom)),
    once(compile_predicate_to_typr(if_then_else_asymmetric_rejoin_clause/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let if_then_else_asymmetric_rejoin_clause <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "else if")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "@{ list(v6, v7) }@")),
    once(sub_string(Code, _, _, _, "let v6 <- @{ .subset2(v5, 1) }@;")),
    once(sub_string(Code, _, _, _, "let v7 <- @{ .subset2(v5, 2) }@;")),
    once(sub_string(Code, _, _, _, "let v9 <- if (@{ v8 > 6 }@)")),
    once(sub_string(Code, _, _, _, "let arg2 <- @{ paste0(v6, v9) }@;")),
    \+ sub_string(Code, _, _, _, "Unknown predicate"),
    \+ sub_string(Code, _, _, _, "local({"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(disjunction_alternative_assignments_lower_natively) :-
    clear_type_declarations,
    assertz(user:(alternative_assign_chain(Name, Out) :- string_lower(Name, Lower), string_length(Lower, Len), ((>(Len, 3), string_upper(Lower, Upper), string_concat(Upper, '!', Mid)); (=<(Len, 3), string_concat(Lower, '?', Mid))), string_concat(Mid, '#', Out))),
    assertz(type_declarations:uw_type(alternative_assign_chain/2, 1, atom)),
    assertz(type_declarations:uw_type(alternative_assign_chain/2, 2, atom)),
    once(compile_predicate_to_typr(alternative_assign_chain/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "let v6 <- @{ paste0(v5, \"!\") }@;")),
    once(sub_string(Code, _, _, _, "else if (@{ v4 <= 3 }@)")),
    once(sub_string(Code, _, _, _, "arg2 <- @{ paste0(v5, \"#\") }@;")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(disjunction_alternative_assignments_stay_native_in_branch_bodies) :-
    clear_type_declarations,
    assertz(user:(alternative_assign_clause_chain(short, Out) :- string_lower('Hello', Lower), string_length(Lower, Len), ((>(Len, 3), string_upper(Lower, Upper), string_concat(Upper, '!', Mid)); (=<(Len, 3), string_concat(Lower, '?', Mid))), string_concat(Mid, '#', Out))),
    assertz(user:(alternative_assign_clause_chain(tiny, Out) :- string_lower('Yo', Lower), string_length(Lower, Len), ((>(Len, 3), string_upper(Lower, Upper), string_concat(Upper, '!', Mid)); (=<(Len, 3), string_concat(Lower, '?', Mid))), string_concat(Mid, '#', Out))),
    assertz(type_declarations:uw_type(alternative_assign_clause_chain/2, 1, atom)),
    assertz(type_declarations:uw_type(alternative_assign_clause_chain/2, 2, atom)),
    once(compile_predicate_to_typr(alternative_assign_clause_chain/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "else if")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "let v6 <- @{ paste0(v5, \"!\") }@;")),
    once(sub_string(Code, _, _, _, "let arg2 <- @{ paste0(v5, \"#\") }@;")),
    \+ sub_string(Code, _, _, _, "local({"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(disjunction_direct_output_selection_stays_native) :-
    clear_type_declarations,
    assertz(user:(direct_output_choice(Name, Out) :- string_lower(Name, Lower), string_length(Lower, Len), ((>(Len, 3), string_upper(Lower, Upper), string_concat(Upper, '!', Out)); (=<(Len, 3), string_concat(Lower, '?', Out))))),
    assertz(type_declarations:uw_type(direct_output_choice/2, 1, atom)),
    assertz(type_declarations:uw_type(direct_output_choice/2, 2, atom)),
    assertz(type_declarations:uw_return_type(direct_output_choice/2, atom)),
    once(compile_predicate_to_typr(direct_output_choice/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let direct_output_choice <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "arg2 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "let arg2 <- @{ paste0(v5, \"!\") }@;")),
    once(sub_string(Code, _, _, _, "else if (@{ v4 <= 3 }@)")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(disjunction_direct_output_selection_stays_native_in_branch_bodies) :-
    clear_type_declarations,
    assertz(user:(direct_output_choice_clause(short, Out) :- string_lower('Hello', Lower), string_length(Lower, Len), ((>(Len, 3), string_upper(Lower, Upper), string_concat(Upper, '!', Out)); (=<(Len, 3), string_concat(Lower, '?', Out))))),
    assertz(user:(direct_output_choice_clause(tiny, Out) :- string_lower('Yo', Lower), string_length(Lower, Len), ((>(Len, 3), string_upper(Lower, Upper), string_concat(Upper, '!', Out)); (=<(Len, 3), string_concat(Lower, '?', Out))))),
    assertz(type_declarations:uw_type(direct_output_choice_clause/2, 1, atom)),
    assertz(type_declarations:uw_type(direct_output_choice_clause/2, 2, atom)),
    assertz(type_declarations:uw_return_type(direct_output_choice_clause/2, atom)),
    once(compile_predicate_to_typr(direct_output_choice_clause/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let direct_output_choice_clause <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "else if")),
    once(sub_string(Code, _, _, _, "arg2 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "let arg2 <- @{ paste0(v5, \"!\") }@;")),
    \+ sub_string(Code, _, _, _, "local({"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(disjunction_branch_local_multi_result_selection_stays_native) :-
    clear_type_declarations,
    assertz(user:(branch_local_multi_result(Name, Out) :- string_lower(Name, Lower), string_length(Lower, Len), ((>(Len, 3), string_upper(Lower, Upper), string_concat(Upper, '!', Mid), string_concat('LONG', ':', Tag)); (=<(Len, 3), string_concat(Lower, '?', Tagged), string_concat(Tagged, '!', Mid), string_concat('SHORT', ':', Tag))), string_concat(Mid, Tag, Out))),
    assertz(type_declarations:uw_type(branch_local_multi_result/2, 1, atom)),
    assertz(type_declarations:uw_type(branch_local_multi_result/2, 2, atom)),
    assertz(type_declarations:uw_return_type(branch_local_multi_result/2, atom)),
    once(compile_predicate_to_typr(branch_local_multi_result/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let branch_local_multi_result <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "let v5 <- @{ toupper(v3) }@;")),
    once(sub_string(Code, _, _, _, "let v5 <- @{ paste0(v3, \"?\") }@;")),
    once(sub_string(Code, _, _, _, "@{ list(v6, v7) }@")),
    once(sub_string(Code, _, _, _, "let v6 <- @{ .subset2(v5, 1) }@;")),
    once(sub_string(Code, _, _, _, "let v7 <- @{ .subset2(v5, 2) }@;")),
    once(sub_string(Code, _, _, _, "arg2 <- @{ paste0(v6, v7) }@;")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(disjunction_branch_local_multi_result_selection_stays_native_in_branch_bodies) :-
    clear_type_declarations,
    assertz(user:(branch_local_multi_result_clause(long, Out) :- string_lower('Hello', Lower), string_length(Lower, Len), ((>(Len, 3), string_upper(Lower, Upper), string_concat(Upper, '!', Mid), string_concat('LONG', ':', Tag)); (=<(Len, 3), string_concat(Lower, '?', Tagged), string_concat(Tagged, '!', Mid), string_concat('SHORT', ':', Tag))), string_concat(Mid, Tag, Out))),
    assertz(user:(branch_local_multi_result_clause(short, Out) :- string_lower('Yo', Lower), string_length(Lower, Len), ((>(Len, 3), string_upper(Lower, Upper), string_concat(Upper, '!', Mid), string_concat('LONG', ':', Tag)); (=<(Len, 3), string_concat(Lower, '?', Tagged), string_concat(Tagged, '!', Mid), string_concat('SHORT', ':', Tag))), string_concat(Mid, Tag, Out))),
    assertz(type_declarations:uw_type(branch_local_multi_result_clause/2, 1, atom)),
    assertz(type_declarations:uw_type(branch_local_multi_result_clause/2, 2, atom)),
    assertz(type_declarations:uw_return_type(branch_local_multi_result_clause/2, atom)),
    once(compile_predicate_to_typr(branch_local_multi_result_clause/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let branch_local_multi_result_clause <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "else if")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "let v5 <- @{ toupper(v3) }@;")),
    once(sub_string(Code, _, _, _, "let v5 <- @{ paste0(v3, \"?\") }@;")),
    once(sub_string(Code, _, _, _, "@{ list(v6, v7) }@")),
    once(sub_string(Code, _, _, _, "let v6 <- @{ .subset2(v5, 1) }@;")),
    once(sub_string(Code, _, _, _, "let v7 <- @{ .subset2(v5, 2) }@;")),
    once(sub_string(Code, _, _, _, "let arg2 <- @{ paste0(v6, v7) }@;")),
    \+ sub_string(Code, _, _, _, "local({"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(disjunction_multi_stage_rejoin_stays_native) :-
    clear_type_declarations,
    assertz(user:(multi_multi_rejoin(Name, Out) :- string_lower(Name, Lower), string_length(Lower, Len), ((>(Len, 3), string_upper(Lower, First), string_concat('LONG', ':', Tag1)); (=<(Len, 3), string_concat(Lower, '?', First), string_concat('SHORT', ':', Tag1))), string_length(First, FirstLen), ((>(FirstLen, 6), string_concat(First, '#', Second), string_concat(Tag1, 'A', Tag2)); (=<(FirstLen, 6), string_concat(First, '!', Second), string_concat(Tag1, 'B', Tag2))), string_concat(Second, Tag2, Out))),
    assertz(type_declarations:uw_type(multi_multi_rejoin/2, 1, atom)),
    assertz(type_declarations:uw_type(multi_multi_rejoin/2, 2, atom)),
    assertz(type_declarations:uw_return_type(multi_multi_rejoin/2, atom)),
    once(compile_predicate_to_typr(multi_multi_rejoin/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let multi_multi_rejoin <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "let v9 <- if (@{ v8 > 6 }@)")),
    once(sub_string(Code, _, _, _, "@{ list(v5, v6) }@")),
    once(sub_string(Code, _, _, _, "@{ list(v9, v10) }@")),
    once(sub_string(Code, _, _, _, "let v6 <- @{ .subset2(v5, 1) }@;")),
    once(sub_string(Code, _, _, _, "let v7 <- @{ .subset2(v5, 2) }@;")),
    once(sub_string(Code, _, _, _, "let v10 <- @{ .subset2(v9, 1) }@;")),
    once(sub_string(Code, _, _, _, "let v11 <- @{ .subset2(v9, 2) }@;")),
    once(sub_string(Code, _, _, _, "arg2 <- @{ paste0(v10, v11) }@;")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(disjunction_multi_stage_rejoin_stays_native_in_branch_bodies) :-
    clear_type_declarations,
    assertz(user:(multi_multi_rejoin_clause(long, Out) :- string_lower('HelloThere', Lower), string_length(Lower, Len), ((>(Len, 3), string_upper(Lower, First), string_concat('LONG', ':', Tag1)); (=<(Len, 3), string_concat(Lower, '?', First), string_concat('SHORT', ':', Tag1))), string_length(First, FirstLen), ((>(FirstLen, 6), string_concat(First, '#', Second), string_concat(Tag1, 'A', Tag2)); (=<(FirstLen, 6), string_concat(First, '!', Second), string_concat(Tag1, 'B', Tag2))), string_concat(Second, Tag2, Out))),
    assertz(user:(multi_multi_rejoin_clause(short, Out) :- string_lower('Yo', Lower), string_length(Lower, Len), ((>(Len, 3), string_upper(Lower, First), string_concat('LONG', ':', Tag1)); (=<(Len, 3), string_concat(Lower, '?', First), string_concat('SHORT', ':', Tag1))), string_length(First, FirstLen), ((>(FirstLen, 6), string_concat(First, '#', Second), string_concat(Tag1, 'A', Tag2)); (=<(FirstLen, 6), string_concat(First, '!', Second), string_concat(Tag1, 'B', Tag2))), string_concat(Second, Tag2, Out))),
    assertz(type_declarations:uw_type(multi_multi_rejoin_clause/2, 1, atom)),
    assertz(type_declarations:uw_type(multi_multi_rejoin_clause/2, 2, atom)),
    assertz(type_declarations:uw_return_type(multi_multi_rejoin_clause/2, atom)),
    once(compile_predicate_to_typr(multi_multi_rejoin_clause/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let multi_multi_rejoin_clause <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "else if")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "let v9 <- if (@{ v8 > 6 }@)")),
    once(sub_string(Code, _, _, _, "@{ list(v5, v6) }@")),
    once(sub_string(Code, _, _, _, "@{ list(v9, v10) }@")),
    once(sub_string(Code, _, _, _, "let v6 <- @{ .subset2(v5, 1) }@;")),
    once(sub_string(Code, _, _, _, "let v7 <- @{ .subset2(v5, 2) }@;")),
    once(sub_string(Code, _, _, _, "let v10 <- @{ .subset2(v9, 1) }@;")),
    once(sub_string(Code, _, _, _, "let v11 <- @{ .subset2(v9, 2) }@;")),
    once(sub_string(Code, _, _, _, "let arg2 <- @{ paste0(v10, v11) }@;")),
    \+ sub_string(Code, _, _, _, "local({"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(disjunction_asymmetric_partial_rejoin_stays_native) :-
    clear_type_declarations,
    assertz(user:(asymmetric_partial_rejoin(Name, Out) :- string_lower(Name, Lower), string_length(Lower, Len), ((>(Len, 3), string_upper(Lower, First)); (=<(Len, 3), string_concat(Lower, '?', First))), string_length(First, FirstLen), string_concat(First, ':', Tag1), ((>(FirstLen, 6), string_concat(First, '#', Second), string_concat(Tag1, 'A', Tag2)); (=<(FirstLen, 6), string_concat(First, '!', Second), string_concat(Tag1, 'B', Tag2))), string_concat(Second, Tag2, Out))),
    assertz(type_declarations:uw_type(asymmetric_partial_rejoin/2, 1, atom)),
    assertz(type_declarations:uw_type(asymmetric_partial_rejoin/2, 2, atom)),
    assertz(type_declarations:uw_return_type(asymmetric_partial_rejoin/2, atom)),
    once(compile_predicate_to_typr(asymmetric_partial_rejoin/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let asymmetric_partial_rejoin <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "let v7 <- @{ paste0(v5, \":\") }@;")),
    once(sub_string(Code, _, _, _, "let v8 <- if (@{ v6 > 6 }@)")),
    once(sub_string(Code, _, _, _, "@{ list(v8, v9) }@")),
    once(sub_string(Code, _, _, _, "let v9 <- @{ .subset2(v8, 1) }@;")),
    once(sub_string(Code, _, _, _, "let v10 <- @{ .subset2(v8, 2) }@;")),
    once(sub_string(Code, _, _, _, "arg2 <- @{ paste0(v9, v10) }@;")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(disjunction_asymmetric_partial_rejoin_stays_native_in_branch_bodies) :-
    clear_type_declarations,
    assertz(user:(asymmetric_partial_rejoin_clause(long, Out) :- string_lower('HelloThere', Lower), string_length(Lower, Len), ((>(Len, 3), string_upper(Lower, First)); (=<(Len, 3), string_concat(Lower, '?', First))), string_length(First, FirstLen), string_concat(First, ':', Tag1), ((>(FirstLen, 6), string_concat(First, '#', Second), string_concat(Tag1, 'A', Tag2)); (=<(FirstLen, 6), string_concat(First, '!', Second), string_concat(Tag1, 'B', Tag2))), string_concat(Second, Tag2, Out))),
    assertz(user:(asymmetric_partial_rejoin_clause(short, Out) :- string_lower('Yo', Lower), string_length(Lower, Len), ((>(Len, 3), string_upper(Lower, First)); (=<(Len, 3), string_concat(Lower, '?', First))), string_length(First, FirstLen), string_concat(First, ':', Tag1), ((>(FirstLen, 6), string_concat(First, '#', Second), string_concat(Tag1, 'A', Tag2)); (=<(FirstLen, 6), string_concat(First, '!', Second), string_concat(Tag1, 'B', Tag2))), string_concat(Second, Tag2, Out))),
    assertz(type_declarations:uw_type(asymmetric_partial_rejoin_clause/2, 1, atom)),
    assertz(type_declarations:uw_type(asymmetric_partial_rejoin_clause/2, 2, atom)),
    assertz(type_declarations:uw_return_type(asymmetric_partial_rejoin_clause/2, atom)),
    once(compile_predicate_to_typr(asymmetric_partial_rejoin_clause/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let asymmetric_partial_rejoin_clause <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "else if")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "let v7 <- @{ paste0(v5, \":\") }@;")),
    once(sub_string(Code, _, _, _, "let v8 <- if (@{ v6 > 6 }@)")),
    once(sub_string(Code, _, _, _, "@{ list(v8, v9) }@")),
    once(sub_string(Code, _, _, _, "let v9 <- @{ .subset2(v8, 1) }@;")),
    once(sub_string(Code, _, _, _, "let v10 <- @{ .subset2(v8, 2) }@;")),
    once(sub_string(Code, _, _, _, "let arg2 <- @{ paste0(v9, v10) }@;")),
    \+ sub_string(Code, _, _, _, "local({"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(disjunction_multi_result_selection_stays_native) :-
    clear_type_declarations,
    assertz(user:(multi_result_choice(Name, Out) :- string_lower(Name, Lower), string_length(Lower, Len), ((>(Len, 3), string_upper(Lower, Upper), string_concat(Upper, '!', Mid), string_concat('LONG', ':', Tag)); (=<(Len, 3), string_concat(Lower, '?', Mid), string_concat('SHORT', ':', Tag))), string_concat(Mid, Tag, Out))),
    assertz(type_declarations:uw_type(multi_result_choice/2, 1, atom)),
    assertz(type_declarations:uw_type(multi_result_choice/2, 2, atom)),
    assertz(type_declarations:uw_return_type(multi_result_choice/2, atom)),
    once(compile_predicate_to_typr(multi_result_choice/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let multi_result_choice <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "@{ list(v6, v7) }@")),
    once(sub_string(Code, _, _, _, "let v6 <- @{ .subset2(v5, 1) }@;")),
    once(sub_string(Code, _, _, _, "let v7 <- @{ .subset2(v5, 2) }@;")),
    once(sub_string(Code, _, _, _, "arg2 <- @{ paste0(v6, v7) }@;")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(disjunction_multi_result_selection_stays_native_in_branch_bodies) :-
    clear_type_declarations,
    assertz(user:(multi_result_choice_clause(short, Out) :- string_lower('Hello', Lower), string_length(Lower, Len), ((>(Len, 3), string_upper(Lower, Upper), string_concat(Upper, '!', Mid), string_concat('LONG', ':', Tag)); (=<(Len, 3), string_concat(Lower, '?', Mid), string_concat('SHORT', ':', Tag))), string_concat(Mid, Tag, Out))),
    assertz(user:(multi_result_choice_clause(tiny, Out) :- string_lower('Yo', Lower), string_length(Lower, Len), ((>(Len, 3), string_upper(Lower, Upper), string_concat(Upper, '!', Mid), string_concat('LONG', ':', Tag)); (=<(Len, 3), string_concat(Lower, '?', Mid), string_concat('SHORT', ':', Tag))), string_concat(Mid, Tag, Out))),
    assertz(type_declarations:uw_type(multi_result_choice_clause/2, 1, atom)),
    assertz(type_declarations:uw_type(multi_result_choice_clause/2, 2, atom)),
    assertz(type_declarations:uw_return_type(multi_result_choice_clause/2, atom)),
    once(compile_predicate_to_typr(multi_result_choice_clause/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let multi_result_choice_clause <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "else if")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "@{ list(v6, v7) }@")),
    once(sub_string(Code, _, _, _, "let v6 <- @{ .subset2(v5, 1) }@;")),
    once(sub_string(Code, _, _, _, "let v7 <- @{ .subset2(v5, 2) }@;")),
    once(sub_string(Code, _, _, _, "let arg2 <- @{ paste0(v6, v7) }@;")),
    \+ sub_string(Code, _, _, _, "local({"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(disjunction_three_result_selection_stays_native) :-
    clear_type_declarations,
    assertz(user:(three_result_choice(Name, Out) :- string_lower(Name, Lower), string_length(Lower, Len), ((>(Len, 3), string_upper(Lower, Upper), string_concat(Upper, '!', Mid), string_concat('LONG', ':', Tag), string_concat('*', Upper, Decor)); (=<(Len, 3), string_concat(Lower, '?', Mid), string_concat('SHORT', ':', Tag), string_concat('*', Lower, Decor))), string_concat(Mid, Tag, Base), string_concat(Base, Decor, Out))),
    assertz(type_declarations:uw_type(three_result_choice/2, 1, atom)),
    assertz(type_declarations:uw_type(three_result_choice/2, 2, atom)),
    assertz(type_declarations:uw_return_type(three_result_choice/2, atom)),
    once(compile_predicate_to_typr(three_result_choice/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let three_result_choice <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "@{ list(v6, v7, v8) }@")),
    once(sub_string(Code, _, _, _, "let v6 <- @{ .subset2(v5, 1) }@;")),
    once(sub_string(Code, _, _, _, "let v7 <- @{ .subset2(v5, 2) }@;")),
    once(sub_string(Code, _, _, _, "let v8 <- @{ .subset2(v5, 3) }@;")),
    once(sub_string(Code, _, _, _, "let v9 <- @{ paste0(v6, v7) }@;")),
    once(sub_string(Code, _, _, _, "arg2 <- @{ paste0(v9, v8) }@;")),
    \+ sub_string(Code, _, _, _, "NULL"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(disjunction_three_result_selection_stays_native_in_branch_bodies) :-
    clear_type_declarations,
    assertz(user:(three_result_choice_clause(short, Out) :- string_lower('Hello', Lower), string_length(Lower, Len), ((>(Len, 3), string_upper(Lower, Upper), string_concat(Upper, '!', Mid), string_concat('LONG', ':', Tag), string_concat('*', Upper, Decor)); (=<(Len, 3), string_concat(Lower, '?', Mid), string_concat('SHORT', ':', Tag), string_concat('*', Lower, Decor))), string_concat(Mid, Tag, Base), string_concat(Base, Decor, Out))),
    assertz(user:(three_result_choice_clause(tiny, Out) :- string_lower('Yo', Lower), string_length(Lower, Len), ((>(Len, 3), string_upper(Lower, Upper), string_concat(Upper, '!', Mid), string_concat('LONG', ':', Tag), string_concat('*', Upper, Decor)); (=<(Len, 3), string_concat(Lower, '?', Mid), string_concat('SHORT', ':', Tag), string_concat('*', Lower, Decor))), string_concat(Mid, Tag, Base), string_concat(Base, Decor, Out))),
    assertz(type_declarations:uw_type(three_result_choice_clause/2, 1, atom)),
    assertz(type_declarations:uw_type(three_result_choice_clause/2, 2, atom)),
    assertz(type_declarations:uw_return_type(three_result_choice_clause/2, atom)),
    once(compile_predicate_to_typr(three_result_choice_clause/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let three_result_choice_clause <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "else if")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "@{ list(v6, v7, v8) }@")),
    once(sub_string(Code, _, _, _, "let v6 <- @{ .subset2(v5, 1) }@;")),
    once(sub_string(Code, _, _, _, "let v7 <- @{ .subset2(v5, 2) }@;")),
    once(sub_string(Code, _, _, _, "let v8 <- @{ .subset2(v5, 3) }@;")),
    once(sub_string(Code, _, _, _, "let v9 <- @{ paste0(v6, v7) }@;")),
    once(sub_string(Code, _, _, _, "let arg2 <- @{ paste0(v9, v8) }@;")),
    \+ sub_string(Code, _, _, _, "NULL"),
    \+ sub_string(Code, _, _, _, "local({"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(nested_guarded_alternatives_stay_native) :-
    clear_type_declarations,
    assertz(user:(nested_guarded_choice(Name, Out) :- string_lower(Name, Lower), string_length(Lower, Len), ((>(Len, 3), ((is_character(Lower), string_upper(Lower, Mid)); (=<(Len, 6), string_concat(Lower, '?', Mid)))); (=<(Len, 3), string_concat(Lower, '!', Mid))), string_concat(Mid, '#', Out))),
    assertz(type_declarations:uw_type(nested_guarded_choice/2, 1, atom)),
    assertz(type_declarations:uw_type(nested_guarded_choice/2, 2, atom)),
    assertz(type_declarations:uw_return_type(nested_guarded_choice/2, atom)),
    once(compile_predicate_to_typr(nested_guarded_choice/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let nested_guarded_choice <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ is.character(v3) }@)")),
    once(sub_string(Code, _, _, _, "else if (@{ v4 <= 6 }@)")),
    once(sub_string(Code, _, _, _, "else if (@{ v4 <= 3 }@)")),
    once(sub_string(Code, _, _, _, "arg2 <- @{ paste0(v5, \"#\") }@;")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(nested_guarded_alternatives_stay_native_in_branch_bodies) :-
    clear_type_declarations,
    assertz(user:(nested_guarded_choice_clause(short, Out) :- string_lower('Hello', Lower), string_length(Lower, Len), ((>(Len, 3), ((is_character(Lower), string_upper(Lower, Mid)); (=<(Len, 6), string_concat(Lower, '?', Mid)))); (=<(Len, 3), string_concat(Lower, '!', Mid))), string_concat(Mid, '#', Out))),
    assertz(user:(nested_guarded_choice_clause(tiny, Out) :- string_lower('Yo', Lower), string_length(Lower, Len), ((>(Len, 3), ((is_character(Lower), string_upper(Lower, Mid)); (=<(Len, 6), string_concat(Lower, '?', Mid)))); (=<(Len, 3), string_concat(Lower, '!', Mid))), string_concat(Mid, '#', Out))),
    assertz(type_declarations:uw_type(nested_guarded_choice_clause/2, 1, atom)),
    assertz(type_declarations:uw_type(nested_guarded_choice_clause/2, 2, atom)),
    assertz(type_declarations:uw_return_type(nested_guarded_choice_clause/2, atom)),
    once(compile_predicate_to_typr(nested_guarded_choice_clause/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let nested_guarded_choice_clause <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "else if")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ is.character(v3) }@)")),
    once(sub_string(Code, _, _, _, "else if (@{ v4 <= 6 }@)")),
    once(sub_string(Code, _, _, _, "let arg2 <- @{ paste0(v5, \"#\") }@;")),
    \+ sub_string(Code, _, _, _, "local({"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(nested_multi_result_alternatives_stay_native) :-
    clear_type_declarations,
    assertz(user:(nested_multi_result_choice(Name, Out) :- string_lower(Name, Lower), string_length(Lower, Len), ((>(Len, 3), ((is_character(Lower), string_upper(Lower, Mid), string_concat('LONG', ':', Tag)); (=<(Len, 6), string_concat(Lower, '?', Mid), string_concat('MID', ':', Tag)))); (=<(Len, 3), string_concat(Lower, '!', Mid), string_concat('SHORT', ':', Tag))), string_concat(Mid, Tag, Out))),
    assertz(type_declarations:uw_type(nested_multi_result_choice/2, 1, atom)),
    assertz(type_declarations:uw_type(nested_multi_result_choice/2, 2, atom)),
    assertz(type_declarations:uw_return_type(nested_multi_result_choice/2, atom)),
    once(compile_predicate_to_typr(nested_multi_result_choice/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let nested_multi_result_choice <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ is.character(v3) }@)")),
    once(sub_string(Code, _, _, _, "@{ list(v5, v6) }@")),
    once(sub_string(Code, _, _, _, "let v6 <- @{ .subset2(v5, 1) }@;")),
    once(sub_string(Code, _, _, _, "let v7 <- @{ .subset2(v5, 2) }@;")),
    once(sub_string(Code, _, _, _, "arg2 <- @{ paste0(v6, v7) }@;")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(nested_multi_result_alternatives_stay_native_in_branch_bodies) :-
    clear_type_declarations,
    assertz(user:(nested_multi_result_choice_clause(short, Out) :- string_lower('Hello', Lower), string_length(Lower, Len), ((>(Len, 3), ((is_character(Lower), string_upper(Lower, Mid), string_concat('LONG', ':', Tag)); (=<(Len, 6), string_concat(Lower, '?', Mid), string_concat('MID', ':', Tag)))); (=<(Len, 3), string_concat(Lower, '!', Mid), string_concat('SHORT', ':', Tag))), string_concat(Mid, Tag, Out))),
    assertz(user:(nested_multi_result_choice_clause(tiny, Out) :- string_lower('Yo', Lower), string_length(Lower, Len), ((>(Len, 3), ((is_character(Lower), string_upper(Lower, Mid), string_concat('LONG', ':', Tag)); (=<(Len, 6), string_concat(Lower, '?', Mid), string_concat('MID', ':', Tag)))); (=<(Len, 3), string_concat(Lower, '!', Mid), string_concat('SHORT', ':', Tag))), string_concat(Mid, Tag, Out))),
    assertz(type_declarations:uw_type(nested_multi_result_choice_clause/2, 1, atom)),
    assertz(type_declarations:uw_type(nested_multi_result_choice_clause/2, 2, atom)),
    assertz(type_declarations:uw_return_type(nested_multi_result_choice_clause/2, atom)),
    once(compile_predicate_to_typr(nested_multi_result_choice_clause/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let nested_multi_result_choice_clause <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "else if")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 3 }@)")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ is.character(v3) }@)")),
    once(sub_string(Code, _, _, _, "@{ list(v5, v6) }@")),
    once(sub_string(Code, _, _, _, "let v6 <- @{ .subset2(v5, 1) }@;")),
    once(sub_string(Code, _, _, _, "let v7 <- @{ .subset2(v5, 2) }@;")),
    once(sub_string(Code, _, _, _, "let arg2 <- @{ paste0(v6, v7) }@;")),
    \+ sub_string(Code, _, _, _, "local({"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(two_level_nested_multi_result_alternatives_stay_native) :-
    clear_type_declarations,
    assertz(user:(two_level_nested_multi_result_choice(Name, Out) :- string_lower(Name, Lower), string_length(Lower, Len), ((>(Len, 8), ((is_character(Lower), ((>(Len, 10), string_upper(Lower, Mid), string_concat('LONG', ':', Tag)); (=<(Len, 10), string_concat(Lower, '?', Mid), string_concat('MID', ':', Tag)))); (=<(Len, 8), string_concat(Lower, '!', Mid), string_concat('SHORT', ':', Tag)))); (=<(Len, 8), string_concat(Lower, '.', Mid), string_concat('TINY', ':', Tag))), string_concat(Mid, Tag, Out))),
    assertz(type_declarations:uw_type(two_level_nested_multi_result_choice/2, 1, atom)),
    assertz(type_declarations:uw_type(two_level_nested_multi_result_choice/2, 2, atom)),
    assertz(type_declarations:uw_return_type(two_level_nested_multi_result_choice/2, atom)),
    once(compile_predicate_to_typr(two_level_nested_multi_result_choice/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let two_level_nested_multi_result_choice <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 8 }@)")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ is.character(v3) }@)")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 10 }@)")),
    once(sub_string(Code, _, _, _, "else if (@{ v4 <= 10 }@)")),
    once(sub_string(Code, _, _, _, "@{ list(v5, v6) }@")),
    once(sub_string(Code, _, _, _, "let v6 <- @{ .subset2(v5, 1) }@;")),
    once(sub_string(Code, _, _, _, "let v7 <- @{ .subset2(v5, 2) }@;")),
    once(sub_string(Code, _, _, _, "arg2 <- @{ paste0(v6, v7) }@;")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(two_level_nested_multi_result_alternatives_stay_native_in_branch_bodies) :-
    clear_type_declarations,
    assertz(user:(two_level_nested_multi_result_choice_clause(long, Out) :- string_lower('HELLOTHERE', Lower), string_length(Lower, Len), ((>(Len, 8), ((is_character(Lower), ((>(Len, 10), string_upper(Lower, Mid), string_concat('LONG', ':', Tag)); (=<(Len, 10), string_concat(Lower, '?', Mid), string_concat('MID', ':', Tag)))); (=<(Len, 8), string_concat(Lower, '!', Mid), string_concat('SHORT', ':', Tag)))); (=<(Len, 8), string_concat(Lower, '.', Mid), string_concat('TINY', ':', Tag))), string_concat(Mid, Tag, Out))),
    assertz(user:(two_level_nested_multi_result_choice_clause(short, Out) :- string_lower('Hey', Lower), string_length(Lower, Len), ((>(Len, 8), ((is_character(Lower), ((>(Len, 10), string_upper(Lower, Mid), string_concat('LONG', ':', Tag)); (=<(Len, 10), string_concat(Lower, '?', Mid), string_concat('MID', ':', Tag)))); (=<(Len, 8), string_concat(Lower, '!', Mid), string_concat('SHORT', ':', Tag)))); (=<(Len, 8), string_concat(Lower, '.', Mid), string_concat('TINY', ':', Tag))), string_concat(Mid, Tag, Out))),
    assertz(type_declarations:uw_type(two_level_nested_multi_result_choice_clause/2, 1, atom)),
    assertz(type_declarations:uw_type(two_level_nested_multi_result_choice_clause/2, 2, atom)),
    assertz(type_declarations:uw_return_type(two_level_nested_multi_result_choice_clause/2, atom)),
    once(compile_predicate_to_typr(two_level_nested_multi_result_choice_clause/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let two_level_nested_multi_result_choice_clause <- fn(arg1: char, arg2: char): char")),
    once(sub_string(Code, _, _, _, "else if")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 8 }@)")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ is.character(v3) }@)")),
    once(sub_string(Code, _, _, _, "let v5 <- if (@{ v4 > 10 }@)")),
    once(sub_string(Code, _, _, _, "@{ list(v5, v6) }@")),
    once(sub_string(Code, _, _, _, "let v6 <- @{ .subset2(v5, 1) }@;")),
    once(sub_string(Code, _, _, _, "let v7 <- @{ .subset2(v5, 2) }@;")),
    once(sub_string(Code, _, _, _, "let arg2 <- @{ paste0(v6, v7) }@;")),
    \+ sub_string(Code, _, _, _, "local({"),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(guard_only_predicates_infer_boolean_return_type) :-
    clear_type_declarations,
    assertz(user:(numeric_input(Value) :- is_numeric(Value))),
    once(compile_predicate_to_typr(numeric_input/1, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "let numeric_input <- fn(arg1): bool")),
    once(sub_string(Code, _, _, _, "is.numeric(arg1)")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(dataframe_sort_predicates_lower_natively) :-
    clear_type_declarations,
    assertz(user:(sort_rows(Input, Output) :- sort_by(Input, name, Output))),
    once(compile_predicate_to_typr(sort_rows/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "order(arg1[[\"name\"]])")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(dataframe_filter_predicates_lower_natively) :-
    clear_type_declarations,
    assertz(user:(filter_rows(Input, Output) :- filter(Input, score > 10, Output))),
    once(compile_predicate_to_typr(filter_rows/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "subset(arg1, (score > 10))")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(dataframe_group_predicates_lower_natively) :-
    clear_type_declarations,
    assertz(user:(group_rows(Input, Output) :- group_by(Input, category, Output))),
    once(compile_predicate_to_typr(group_rows/2, [typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "aggregate(. ~ \"category\", data=arg1, FUN=list)")),
    \+ sub_string(Code, _, _, _, "(function("),
    generated_typr_is_valid(Code, exit(0)).

test(transitive_closure_template_is_valid_typr) :-
    clear_type_declarations,
    assertz(type_declarations:uw_type(edge/2, 1, atom)),
    assertz(type_declarations:uw_type(edge/2, 2, atom)),
    once(compile_predicate_to_typr(tc/2, [base_pred(edge), typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "while (length(queue) > 0)")),
    generated_typr_is_valid(Code, exit(0)).

test(transitive_closure_seeds_known_base_facts) :-
    clear_type_declarations,
    assertz(user:edge(a, b)),
    assertz(user:edge(b, c)),
    assertz(type_declarations:uw_type(edge/2, 1, atom)),
    assertz(type_declarations:uw_type(edge/2, 2, atom)),
    once(compile_predicate_to_typr(tc/2, [base_pred(edge), typed_mode(explicit)], Code)),
    once(sub_string(Code, _, _, _, "_seed_edge_1 <- add_edge(\"a\", \"b\");")),
    once(sub_string(Code, _, _, _, "_seed_edge_2 <- add_edge(\"b\", \"c\");")).

:- end_tests(typr_target).
