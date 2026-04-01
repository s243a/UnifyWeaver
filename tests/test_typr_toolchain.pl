:- module(test_typr_toolchain, [run_all_tests/0]).

:- use_module(library(filesex)).
:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/type_declarations').
:- use_module('../src/unifyweaver/targets/typr_target').
:- use_module('../src/unifyweaver/core/recursive_compiler').

run_all_tests :-
    run_tests([typr_toolchain]).

setup_typr_mixed_tree_numeric_helper_toolchain_state :-
    cleanup_typr_mixed_tree_numeric_helper_toolchain_state.

cleanup_typr_mixed_tree_numeric_helper_toolchain_state :-
    clear_type_declarations,
    retractall(user:typr_tree_num_ok_helper(_)),
    retractall(user:typr_num_tree_ok_helper(_)),
    retractall(user:typr_tree_num_sum_helper(_, _)),
    retractall(user:typr_num_tree_sum_helper(_, _)),
    retractall(user:typr_tree_num_weight_helper(_, _, _)),
    retractall(user:typr_num_tree_weight_helper(_, _, _)),
    retractall(user:typr_pick_tree_helper(_, _, _, _)).

:- begin_tests(typr_toolchain).

test(generated_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:simple_fact(hello)),
    assertz(type_declarations:uw_type(simple_fact/1, 1, atom)),
    once(compile_predicate_to_typr(simple_fact/1, [typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:simple_fact(_)).

test(alias_assignments_after_native_outputs_check_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:(alias_after_output(Name, Out) :- string_lower(Name, Lower), Out = Lower)),
    assertz(type_declarations:uw_type(alias_after_output/2, 1, atom)),
    assertz(type_declarations:uw_type(alias_after_output/2, 2, atom)),
    once(compile_predicate_to_typr(alias_after_output/2, [typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:alias_after_output(_, _)).

test(arithmetic_assignments_after_native_outputs_check_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:(arith_after_output(Name, Out) :- string_length(Name, Len), Out is Len + 1)),
    assertz(type_declarations:uw_type(arith_after_output/2, 1, atom)),
    assertz(type_declarations:uw_type(arith_after_output/2, 2, integer)),
    once(compile_predicate_to_typr(arith_after_output/2, [typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:arith_after_output(_, _)).

test(cat_command_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:(say_cat(Msg) :- cat(Msg))),
    assertz(type_declarations:uw_type(say_cat/1, 1, atom)),
    once(compile_predicate_to_typr(say_cat/1, [typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:say_cat(_)).

test(print_command_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:(say_print(Msg) :- print(Msg))),
    assertz(type_declarations:uw_type(say_print/1, 1, atom)),
    once(compile_predicate_to_typr(say_print/1, [typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:say_print(_)).

test(transitive_closure_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:edge(a, b)),
    assertz(user:edge(b, c)),
    assertz(type_declarations:uw_type(edge/2, 1, atom)),
    assertz(type_declarations:uw_type(edge/2, 2, atom)),
    once(compile_predicate_to_typr(tc/2, [base_pred(edge), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:edge(_, _)).

test(transitive_closure_runtime_vector_api_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(type_declarations:uw_type(edge/2, 1, atom)),
    assertz(type_declarations:uw_type(edge/2, 2, atom)),
    once(compile_predicate_to_typr(tc/2, [base_pred(edge), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ).

test(transitive_closure_stdin_input_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(type_declarations:uw_type(edge/2, 1, atom)),
    assertz(type_declarations:uw_type(edge/2, 2, atom)),
    once(compile_predicate_to_typr(tc/2, [base_pred(edge), typed_mode(explicit), input(stdin)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ).

test(transitive_closure_file_input_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(type_declarations:uw_type(edge/2, 1, atom)),
    assertz(type_declarations:uw_type(edge/2, 2, atom)),
    once(compile_predicate_to_typr(tc/2, [base_pred(edge), typed_mode(explicit), input(file("data.txt"))], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ).

test(transitive_closure_vfs_input_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(type_declarations:uw_type(edge/2, 1, atom)),
    assertz(type_declarations:uw_type(edge/2, 2, atom)),
    once(compile_predicate_to_typr(tc/2, [base_pred(edge), typed_mode(explicit), input(vfs("family_tree"))], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check'])
        ),
        delete_directory_and_contents(ProjectDir)
    ).

test(transitive_closure_function_input_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(type_declarations:uw_type(edge/2, 1, atom)),
    assertz(type_declarations:uw_type(edge/2, 2, atom)),
    once(compile_predicate_to_typr(tc/2, [base_pred(edge), typed_mode(explicit), input(function)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ).

test(transitive_closure_integer_runtime_loader_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(type_declarations:uw_type(edge/2, 1, integer)),
    assertz(type_declarations:uw_type(edge/2, 2, integer)),
    once(compile_predicate_to_typr(tc/2, [base_pred(edge), typed_mode(explicit), input(stdin)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ).

test(transitive_closure_numeric_runtime_loader_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(type_declarations:uw_type(edge/2, 1, number)),
    assertz(type_declarations:uw_type(edge/2, 2, number)),
    once(compile_predicate_to_typr(tc/2, [base_pred(edge), typed_mode(explicit), input(vfs("family_tree"))], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check'])
        ),
        delete_directory_and_contents(ProjectDir)
    ).

test(transitive_closure_pair_integer_runtime_loader_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(type_declarations:uw_type(edge/2, 1, pair(integer, integer))),
    assertz(type_declarations:uw_type(edge/2, 2, pair(integer, integer))),
    once(compile_predicate_to_typr(tc/2, [base_pred(edge), typed_mode(explicit), input(stdin)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check'])
        ),
        delete_directory_and_contents(ProjectDir)
    ).

test(transitive_closure_pair_numeric_runtime_loader_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(type_declarations:uw_type(edge/2, 1, pair(number, number))),
    assertz(type_declarations:uw_type(edge/2, 2, pair(number, number))),
    once(compile_predicate_to_typr(tc/2, [base_pred(edge), typed_mode(explicit), input(vfs("family_tree"))], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check'])
        ),
        delete_directory_and_contents(ProjectDir)
    ).

test(tail_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
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
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:factorial_acc(_, _, _)).

test(linear_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
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
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:factorial_linear(_, _)).

test(list_linear_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
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
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:list_length(_, _)).

test(tree_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:fib(0, 0)),
    assertz(user:fib(1, 1)),
    assertz(user:(fib(N, F) :-
        N > 1,
        N1 is N - 1,
        N2 is N - 2,
        fib(N1, F1),
        fib(N2, F2),
        F is F1 + F2
    )),
    assertz(type_declarations:uw_type(fib/2, 1, integer)),
    assertz(type_declarations:uw_type(fib/2, 2, integer)),
    assertz(type_declarations:uw_return_type(fib/2, integer)),
    once(recursive_compiler:compile_recursive(fib/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:fib(_, _)).

test(mutual_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_even(0)),
    assertz(user:(typr_mutual_even(N) :-
        N > 0,
        N1 is N - 1,
        typr_mutual_odd(N1)
    )),
    assertz(user:typr_mutual_odd(1)),
    assertz(user:(typr_mutual_odd(N) :-
        N > 1,
        N1 is N - 1,
        typr_mutual_even(N1)
    )),
    assertz(type_declarations:uw_type(typr_mutual_even/1, 1, integer)),
    assertz(type_declarations:uw_type(typr_mutual_odd/1, 1, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_even/1, bool)),
    assertz(type_declarations:uw_return_type(typr_mutual_odd/1, bool)),
    once(recursive_compiler:compile_recursive(typr_mutual_even/1, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_even(_)),
    retractall(user:typr_mutual_odd(_)).

test(structural_mutual_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_even_list([])),
    assertz(user:(typr_mutual_even_list([_|T]) :-
        typr_mutual_odd_list(T)
    )),
    assertz(user:typr_mutual_odd_list([_])),
    assertz(user:(typr_mutual_odd_list([_|T]) :-
        typr_mutual_even_list(T)
    )),
    assertz(type_declarations:uw_type(typr_mutual_even_list/1, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_odd_list/1, 1, list(any))),
    assertz(type_declarations:uw_return_type(typr_mutual_even_list/1, bool)),
    assertz(type_declarations:uw_return_type(typr_mutual_odd_list/1, bool)),
    once(recursive_compiler:compile_recursive(typr_mutual_even_list/1, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_even_list(_)),
    retractall(user:typr_mutual_odd_list(_)).

test(structural_tree_mutual_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_even_left_tree([])),
    assertz(user:(typr_mutual_even_left_tree([_, L, _]) :-
        typr_mutual_odd_left_tree(L)
    )),
    assertz(user:typr_mutual_odd_left_tree([_, [], []])),
    assertz(user:(typr_mutual_odd_left_tree([_, L, _]) :-
        typr_mutual_even_left_tree(L)
    )),
    assertz(type_declarations:uw_type(typr_mutual_even_left_tree/1, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_odd_left_tree/1, 1, list(any))),
    assertz(type_declarations:uw_return_type(typr_mutual_even_left_tree/1, bool)),
    assertz(type_declarations:uw_return_type(typr_mutual_odd_left_tree/1, bool)),
    once(recursive_compiler:compile_recursive(typr_mutual_even_left_tree/1, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_even_left_tree(_)),
    retractall(user:typr_mutual_odd_left_tree(_)).

test(structural_tree_dual_mutual_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_even_tree([])),
    assertz(user:(typr_mutual_even_tree([_, L, R]) :-
        typr_mutual_odd_tree(L),
        typr_mutual_odd_tree(R)
    )),
    assertz(user:typr_mutual_odd_tree([_, [], []])),
    assertz(user:(typr_mutual_odd_tree([_, L, R]) :-
        typr_mutual_even_tree(L),
        typr_mutual_even_tree(R)
    )),
    assertz(type_declarations:uw_type(typr_mutual_even_tree/1, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree/1, 1, list(any))),
    assertz(type_declarations:uw_return_type(typr_mutual_even_tree/1, bool)),
    assertz(type_declarations:uw_return_type(typr_mutual_odd_tree/1, bool)),
    once(recursive_compiler:compile_recursive(typr_mutual_even_tree/1, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_even_tree(_)),
    retractall(user:typr_mutual_odd_tree(_)).

test(structural_tree_dual_mutual_prework_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_even_tree_pre([])),
    assertz(user:(typr_mutual_even_tree_pre([_, L, R]) :-
        Left = L,
        Right = R,
        typr_mutual_odd_tree_pre(Left),
        typr_mutual_odd_tree_pre(Right)
    )),
    assertz(user:typr_mutual_odd_tree_pre([_, [], []])),
    assertz(user:(typr_mutual_odd_tree_pre([_, L, R]) :-
        Left = L,
        Right = R,
        typr_mutual_even_tree_pre(Left),
        typr_mutual_even_tree_pre(Right)
    )),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_pre/1, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_pre/1, 1, list(any))),
    assertz(type_declarations:uw_return_type(typr_mutual_even_tree_pre/1, bool)),
    assertz(type_declarations:uw_return_type(typr_mutual_odd_tree_pre/1, bool)),
    once(recursive_compiler:compile_recursive(typr_mutual_even_tree_pre/1, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_even_tree_pre(_)),
    retractall(user:typr_mutual_odd_tree_pre(_)).

test(structural_tree_dual_mutual_branch_prework_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_even_tree_branch_pre([])),
    assertz(user:(typr_mutual_even_tree_branch_pre([V, L, R]) :-
        ( V > 0 ->
            Left = L,
            Right = R
        ;   Left = R,
            Right = L
        ),
        typr_mutual_odd_tree_branch_pre(Left),
        typr_mutual_odd_tree_branch_pre(Right)
    )),
    assertz(user:typr_mutual_odd_tree_branch_pre([_, [], []])),
    assertz(user:(typr_mutual_odd_tree_branch_pre([V, L, R]) :-
        ( V > 0 ->
            Left = L,
            Right = R
        ;   Left = R,
            Right = L
        ),
        typr_mutual_even_tree_branch_pre(Left),
        typr_mutual_even_tree_branch_pre(Right)
    )),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_branch_pre/1, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_branch_pre/1, 1, list(any))),
    assertz(type_declarations:uw_return_type(typr_mutual_even_tree_branch_pre/1, bool)),
    assertz(type_declarations:uw_return_type(typr_mutual_odd_tree_branch_pre/1, bool)),
    once(recursive_compiler:compile_recursive(typr_mutual_even_tree_branch_pre/1, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_even_tree_branch_pre(_)),
    retractall(user:typr_mutual_odd_tree_branch_pre(_)).

test(structural_tree_dual_mutual_recursive_branch_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_even_tree_recursive_branch([])),
    assertz(user:(typr_mutual_even_tree_recursive_branch([V, L, R]) :-
        ( V > 0 ->
            typr_mutual_odd_tree_recursive_branch(L),
            typr_mutual_odd_tree_recursive_branch(R)
        ;   typr_mutual_odd_tree_recursive_branch(R),
            typr_mutual_odd_tree_recursive_branch(L)
        )
    )),
    assertz(user:typr_mutual_odd_tree_recursive_branch([_, [], []])),
    assertz(user:(typr_mutual_odd_tree_recursive_branch([V, L, R]) :-
        ( V > 0 ->
            typr_mutual_even_tree_recursive_branch(L),
            typr_mutual_even_tree_recursive_branch(R)
        ;   typr_mutual_even_tree_recursive_branch(R),
            typr_mutual_even_tree_recursive_branch(L)
        )
    )),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_recursive_branch/1, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_recursive_branch/1, 1, list(any))),
    assertz(type_declarations:uw_return_type(typr_mutual_even_tree_recursive_branch/1, bool)),
    assertz(type_declarations:uw_return_type(typr_mutual_odd_tree_recursive_branch/1, bool)),
    once(recursive_compiler:compile_recursive(typr_mutual_even_tree_recursive_branch/1, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_even_tree_recursive_branch(_)),
    retractall(user:typr_mutual_odd_tree_recursive_branch(_)).

test(structural_tree_dual_mutual_nested_branch_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_even_tree_nested_branch([])),
    assertz(user:(typr_mutual_even_tree_nested_branch([V, L, R]) :-
        ( V > 0 ->
            ( V > 10 ->
                typr_mutual_odd_tree_nested_branch(L),
                typr_mutual_odd_tree_nested_branch(R)
            ;   typr_mutual_odd_tree_nested_branch(R),
                typr_mutual_odd_tree_nested_branch(L)
            )
        ;   typr_mutual_odd_tree_nested_branch(L),
            typr_mutual_odd_tree_nested_branch(R)
        )
    )),
    assertz(user:typr_mutual_odd_tree_nested_branch([_, [], []])),
    assertz(user:(typr_mutual_odd_tree_nested_branch([V, L, R]) :-
        ( V > 0 ->
            ( V > 10 ->
                typr_mutual_even_tree_nested_branch(L),
                typr_mutual_even_tree_nested_branch(R)
            ;   typr_mutual_even_tree_nested_branch(R),
                typr_mutual_even_tree_nested_branch(L)
            )
        ;   typr_mutual_even_tree_nested_branch(L),
            typr_mutual_even_tree_nested_branch(R)
        )
    )),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_nested_branch/1, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_nested_branch/1, 1, list(any))),
    assertz(type_declarations:uw_return_type(typr_mutual_even_tree_nested_branch/1, bool)),
    assertz(type_declarations:uw_return_type(typr_mutual_odd_tree_nested_branch/1, bool)),
    once(recursive_compiler:compile_recursive(typr_mutual_even_tree_nested_branch/1, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_even_tree_nested_branch(_)),
    retractall(user:typr_mutual_odd_tree_nested_branch(_)).

test(structural_tree_dual_mutual_nested_branch_prework_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_even_tree_nested_branch_pre([])),
    assertz(user:(typr_mutual_even_tree_nested_branch_pre([V, L, R]) :-
        ( V > 0 ->
            V >= 0,
            ( V > 10 ->
                typr_mutual_odd_tree_nested_branch_pre(L),
                typr_mutual_odd_tree_nested_branch_pre(R)
            ;   typr_mutual_odd_tree_nested_branch_pre(R),
                typr_mutual_odd_tree_nested_branch_pre(L)
            )
        ;   typr_mutual_odd_tree_nested_branch_pre(L),
            typr_mutual_odd_tree_nested_branch_pre(R)
        )
    )),
    assertz(user:typr_mutual_odd_tree_nested_branch_pre([_, [], []])),
    assertz(user:(typr_mutual_odd_tree_nested_branch_pre([V, L, R]) :-
        ( V > 0 ->
            V >= 0,
            ( V > 10 ->
                typr_mutual_even_tree_nested_branch_pre(L),
                typr_mutual_even_tree_nested_branch_pre(R)
            ;   typr_mutual_even_tree_nested_branch_pre(R),
                typr_mutual_even_tree_nested_branch_pre(L)
            )
        ;   typr_mutual_even_tree_nested_branch_pre(L),
            typr_mutual_even_tree_nested_branch_pre(R)
        )
    )),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_nested_branch_pre/1, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_nested_branch_pre/1, 1, list(any))),
    assertz(type_declarations:uw_return_type(typr_mutual_even_tree_nested_branch_pre/1, bool)),
    assertz(type_declarations:uw_return_type(typr_mutual_odd_tree_nested_branch_pre/1, bool)),
    once(recursive_compiler:compile_recursive(typr_mutual_even_tree_nested_branch_pre/1, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_even_tree_nested_branch_pre(_)),
    retractall(user:typr_mutual_odd_tree_nested_branch_pre(_)).

test(structural_tree_dual_mutual_intercall_guard_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_even_tree_call_guard([])),
    assertz(user:(typr_mutual_even_tree_call_guard([V, L, R]) :-
        ( V > 0 ->
            typr_mutual_odd_tree_call_guard(L),
            V >= 0,
            typr_mutual_odd_tree_call_guard(R)
        ;   typr_mutual_odd_tree_call_guard(R),
            typr_mutual_odd_tree_call_guard(L)
        )
    )),
    assertz(user:typr_mutual_odd_tree_call_guard([_, [], []])),
    assertz(user:(typr_mutual_odd_tree_call_guard([V, L, R]) :-
        ( V > 0 ->
            typr_mutual_even_tree_call_guard(L),
            V >= 0,
            typr_mutual_even_tree_call_guard(R)
        ;   typr_mutual_even_tree_call_guard(R),
            typr_mutual_even_tree_call_guard(L)
        )
    )),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_call_guard/1, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_call_guard/1, 1, list(any))),
    assertz(type_declarations:uw_return_type(typr_mutual_even_tree_call_guard/1, bool)),
    assertz(type_declarations:uw_return_type(typr_mutual_odd_tree_call_guard/1, bool)),
    once(recursive_compiler:compile_recursive(typr_mutual_even_tree_call_guard/1, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_even_tree_call_guard(_)),
    retractall(user:typr_mutual_odd_tree_call_guard(_)).

test(structural_tree_dual_mutual_intercall_alias_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_even_tree_call_alias([])),
    assertz(user:(typr_mutual_even_tree_call_alias([V, L, R]) :-
        ( V > 0 ->
            typr_mutual_odd_tree_call_alias(L),
            Right = R,
            typr_mutual_odd_tree_call_alias(Right)
        ;   typr_mutual_odd_tree_call_alias(R),
            typr_mutual_odd_tree_call_alias(L)
        )
    )),
    assertz(user:typr_mutual_odd_tree_call_alias([_, [], []])),
    assertz(user:(typr_mutual_odd_tree_call_alias([V, L, R]) :-
        ( V > 0 ->
            typr_mutual_even_tree_call_alias(L),
            Right = R,
            typr_mutual_even_tree_call_alias(Right)
        ;   typr_mutual_even_tree_call_alias(R),
            typr_mutual_even_tree_call_alias(L)
        )
    )),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_call_alias/1, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_call_alias/1, 1, list(any))),
    assertz(type_declarations:uw_return_type(typr_mutual_even_tree_call_alias/1, bool)),
    assertz(type_declarations:uw_return_type(typr_mutual_odd_tree_call_alias/1, bool)),
    once(recursive_compiler:compile_recursive(typr_mutual_even_tree_call_alias/1, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_even_tree_call_alias(_)),
    retractall(user:typr_mutual_odd_tree_call_alias(_)).

test(structural_tree_dual_mutual_nested_intercall_guard_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_even_tree_nested_call_guard([])),
    assertz(user:(typr_mutual_even_tree_nested_call_guard([V, L, R]) :-
        ( V > 0 ->
            typr_mutual_odd_tree_nested_call_guard(L),
            ( V > 10 ->
                V >= 0,
                typr_mutual_odd_tree_nested_call_guard(R)
            ;   typr_mutual_odd_tree_nested_call_guard(R)
            )
        ;   typr_mutual_odd_tree_nested_call_guard(R),
            typr_mutual_odd_tree_nested_call_guard(L)
        )
    )),
    assertz(user:typr_mutual_odd_tree_nested_call_guard([_, [], []])),
    assertz(user:(typr_mutual_odd_tree_nested_call_guard([V, L, R]) :-
        ( V > 0 ->
            typr_mutual_even_tree_nested_call_guard(L),
            ( V > 10 ->
                V >= 0,
                typr_mutual_even_tree_nested_call_guard(R)
            ;   typr_mutual_even_tree_nested_call_guard(R)
            )
        ;   typr_mutual_even_tree_nested_call_guard(R),
            typr_mutual_even_tree_nested_call_guard(L)
        )
    )),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_nested_call_guard/1, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_nested_call_guard/1, 1, list(any))),
    assertz(type_declarations:uw_return_type(typr_mutual_even_tree_nested_call_guard/1, bool)),
    assertz(type_declarations:uw_return_type(typr_mutual_odd_tree_nested_call_guard/1, bool)),
    once(recursive_compiler:compile_recursive(typr_mutual_even_tree_nested_call_guard/1, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_even_tree_nested_call_guard(_)),
    retractall(user:typr_mutual_odd_tree_nested_call_guard(_)).

test(structural_tree_dual_mutual_nested_intercall_alias_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_even_tree_nested_call_alias([])),
    assertz(user:(typr_mutual_even_tree_nested_call_alias([V, L, R]) :-
        ( V > 0 ->
            typr_mutual_odd_tree_nested_call_alias(L),
            ( V > 10 ->
                Right = R,
                typr_mutual_odd_tree_nested_call_alias(Right)
            ;   typr_mutual_odd_tree_nested_call_alias(R)
            )
        ;   typr_mutual_odd_tree_nested_call_alias(R),
            typr_mutual_odd_tree_nested_call_alias(L)
        )
    )),
    assertz(user:typr_mutual_odd_tree_nested_call_alias([_, [], []])),
    assertz(user:(typr_mutual_odd_tree_nested_call_alias([V, L, R]) :-
        ( V > 0 ->
            typr_mutual_even_tree_nested_call_alias(L),
            ( V > 10 ->
                Right = R,
                typr_mutual_even_tree_nested_call_alias(Right)
            ;   typr_mutual_even_tree_nested_call_alias(R)
            )
        ;   typr_mutual_even_tree_nested_call_alias(R),
            typr_mutual_even_tree_nested_call_alias(L)
        )
    )),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_nested_call_alias/1, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_nested_call_alias/1, 1, list(any))),
    assertz(type_declarations:uw_return_type(typr_mutual_even_tree_nested_call_alias/1, bool)),
    assertz(type_declarations:uw_return_type(typr_mutual_odd_tree_nested_call_alias/1, bool)),
    once(recursive_compiler:compile_recursive(typr_mutual_even_tree_nested_call_alias/1, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_even_tree_nested_call_alias(_)),
    retractall(user:typr_mutual_odd_tree_nested_call_alias(_)).

test(structural_tree_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:tree_sum([], 0)),
    assertz(user:(tree_sum([V, L, R], Sum) :-
        tree_sum(L, LS),
        tree_sum(R, RS),
        Sum is V + LS + RS
    )),
    assertz(type_declarations:uw_type(tree_sum/2, 1, list(any))),
    assertz(type_declarations:uw_type(tree_sum/2, 2, integer)),
    assertz(type_declarations:uw_return_type(tree_sum/2, integer)),
    once(recursive_compiler:compile_recursive(tree_sum/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:tree_sum(_, _)).

test(structural_tree_height_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:tree_height([], 0)),
    assertz(user:(tree_height([_V, L, R], H) :-
        tree_height(L, HL),
        tree_height(R, HR),
        H is 1 + max(HL, HR)
    )),
    assertz(type_declarations:uw_type(tree_height/2, 1, list(any))),
    assertz(type_declarations:uw_type(tree_height/2, 2, integer)),
    assertz(type_declarations:uw_return_type(tree_height/2, integer)),
    once(recursive_compiler:compile_recursive(tree_height/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:tree_height(_, _)).

test(structural_tree_prework_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:tree_sum_prework([], 0)),
    assertz(user:(tree_sum_prework([V, L, R], Sum) :-
        V >= 0,
        W is V + 1,
        tree_sum_prework(L, LS),
        tree_sum_prework(R, RS),
        Sum is W + LS + RS
    )),
    assertz(type_declarations:uw_type(tree_sum_prework/2, 1, list(any))),
    assertz(type_declarations:uw_type(tree_sum_prework/2, 2, integer)),
    assertz(type_declarations:uw_return_type(tree_sum_prework/2, integer)),
    once(recursive_compiler:compile_recursive(tree_sum_prework/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:tree_sum_prework(_, _)).

test(structural_tree_branching_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:tree_sum_branch([], 0)),
    assertz(user:(tree_sum_branch([V, L, R], Sum) :-
        ( V > 0 ->
            Base is V + 1,
            W is Base + 1
        ;   Base is V + 2,
            W is Base + 2
        ),
        tree_sum_branch(L, LS),
        tree_sum_branch(R, RS),
        Sum is W + LS + RS
    )),
    assertz(type_declarations:uw_type(tree_sum_branch/2, 1, list(any))),
    assertz(type_declarations:uw_type(tree_sum_branch/2, 2, integer)),
    assertz(type_declarations:uw_return_type(tree_sum_branch/2, integer)),
    once(recursive_compiler:compile_recursive(tree_sum_branch/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:tree_sum_branch(_, _)).

test(structural_tree_asymmetric_branching_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:tree_sum_asym_branch([], 0)),
    assertz(user:(tree_sum_asym_branch([V, L, R], Sum) :-
        ( V > 0 -> A is V + 1 ; B is V + 2 ),
        tree_sum_asym_branch(L, LS),
        tree_sum_asym_branch(R, RS),
        ( V > 0 -> Sum is A + LS + RS ; Sum is B + LS + RS )
    )),
    assertz(type_declarations:uw_type(tree_sum_asym_branch/2, 1, list(any))),
    assertz(type_declarations:uw_type(tree_sum_asym_branch/2, 2, integer)),
    assertz(type_declarations:uw_return_type(tree_sum_asym_branch/2, integer)),
    once(recursive_compiler:compile_recursive(tree_sum_asym_branch/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:tree_sum_asym_branch(_, _)).

test(nary_structural_tree_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:weighted_tree_sum([], _Scale, 0)),
    assertz(user:(weighted_tree_sum([V, L, R], Scale, Sum) :-
        weighted_tree_sum(L, Scale, LS),
        weighted_tree_sum(R, Scale, RS),
        Sum is (V * Scale) + LS + RS
    )),
    assertz(type_declarations:uw_type(weighted_tree_sum/3, 1, list(any))),
    assertz(type_declarations:uw_type(weighted_tree_sum/3, 2, integer)),
    assertz(type_declarations:uw_type(weighted_tree_sum/3, 3, integer)),
    assertz(type_declarations:uw_return_type(weighted_tree_sum/3, integer)),
    once(recursive_compiler:compile_recursive(weighted_tree_sum/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:weighted_tree_sum(_, _, _)).

test(nary_structural_tree_invariant_step_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:weighted_tree_sum_scale_step([], _Scale, 0)),
    assertz(user:(weighted_tree_sum_scale_step([V, L, R], Scale, Sum) :-
        Scale1 is Scale + 1,
        weighted_tree_sum_scale_step(L, Scale1, LS),
        weighted_tree_sum_scale_step(R, Scale1, RS),
        Sum is (V * Scale) + LS + RS
    )),
    assertz(type_declarations:uw_type(weighted_tree_sum_scale_step/3, 1, list(any))),
    assertz(type_declarations:uw_type(weighted_tree_sum_scale_step/3, 2, integer)),
    assertz(type_declarations:uw_type(weighted_tree_sum_scale_step/3, 3, integer)),
    assertz(type_declarations:uw_return_type(weighted_tree_sum_scale_step/3, integer)),
    once(recursive_compiler:compile_recursive(weighted_tree_sum_scale_step/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:weighted_tree_sum_scale_step(_, _, _)).

test(nary_structural_tree_invariant_branch_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:weighted_tree_sum_scale_branch([], _Scale, 0)),
    assertz(user:(weighted_tree_sum_scale_branch([V, L, R], Scale, Sum) :-
        ( Scale > 1 -> Scale1 is Scale + 1 ; Scale1 is Scale + 2 ),
        weighted_tree_sum_scale_branch(L, Scale1, LS),
        weighted_tree_sum_scale_branch(R, Scale1, RS),
        Sum is (V * Scale1) + LS + RS
    )),
    assertz(type_declarations:uw_type(weighted_tree_sum_scale_branch/3, 1, list(any))),
    assertz(type_declarations:uw_type(weighted_tree_sum_scale_branch/3, 2, integer)),
    assertz(type_declarations:uw_type(weighted_tree_sum_scale_branch/3, 3, integer)),
    assertz(type_declarations:uw_return_type(weighted_tree_sum_scale_branch/3, integer)),
    once(recursive_compiler:compile_recursive(weighted_tree_sum_scale_branch/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:weighted_tree_sum_scale_branch(_, _, _)).

test(structural_tree_branch_local_calls_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:tree_sum_branch_calls([], 0)),
    assertz(user:(tree_sum_branch_calls([V, L, R], Sum) :-
        ( V > 0 -> LeftTree = L, RightTree = R ; LeftTree = R, RightTree = L ),
        tree_sum_branch_calls(LeftTree, LS),
        tree_sum_branch_calls(RightTree, RS),
        Sum is V + LS + RS
    )),
    assertz(type_declarations:uw_type(tree_sum_branch_calls/2, 1, list(any))),
    assertz(type_declarations:uw_type(tree_sum_branch_calls/2, 2, integer)),
    assertz(type_declarations:uw_return_type(tree_sum_branch_calls/2, integer)),
    once(recursive_compiler:compile_recursive(tree_sum_branch_calls/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:tree_sum_branch_calls(_, _)).

test(weighted_structural_tree_branch_local_calls_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:weighted_tree_branch_calls([], _Scale, 0)),
    assertz(user:(weighted_tree_branch_calls([V, L, R], Scale, Sum) :-
        ( Scale > 1 -> LeftTree = L, RightTree = R ; LeftTree = R, RightTree = L ),
        Scale1 is Scale + 1,
        weighted_tree_branch_calls(LeftTree, Scale1, LS),
        weighted_tree_branch_calls(RightTree, Scale, RS),
        Sum is (V * Scale) + LS + RS
    )),
    assertz(type_declarations:uw_type(weighted_tree_branch_calls/3, 1, list(any))),
    assertz(type_declarations:uw_type(weighted_tree_branch_calls/3, 2, integer)),
    assertz(type_declarations:uw_type(weighted_tree_branch_calls/3, 3, integer)),
    assertz(type_declarations:uw_return_type(weighted_tree_branch_calls/3, integer)),
    once(recursive_compiler:compile_recursive(weighted_tree_branch_calls/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:weighted_tree_branch_calls(_, _, _)).

test(structural_tree_recursive_branch_body_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:tree_sum_recursive_branch([], 0)),
    assertz(user:(tree_sum_recursive_branch([V, L, R], Sum) :-
        ( V > 0 ->
            tree_sum_recursive_branch(L, LS),
            tree_sum_recursive_branch(R, RS),
            Sum is V + LS + RS
        ;   tree_sum_recursive_branch(R, RS),
            tree_sum_recursive_branch(L, LS),
            Sum is V + LS + RS
        )
    )),
    assertz(type_declarations:uw_type(tree_sum_recursive_branch/2, 1, list(any))),
    assertz(type_declarations:uw_type(tree_sum_recursive_branch/2, 2, integer)),
    assertz(type_declarations:uw_return_type(tree_sum_recursive_branch/2, integer)),
    once(recursive_compiler:compile_recursive(tree_sum_recursive_branch/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:tree_sum_recursive_branch(_, _)).

test(weighted_structural_tree_recursive_branch_body_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:weighted_tree_recursive_branch([], _Scale, 0)),
    assertz(user:(weighted_tree_recursive_branch([V, L, R], Scale, Sum) :-
        ( Scale > 1 ->
            weighted_tree_recursive_branch(L, Scale, LS),
            weighted_tree_recursive_branch(R, Scale, RS),
            Sum is (V * Scale) + LS + RS
        ;   weighted_tree_recursive_branch(R, Scale, RS),
            weighted_tree_recursive_branch(L, Scale, LS),
            Sum is (V * Scale) + LS + RS
        )
    )),
    assertz(type_declarations:uw_type(weighted_tree_recursive_branch/3, 1, list(any))),
    assertz(type_declarations:uw_type(weighted_tree_recursive_branch/3, 2, integer)),
    assertz(type_declarations:uw_type(weighted_tree_recursive_branch/3, 3, integer)),
    assertz(type_declarations:uw_return_type(weighted_tree_recursive_branch/3, integer)),
    once(recursive_compiler:compile_recursive(weighted_tree_recursive_branch/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:weighted_tree_recursive_branch(_, _, _)).

test(nested_structural_tree_recursive_branch_body_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:tree_sum_nested_recursive_branch([], 0)),
    assertz(user:(tree_sum_nested_recursive_branch([V, L, R], Sum) :-
        ( V > 0 ->
            ( V > 1 ->
                tree_sum_nested_recursive_branch(L, LS),
                tree_sum_nested_recursive_branch(R, RS)
            ;   tree_sum_nested_recursive_branch(R, RS),
                tree_sum_nested_recursive_branch(L, LS)
            ),
            Sum is V + LS + RS
        ;   tree_sum_nested_recursive_branch(L, LS),
            tree_sum_nested_recursive_branch(R, RS),
            Sum is V + LS + RS
        )
    )),
    assertz(type_declarations:uw_type(tree_sum_nested_recursive_branch/2, 1, list(any))),
    assertz(type_declarations:uw_type(tree_sum_nested_recursive_branch/2, 2, integer)),
    assertz(type_declarations:uw_return_type(tree_sum_nested_recursive_branch/2, integer)),
    once(recursive_compiler:compile_recursive(tree_sum_nested_recursive_branch/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:tree_sum_nested_recursive_branch(_, _)).

test(weighted_nested_structural_tree_recursive_branch_body_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:weighted_tree_nested_recursive_branch([], _Scale, 0)),
    assertz(user:(weighted_tree_nested_recursive_branch([V, L, R], Scale, Sum) :-
        ( Scale > 1 ->
            ( V > 0 ->
                weighted_tree_nested_recursive_branch(L, Scale, LS),
                weighted_tree_nested_recursive_branch(R, Scale, RS)
            ;   weighted_tree_nested_recursive_branch(R, Scale, RS),
                weighted_tree_nested_recursive_branch(L, Scale, LS)
            ),
            Sum is (V * Scale) + LS + RS
        ;   weighted_tree_nested_recursive_branch(L, Scale, LS),
            weighted_tree_nested_recursive_branch(R, Scale, RS),
            Sum is (V * Scale) + LS + RS
        )
    )),
    assertz(type_declarations:uw_type(weighted_tree_nested_recursive_branch/3, 1, list(any))),
    assertz(type_declarations:uw_type(weighted_tree_nested_recursive_branch/3, 2, integer)),
    assertz(type_declarations:uw_type(weighted_tree_nested_recursive_branch/3, 3, integer)),
    assertz(type_declarations:uw_return_type(weighted_tree_nested_recursive_branch/3, integer)),
    once(recursive_compiler:compile_recursive(weighted_tree_nested_recursive_branch/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:weighted_tree_nested_recursive_branch(_, _, _)).

test(nested_structural_tree_branch_recombine_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:tree_sum_nested_branch_recombine([], 0)),
    assertz(user:(tree_sum_nested_branch_recombine([V, L, R], Sum) :-
        ( V > 0 ->
            ( V > 1 ->
                tree_sum_nested_branch_recombine(L, LS),
                tree_sum_nested_branch_recombine(R, RS),
                Part is V + LS + RS
            ;   tree_sum_nested_branch_recombine(R, RS),
                tree_sum_nested_branch_recombine(L, LS),
                Part is V + LS + RS
            ),
            Sum is Part + 1
        ;   tree_sum_nested_branch_recombine(L, LS),
            tree_sum_nested_branch_recombine(R, RS),
            Sum is V + LS + RS
        )
    )),
    assertz(type_declarations:uw_type(tree_sum_nested_branch_recombine/2, 1, list(any))),
    assertz(type_declarations:uw_type(tree_sum_nested_branch_recombine/2, 2, integer)),
    assertz(type_declarations:uw_return_type(tree_sum_nested_branch_recombine/2, integer)),
    once(recursive_compiler:compile_recursive(tree_sum_nested_branch_recombine/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:tree_sum_nested_branch_recombine(_, _)).

test(weighted_nested_structural_tree_branch_recombine_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:weighted_tree_nested_branch_recombine([], _Scale, 0)),
    assertz(user:(weighted_tree_nested_branch_recombine([V, L, R], Scale, Sum) :-
        ( Scale > 1 ->
            ( V > 0 ->
                weighted_tree_nested_branch_recombine(L, Scale, LS),
                weighted_tree_nested_branch_recombine(R, Scale, RS),
                Part is (V * Scale) + LS + RS
            ;   weighted_tree_nested_branch_recombine(R, Scale, RS),
                weighted_tree_nested_branch_recombine(L, Scale, LS),
                Part is (V * Scale) + LS + RS
            ),
            Sum is Part + 1
        ;   weighted_tree_nested_branch_recombine(L, Scale, LS),
            weighted_tree_nested_branch_recombine(R, Scale, RS),
            Sum is (V * Scale) + LS + RS
        )
    )),
    assertz(type_declarations:uw_type(weighted_tree_nested_branch_recombine/3, 1, list(any))),
    assertz(type_declarations:uw_type(weighted_tree_nested_branch_recombine/3, 2, integer)),
    assertz(type_declarations:uw_type(weighted_tree_nested_branch_recombine/3, 3, integer)),
    assertz(type_declarations:uw_return_type(weighted_tree_nested_branch_recombine/3, integer)),
    once(recursive_compiler:compile_recursive(weighted_tree_nested_branch_recombine/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:weighted_tree_nested_branch_recombine(_, _, _)).

test(nested_structural_tree_branch_prework_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:tree_sum_nested_branch_prework([], 0)),
    assertz(user:(tree_sum_nested_branch_prework([V, L, R], Sum) :-
        ( V > 0 ->
            Bias is V + 1,
            ( V > 1 ->
                tree_sum_nested_branch_prework(L, LS),
                tree_sum_nested_branch_prework(R, RS)
            ;   tree_sum_nested_branch_prework(R, RS),
                tree_sum_nested_branch_prework(L, LS)
            ),
            Sum is Bias + LS + RS
        ;   tree_sum_nested_branch_prework(L, LS),
            tree_sum_nested_branch_prework(R, RS),
            Sum is V + LS + RS
        )
    )),
    assertz(type_declarations:uw_type(tree_sum_nested_branch_prework/2, 1, list(any))),
    assertz(type_declarations:uw_type(tree_sum_nested_branch_prework/2, 2, integer)),
    assertz(type_declarations:uw_return_type(tree_sum_nested_branch_prework/2, integer)),
    once(recursive_compiler:compile_recursive(tree_sum_nested_branch_prework/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:tree_sum_nested_branch_prework(_, _)).

test(weighted_nested_structural_tree_branch_prework_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:weighted_tree_nested_branch_prework([], _Scale, 0)),
    assertz(user:(weighted_tree_nested_branch_prework([V, L, R], Scale, Sum) :-
        ( Scale > 1 ->
            Bias is V * Scale,
            ( V > 0 ->
                weighted_tree_nested_branch_prework(L, Scale, LS),
                weighted_tree_nested_branch_prework(R, Scale, RS)
            ;   weighted_tree_nested_branch_prework(R, Scale, RS),
                weighted_tree_nested_branch_prework(L, Scale, LS)
            ),
            Sum is Bias + LS + RS
        ;   weighted_tree_nested_branch_prework(L, Scale, LS),
            weighted_tree_nested_branch_prework(R, Scale, RS),
            Sum is (V * Scale) + LS + RS
        )
    )),
    assertz(type_declarations:uw_type(weighted_tree_nested_branch_prework/3, 1, list(any))),
    assertz(type_declarations:uw_type(weighted_tree_nested_branch_prework/3, 2, integer)),
    assertz(type_declarations:uw_type(weighted_tree_nested_branch_prework/3, 3, integer)),
    assertz(type_declarations:uw_return_type(weighted_tree_nested_branch_prework/3, integer)),
    once(recursive_compiler:compile_recursive(weighted_tree_nested_branch_prework/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:weighted_tree_nested_branch_prework(_, _, _)).

test(double_nested_structural_tree_calls_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:tree_sum_double_nested_calls([], 0)),
    assertz(user:(tree_sum_double_nested_calls([V, L, R], Sum) :-
        ( V > 0 ->
            Bias is V + 1,
            ( V > 1 ->
                ( V > 2 ->
                    tree_sum_double_nested_calls(L, LS),
                    tree_sum_double_nested_calls(R, RS)
                ;   tree_sum_double_nested_calls(R, RS),
                    tree_sum_double_nested_calls(L, LS)
                ),
                Part is Bias + LS + RS
            ;   tree_sum_double_nested_calls(L, LS),
                tree_sum_double_nested_calls(R, RS),
                Part is V + LS + RS
            ),
            Sum is Part + 1
        ;   tree_sum_double_nested_calls(L, LS),
            tree_sum_double_nested_calls(R, RS),
            Sum is V + LS + RS
        )
    )),
    assertz(type_declarations:uw_type(tree_sum_double_nested_calls/2, 1, list(any))),
    assertz(type_declarations:uw_type(tree_sum_double_nested_calls/2, 2, integer)),
    assertz(type_declarations:uw_return_type(tree_sum_double_nested_calls/2, integer)),
    once(recursive_compiler:compile_recursive(tree_sum_double_nested_calls/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:tree_sum_double_nested_calls(_, _)).

test(weighted_double_nested_structural_tree_calls_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:weighted_tree_double_nested_calls([], _Scale, 0)),
    assertz(user:(weighted_tree_double_nested_calls([V, L, R], Scale, Sum) :-
        ( Scale > 1 ->
            Bias is V * Scale,
            ( V > 0 ->
                ( Scale > 2 ->
                    weighted_tree_double_nested_calls(L, Scale, LS),
                    weighted_tree_double_nested_calls(R, Scale, RS)
                ;   weighted_tree_double_nested_calls(R, Scale, RS),
                    weighted_tree_double_nested_calls(L, Scale, LS)
                ),
                Part is Bias + LS + RS
            ;   weighted_tree_double_nested_calls(L, Scale, LS),
                weighted_tree_double_nested_calls(R, Scale, RS),
                Part is (V * Scale) + LS + RS
            ),
            Sum is Part + 1
        ;   weighted_tree_double_nested_calls(L, Scale, LS),
            weighted_tree_double_nested_calls(R, Scale, RS),
            Sum is (V * Scale) + LS + RS
        )
    )),
    assertz(type_declarations:uw_type(weighted_tree_double_nested_calls/3, 1, list(any))),
    assertz(type_declarations:uw_type(weighted_tree_double_nested_calls/3, 2, integer)),
    assertz(type_declarations:uw_type(weighted_tree_double_nested_calls/3, 3, integer)),
    assertz(type_declarations:uw_return_type(weighted_tree_double_nested_calls/3, integer)),
    once(recursive_compiler:compile_recursive(weighted_tree_double_nested_calls/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:weighted_tree_double_nested_calls(_, _, _)).

test(nary_structural_tree_subtree_invariant_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:weighted_tree_sum_subtree_scale([], _Scale, 0)),
    assertz(user:(weighted_tree_sum_subtree_scale([V, L, R], Scale, Sum) :-
        ScaleL is Scale + 1,
        ScaleR is Scale + 2,
        weighted_tree_sum_subtree_scale(L, ScaleL, LS),
        weighted_tree_sum_subtree_scale(R, ScaleR, RS),
        Sum is (V * Scale) + LS + RS
    )),
    assertz(type_declarations:uw_type(weighted_tree_sum_subtree_scale/3, 1, list(any))),
    assertz(type_declarations:uw_type(weighted_tree_sum_subtree_scale/3, 2, integer)),
    assertz(type_declarations:uw_type(weighted_tree_sum_subtree_scale/3, 3, integer)),
    assertz(type_declarations:uw_return_type(weighted_tree_sum_subtree_scale/3, integer)),
    once(recursive_compiler:compile_recursive(weighted_tree_sum_subtree_scale/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:weighted_tree_sum_subtree_scale(_, _, _)).

test(nary_structural_tree_subtree_branch_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:weighted_tree_sum_subtree_branch([], _Scale, 0)),
    assertz(user:(weighted_tree_sum_subtree_branch([V, L, R], Scale, Sum) :-
        ( Scale > 1 ->
            ScaleL is Scale + 1,
            ScaleR is Scale + 2
        ;   ScaleL is Scale + 3,
            ScaleR is Scale + 4
        ),
        weighted_tree_sum_subtree_branch(L, ScaleL, LS),
        weighted_tree_sum_subtree_branch(R, ScaleR, RS),
        Sum is (V * ScaleL) + LS + RS + ScaleR
    )),
    assertz(type_declarations:uw_type(weighted_tree_sum_subtree_branch/3, 1, list(any))),
    assertz(type_declarations:uw_type(weighted_tree_sum_subtree_branch/3, 2, integer)),
    assertz(type_declarations:uw_type(weighted_tree_sum_subtree_branch/3, 3, integer)),
    assertz(type_declarations:uw_return_type(weighted_tree_sum_subtree_branch/3, integer)),
    once(recursive_compiler:compile_recursive(weighted_tree_sum_subtree_branch/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:weighted_tree_sum_subtree_branch(_, _, _)).

test(double_nested_structural_tree_subtree_context_guard_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:weighted_tree_double_nested_subtree_context_guard([], _Scale, 0)),
    assertz(user:(weighted_tree_double_nested_subtree_context_guard([V, L, R], Scale, Sum) :-
        ( Scale > 1 ->
            Bias is V * Scale,
            ( V > 0 ->
                ( Scale > 2 ->
                    ScaleL is Scale + 1,
                    ScaleR is Scale + 2
                ;   ScaleL is Scale + 3,
                    ScaleR is Scale + 4
                ),
                weighted_tree_double_nested_subtree_context_guard(L, ScaleL, LS),
                weighted_tree_double_nested_subtree_context_guard(R, ScaleR, RS),
                Part is Bias + LS + RS + ScaleR
            ;   weighted_tree_double_nested_subtree_context_guard(L, Scale, LS),
                weighted_tree_double_nested_subtree_context_guard(R, Scale, RS),
                Part is (V * Scale) + LS + RS
            ),
            Sum is Part + 1
        ;   weighted_tree_double_nested_subtree_context_guard(L, Scale, LS),
            weighted_tree_double_nested_subtree_context_guard(R, Scale, RS),
            Sum is (V * Scale) + LS + RS
        )
    )),
    assertz(type_declarations:uw_type(weighted_tree_double_nested_subtree_context_guard/3, 1, list(any))),
    assertz(type_declarations:uw_type(weighted_tree_double_nested_subtree_context_guard/3, 2, integer)),
    assertz(type_declarations:uw_type(weighted_tree_double_nested_subtree_context_guard/3, 3, integer)),
    assertz(type_declarations:uw_return_type(weighted_tree_double_nested_subtree_context_guard/3, integer)),
    once(recursive_compiler:compile_recursive(weighted_tree_double_nested_subtree_context_guard/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:weighted_tree_double_nested_subtree_context_guard(_, _, _)).

test(nary_structural_tree_prework_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:weighted_tree_sum_prework([], _Scale, 0)),
    assertz(user:(weighted_tree_sum_prework([V, L, R], Scale, Sum) :-
        Scale > 0,
        W is V * Scale,
        weighted_tree_sum_prework(L, Scale, LS),
        weighted_tree_sum_prework(R, Scale, RS),
        Sum is W + LS + RS
    )),
    assertz(type_declarations:uw_type(weighted_tree_sum_prework/3, 1, list(any))),
    assertz(type_declarations:uw_type(weighted_tree_sum_prework/3, 2, integer)),
    assertz(type_declarations:uw_type(weighted_tree_sum_prework/3, 3, integer)),
    assertz(type_declarations:uw_return_type(weighted_tree_sum_prework/3, integer)),
    once(recursive_compiler:compile_recursive(weighted_tree_sum_prework/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:weighted_tree_sum_prework(_, _, _)).

test(nary_structural_tree_branching_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:weighted_tree_sum_branch([], _Scale, 0)),
    assertz(user:(weighted_tree_sum_branch([V, L, R], Scale, Sum) :-
        ( Scale > 1 ->
            Base is V * Scale,
            W is Base + 1
        ;   Base is V + Scale,
            W is Base + 2
        ),
        weighted_tree_sum_branch(L, Scale, LS),
        weighted_tree_sum_branch(R, Scale, RS),
        Sum is W + LS + RS
    )),
    assertz(type_declarations:uw_type(weighted_tree_sum_branch/3, 1, list(any))),
    assertz(type_declarations:uw_type(weighted_tree_sum_branch/3, 2, integer)),
    assertz(type_declarations:uw_type(weighted_tree_sum_branch/3, 3, integer)),
    assertz(type_declarations:uw_return_type(weighted_tree_sum_branch/3, integer)),
    once(recursive_compiler:compile_recursive(weighted_tree_sum_branch/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:weighted_tree_sum_branch(_, _, _)).

test(nary_structural_tree_asymmetric_branching_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:weighted_tree_sum_asym_branch([], _Scale, 0)),
    assertz(user:(weighted_tree_sum_asym_branch([V, L, R], Scale, Sum) :-
        ( Scale > 1 -> A is V * Scale ; B is V + Scale ),
        weighted_tree_sum_asym_branch(L, Scale, LS),
        weighted_tree_sum_asym_branch(R, Scale, RS),
        ( Scale > 1 -> Sum is A + LS + RS ; Sum is B + LS + RS )
    )),
    assertz(type_declarations:uw_type(weighted_tree_sum_asym_branch/3, 1, list(any))),
    assertz(type_declarations:uw_type(weighted_tree_sum_asym_branch/3, 2, integer)),
    assertz(type_declarations:uw_type(weighted_tree_sum_asym_branch/3, 3, integer)),
    assertz(type_declarations:uw_return_type(weighted_tree_sum_asym_branch/3, integer)),
    once(recursive_compiler:compile_recursive(weighted_tree_sum_asym_branch/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:weighted_tree_sum_asym_branch(_, _, _)).

test(nary_structural_tree_recursive_nonleading_driver_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:weighted_tree_affine_sum(_Scale, _Offset, [], 0)),
    assertz(user:(weighted_tree_affine_sum(Scale, Offset, [V, L, R], Sum) :-
        weighted_tree_affine_sum(Scale, Offset, L, LS),
        weighted_tree_affine_sum(Scale, Offset, R, RS),
        Sum is ((V * Scale) + Offset) + LS + RS
    )),
    assertz(type_declarations:uw_type(weighted_tree_affine_sum/4, 1, integer)),
    assertz(type_declarations:uw_type(weighted_tree_affine_sum/4, 2, integer)),
    assertz(type_declarations:uw_type(weighted_tree_affine_sum/4, 3, list(any))),
    assertz(type_declarations:uw_type(weighted_tree_affine_sum/4, 4, integer)),
    assertz(type_declarations:uw_return_type(weighted_tree_affine_sum/4, integer)),
    once(recursive_compiler:compile_recursive(weighted_tree_affine_sum/4, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:weighted_tree_affine_sum(_, _, _, _)).

test(nary_linear_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
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
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:power(_, _, _)).

test(nary_list_linear_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
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
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:list_length_from(_, _, _)).

test(guarded_nary_linear_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
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
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:power_if(_, _, _)).

test(guarded_nary_list_linear_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
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
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:count_occ(_, _, _)).

test(multistate_nary_linear_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
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
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:power_multistate(_, _, _)).

test(multistate_nary_list_linear_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
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
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:count_weighted(_, _, _)).

test(asymmetric_nary_linear_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:asym_rec_a(_Base, 0, 1)),
    assertz(user:(asym_rec_a(Base, N, Result) :-
        N > 0,
        N1 is N - 1,
        asym_rec_a(Base, N1, Prev),
        ( Base > 1 ->
            Step is Base * Prev,
            Result is Step + 1
        ;   Temp is Prev + 2,
            Result is Temp + Prev
        )
    )),
    assertz(type_declarations:uw_type(asym_rec_a/3, 1, integer)),
    assertz(type_declarations:uw_type(asym_rec_a/3, 2, integer)),
    assertz(type_declarations:uw_type(asym_rec_a/3, 3, integer)),
    assertz(type_declarations:uw_return_type(asym_rec_a/3, integer)),
    once(recursive_compiler:compile_recursive(asym_rec_a/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:asym_rec_a(_, _, _)).

test(asymmetric_nary_list_linear_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:asym_rec_c(_, [], 0)),
    assertz(user:(asym_rec_c(X, [Y|T], N) :-
        asym_rec_c(X, T, N1),
        ( X == Y ->
            Delta is N1 + 1,
            N is Delta + 1
        ;   Extra is N1 + 2,
            N is Extra + N1
        )
    )),
    assertz(type_declarations:uw_type(asym_rec_c/3, 1, integer)),
    assertz(type_declarations:uw_type(asym_rec_c/3, 2, list(integer))),
    assertz(type_declarations:uw_type(asym_rec_c/3, 3, integer)),
    assertz(type_declarations:uw_return_type(asym_rec_c/3, integer)),
    once(recursive_compiler:compile_recursive(asym_rec_c/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:asym_rec_c(_, _, _)).

test(structural_tree_dual_mutual_context_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_even_tree_ctx([], _T0)),
    assertz(user:(typr_mutual_even_tree_ctx([V, L, R], T) :-
        V >= T,
        typr_mutual_odd_tree_ctx(L, T),
        typr_mutual_odd_tree_ctx(R, T)
    )),
    assertz(user:typr_mutual_odd_tree_ctx([_, [], []], _T1)),
    assertz(user:(typr_mutual_odd_tree_ctx([V, L, R], T) :-
        V >= T,
        typr_mutual_even_tree_ctx(L, T),
        typr_mutual_even_tree_ctx(R, T)
    )),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_ctx/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_ctx/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_ctx/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_ctx/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_even_tree_ctx/2, bool)),
    assertz(type_declarations:uw_return_type(typr_mutual_odd_tree_ctx/2, bool)),
    once(recursive_compiler:compile_recursive(typr_mutual_even_tree_ctx/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_even_tree_ctx(_, _)),
    retractall(user:typr_mutual_odd_tree_ctx(_, _)).

test(structural_tree_dual_mutual_context_step_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_even_tree_ctx_step([], _T0)),
    assertz(user:(typr_mutual_even_tree_ctx_step([V, L, R], T) :-
        V >= T,
        T1 is T + 1,
        typr_mutual_odd_tree_ctx_step(L, T1),
        typr_mutual_odd_tree_ctx_step(R, T1)
    )),
    assertz(user:typr_mutual_odd_tree_ctx_step([_, [], []], _T1)),
    assertz(user:(typr_mutual_odd_tree_ctx_step([V, L, R], T) :-
        V >= T,
        T1 is T + 1,
        typr_mutual_even_tree_ctx_step(L, T1),
        typr_mutual_even_tree_ctx_step(R, T1)
    )),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_ctx_step/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_ctx_step/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_ctx_step/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_ctx_step/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_even_tree_ctx_step/2, bool)),
    assertz(type_declarations:uw_return_type(typr_mutual_odd_tree_ctx_step/2, bool)),
    once(recursive_compiler:compile_recursive(typr_mutual_even_tree_ctx_step/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_even_tree_ctx_step(_, _)),
    retractall(user:typr_mutual_odd_tree_ctx_step(_, _)).

test(structural_tree_dual_mutual_context_branch_step_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_even_tree_ctx_branch_step([], _T0)),
    assertz(user:(typr_mutual_even_tree_ctx_branch_step([V, L, R], T) :-
        ( V > T ->
            T1 is T + 1
        ;   T1 is T + 2
        ),
        typr_mutual_odd_tree_ctx_branch_step(L, T1),
        typr_mutual_odd_tree_ctx_branch_step(R, T1)
    )),
    assertz(user:typr_mutual_odd_tree_ctx_branch_step([_, [], []], _T1)),
    assertz(user:(typr_mutual_odd_tree_ctx_branch_step([V, L, R], T) :-
        ( V > T ->
            T1 is T + 1
        ;   T1 is T + 2
        ),
        typr_mutual_even_tree_ctx_branch_step(L, T1),
        typr_mutual_even_tree_ctx_branch_step(R, T1)
    )),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_ctx_branch_step/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_ctx_branch_step/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_ctx_branch_step/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_ctx_branch_step/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_even_tree_ctx_branch_step/2, bool)),
    assertz(type_declarations:uw_return_type(typr_mutual_odd_tree_ctx_branch_step/2, bool)),
    once(recursive_compiler:compile_recursive(typr_mutual_even_tree_ctx_branch_step/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_even_tree_ctx_branch_step(_, _)),
    retractall(user:typr_mutual_odd_tree_ctx_branch_step(_, _)).

test(structural_tree_dual_mutual_context_branch_pre_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_even_tree_ctx_branch_pre([], _T0)),
    assertz(user:(typr_mutual_even_tree_ctx_branch_pre([V, L, R], T) :-
        ( V > T ->
            Left = L,
            Right = R
        ;   Left = R,
            Right = L
        ),
        typr_mutual_odd_tree_ctx_branch_pre(Left, T),
        typr_mutual_odd_tree_ctx_branch_pre(Right, T)
    )),
    assertz(user:typr_mutual_odd_tree_ctx_branch_pre([_, [], []], _T1)),
    assertz(user:(typr_mutual_odd_tree_ctx_branch_pre([V, L, R], T) :-
        ( V > T ->
            Left = L,
            Right = R
        ;   Left = R,
            Right = L
        ),
        typr_mutual_even_tree_ctx_branch_pre(Left, T),
        typr_mutual_even_tree_ctx_branch_pre(Right, T)
    )),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_ctx_branch_pre/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_ctx_branch_pre/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_ctx_branch_pre/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_ctx_branch_pre/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_even_tree_ctx_branch_pre/2, bool)),
    assertz(type_declarations:uw_return_type(typr_mutual_odd_tree_ctx_branch_pre/2, bool)),
    once(recursive_compiler:compile_recursive(typr_mutual_even_tree_ctx_branch_pre/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_even_tree_ctx_branch_pre(_, _)),
    retractall(user:typr_mutual_odd_tree_ctx_branch_pre(_, _)).

test(structural_tree_dual_mutual_context_recursive_branch_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_even_tree_ctx_recursive_branch([], _T0)),
    assertz(user:(typr_mutual_even_tree_ctx_recursive_branch([V, L, R], T) :-
        ( V > T ->
            typr_mutual_odd_tree_ctx_recursive_branch(L, T),
            typr_mutual_odd_tree_ctx_recursive_branch(R, T)
        ;   typr_mutual_odd_tree_ctx_recursive_branch(R, T),
            typr_mutual_odd_tree_ctx_recursive_branch(L, T)
        )
    )),
    assertz(user:typr_mutual_odd_tree_ctx_recursive_branch([_, [], []], _T1)),
    assertz(user:(typr_mutual_odd_tree_ctx_recursive_branch([V, L, R], T) :-
        ( V > T ->
            typr_mutual_even_tree_ctx_recursive_branch(L, T),
            typr_mutual_even_tree_ctx_recursive_branch(R, T)
        ;   typr_mutual_even_tree_ctx_recursive_branch(R, T),
            typr_mutual_even_tree_ctx_recursive_branch(L, T)
        )
    )),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_ctx_recursive_branch/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_ctx_recursive_branch/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_ctx_recursive_branch/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_ctx_recursive_branch/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_even_tree_ctx_recursive_branch/2, bool)),
    assertz(type_declarations:uw_return_type(typr_mutual_odd_tree_ctx_recursive_branch/2, bool)),
    once(recursive_compiler:compile_recursive(typr_mutual_even_tree_ctx_recursive_branch/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_even_tree_ctx_recursive_branch(_, _)),
    retractall(user:typr_mutual_odd_tree_ctx_recursive_branch(_, _)).

test(structural_tree_dual_mutual_context_nested_branch_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_even_tree_ctx_nested_branch([], _T0)),
    assertz(user:(typr_mutual_even_tree_ctx_nested_branch([V, L, R], T) :-
        ( V > T ->
            ( V > T + 10 ->
                typr_mutual_odd_tree_ctx_nested_branch(L, T),
                typr_mutual_odd_tree_ctx_nested_branch(R, T)
            ;   typr_mutual_odd_tree_ctx_nested_branch(R, T),
                typr_mutual_odd_tree_ctx_nested_branch(L, T)
            )
        ;   typr_mutual_odd_tree_ctx_nested_branch(L, T),
            typr_mutual_odd_tree_ctx_nested_branch(R, T)
        )
    )),
    assertz(user:typr_mutual_odd_tree_ctx_nested_branch([_, [], []], _T1)),
    assertz(user:(typr_mutual_odd_tree_ctx_nested_branch([V, L, R], T) :-
        ( V > T ->
            ( V > T + 10 ->
                typr_mutual_even_tree_ctx_nested_branch(L, T),
                typr_mutual_even_tree_ctx_nested_branch(R, T)
            ;   typr_mutual_even_tree_ctx_nested_branch(R, T),
                typr_mutual_even_tree_ctx_nested_branch(L, T)
            )
        ;   typr_mutual_even_tree_ctx_nested_branch(L, T),
            typr_mutual_even_tree_ctx_nested_branch(R, T)
        )
    )),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_ctx_nested_branch/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_ctx_nested_branch/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_ctx_nested_branch/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_ctx_nested_branch/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_even_tree_ctx_nested_branch/2, bool)),
    assertz(type_declarations:uw_return_type(typr_mutual_odd_tree_ctx_nested_branch/2, bool)),
    once(recursive_compiler:compile_recursive(typr_mutual_even_tree_ctx_nested_branch/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_even_tree_ctx_nested_branch(_, _)),
    retractall(user:typr_mutual_odd_tree_ctx_nested_branch(_, _)).

test(structural_tree_dual_mutual_context_postcall_guard_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_even_tree_ctx_postcall_guard([], _T0)),
    assertz(user:(typr_mutual_even_tree_ctx_postcall_guard([V, L, R], T) :-
        ( V > T ->
            typr_mutual_odd_tree_ctx_postcall_guard(L, T),
            typr_mutual_odd_tree_ctx_postcall_guard(R, T),
            ( V > T + 10 ->
                V >= T
            ;   V >= T - 1
            )
        ;   typr_mutual_odd_tree_ctx_postcall_guard(L, T),
            typr_mutual_odd_tree_ctx_postcall_guard(R, T)
        )
    )),
    assertz(user:typr_mutual_odd_tree_ctx_postcall_guard([_, [], []], _T1)),
    assertz(user:(typr_mutual_odd_tree_ctx_postcall_guard([V, L, R], T) :-
        ( V > T ->
            typr_mutual_even_tree_ctx_postcall_guard(L, T),
            typr_mutual_even_tree_ctx_postcall_guard(R, T),
            ( V > T + 10 ->
                V >= T
            ;   V >= T - 1
            )
        ;   typr_mutual_even_tree_ctx_postcall_guard(L, T),
            typr_mutual_even_tree_ctx_postcall_guard(R, T)
        )
    )),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_ctx_postcall_guard/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_ctx_postcall_guard/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_ctx_postcall_guard/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_ctx_postcall_guard/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_even_tree_ctx_postcall_guard/2, bool)),
    assertz(type_declarations:uw_return_type(typr_mutual_odd_tree_ctx_postcall_guard/2, bool)),
    once(recursive_compiler:compile_recursive(typr_mutual_even_tree_ctx_postcall_guard/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_even_tree_ctx_postcall_guard(_, _)),
    retractall(user:typr_mutual_odd_tree_ctx_postcall_guard(_, _)).

test(structural_tree_dual_mutual_context_intercall_postcall_control_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_even_tree_ctx_intercall_post([], _T0)),
    assertz(user:(typr_mutual_even_tree_ctx_intercall_post([V, L, R], T) :-
        T1 is T + 1,
        typr_mutual_odd_tree_ctx_intercall_post(L, T1),
        T2 is T1 + 1,
        typr_mutual_odd_tree_ctx_intercall_post(R, T2),
        ( V > T2 ->
            V >= T
        ;   V >= T - 1
        )
    )),
    assertz(user:typr_mutual_odd_tree_ctx_intercall_post([_, [], []], _T1)),
    assertz(user:(typr_mutual_odd_tree_ctx_intercall_post([V, L, R], T) :-
        T1 is T + 1,
        typr_mutual_even_tree_ctx_intercall_post(L, T1),
        T2 is T1 + 1,
        typr_mutual_even_tree_ctx_intercall_post(R, T2),
        ( V > T2 ->
            V >= T
        ;   V >= T - 1
        )
    )),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_ctx_intercall_post/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_ctx_intercall_post/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_ctx_intercall_post/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_ctx_intercall_post/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_even_tree_ctx_intercall_post/2, bool)),
    assertz(type_declarations:uw_return_type(typr_mutual_odd_tree_ctx_intercall_post/2, bool)),
    once(recursive_compiler:compile_recursive(typr_mutual_even_tree_ctx_intercall_post/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_even_tree_ctx_intercall_post(_, _)),
    retractall(user:typr_mutual_odd_tree_ctx_intercall_post(_, _)).

test(structural_tree_dual_mutual_context_nested_postcall_guard_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_even_tree_ctx_nested_post([], _T0)),
    assertz(user:(typr_mutual_even_tree_ctx_nested_post([V, L, R], T) :-
        typr_mutual_odd_tree_ctx_nested_post(L, T),
        typr_mutual_odd_tree_ctx_nested_post(R, T),
        ( V > T + 10 ->
            ( V >= T ->
                V >= T - 1
            ;   V >= T - 2
            )
        ;   V >= T - 3
        )
    )),
    assertz(user:typr_mutual_odd_tree_ctx_nested_post([_, [], []], _T1)),
    assertz(user:(typr_mutual_odd_tree_ctx_nested_post([V, L, R], T) :-
        typr_mutual_even_tree_ctx_nested_post(L, T),
        typr_mutual_even_tree_ctx_nested_post(R, T),
        ( V > T + 10 ->
            ( V >= T ->
                V >= T - 1
            ;   V >= T - 2
            )
        ;   V >= T - 3
        )
    )),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_ctx_nested_post/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_ctx_nested_post/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_ctx_nested_post/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_ctx_nested_post/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_even_tree_ctx_nested_post/2, bool)),
    assertz(type_declarations:uw_return_type(typr_mutual_odd_tree_ctx_nested_post/2, bool)),
    once(recursive_compiler:compile_recursive(typr_mutual_even_tree_ctx_nested_post/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_even_tree_ctx_nested_post(_, _)),
    retractall(user:typr_mutual_odd_tree_ctx_nested_post(_, _)).

test(structural_tree_dual_mutual_context_branch_second_call_nested_postcall_control_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_even_tree_ctx_call_nested_post([], _T0)),
    assertz(user:(typr_mutual_even_tree_ctx_call_nested_post([V, L, R], T) :-
        typr_mutual_odd_tree_ctx_call_nested_post(L, T),
        ( V > T ->
            typr_mutual_odd_tree_ctx_call_nested_post(R, T),
            ( V > T + 10 ->
                V >= T
            ;   V >= T - 1
            )
        ;   typr_mutual_odd_tree_ctx_call_nested_post(R, T)
        )
    )),
    assertz(user:typr_mutual_odd_tree_ctx_call_nested_post([_, [], []], _T1)),
    assertz(user:(typr_mutual_odd_tree_ctx_call_nested_post([V, L, R], T) :-
        typr_mutual_even_tree_ctx_call_nested_post(L, T),
        ( V > T ->
            typr_mutual_even_tree_ctx_call_nested_post(R, T),
            ( V > T + 10 ->
                V >= T
            ;   V >= T - 1
            )
        ;   typr_mutual_even_tree_ctx_call_nested_post(R, T)
        )
    )),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_ctx_call_nested_post/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_ctx_call_nested_post/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_ctx_call_nested_post/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_ctx_call_nested_post/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_even_tree_ctx_call_nested_post/2, bool)),
    assertz(type_declarations:uw_return_type(typr_mutual_odd_tree_ctx_call_nested_post/2, bool)),
    once(recursive_compiler:compile_recursive(typr_mutual_even_tree_ctx_call_nested_post/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_even_tree_ctx_call_nested_post(_, _)),
    retractall(user:typr_mutual_odd_tree_ctx_call_nested_post(_, _)).

test(structural_tree_dual_mutual_context_prework_plus_branch_second_call_post_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_even_tree_ctx_pre_branch_call_post([], _T0)),
    assertz(user:(typr_mutual_even_tree_ctx_pre_branch_call_post([V, L, R], T) :-
        T1 is T + 1,
        typr_mutual_odd_tree_ctx_pre_branch_call_post(L, T1),
        ( V > T ->
            typr_mutual_odd_tree_ctx_pre_branch_call_post(R, T1),
            ( V > T1 + 10 ->
                V >= T1
            ;   V >= T1 - 1
            )
        ;   typr_mutual_odd_tree_ctx_pre_branch_call_post(R, T1)
        )
    )),
    assertz(user:typr_mutual_odd_tree_ctx_pre_branch_call_post([_, [], []], _T1)),
    assertz(user:(typr_mutual_odd_tree_ctx_pre_branch_call_post([V, L, R], T) :-
        T1 is T + 1,
        typr_mutual_even_tree_ctx_pre_branch_call_post(L, T1),
        ( V > T ->
            typr_mutual_even_tree_ctx_pre_branch_call_post(R, T1),
            ( V > T1 + 10 ->
                V >= T1
            ;   V >= T1 - 1
            )
        ;   typr_mutual_even_tree_ctx_pre_branch_call_post(R, T1)
        )
    )),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_ctx_pre_branch_call_post/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_ctx_pre_branch_call_post/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_ctx_pre_branch_call_post/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_ctx_pre_branch_call_post/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_even_tree_ctx_pre_branch_call_post/2, bool)),
    assertz(type_declarations:uw_return_type(typr_mutual_odd_tree_ctx_pre_branch_call_post/2, bool)),
    once(recursive_compiler:compile_recursive(typr_mutual_even_tree_ctx_pre_branch_call_post/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_even_tree_ctx_pre_branch_call_post(_, _)),
    retractall(user:typr_mutual_odd_tree_ctx_pre_branch_call_post(_, _)).

test(structural_tree_dual_mutual_context_branch_step_second_call_post_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_even_tree_ctx_branch_step_post([], _T0)),
    assertz(user:(typr_mutual_even_tree_ctx_branch_step_post([V, L, R], T) :-
        typr_mutual_odd_tree_ctx_branch_step_post(L, T),
        ( V > T ->
            T1 is T + 1,
            typr_mutual_odd_tree_ctx_branch_step_post(R, T1),
            ( V >= T1 ->
                V >= T
            ;   V >= T - 1
            )
        ;   T2 is T + 2,
            typr_mutual_odd_tree_ctx_branch_step_post(R, T2),
            ( V >= T2 ->
                V >= T
            ;   V >= T - 2
            )
        )
    )),
    assertz(user:typr_mutual_odd_tree_ctx_branch_step_post([_, [], []], _T1)),
    assertz(user:(typr_mutual_odd_tree_ctx_branch_step_post([V, L, R], T) :-
        typr_mutual_even_tree_ctx_branch_step_post(L, T),
        ( V > T ->
            T1 is T + 1,
            typr_mutual_even_tree_ctx_branch_step_post(R, T1),
            ( V >= T1 ->
                V >= T
            ;   V >= T - 1
            )
        ;   T2 is T + 2,
            typr_mutual_even_tree_ctx_branch_step_post(R, T2),
            ( V >= T2 ->
                V >= T
            ;   V >= T - 2
            )
        )
    )),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_ctx_branch_step_post/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_ctx_branch_step_post/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_ctx_branch_step_post/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_ctx_branch_step_post/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_even_tree_ctx_branch_step_post/2, bool)),
    assertz(type_declarations:uw_return_type(typr_mutual_odd_tree_ctx_branch_step_post/2, bool)),
    once(recursive_compiler:compile_recursive(typr_mutual_even_tree_ctx_branch_step_post/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_even_tree_ctx_branch_step_post(_, _)),
    retractall(user:typr_mutual_odd_tree_ctx_branch_step_post(_, _)).

test(structural_tree_dual_mutual_multi_context_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_even_tree_ctx2([], _T0, _B0)),
    assertz(user:(typr_mutual_even_tree_ctx2([V, L, R], T, B) :-
        T1 is T + 1,
        B1 is B + 2,
        typr_mutual_odd_tree_ctx2(L, T1, B1),
        typr_mutual_odd_tree_ctx2(R, T1, B1),
        V >= T,
        V >= B
    )),
    assertz(user:typr_mutual_odd_tree_ctx2([_, [], []], _T1, _B1)),
    assertz(user:(typr_mutual_odd_tree_ctx2([V, L, R], T, B) :-
        T1 is T + 1,
        B1 is B + 2,
        typr_mutual_even_tree_ctx2(L, T1, B1),
        typr_mutual_even_tree_ctx2(R, T1, B1),
        V >= T,
        V >= B
    )),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_ctx2/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_ctx2/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_ctx2/3, 3, integer)),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_ctx2/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_ctx2/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_ctx2/3, 3, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_even_tree_ctx2/3, bool)),
    assertz(type_declarations:uw_return_type(typr_mutual_odd_tree_ctx2/3, bool)),
    once(recursive_compiler:compile_recursive(typr_mutual_even_tree_ctx2/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_even_tree_ctx2(_, _, _)),
    retractall(user:typr_mutual_odd_tree_ctx2(_, _, _)).

test(structural_tree_dual_mutual_integer_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_sum_even_tree([], 0)),
    assertz(user:(typr_mutual_sum_even_tree([V, L, R], S) :-
        typr_mutual_sum_odd_tree(L, SL),
        typr_mutual_sum_odd_tree(R, SR),
        S is V + SL + SR
    )),
    assertz(user:typr_mutual_sum_odd_tree([_, [], []], 1)),
    assertz(user:(typr_mutual_sum_odd_tree([V, L, R], S) :-
        typr_mutual_sum_even_tree(L, SL),
        typr_mutual_sum_even_tree(R, SR),
        S is V + SL + SR + 1
    )),
    assertz(type_declarations:uw_type(typr_mutual_sum_even_tree/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_sum_even_tree/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_sum_odd_tree/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_sum_odd_tree/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_sum_even_tree/2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_sum_odd_tree/2, integer)),
    once(recursive_compiler:compile_recursive(typr_mutual_sum_even_tree/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_sum_even_tree(_, _)),
    retractall(user:typr_mutual_sum_odd_tree(_, _)).

test(structural_tree_single_subtree_mutual_integer_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_sum_even_left_tree([], 0)),
    assertz(user:(typr_mutual_sum_even_left_tree([V, L, _], S) :-
        typr_mutual_sum_odd_left_tree(L, SL),
        S is V + SL
    )),
    assertz(user:typr_mutual_sum_odd_left_tree([_, [], []], 1)),
    assertz(user:(typr_mutual_sum_odd_left_tree([V, L, _], S) :-
        typr_mutual_sum_even_left_tree(L, SL),
        S is V - SL
    )),
    assertz(type_declarations:uw_type(typr_mutual_sum_even_left_tree/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_sum_even_left_tree/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_sum_odd_left_tree/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_sum_odd_left_tree/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_sum_even_left_tree/2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_sum_odd_left_tree/2, integer)),
    once(recursive_compiler:compile_recursive(typr_mutual_sum_even_left_tree/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_sum_even_left_tree(_, _)),
    retractall(user:typr_mutual_sum_odd_left_tree(_, _)).

test(structural_tree_mixed_mutual_integer_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_sum_even_mixed_tree([], 0)),
    assertz(user:(typr_mutual_sum_even_mixed_tree([V, L, _], S) :-
        typr_mutual_sum_odd_mixed_tree(L, SL),
        S is V + SL
    )),
    assertz(user:typr_mutual_sum_odd_mixed_tree([_, [], []], 1)),
    assertz(user:(typr_mutual_sum_odd_mixed_tree([V, L, R], S) :-
        typr_mutual_sum_even_mixed_tree(L, SL),
        typr_mutual_sum_even_mixed_tree(R, SR),
        S is V + SL + SR
    )),
    assertz(type_declarations:uw_type(typr_mutual_sum_even_mixed_tree/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_sum_even_mixed_tree/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_sum_odd_mixed_tree/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_sum_odd_mixed_tree/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_sum_even_mixed_tree/2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_sum_odd_mixed_tree/2, integer)),
    once(recursive_compiler:compile_recursive(typr_mutual_sum_even_mixed_tree/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_sum_even_mixed_tree(_, _)),
    retractall(user:typr_mutual_sum_odd_mixed_tree(_, _)).

test(structural_tree_dual_mutual_integer_context_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_weight_even_tree([], _W0, 0)),
    assertz(user:(typr_mutual_weight_even_tree([V, L, R], W, S) :-
        typr_mutual_weight_odd_tree(L, W, SL),
        typr_mutual_weight_odd_tree(R, W, SR),
        S is (V * W) + SL + SR
    )),
    assertz(user:typr_mutual_weight_odd_tree([_, [], []], _W1, 1)),
    assertz(user:(typr_mutual_weight_odd_tree([V, L, R], W, S) :-
        typr_mutual_weight_even_tree(L, W, SL),
        typr_mutual_weight_even_tree(R, W, SR),
        S is (V * W) + SL + SR + 1
    )),
    assertz(type_declarations:uw_type(typr_mutual_weight_even_tree/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_weight_even_tree/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_weight_even_tree/3, 3, integer)),
    assertz(type_declarations:uw_type(typr_mutual_weight_odd_tree/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_weight_odd_tree/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_weight_odd_tree/3, 3, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_weight_even_tree/3, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_weight_odd_tree/3, integer)),
    once(recursive_compiler:compile_recursive(typr_mutual_weight_even_tree/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_weight_even_tree(_, _, _)),
    retractall(user:typr_mutual_weight_odd_tree(_, _, _)).

test(structural_tree_dual_mutual_integer_branch_calls_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_sum_even_tree_branch_calls([], 0)),
    assertz(user:(typr_mutual_sum_even_tree_branch_calls([V, L, R], S) :-
        ( V > 0 -> Left = L, Right = R ; Left = R, Right = L ),
        typr_mutual_sum_odd_tree_branch_calls(Left, SL),
        typr_mutual_sum_odd_tree_branch_calls(Right, SR),
        S is V + SL + SR
    )),
    assertz(user:typr_mutual_sum_odd_tree_branch_calls([_, [], []], 1)),
    assertz(user:(typr_mutual_sum_odd_tree_branch_calls([V, L, R], S) :-
        ( V > 0 -> Left = L, Right = R ; Left = R, Right = L ),
        typr_mutual_sum_even_tree_branch_calls(Left, SL),
        typr_mutual_sum_even_tree_branch_calls(Right, SR),
        S is V + SL + SR
    )),
    assertz(type_declarations:uw_type(typr_mutual_sum_even_tree_branch_calls/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_sum_even_tree_branch_calls/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_sum_odd_tree_branch_calls/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_sum_odd_tree_branch_calls/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_sum_even_tree_branch_calls/2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_sum_odd_tree_branch_calls/2, integer)),
    once(recursive_compiler:compile_recursive(typr_mutual_sum_even_tree_branch_calls/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_sum_even_tree_branch_calls(_, _)),
    retractall(user:typr_mutual_sum_odd_tree_branch_calls(_, _)).

test(structural_tree_dual_mutual_integer_context_branch_calls_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_weight_even_tree_branch_calls([], _W0, 0)),
    assertz(user:(typr_mutual_weight_even_tree_branch_calls([V, L, R], W, S) :-
        ( V > W -> Left = L, Right = R ; Left = R, Right = L ),
        typr_mutual_weight_odd_tree_branch_calls(Left, W, SL),
        typr_mutual_weight_odd_tree_branch_calls(Right, W, SR),
        S is (V * W) + SL + SR
    )),
    assertz(user:typr_mutual_weight_odd_tree_branch_calls([_, [], []], _W1, 1)),
    assertz(user:(typr_mutual_weight_odd_tree_branch_calls([V, L, R], W, S) :-
        ( V > W -> Left = L, Right = R ; Left = R, Right = L ),
        typr_mutual_weight_even_tree_branch_calls(Left, W, SL),
        typr_mutual_weight_even_tree_branch_calls(Right, W, SR),
        S is (V * W) + SL + SR
    )),
    assertz(type_declarations:uw_type(typr_mutual_weight_even_tree_branch_calls/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_weight_even_tree_branch_calls/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_weight_even_tree_branch_calls/3, 3, integer)),
    assertz(type_declarations:uw_type(typr_mutual_weight_odd_tree_branch_calls/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_weight_odd_tree_branch_calls/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_weight_odd_tree_branch_calls/3, 3, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_weight_even_tree_branch_calls/3, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_weight_odd_tree_branch_calls/3, integer)),
    once(recursive_compiler:compile_recursive(typr_mutual_weight_even_tree_branch_calls/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_weight_even_tree_branch_calls(_, _, _)),
    retractall(user:typr_mutual_weight_odd_tree_branch_calls(_, _, _)).

test(structural_tree_dual_mutual_integer_recursive_branch_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_sum_even_tree_recursive_branch([], 0)),
    assertz(user:(typr_mutual_sum_even_tree_recursive_branch([V, L, R], S) :-
        ( V > 0 ->
            typr_mutual_sum_odd_tree_recursive_branch(L, SL),
            typr_mutual_sum_odd_tree_recursive_branch(R, SR)
        ;   typr_mutual_sum_odd_tree_recursive_branch(R, SR),
            typr_mutual_sum_odd_tree_recursive_branch(L, SL)
        ),
        S is V + SL + SR
    )),
    assertz(user:typr_mutual_sum_odd_tree_recursive_branch([_, [], []], 1)),
    assertz(user:(typr_mutual_sum_odd_tree_recursive_branch([V, L, R], S) :-
        ( V > 0 ->
            typr_mutual_sum_even_tree_recursive_branch(L, SL),
            typr_mutual_sum_even_tree_recursive_branch(R, SR)
        ;   typr_mutual_sum_even_tree_recursive_branch(R, SR),
            typr_mutual_sum_even_tree_recursive_branch(L, SL)
        ),
        S is V + SL + SR
    )),
    assertz(type_declarations:uw_type(typr_mutual_sum_even_tree_recursive_branch/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_sum_even_tree_recursive_branch/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_sum_odd_tree_recursive_branch/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_sum_odd_tree_recursive_branch/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_sum_even_tree_recursive_branch/2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_sum_odd_tree_recursive_branch/2, integer)),
    once(recursive_compiler:compile_recursive(typr_mutual_sum_even_tree_recursive_branch/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_sum_even_tree_recursive_branch(_, _)),
    retractall(user:typr_mutual_sum_odd_tree_recursive_branch(_, _)).

test(structural_tree_dual_mutual_integer_context_recursive_branch_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_weight_even_tree_recursive_branch([], _W0, 0)),
    assertz(user:(typr_mutual_weight_even_tree_recursive_branch([V, L, R], W, S) :-
        ( V > W ->
            typr_mutual_weight_odd_tree_recursive_branch(L, W, SL),
            typr_mutual_weight_odd_tree_recursive_branch(R, W, SR)
        ;   typr_mutual_weight_odd_tree_recursive_branch(R, W, SR),
            typr_mutual_weight_odd_tree_recursive_branch(L, W, SL)
        ),
        S is (V * W) + SL + SR
    )),
    assertz(user:typr_mutual_weight_odd_tree_recursive_branch([_, [], []], _W1, 1)),
    assertz(user:(typr_mutual_weight_odd_tree_recursive_branch([V, L, R], W, S) :-
        ( V > W ->
            typr_mutual_weight_even_tree_recursive_branch(L, W, SL),
            typr_mutual_weight_even_tree_recursive_branch(R, W, SR)
        ;   typr_mutual_weight_even_tree_recursive_branch(R, W, SR),
            typr_mutual_weight_even_tree_recursive_branch(L, W, SL)
        ),
        S is (V * W) + SL + SR
    )),
    assertz(type_declarations:uw_type(typr_mutual_weight_even_tree_recursive_branch/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_weight_even_tree_recursive_branch/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_weight_even_tree_recursive_branch/3, 3, integer)),
    assertz(type_declarations:uw_type(typr_mutual_weight_odd_tree_recursive_branch/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_weight_odd_tree_recursive_branch/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_weight_odd_tree_recursive_branch/3, 3, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_weight_even_tree_recursive_branch/3, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_weight_odd_tree_recursive_branch/3, integer)),
    once(recursive_compiler:compile_recursive(typr_mutual_weight_even_tree_recursive_branch/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_weight_even_tree_recursive_branch(_, _, _)),
    retractall(user:typr_mutual_weight_odd_tree_recursive_branch(_, _, _)).

test(structural_tree_dual_mutual_integer_nested_recursive_branch_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_sum_even_tree_nested_recursive_branch([], 0)),
    assertz(user:(typr_mutual_sum_even_tree_nested_recursive_branch([V, L, R], S) :-
        ( V > 0 ->
            ( V > 10 ->
                typr_mutual_sum_odd_tree_nested_recursive_branch(L, SL),
                typr_mutual_sum_odd_tree_nested_recursive_branch(R, SR)
            ;   typr_mutual_sum_odd_tree_nested_recursive_branch(R, SR),
                typr_mutual_sum_odd_tree_nested_recursive_branch(L, SL)
            )
        ;   typr_mutual_sum_odd_tree_nested_recursive_branch(L, SL),
            typr_mutual_sum_odd_tree_nested_recursive_branch(R, SR)
        ),
        S is V + SL + SR
    )),
    assertz(user:typr_mutual_sum_odd_tree_nested_recursive_branch([_, [], []], 1)),
    assertz(user:(typr_mutual_sum_odd_tree_nested_recursive_branch([V, L, R], S) :-
        ( V > 0 ->
            ( V > 10 ->
                typr_mutual_sum_even_tree_nested_recursive_branch(L, SL),
                typr_mutual_sum_even_tree_nested_recursive_branch(R, SR)
            ;   typr_mutual_sum_even_tree_nested_recursive_branch(R, SR),
                typr_mutual_sum_even_tree_nested_recursive_branch(L, SL)
            )
        ;   typr_mutual_sum_even_tree_nested_recursive_branch(L, SL),
            typr_mutual_sum_even_tree_nested_recursive_branch(R, SR)
        ),
        S is V + SL + SR
    )),
    assertz(type_declarations:uw_type(typr_mutual_sum_even_tree_nested_recursive_branch/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_sum_even_tree_nested_recursive_branch/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_sum_odd_tree_nested_recursive_branch/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_sum_odd_tree_nested_recursive_branch/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_sum_even_tree_nested_recursive_branch/2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_sum_odd_tree_nested_recursive_branch/2, integer)),
    once(recursive_compiler:compile_recursive(typr_mutual_sum_even_tree_nested_recursive_branch/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_sum_even_tree_nested_recursive_branch(_, _)),
    retractall(user:typr_mutual_sum_odd_tree_nested_recursive_branch(_, _)).

test(structural_tree_dual_mutual_integer_context_nested_recursive_branch_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_weight_even_tree_nested_recursive_branch([], _W0, 0)),
    assertz(user:(typr_mutual_weight_even_tree_nested_recursive_branch([V, L, R], W, S) :-
        ( V > W ->
            ( V > W + 10 ->
                typr_mutual_weight_odd_tree_nested_recursive_branch(L, W, SL),
                typr_mutual_weight_odd_tree_nested_recursive_branch(R, W, SR)
            ;   typr_mutual_weight_odd_tree_nested_recursive_branch(R, W, SR),
                typr_mutual_weight_odd_tree_nested_recursive_branch(L, W, SL)
            )
        ;   typr_mutual_weight_odd_tree_nested_recursive_branch(L, W, SL),
            typr_mutual_weight_odd_tree_nested_recursive_branch(R, W, SR)
        ),
        S is (V * W) + SL + SR
    )),
    assertz(user:typr_mutual_weight_odd_tree_nested_recursive_branch([_, [], []], _W1, 1)),
    assertz(user:(typr_mutual_weight_odd_tree_nested_recursive_branch([V, L, R], W, S) :-
        ( V > W ->
            ( V > W + 10 ->
                typr_mutual_weight_even_tree_nested_recursive_branch(L, W, SL),
                typr_mutual_weight_even_tree_nested_recursive_branch(R, W, SR)
            ;   typr_mutual_weight_even_tree_nested_recursive_branch(R, W, SR),
                typr_mutual_weight_even_tree_nested_recursive_branch(L, W, SL)
            )
        ;   typr_mutual_weight_even_tree_nested_recursive_branch(L, W, SL),
            typr_mutual_weight_even_tree_nested_recursive_branch(R, W, SR)
        ),
        S is (V * W) + SL + SR
    )),
    assertz(type_declarations:uw_type(typr_mutual_weight_even_tree_nested_recursive_branch/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_weight_even_tree_nested_recursive_branch/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_weight_even_tree_nested_recursive_branch/3, 3, integer)),
    assertz(type_declarations:uw_type(typr_mutual_weight_odd_tree_nested_recursive_branch/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_weight_odd_tree_nested_recursive_branch/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_weight_odd_tree_nested_recursive_branch/3, 3, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_weight_even_tree_nested_recursive_branch/3, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_weight_odd_tree_nested_recursive_branch/3, integer)),
    once(recursive_compiler:compile_recursive(typr_mutual_weight_even_tree_nested_recursive_branch/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_weight_even_tree_nested_recursive_branch(_, _, _)),
    retractall(user:typr_mutual_weight_odd_tree_nested_recursive_branch(_, _, _)).

test(structural_tree_dual_mutual_integer_branch_local_recombine_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_sum_even_tree_branch_part([], 0)),
    assertz(user:(typr_mutual_sum_even_tree_branch_part([V, L, R], S) :-
        ( V > 0 ->
            typr_mutual_sum_odd_tree_branch_part(L, SL),
            typr_mutual_sum_odd_tree_branch_part(R, SR),
            Part is V + SL + SR
        ;   typr_mutual_sum_odd_tree_branch_part(R, SR),
            typr_mutual_sum_odd_tree_branch_part(L, SL),
            Part is V - SL + SR
        ),
        S is Part + 1
    )),
    assertz(user:typr_mutual_sum_odd_tree_branch_part([_, [], []], 1)),
    assertz(user:(typr_mutual_sum_odd_tree_branch_part([V, L, R], S) :-
        ( V > 0 ->
            typr_mutual_sum_even_tree_branch_part(L, SL),
            typr_mutual_sum_even_tree_branch_part(R, SR),
            Part is V + SL + SR
        ;   typr_mutual_sum_even_tree_branch_part(R, SR),
            typr_mutual_sum_even_tree_branch_part(L, SL),
            Part is V - SL + SR
        ),
        S is Part + 1
    )),
    assertz(type_declarations:uw_type(typr_mutual_sum_even_tree_branch_part/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_sum_even_tree_branch_part/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_sum_odd_tree_branch_part/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_sum_odd_tree_branch_part/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_sum_even_tree_branch_part/2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_sum_odd_tree_branch_part/2, integer)),
    once(recursive_compiler:compile_recursive(typr_mutual_sum_even_tree_branch_part/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_sum_even_tree_branch_part(_, _)),
    retractall(user:typr_mutual_sum_odd_tree_branch_part(_, _)).

test(structural_tree_dual_mutual_integer_context_branch_local_recombine_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_weight_even_tree_branch_part([], _W0, 0)),
    assertz(user:(typr_mutual_weight_even_tree_branch_part([V, L, R], W, S) :-
        ( V > W ->
            typr_mutual_weight_odd_tree_branch_part(L, W, SL),
            typr_mutual_weight_odd_tree_branch_part(R, W, SR),
            Part is (V * W) + SL + SR
        ;   typr_mutual_weight_odd_tree_branch_part(R, W, SR),
            typr_mutual_weight_odd_tree_branch_part(L, W, SL),
            Part is (V * W) - SL + SR
        ),
        S is Part + W
    )),
    assertz(user:typr_mutual_weight_odd_tree_branch_part([_, [], []], _W1, 1)),
    assertz(user:(typr_mutual_weight_odd_tree_branch_part([V, L, R], W, S) :-
        ( V > W ->
            typr_mutual_weight_even_tree_branch_part(L, W, SL),
            typr_mutual_weight_even_tree_branch_part(R, W, SR),
            Part is (V * W) + SL + SR
        ;   typr_mutual_weight_even_tree_branch_part(R, W, SR),
            typr_mutual_weight_even_tree_branch_part(L, W, SL),
            Part is (V * W) - SL + SR
        ),
        S is Part + W
    )),
    assertz(type_declarations:uw_type(typr_mutual_weight_even_tree_branch_part/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_weight_even_tree_branch_part/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_weight_even_tree_branch_part/3, 3, integer)),
    assertz(type_declarations:uw_type(typr_mutual_weight_odd_tree_branch_part/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_weight_odd_tree_branch_part/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_weight_odd_tree_branch_part/3, 3, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_weight_even_tree_branch_part/3, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_weight_odd_tree_branch_part/3, integer)),
    once(recursive_compiler:compile_recursive(typr_mutual_weight_even_tree_branch_part/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_weight_even_tree_branch_part(_, _, _)),
    retractall(user:typr_mutual_weight_odd_tree_branch_part(_, _, _)).

test(structural_tree_dual_mutual_asymmetric_boolean_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_even_tree_asym([])),
    assertz(user:(typr_mutual_even_tree_asym([_, L, R]) :-
        typr_mutual_odd_tree_asym(L),
        typr_mutual_odd_tree_asym(R)
    )),
    assertz(user:typr_mutual_odd_tree_asym([_, [], []])),
    assertz(user:(typr_mutual_odd_tree_asym([V, L, R]) :-
        ( V > 0 ->
            typr_mutual_even_tree_asym(L),
            typr_mutual_even_tree_asym(R)
        ;   typr_mutual_even_tree_asym(R),
            typr_mutual_even_tree_asym(L)
        )
    )),
    assertz(type_declarations:uw_type(typr_mutual_even_tree_asym/1, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_odd_tree_asym/1, 1, list(any))),
    assertz(type_declarations:uw_return_type(typr_mutual_even_tree_asym/1, bool)),
    assertz(type_declarations:uw_return_type(typr_mutual_odd_tree_asym/1, bool)),
    once(recursive_compiler:compile_recursive(typr_mutual_even_tree_asym/1, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_even_tree_asym(_)),
    retractall(user:typr_mutual_odd_tree_asym(_)).

test(structural_tree_dual_mutual_integer_asymmetric_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_sum_even_tree_asym([], 0)),
    assertz(user:(typr_mutual_sum_even_tree_asym([V, L, R], S) :-
        typr_mutual_sum_odd_tree_asym(L, SL),
        typr_mutual_sum_odd_tree_asym(R, SR),
        S is V + SL + SR
    )),
    assertz(user:typr_mutual_sum_odd_tree_asym([_, [], []], 1)),
    assertz(user:(typr_mutual_sum_odd_tree_asym([V, L, R], S) :-
        ( V > 0 ->
            typr_mutual_sum_even_tree_asym(L, SL),
            typr_mutual_sum_even_tree_asym(R, SR)
        ;   typr_mutual_sum_even_tree_asym(R, SR),
            typr_mutual_sum_even_tree_asym(L, SL)
        ),
        S is V - SL + SR
    )),
    assertz(type_declarations:uw_type(typr_mutual_sum_even_tree_asym/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_sum_even_tree_asym/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_sum_odd_tree_asym/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_sum_odd_tree_asym/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_sum_even_tree_asym/2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_sum_odd_tree_asym/2, integer)),
    once(recursive_compiler:compile_recursive(typr_mutual_sum_even_tree_asym/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_sum_even_tree_asym(_, _)),
    retractall(user:typr_mutual_sum_odd_tree_asym(_, _)).

test(structural_tree_dual_mutual_integer_context_asymmetric_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_weight_even_tree_asym([], _W0, 0)),
    assertz(user:(typr_mutual_weight_even_tree_asym([V, L, R], W, S) :-
        typr_mutual_weight_odd_tree_asym(L, W, SL),
        typr_mutual_weight_odd_tree_asym(R, W, SR),
        S is (V * W) + SL + SR
    )),
    assertz(user:typr_mutual_weight_odd_tree_asym([_, [], []], _W1, 1)),
    assertz(user:(typr_mutual_weight_odd_tree_asym([V, L, R], W, S) :-
        ( V > W ->
            typr_mutual_weight_even_tree_asym(L, W, SL),
            typr_mutual_weight_even_tree_asym(R, W, SR)
        ;   typr_mutual_weight_even_tree_asym(R, W, SR),
            typr_mutual_weight_even_tree_asym(L, W, SL)
        ),
        S is (V * W) - SL + SR
    )),
    assertz(type_declarations:uw_type(typr_mutual_weight_even_tree_asym/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_weight_even_tree_asym/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_weight_even_tree_asym/3, 3, integer)),
    assertz(type_declarations:uw_type(typr_mutual_weight_odd_tree_asym/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_weight_odd_tree_asym/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_weight_odd_tree_asym/3, 3, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_weight_even_tree_asym/3, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_weight_odd_tree_asym/3, integer)),
    once(recursive_compiler:compile_recursive(typr_mutual_weight_even_tree_asym/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_weight_even_tree_asym(_, _, _)),
    retractall(user:typr_mutual_weight_odd_tree_asym(_, _, _)).

test(mixed_tree_list_mutual_boolean_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_tree_ok([])),
    assertz(user:(typr_tree_ok([_, L, R]) :-
        typr_forest_ok([L, R])
    )),
    assertz(user:typr_forest_ok([])),
    assertz(user:(typr_forest_ok([T|Ts]) :-
        typr_tree_ok(T),
        typr_forest_ok(Ts)
    )),
    assertz(type_declarations:uw_type(typr_tree_ok/1, 1, list(any))),
    assertz(type_declarations:uw_type(typr_forest_ok/1, 1, list(any))),
    assertz(type_declarations:uw_return_type(typr_tree_ok/1, bool)),
    assertz(type_declarations:uw_return_type(typr_forest_ok/1, bool)),
    once(recursive_compiler:compile_recursive(typr_tree_ok/1, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_tree_ok(_)),
    retractall(user:typr_forest_ok(_)).

test(mixed_tree_list_mutual_integer_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_tree_sum([], 0)),
    assertz(user:(typr_tree_sum([V, L, R], S) :-
        typr_forest_sum([L, R], Parts),
        S is V + Parts
    )),
    assertz(user:typr_forest_sum([], 0)),
    assertz(user:(typr_forest_sum([T|Ts], S) :-
        typr_tree_sum(T, ST),
        typr_forest_sum(Ts, SS),
        S is ST + SS
    )),
    assertz(type_declarations:uw_type(typr_tree_sum/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_tree_sum/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_forest_sum/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_forest_sum/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_tree_sum/2, integer)),
    assertz(type_declarations:uw_return_type(typr_forest_sum/2, integer)),
    once(recursive_compiler:compile_recursive(typr_tree_sum/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_tree_sum(_, _)),
    retractall(user:typr_forest_sum(_, _)).

test(mixed_tree_list_mutual_integer_context_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_tree_weight_sum([], _W0, 0)),
    assertz(user:(typr_tree_weight_sum([V, L, R], W, S) :-
        typr_forest_weight_sum([L, R], W, Parts),
        S is (V * W) + Parts
    )),
    assertz(user:typr_forest_weight_sum([], _W1, 0)),
    assertz(user:(typr_forest_weight_sum([T|Ts], W, S) :-
        typr_tree_weight_sum(T, W, ST),
        typr_forest_weight_sum(Ts, W, SS),
        S is ST + SS
    )),
    assertz(type_declarations:uw_type(typr_tree_weight_sum/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_tree_weight_sum/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_tree_weight_sum/3, 3, integer)),
    assertz(type_declarations:uw_type(typr_forest_weight_sum/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_forest_weight_sum/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_forest_weight_sum/3, 3, integer)),
    assertz(type_declarations:uw_return_type(typr_tree_weight_sum/3, integer)),
    assertz(type_declarations:uw_return_type(typr_forest_weight_sum/3, integer)),
    once(recursive_compiler:compile_recursive(typr_tree_weight_sum/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_tree_weight_sum(_, _, _)),
    retractall(user:typr_forest_weight_sum(_, _, _)).

test(mixed_tree_list_mutual_boolean_branch_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_tree_ok_branch([])),
    assertz(user:(typr_tree_ok_branch([V, L, R]) :-
        ( V > 0 -> typr_forest_ok_branch([L, R]) ; typr_forest_ok_branch([R, L]) )
    )),
    assertz(user:typr_forest_ok_branch([])),
    assertz(user:(typr_forest_ok_branch([T|Ts]) :-
        typr_tree_ok_branch(T),
        typr_forest_ok_branch(Ts)
    )),
    assertz(type_declarations:uw_type(typr_tree_ok_branch/1, 1, list(any))),
    assertz(type_declarations:uw_type(typr_forest_ok_branch/1, 1, list(any))),
    assertz(type_declarations:uw_return_type(typr_tree_ok_branch/1, bool)),
    assertz(type_declarations:uw_return_type(typr_forest_ok_branch/1, bool)),
    once(recursive_compiler:compile_recursive(typr_tree_ok_branch/1, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_tree_ok_branch(_)),
    retractall(user:typr_forest_ok_branch(_)).

test(mixed_tree_list_mutual_integer_branch_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_tree_sum_branch([], 0)),
    assertz(user:(typr_tree_sum_branch([V, L, R], S) :-
        ( V > 0 -> typr_forest_sum_branch([L, R], Parts) ; typr_forest_sum_branch([R, L], Parts) ),
        S is V + Parts
    )),
    assertz(user:typr_forest_sum_branch([], 0)),
    assertz(user:(typr_forest_sum_branch([T|Ts], S) :-
        typr_tree_sum_branch(T, ST),
        typr_forest_sum_branch(Ts, SS),
        S is ST + SS
    )),
    assertz(type_declarations:uw_type(typr_tree_sum_branch/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_tree_sum_branch/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_forest_sum_branch/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_forest_sum_branch/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_tree_sum_branch/2, integer)),
    assertz(type_declarations:uw_return_type(typr_forest_sum_branch/2, integer)),
    once(recursive_compiler:compile_recursive(typr_tree_sum_branch/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_tree_sum_branch(_, _)),
    retractall(user:typr_forest_sum_branch(_, _)).

test(mixed_tree_list_mutual_integer_context_branch_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_tree_weight_sum_branch([], _W0, 0)),
    assertz(user:(typr_tree_weight_sum_branch([V, L, R], W, S) :-
        ( V > W -> typr_forest_weight_sum_branch([L, R], W, Parts) ; typr_forest_weight_sum_branch([R, L], W, Parts) ),
        S is (V * W) + Parts
    )),
    assertz(user:typr_forest_weight_sum_branch([], _W1, 0)),
    assertz(user:(typr_forest_weight_sum_branch([T|Ts], W, S) :-
        typr_tree_weight_sum_branch(T, W, ST),
        typr_forest_weight_sum_branch(Ts, W, SS),
        S is ST + SS
    )),
    assertz(type_declarations:uw_type(typr_tree_weight_sum_branch/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_tree_weight_sum_branch/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_tree_weight_sum_branch/3, 3, integer)),
    assertz(type_declarations:uw_type(typr_forest_weight_sum_branch/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_forest_weight_sum_branch/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_forest_weight_sum_branch/3, 3, integer)),
    assertz(type_declarations:uw_return_type(typr_tree_weight_sum_branch/3, integer)),
    assertz(type_declarations:uw_return_type(typr_forest_weight_sum_branch/3, integer)),
    once(recursive_compiler:compile_recursive(typr_tree_weight_sum_branch/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_tree_weight_sum_branch(_, _, _)),
    retractall(user:typr_forest_weight_sum_branch(_, _, _)).

test(mixed_list_numeric_mutual_boolean_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_list_num_ok([])),
    assertz(user:(typr_list_num_ok([N|Ns]) :-
        typr_num_list_ok(N),
        typr_list_num_ok(Ns)
    )),
    assertz(user:typr_num_list_ok(0)),
    assertz(user:(typr_num_list_ok(N) :-
        N > 0,
        N1 is N - 1,
        typr_list_num_ok([N1])
    )),
    assertz(type_declarations:uw_type(typr_list_num_ok/1, 1, list(any))),
    assertz(type_declarations:uw_type(typr_num_list_ok/1, 1, integer)),
    assertz(type_declarations:uw_return_type(typr_list_num_ok/1, bool)),
    assertz(type_declarations:uw_return_type(typr_num_list_ok/1, bool)),
    once(recursive_compiler:compile_recursive(typr_list_num_ok/1, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_list_num_ok(_)),
    retractall(user:typr_num_list_ok(_)).

test(mixed_list_numeric_mutual_integer_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_sum_list_num([], 0)),
    assertz(user:(typr_mutual_sum_list_num([N|Ns], S) :-
        typr_mutual_sum_num_list(N, SN),
        typr_mutual_sum_list_num(Ns, SS),
        S is SN + SS
    )),
    assertz(user:typr_mutual_sum_num_list(0, 0)),
    assertz(user:(typr_mutual_sum_num_list(N, S) :-
        N > 0,
        N1 is N - 1,
        typr_mutual_sum_list_num([N1], Parts),
        S is N + Parts
    )),
    assertz(type_declarations:uw_type(typr_mutual_sum_list_num/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_sum_list_num/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_sum_num_list/2, 1, integer)),
    assertz(type_declarations:uw_type(typr_mutual_sum_num_list/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_sum_list_num/2, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_sum_num_list/2, integer)),
    once(recursive_compiler:compile_recursive(typr_mutual_sum_list_num/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_sum_list_num(_, _)),
    retractall(user:typr_mutual_sum_num_list(_, _)).

test(mixed_list_numeric_mutual_integer_context_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_mutual_weight_list_num([], _W0, 0)),
    assertz(user:(typr_mutual_weight_list_num([N|Ns], W, S) :-
        typr_mutual_weight_num_list(N, W, SN),
        typr_mutual_weight_list_num(Ns, W, SS),
        S is SN + SS
    )),
    assertz(user:typr_mutual_weight_num_list(0, _W1, 0)),
    assertz(user:(typr_mutual_weight_num_list(N, W, S) :-
        N > 0,
        N1 is N - 1,
        typr_mutual_weight_list_num([N1], W, Parts),
        S is (N * W) + Parts
    )),
    assertz(type_declarations:uw_type(typr_mutual_weight_list_num/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_mutual_weight_list_num/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_weight_list_num/3, 3, integer)),
    assertz(type_declarations:uw_type(typr_mutual_weight_num_list/3, 1, integer)),
    assertz(type_declarations:uw_type(typr_mutual_weight_num_list/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_mutual_weight_num_list/3, 3, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_weight_list_num/3, integer)),
    assertz(type_declarations:uw_return_type(typr_mutual_weight_num_list/3, integer)),
    once(recursive_compiler:compile_recursive(typr_mutual_weight_list_num/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_mutual_weight_list_num(_, _, _)),
    retractall(user:typr_mutual_weight_num_list(_, _, _)).

test(mixed_list_numeric_pair_tail_mutual_boolean_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_pair_tail_list_num_ok([])),
    assertz(user:(typr_pair_tail_list_num_ok([A, B|Ts]) :-
        typr_pair_tail_num_list_ok(A),
        typr_pair_tail_num_list_ok(B),
        typr_pair_tail_list_num_ok(Ts)
    )),
    assertz(user:typr_pair_tail_num_list_ok(0)),
    assertz(user:typr_pair_tail_num_list_ok(1)),
    assertz(user:(typr_pair_tail_num_list_ok(N) :-
        N > 1,
        N1 is N - 1,
        N2 is N - 2,
        typr_pair_tail_list_num_ok([N1, N2])
    )),
    assertz(type_declarations:uw_type(typr_pair_tail_list_num_ok/1, 1, list(any))),
    assertz(type_declarations:uw_type(typr_pair_tail_num_list_ok/1, 1, integer)),
    assertz(type_declarations:uw_return_type(typr_pair_tail_list_num_ok/1, bool)),
    assertz(type_declarations:uw_return_type(typr_pair_tail_num_list_ok/1, bool)),
    once(recursive_compiler:compile_recursive(typr_pair_tail_list_num_ok/1, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_pair_tail_list_num_ok(_)),
    retractall(user:typr_pair_tail_num_list_ok(_)).

test(mixed_list_numeric_pair_tail_mutual_integer_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_pair_tail_list_num_sum([], 0)),
    assertz(user:(typr_pair_tail_list_num_sum([A, B|Ts], S) :-
        typr_pair_tail_num_list_sum(A, SA),
        typr_pair_tail_num_list_sum(B, SB),
        typr_pair_tail_list_num_sum(Ts, ST),
        S is SA + SB + ST
    )),
    assertz(user:typr_pair_tail_num_list_sum(0, 0)),
    assertz(user:typr_pair_tail_num_list_sum(1, 1)),
    assertz(user:(typr_pair_tail_num_list_sum(N, S) :-
        N > 1,
        N1 is N - 1,
        N2 is N - 2,
        typr_pair_tail_list_num_sum([N1, N2], Parts),
        S is N + Parts
    )),
    assertz(type_declarations:uw_type(typr_pair_tail_list_num_sum/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_pair_tail_list_num_sum/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_pair_tail_num_list_sum/2, 1, integer)),
    assertz(type_declarations:uw_type(typr_pair_tail_num_list_sum/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_pair_tail_list_num_sum/2, integer)),
    assertz(type_declarations:uw_return_type(typr_pair_tail_num_list_sum/2, integer)),
    once(recursive_compiler:compile_recursive(typr_pair_tail_list_num_sum/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_pair_tail_list_num_sum(_, _)),
    retractall(user:typr_pair_tail_num_list_sum(_, _)).

test(mixed_list_numeric_pair_tail_mutual_integer_context_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_pair_tail_list_num_weight([], _W0, 0)),
    assertz(user:(typr_pair_tail_list_num_weight([A, B|Ts], W, S) :-
        typr_pair_tail_num_list_weight(A, W, SA),
        typr_pair_tail_num_list_weight(B, W, SB),
        typr_pair_tail_list_num_weight(Ts, W, ST),
        S is SA + SB + ST
    )),
    assertz(user:typr_pair_tail_num_list_weight(0, _W1, 0)),
    assertz(user:typr_pair_tail_num_list_weight(1, W, W)),
    assertz(user:(typr_pair_tail_num_list_weight(N, W, S) :-
        N > 1,
        N1 is N - 1,
        N2 is N - 2,
        typr_pair_tail_list_num_weight([N1, N2], W, Parts),
        S is (N * W) + Parts
    )),
    assertz(type_declarations:uw_type(typr_pair_tail_list_num_weight/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_pair_tail_list_num_weight/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_pair_tail_list_num_weight/3, 3, integer)),
    assertz(type_declarations:uw_type(typr_pair_tail_num_list_weight/3, 1, integer)),
    assertz(type_declarations:uw_type(typr_pair_tail_num_list_weight/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_pair_tail_num_list_weight/3, 3, integer)),
    assertz(type_declarations:uw_return_type(typr_pair_tail_list_num_weight/3, integer)),
    assertz(type_declarations:uw_return_type(typr_pair_tail_num_list_weight/3, integer)),
    once(recursive_compiler:compile_recursive(typr_pair_tail_list_num_weight/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_pair_tail_list_num_weight(_, _, _)),
    retractall(user:typr_pair_tail_num_list_weight(_, _, _)).

test(mixed_tree_numeric_mutual_boolean_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_tree_num_ok([])),
    assertz(user:(typr_tree_num_ok([V, L, _]) :-
        typr_num_tree_ok(V),
        typr_tree_num_ok(L)
    )),
    assertz(user:typr_num_tree_ok(0)),
    assertz(user:(typr_num_tree_ok(N) :-
        N > 0,
        N1 is N - 1,
        typr_tree_num_ok([N1, [], []])
    )),
    assertz(type_declarations:uw_type(typr_tree_num_ok/1, 1, list(any))),
    assertz(type_declarations:uw_type(typr_num_tree_ok/1, 1, integer)),
    assertz(type_declarations:uw_return_type(typr_tree_num_ok/1, bool)),
    assertz(type_declarations:uw_return_type(typr_num_tree_ok/1, bool)),
    once(recursive_compiler:compile_recursive(typr_tree_num_ok/1, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_tree_num_ok(_)),
    retractall(user:typr_num_tree_ok(_)).

test(mixed_tree_numeric_mutual_integer_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_tree_num_sum([], 0)),
    assertz(user:(typr_tree_num_sum([V, L, _], S) :-
        typr_num_tree_sum(V, SV),
        typr_tree_num_sum(L, SL),
        S is SV + SL
    )),
    assertz(user:typr_num_tree_sum(0, 0)),
    assertz(user:(typr_num_tree_sum(N, S) :-
        N > 0,
        N1 is N - 1,
        typr_tree_num_sum([N1, [], []], Parts),
        S is N + Parts
    )),
    assertz(type_declarations:uw_type(typr_tree_num_sum/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_tree_num_sum/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_num_tree_sum/2, 1, integer)),
    assertz(type_declarations:uw_type(typr_num_tree_sum/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_tree_num_sum/2, integer)),
    assertz(type_declarations:uw_return_type(typr_num_tree_sum/2, integer)),
    once(recursive_compiler:compile_recursive(typr_tree_num_sum/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_tree_num_sum(_, _)),
    retractall(user:typr_num_tree_sum(_, _)).

test(mixed_tree_numeric_mutual_integer_context_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_tree_num_weight([], _W0, 0)),
    assertz(user:(typr_tree_num_weight([V, L, _], W, S) :-
        typr_num_tree_weight(V, W, SV),
        typr_tree_num_weight(L, W, SL),
        S is SV + SL
    )),
    assertz(user:typr_num_tree_weight(0, _W1, 0)),
    assertz(user:(typr_num_tree_weight(N, W, S) :-
        N > 0,
        N1 is N - 1,
        typr_tree_num_weight([N1, [], []], W, Parts),
        S is (N * W) + Parts
    )),
    assertz(type_declarations:uw_type(typr_tree_num_weight/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_tree_num_weight/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_tree_num_weight/3, 3, integer)),
    assertz(type_declarations:uw_type(typr_num_tree_weight/3, 1, integer)),
    assertz(type_declarations:uw_type(typr_num_tree_weight/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_num_tree_weight/3, 3, integer)),
    assertz(type_declarations:uw_return_type(typr_tree_num_weight/3, integer)),
    assertz(type_declarations:uw_return_type(typr_num_tree_weight/3, integer)),
    once(recursive_compiler:compile_recursive(typr_tree_num_weight/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_tree_num_weight(_, _, _)),
    retractall(user:typr_num_tree_weight(_, _, _)).

test(mixed_tree_numeric_mutual_boolean_helper_output_checks_with_typr, [condition(typr_cli_available)]) :-
    setup_call_cleanup(
        setup_typr_mixed_tree_numeric_helper_toolchain_state,
        (
            assertz(user:typr_tree_num_ok_helper([])),
            assertz(user:(typr_tree_num_ok_helper([V, L, R]) :-
                typr_num_tree_ok_helper(V),
                typr_pick_tree_helper(V, L, R, T),
                typr_tree_num_ok_helper(T)
            )),
            assertz(user:typr_num_tree_ok_helper(0)),
            assertz(user:(typr_num_tree_ok_helper(N) :-
                N > 0,
                N1 is N - 1,
                typr_tree_num_ok_helper([N1, [], []])
            )),
            assertz(user:(typr_pick_tree_helper(V, L, R, T) :-
                ( V > 0 -> T = L ; T = R )
            )),
            assertz(type_declarations:uw_type(typr_tree_num_ok_helper/1, 1, list(any))),
            assertz(type_declarations:uw_type(typr_num_tree_ok_helper/1, 1, integer)),
            assertz(type_declarations:uw_return_type(typr_tree_num_ok_helper/1, bool)),
            assertz(type_declarations:uw_return_type(typr_num_tree_ok_helper/1, bool)),
            once(recursive_compiler:compile_recursive(typr_tree_num_ok_helper/1, [target(typr), typed_mode(explicit)], Code)),
            setup_call_cleanup(
                create_smoke_project(ProjectDir),
                (
                    write_generated_typr_program(ProjectDir, Code),
                    run_typr(ProjectDir, ['check']),
                    maybe_build_with_r(ProjectDir)
                ),
                delete_directory_and_contents(ProjectDir)
            )
        ),
        cleanup_typr_mixed_tree_numeric_helper_toolchain_state
    ).

test(mixed_tree_numeric_mutual_integer_helper_output_checks_with_typr, [condition(typr_cli_available)]) :-
    setup_call_cleanup(
        setup_typr_mixed_tree_numeric_helper_toolchain_state,
        (
            assertz(user:typr_tree_num_sum_helper([], 0)),
            assertz(user:(typr_tree_num_sum_helper([V, L, R], S) :-
                typr_num_tree_sum_helper(V, SV),
                typr_pick_tree_helper(V, L, R, T),
                typr_tree_num_sum_helper(T, ST),
                S is SV + ST
            )),
            assertz(user:typr_num_tree_sum_helper(0, 0)),
            assertz(user:(typr_num_tree_sum_helper(N, S) :-
                N > 0,
                N1 is N - 1,
                typr_tree_num_sum_helper([N1, [], []], Parts),
                S is N + Parts
            )),
            assertz(user:(typr_pick_tree_helper(V, L, R, T) :-
                ( V > 0 -> T = L ; T = R )
            )),
            assertz(type_declarations:uw_type(typr_tree_num_sum_helper/2, 1, list(any))),
            assertz(type_declarations:uw_type(typr_tree_num_sum_helper/2, 2, integer)),
            assertz(type_declarations:uw_type(typr_num_tree_sum_helper/2, 1, integer)),
            assertz(type_declarations:uw_type(typr_num_tree_sum_helper/2, 2, integer)),
            assertz(type_declarations:uw_return_type(typr_tree_num_sum_helper/2, integer)),
            assertz(type_declarations:uw_return_type(typr_num_tree_sum_helper/2, integer)),
            once(recursive_compiler:compile_recursive(typr_tree_num_sum_helper/2, [target(typr), typed_mode(explicit)], Code)),
            setup_call_cleanup(
                create_smoke_project(ProjectDir),
                (
                    write_generated_typr_program(ProjectDir, Code),
                    run_typr(ProjectDir, ['check']),
                    maybe_build_with_r(ProjectDir)
                ),
                delete_directory_and_contents(ProjectDir)
            )
        ),
        cleanup_typr_mixed_tree_numeric_helper_toolchain_state
    ).

test(mixed_tree_numeric_mutual_integer_context_helper_output_checks_with_typr, [condition(typr_cli_available)]) :-
    setup_call_cleanup(
        setup_typr_mixed_tree_numeric_helper_toolchain_state,
        (
            assertz(user:typr_tree_num_weight_helper([], _W0, 0)),
            assertz(user:(typr_tree_num_weight_helper([V, L, R], W, S) :-
                typr_num_tree_weight_helper(V, W, SV),
                typr_pick_tree_helper(V, L, R, T),
                typr_tree_num_weight_helper(T, W, ST),
                S is SV + ST
            )),
            assertz(user:typr_num_tree_weight_helper(0, _W1, 0)),
            assertz(user:(typr_num_tree_weight_helper(N, W, S) :-
                N > 0,
                N1 is N - 1,
                typr_tree_num_weight_helper([N1, [], []], W, Parts),
                S is (N * W) + Parts
            )),
            assertz(user:(typr_pick_tree_helper(V, L, R, T) :-
                ( V > 0 -> T = L ; T = R )
            )),
            assertz(type_declarations:uw_type(typr_tree_num_weight_helper/3, 1, list(any))),
            assertz(type_declarations:uw_type(typr_tree_num_weight_helper/3, 2, integer)),
            assertz(type_declarations:uw_type(typr_tree_num_weight_helper/3, 3, integer)),
            assertz(type_declarations:uw_type(typr_num_tree_weight_helper/3, 1, integer)),
            assertz(type_declarations:uw_type(typr_num_tree_weight_helper/3, 2, integer)),
            assertz(type_declarations:uw_type(typr_num_tree_weight_helper/3, 3, integer)),
            assertz(type_declarations:uw_return_type(typr_tree_num_weight_helper/3, integer)),
            assertz(type_declarations:uw_return_type(typr_num_tree_weight_helper/3, integer)),
            once(recursive_compiler:compile_recursive(typr_tree_num_weight_helper/3, [target(typr), typed_mode(explicit)], Code)),
            setup_call_cleanup(
                create_smoke_project(ProjectDir),
                (
                    write_generated_typr_program(ProjectDir, Code),
                    run_typr(ProjectDir, ['check']),
                    maybe_build_with_r(ProjectDir)
                ),
                delete_directory_and_contents(ProjectDir)
            )
        ),
        cleanup_typr_mixed_tree_numeric_helper_toolchain_state
    ).

test(mixed_tree_list_numeric_mutual_boolean_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_tree_forest_num_ok([])),
    assertz(user:(typr_tree_forest_num_ok([V, L, R]) :-
        typr_num_forest_tree_ok(V),
        typr_forest_tree_num_ok([L, R])
    )),
    assertz(user:typr_forest_tree_num_ok([])),
    assertz(user:(typr_forest_tree_num_ok([T|Ts]) :-
        typr_tree_forest_num_ok(T),
        typr_forest_tree_num_ok(Ts)
    )),
    assertz(user:typr_num_forest_tree_ok(0)),
    assertz(user:(typr_num_forest_tree_ok(N) :-
        N > 0,
        N1 is N - 1,
        typr_tree_forest_num_ok([N1, [], []])
    )),
    assertz(type_declarations:uw_type(typr_tree_forest_num_ok/1, 1, list(any))),
    assertz(type_declarations:uw_type(typr_forest_tree_num_ok/1, 1, list(any))),
    assertz(type_declarations:uw_type(typr_num_forest_tree_ok/1, 1, integer)),
    assertz(type_declarations:uw_return_type(typr_tree_forest_num_ok/1, bool)),
    assertz(type_declarations:uw_return_type(typr_forest_tree_num_ok/1, bool)),
    assertz(type_declarations:uw_return_type(typr_num_forest_tree_ok/1, bool)),
    once(recursive_compiler:compile_recursive(typr_tree_forest_num_ok/1, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_tree_forest_num_ok(_)),
    retractall(user:typr_forest_tree_num_ok(_)),
    retractall(user:typr_num_forest_tree_ok(_)).

test(mixed_tree_list_numeric_mutual_integer_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_tree_forest_num_sum([], 0)),
    assertz(user:(typr_tree_forest_num_sum([V, L, R], S) :-
        typr_num_forest_tree_sum(V, SV),
        typr_forest_tree_num_sum([L, R], Parts),
        S is SV + Parts
    )),
    assertz(user:typr_forest_tree_num_sum([], 0)),
    assertz(user:(typr_forest_tree_num_sum([T|Ts], S) :-
        typr_tree_forest_num_sum(T, ST),
        typr_forest_tree_num_sum(Ts, SS),
        S is ST + SS
    )),
    assertz(user:typr_num_forest_tree_sum(0, 0)),
    assertz(user:(typr_num_forest_tree_sum(N, S) :-
        N > 0,
        N1 is N - 1,
        typr_tree_forest_num_sum([N1, [], []], Parts),
        S is N + Parts
    )),
    assertz(type_declarations:uw_type(typr_tree_forest_num_sum/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_tree_forest_num_sum/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_forest_tree_num_sum/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_forest_tree_num_sum/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_num_forest_tree_sum/2, 1, integer)),
    assertz(type_declarations:uw_type(typr_num_forest_tree_sum/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_tree_forest_num_sum/2, integer)),
    assertz(type_declarations:uw_return_type(typr_forest_tree_num_sum/2, integer)),
    assertz(type_declarations:uw_return_type(typr_num_forest_tree_sum/2, integer)),
    once(recursive_compiler:compile_recursive(typr_tree_forest_num_sum/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_tree_forest_num_sum(_, _)),
    retractall(user:typr_forest_tree_num_sum(_, _)),
    retractall(user:typr_num_forest_tree_sum(_, _)).

test(mixed_tree_list_numeric_mutual_integer_context_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_tree_forest_num_weight([], _W0, 0)),
    assertz(user:(typr_tree_forest_num_weight([V, L, R], W, S) :-
        typr_num_forest_tree_weight(V, W, SV),
        typr_forest_tree_num_weight([L, R], W, Parts),
        S is SV + Parts
    )),
    assertz(user:typr_forest_tree_num_weight([], _W1, 0)),
    assertz(user:(typr_forest_tree_num_weight([T|Ts], W, S) :-
        typr_tree_forest_num_weight(T, W, ST),
        typr_forest_tree_num_weight(Ts, W, SS),
        S is ST + SS
    )),
    assertz(user:typr_num_forest_tree_weight(0, _W2, 0)),
    assertz(user:(typr_num_forest_tree_weight(N, W, S) :-
        N > 0,
        N1 is N - 1,
        typr_tree_forest_num_weight([N1, [], []], W, Parts),
        S is (N * W) + Parts
    )),
    assertz(type_declarations:uw_type(typr_tree_forest_num_weight/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_tree_forest_num_weight/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_tree_forest_num_weight/3, 3, integer)),
    assertz(type_declarations:uw_type(typr_forest_tree_num_weight/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_forest_tree_num_weight/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_forest_tree_num_weight/3, 3, integer)),
    assertz(type_declarations:uw_type(typr_num_forest_tree_weight/3, 1, integer)),
    assertz(type_declarations:uw_type(typr_num_forest_tree_weight/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_num_forest_tree_weight/3, 3, integer)),
    assertz(type_declarations:uw_return_type(typr_tree_forest_num_weight/3, integer)),
    assertz(type_declarations:uw_return_type(typr_forest_tree_num_weight/3, integer)),
    assertz(type_declarations:uw_return_type(typr_num_forest_tree_weight/3, integer)),
    once(recursive_compiler:compile_recursive(typr_tree_forest_num_weight/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_tree_forest_num_weight(_, _, _)),
    retractall(user:typr_forest_tree_num_weight(_, _, _)),
    retractall(user:typr_num_forest_tree_weight(_, _, _)).

test(mixed_tree_numeric_mutual_boolean_branch_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_tree_num_ok_branch([])),
    assertz(user:(typr_tree_num_ok_branch([V, L, R]) :-
        typr_num_tree_ok_branch(V),
        ( V > 0 -> typr_tree_num_ok_branch(L) ; typr_tree_num_ok_branch(R) )
    )),
    assertz(user:typr_num_tree_ok_branch(0)),
    assertz(user:(typr_num_tree_ok_branch(N) :-
        N > 0,
        N1 is N - 1,
        typr_tree_num_ok_branch([N1, [], []])
    )),
    assertz(type_declarations:uw_type(typr_tree_num_ok_branch/1, 1, list(any))),
    assertz(type_declarations:uw_type(typr_num_tree_ok_branch/1, 1, integer)),
    assertz(type_declarations:uw_return_type(typr_tree_num_ok_branch/1, bool)),
    assertz(type_declarations:uw_return_type(typr_num_tree_ok_branch/1, bool)),
    once(recursive_compiler:compile_recursive(typr_tree_num_ok_branch/1, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_tree_num_ok_branch(_)),
    retractall(user:typr_num_tree_ok_branch(_)).

test(mixed_tree_numeric_mutual_integer_branch_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_tree_num_sum_branch([], 0)),
    assertz(user:(typr_tree_num_sum_branch([V, L, R], S) :-
        typr_num_tree_sum_branch(V, SV),
        ( V > 0 -> typr_tree_num_sum_branch(L, SL) ; typr_tree_num_sum_branch(R, SL) ),
        S is SV + SL
    )),
    assertz(user:typr_num_tree_sum_branch(0, 0)),
    assertz(user:(typr_num_tree_sum_branch(N, S) :-
        N > 0,
        N1 is N - 1,
        typr_tree_num_sum_branch([N1, [], []], Parts),
        S is N + Parts
    )),
    assertz(type_declarations:uw_type(typr_tree_num_sum_branch/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_tree_num_sum_branch/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_num_tree_sum_branch/2, 1, integer)),
    assertz(type_declarations:uw_type(typr_num_tree_sum_branch/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_tree_num_sum_branch/2, integer)),
    assertz(type_declarations:uw_return_type(typr_num_tree_sum_branch/2, integer)),
    once(recursive_compiler:compile_recursive(typr_tree_num_sum_branch/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_tree_num_sum_branch(_, _)),
    retractall(user:typr_num_tree_sum_branch(_, _)).

test(mixed_tree_list_numeric_mutual_integer_context_branch_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_tree_forest_num_weight_branch([], _W0, 0)),
    assertz(user:(typr_tree_forest_num_weight_branch([V, L, R], W, S) :-
        typr_num_forest_tree_weight_branch(V, W, SV),
        ( V > W -> typr_forest_tree_num_weight_branch([L, R], W, Parts)
        ; typr_forest_tree_num_weight_branch([R, L], W, Parts)
        ),
        S is SV + Parts
    )),
    assertz(user:typr_forest_tree_num_weight_branch([], _W1, 0)),
    assertz(user:(typr_forest_tree_num_weight_branch([T|Ts], W, S) :-
        typr_tree_forest_num_weight_branch(T, W, ST),
        typr_forest_tree_num_weight_branch(Ts, W, SS),
        S is ST + SS
    )),
    assertz(user:typr_num_forest_tree_weight_branch(0, _W2, 0)),
    assertz(user:(typr_num_forest_tree_weight_branch(N, W, S) :-
        N > 0,
        N1 is N - 1,
        typr_tree_forest_num_weight_branch([N1, [], []], W, Parts),
        S is (N * W) + Parts
    )),
    assertz(type_declarations:uw_type(typr_tree_forest_num_weight_branch/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_tree_forest_num_weight_branch/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_tree_forest_num_weight_branch/3, 3, integer)),
    assertz(type_declarations:uw_type(typr_forest_tree_num_weight_branch/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_forest_tree_num_weight_branch/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_forest_tree_num_weight_branch/3, 3, integer)),
    assertz(type_declarations:uw_type(typr_num_forest_tree_weight_branch/3, 1, integer)),
    assertz(type_declarations:uw_type(typr_num_forest_tree_weight_branch/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_num_forest_tree_weight_branch/3, 3, integer)),
    assertz(type_declarations:uw_return_type(typr_tree_forest_num_weight_branch/3, integer)),
    assertz(type_declarations:uw_return_type(typr_forest_tree_num_weight_branch/3, integer)),
    assertz(type_declarations:uw_return_type(typr_num_forest_tree_weight_branch/3, integer)),
    once(recursive_compiler:compile_recursive(typr_tree_forest_num_weight_branch/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_tree_forest_num_weight_branch(_, _, _)),
    retractall(user:typr_forest_tree_num_weight_branch(_, _, _)),
    retractall(user:typr_num_forest_tree_weight_branch(_, _, _)).

test(tree_pair_mutual_boolean_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_tree_pair_ok([])),
    assertz(user:(typr_tree_pair_ok([_, L, R]) :-
        typr_forest_pair_ok([L, R])
    )),
    assertz(user:typr_forest_pair_ok([])),
    assertz(user:(typr_forest_pair_ok([L, R]) :-
        typr_tree_pair_ok(L),
        typr_tree_pair_ok(R)
    )),
    assertz(type_declarations:uw_type(typr_tree_pair_ok/1, 1, list(any))),
    assertz(type_declarations:uw_type(typr_forest_pair_ok/1, 1, list(any))),
    assertz(type_declarations:uw_return_type(typr_tree_pair_ok/1, bool)),
    assertz(type_declarations:uw_return_type(typr_forest_pair_ok/1, bool)),
    once(recursive_compiler:compile_recursive(typr_tree_pair_ok/1, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_tree_pair_ok(_)),
    retractall(user:typr_forest_pair_ok(_)).

test(tree_pair_mutual_integer_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_tree_pair_sum([], 0)),
    assertz(user:(typr_tree_pair_sum([V, L, R], S) :-
        typr_forest_pair_sum([L, R], Parts),
        S is V + Parts
    )),
    assertz(user:typr_forest_pair_sum([], 0)),
    assertz(user:(typr_forest_pair_sum([L, R], S) :-
        typr_tree_pair_sum(L, SL),
        typr_tree_pair_sum(R, SR),
        S is SL + SR
    )),
    assertz(type_declarations:uw_type(typr_tree_pair_sum/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_tree_pair_sum/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_forest_pair_sum/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_forest_pair_sum/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_tree_pair_sum/2, integer)),
    assertz(type_declarations:uw_return_type(typr_forest_pair_sum/2, integer)),
    once(recursive_compiler:compile_recursive(typr_tree_pair_sum/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_tree_pair_sum(_, _)),
    retractall(user:typr_forest_pair_sum(_, _)).

test(tree_pair_mutual_integer_context_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_tree_pair_weight([], _W0, 0)),
    assertz(user:(typr_tree_pair_weight([V, L, R], W, S) :-
        typr_forest_pair_weight([L, R], W, Parts),
        S is (V * W) + Parts
    )),
    assertz(user:typr_forest_pair_weight([], _W1, 0)),
    assertz(user:(typr_forest_pair_weight([L, R], W, S) :-
        typr_tree_pair_weight(L, W, SL),
        typr_tree_pair_weight(R, W, SR),
        S is SL + SR
    )),
    assertz(type_declarations:uw_type(typr_tree_pair_weight/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_tree_pair_weight/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_tree_pair_weight/3, 3, integer)),
    assertz(type_declarations:uw_type(typr_forest_pair_weight/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_forest_pair_weight/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_forest_pair_weight/3, 3, integer)),
    assertz(type_declarations:uw_return_type(typr_tree_pair_weight/3, integer)),
    assertz(type_declarations:uw_return_type(typr_forest_pair_weight/3, integer)),
    once(recursive_compiler:compile_recursive(typr_tree_pair_weight/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_tree_pair_weight(_, _, _)),
    retractall(user:typr_forest_pair_weight(_, _, _)).

test(tree_pair_tail_mutual_boolean_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_tree_pair_tail_ok([])),
    assertz(user:(typr_tree_pair_tail_ok([_, L, R]) :-
        typr_forest_pair_tail_ok([L, R])
    )),
    assertz(user:typr_forest_pair_tail_ok([])),
    assertz(user:(typr_forest_pair_tail_ok([A, B|Ts]) :-
        typr_tree_pair_tail_ok(A),
        typr_tree_pair_tail_ok(B),
        typr_forest_pair_tail_ok(Ts)
    )),
    assertz(type_declarations:uw_type(typr_tree_pair_tail_ok/1, 1, list(any))),
    assertz(type_declarations:uw_type(typr_forest_pair_tail_ok/1, 1, list(any))),
    assertz(type_declarations:uw_return_type(typr_tree_pair_tail_ok/1, bool)),
    assertz(type_declarations:uw_return_type(typr_forest_pair_tail_ok/1, bool)),
    once(recursive_compiler:compile_recursive(typr_tree_pair_tail_ok/1, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_tree_pair_tail_ok(_)),
    retractall(user:typr_forest_pair_tail_ok(_)).

test(tree_pair_tail_mutual_integer_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_tree_pair_tail_sum([], 0)),
    assertz(user:(typr_tree_pair_tail_sum([V, L, R], S) :-
        typr_forest_pair_tail_sum([L, R], Parts),
        S is V + Parts
    )),
    assertz(user:typr_forest_pair_tail_sum([], 0)),
    assertz(user:(typr_forest_pair_tail_sum([A, B|Ts], S) :-
        typr_tree_pair_tail_sum(A, SA),
        typr_tree_pair_tail_sum(B, SB),
        typr_forest_pair_tail_sum(Ts, ST),
        S is SA + SB + ST
    )),
    assertz(type_declarations:uw_type(typr_tree_pair_tail_sum/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_tree_pair_tail_sum/2, 2, integer)),
    assertz(type_declarations:uw_type(typr_forest_pair_tail_sum/2, 1, list(any))),
    assertz(type_declarations:uw_type(typr_forest_pair_tail_sum/2, 2, integer)),
    assertz(type_declarations:uw_return_type(typr_tree_pair_tail_sum/2, integer)),
    assertz(type_declarations:uw_return_type(typr_forest_pair_tail_sum/2, integer)),
    once(recursive_compiler:compile_recursive(typr_tree_pair_tail_sum/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_tree_pair_tail_sum(_, _)),
    retractall(user:typr_forest_pair_tail_sum(_, _)).

test(tree_pair_tail_mutual_integer_context_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:typr_tree_pair_tail_weight([], _W0, 0)),
    assertz(user:(typr_tree_pair_tail_weight([V, L, R], W, S) :-
        typr_forest_pair_tail_weight([L, R], W, Parts),
        S is (V * W) + Parts
    )),
    assertz(user:typr_forest_pair_tail_weight([], _W1, 0)),
    assertz(user:(typr_forest_pair_tail_weight([A, B|Ts], W, S) :-
        typr_tree_pair_tail_weight(A, W, SA),
        typr_tree_pair_tail_weight(B, W, SB),
        typr_forest_pair_tail_weight(Ts, W, ST),
        S is SA + SB + ST
    )),
    assertz(type_declarations:uw_type(typr_tree_pair_tail_weight/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_tree_pair_tail_weight/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_tree_pair_tail_weight/3, 3, integer)),
    assertz(type_declarations:uw_type(typr_forest_pair_tail_weight/3, 1, list(any))),
    assertz(type_declarations:uw_type(typr_forest_pair_tail_weight/3, 2, integer)),
    assertz(type_declarations:uw_type(typr_forest_pair_tail_weight/3, 3, integer)),
    assertz(type_declarations:uw_return_type(typr_tree_pair_tail_weight/3, integer)),
    assertz(type_declarations:uw_return_type(typr_forest_pair_tail_weight/3, integer)),
    once(recursive_compiler:compile_recursive(typr_tree_pair_tail_weight/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:typr_tree_pair_tail_weight(_, _, _)),
    retractall(user:typr_forest_pair_tail_weight(_, _, _)).

test(per_path_visited_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:category_parent(a, b)),
    assertz(user:category_parent(b, c)),
    assertz(user:category_parent(c, a)),
    assertz(user:category_parent(b, d)),
    assertz(user:(category_ancestor(Cat, Parent, 1, Visited) :-
        category_parent(Cat, Parent),
        \+ member(Parent, Visited)
    )),
    assertz(user:(category_ancestor(Cat, Ancestor, Hops, Visited) :-
        category_parent(Cat, Mid),
        \+ member(Mid, Visited),
        category_ancestor(Mid, Ancestor, H1, [Mid|Visited]),
        Hops is H1 + 1
    )),
    assertz(type_declarations:uw_type(category_parent/2, 1, atom)),
    assertz(type_declarations:uw_type(category_parent/2, 2, atom)),
    once(recursive_compiler:compile_recursive(category_ancestor/4, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:category_parent(_, _)),
    retractall(user:category_ancestor(_, _, _, _)).

test(weighted_per_path_visited_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:category_parent(a, b)),
    assertz(user:category_parent(b, c)),
    assertz(user:category_parent(c, a)),
    assertz(user:category_parent(b, d)),
    assertz(user:(category_ancestor_weight(Cat, Parent, 1, 10, Visited) :-
        category_parent(Cat, Parent),
        \+ member(Parent, Visited)
    )),
    assertz(user:(category_ancestor_weight(Cat, Ancestor, Hops, Cost, Visited) :-
        category_parent(Cat, Mid),
        \+ member(Mid, Visited),
        category_ancestor_weight(Mid, Ancestor, H1, Cost1, [Mid|Visited]),
        Hops is H1 + 1,
        Cost is Cost1 + 10
    )),
    assertz(type_declarations:uw_type(category_parent/2, 1, atom)),
    assertz(type_declarations:uw_type(category_parent/2, 2, atom)),
    once(recursive_compiler:compile_recursive(category_ancestor_weight/5, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:category_parent(_, _)),
    retractall(user:category_ancestor_weight(_, _, _, _, _)).

test(mode_driven_per_path_visited_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:category_parent(a, b)),
    assertz(user:category_parent(b, c)),
    assertz(user:category_parent(c, a)),
    assertz(user:category_parent(b, d)),
    assertz(user:mode(category_ancestor_from_end(-, +, -, +))),
    assertz(user:(category_ancestor_from_end(Parent, Cat, 1, Visited) :-
        category_parent(Cat, Parent),
        \+ member(Parent, Visited)
    )),
    assertz(user:(category_ancestor_from_end(Ancestor, Cat, Hops, Visited) :-
        category_parent(Cat, Mid),
        \+ member(Mid, Visited),
        category_ancestor_from_end(Ancestor, Mid, H1, [Mid|Visited]),
        Hops is H1 + 1
    )),
    assertz(type_declarations:uw_type(category_parent/2, 1, atom)),
    assertz(type_declarations:uw_type(category_parent/2, 2, atom)),
    once(recursive_compiler:compile_recursive(category_ancestor_from_end/4, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:category_parent(_, _)),
    retractall(user:category_ancestor_from_end(_, _, _, _)),
    retractall(user:mode(category_ancestor_from_end(_, _, _, _))).

test(mode_driven_weighted_per_path_visited_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:category_parent(a, b)),
    assertz(user:category_parent(b, c)),
    assertz(user:category_parent(c, a)),
    assertz(user:category_parent(b, d)),
    assertz(user:mode(category_ancestor_weight_from_end(-, +, -, -, +))),
    assertz(user:(category_ancestor_weight_from_end(Parent, Cat, 1, 10, Visited) :-
        category_parent(Cat, Parent),
        \+ member(Parent, Visited)
    )),
    assertz(user:(category_ancestor_weight_from_end(Ancestor, Cat, Hops, Cost, Visited) :-
        category_parent(Cat, Mid),
        \+ member(Mid, Visited),
        category_ancestor_weight_from_end(Ancestor, Mid, H1, Cost1, [Mid|Visited]),
        Hops is H1 + 1,
        Cost is Cost1 + 10
    )),
    assertz(type_declarations:uw_type(category_parent/2, 1, atom)),
    assertz(type_declarations:uw_type(category_parent/2, 2, atom)),
    once(recursive_compiler:compile_recursive(category_ancestor_weight_from_end/5, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:category_parent(_, _)),
    retractall(user:category_ancestor_weight_from_end(_, _, _, _, _)),
    retractall(user:mode(category_ancestor_weight_from_end(_, _, _, _, _))).

test(per_path_visited_stdin_input_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:(category_ancestor(Cat, Parent, 1, Visited) :-
        category_parent(Cat, Parent),
        \+ member(Parent, Visited)
    )),
    assertz(user:(category_ancestor(Cat, Ancestor, Hops, Visited) :-
        category_parent(Cat, Mid),
        \+ member(Mid, Visited),
        category_ancestor(Mid, Ancestor, H1, [Mid|Visited]),
        Hops is H1 + 1
    )),
    assertz(type_declarations:uw_type(category_parent/2, 1, atom)),
    assertz(type_declarations:uw_type(category_parent/2, 2, atom)),
    once(recursive_compiler:compile_recursive(category_ancestor/4, [target(typr), typed_mode(explicit), input(stdin)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:category_ancestor(_, _, _, _)).

test(per_path_visited_file_input_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:(category_ancestor(Cat, Parent, 1, Visited) :-
        category_parent(Cat, Parent),
        \+ member(Parent, Visited)
    )),
    assertz(user:(category_ancestor(Cat, Ancestor, Hops, Visited) :-
        category_parent(Cat, Mid),
        \+ member(Mid, Visited),
        category_ancestor(Mid, Ancestor, H1, [Mid|Visited]),
        Hops is H1 + 1
    )),
    assertz(type_declarations:uw_type(category_parent/2, 1, atom)),
    assertz(type_declarations:uw_type(category_parent/2, 2, atom)),
    once(recursive_compiler:compile_recursive(category_ancestor/4, [target(typr), typed_mode(explicit), input(file("data.txt"))], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:category_ancestor(_, _, _, _)).

test(per_path_visited_vfs_input_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:(category_ancestor(Cat, Parent, 1, Visited) :-
        category_parent(Cat, Parent),
        \+ member(Parent, Visited)
    )),
    assertz(user:(category_ancestor(Cat, Ancestor, Hops, Visited) :-
        category_parent(Cat, Mid),
        \+ member(Mid, Visited),
        category_ancestor(Mid, Ancestor, H1, [Mid|Visited]),
        Hops is H1 + 1
    )),
    assertz(type_declarations:uw_type(category_parent/2, 1, atom)),
    assertz(type_declarations:uw_type(category_parent/2, 2, atom)),
    once(recursive_compiler:compile_recursive(category_ancestor/4, [target(typr), typed_mode(explicit), input(vfs("family_tree"))], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check'])
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:category_ancestor(_, _, _, _)).

test(per_path_visited_function_input_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:(category_ancestor(Cat, Parent, 1, Visited) :-
        category_parent(Cat, Parent),
        \+ member(Parent, Visited)
    )),
    assertz(user:(category_ancestor(Cat, Ancestor, Hops, Visited) :-
        category_parent(Cat, Mid),
        \+ member(Mid, Visited),
        category_ancestor(Mid, Ancestor, H1, [Mid|Visited]),
        Hops is H1 + 1
    )),
    assertz(type_declarations:uw_type(category_parent/2, 1, atom)),
    assertz(type_declarations:uw_type(category_parent/2, 2, atom)),
    once(recursive_compiler:compile_recursive(category_ancestor/4, [target(typr), typed_mode(explicit), input(function)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:category_ancestor(_, _, _, _)).

test(per_path_visited_integer_stdin_loader_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    retractall(user:category_parent(_, _)),
    assertz(user:(category_ancestor(Cat, Parent, 1, Visited) :-
        category_parent(Cat, Parent),
        \+ member(Parent, Visited)
    )),
    assertz(user:(category_ancestor(Cat, Ancestor, Hops, Visited) :-
        category_parent(Cat, Mid),
        \+ member(Mid, Visited),
        category_ancestor(Mid, Ancestor, H1, [Mid|Visited]),
        Hops is H1 + 1
    )),
    assertz(type_declarations:uw_type(category_parent/2, 1, integer)),
    assertz(type_declarations:uw_type(category_parent/2, 2, integer)),
    once(recursive_compiler:compile_recursive(category_ancestor/4, [target(typr), typed_mode(explicit), input(stdin)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:category_ancestor(_, _, _, _)),
    retractall(user:category_parent(_, _)).

test(per_path_visited_number_stdin_loader_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    retractall(user:category_parent(_, _)),
    assertz(user:(category_ancestor(Cat, Parent, 1, Visited) :-
        category_parent(Cat, Parent),
        \+ member(Parent, Visited)
    )),
    assertz(user:(category_ancestor(Cat, Ancestor, Hops, Visited) :-
        category_parent(Cat, Mid),
        \+ member(Mid, Visited),
        category_ancestor(Mid, Ancestor, H1, [Mid|Visited]),
        Hops is H1 + 1
    )),
    assertz(type_declarations:uw_type(category_parent/2, 1, number)),
    assertz(type_declarations:uw_type(category_parent/2, 2, number)),
    once(recursive_compiler:compile_recursive(category_ancestor/4, [target(typr), typed_mode(explicit), input(stdin)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:category_ancestor(_, _, _, _)),
    retractall(user:category_parent(_, _)).

test(per_path_visited_pair_integer_stdin_loader_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    retractall(user:category_parent(_, _)),
    assertz(user:(category_ancestor(Cat, Parent, 1, Visited) :-
        category_parent(Cat, Parent),
        \+ member(Parent, Visited)
    )),
    assertz(user:(category_ancestor(Cat, Ancestor, Hops, Visited) :-
        category_parent(Cat, Mid),
        \+ member(Mid, Visited),
        category_ancestor(Mid, Ancestor, H1, [Mid|Visited]),
        Hops is H1 + 1
    )),
    assertz(type_declarations:uw_type(category_parent/2, 1, pair(integer, integer))),
    assertz(type_declarations:uw_type(category_parent/2, 2, pair(integer, integer))),
    once(recursive_compiler:compile_recursive(category_ancestor/4, [target(typr), typed_mode(explicit), input(stdin)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:category_ancestor(_, _, _, _)),
    retractall(user:category_parent(_, _)).

test(per_path_visited_pair_number_vfs_loader_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    retractall(user:category_parent(_, _)),
    assertz(user:(category_ancestor(Cat, Parent, 1, Visited) :-
        category_parent(Cat, Parent),
        \+ member(Parent, Visited)
    )),
    assertz(user:(category_ancestor(Cat, Ancestor, Hops, Visited) :-
        category_parent(Cat, Mid),
        \+ member(Mid, Visited),
        category_ancestor(Mid, Ancestor, H1, [Mid|Visited]),
        Hops is H1 + 1
    )),
    assertz(type_declarations:uw_type(category_parent/2, 1, pair(number, number))),
    assertz(type_declarations:uw_type(category_parent/2, 2, pair(number, number))),
    once(recursive_compiler:compile_recursive(category_ancestor/4, [target(typr), typed_mode(explicit), input(vfs("family_tree"))], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:category_ancestor(_, _, _, _)),
    retractall(user:category_parent(_, _)).

test(invariant_per_path_visited_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    retractall(user:category_parent(_, _)),
    retractall(user:category_ancestor_limited(_, _, _, _, _)),
    retractall(user:mode(category_ancestor_limited(_, _, _, _, _))),
    assertz(user:category_parent(a, b)),
    assertz(user:category_parent(b, c)),
    assertz(user:mode(category_ancestor_limited(+, +, -, -, +))),
    assertz(user:(category_ancestor_limited(Cat, Limit, Parent, 1, Visited) :-
        category_parent(Cat, Parent),
        \+ member(Parent, Visited)
    )),
    assertz(user:(category_ancestor_limited(Cat, Limit, Ancestor, Hops, Visited) :-
        category_parent(Cat, Mid),
        \+ member(Mid, Visited),
        category_ancestor_limited(Mid, Limit, Ancestor, H1, [Mid|Visited]),
        Hops is H1 + 1
    )),
    assertz(type_declarations:uw_type(category_parent/2, 1, atom)),
    assertz(type_declarations:uw_type(category_parent/2, 2, atom)),
    once(recursive_compiler:compile_recursive(category_ancestor_limited/5, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:category_parent(_, _)),
    retractall(user:category_ancestor_limited(_, _, _, _, _)),
    retractall(user:mode(category_ancestor_limited(_, _, _, _, _))).

test(weighted_invariant_per_path_visited_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    retractall(user:category_parent(_, _)),
    retractall(user:category_ancestor_limit_weight(_, _, _, _, _, _)),
    retractall(user:mode(category_ancestor_limit_weight(_, _, _, _, _, _))),
    assertz(user:category_parent(a, b)),
    assertz(user:category_parent(b, c)),
    assertz(user:mode(category_ancestor_limit_weight(+, +, -, -, -, +))),
    assertz(user:(category_ancestor_limit_weight(Cat, Limit, Parent, 1, 10, Visited) :-
        category_parent(Cat, Parent),
        \+ member(Parent, Visited)
    )),
    assertz(user:(category_ancestor_limit_weight(Cat, Limit, Ancestor, Hops, Cost, Visited) :-
        category_parent(Cat, Mid),
        \+ member(Mid, Visited),
        category_ancestor_limit_weight(Mid, Limit, Ancestor, H1, Cost1, [Mid|Visited]),
        Hops is H1 + 1,
        Cost is Cost1 + 10
    )),
    assertz(type_declarations:uw_type(category_parent/2, 1, atom)),
    assertz(type_declarations:uw_type(category_parent/2, 2, atom)),
    once(recursive_compiler:compile_recursive(category_ancestor_limit_weight/6, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:category_parent(_, _)),
    retractall(user:category_ancestor_limit_weight(_, _, _, _, _, _)),
    retractall(user:mode(category_ancestor_limit_weight(_, _, _, _, _, _))).

test(invariant_per_path_visited_function_input_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    retractall(user:category_parent(_, _)),
    retractall(user:category_ancestor_limited_fn(_, _, _, _, _)),
    retractall(user:mode(category_ancestor_limited_fn(_, _, _, _, _))),
    assertz(user:mode(category_ancestor_limited_fn(+, +, -, -, +))),
    assertz(user:(category_ancestor_limited_fn(Cat, Limit, Parent, 1, Visited) :-
        category_parent(Cat, Parent),
        \+ member(Parent, Visited)
    )),
    assertz(user:(category_ancestor_limited_fn(Cat, Limit, Ancestor, Hops, Visited) :-
        category_parent(Cat, Mid),
        \+ member(Mid, Visited),
        category_ancestor_limited_fn(Mid, Limit, Ancestor, H1, [Mid|Visited]),
        Hops is H1 + 1
    )),
    assertz(type_declarations:uw_type(category_parent/2, 1, atom)),
    assertz(type_declarations:uw_type(category_parent/2, 2, atom)),
    once(recursive_compiler:compile_recursive(category_ancestor_limited_fn/5, [target(typr), typed_mode(explicit), input(function)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:category_parent(_, _)),
    retractall(user:category_ancestor_limited_fn(_, _, _, _, _)),
    retractall(user:mode(category_ancestor_limited_fn(_, _, _, _, _))).

:- end_tests(typr_toolchain).

typr_cli_available :-
    process_create(path(sh), ['-c', 'command -v typr >/dev/null 2>&1'], [process(Pid)]),
    process_wait(Pid, exit(0)).

create_smoke_project(ProjectDir) :-
    tmp_file(typr_smoke, RootDir),
    make_directory(RootDir),
    run_typr(RootDir, ['new', 'smoke_project']),
    directory_file_path(RootDir, 'smoke_project', ProjectDir),
    exists_directory(ProjectDir).

write_generated_typr_program(ProjectDir, Code) :-
    directory_file_path(ProjectDir, 'TypR/main.ty', MainFile),
    setup_call_cleanup(
        open(MainFile, write, Stream),
        write(Stream, Code),
        close(Stream)
    ).

maybe_build_with_r(ProjectDir) :-
    (   rscript_available
    ->  run_typr(ProjectDir, ['build'])
    ;   true
    ).

rscript_available :-
    process_create(path(sh), ['-c', 'command -v Rscript >/dev/null 2>&1'], [process(Pid)]),
    process_wait(Pid, exit(0)).

run_typr(ProjectDir, Args) :-
    process_create(
        path(typr),
        Args,
        [ cwd(ProjectDir),
          stdout(pipe(Stdout)),
          stderr(pipe(Stderr)),
          process(Pid)
        ]
    ),
    read_string(Stdout, _, _),
    read_string(Stderr, _, _),
    close(Stdout),
    close(Stderr),
    process_wait(Pid, exit(0)).
