:- module(test_go_generator_aggregate_subplans, [
    test_go_generator_aggregate_subplans/0
]).

:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/go_target.pl').

setup_filtered_count_example :-
    cleanup_all,
    assertz(user:sale(alice, 100)),
    assertz(user:sale(bob, 250)),
    assertz(user:sale(carol, 175)),
    assertz(user:(high_sale_count(C) :- aggregate_all(count, (sale(_, A), A > 150), C))).

setup_grouped_filtered_sum_example :-
    cleanup_all,
    assertz(user:salary(eng, 1000)),
    assertz(user:salary(eng, 1500)),
    assertz(user:salary(sales, 500)),
    assertz(user:salary(sales, 2000)),
    assertz(user:(dept_big_total(Dept, Total) :- aggregate_all(sum(S), (salary(Dept, S), S > 1000), Dept, Total))).

setup_correlated_aggregate_example :-
    cleanup_all,
    assertz(user:base(x)),
    assertz(user:rel(x, 1)),
    assertz(user:(outer_sum(X, S) :- base(X), aggregate_all(sum(V), rel(X, V), S))).

cleanup_all :-
    catch(abolish(user:sale/2), _, true),
    catch(abolish(user:high_sale_count/1), _, true),
    catch(abolish(user:salary/2), _, true),
    catch(abolish(user:dept_big_total/2), _, true),
    catch(abolish(user:base/1), _, true),
    catch(abolish(user:rel/2), _, true),
    catch(abolish(user:outer_sum/2), _, true).

:- begin_tests(go_generator_aggregate_subplans).

test(compile_filtered_count_aggregate, [
    setup(setup_filtered_count_example),
    cleanup(cleanup_all)
]) :-
    compile_predicate_to_go(high_sale_count/1, [mode(generator)], Code),
    sub_string(Code, _, _, _, "toFloat64Must(f.Args[\"arg1\"]) > 150"),
    sub_string(Code, _, _, _, "count := 0.0"),
    sub_string(Code, _, _, _, "count += 1.0"),
    sub_string(Code, _, _, _, "high_sale_count").

test(compile_grouped_filtered_sum_aggregate, [
    setup(setup_grouped_filtered_sum_example),
    cleanup(cleanup_all)
]) :-
    compile_predicate_to_go(dept_big_total/2, [mode(generator)], Code),
    sub_string(Code, _, _, _, "toFloat64Must(f.Args[\"arg1\"]) > 1000"),
    sub_string(Code, _, _, _, "groups := make(map[interface{}][]float64)"),
    sub_string(Code, _, _, _, "dept_big_total").

test(reject_correlated_aggregate, [
    setup(setup_correlated_aggregate_example),
    cleanup(cleanup_all)
]) :-
    compile_predicate_to_go(outer_sum/2, [mode(generator)], Code),
    sub_string(Code, _, _, _, "Unsupported correlated aggregate form").

:- end_tests(go_generator_aggregate_subplans).

test_go_generator_aggregate_subplans :-
    run_tests([go_generator_aggregate_subplans]).
