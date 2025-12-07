:- module(test_go_generator_aggregates, [
    run_tests/0
]).

:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/go_target.pl').

%% Setup test predicates
setup_count_example :-
    cleanup_all,
    % Items to count
    assertz(user:item(a, 10)),
    assertz(user:item(b, 20)),
    assertz(user:item(c, 30)),
    % Count rule
    assertz(user:(item_count(N) :- aggregate_all(count, item(_, _), N))).

setup_sum_example :-
    cleanup_all,
    assertz(user:sale(100)),
    assertz(user:sale(200)),
    assertz(user:sale(50)),
    assertz(user:(total_sales(T) :- aggregate_all(sum(X), sale(X), T))).

setup_grouped_sum_example :-
    cleanup_all,
    assertz(user:salary(eng, 1000)),
    assertz(user:salary(sales, 2000)),
    assertz(user:salary(eng, 1500)),
    assertz(user:salary(sales, 500)),
    assertz(user:(dept_total(Dept, Total) :- aggregate_all(sum(S), salary(Dept, S), Dept, Total))).

cleanup_all :-
    catch(abolish(user:item/2), _, true),
    catch(abolish(user:item_count/1), _, true),
    catch(abolish(user:sale/1), _, true),
    catch(abolish(user:total_sales/1), _, true),
    catch(abolish(user:salary/2), _, true),
    catch(abolish(user:dept_total/2), _, true).

:- begin_tests(go_generator_aggregates).

test(compile_count_aggregate, [
    setup(setup_count_example),
    cleanup(cleanup_all)
]) :-
    compile_predicate_to_go(item_count/1, [mode(generator)], Code),
    % Check that code contains aggregate elements
    sub_string(Code, _, _, _, "values []float64"),
    sub_string(Code, _, _, _, "len(values)"),
    sub_string(Code, _, _, _, "toFloat64"),
    format('~n=== Count Aggregate Code ===~n~w~n', [Code]).

test(compile_sum_aggregate, [
    setup(setup_sum_example),
    cleanup(cleanup_all)
]) :-
    compile_predicate_to_go(total_sales/1, [mode(generator)], Code),
    % Check sum aggregate
    sub_string(Code, _, _, _, "agg += v"),
    format('~n=== Sum Aggregate Code ===~n~w~n', [Code]).

test(compile_grouped_aggregate, [
    setup(setup_grouped_sum_example),
    cleanup(cleanup_all)
]) :-
    compile_predicate_to_go(dept_total/2, [mode(generator)], Code),
    % Check grouped aggregate elements
    sub_string(Code, _, _, _, "groups := make(map[interface{}][]float64)"),
    sub_string(Code, _, _, _, "for key, values := range groups"),
    format('~n=== Grouped Aggregate Code ===~n~w~n', [Code]).

:- end_tests(go_generator_aggregates).

run_tests :-
    run_tests(go_generator_aggregates).
