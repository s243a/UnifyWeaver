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

setup_grouped_count_example :-
    cleanup_all,
    assertz(user:salary(eng, 1000)),
    assertz(user:salary(sales, 2000)),
    assertz(user:salary(eng, 1500)),
    assertz(user:(dept_count(Dept, Count) :- aggregate_all(count, salary(Dept, _), Dept, Count))).

setup_grouped_recursive_count_example :-
    cleanup_all,
    assertz(user:r_edge(a, b)),
    assertz(user:r_edge(a, c)),
    assertz(user:r_edge(b, c)),
    assertz(user:(r_reach(X, Y, H) :- r_edge(X, Y), H is 1)),
    assertz(user:(r_reach(X, Z, H) :- r_edge(X, Y), r_reach(Y, Z, H1), H is H1 + 1)),
    assertz(user:(r_count_by_target(X, Y, N) :- aggregate_all(count, r_reach(X, Y, _), Y, N))).

setup_grouped_recursive_weighted_sum_example :-
    cleanup_all,
    assertz(user:wa_edge(a, b)),
    assertz(user:wa_edge(a, c)),
    assertz(user:wa_edge(b, c)),
    assertz(user:wa_cost(a, 2)),
    assertz(user:wa_cost(b, 5)),
    assertz(user:wa_cost(c, 7)),
    assertz(user:(wa_path(X, Y, Acc) :- wa_edge(X, Y), wa_cost(X, Cost), Acc is Cost)),
    assertz(user:(wa_path(X, Z, Acc) :- wa_edge(X, Y), wa_cost(X, Cost), wa_path(Y, Z, PrevAcc), Acc is PrevAcc + Cost)),
    assertz(user:(wa_sum_by_target(X, Y, Total) :- aggregate_all(sum(Acc), wa_path(X, Y, Acc), Y, Total))).

cleanup_all :-
    catch(abolish(user:item/2), _, true),
    catch(abolish(user:item_count/1), _, true),
    catch(abolish(user:sale/1), _, true),
    catch(abolish(user:total_sales/1), _, true),
    catch(abolish(user:salary/2), _, true),
    catch(abolish(user:dept_total/2), _, true),
    catch(abolish(user:dept_count/2), _, true),
    catch(abolish(user:r_edge/2), _, true),
    catch(abolish(user:r_reach/3), _, true),
    catch(abolish(user:r_count_by_target/3), _, true),
    catch(abolish(user:wa_edge/2), _, true),
    catch(abolish(user:wa_cost/2), _, true),
    catch(abolish(user:wa_path/3), _, true),
    catch(abolish(user:wa_sum_by_target/3), _, true).

:- begin_tests(go_generator_aggregates).

test(compile_count_aggregate, [
    setup(setup_count_example),
    cleanup(cleanup_all)
]) :-
    compile_predicate_to_go(item_count/1, [mode(generator)], Code),
    sub_string(Code, _, _, _, "count := 0.0"),
    sub_string(Code, _, _, _, "if f.Relation == \"item\""),
    sub_string(Code, _, _, _, "count += 1.0").

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

test(compile_grouped_count_aggregate, [
    setup(setup_grouped_count_example),
    cleanup(cleanup_all)
]) :-
    compile_predicate_to_go(dept_count/2, [mode(generator)], Code),
    sub_string(Code, _, _, _, "groups := make(map[interface{}]float64)"),
    sub_string(Code, _, _, _, "groups[key] += 1.0"),
    sub_string(Code, _, _, _, "\"arg1\": agg").

test(compile_grouped_recursive_count_aggregate, [
    setup(setup_grouped_recursive_count_example),
    cleanup(cleanup_all)
]) :-
    compile_predicate_to_go(r_count_by_target/3, [mode(generator)], Code),
    sub_string(Code, _, _, _, "groups := make(map[interface{}]float64)"),
    sub_string(Code, _, _, _, "if f.Relation == \"r_reach\""),
    sub_string(Code, _, _, _, "groups[key] += 1.0"),
    sub_string(Code, _, _, _, "r_count_by_target").

test(compile_grouped_recursive_weighted_sum_aggregate, [
    setup(setup_grouped_recursive_weighted_sum_example),
    cleanup(cleanup_all)
]) :-
    compile_predicate_to_go(wa_sum_by_target/3, [mode(generator)], Code),
    sub_string(Code, _, _, _, "groups := make(map[interface{}][]float64)"),
    sub_string(Code, _, _, _, "if f.Relation == \"wa_path\""),
    sub_string(Code, _, _, _, "val, ok := toFloat64(f.Args[\"arg2\"])"),
    sub_string(Code, _, _, _, "wa_sum_by_target").

:- end_tests(go_generator_aggregates).

run_tests :-
    run_tests(go_generator_aggregates).
