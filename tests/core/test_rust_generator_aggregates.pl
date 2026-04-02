:- module(test_rust_generator_aggregates, [test_rust_generator_aggregates/0]).

:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/rust_target').

setup_count_example :-
    cleanup_all,
    assertz(user:item(a, 10)),
    assertz(user:item(b, 20)),
    assertz(user:item(c, 30)),
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
    catch(abolish(user:dept_total/2), _, true),
    catch(abolish(user:r_edge/2), _, true),
    catch(abolish(user:r_reach/3), _, true),
    catch(abolish(user:r_count/2), _, true),
    catch(abolish(user:wa_edge/2), _, true),
    catch(abolish(user:wa_cost/2), _, true),
    catch(abolish(user:wa_path/3), _, true),
    catch(abolish(user:wa_sum/2), _, true).

:- begin_tests(rust_generator_aggregates).

test(compile_count_aggregate, [
    setup(setup_count_example),
    cleanup(cleanup_all)
]) :-
    once(rust_target:compile_predicate_to_rust_normal(item_count, 1, [include_main(false)], Code)),
    sub_string(Code, _, _, _, "fn item_count() -> usize"),
    sub_string(Code, _, _, _, "vec![(); 3]"),
    sub_string(Code, _, _, _, "agg += 1;").

test(compile_sum_aggregate, [
    setup(setup_sum_example),
    cleanup(cleanup_all)
]) :-
    once(rust_target:compile_predicate_to_rust_normal(total_sales, 1, [include_main(false)], Code)),
    sub_string(Code, _, _, _, "fn total_sales() -> f64"),
    sub_string(Code, _, _, _, "100_f64"),
    sub_string(Code, _, _, _, "agg += *value;").

test(compile_grouped_aggregate, [
    setup(setup_grouped_sum_example),
    cleanup(cleanup_all)
]) :-
    once(rust_target:compile_predicate_to_rust_normal(dept_total, 2, [include_main(false)], Code)),
    sub_string(Code, _, _, _, "fn dept_total() -> HashMap<String, f64>"),
    sub_string(Code, _, _, _, "let mut groups: HashMap<String, f64> = HashMap::new();"),
    sub_string(Code, _, _, _, "\"eng\".to_string()"),
    sub_string(Code, _, _, _, "\"sales\".to_string()"),
    sub_string(Code, _, _, _, "*entry += value;").

test(compile_recursive_count_aggregate, [
    cleanup(cleanup_all)
]) :-
    assertz(user:r_edge(a, b)),
    assertz(user:r_edge(b, c)),
    assertz(user:(r_reach(X, Y, H) :- r_edge(X, Y), H is 1)),
    assertz(user:(r_reach(X, Z, H) :- r_edge(X, Y), r_reach(Y, Z, H1), H is H1 + 1)),
    assertz(user:(r_count(X, N) :- aggregate_all(count, r_reach(X, _, _), N))),
    once(rust_target:compile_predicate_to_rust_normal(r_count, 2, [include_main(false)], Code)),
    sub_string(Code, _, _, _, "aggregate_all over recursive predicate"),
    sub_string(Code, _, _, _, "fn r_reach_worker"),
    sub_string(Code, _, _, _, "fn r_count(arg1: &str) -> usize"),
    sub_string(Code, _, _, _, "agg += 1;").

test(compile_recursive_weighted_sum_aggregate, [
    cleanup(cleanup_all)
]) :-
    assertz(user:wa_edge(a, b)),
    assertz(user:wa_edge(b, c)),
    assertz(user:wa_cost(a, 2)),
    assertz(user:wa_cost(b, 5)),
    assertz(user:(wa_path(X, Y, Acc) :- wa_edge(X, Y), wa_cost(X, Cost), Acc is Cost)),
    assertz(user:(wa_path(X, Z, Acc) :- wa_edge(X, Y), wa_cost(X, Cost), wa_path(Y, Z, PrevAcc), Acc is PrevAcc + Cost)),
    assertz(user:(wa_sum(X, Total) :- aggregate_all(sum(Acc), wa_path(X, _, Acc), Total))),
    once(rust_target:compile_predicate_to_rust_normal(wa_sum, 2, [include_main(false)], Code)),
    sub_string(Code, _, _, _, "aggregate_all over recursive accumulation predicate"),
    sub_string(Code, _, _, _, "fn wa_path_worker"),
    sub_string(Code, _, _, _, "fn wa_sum(arg1: &str) -> f64"),
    sub_string(Code, _, _, _, "agg += (row.1 as f64);").

:- end_tests(rust_generator_aggregates).

test_rust_generator_aggregates :-
    run_tests([rust_generator_aggregates]).
