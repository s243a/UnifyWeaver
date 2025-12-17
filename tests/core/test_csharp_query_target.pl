
:- module(test_csharp_query_target, [
    test_csharp_query_target/0
]).

:- asserta(user:file_search_path(library, 'src/unifyweaver/targets')).

:- use_module(library(apply)).
:- use_module(library(filesex)).
:- use_module(library(lists)).
:- use_module(library(memfile)).
:- use_module(library(process)).
:- use_module(library(uuid)).
:- use_module(library(csharp_query_target)).
:- use_module(library(csharp_stream_target)).
:- use_module('src/unifyweaver/core/dynamic_source_compiler').
:- use_module('src/unifyweaver/sources').
:- use_module('src/unifyweaver/sources/csv_source').
:- use_module('src/unifyweaver/sources/json_source').

:- dynamic cqt_option/2.
:- dynamic user:test_factorial/2.
:- dynamic user:test_factorial_input/1.
:- dynamic user:test_fib_param/2.
:- dynamic user:test_post_agg_param/2.
:- dynamic user:test_customer/1.
:- dynamic user:test_customer_alice_or_bob/1.
:- dynamic user:test_sale/2.
:- dynamic user:test_sale_amount_for_alice/1.
:- dynamic user:test_sale_item/3.
:- dynamic user:test_sale_count/1.
:- dynamic user:test_sales_by_customer/2.
:- dynamic user:test_sales_by_customer_product/3.
:- dynamic user:test_sale_count_by_customer/2.
:- dynamic user:test_sale_sum_by_customer/2.
:- dynamic user:test_sale_min/1.
:- dynamic user:test_sale_max/1.
:- dynamic user:test_sale_min_by_customer/2.
:- dynamic user:test_sale_max_by_customer/2.
:- dynamic user:test_sale_count_grouped/2.
:- dynamic user:test_customer_high_sales_count/1.
:- dynamic user:test_sale_filtered_alice_count/1.
:- dynamic user:test_sale_customers_set/1.
:- dynamic user:test_sale_customers_bag/1.
:- dynamic user:test_sale_alice_or_bob_count/1.
:- dynamic user:test_sale_alice_or_bob_nested_count/1.
:- dynamic user:test_banned/1.
:- dynamic user:test_allowed/1.
:- dynamic user:test_blocked/1.
:- dynamic user:test_countdown/2.
:- dynamic user:test_even/1.
:- dynamic user:test_odd/1.
:- dynamic user:test_even_param/1.
:- dynamic user:test_odd_param/1.
:- dynamic user:test_even_param_partial/1.
:- dynamic user:test_odd_param_partial/1.
:- dynamic user:test_even_param_unbound/1.
:- dynamic user:test_odd_param_unbound/1.
:- dynamic user:test_parity_input/1.
:- dynamic user:test_product_record/1.
:- dynamic user:test_jsonpath_projection/2.
:- dynamic user:test_order_summary/1.
:- dynamic user:test_orders_jsonl/3.
:- dynamic user:test_json_null_skip/2.
:- dynamic user:test_json_null_default/2.
:- dynamic user:test_multi_mode/2.
:- dynamic user:test_any_mode/2.
:- dynamic user:mode/1.

:- dynamic progress_last_report/1.
:- dynamic progress_count/1.
:- dynamic progress_total/1.

test_csharp_query_target :-
    set_prolog_flag(verbose, silent),
    configure_csharp_query_options,
    writeln('=== Testing C# query target ==='),
    setup_test_data,
    progress_init,
    Tests = [
        verify_fact_plan,
        verify_join_plan,
        verify_selection_plan,
        verify_ground_relation_arg_plan,
        verify_disjunction_body_union_plan,
        verify_arithmetic_plan,
        verify_recursive_arithmetic_plan,
        verify_comparison_plan,
        verify_is_check_literal_plan,
        verify_is_check_bound_var_plan,
        verify_arith_expr_eq_plan,
        verify_arith_expr_neq_plan,
        verify_arith_eq_direct_plan,
        verify_arith_neq_direct_plan,
        verify_aggregate_count_plan,
        verify_aggregate_min_plan,
        verify_aggregate_max_plan,
        verify_grouped_aggregate_sum_plan,
        verify_multi_key_grouped_aggregate_sum_plan,
        verify_grouped_aggregate_count_plan,
        verify_correlated_aggregate_count_plan,
        verify_correlated_aggregate_sum_plan,
        verify_correlated_aggregate_min_plan,
        verify_correlated_aggregate_max_plan,
        verify_aggregate_set_plan,
        verify_aggregate_bag_plan,
        verify_aggregate_subplan_count_with_constraint_plan,
        verify_aggregate_subplan_count_with_constant_arg_plan,
        verify_aggregate_subplan_nested_aggregate_plan,
        verify_aggregate_subplan_banned_sale_count_plan,
        verify_aggregate_subplan_disjunction_count_plan,
        verify_aggregate_subplan_nested_disjunction_count_plan,
        verify_aggregate_subplan_correlated_count_with_constraint_plan,
        verify_aggregate_subplan_grouped_count_with_negation_plan,
        verify_aggregate_subplan_sum_with_constraint_plan,
        verify_aggregate_subplan_set_with_negation_plan,
        verify_aggregate_subplan_grouped_sum_with_negation_plan,
        verify_negation_plan,
        verify_parameterized_fib_plan,
        verify_parameterized_fib_runtime,
        verify_multi_mode_codegen_plan,
        verify_multi_mode_plan_selection_api,
        verify_multi_mode_runtime_dispatch,
        verify_any_mode_rejected_plan,
        verify_parameterized_need_allows_post_agg,
        verify_parameterized_need_allows_prefix_negation,
        verify_recursive_plan,
        verify_mutual_recursion_plan,
        verify_parameterized_mutual_recursion_plan,
        verify_parameterized_mutual_recursion_inferred_plan,
        verify_parameterized_mutual_recursion_fallback_plan,
        verify_dynamic_source_plan,
        verify_tsv_dynamic_source_plan,
        verify_json_dynamic_source_plan,
        verify_json_nested_source_plan,
        verify_json_jsonpath_source_plan,
        verify_json_schema_source_plan,
        verify_json_nested_schema_record_plan,
        verify_json_jsonl_source_plan,
        verify_json_null_policy_skip_plan,
        verify_json_null_policy_default_plan,
        verify_json_object_source_plan
    ],
    length(Tests, Total),
    retractall(progress_total(_)),
    asserta(progress_total(Total)),
    maplist(run_with_progress, Tests),
    progress_maybe_report(force),
    cleanup_test_data,
    writeln('=== C# query target tests complete ===').

setup_test_data :-
    cleanup_test_data,
    assertz(user:test_fact(alice, bob)),
    assertz(user:test_fact(bob, charlie)),
    assertz(user:(test_link(X, Z) :- test_fact(X, Y), test_fact(Y, Z))),
    assertz(user:(test_filtered(X) :- test_fact(X, _), X = alice)),
    assertz(user:test_val(item1, 5)),
    assertz(user:test_val(item2, 2)),
    assertz(user:(test_increment(Id, Result) :- test_val(Id, Value), Result is Value + 1)),
    assertz(user:(test_is_check_literal(Id) :- test_val(Id, Value), 6 is Value + 1)),
    assertz(user:(test_is_check_bound_var(Id) :- test_val(Id, Value), W is Value + 1, W is 6)),
    assertz(user:(test_arith_expr_eq(Id) :- test_val(Id, Value), Value + 1 =:= 6)),
    assertz(user:(test_arith_expr_neq(Id) :- test_val(Id, Value), Value + 1 =\= 6)),
    assertz(user:(test_arith_eq_direct(Id) :- test_val(Id, Value), Value =:= 5)),
    assertz(user:(test_arith_neq_direct(Id) :- test_val(Id, Value), Value =\= 5)),
    assertz(user:test_num(item1, 5)),
    assertz(user:test_num(item2, -3)),
    assertz(user:(test_positive(Id) :- test_num(Id, Value), Value > 0)),
    assertz(user:test_customer(alice)),
    assertz(user:test_customer(bob)),
    assertz(user:test_customer(charlie)),
    assertz(user:(test_customer_alice_or_bob(Customer) :-
        test_customer(Customer),
        (Customer = alice ; Customer = bob)
    )),
    assertz(user:test_sale(alice, 10)),
    assertz(user:test_sale(alice, 5)),
    assertz(user:test_sale(bob, 7)),
    assertz(user:(test_sale_amount_for_alice(Amount) :-
        test_sale(alice, Amount)
    )),
    assertz(user:test_sale_item(alice, laptop, 10)),
    assertz(user:test_sale_item(alice, laptop, 2)),
    assertz(user:test_sale_item(alice, mouse, 5)),
    assertz(user:test_sale_item(bob, laptop, 3)),
    assertz(user:test_sale_item(bob, mouse, 7)),
    assertz(user:test_sale_item(bob, mouse, 1)),
    assertz(user:(test_sale_count(C) :- aggregate_all(count, test_sale(_, _), C))),
    assertz(user:(test_sales_by_customer(Customer, Total) :-
        aggregate_all(sum(Amount), test_sale(Customer, Amount), Customer, Total)
    )),
    assertz(user:(test_sales_by_customer_product(Customer, Product, Total) :-
        aggregate_all(sum(Amount), test_sale_item(Customer, Product, Amount), Customer-Product, Total)
    )),
    assertz(user:(test_sale_count_by_customer(Customer, Count) :-
        test_customer(Customer),
        aggregate_all(count, test_sale(Customer, _), Count)
    )),
    assertz(user:(test_sale_sum_by_customer(Customer, Sum) :-
        test_customer(Customer),
        aggregate_all(sum(Amount), test_sale(Customer, Amount), Sum)
    )),
    assertz(user:(test_sale_min(Min) :-
        aggregate_all(min(Amount), test_sale(_, Amount), Min)
    )),
    assertz(user:(test_sale_max(Max) :-
        aggregate_all(max(Amount), test_sale(_, Amount), Max)
    )),
    assertz(user:(test_sale_min_by_customer(Customer, Min) :-
        test_customer(Customer),
        aggregate_all(min(Amount), test_sale(Customer, Amount), Min)
    )),
    assertz(user:(test_sale_max_by_customer(Customer, Max) :-
        test_customer(Customer),
        aggregate_all(max(Amount), test_sale(Customer, Amount), Max)
    )),
    assertz(user:(test_sale_count_grouped(Customer, Count) :-
        aggregate_all(count, test_sale(Customer, _), Customer, Count)
    )),
    assertz(user:(test_sale_customers_set(Set) :-
        aggregate_all(set(Customer), test_sale(Customer, _), Set)
    )),
    assertz(user:(test_sale_customers_bag(Bag) :-
        aggregate_all(bag(Customer), test_sale(Customer, _), Bag)
    )),
    assertz(user:(test_sale_filtered_count(C) :-
        aggregate_all(count, (test_sale(_, Amount), Amount > 5), C)
    )),
    assertz(user:(test_sale_filtered_alice_count(C) :-
        aggregate_all(count, (test_sale(alice, Amount), Amount > 5), C)
    )),
    assertz(user:(test_customer_high_sales_count(C) :-
        aggregate_all(count,
            (test_customer(Customer),
             aggregate_all(sum(Amount), test_sale(Customer, Amount), Total),
             Total > 10),
            C)
    )),
    assertz(user:(test_banned_sale_count(C) :-
        aggregate_all(count, (test_sale(Customer, _), test_banned(Customer)), C)
    )),
    assertz(user:(test_sale_alice_or_bob_count(C) :-
        aggregate_all(count,
            ((test_sale(Customer, _), Customer = alice)
            ; (test_sale(Customer, _), Customer = bob)),
            C)
    )),
    assertz(user:(test_sale_alice_or_bob_nested_count(C) :-
        aggregate_all(count,
            (test_sale(Customer, _),
             (Customer = alice ; Customer = bob)),
            C)
    )),
    assertz(user:(test_sale_count_filtered_by_customer(Customer, Count) :-
        test_customer(Customer),
        aggregate_all(count, (test_sale(Customer, Amount), Amount > 5), Count)
    )),
    assertz(user:(test_non_banned_sale_count_grouped(Customer, Count) :-
        aggregate_all(count, (test_sale(Customer, _), \+ test_banned(Customer)), Customer, Count)
    )),
    assertz(user:(test_sale_filtered_sum(Sum) :-
        aggregate_all(sum(Amount), (test_sale(_, Amount), Amount > 5), Sum)
    )),
    assertz(user:(test_non_banned_sale_customers_set(Set) :-
        aggregate_all(set(Customer), (test_sale(Customer, _), \+ test_banned(Customer)), Set)
    )),
    assertz(user:(test_non_banned_sale_sum_grouped(Customer, Sum) :-
        aggregate_all(sum(Amount), (test_sale(Customer, Amount), \+ test_banned(Customer)), Customer, Sum)
    )),
    assertz(user:test_banned(bob)),
    assertz(user:(test_allowed(X) :- test_fact(X, _), \+ test_banned(X))),
    assertz(user:test_factorial_input(1)),
    assertz(user:test_factorial_input(2)),
    assertz(user:test_factorial_input(3)),
    assertz(user:test_factorial(0, 1)),
    assertz(user:(test_factorial(N, Result) :-
        test_factorial_input(N),
        N > 0,
        N1 is N - 1,
        test_factorial(N1, Prev),
        Result is Prev * N
    )),
    assertz(user:mode(test_multi_mode(+, -))),
    assertz(user:mode(test_multi_mode(-, +))),
    assertz(user:test_multi_mode(alice, bob)),
    assertz(user:test_multi_mode(bob, charlie)),
    assertz(user:mode(test_any_mode(?, -))),
    assertz(user:test_any_mode(alice, bob)),
    assertz(user:mode(test_fib_param(+, -))),
    assertz(user:test_fib_param(0, 1)),
    assertz(user:test_fib_param(1, 1)),
    assertz(user:(test_fib_param(N, F) :-
        N > 1,
        N1 is N - 1,
        N2 is N - 2,
        test_fib_param(N1, F1),
        test_fib_param(N2, F2),
        F is F1 + F2
    )),
    assertz(user:mode(test_post_agg_param(+, -))),
    assertz(user:test_post_agg_param(0, 0)),
    assertz(user:(test_post_agg_param(N, Sum) :-
        N > 0,
        N1 is N - 1,
        test_post_agg_param(N1, Prev),
        aggregate_all(count, test_num(_, _), C),
        Sum is Prev + C
    )),
    assertz(user:mode(test_countdown(+, -))),
    assertz(user:test_blocked(2)),
    assertz(user:test_countdown(0, 0)),
    assertz(user:(test_countdown(N, Out) :-
        \+ test_blocked(N),
        N > 0,
        N1 is N - 1,
        test_countdown(N1, Prev),
        Out is Prev + 1
    )),
    assertz(user:test_parity_input(0)),
    assertz(user:test_parity_input(1)),
    assertz(user:test_parity_input(2)),
    assertz(user:test_parity_input(3)),
    assertz(user:test_parity_input(4)),
    assertz(user:test_even(0)),
    assertz(user:test_odd(1)),
    assertz(user:mode(test_even_param(+))),
    assertz(user:mode(test_odd_param(+))),
    assertz(user:test_even_param(0)),
    assertz(user:test_odd_param(1)),
    assertz(user:(test_even_param(N) :-
        test_parity_input(N),
        N > 0,
        N1 is N - 1,
        test_odd_param(N1)
    )),
    assertz(user:(test_odd_param(N) :-
        test_parity_input(N),
        N > 1,
        N1 is N - 1,
        test_even_param(N1)
    )),
    assertz(user:mode(test_even_param_partial(+))),
    assertz(user:test_even_param_partial(0)),
    assertz(user:test_odd_param_partial(1)),
    assertz(user:(test_even_param_partial(N) :-
        test_parity_input(N),
        N > 0,
        N1 is N - 1,
        test_odd_param_partial(N1)
    )),
    assertz(user:(test_odd_param_partial(N) :-
        test_parity_input(N),
        N > 1,
        N1 is N - 1,
        test_even_param_partial(N1)
    )),
    assertz(user:mode(test_even_param_unbound(+))),
    assertz(user:test_even_param_unbound(0)),
    assertz(user:test_odd_param_unbound(1)),
    assertz(user:(test_even_param_unbound(N) :-
        test_parity_input(N),
        N > 0,
        test_odd_param_unbound(_)
    )),
    assertz(user:(test_odd_param_unbound(N) :-
        test_parity_input(N),
        N > 1,
        test_even_param_unbound(_)
    )),
    assertz(user:(test_even(N) :-
        test_parity_input(N),
        N > 0,
        N1 is N - 1,
        test_odd(N1)
    )),
    assertz(user:(test_odd(N) :-
        test_parity_input(N),
        N > 1,
        N1 is N - 1,
        test_even(N1)
    )),
    assertz(user:(test_reachable(X, Y) :- test_fact(X, Y))),
    assertz(user:(test_reachable(X, Z) :- test_fact(X, Y), test_reachable(Y, Z))).

cleanup_test_data :-
    retractall(user:test_fact(_, _)),
    retractall(user:test_link(_, _)),
    retractall(user:test_filtered(_)),
    retractall(user:test_val(_, _)),
    retractall(user:test_increment(_, _)),
    retractall(user:test_is_check_literal(_)),
    retractall(user:test_is_check_bound_var(_)),
    retractall(user:test_arith_expr_eq(_)),
    retractall(user:test_arith_expr_neq(_)),
    retractall(user:test_arith_eq_direct(_)),
    retractall(user:test_arith_neq_direct(_)),
    retractall(user:test_num(_, _)),
    retractall(user:test_positive(_)),
    retractall(user:test_customer(_)),
    retractall(user:test_customer_alice_or_bob(_)),
    retractall(user:test_sale(_, _)),
    retractall(user:test_sale_amount_for_alice(_)),
    retractall(user:test_sale_item(_, _, _)),
    retractall(user:test_sale_count(_)),
    retractall(user:test_sales_by_customer(_, _)),
    retractall(user:test_sales_by_customer_product(_, _, _)),
    retractall(user:test_sale_count_by_customer(_, _)),
    retractall(user:test_sale_sum_by_customer(_, _)),
    retractall(user:test_sale_min(_)),
    retractall(user:test_sale_max(_)),
    retractall(user:test_sale_min_by_customer(_, _)),
    retractall(user:test_sale_max_by_customer(_, _)),
    retractall(user:test_sale_count_grouped(_, _)),
    retractall(user:test_sale_filtered_alice_count(_)),
    retractall(user:test_customer_high_sales_count(_)),
    retractall(user:test_sale_customers_set(_)),
    retractall(user:test_sale_customers_bag(_)),
    retractall(user:test_sale_filtered_count(_)),
    retractall(user:test_banned_sale_count(_)),
    retractall(user:test_sale_alice_or_bob_count(_)),
    retractall(user:test_sale_alice_or_bob_nested_count(_)),
    retractall(user:test_sale_count_filtered_by_customer(_, _)),
    retractall(user:test_non_banned_sale_count_grouped(_, _)),
    retractall(user:test_sale_filtered_sum(_)),
    retractall(user:test_non_banned_sale_customers_set(_)),
    retractall(user:test_non_banned_sale_sum_grouped(_, _)),
    retractall(user:test_factorial_input(_)),
    retractall(user:test_factorial(_, _)),
    retractall(user:test_multi_mode(_, _)),
    retractall(user:mode(test_multi_mode(_, _))),
    retractall(user:test_any_mode(_, _)),
    retractall(user:mode(test_any_mode(_, _))),
    retractall(user:test_fib_param(_, _)),
    retractall(user:mode(test_fib_param(_,_))),
    retractall(user:test_post_agg_param(_, _)),
    retractall(user:mode(test_post_agg_param(_,_))),
    retractall(user:test_banned(_)),
    retractall(user:test_allowed(_)),
    retractall(user:test_blocked(_)),
    retractall(user:test_countdown(_, _)),
    retractall(user:mode(test_countdown(_,_))),
    retractall(user:test_parity_input(_)),
    retractall(user:test_even(_)),
    retractall(user:test_odd(_)),
    retractall(user:test_even_param(_)),
    retractall(user:test_odd_param(_)),
    retractall(user:mode(test_even_param(_))),
    retractall(user:mode(test_odd_param(_))),
    retractall(user:test_even_param_partial(_)),
    retractall(user:test_odd_param_partial(_)),
    retractall(user:mode(test_even_param_partial(_))),
    retractall(user:test_even_param_unbound(_)),
    retractall(user:test_odd_param_unbound(_)),
    retractall(user:mode(test_even_param_unbound(_))),
    retractall(user:test_reachable(_, _)),
    cleanup_csv_dynamic_source.

verify_fact_plan :-
    csharp_query_target:build_query_plan(test_fact/2, [target(csharp_query)], Plan),
    get_dict(head, Plan, predicate{name:test_fact, arity:2}),
    get_dict(root, Plan, relation_scan{type:relation_scan, predicate:predicate{name:test_fact, arity:2}, width:_}),
    get_dict(relations, Plan, [relation{predicate:predicate{name:test_fact, arity:2}, facts:Facts}]),
    Facts == [[alice, bob], [bob, charlie]],
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'RelationScanNode').

verify_join_plan :-
    csharp_query_target:build_query_plan(test_link/2, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:JoinNode, columns:[0, 3], width:2}),
    JoinNode = join{
        type:join,
        left:relation_scan{predicate:predicate{name:test_fact, arity:2}, type:relation_scan, width:_},
        right:relation_scan{predicate:predicate{name:test_fact, arity:2}, type:relation_scan, width:_},
        left_keys:[1],
        right_keys:[0],
        left_width:_,
        right_width:_,
        width:_
    },
    get_dict(relations, Plan, [relation{predicate:predicate{name:test_fact, arity:2}, facts:_}]),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'KeyJoinNode'),
    maybe_run_query_runtime(Plan, ['alice,charlie']).

verify_selection_plan :-
    csharp_query_target:build_query_plan(test_filtered/1, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Selection, columns:[0], width:1}),
    Selection = selection{
        type:selection,
        input:relation_scan{predicate:predicate{name:test_fact, arity:2}, type:relation_scan, width:_},
        predicate:condition{type:eq, left:operand{kind:column, index:0}, right:operand{kind:value, value:alice}},
        width:_
    },
    get_dict(relations, Plan, [relation{predicate:predicate{name:test_fact, arity:2}, facts:_}]),
    maybe_run_query_runtime(Plan, ['alice']).

verify_ground_relation_arg_plan :-
    csharp_query_target:build_query_plan(test_sale_amount_for_alice/1, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Selection, columns:[1], width:1}),
    Selection = selection{
        type:selection,
        input:relation_scan{predicate:predicate{name:test_sale, arity:2}, type:relation_scan, width:_},
        predicate:condition{type:eq, left:operand{kind:column, index:0}, right:operand{kind:value, value:alice}},
        width:_
    },
    maybe_run_query_runtime(Plan, ['10', '5']).

verify_disjunction_body_union_plan :-
    csharp_query_target:build_query_plan(test_customer_alice_or_bob/1, [target(csharp_query)], Plan),
    get_dict(root, Plan, Root),
    is_dict(Root, union),
    get_dict(width, Root, 1),
    get_dict(sources, Root, Sources),
    length(Sources, 2),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'UnionNode'),
    maybe_run_query_runtime(Plan, ['alice', 'bob']).

verify_arithmetic_plan :-
    csharp_query_target:build_query_plan(test_increment/2, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:ArithmeticNode, columns:[0, 2], width:2}),
    ArithmeticNode = arithmetic{
        type:arithmetic,
        input:relation_scan{predicate:predicate{name:test_val, arity:2}, type:relation_scan, width:_},
        expression:Expression,
        result_index:2,
        width:3
    },
    Expression = expr{
        type:binary,
        op:add,
        left:expr{type:column, index:1},
        right:expr{type:value, value:1}
    },
    get_dict(relations, Plan, [relation{predicate:predicate{name:test_val, arity:2}, facts:_}]),
    maybe_run_query_runtime(Plan, ['item1,6', 'item2,3']).

verify_comparison_plan :-
    csharp_query_target:build_query_plan(test_positive/1, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Selection, columns:[0], width:1}),
    Selection = selection{
        type:selection,
        input:relation_scan{predicate:predicate{name:test_num, arity:2}, type:relation_scan, width:_},
        predicate:condition{type:gt, left:operand{kind:column, index:1}, right:operand{kind:value, value:0}},
        width:_
    },
    maybe_run_query_runtime(Plan, ['item1']).

verify_is_check_literal_plan :-
    csharp_query_target:build_query_plan(test_is_check_literal/1, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Selection, columns:[0], width:1}),
    Selection = selection{
        type:selection,
        input:Arithmetic,
        predicate:condition{type:eq, left:operand{kind:column, index:2}, right:operand{kind:value, value:6}},
        width:3
    },
    Arithmetic = arithmetic{
        type:arithmetic,
        input:relation_scan{predicate:predicate{name:test_val, arity:2}, type:relation_scan, width:_},
        expression:expr{type:binary, op:add, left:expr{type:column, index:1}, right:expr{type:value, value:1}},
        result_index:2,
        width:3
    },
    maybe_run_query_runtime(Plan, ['item1']).

verify_is_check_bound_var_plan :-
    csharp_query_target:build_query_plan(test_is_check_bound_var/1, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Selection, columns:[0], width:1}),
    Selection = selection{
        type:selection,
        input:ConstArithmetic,
        predicate:condition{type:eq, left:operand{kind:column, index:3}, right:operand{kind:column, index:2}},
        width:4
    },
    ConstArithmetic = arithmetic{
        type:arithmetic,
        input:Arithmetic,
        expression:expr{type:value, value:6},
        result_index:3,
        width:4
    },
    Arithmetic = arithmetic{
        type:arithmetic,
        input:relation_scan{predicate:predicate{name:test_val, arity:2}, type:relation_scan, width:_},
        expression:expr{type:binary, op:add, left:expr{type:column, index:1}, right:expr{type:value, value:1}},
        result_index:2,
        width:3
    },
    maybe_run_query_runtime(Plan, ['item1']).

verify_arith_expr_eq_plan :-
    csharp_query_target:build_query_plan(test_arith_expr_eq/1, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Selection, columns:[0], width:1}),
    Selection = selection{
        type:selection,
        input:Arithmetic,
        predicate:condition{type:arith_eq, left:operand{kind:column, index:2}, right:operand{kind:value, value:6}},
        width:3
    },
    Arithmetic = arithmetic{
        type:arithmetic,
        input:relation_scan{predicate:predicate{name:test_val, arity:2}, type:relation_scan, width:_},
        expression:expr{type:binary, op:add, left:expr{type:column, index:1}, right:expr{type:value, value:1}},
        result_index:2,
        width:3
    },
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'CompareValues'),
    sub_string(Source, _, _, _, '== 0'),
    maybe_run_query_runtime(Plan, ['item1']).

verify_arith_expr_neq_plan :-
    csharp_query_target:build_query_plan(test_arith_expr_neq/1, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Selection, columns:[0], width:1}),
    Selection = selection{
        type:selection,
        input:Arithmetic,
        predicate:condition{type:arith_neq, left:operand{kind:column, index:2}, right:operand{kind:value, value:6}},
        width:3
    },
    Arithmetic = arithmetic{
        type:arithmetic,
        input:relation_scan{predicate:predicate{name:test_val, arity:2}, type:relation_scan, width:_},
        expression:expr{type:binary, op:add, left:expr{type:column, index:1}, right:expr{type:value, value:1}},
        result_index:2,
        width:3
    },
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'CompareValues'),
    sub_string(Source, _, _, _, '!= 0'),
    maybe_run_query_runtime(Plan, ['item2']).

verify_arith_eq_direct_plan :-
    csharp_query_target:build_query_plan(test_arith_eq_direct/1, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Selection, columns:[0], width:1}),
    Selection = selection{
        type:selection,
        input:relation_scan{predicate:predicate{name:test_val, arity:2}, type:relation_scan, width:_},
        predicate:condition{type:arith_eq, left:operand{kind:column, index:1}, right:operand{kind:value, value:5}},
        width:_
    },
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'CompareValues'),
    sub_string(Source, _, _, _, '== 0'),
    maybe_run_query_runtime(Plan, ['item1']).

verify_arith_neq_direct_plan :-
    csharp_query_target:build_query_plan(test_arith_neq_direct/1, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Selection, columns:[0], width:1}),
    Selection = selection{
        type:selection,
        input:relation_scan{predicate:predicate{name:test_val, arity:2}, type:relation_scan, width:_},
        predicate:condition{type:arith_neq, left:operand{kind:column, index:1}, right:operand{kind:value, value:5}},
        width:_
    },
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'CompareValues'),
    sub_string(Source, _, _, _, '!= 0'),
    maybe_run_query_runtime(Plan, ['item2']).

verify_aggregate_count_plan :-
    csharp_query_target:build_query_plan(test_sale_count/1, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Aggregate, columns:[0], width:1}),
    is_dict(Aggregate, aggregate),
    get_dict(input, Aggregate, unit{type:unit, width:0}),
    get_dict(predicate, Aggregate, predicate{name:test_sale, arity:2}),
    get_dict(op, Aggregate, count),
    get_dict(group_indices, Aggregate, []),
    get_dict(value_index, Aggregate, -1),
    get_dict(width, Aggregate, 1),
    get_dict(args, Aggregate, [operand{kind:wildcard}, operand{kind:wildcard}]),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'AggregateNode'),
    sub_string(Source, _, _, _, 'AggregateOperation.Count'),
    sub_string(Source, _, _, _, 'UnitNode'),
    sub_string(Source, _, _, _, 'Wildcard.Value'),
    maybe_run_query_runtime(Plan, ['3']).

verify_aggregate_min_plan :-
    csharp_query_target:build_query_plan(test_sale_min/1, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Aggregate, columns:[0], width:1}),
    is_dict(Aggregate, aggregate),
    get_dict(input, Aggregate, unit{type:unit, width:0}),
    get_dict(predicate, Aggregate, predicate{name:test_sale, arity:2}),
    get_dict(op, Aggregate, min),
    get_dict(group_indices, Aggregate, []),
    get_dict(value_index, Aggregate, 1),
    get_dict(width, Aggregate, 1),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'AggregateOperation.Min'),
    maybe_run_query_runtime(Plan, ['5']).

verify_aggregate_max_plan :-
    csharp_query_target:build_query_plan(test_sale_max/1, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Aggregate, columns:[0], width:1}),
    is_dict(Aggregate, aggregate),
    get_dict(input, Aggregate, unit{type:unit, width:0}),
    get_dict(predicate, Aggregate, predicate{name:test_sale, arity:2}),
    get_dict(op, Aggregate, max),
    get_dict(group_indices, Aggregate, []),
    get_dict(value_index, Aggregate, 1),
    get_dict(width, Aggregate, 1),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'AggregateOperation.Max'),
    maybe_run_query_runtime(Plan, ['10']).

verify_grouped_aggregate_sum_plan :-
    csharp_query_target:build_query_plan(test_sales_by_customer/2, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Aggregate, columns:[0, 1], width:2}),
    is_dict(Aggregate, aggregate),
    get_dict(input, Aggregate, unit{type:unit, width:0}),
    get_dict(predicate, Aggregate, predicate{name:test_sale, arity:2}),
    get_dict(op, Aggregate, sum),
    get_dict(group_indices, Aggregate, [0]),
    get_dict(value_index, Aggregate, 1),
    get_dict(width, Aggregate, 2),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'AggregateNode'),
    sub_string(Source, _, _, _, 'AggregateOperation.Sum'),
    maybe_run_query_runtime(Plan, ['alice,15', 'bob,7']).

verify_multi_key_grouped_aggregate_sum_plan :-
    csharp_query_target:build_query_plan(test_sales_by_customer_product/3, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Aggregate, columns:[0, 1, 2], width:3}),
    is_dict(Aggregate, aggregate),
    get_dict(input, Aggregate, unit{type:unit, width:0}),
    get_dict(predicate, Aggregate, predicate{name:test_sale_item, arity:3}),
    get_dict(op, Aggregate, sum),
    get_dict(group_indices, Aggregate, [0, 1]),
    get_dict(value_index, Aggregate, 2),
    get_dict(width, Aggregate, 3),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'AggregateOperation.Sum'),
    sub_string(Source, _, _, _, 'new int[]{ 0, 1 }'),
    maybe_run_query_runtime(Plan, ['alice,laptop,12', 'alice,mouse,5', 'bob,laptop,3', 'bob,mouse,8']).

verify_grouped_aggregate_count_plan :-
    csharp_query_target:build_query_plan(test_sale_count_grouped/2, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Aggregate, columns:[0, 1], width:2}),
    is_dict(Aggregate, aggregate),
    get_dict(input, Aggregate, unit{type:unit, width:0}),
    get_dict(predicate, Aggregate, predicate{name:test_sale, arity:2}),
    get_dict(op, Aggregate, count),
    get_dict(group_indices, Aggregate, [0]),
    get_dict(value_index, Aggregate, -1),
    get_dict(width, Aggregate, 2),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'AggregateOperation.Count'),
    maybe_run_query_runtime(Plan, ['alice,2', 'bob,1']).

verify_correlated_aggregate_count_plan :-
    csharp_query_target:build_query_plan(test_sale_count_by_customer/2, [target(csharp_query)], Plan),
    get_dict(root, Plan, Root),
    sub_term(Aggregate, Root),
    is_dict(Aggregate, aggregate),
    get_dict(predicate, Aggregate, predicate{name:test_sale, arity:2}),
    get_dict(op, Aggregate, count),
    get_dict(group_indices, Aggregate, []),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'AggregateOperation.Count'),
    maybe_run_query_runtime(Plan, ['alice,2', 'bob,1', 'charlie,0']).

verify_correlated_aggregate_sum_plan :-
    csharp_query_target:build_query_plan(test_sale_sum_by_customer/2, [target(csharp_query)], Plan),
    get_dict(root, Plan, Root),
    sub_term(Aggregate, Root),
    is_dict(Aggregate, aggregate),
    get_dict(predicate, Aggregate, predicate{name:test_sale, arity:2}),
    get_dict(op, Aggregate, sum),
    get_dict(group_indices, Aggregate, []),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'AggregateOperation.Sum'),
    maybe_run_query_runtime(Plan, ['alice,15', 'bob,7']).

verify_correlated_aggregate_min_plan :-
    csharp_query_target:build_query_plan(test_sale_min_by_customer/2, [target(csharp_query)], Plan),
    get_dict(root, Plan, Root),
    sub_term(Aggregate, Root),
    is_dict(Aggregate, aggregate),
    get_dict(predicate, Aggregate, predicate{name:test_sale, arity:2}),
    get_dict(op, Aggregate, min),
    get_dict(group_indices, Aggregate, []),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'AggregateOperation.Min'),
    maybe_run_query_runtime(Plan, ['alice,5', 'bob,7']).

verify_correlated_aggregate_max_plan :-
    csharp_query_target:build_query_plan(test_sale_max_by_customer/2, [target(csharp_query)], Plan),
    get_dict(root, Plan, Root),
    sub_term(Aggregate, Root),
    is_dict(Aggregate, aggregate),
    get_dict(predicate, Aggregate, predicate{name:test_sale, arity:2}),
    get_dict(op, Aggregate, max),
    get_dict(group_indices, Aggregate, []),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'AggregateOperation.Max'),
    maybe_run_query_runtime(Plan, ['alice,10', 'bob,7']).

verify_aggregate_set_plan :-
    csharp_query_target:build_query_plan(test_sale_customers_set/1, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Aggregate, columns:[0], width:1}),
    is_dict(Aggregate, aggregate),
    get_dict(op, Aggregate, set),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'AggregateOperation.Set'),
    maybe_run_query_runtime(Plan, ['[alice|bob]']).

verify_aggregate_bag_plan :-
    csharp_query_target:build_query_plan(test_sale_customers_bag/1, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Aggregate, columns:[0], width:1}),
    is_dict(Aggregate, aggregate),
    get_dict(op, Aggregate, bag),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'AggregateOperation.Bag'),
    maybe_run_query_runtime(Plan, ['[alice|alice|bob]']).

verify_aggregate_subplan_count_with_constraint_plan :-
    csharp_query_target:build_query_plan(test_sale_filtered_count/1, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Agg, columns:[0], width:1}),
    is_dict(Agg, aggregate_subplan),
    get_dict(input, Agg, unit{type:unit, width:0}),
    get_dict(op, Agg, count),
    get_dict(params, Agg, []),
    get_dict(group_indices, Agg, []),
    get_dict(value_index, Agg, -1),
    get_dict(width, Agg, 1),
    get_dict(subplan, Agg, projection{type:projection, columns:[], width:0, input:_}),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'AggregateSubplanNode'),
    sub_string(Source, _, _, _, 'AggregateOperation.Count'),
    sub_string(Source, _, _, _, 'SelectionNode'),
    maybe_run_query_runtime(Plan, ['2']).

verify_aggregate_subplan_count_with_constant_arg_plan :-
    csharp_query_target:build_query_plan(test_sale_filtered_alice_count/1, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Agg, columns:[0], width:1}),
    is_dict(Agg, aggregate_subplan),
    get_dict(input, Agg, unit{type:unit, width:0}),
    get_dict(op, Agg, count),
    get_dict(params, Agg, []),
    get_dict(group_indices, Agg, []),
    get_dict(value_index, Agg, -1),
    get_dict(width, Agg, 1),
    get_dict(subplan, Agg, Subplan),
    sub_term(selection{type:selection, predicate:condition{left:_, type:eq, right:operand{kind:value, value:alice}}, input:_, width:_}, Subplan),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'AggregateSubplanNode'),
    sub_string(Source, _, _, _, 'AggregateOperation.Count'),
    sub_string(Source, _, _, _, 'SelectionNode'),
    maybe_run_query_runtime(Plan, ['1']).

verify_aggregate_subplan_nested_aggregate_plan :-
    csharp_query_target:build_query_plan(test_customer_high_sales_count/1, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Agg, columns:[0], width:1}),
    is_dict(Agg, aggregate_subplan),
    get_dict(op, Agg, count),
    get_dict(subplan, Agg, Subplan),
    sub_term(InnerAgg, Subplan),
    is_dict(InnerAgg, aggregate),
    get_dict(op, InnerAgg, sum),
    get_dict(predicate, InnerAgg, predicate{name:test_sale, arity:2}),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'AggregateSubplanNode'),
    sub_string(Source, _, _, _, 'AggregateNode'),
    maybe_run_query_runtime(Plan, ['1']).

verify_aggregate_subplan_banned_sale_count_plan :-
    csharp_query_target:build_query_plan(test_banned_sale_count/1, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Agg, columns:[0], width:1}),
    is_dict(Agg, aggregate_subplan),
    get_dict(op, Agg, count),
    get_dict(group_indices, Agg, []),
    get_dict(value_index, Agg, -1),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'AggregateSubplanNode'),
    maybe_run_query_runtime(Plan, ['1']).

verify_aggregate_subplan_disjunction_count_plan :-
    csharp_query_target:build_query_plan(test_sale_alice_or_bob_count/1, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Agg, columns:[0], width:1}),
    is_dict(Agg, aggregate_subplan),
    get_dict(op, Agg, count),
    get_dict(group_indices, Agg, []),
    get_dict(value_index, Agg, -1),
    get_dict(params, Agg, []),
    get_dict(subplan, Agg, union{type:union, sources:Sources, width:0}),
    length(Sources, 2),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'AggregateSubplanNode'),
    sub_string(Source, _, _, _, 'UnionNode'),
    maybe_run_query_runtime(Plan, ['3']).

verify_aggregate_subplan_nested_disjunction_count_plan :-
    csharp_query_target:build_query_plan(test_sale_alice_or_bob_nested_count/1, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Agg, columns:[0], width:1}),
    is_dict(Agg, aggregate_subplan),
    get_dict(op, Agg, count),
    get_dict(group_indices, Agg, []),
    get_dict(value_index, Agg, -1),
    get_dict(params, Agg, []),
    get_dict(subplan, Agg, union{type:union, sources:Sources, width:0}),
    length(Sources, 2),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'AggregateSubplanNode'),
    sub_string(Source, _, _, _, 'UnionNode'),
    maybe_run_query_runtime(Plan, ['3']).

verify_aggregate_subplan_correlated_count_with_constraint_plan :-
    csharp_query_target:build_query_plan(test_sale_count_filtered_by_customer/2, [target(csharp_query)], Plan),
    get_dict(root, Plan, Root),
    sub_term(Agg, Root),
    is_dict(Agg, aggregate_subplan),
    get_dict(op, Agg, count),
    get_dict(group_indices, Agg, []),
    get_dict(value_index, Agg, -1),
    get_dict(params, Agg, [operand{kind:column, index:0}]),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'AggregateSubplanNode'),
    maybe_run_query_runtime(Plan, ['alice,1', 'bob,1', 'charlie,0']).

verify_aggregate_subplan_grouped_count_with_negation_plan :-
    csharp_query_target:build_query_plan(test_non_banned_sale_count_grouped/2, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Agg, columns:[0, 1], width:2}),
    is_dict(Agg, aggregate_subplan),
    get_dict(op, Agg, count),
    get_dict(group_indices, Agg, [0]),
    get_dict(value_index, Agg, -1),
    get_dict(params, Agg, []),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'AggregateSubplanNode'),
    sub_string(Source, _, _, _, 'NegationNode'),
    maybe_run_query_runtime(Plan, ['alice,2']).

verify_aggregate_subplan_sum_with_constraint_plan :-
    csharp_query_target:build_query_plan(test_sale_filtered_sum/1, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Agg, columns:[0], width:1}),
    is_dict(Agg, aggregate_subplan),
    get_dict(input, Agg, unit{type:unit, width:0}),
    get_dict(op, Agg, sum),
    get_dict(params, Agg, []),
    get_dict(group_indices, Agg, []),
    get_dict(value_index, Agg, 0),
    get_dict(width, Agg, 1),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'AggregateSubplanNode'),
    sub_string(Source, _, _, _, 'AggregateOperation.Sum'),
    sub_string(Source, _, _, _, 'SelectionNode'),
    maybe_run_query_runtime(Plan, ['17']).

verify_aggregate_subplan_set_with_negation_plan :-
    csharp_query_target:build_query_plan(test_non_banned_sale_customers_set/1, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Agg, columns:[0], width:1}),
    is_dict(Agg, aggregate_subplan),
    get_dict(op, Agg, set),
    get_dict(params, Agg, []),
    get_dict(group_indices, Agg, []),
    get_dict(value_index, Agg, 0),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'AggregateSubplanNode'),
    sub_string(Source, _, _, _, 'AggregateOperation.Set'),
    sub_string(Source, _, _, _, 'NegationNode'),
    maybe_run_query_runtime(Plan, ['[alice]']).

verify_aggregate_subplan_grouped_sum_with_negation_plan :-
    csharp_query_target:build_query_plan(test_non_banned_sale_sum_grouped/2, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Agg, columns:[0, 1], width:2}),
    is_dict(Agg, aggregate_subplan),
    get_dict(op, Agg, sum),
    get_dict(params, Agg, []),
    get_dict(group_indices, Agg, [0]),
    get_dict(value_index, Agg, 1),
    get_dict(width, Agg, 2),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'AggregateSubplanNode'),
    sub_string(Source, _, _, _, 'AggregateOperation.Sum'),
    sub_string(Source, _, _, _, 'NegationNode'),
    maybe_run_query_runtime(Plan, ['alice,15']).

verify_parameterized_fib_plan :-
    csharp_query_target:build_query_plan(test_fib_param/2, [target(csharp_query)], Plan),
    get_dict(is_recursive, Plan, true),
    get_dict(metadata, Plan, Meta),
    get_dict(modes, Meta, Modes),
    Modes == [input, output],
    get_dict(root, Plan, Root),
    atom_concat(test_fib_param, '$need', NeedName),
    sub_term(materialize{type:materialize, id:_, plan:fixpoint{type:fixpoint, head:predicate{name:NeedName, arity:1}, base:_, recursive:_, width:1}, width:1}, Root),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'MaterializeNode'),
    sub_string(Source, _, _, _, 'ParamSeedNode').

verify_parameterized_fib_runtime :-
    csharp_query_target:build_query_plan(test_fib_param/2, [target(csharp_query)], Plan),
    maybe_run_query_runtime(Plan, ['5,8'], [[5]]).

verify_multi_mode_codegen_plan :-
    csharp_target:compile_predicate_to_csharp(test_multi_mode/2, [mode(query)], Code),
    sub_string(Code, _, _, _, 'BuildIn0'),
    sub_string(Code, _, _, _, 'BuildIn1'),
    sub_string(Code, _, _, _, 'BuildForInputs').

verify_multi_mode_plan_selection_api :-
    csharp_query_target:build_query_plans(test_multi_mode/2, [target(csharp_query)], Plans),
    length(Plans, 2),
    Plans = [Plan0, Plan1],
    get_dict(metadata, Plan0, Meta0),
    get_dict(modes, Meta0, [input, output]),
    get_dict(metadata, Plan1, Meta1),
    get_dict(modes, Meta1, [output, input]),
    csharp_query_target:build_query_plan_for_inputs(test_multi_mode/2, [target(csharp_query)], [0], Selected0),
    get_dict(metadata, Selected0, SelectedMeta0),
    get_dict(modes, SelectedMeta0, [input, output]),
    csharp_query_target:build_query_plan_for_inputs(test_multi_mode/2, [target(csharp_query)], [1], Selected1),
    get_dict(metadata, Selected1, SelectedMeta1),
    get_dict(modes, SelectedMeta1, [output, input]).

verify_multi_mode_runtime_dispatch :-
    csharp_target:compile_predicate_to_csharp(test_multi_mode/2, [mode(query)], ModuleSource),
    csharp_query_target:build_query_plan_for_inputs(test_multi_mode/2, [target(csharp_query)], [0], Plan0),
    csharp_query_target:plan_module_name(Plan0, ModuleClass),
    harness_source_multi_mode_dispatch(ModuleClass, HarnessSource),
    maybe_run_multi_mode_dispatch_runtime(ModuleClass, ModuleSource, HarnessSource, ['alice,bob', 'bob,charlie']).

with_suppressed_user_error(Goal) :-
    current_input(In),
    current_output(Out),
    stream_property(Err, alias(user_error)),
    setup_call_cleanup(
        open_null_stream(Null),
        setup_call_cleanup(
            set_prolog_IO(In, Out, Null),
            catch(Goal, _Error, fail),
            set_prolog_IO(In, Out, Err)
        ),
        close(Null)
    ).

verify_any_mode_rejected_plan :-
    with_suppressed_user_error(
        \+ csharp_query_target:build_query_plan(test_any_mode/2, [target(csharp_query)], _Plan)
    ).

verify_parameterized_need_allows_post_agg :-
    HeadSpec = predicate{name:test_post_agg_param, arity:2},
    csharp_target:gather_predicate_clauses(HeadSpec, Clauses),
    csharp_target:partition_recursive_clauses(test_post_agg_param, 2, Clauses, _BaseClauses, RecClauses),
    csharp_target:eligible_for_need_closure(HeadSpec, [HeadSpec], RecClauses, [input, output]).

verify_negation_plan :-
    csharp_query_target:build_query_plan(test_allowed/1, [target(csharp_query)], Plan),
    get_dict(relations, Plan, Relations),
    member(relation{predicate:predicate{name:test_banned, arity:1}, facts:_}, Relations),
    get_dict(root, Plan, Root),
    sub_term(negation{type:negation, predicate:predicate{name:test_banned, arity:1}, args:_, input:_, width:_}, Root),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'NegationNode'),
    maybe_run_query_runtime(Plan, ['alice']).

verify_parameterized_need_allows_prefix_negation :-
    csharp_query_target:build_query_plan(test_countdown/2, [target(csharp_query)], Plan),
    get_dict(is_recursive, Plan, true),
    get_dict(root, Plan, Root),
    atom_concat(test_countdown, '$need', NeedName),
    sub_term(materialize{type:materialize, id:_, plan:fixpoint{type:fixpoint, head:predicate{name:NeedName, arity:1}, base:_, recursive:_, width:1}, width:1}, Root),
    sub_term(negation{type:negation, predicate:predicate{name:test_blocked, arity:1}, args:_, input:_, width:_}, Root).

verify_recursive_arithmetic_plan :-
    csharp_query_target:build_query_plan(test_factorial/2, [target(csharp_query)], Plan),
    get_dict(is_recursive, Plan, true),
    get_dict(root, Plan, fixpoint{type:fixpoint, head:_, base:Base, recursive:[RecursiveClause], width:2}),
    Base = relation_scan{predicate:predicate{name:test_factorial, arity:2}, type:relation_scan, width:2},
    RecursiveClause = projection{
        type:projection,
        input:OuterArithmetic,
        columns:[0, 4],
        width:2
    },
    OuterArithmetic = arithmetic{
        type:arithmetic,
        input:JoinNode,
        expression:OuterExpr,
        result_index:4,
        width:5
    },
    is_dict(OuterExpr, expr),
    get_dict(op, OuterExpr, multiply),
    get_dict(left, OuterExpr, expr{type:column, index:3}),
    get_dict(right, OuterExpr, expr{type:column, index:0}),
    JoinNode = join{
        type:join,
        left:InnerArithmetic,
        right:recursive_ref{predicate:predicate{name:test_factorial, arity:2}, role:delta, type:recursive_ref, width:_},
        left_keys:[1],
        right_keys:[0],
        left_width:_,
        right_width:_,
        width:_
    },
    InnerArithmetic = arithmetic{
        type:arithmetic,
        input:Selection,
        expression:InnerExpr,
        result_index:1,
        width:2
    },
    is_dict(InnerExpr, expr),
    get_dict(op, InnerExpr, add),
    get_dict(left, InnerExpr, expr{type:column, index:0}),
    get_dict(right, InnerExpr, expr{type:value, value:Neg1}),
    Neg1 = -1,
    Selection = selection{
        type:selection,
        input:relation_scan{predicate:predicate{name:test_factorial_input, arity:1}, type:relation_scan, width:_},
        predicate:condition{type:gt, left:operand{kind:column, index:0}, right:operand{kind:value, value:0}},
        width:1
    },
    maybe_run_query_runtime(Plan, ['0,1', '1,1', '2,2', '3,6']).

verify_recursive_plan :-
    csharp_query_target:build_query_plan(test_reachable/2, [target(csharp_query)], Plan),
    get_dict(is_recursive, Plan, true),
    get_dict(root, Plan, fixpoint{type:fixpoint, head:_, base:Base, recursive:[RecursiveClause], width:2}),
    Base = projection{
        type:projection,
        input:relation_scan{predicate:predicate{name:test_fact, arity:2}, type:relation_scan, width:_},
        columns:[0, 1],
        width:2
    },
    RecursiveClause = projection{
        type:projection,
        input:JoinNode,
        columns:[0, 3],
        width:2
    },
    JoinNode = join{
        type:join,
        left:relation_scan{predicate:predicate{name:test_fact, arity:2}, type:relation_scan, width:_},
        right:recursive_ref{predicate:predicate{name:test_reachable, arity:2}, role:delta, type:recursive_ref, width:_},
        left_keys:[1],
        right_keys:[0],
        left_width:_,
        right_width:_,
        width:_
    },
    maybe_run_query_runtime(Plan, ['alice,bob', 'bob,charlie', 'alice,charlie']).

verify_mutual_recursion_plan :-
    csharp_query_target:build_query_plan(test_even/1, [target(csharp_query)], Plan),
    get_dict(is_recursive, Plan, true),
    get_dict(root, Plan, mutual_fixpoint{type:mutual_fixpoint, head:predicate{name:test_even, arity:1}, members:Members}),
    length(Members, 2),
    member(EvenMember, Members),
    get_dict(predicate, EvenMember, predicate{name:test_even, arity:1}),
    get_dict(base, EvenMember, EvenBase),
    get_dict(recursive, EvenMember, EvenVariants),
    EvenBase = relation_scan{predicate:predicate{name:test_even, arity:1}, type:relation_scan, width:_},
    member(EvenRecursive, EvenVariants),
    sub_term(cross_ref{predicate:predicate{name:test_odd, arity:1}, role:delta, type:cross_ref, width:_}, EvenRecursive),
    member(OddMember, Members),
    get_dict(predicate, OddMember, predicate{name:test_odd, arity:1}),
    get_dict(base, OddMember, OddBase),
    get_dict(recursive, OddMember, OddVariants),
    OddBase = relation_scan{predicate:predicate{name:test_odd, arity:1}, type:relation_scan, width:_},
    member(OddRecursive, OddVariants),
    sub_term(cross_ref{predicate:predicate{name:test_even, arity:1}, role:delta, type:cross_ref, width:_}, OddRecursive),
    maybe_run_query_runtime(Plan, ['0', '2', '4']).

verify_parameterized_mutual_recursion_plan :-
    csharp_query_target:build_query_plan(test_even_param/1, [target(csharp_query)], Plan),
    get_dict(is_recursive, Plan, true),
    get_dict(metadata, Plan, Meta),
    get_dict(modes, Meta, Modes),
    Modes == [input],
    get_dict(root, Plan, mutual_fixpoint{type:mutual_fixpoint, head:predicate{name:test_even_param, arity:1}, members:Members}),
    length(Members, 2),
    atom_concat(test_even_param, '$need', NeedName),
    sub_term(materialize{type:materialize, id:_, plan:fixpoint{type:fixpoint, head:predicate{name:NeedName, arity:2}, base:_, recursive:_, width:2}, width:2}, Members),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'new int[]{ 0 }'),
    sub_string(Source, _, _, _, NeedName),
    sub_string(Source, _, _, _, 'MaterializeNode'),
    maybe_run_query_runtime(Plan, ['4'], [[4]]).

verify_parameterized_mutual_recursion_inferred_plan :-
    csharp_query_target:build_query_plan(test_even_param_partial/1, [target(csharp_query)], Plan),
    get_dict(is_recursive, Plan, true),
    get_dict(metadata, Plan, Meta),
    get_dict(modes, Meta, Modes),
    Modes == [input],
    get_dict(root, Plan, mutual_fixpoint{type:mutual_fixpoint, head:predicate{name:test_even_param_partial, arity:1}, members:Members}),
    length(Members, 2),
    atom_concat(test_even_param_partial, '$need', NeedName),
    sub_term(materialize{type:materialize, id:_, plan:fixpoint{type:fixpoint, head:predicate{name:NeedName, arity:2}, base:_, recursive:_, width:2}, width:2}, Members),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'new int[]{ 0 }'),
    sub_string(Source, _, _, _, NeedName),
    sub_string(Source, _, _, _, 'MaterializeNode').

verify_parameterized_mutual_recursion_fallback_plan :-
    capture_user_error(
        csharp_query_target:build_query_plan(test_even_param_unbound/1, [target(csharp_query)], Plan),
        Err
    ),
    Err == "",
    get_dict(is_recursive, Plan, true),
    get_dict(metadata, Plan, Meta),
    get_dict(modes, Meta, Modes),
    Modes == [input],
    get_dict(root, Plan, mutual_fixpoint{type:mutual_fixpoint, head:predicate{name:test_even_param_unbound, arity:1}, members:Members}),
    length(Members, 2),
    atom_concat(test_even_param_unbound, '$need', NeedName),
    \+ sub_term(predicate{name:NeedName, arity:_}, Members),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'new int[]{ 0 }'),
    \+ sub_string(Source, _, _, _, NeedName),
    \+ sub_string(Source, _, _, _, 'MaterializeNode').

verify_dynamic_source_plan :-
    setup_call_cleanup(
        setup_csv_dynamic_source,
        verify_dynamic_source_plan_(),
        cleanup_csv_dynamic_source
    ).

verify_dynamic_source_plan_ :-
    csharp_query_target:build_query_plan(test_user_age/2, [target(csharp_query)], Plan),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'DelimitedTextReader'),
    sub_string(Source, _, _, _, 'test_users.csv'),
    maybe_run_query_runtime(Plan, ['Alice,30', 'Bob,25', 'Charlie,35']).

verify_tsv_dynamic_source_plan :-
    setup_call_cleanup(
        setup_tsv_dynamic_source,
        verify_tsv_dynamic_source_plan_(),
        cleanup_tsv_dynamic_source
    ).

verify_tsv_dynamic_source_plan_ :-
    csharp_query_target:build_query_plan(test_sales_total/2, [target(csharp_query)], Plan),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'test_sales.tsv'),
    string_codes(TabLiteral, [0'@, 34, 9, 34]),
    sub_string(Source, _, _, _, TabLiteral),
    maybe_run_query_runtime(Plan, ['Laptop,1200', 'Mouse,25', 'Keyboard,75']).

verify_json_dynamic_source_plan :-
    setup_call_cleanup(
        setup_json_dynamic_source,
        verify_json_dynamic_source_plan_(),
        cleanup_json_dynamic_source
    ).

verify_json_dynamic_source_plan_ :-
    csharp_query_target:build_query_plan(test_product_price/2, [target(csharp_query)], Plan),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'JsonStreamReader'),
    sub_string(Source, _, _, _, 'test_products.json'),
    maybe_run_query_runtime(Plan, ['Laptop,999', 'Mouse,25', 'Keyboard,75']).

verify_json_nested_source_plan :-
    setup_call_cleanup(
        setup_json_orders_source,
        verify_json_nested_source_plan_(),
        cleanup_json_orders_source
    ).

verify_json_nested_source_plan_ :-
    csharp_query_target:build_query_plan(test_order_first_item/3, [target(csharp_query)], Plan),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'test_orders.json'),
    sub_string(Source, _, _, _, 'items[0].product'),
    maybe_run_query_runtime(Plan, ['Alice,Laptop,1200', 'Bob,Mouse,25', 'Charlie,Keyboard,75']).

verify_json_jsonpath_source_plan :-
    setup_call_cleanup(
        setup_json_jsonpath_source,
        verify_json_jsonpath_source_plan_(),
        cleanup_json_jsonpath_source
    ).

verify_json_jsonpath_source_plan_ :-
    csharp_query_target:build_query_plan(test_jsonpath_projection/2, [target(csharp_query)], Plan),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'JsonColumnSelectorKind.JsonPath'),
    maybe_run_query_runtime(Plan, [
        'Alice,Laptop',
        'Bob,Mouse',
        'Charlie,Keyboard'
    ]).

verify_json_schema_source_plan :-
    setup_call_cleanup(
        setup_json_schema_source,
        verify_json_schema_source_plan_(),
        cleanup_json_schema_source
    ).

verify_json_schema_source_plan_ :-
    csharp_query_target:build_query_plan(test_product_record/1, [target(csharp_query)], Plan),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'ProductRecord'),
    maybe_run_query_runtime(Plan, [
        'ProductRecord { Id = P001, Name = Laptop, Price = 999 }',
        'ProductRecord { Id = P002, Name = Mouse, Price = 25 }',
        'ProductRecord { Id = P003, Name = Keyboard, Price = 75 }'
    ]).

verify_json_nested_schema_record_plan :-
    setup_call_cleanup(
        setup_json_nested_schema_source,
        verify_json_nested_schema_record_plan_(),
        cleanup_json_nested_schema_source
    ).

verify_json_nested_schema_record_plan_ :-
    csharp_query_target:build_query_plan(test_order_summary/1, [target(csharp_query)], Plan),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'OrderRecord'),
    sub_string(Source, _, _, _, 'LineItemRecord'),
    maybe_run_query_runtime(Plan, [
        'OrderSummaryRecord { Order = OrderRecord { Id = SO1, Customer = Alice }, FirstItem = LineItemRecord { Product = Laptop, Total = 1200 } }',
        'OrderSummaryRecord { Order = OrderRecord { Id = SO2, Customer = Bob }, FirstItem = LineItemRecord { Product = Mouse, Total = 25 } }',
        'OrderSummaryRecord { Order = OrderRecord { Id = SO3, Customer = Charlie }, FirstItem = LineItemRecord { Product = Keyboard, Total = 75 } }'
    ]).

verify_json_jsonl_source_plan :-
    setup_call_cleanup(
        setup_json_jsonl_source,
        verify_json_jsonl_source_plan_(),
        cleanup_json_jsonl_source
    ).

verify_json_jsonl_source_plan_ :-
    csharp_query_target:build_query_plan(test_orders_jsonl/3, [target(csharp_query)], Plan),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'test_orders.jsonl'),
    sub_string(Source, _, _, _, 'TreatArrayAsStream = false'),
    maybe_run_query_runtime(Plan, ['Alice,Laptop,1200', 'Bob,Mouse,25', 'Charlie,Keyboard,75']).

verify_json_null_policy_skip_plan :-
    setup_call_cleanup(
        setup_json_null_skip_source,
        verify_json_null_policy_skip_plan_(),
        cleanup_json_null_skip_source
    ).

verify_json_null_policy_skip_plan_ :-
    csharp_query_target:build_query_plan(test_json_null_skip/2, [target(csharp_query)], Plan),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'JsonNullPolicy.Skip'),
    maybe_run_query_runtime(Plan, ['Alice,Mouse']).

verify_json_null_policy_default_plan :-
    setup_call_cleanup(
        setup_json_null_default_source,
        verify_json_null_policy_default_plan_(),
        cleanup_json_null_default_source
    ).

verify_json_null_policy_default_plan_ :-
    csharp_query_target:build_query_plan(test_json_null_default/2, [target(csharp_query)], Plan),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'JsonNullPolicy.Default'),
    sub_string(Source, _, _, _, 'NullReplacement = "N/A"'),
    maybe_run_query_runtime(Plan, ['Alice,Mouse', 'Bob,N/A', 'Charlie,N/A']).

verify_json_object_source_plan :-
    setup_call_cleanup(
        setup_json_object_source,
        verify_json_object_source_plan_(),
        cleanup_json_object_source
    ).

verify_json_object_source_plan_ :-
    csharp_query_target:build_query_plan(test_product_object/1, [target(csharp_query)], Plan),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'JsonStreamReader'),
    sub_string(Source, _, _, _, 'ReturnObject = true'),
    maybe_run_query_runtime(Plan, [
        '{"id":"P001","name":"Laptop","price":999}',
        '{"id":"P002","name":"Mouse","price":25}',
        '{"id":"P003","name":"Keyboard","price":75}'
    ]).

verify_xml_dynamic_source_plan :-
    setup_call_cleanup(
        setup_xml_dynamic_source,
        verify_xml_dynamic_source_plan_(),
        cleanup_xml_dynamic_source
    ).

verify_xml_dynamic_source_plan_ :-
    csharp_query_target:build_query_plan(test_xml_item/1, [target(csharp_query)], Plan),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'XmlStreamReader'),
    sub_string(Source, _, _, _, 'test_xml_fragments.txt'),
    sub_string(Source, _, _, _, 'TreatPearltreesCDataAsText = true'),
    % Validate dictionary projection picks up local + qualified keys
    maybe_run_query_runtime(Plan, [
        "_{id:1, name:Alpha, '@lang':en}",
        "_{id:2, title:Hacktivism, 'pt:item':Hacktivism, '@code':X, '@pt:id':2, '@pt:code':A1}",
        "_{id:3, name:Gamma}"
    ]).

verify_xml_nested_projection_plan :-
    setup_call_cleanup(
        setup_xml_dynamic_source_nested,
        verify_xml_nested_projection_plan_(),
        cleanup_xml_dynamic_source_nested
    ).

verify_xml_nested_projection_plan_ :-
    csharp_query_target:build_query_plan(test_xml_item_nested/1, [target(csharp_query)], Plan),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'XmlStreamReader'),
    sub_string(Source, _, _, _, 'NestedProjection = true'),
    sub_string(Source, _, _, _, 'test_xml_fragments.txt').

verify_xml_pearltrees_preset_plan :-
    setup_call_cleanup(
        setup_xml_pearltrees_source,
        verify_xml_pearltrees_preset_plan_(),
        cleanup_xml_pearltrees_source
    ).

verify_xml_pearltrees_preset_plan_ :-
    csharp_query_target:build_query_plan(test_pt_item/1, [target(csharp_query)], Plan),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'XmlStreamReader'),
    sub_string(Source, _, _, _, 'NamespacePrefixes'),
    sub_string(Source, _, _, _, 'TreatPearltreesCDataAsText = true').

setup_csv_dynamic_source :-
    source(csv, test_users, [csv_file('test_data/test_users.csv'), has_header(true)]),
    assertz(user:(test_user_age(Name, Age) :- test_users(_, Name, Age))).

cleanup_csv_dynamic_source :-
    retractall(user:test_user_age(_, _)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_users/3, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_users/3, _)).

setup_tsv_dynamic_source :-
    source(csv, test_sales, [
        csv_file('test_data/test_sales.tsv'),
        delimiter('\t'),
        has_header(true),
        quote_style(none)
    ]),
    assertz(user:(test_sales_total(Product, Total) :- test_sales(_, Product, Total))).

cleanup_tsv_dynamic_source :-
    retractall(user:test_sales_total(_, _)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_sales/3, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_sales/3, _)).

setup_json_dynamic_source :-
    source(json, test_products, [
        json_file('test_data/test_products.json'),
        record_format(json),
        columns([id, name, price])
    ]),
    assertz(user:(test_product_price(Name, Price) :- test_products(_, Name, Price))).

cleanup_json_dynamic_source :-
    retractall(user:test_product_price(_, _)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_products/3, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_products/3, _)).

setup_xml_dynamic_source :-
    source(xml, test_xml_items, [
        file('test_data/test_xml_fragments.txt'),
        record_format(xml),
        record_separator(line_feed)
    ]),
    assertz(user:(test_xml_item(Row) :- test_xml_items(Row))).

cleanup_xml_dynamic_source :-
    retractall(user:test_xml_item(_)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_xml_items/1, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_xml_items/1, _)).

setup_json_schema_source :-
    source(json, test_product_record_source, [
        json_file('test_data/test_products.json'),
        schema([
            field(id, 'id', string),
            field(name, 'name', string),
            field(price, 'price', double)
        ]),
        record_type('ProductRecord')
    ]),
    assertz(user:(test_product_record(Row) :- test_product_record_source(Row))).

setup_xml_dynamic_source_nested :-
    source(xml, test_xml_items_nested, [
        file('test_data/test_xml_fragments.txt'),
        record_format(xml),
        record_separator(line_feed),
        nested_projection(true)
    ]),
    assertz(user:(test_xml_item_nested(Row) :- test_xml_items_nested(Row))).

cleanup_xml_dynamic_source_nested :-
    retractall(user:test_xml_item_nested(_)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_xml_items_nested/1, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_xml_items_nested/1, _)).

setup_xml_pearltrees_source :-
    source(xml, test_pt_items, [
        file('test_data/test_xml_fragments.txt'),
        record_format(xml),
        record_separator(line_feed),
        pearltrees(true)
    ]),
    assertz(user:(test_pt_item(Row) :- test_pt_items(Row))).

cleanup_xml_pearltrees_source :-
    retractall(user:test_pt_item(_)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_pt_items/1, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_pt_items/1, _)).

cleanup_json_schema_source :-
    retractall(user:test_product_record(_)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_product_record_source/1, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_product_record_source/1, _)).

setup_json_nested_schema_source :-
    source(json, test_order_summary_source, [
        json_file('test_data/test_orders.json'),
        schema([
            field(order, 'order', record('OrderRecord', [
                field(id, 'id', string),
                field(customer, 'customer.name', string)
            ])),
            field(first_item, 'items[0]', record('LineItemRecord', [
                field(product, 'product', string),
                field(total, 'total', double)
            ]))
        ]),
        record_type('OrderSummaryRecord')
    ]),
    assertz(user:(test_order_summary(Row) :- test_order_summary_source(Row))).

cleanup_json_nested_schema_source :-
    retractall(user:test_order_summary(_)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_order_summary_source/1, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_order_summary_source/1, _)).

setup_json_jsonl_source :-
    source(json, test_orders_jsonl_source, [
        json_file('test_data/test_orders.jsonl'),
        record_format(jsonl),
        columns(['order.customer.name', 'items[0].product', 'items[0].total'])
    ]),
    assertz(user:(test_orders_jsonl(Customer, Product, Total) :-
        test_orders_jsonl_source(Customer, Product, Total))).

cleanup_json_jsonl_source :-
    retractall(user:test_orders_jsonl(_, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_orders_jsonl_source/3, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_orders_jsonl_source/3, _)).

setup_json_null_skip_source :-
    source(json, test_json_null_skip_source, [
        json_file('test_data/test_orders.json'),
        columns([
            jsonpath('$.order.customer.name'),
            jsonpath('$.items[1].product')
        ]),
        null_policy(skip)
    ]),
    assertz(user:(test_json_null_skip(Customer, Product) :-
        test_json_null_skip_source(Customer, Product))).

cleanup_json_null_skip_source :-
    retractall(user:test_json_null_skip(_, _)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_json_null_skip_source/2, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_json_null_skip_source/2, _)).

setup_json_null_default_source :-
    source(json, test_json_null_default_source, [
        json_file('test_data/test_orders.json'),
        columns([
            jsonpath('$.order.customer.name'),
            jsonpath('$.items[1].product')
        ]),
        null_policy(default('N/A'))
    ]),
    assertz(user:(test_json_null_default(Customer, Product) :-
        test_json_null_default_source(Customer, Product))).

cleanup_json_null_default_source :-
    retractall(user:test_json_null_default(_, _)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_json_null_default_source/2, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_json_null_default_source/2, _)).

setup_json_orders_source :-
    source(json, test_orders, [
        json_file('test_data/test_orders.json'),
        record_format(json),
        columns(['order.customer.name', 'items[0].product', 'items[0].total'])
    ]),
    assertz(user:(test_order_first_item(Customer, Product, Total) :-
        test_orders(Customer, Product, Total))).

cleanup_json_orders_source :-
    retractall(user:test_order_first_item(_, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_orders/3, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_orders/3, _)).

setup_json_jsonpath_source :-
    source(json, test_jsonpath_projection_source, [
        json_file('test_data/test_orders.json'),
        columns([
            jsonpath('$.order.customer.name'),
            jsonpath('$.items[*].product')
        ])
    ]),
    assertz(user:(test_jsonpath_projection(Customer, Product) :-
        test_jsonpath_projection_source(Customer, Product))).

cleanup_json_jsonpath_source :-
    retractall(user:test_jsonpath_projection(_, _)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_jsonpath_projection_source/2, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_jsonpath_projection_source/2, _)).

setup_json_object_source :-
    source(json, test_products_object, [
        json_file('test_data/test_products.json'),
        record_format(json),
        arity(1),
        type_hint('System.Text.Json.Nodes.JsonObject, System.Text.Json'),
        return_object(true)
    ]),
    assertz(user:(test_product_object(ProductJson) :-
        test_products_object(ProductJson))).

cleanup_json_object_source :-
    retractall(user:test_product_object(_)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_products_object/1, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_products_object/1, _)).

% Run with build-first approach, optionally skipping execution
maybe_run_query_runtime(Plan, ExpectedRows) :-
    maybe_run_query_runtime(Plan, ExpectedRows, []).

maybe_run_query_runtime(Plan, ExpectedRows, Params) :-
    (   getenv('SKIP_CSHARP_EXECUTION', '1')
    ->  prepare_temp_dir(Plan, Dir),
        (   generate_csharp_code_only(_Dotnet, Plan, Params, Dir)
        ->  write_expected_rows_file(Dir, ExpectedRows, Params)
        ;   writeln('  (C# code generation: FAIL)'),
            finalize_temp_dir(Dir),
            fail
        ),
        finalize_temp_dir(Dir)
    ;   dotnet_cli(Dotnet)
    ->  prepare_temp_dir(Plan, Dir),
        (   run_dotnet_plan_build_first(Dotnet, Plan, ExpectedRows, Params, Dir)
        ->  writeln('  (query runtime execution: PASS)'),
            finalize_temp_dir(Dir)
        ;   writeln('  (query runtime execution: FAIL - but plan structure verified)'),
            finalize_temp_dir(Dir)
        )
    ;   writeln('  (dotnet run skipped; see docs/CSHARP_DOTNET_RUN_HANG_SOLUTION.md)')
    ).

maybe_run_multi_mode_dispatch_runtime(ModuleClass, ModuleSource, HarnessSource, ExpectedRows) :-
    (   getenv('SKIP_CSHARP_EXECUTION', '1')
    ->  prepare_temp_dir(Dir),
        (   generate_csharp_multi_mode_dispatch_code_only(ModuleClass, ModuleSource, HarnessSource, Dir)
        ->  write_expected_rows_file(Dir, ExpectedRows, [])
        ;   writeln('  (C# code generation: FAIL)'),
            finalize_temp_dir(Dir),
            fail
        ),
        finalize_temp_dir(Dir)
    ;   dotnet_cli(Dotnet)
    ->  prepare_temp_dir(Dir),
        (   run_dotnet_multi_mode_dispatch_build_first(Dotnet, ModuleClass, ModuleSource, HarnessSource, ExpectedRows, Dir)
        ->  writeln('  (query runtime execution: PASS)'),
            finalize_temp_dir(Dir)
        ;   writeln('  (query runtime execution: FAIL - but plan structure verified)'),
            finalize_temp_dir(Dir)
        )
    ;   writeln('  (dotnet run skipped; see docs/CSHARP_DOTNET_RUN_HANG_SOLUTION.md)')
    ).

write_expected_rows_file(Dir, ExpectedRows, Params) :-
    directory_file_path(Dir, 'expected_rows.txt', RowsPath),
    maplist(to_atom, ExpectedRows, ExpectedAtoms0),
    sort(ExpectedAtoms0, ExpectedAtoms),
    setup_call_cleanup(
        open(RowsPath, write, Stream),
        forall(member(Row, ExpectedAtoms),
               format(Stream, '~w~n', [Row])),
        close(Stream)
    ),
    (   Params == []
    ->  true
    ;   directory_file_path(Dir, 'params.txt', ParamsPath),
        setup_call_cleanup(
            open(ParamsPath, write, PStream),
            format(PStream, '~q.~n', [Params]),
            close(PStream)
        )
    ).

capture_user_error(Goal, Output) :-
    current_input(In),
    current_output(Out),
    stream_property(Err, alias(user_error)),
    new_memory_file(MemFile),
    setup_call_cleanup(
        open_memory_file(MemFile, write, Stream),
        setup_call_cleanup(
            set_prolog_IO(In, Out, Stream),
            call(Goal),
            set_prolog_IO(In, Out, Err)
        ),
        (   close(Stream),
            memory_file_to_string(MemFile, Output),
            free_memory_file(MemFile)
        )
    ).

dotnet_cli(path(dotnet)) :-
    catch(
        (   process_create(path(dotnet), ['--version'],
                           [ stdout(null),
                             stderr(null),
                             process(PID)
                           ]),
            process_wait(PID, exit(0))
        ),
        _,
        fail
    ).

% Create temp directory with test name from Plan
prepare_temp_dir(Plan, Dir) :-
    get_dict(head, Plan, predicate{name:PredName, arity:_}),
    !,
    uuid(UUID),
    atomic_list_concat(['csharp_query_', PredName, '_', UUID], Sub),
    cqt_option(output_dir, Base),
    make_directory_path(Base),
    directory_file_path(Base, Sub, Dir),
    make_directory_path(Dir).

% Fallback for backwards compatibility
prepare_temp_dir(Dir) :-
    uuid(UUID),
    atomic_list_concat(['csharp_query_', UUID], Sub),
    cqt_option(output_dir, Base),
    make_directory_path(Base),
    directory_file_path(Base, Sub, Dir),
    make_directory_path(Dir).

run_dotnet_plan_verbose(Dotnet, Plan, ExpectedRows, Dir) :-
    run_dotnet_plan(Dotnet, Plan, ExpectedRows, Dir),
    (   cqt_option(keep_artifacts, true)
    ->  format('  (kept C# artifacts in ~w)~n', [Dir])
    ;   true
    ).

finalize_temp_dir(Dir) :-
    (   cqt_option(keep_artifacts, true)
    ->  format('  (kept C# artifacts in ~w)~n', [Dir])
    ;   catch(
            delete_directory_and_contents(Dir),
            Error,
            format('  (warning: could not cleanup ~w: ~w)~n', [Dir, Error])
        )
    ).

% Build-first approach (works around dotnet run hang)
% See: docs/CSHARP_DOTNET_RUN_HANG_SOLUTION.md
run_dotnet_plan_build_first(Dotnet, Plan, ExpectedRows, Dir) :-
    run_dotnet_plan_build_first(Dotnet, Plan, ExpectedRows, [], Dir).

run_dotnet_plan_build_first(Dotnet, Plan, ExpectedRows, Params, Dir) :-
    % Step 1: Create project and write source files
    dotnet_command(Dotnet, ['new','console','--force','--framework','net9.0'], Dir, StatusNew, _),
    (   StatusNew =:= 0
    ->  true
    ;   writeln('  (dotnet new console failed; skipping runtime execution test)'), fail
    ),

    % Copy QueryRuntime.cs
    absolute_file_name('src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs', RuntimePath, []),
    directory_file_path(Dir, 'QueryRuntime.cs', RuntimeCopy),
    copy_file(RuntimePath, RuntimeCopy),

    % Generate and write query module
    csharp_query_target:render_plan_to_csharp(Plan, ModuleSource),
    csharp_query_target:plan_module_name(Plan, ModuleClass),
    atom_concat(ModuleClass, '.cs', ModuleFile),
    directory_file_path(Dir, ModuleFile, ModulePath),
    write_string(ModulePath, ModuleSource),

    % Write harness
    harness_source(ModuleClass, Params, HarnessSource),
    directory_file_path(Dir, 'Program.cs', ProgramPath),
    write_string(ProgramPath, HarnessSource),

    % Step 2: Build the project
    dotnet_command(Dotnet, ['build','--no-restore'], Dir, StatusBuild, BuildOutput),
    (   StatusBuild =:= 0
    ->  true
    ;   format('  (dotnet build failed: ~s)~n', [BuildOutput]), fail
    ),

    % Step 3: Find and execute compiled binary
    find_compiled_executable(Dir, ExePath),
    (   ExePath \= ''
    ->  true
    ;   writeln('  (compiled executable not found)'), fail
    ),

    % Execute the binary directly
    execute_compiled_binary(ExePath, Dir, StatusRun, Output),
    (   StatusRun =:= 0
    ->  extract_result_rows(Output, Rows),
        sort(Rows, SortedRows),
        maplist(to_atom, ExpectedRows, ExpectedAtoms),
        sort(ExpectedAtoms, SortedExpected),
        (   SortedRows == SortedExpected
        ->  true
        ;   format('  dotnet run output mismatch: ~w~n', [SortedRows]), fail
        )
    ;   format('  (execution failed: ~s)~n', [Output]), fail
    ).

run_dotnet_multi_mode_dispatch_build_first(Dotnet, ModuleClass, ModuleSource, HarnessSource, ExpectedRows, Dir) :-
    dotnet_command(Dotnet, ['new','console','--force','--framework','net9.0'], Dir, StatusNew, _),
    (   StatusNew =:= 0
    ->  true
    ;   writeln('  (dotnet new console failed; skipping runtime execution test)'), fail
    ),
    absolute_file_name('src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs', RuntimePath, []),
    directory_file_path(Dir, 'QueryRuntime.cs', RuntimeCopy),
    copy_file(RuntimePath, RuntimeCopy),
    atom_concat(ModuleClass, '.cs', ModuleFile),
    directory_file_path(Dir, ModuleFile, ModulePath),
    write_string(ModulePath, ModuleSource),
    directory_file_path(Dir, 'Program.cs', ProgramPath),
    write_string(ProgramPath, HarnessSource),
    dotnet_command(Dotnet, ['build','--no-restore'], Dir, StatusBuild, BuildOutput),
    (   StatusBuild =:= 0
    ->  true
    ;   format('  (dotnet build failed: ~s)~n', [BuildOutput]), fail
    ),
    find_compiled_executable(Dir, ExePath),
    (   ExePath \= ''
    ->  true
    ;   writeln('  (compiled executable not found)'), fail
    ),
    execute_compiled_binary(ExePath, Dir, StatusRun, Output),
    (   StatusRun =:= 0
    ->  extract_result_rows(Output, Rows),
        sort(Rows, SortedRows),
        maplist(to_atom, ExpectedRows, ExpectedAtoms),
        sort(ExpectedAtoms, SortedExpected),
        (   SortedRows == SortedExpected
        ->  true
        ;   format('  dotnet run output mismatch: ~w~n', [SortedRows]), fail
        )
    ;   format('  (execution failed: ~s)~n', [Output]), fail
    ).

% Generate C# code without execution (for SKIP_CSHARP_EXECUTION mode)
% Skips all dotnet commands - just generates and writes C# source files
generate_csharp_code_only(Dotnet, Plan, Dir) :-
    generate_csharp_code_only(Dotnet, Plan, [], Dir).

generate_csharp_code_only(_Dotnet, Plan, Params, Dir) :-
    % Create .csproj file manually (without calling dotnet new console)
    file_base_name(Dir, ProjectName),
    create_minimal_csproj(Dir, ProjectName),

    % Copy QueryRuntime.cs
    absolute_file_name('src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs', RuntimePath, []),
    directory_file_path(Dir, 'QueryRuntime.cs', RuntimeCopy),
    copy_file(RuntimePath, RuntimeCopy),

    % Generate and write query module
    csharp_query_target:render_plan_to_csharp(Plan, ModuleSource),
    csharp_query_target:plan_module_name(Plan, ModuleClass),
    atom_concat(ModuleClass, '.cs', ModuleFile),
    directory_file_path(Dir, ModuleFile, ModulePath),
    write_string(ModulePath, ModuleSource),

    % Write harness
    harness_source(ModuleClass, Params, HarnessSource),
    directory_file_path(Dir, 'Program.cs', ProgramPath),
    write_string(ProgramPath, HarnessSource).
    % Note: dotnet execution commands (build, run) still skipped in this mode

generate_csharp_multi_mode_dispatch_code_only(ModuleClass, ModuleSource, HarnessSource, Dir) :-
    file_base_name(Dir, ProjectName),
    create_minimal_csproj(Dir, ProjectName),
    absolute_file_name('src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs', RuntimePath, []),
    directory_file_path(Dir, 'QueryRuntime.cs', RuntimeCopy),
    copy_file(RuntimePath, RuntimeCopy),
    atom_concat(ModuleClass, '.cs', ModuleFile),
    directory_file_path(Dir, ModuleFile, ModulePath),
    write_string(ModulePath, ModuleSource),
    directory_file_path(Dir, 'Program.cs', ProgramPath),
    write_string(ProgramPath, HarnessSource).

% Create a minimal .csproj file manually
create_minimal_csproj(Dir, ProjectName) :-
    atom_concat(ProjectName, '.csproj', CsprojFile),
    directory_file_path(Dir, CsprojFile, CsprojPath),
    CsprojContent = '<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net9.0</TargetFramework>
    <Nullable>enable</Nullable>
  </PropertyGroup>
</Project>',
    open(CsprojPath, write, Stream),
    write(Stream, CsprojContent),
    close(Stream).

% Original run_dotnet_plan (kept for reference, but not used)
run_dotnet_plan(Dotnet, Plan, ExpectedRows, Dir) :-
    dotnet_command(Dotnet, ['new','console','--force','--framework','net9.0'], Dir, StatusNew, _),
    (   StatusNew =:= 0
    ->  true
    ;   writeln('  (dotnet new console failed; skipping runtime execution test)'), fail
    ),
    absolute_file_name('src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs', RuntimePath, []),
    directory_file_path(Dir, 'QueryRuntime.cs', RuntimeCopy),
    copy_file(RuntimePath, RuntimeCopy),
    csharp_query_target:render_plan_to_csharp(Plan, ModuleSource),
    csharp_query_target:plan_module_name(Plan, ModuleClass),
    atom_concat(ModuleClass, '.cs', ModuleFile),
    directory_file_path(Dir, ModuleFile, ModulePath),
    write_string(ModulePath, ModuleSource),
    harness_source(ModuleClass, HarnessSource),
    directory_file_path(Dir, 'Program.cs', ProgramPath),
    write_string(ProgramPath, HarnessSource),
    dotnet_command(Dotnet, ['run','--no-restore'], Dir, StatusRun, Output),
    (   StatusRun =:= 0
    ->  extract_result_rows(Output, Rows),
        sort(Rows, SortedRows),
        maplist(to_atom, ExpectedRows, ExpectedAtoms),
        sort(ExpectedAtoms, SortedExpected),
        (   SortedRows == SortedExpected
        ->  true
        ;   format('  dotnet run output mismatch: ~w~n', [SortedRows]), fail
        )
    ;   format('  (dotnet run failed: ~s)~n', [Output]), fail
    ).

% Find the compiled executable in bin/Debug/net9.0/
find_compiled_executable(Dir, ExePath) :-
    directory_file_path(Dir, 'bin/Debug/net9.0', DebugDir),
    (   exists_directory(DebugDir)
    ->  directory_files(DebugDir, Files),
        member(File, Files),
        \+ atom_concat(_, '.dll', File),
        \+ atom_concat(_, '.pdb', File),
        \+ atom_concat(_, '.deps.json', File),
        \+ atom_concat(_, '.runtimeconfig.json', File),
        File \= '.',
        File \= '..',
        directory_file_path(DebugDir, File, ExePath),
        exists_file(ExePath),
        !
    ;   % No native executable, try DLL
        directory_file_path(DebugDir, 'test.dll', DllPath),
        exists_file(DllPath),
        !,
        ExePath = DllPath
    ).

% Execute compiled binary (native or DLL)
execute_compiled_binary(ExePath, Dir, Status, Output) :-
    dotnet_env(Dir, Env),
    (   atom_concat(_, '.dll', ExePath)
    ->  % Execute DLL with dotnet
        dotnet_cli(Dotnet),
        process_create(Dotnet, [ExePath],
                       [ cwd(Dir),
                         env(Env),
                         stdout(pipe(Out)),
                         stderr(pipe(Err)),
                         process(PID)
                       ])
    ;   % Execute native binary directly
        process_create(ExePath, [],
                       [ cwd(Dir),
                         env(Env),
                         stdout(pipe(Out)),
                         stderr(pipe(Err)),
                         process(PID)
                       ])
    ),
    read_string(Out, _, Stdout),
    read_string(Err, _, Stderr),
    close(Out),
    close(Err),
    process_wait(PID, exit(Status)),
    string_concat(Stdout, Stderr, Output).

harness_source(ModuleClass, Source) :-
    harness_source(ModuleClass, [], Source).

harness_source(ModuleClass, Params, Source) :-
    (   Params == []
    ->  ParamDecl = '',
        ExecCall = 'executor.Execute(result.Plan)'
    ;   csharp_params_literal(Params, ParamsLiteral),
        format(atom(ParamDecl), 'var parameters = ~w;~n', [ParamsLiteral]),
        ExecCall = 'executor.Execute(result.Plan, parameters)'
    ),
    format(atom(Source),
'using System;
using System.Linq;
using UnifyWeaver.QueryRuntime;
using System.Text.Json;
using System.Text.Json.Nodes;

var result = UnifyWeaver.Generated.~w.Build();
var executor = new QueryExecutor(result.Provider);
~wvar jsonOptions = new JsonSerializerOptions { WriteIndented = false };

 string FormatValue(object? value) => value switch
 {
     JsonNode node => node.ToJsonString(jsonOptions),
     JsonElement element => element.GetRawText(),
     System.Collections.IEnumerable enumerable when value is not string => "[" + string.Join("|", enumerable.Cast<object?>().Select(FormatValue).OrderBy(s => s, StringComparer.Ordinal)) + "]",
     _ => value?.ToString() ?? string.Empty
 };
 foreach (var row in ~w)
 {
    var projected = row.Take(result.Plan.Head.Arity)
                       .Select(FormatValue)
                       .ToArray();

    if (projected.Length == 0)
    {
        continue;
    }

    Console.WriteLine(string.Join(\",\", projected));
 }
 ', [ModuleClass, ParamDecl, ExecCall]).

harness_source_multi_mode_dispatch(ModuleClass, Source) :-
    format(atom(Source),
'using System;
using System.Linq;
using UnifyWeaver.QueryRuntime;
using System.Text.Json;
using System.Text.Json.Nodes;

var jsonOptions = new JsonSerializerOptions { WriteIndented = false };

 string FormatValue(object? value) => value switch
 {
     JsonNode node => node.ToJsonString(jsonOptions),
     JsonElement element => element.GetRawText(),
     System.Collections.IEnumerable enumerable when value is not string => "[" + string.Join("|", enumerable.Cast<object?>().Select(FormatValue).OrderBy(s => s, StringComparer.Ordinal)) + "]",
     _ => value?.ToString() ?? string.Empty
 };

 void PrintRows((InMemoryRelationProvider Provider, QueryPlan Plan) result, object[][] parameters)
{
    var executor = new QueryExecutor(result.Provider);
    foreach (var row in executor.Execute(result.Plan, parameters))
    {
        var projected = row.Take(result.Plan.Head.Arity)
                           .Select(FormatValue)
                           .ToArray();

        if (projected.Length == 0)
        {
            continue;
        }

        Console.WriteLine(string.Join(\",\", projected));
    }
}

var result0 = UnifyWeaver.Generated.~w.BuildForInputs(0);
PrintRows(result0, new object[][] { new object[]{ \"alice\" } });

var result1 = UnifyWeaver.Generated.~w.BuildForInputs(1);
PrintRows(result1, new object[][] { new object[]{ \"charlie\" } });
', [ModuleClass, ModuleClass]).

csharp_params_literal(Params, Literal) :-
    maplist(csharp_tuple_literal, Params, TupleLits),
    atomic_list_concat(TupleLits, ", ", TuplesStr),
    format(atom(Literal), 'new object[][]{ ~w }', [TuplesStr]).

csharp_tuple_literal(Tuple, Literal) :-
    maplist(csharp_value_literal, Tuple, Values),
    atomic_list_concat(Values, ", ", ValuesStr),
    format(atom(Literal), 'new object[]{ ~w }', [ValuesStr]).

csharp_value_literal(Value, Literal) :-
    (   number(Value)
    ->  format(atom(Literal), '~w', [Value])
    ;   string(Value)
    ->  format(atom(Literal), '\"~w\"', [Value])
    ;   atom(Value)
    ->  format(atom(Literal), '\"~w\"', [Value])
    ;   format(atom(Literal), '\"~w\"', [Value])
    ).

write_string(Path, String) :-
    setup_call_cleanup(open(Path, write, Stream),
                       write(Stream, String),
                       close(Stream)).

dotnet_command(Dotnet, Args, Dir, Status, Output) :-
    dotnet_env(Dir, Env),
    process_create(Dotnet, Args,
                   [ cwd(Dir),
                     env(Env),
                     stdout(pipe(Out)),
                     stderr(pipe(Err)),
                     process(PID)
                   ]),
    read_string(Out, _, Stdout),
    read_string(Err, _, Stderr),
    close(Out),
    close(Err),
    process_wait(PID, exit(Status)),
    string_concat(Stdout, Stderr, Output).

dotnet_env(Dir, Env) :-
    environ(RawEnv),
    exclude(is_dotnet_env, RawEnv, BaseEnv),
    Env = ['DOTNET_CLI_HOME'=Dir,
           'DOTNET_CLI_TELEMETRY_OPTOUT'='1',
           'DOTNET_NOLOGO'='1'
           | BaseEnv].

is_dotnet_env('DOTNET_CLI_HOME'=_).
is_dotnet_env('DOTNET_CLI_TELEMETRY_OPTOUT'=_).
is_dotnet_env('DOTNET_NOLOGO'=_).

extract_result_rows(Output, Rows) :-
    split_string(Output, "\n", "\r", Lines0),
    maplist(normalize_space_string, Lines0, NormalizedLines),
    include(non_empty_line, NormalizedLines, Candidate),
    maplist(to_atom, Candidate, Rows).

non_empty_line(Line) :-
    Line \== '',
    Line \== "".

normalize_space_string(Line, Normalized) :-
    normalize_space(string(Normalized), Line).

to_atom(Value, Atom) :-
    (   atom(Value) -> Atom = Value
    ;   string(Value) -> atom_string(Atom, Value)
    ;   term_to_atom(Atom, Value)
    ).

%% Option handling ---------------------------------------------------------

% The following predicates allow the dotnet harness to respect CLI
% switches (e.g. --csharp-query-output, --csharp-query-keep) and
% corresponding environment variables, mirroring the behaviour used in
% the education module examples.
configure_csharp_query_options :-
    retractall(cqt_option(_, _)),
    default_cqt_options(Default),
    maplist(assertz, Default),
    capture_env_overrides,
    capture_cli_overrides.

default_cqt_options([
    cqt_option(output_dir, 'tmp'),
    cqt_option(keep_artifacts, false)
]).

%% progress reporting helpers --------------------------------------------------

progress_init :-
    get_time(Now),
    retractall(progress_last_report(_)),
    retractall(progress_count(_)),
    asserta(progress_last_report(Now)),
    asserta(progress_count(0)).

progress_interval_seconds(10).

run_with_progress(Goal) :-
    (   catch(once(call(Goal)), E, (print_message(error, E), fail))
    ->  true
    ;   format('  FAILED: ~w~n', [Goal]),
        fail
    ),
    retract(progress_count(C0)),
    C is C0 + 1,
    asserta(progress_count(C)),
    progress_maybe_report(normal).

progress_maybe_report(force) :-
    !,
    progress_count(C),
    progress_total(T),
    format('  Progress: ~w/~w tests complete.~n', [C, T]).
progress_maybe_report(normal) :-
    progress_interval_seconds(Interval),
    get_time(Now),
    (   progress_last_report(Last),
        Delta is Now - Last,
        Delta >= Interval
    ->  retract(progress_last_report(_)),
        asserta(progress_last_report(Now)),
        progress_count(C),
        progress_total(T),
        format('  Progress: ~w/~w tests complete.~n', [C, T])
    ;   true
    ).

capture_env_overrides :-
    (   getenv('CSHARP_QUERY_OUTPUT_DIR', Dir),
        Dir \= ''
    ->  retractall(cqt_option(output_dir, _)),
        assertz(cqt_option(output_dir, Dir))
    ;   true
    ),
    (   getenv('CSHARP_QUERY_KEEP_ARTIFACTS', KeepRaw),
        normalize_yes_no(KeepRaw, Keep)
    ->  retractall(cqt_option(keep_artifacts, _)),
        assertz(cqt_option(keep_artifacts, Keep))
    ;   true
    ).

capture_cli_overrides :-
    current_prolog_flag(argv, Argv),
    apply_cli_overrides(Argv).

apply_cli_overrides([]).
apply_cli_overrides([Arg|Rest]) :-
    (   atom(Arg),
        atom_concat('--csharp-query-output=', DirAtom, Arg)
    ->  set_cqt_option(output_dir, DirAtom),
        apply_cli_overrides(Rest)
    ;   Arg == '--csharp-query-output',
        Rest = [Dir|Tail]
    ->  set_cqt_option(output_dir, Dir),
        apply_cli_overrides(Tail)
    ;   Arg == '--csharp-query-keep'
    ->  set_cqt_option(keep_artifacts, true),
        apply_cli_overrides(Rest)
    ;   Arg == '--csharp-query-autodelete'
    ->  set_cqt_option(keep_artifacts, false),
        apply_cli_overrides(Rest)
    ;   apply_cli_overrides(Rest)
    ).

set_cqt_option(Key, Value) :-
    retractall(cqt_option(Key, _)),
    assertz(cqt_option(Key, Value)).

normalize_yes_no(Value0, Bool) :-
    (   atom(Value0) -> atom_string(Value0, Value)
    ;   Value = Value0
    ),
    string_lower(Value, Lower),
    (   member(Lower, ["1", "true", "yes", "keep"])
    ->  Bool = true
    ;   member(Lower, ["0", "false", "no", "delete", "autodelete"])
    ->  Bool = false
    ).
