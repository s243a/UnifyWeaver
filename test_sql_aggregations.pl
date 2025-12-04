% test_sql_aggregations.pl - Test SQL aggregation generation

:- use_module('src/unifyweaver/targets/sql_target').

% Define schema - orders table
:- sql_table(orders, [customer-text, product-text, amount-integer, region-text]).

% Test 1: Simple GROUP BY with COUNT
% Count orders per customer
customer_order_count(Customer, Count) :-
    group_by(Customer, orders(Customer, _, _, _), count, Count).

% Test 2: GROUP BY with SUM
% Total order amount per customer
customer_total(Customer, Total) :-
    group_by(Customer, orders(Customer, _, Amount, _), sum, Total).

% Test 3: GROUP BY with AVG
% Average order amount per customer
customer_avg(Customer, Avg) :-
    group_by(Customer, orders(Customer, _, Amount, _), avg, Avg).

% Test 4: GROUP BY with MAX
% Maximum order amount per customer
customer_max(Customer, Max) :-
    group_by(Customer, orders(Customer, _, Amount, _), max, Max).

% Test 5: GROUP BY with MIN
% Minimum order amount per customer
customer_min(Customer, Min) :-
    group_by(Customer, orders(Customer, _, Amount, _), min, Min).

% Test 6: GROUP BY with WHERE constraints
% Total order amount per customer in specific region
customer_total_by_region(Customer, Total) :-
    group_by(Customer, (orders(Customer, _, Amount, Region), Region = "West"), sum, Total).

% Test 7: GROUP BY with HAVING
% Customers with more than 2 orders
high_volume_customers(Customer, Count) :-
    group_by(Customer, orders(Customer, _, _, _), count, Count),
    Count > 2.

% Generate SQL
main :-
    write('Generating SQL aggregation views...'), nl, nl,

    % Test 1
    compile_predicate_to_sql(customer_order_count/2, [format(view)], SQL1),
    write_sql_file(SQL1, 'output_sql_test/customer_order_count.sql'),
    write('✓ Test 1: COUNT'), nl,

    % Test 2
    compile_predicate_to_sql(customer_total/2, [format(view)], SQL2),
    write_sql_file(SQL2, 'output_sql_test/customer_total.sql'),
    write('✓ Test 2: SUM'), nl,

    % Test 3
    compile_predicate_to_sql(customer_avg/2, [format(view)], SQL3),
    write_sql_file(SQL3, 'output_sql_test/customer_avg.sql'),
    write('✓ Test 3: AVG'), nl,

    % Test 4
    compile_predicate_to_sql(customer_max/2, [format(view)], SQL4),
    write_sql_file(SQL4, 'output_sql_test/customer_max.sql'),
    write('✓ Test 4: MAX'), nl,

    % Test 5
    compile_predicate_to_sql(customer_min/2, [format(view)], SQL5),
    write_sql_file(SQL5, 'output_sql_test/customer_min.sql'),
    write('✓ Test 5: MIN'), nl,

    % Test 6
    compile_predicate_to_sql(customer_total_by_region/2, [format(view)], SQL6),
    write_sql_file(SQL6, 'output_sql_test/customer_total_by_region.sql'),
    write('✓ Test 6: GROUP BY with WHERE'), nl,

    % Test 7
    compile_predicate_to_sql(high_volume_customers/2, [format(view)], SQL7),
    write_sql_file(SQL7, 'output_sql_test/high_volume_customers.sql'),
    write('✓ Test 7: GROUP BY with HAVING'), nl,

    nl,
    write('All SQL aggregation views generated successfully!'), nl.

:- initialization(main, main).
