% test_sql_left_join.pl - Prototype LEFT JOIN via disjunction

:- use_module('src/unifyweaver/targets/sql_target').

% Define schemas
:- sql_table(customers, [id-integer, name-text, region-text]).
:- sql_table(orders, [id-integer, customer_id-integer, product-text, amount-real]).
:- sql_table(shipments, [id-integer, order_id-integer, tracking-text]).

% Test 1: Basic LEFT JOIN - single column
% All customers, including those without orders
customer_orders(Name, Product) :-
    customers(CustomerId, Name, _),
    ( orders(_, CustomerId, Product, _)
    ; Product = null
    ).

% Test 2: Multi-column LEFT JOIN
% All customers with order details (or NULLs)
customer_order_details(Name, Product, Amount) :-
    customers(CustomerId, Name, _),
    ( orders(_, CustomerId, Product, Amount)
    ; Product = null, Amount = null
    ).

% Test 3: Nested LEFT JOINs
% All customers → orders → shipments (with NULLs at each level)
customer_shipments(Name, Product, Tracking) :-
    customers(CustomerId, Name, _),
    ( orders(OrderId, CustomerId, Product, _)
    ; Product = null, OrderId = null
    ),
    ( shipments(_, OrderId, Tracking)
    ; Tracking = null
    ).

% Test 4: LEFT JOIN with WHERE clause
% Customers from EU region with their orders (or NULLs)
eu_customer_orders(Name, Product) :-
    customers(CustomerId, Name, Region),
    Region = 'EU',
    ( orders(_, CustomerId, Product, _)
    ; Product = null
    ).

% Generate SQL
main :-
    write('Testing LEFT JOIN prototype...'), nl, nl,

    % Test 1: Basic LEFT JOIN
    (   compile_predicate_to_sql(customer_orders/2, [format(view)], SQL1)
    ->  write_sql_file(SQL1, 'output_sql_test/left_join_basic.sql'),
        write('✓ Test 1: Basic LEFT JOIN'), nl
    ;   write('✗ Test 1 FAILED: Could not compile'), nl
    ),

    % Test 2: Multi-column LEFT JOIN
    (   compile_predicate_to_sql(customer_order_details/3, [format(view)], SQL2)
    ->  write_sql_file(SQL2, 'output_sql_test/left_join_multi.sql'),
        write('✓ Test 2: Multi-column LEFT JOIN'), nl
    ;   write('✗ Test 2 FAILED: Could not compile'), nl
    ),

    % Test 3: Nested LEFT JOINs
    (   compile_predicate_to_sql(customer_shipments/3, [format(view)], SQL3)
    ->  write_sql_file(SQL3, 'output_sql_test/left_join_nested.sql'),
        write('✓ Test 3: Nested LEFT JOINs'), nl
    ;   write('✗ Test 3 FAILED: Could not compile'), nl
    ),

    % Test 4: LEFT JOIN with WHERE
    (   compile_predicate_to_sql(eu_customer_orders/2, [format(view)], SQL4)
    ->  write_sql_file(SQL4, 'output_sql_test/left_join_where.sql'),
        write('✓ Test 4: LEFT JOIN with WHERE'), nl
    ;   write('✗ Test 4 FAILED: Could not compile'), nl
    ),

    nl,
    write('LEFT JOIN prototype tests complete!'), nl.

:- initialization(main, main).
