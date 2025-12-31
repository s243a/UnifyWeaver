% test_sql_joins.pl - Test SQL JOIN generation

:- use_module('src/unifyweaver/targets/sql_target').

% Define schemas
:- sql_table(customers, [customer_id-integer, name-text, city-text]).
:- sql_table(orders, [order_id-integer, customer_id-integer, product-text, amount-integer]).
:- sql_table(products, [product_id-integer, product_name-text, category-text, price-integer]).

% Test 1: Simple INNER JOIN on customer_id
% Get customer name and their order details
customer_orders(CustomerName, Product, Amount) :-
    customers(CustomerId, CustomerName, _),
    orders(_, CustomerId, Product, Amount).

% Test 2: JOIN with WHERE constraint
% Get orders for customers in NYC
nyc_customer_orders(CustomerName, Product) :-
    customers(CustomerId, CustomerName, City),
    orders(_, CustomerId, Product, _),
    City = "NYC".

% Test 3: Three-table JOIN
% Customer orders with product details (if we had product_id in orders)
% For now, simpler: customers with orders and their cities
customer_order_summary(CustomerName, City, TotalAmount) :-
    customers(CustomerId, CustomerName, City),
    orders(_, CustomerId, _, TotalAmount).

% Generate SQL
main :-
    write('Generating SQL JOIN views...'), nl, nl,

    % Test 1
    compile_predicate_to_sql(customer_orders/3, [format(view)], SQL1),
    write_sql_file(SQL1, 'output_sql_test/customer_orders.sql'),
    write('✓ Test 1: Simple INNER JOIN'), nl,

    % Test 2
    compile_predicate_to_sql(nyc_customer_orders/2, [format(view)], SQL2),
    write_sql_file(SQL2, 'output_sql_test/nyc_customer_orders.sql'),
    write('✓ Test 2: JOIN with WHERE'), nl,

    % Test 3
    compile_predicate_to_sql(customer_order_summary/3, [format(view)], SQL3),
    write_sql_file(SQL3, 'output_sql_test/customer_order_summary.sql'),
    write('✓ Test 3: JOIN with multiple columns'), nl,

    nl,
    write('All SQL JOIN views generated successfully!'), nl.

:- initialization(main, main).
