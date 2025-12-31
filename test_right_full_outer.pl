% test_right_full_outer.pl - Test RIGHT JOIN and FULL OUTER JOIN

:- use_module('src/unifyweaver/targets/sql_target').

% Define schemas
:- sql_table(customers, [id-integer, name-text, region-text]).
:- sql_table(orders, [id-integer, customer_id-integer, product-text, amount-real]).
:- sql_table(t1, [x-integer, a-text]).
:- sql_table(t2, [y-integer, x-integer, b-text]).
:- sql_table(t3, [z-integer, y-integer, c-text]).

% Test 1: Simple RIGHT JOIN
% Keep all orders, NULL for customers without orders
order_customers(Product, Name) :-
    (customers(CustomerId, Name, _) ; Name = null),
    orders(_, CustomerId, Product, _).

% Test 2: Simple FULL OUTER JOIN
% Keep all customers AND all orders
all_customer_orders(Name, Product) :-
    (customers(CustomerId, Name, _) ; Name = null),
    (orders(_, CustomerId, Product, _) ; Product = null).

% Test 3: RIGHT JOIN Chain
% t1 (RIGHT) t2 (RIGHT) t3
right_chain(A, B, C) :-
    (t1(X, A) ; A = null),
    (t2(Y, X, B) ; B = null, X = null),
    t3(_, Y, C).

% Helper to write SQL to file
write_sql_file(SQL, FilePath) :-
    open(FilePath, write, Stream),
    write(Stream, SQL),
    close(Stream).

main :-
    write('=== Testing RIGHT JOIN and FULL OUTER JOIN ==='), nl, nl,

    % Test 1: RIGHT JOIN
    write('Test 1: Simple RIGHT JOIN'), nl,
    write('Pattern: (customers ; null), orders'), nl,
    (   compile_predicate_to_sql(order_customers/2, [format(view)], SQL1)
    ->  write_sql_file(SQL1, 'output_sql_test/right_join_simple.sql'),
        write('✓ Generated SQL'), nl,
        write(SQL1), nl
    ;   write('✗ FAILED to compile'), nl
    ),
    nl,

    % Test 2: FULL OUTER JOIN
    write('Test 2: FULL OUTER JOIN'), nl,
    write('Pattern: (customers ; null), (orders ; null)'), nl,
    (   compile_predicate_to_sql(all_customer_orders/2, [format(view)], SQL2)
    ->  write_sql_file(SQL2, 'output_sql_test/full_outer_join.sql'),
        write('✓ Generated SQL'), nl,
        write(SQL2), nl
    ;   write('✗ FAILED to compile'), nl
    ),
    nl,

    % Test 3: RIGHT JOIN Chain
    write('Test 3: RIGHT JOIN Chain'), nl,
    write('Pattern: (t1 ; null), (t2 ; null), t3'), nl,
    (   compile_predicate_to_sql(right_chain/3, [format(view)], SQL3)
    ->  write_sql_file(SQL3, 'output_sql_test/right_join_chain.sql'),
        write('✓ Generated SQL'), nl,
        write(SQL3), nl
    ;   write('✗ FAILED to compile'), nl
    ),
    nl,

    write('=== RIGHT/FULL OUTER JOIN tests complete ==='), nl.

:- initialization(main, main).
