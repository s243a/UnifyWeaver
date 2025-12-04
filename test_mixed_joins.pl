% test_mixed_joins.pl - Test mixed INNER/LEFT JOINs

:- use_module('src/unifyweaver/targets/sql_target').

% Define schemas
:- sql_table(customers, [id-integer, name-text, region-text]).
:- sql_table(categories, [id-integer, customer_id-integer, category-text]).
:- sql_table(orders, [id-integer, customer_id-integer, category_id-integer, product-text]).
:- sql_table(t1, [x-integer, a-text]).
:- sql_table(t2, [y-integer, x-integer, b-text]).
:- sql_table(t3, [z-integer, y-integer, c-text]).
:- sql_table(t4, [id-integer, z-integer, d-text]).

% Test 1: One INNER JOIN, One LEFT JOIN
% customers → categories (INNER) → orders (LEFT)
customer_category_orders(Name, Category, Product) :-
    customers(CustomerId, Name, _),
    categories(CategoryId, CustomerId, Category),
    ( orders(_, CustomerId, CategoryId, Product)
    ; Product = null
    ).

% Test 2: Two INNER JOINs, One LEFT JOIN
% t1 → t2 (INNER) → t3 (INNER) → t4 (LEFT)
chain_inner_left(A, B, C, D) :-
    t1(X, A),
    t2(Y, X, B),
    t3(Z, Y, C),
    ( t4(_, Z, D)
    ; D = null
    ).

% Test 3: One INNER JOIN, Two LEFT JOINs
% t1 → t2 (INNER) → t3 (LEFT) → t4 (LEFT)
inner_then_nested_left(A, B, C, D) :-
    t1(X, A),
    t2(Y, X, B),
    ( t3(Z, Y, C)
    ; C = null, Z = null
    ),
    ( t4(_, Z, D)
    ; D = null
    ).

% Helper to write SQL to file
write_sql_file(SQL, FilePath) :-
    open(FilePath, write, Stream),
    write(Stream, SQL),
    close(Stream).

main :-
    write('=== Testing Mixed INNER/LEFT JOINs ==='), nl, nl,

    % Test 1
    write('Test 1: One INNER, One LEFT'), nl,
    (   compile_predicate_to_sql(customer_category_orders/3, [format(view)], SQL1)
    ->  write_sql_file(SQL1, 'output_sql_test/mixed_one_inner_one_left.sql'),
        write('✓ Generated SQL'), nl,
        write(SQL1), nl
    ;   write('✗ FAILED to compile'), nl
    ),
    nl,

    % Test 2
    write('Test 2: Two INNER, One LEFT'), nl,
    (   compile_predicate_to_sql(chain_inner_left/4, [format(view)], SQL2)
    ->  write_sql_file(SQL2, 'output_sql_test/mixed_two_inner_one_left.sql'),
        write('✓ Generated SQL'), nl,
        write(SQL2), nl
    ;   write('✗ FAILED to compile'), nl
    ),
    nl,

    % Test 3
    write('Test 3: One INNER, Two LEFT'), nl,
    (   compile_predicate_to_sql(inner_then_nested_left/4, [format(view)], SQL3)
    ->  write_sql_file(SQL3, 'output_sql_test/mixed_one_inner_two_left.sql'),
        write('✓ Generated SQL'), nl,
        write(SQL3), nl
    ;   write('✗ FAILED to compile'), nl
    ),
    nl,

    write('=== Mixed JOIN tests complete ==='), nl.

:- initialization(main, main).
