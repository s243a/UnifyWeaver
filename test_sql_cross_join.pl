% test_sql_cross_join.pl - Test CROSS JOIN support

:- use_module('src/unifyweaver/targets/sql_target').

% Define schemas
:- sql_table(colors, [id-integer, name-text]).
:- sql_table(sizes, [id-integer, size-text]).
:- sql_table(products, [id-integer, product-text]).
:- sql_table(categories, [id-integer, name-text]).
:- sql_table(tags, [tag_id-integer, tag_name-text]).
:- sql_table(products_cat, [prod_id-integer, cat_id-integer, name-text]).

% Test 1: Simple CROSS JOIN (2 tables, no shared variables)
color_size_combos(Color, Size) :-
    colors(_, Color),
    sizes(_, Size).

% Test 2: Triple CROSS JOIN (3 tables, no shared variables)
all_combinations(Color, Size, Product) :-
    colors(_, Color),
    sizes(_, Size),
    products(_, Product).

% Test 3: Mixed INNER JOIN + CROSS JOIN
products_with_tags(ProdName, CatName, TagName) :-
    categories(CatId, CatName),
    products_cat(_, CatId, ProdName),  % INNER JOIN on CatId
    tags(_, TagName).                   % CROSS JOIN (no shared vars)

% Helper to write SQL to file
write_sql_file(SQL, FilePath) :-
    open(FilePath, write, Stream),
    write(Stream, SQL),
    close(Stream).

main :-
    write('=== Testing CROSS JOIN Support ==='), nl, nl,

    % Test 1: Simple CROSS JOIN
    write('Test 1: Simple CROSS JOIN (2 tables)'), nl,
    write('Pattern: colors, sizes (no shared variables)'), nl,
    (   compile_predicate_to_sql(color_size_combos/2, [format(view)], SQL1)
    ->  write_sql_file(SQL1, 'output_sql_test/cross_join_simple.sql'),
        write('✓ Generated SQL'), nl,
        write(SQL1), nl
    ;   write('✗ FAILED to compile'), nl
    ),
    nl,

    % Test 2: Triple CROSS JOIN
    write('Test 2: Triple CROSS JOIN (3 tables)'), nl,
    write('Pattern: colors, sizes, products (all independent)'), nl,
    (   compile_predicate_to_sql(all_combinations/3, [format(view)], SQL2)
    ->  write_sql_file(SQL2, 'output_sql_test/cross_join_triple.sql'),
        write('✓ Generated SQL'), nl,
        write(SQL2), nl
    ;   write('✗ FAILED to compile'), nl
    ),
    nl,

    % Test 3: Mixed INNER + CROSS
    write('Test 3: Mixed INNER JOIN + CROSS JOIN'), nl,
    write('Pattern: categories INNER products_cat, CROSS tags'), nl,
    (   compile_predicate_to_sql(products_with_tags/3, [format(view)], SQL3)
    ->  write_sql_file(SQL3, 'output_sql_test/cross_join_mixed.sql'),
        write('✓ Generated SQL'), nl,
        write(SQL3), nl
    ;   write('✗ FAILED to compile'), nl
    ),
    nl,

    write('=== CROSS JOIN tests complete ==='), nl.

:- initialization(main, main).
