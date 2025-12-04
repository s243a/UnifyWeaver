% generate_sql_views.pl - Generate SQL views for testing

:- use_module('src/unifyweaver/targets/sql_target').

% Define schema
:- sql_table(person, [name-text, age-integer, city-text]).

% Define predicates
adult(Name) :- person(Name, Age, _), Age >= 18.
nyc_adults(Name) :- person(Name, Age, "NYC"), Age >= 21.

% Generate SQL
main :-
    % Generate adult view
    compile_predicate_to_sql(adult/1, [format(view)], SQL1),
    write_sql_file(SQL1, 'output_sql_test/adult.sql'),

    % Generate nyc_adults view
    compile_predicate_to_sql(nyc_adults/1, [format(view)], SQL2),
    write_sql_file(SQL2, 'output_sql_test/nyc_adults.sql'),

    write('SQL views generated successfully'), nl.

:- initialization(main, main).
