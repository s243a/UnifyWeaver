% test_sql_setops.pl - Test INTERSECT and EXCEPT set operations

:- use_module('src/unifyweaver/targets/sql_target').

% Define schemas
:- sql_table(person, [name-text, age-integer, city-text]).
:- sql_table(special_members, [name-text, member_type-text]).
:- sql_table(employees, [name-text, dept-text]).

% Define predicates to combine with set operations
adults(Name) :- person(Name, Age, _), Age >= 18.
members(Name) :- special_members(Name, _).
staff(Name) :- employees(Name, _).

% Generate SQL
main :-
    write('Generating SQL set operation views...'), nl, nl,

    % Test 1: INTERSECT - Find people who are both adults AND members
    compile_set_operation(intersect, [adults/1, members/1],
                         [format(view), view_name(adult_members)], SQL1),
    write_sql_file(SQL1, 'output_sql_test/intersect_test.sql'),
    write('✓ Test 1: INTERSECT (adult_members)'), nl,

    % Test 2: EXCEPT - Find adults who are NOT members
    compile_set_operation(except, [adults/1, members/1],
                         [format(view), view_name(adults_only)], SQL2),
    write_sql_file(SQL2, 'output_sql_test/except_test.sql'),
    write('✓ Test 2: EXCEPT (adults_only)'), nl,

    % Test 3: Three-way INTERSECT - Find common names across all three groups
    compile_set_operation(intersect, [adults/1, members/1, staff/1],
                         [format(view), view_name(common_all)], SQL3),
    write_sql_file(SQL3, 'output_sql_test/intersect_three.sql'),
    write('✓ Test 3: Three-way INTERSECT (common_all)'), nl,

    nl,
    write('All SQL set operation views generated successfully!'), nl.

:- initialization(main, main).
