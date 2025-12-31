% test_sql_union.pl - Test UNION/INTERSECT/EXCEPT generation

:- use_module('src/unifyweaver/targets/sql_target').

% Define schemas
:- sql_table(person, [name-text, age-integer, city-text]).
:- sql_table(special_members, [name-text, member_type-text]).
:- sql_table(vip_list, [name-text, level-integer]).

% Test 1: Simple UNION - multiple clauses
% Adults can be from regular table OR special members
adult(Name) :- person(Name, Age, _), Age >= 18.
adult(Name) :- special_members(Name, _).

% Test 2: Three-way UNION
% VIPs are adults, special members, OR explicitly on VIP list
vip(Name) :- person(Name, Age, _), Age >= 21.
vip(Name) :- special_members(Name, 'gold').
vip(Name) :- vip_list(Name, _).

% Generate SQL
main :-
    write('Generating SQL UNION views...'), nl, nl,

    % Test 1: Two-clause UNION
    compile_predicate_to_sql(adult/1, [format(view)], SQL1),
    write_sql_file(SQL1, 'output_sql_test/adult_union.sql'),
    write('✓ Test 1: Two-clause UNION'), nl,

    % Test 2: Three-clause UNION
    compile_predicate_to_sql(vip/1, [format(view)], SQL2),
    write_sql_file(SQL2, 'output_sql_test/vip_union.sql'),
    write('✓ Test 2: Three-clause UNION'), nl,

    nl,
    write('All SQL UNION views generated successfully!'), nl.

:- initialization(main, main).
