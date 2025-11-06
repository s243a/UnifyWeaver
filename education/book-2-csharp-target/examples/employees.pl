% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

/**
 * Employee Database Example for Chapter 2
 *
 * This example demonstrates Stream Target compilation with
 * a simple employee management system.
 *
 * Usage:
 *   ?- [employees].
 *   ?- compile_all_queries.
 */

:- use_module('../../src/unifyweaver/targets/csharp_stream_target').

% Employee facts: employee(Name, Department)
employee(alice, engineering).
employee(bob, engineering).
employee(charlie, sales).
employee(diane, sales).
employee(eve, hr).

% Manager relationships: manages(Manager, Employee)
manages(alice, bob).
manages(charlie, diane).

% Query predicates to compile
engineer(Name) :- employee(Name, engineering).
sales_person(Name) :- employee(Name, sales).
team_member(Manager, Employee) :- manages(Manager, Employee).

% Helper to compile all queries
compile_all_queries :-
    writeln('Compiling engineer/1...'),
    compile_predicate_to_csharp(engineer/1, [], EngineerCode),
    open('Engineer.cs', write, S1),
    write(S1, EngineerCode),
    close(S1),

    writeln('Compiling sales_person/1...'),
    compile_predicate_to_csharp(sales_person/1, [], SalesCode),
    open('SalesPerson.cs', write, S2),
    write(S2, SalesCode),
    close(S2),

    writeln('Compiling team_member/2...'),
    compile_predicate_to_csharp(team_member/2, [], TeamCode),
    open('TeamMember.cs', write, S3),
    write(S3, TeamCode),
    close(S3),

    writeln('Done! Generated files:'),
    writeln('  - Engineer.cs'),
    writeln('  - SalesPerson.cs'),
    writeln('  - TeamMember.cs').
