:- encoding(utf8).
% AWK Target Demo - Shows how to use the AWK target in UnifyWeaver

:- use_module('../src/unifyweaver/core/recursive_compiler').

% Example 1: Simple facts (arity 1)
person(alice).
person(bob).
person(charlie).

% Example 2: Binary facts (arity 2)
parent(alice, bob).
parent(bob, charlie).
parent(alice, charlie).

% Example 3: Facts with more fields
employee(alice, engineering, 5).
employee(bob, marketing, 3).
employee(charlie, engineering, 7).

demo_person_awk :-
    write('=== Demo 1: Compiling person/1 to AWK ==='), nl, nl,
    compile_recursive(person/1, [target(awk)], AwkCode),
    write(AwkCode), nl, nl.

demo_parent_awk :-
    write('=== Demo 2: Compiling parent/2 to AWK ==='), nl, nl,
    compile_recursive(parent/2, [target(awk)], AwkCode),
    write(AwkCode), nl, nl.

demo_employee_awk :-
    write('=== Demo 3: Compiling employee/3 to AWK ==='), nl, nl,
    compile_recursive(employee/3, [target(awk)], AwkCode),
    write(AwkCode), nl, nl.

% Run all demos
run_all :-
    demo_person_awk,
    demo_parent_awk,
    demo_employee_awk,
    write('All demos completed!'), nl.

% Usage:
% ?- consult('examples/awk_target_demo.pl').
% ?- run_all.
