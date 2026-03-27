:- encoding(utf8).

:- ['src/unifyweaver/init'].
:- use_module('src/unifyweaver/core/recursive_compiler').

:- dynamic parent/2, ancestor/2.

parent(alice, bob).
parent(bob, charlie).
parent(bob, diana).
parent(charlie, eve).
parent(diana, frank).

ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).

test_stdin :-
    writeln('=== TEST: stdin mode (default, backward compat) ==='),
    recursive_compiler:compile_recursive(ancestor/2, [target(lua)], Code),
    writeln(Code), nl.

test_embedded :-
    writeln('=== TEST: embedded mode ==='),
    recursive_compiler:compile_recursive(ancestor/2, [target(lua), input(embedded)], Code),
    writeln(Code), nl.

test_file :-
    writeln('=== TEST: file mode ==='),
    recursive_compiler:compile_recursive(ancestor/2, [target(lua), input(file("facts.txt"))], Code),
    writeln(Code), nl.

test_vfs :-
    writeln('=== TEST: vfs mode ==='),
    recursive_compiler:compile_recursive(ancestor/2, [target(lua), input(vfs(family_tree))], Code),
    writeln(Code), nl.

test_vfs_output :-
    writeln('=== TEST: vfs mode (read from .output) ==='),
    recursive_compiler:compile_recursive(ancestor/2, [target(lua), input(vfs(family_tree, '.output'))], Code),
    writeln(Code), nl.

test_function :-
    writeln('=== TEST: function mode ==='),
    recursive_compiler:compile_recursive(ancestor/2, [target(lua), input(function)], Code),
    writeln(Code), nl.

run_tests :-
    test_stdin,
    test_embedded,
    test_file,
    test_vfs,
    test_vfs_output,
    test_function,
    writeln('=== ALL INPUT SOURCE TESTS PASSED ===').
