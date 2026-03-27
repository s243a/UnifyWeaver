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

%% --- Lua tests ---
test_lua_stdin :-
    writeln('=== TEST: Lua stdin (default) ==='),
    recursive_compiler:compile_recursive(ancestor/2, [target(lua)], Code),
    (sub_string(Code, _, _, _, "io.lines()") -> writeln('  PASS: contains io.lines()') ; (writeln('  FAIL'), fail)).

test_lua_embedded :-
    writeln('=== TEST: Lua embedded ==='),
    recursive_compiler:compile_recursive(ancestor/2, [target(lua), input(embedded)], Code),
    (sub_string(Code, _, _, _, "add_fact(\"alice\"") -> writeln('  PASS: contains seed facts') ; (writeln('  FAIL'), fail)),
    (\+ sub_string(Code, _, _, _, "io.lines()") -> writeln('  PASS: no io.lines()') ; (writeln('  FAIL'), fail)).

test_lua_vfs :-
    writeln('=== TEST: Lua vfs ==='),
    recursive_compiler:compile_recursive(ancestor/2, [target(lua), input(vfs(family_tree))], Code),
    (sub_string(Code, _, _, _, "nb.read(\"family_tree\"") -> writeln('  PASS: contains nb.read()') ; (writeln('  FAIL'), fail)).

test_lua_file :-
    writeln('=== TEST: Lua file ==='),
    recursive_compiler:compile_recursive(ancestor/2, [target(lua), input(file("data.txt"))], Code),
    (sub_string(Code, _, _, _, "io.lines(\"data.txt\")") -> writeln('  PASS: contains io.lines(path)') ; (writeln('  FAIL'), fail)).

test_lua_function :-
    writeln('=== TEST: Lua function ==='),
    recursive_compiler:compile_recursive(ancestor/2, [target(lua), input(function)], Code),
    (sub_string(Code, _, _, _, "ancestor_from_pairs") -> writeln('  PASS: contains function API') ; (writeln('  FAIL'), fail)).

%% --- Python tests ---
test_python_stdin :-
    writeln('=== TEST: Python stdin (default) ==='),
    recursive_compiler:compile_recursive(ancestor/2, [target(python)], Code),
    (sub_string(Code, _, _, _, "sys.stdin") -> writeln('  PASS: contains sys.stdin') ; (writeln('  FAIL'), fail)).

test_python_embedded :-
    writeln('=== TEST: Python embedded ==='),
    recursive_compiler:compile_recursive(ancestor/2, [target(python), input(embedded)], Code),
    (sub_string(Code, _, _, _, "query.add_fact(\"alice\"") -> writeln('  PASS: contains seed facts') ; (writeln('  FAIL'), fail)),
    (\+ sub_string(Code, _, _, _, "sys.stdin") -> writeln('  PASS: no sys.stdin') ; (writeln('  FAIL'), fail)).

test_python_vfs :-
    writeln('=== TEST: Python vfs ==='),
    recursive_compiler:compile_recursive(ancestor/2, [target(python), input(vfs(family_tree))], Code),
    (sub_string(Code, _, _, _, "family_tree") -> writeln('  PASS: contains cell name') ; (writeln('  FAIL'), fail)).

test_python_function :-
    writeln('=== TEST: Python function ==='),
    recursive_compiler:compile_recursive(ancestor/2, [target(python), input(function)], Code),
    (sub_string(Code, _, _, _, "ancestor_from_pairs") -> writeln('  PASS: contains function API') ; (writeln('  FAIL'), fail)).

%% --- R tests ---
test_r_stdin :-
    writeln('=== TEST: R stdin (default) ==='),
    recursive_compiler:compile_recursive(ancestor/2, [target(r)], Code),
    (sub_string(Code, _, _, _, "readLines(file(\"stdin\"))") -> writeln('  PASS: contains readLines(stdin)') ; (writeln('  FAIL'), fail)).

test_r_embedded :-
    writeln('=== TEST: R embedded ==='),
    recursive_compiler:compile_recursive(ancestor/2, [target(r), input(embedded)], Code),
    (sub_string(Code, _, _, _, "add_parent(\"alice\"") -> writeln('  PASS: contains seed facts') ; (writeln('  FAIL'), fail)),
    (\+ sub_string(Code, _, _, _, "readLines") -> writeln('  PASS: no readLines') ; (writeln('  FAIL'), fail)).

test_r_vfs :-
    writeln('=== TEST: R vfs ==='),
    recursive_compiler:compile_recursive(ancestor/2, [target(r), input(vfs(family_tree))], Code),
    (sub_string(Code, _, _, _, "nb_read(\"family_tree\"") -> writeln('  PASS: contains nb_read()') ; (writeln('  FAIL'), fail)).

%% --- Bash tests ---
test_bash_stdin :-
    writeln('=== TEST: Bash stdin (default) ==='),
    recursive_compiler:compile_recursive(ancestor/2, [target(bash)], Code),
    (sub_string(Code, _, _, _, "ancestor") -> writeln('  PASS: generates bash code') ; (writeln('  FAIL'), fail)).

test_bash_embedded :-
    writeln('=== TEST: Bash embedded ==='),
    recursive_compiler:compile_recursive(ancestor/2, [target(bash), input(embedded)], Code),
    (sub_string(Code, _, _, _, "add_fact \"alice\"") -> writeln('  PASS: contains seed facts') ; (writeln('  FAIL'), fail)).

test_bash_vfs :-
    writeln('=== TEST: Bash vfs ==='),
    recursive_compiler:compile_recursive(ancestor/2, [target(bash), input(vfs(family_tree))], Code),
    (sub_string(Code, _, _, _, "/nb/family_tree") -> writeln('  PASS: contains VFS path') ; (writeln('  FAIL'), fail)).

test_bash_file :-
    writeln('=== TEST: Bash file ==='),
    recursive_compiler:compile_recursive(ancestor/2, [target(bash), input(file("facts.txt"))], Code),
    (sub_string(Code, _, _, _, "facts.txt") -> writeln('  PASS: contains file path') ; (writeln('  FAIL'), fail)).

%% --- Context-aware default tests ---
test_context_notebook :-
    writeln('=== TEST: context(notebook) defaults to embedded ==='),
    recursive_compiler:compile_recursive(ancestor/2, [target(lua), context(notebook)], Code),
    (sub_string(Code, _, _, _, "add_fact(\"alice\"") -> writeln('  PASS: embedded seed facts') ; (writeln('  FAIL'), fail)),
    (\+ sub_string(Code, _, _, _, "io.lines()") -> writeln('  PASS: no stdin') ; (writeln('  FAIL'), fail)).

test_context_workbook :-
    writeln('=== TEST: context(workbook) defaults to embedded ==='),
    recursive_compiler:compile_recursive(ancestor/2, [target(python), context(workbook)], Code),
    (sub_string(Code, _, _, _, "query.add_fact(\"alice\"") -> writeln('  PASS: embedded seed facts') ; (writeln('  FAIL'), fail)).

test_context_cli :-
    writeln('=== TEST: context(cli) defaults to stdin ==='),
    recursive_compiler:compile_recursive(ancestor/2, [target(lua), context(cli)], Code),
    (sub_string(Code, _, _, _, "io.lines()") -> writeln('  PASS: stdin') ; (writeln('  FAIL'), fail)).

test_context_override :-
    writeln('=== TEST: explicit input() overrides context ==='),
    recursive_compiler:compile_recursive(ancestor/2, [target(lua), context(notebook), input(stdin)], Code),
    (sub_string(Code, _, _, _, "io.lines()") -> writeln('  PASS: stdin overrides notebook') ; (writeln('  FAIL'), fail)).

%% --- Run all ---
run_tests :-
    %% Lua
    test_lua_stdin, test_lua_embedded, test_lua_vfs, test_lua_file, test_lua_function,
    %% Python
    test_python_stdin, test_python_embedded, test_python_vfs, test_python_function,
    %% R
    test_r_stdin, test_r_embedded, test_r_vfs,
    %% Bash
    test_bash_stdin, test_bash_embedded, test_bash_vfs, test_bash_file,
    %% Context defaults
    test_context_notebook, test_context_workbook, test_context_cli, test_context_override,
    nl, writeln('=== ALL 20 INPUT SOURCE TESTS PASSED ===').
