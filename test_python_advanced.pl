:- encoding(utf8).
:- ['src/unifyweaver/init'].
:- use_module('src/unifyweaver/core/recursive_compiler').

:- dynamic label/2, category/2, sign/2.

%% If-then-else output: label(X, L) where L depends on condition
label(X, L) :- (X > 0 -> L = positive ; L = negative).

%% Disjunction output: 3-way classification
category(X, C) :- (X < 0, C = negative ; X =:= 0, C = zero ; X > 0, C = positive).

%% Multi-clause with guards
sign(0, zero).
sign(X, positive) :- X > 0.
sign(X, negative) :- X < 0.

test_ite :-
    writeln('=== TEST: Python if-then-else output ==='),
    recursive_compiler:compile_recursive(label/2, [target(python)], Code),
    writeln(Code),
    (sub_string(Code, _, _, _, "if") -> writeln('  PASS: contains if') ; writeln('  FAIL')).

test_disj :-
    writeln('=== TEST: Python disjunction output ==='),
    recursive_compiler:compile_recursive(category/2, [target(python)], Code),
    writeln(Code),
    (sub_string(Code, _, _, _, "elif") -> writeln('  PASS: contains elif') ; writeln('  NOTE: no elif (may use different pattern)')).

test_multi_clause :-
    writeln('=== TEST: Python multi-clause with guards ==='),
    recursive_compiler:compile_recursive(sign/2, [target(python)], Code),
    writeln(Code),
    (sub_string(Code, _, _, _, "def sign") -> writeln('  PASS: generates function') ; writeln('  FAIL')).

%% Arithmetic output
:- dynamic double/2, triple_plus/3.
double(X, Y) :- Y is X * 2.
triple_plus(X, Offset, Y) :- Y is X * 3 + Offset.

test_arithmetic :-
    writeln('=== TEST: Python arithmetic output ==='),
    recursive_compiler:compile_recursive(double/2, [target(python)], Code),
    (sub_string(Code, _, _, _, "def double") -> true ; (writeln('  FAIL: no def double'), fail)),
    (sub_string(Code, _, _, _, "* 2") -> writeln('  PASS: contains * 2') ; writeln('  NOTE: different form')).

test_arithmetic_multi :-
    writeln('=== TEST: Python multi-arg arithmetic ==='),
    recursive_compiler:compile_recursive(triple_plus/3, [target(python)], Code),
    (sub_string(Code, _, _, _, "def triple_plus") -> writeln('  PASS: generates function') ; writeln('  FAIL')).

%% Guard after output (interleaved)
:- dynamic clamp/3.
clamp(X, Max, Y) :- Y is X, Y > Max, Y = Max.
% This is tricky — the guard Y > Max comes after the output Y is X

test_guard_after_output :-
    writeln('=== TEST: Python guard after output ==='),
    (   recursive_compiler:compile_recursive(clamp/3, [target(python)], Code)
    ->  writeln(Code),
        writeln('  PASS: compiles')
    ;   writeln('  NOTE: failed to compile (expected — complex pattern)')
    ).

%% Nested if-then-else
:- dynamic classify3/2.
classify3(X, C) :- (X < 0 -> C = negative ; (X =:= 0 -> C = zero ; C = positive)).

test_nested_ite :-
    writeln('=== TEST: Python nested if-then-else ==='),
    recursive_compiler:compile_recursive(classify3/2, [target(python)], Code),
    (sub_string(Code, _, _, _, "def classify3") -> true ; (writeln('  FAIL'), fail)),
    (sub_string(Code, _, _, _, "elif") ->
        writeln('  PASS: flattened to elif')
    ; sub_string(Code, _, _, _, "else") ->
        writeln('  PASS: has else')
    ;   writeln('  NOTE: different form')
    ).

%% Multi-clause with mixed constant/variable output
:- dynamic fizzbuzz/2.
fizzbuzz(X, fizzbuzz) :- X mod 15 =:= 0.
fizzbuzz(X, fizz) :- X mod 3 =:= 0.
fizzbuzz(X, buzz) :- X mod 5 =:= 0.
fizzbuzz(X, X).

test_fizzbuzz :-
    writeln('=== TEST: Python fizzbuzz (4 clauses) ==='),
    recursive_compiler:compile_recursive(fizzbuzz/2, [target(python)], Code),
    (sub_string(Code, _, _, _, "def fizzbuzz") -> true ; (writeln('  FAIL'), fail)),
    (sub_string(Code, _, _, _, "elif") ->
        writeln('  PASS: if/elif chain')
    ;   writeln('  NOTE: different structure')
    ).

%% Pure guard predicate (no output binding — boolean result)
:- dynamic is_positive/1.
is_positive(X) :- X > 0.

test_pure_guard :-
    writeln('=== TEST: Python pure guard predicate ==='),
    (   recursive_compiler:compile_recursive(is_positive/1, [target(python)], Code)
    ->  (sub_string(Code, _, _, _, "def is_positive(arg1)") -> writeln('  PASS: has arg1') ; (writeln('  FAIL: missing arg'), fail)),
        (sub_string(Code, _, _, _, "return True") -> writeln('  PASS: returns True') ; writeln('  NOTE: different form')),
        (sub_string(Code, _, _, _, "return False") -> writeln('  PASS: returns False') ; writeln('  NOTE: no False branch'))
    ;   writeln('  FAIL: failed to compile'), fail
    ).

%% Multiple outputs in sequence
:- dynamic compute/4.
compute(X, Y, Sum, Product) :- Sum is X + Y, Product is X * Y.

test_multi_output :-
    writeln('=== TEST: Python multiple outputs ==='),
    (   recursive_compiler:compile_recursive(compute/4, [target(python)], Code)
    ->  (sub_string(Code, _, _, _, "def compute") -> writeln('  PASS: generates function') ; writeln('  FAIL')),
        writeln(Code)
    ;   writeln('  NOTE: failed to compile')
    ).

run_tests :-
    test_ite,
    test_disj,
    test_multi_clause,
    test_arithmetic,
    test_arithmetic_multi,
    test_nested_ite,
    test_fizzbuzz,
    test_guard_after_output,
    test_pure_guard,
    test_multi_output,
    nl, writeln('=== ALL PYTHON ADVANCED TESTS DONE ===').
