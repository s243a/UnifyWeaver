:- encoding(utf8).
:- ['src/unifyweaver/init'].
:- use_module('src/unifyweaver/core/recursive_compiler').
:- use_module('src/unifyweaver/core/clause_body_analysis').

%% ========================================================
%% Test predicates — increasing complexity
%% ========================================================

%% --- Group 1: Multiple sequential outputs ---
:- dynamic swap/4, min_max/4.
swap(X, Y, Y, X).
min_max(X, Y, X, Y) :- X =< Y.
min_max(X, Y, Y, X) :- X > Y.

%% --- Group 2: Chained arithmetic ---
:- dynamic celsius_to_fahrenheit/2, bmi/3.
celsius_to_fahrenheit(C, F) :- F is C * 9 / 5 + 32.
bmi(Weight, Height, BMI) :- BMI is Weight / (Height * Height).

%% --- Group 3: Guard with fallback default ---
:- dynamic safe_sqrt/2, clamp_range/4.
safe_sqrt(X, Y) :- X >= 0, Y is sqrt(X).
safe_sqrt(X, -1) :- X < 0.
clamp_range(X, Lo, Hi, Lo) :- X < Lo.
clamp_range(X, _Lo, Hi, Hi) :- X > Hi.
clamp_range(X, _Lo, _Hi, X) :- true.

%% --- Group 4: String/atom output ---
:- dynamic greet/2, day_type/2.
greet(hello, 'Hello, World!').
greet(goodbye, 'Farewell!').
greet(_, 'Hi!').
day_type(saturday, weekend).
day_type(sunday, weekend).
day_type(_, weekday).

%% --- Group 5: Nested guard conditions ---
:- dynamic tax_bracket/2, letter_grade/2.
tax_bracket(Income, Rate) :-
    (Income =< 10000 -> Rate = 0.0
    ; Income =< 50000 -> Rate = 0.2
    ; Income =< 100000 -> Rate = 0.3
    ; Rate = 0.4).
letter_grade(Score, Grade) :-
    (Score >= 90 -> Grade = 'A'
    ; Score >= 80 -> Grade = 'B'
    ; Score >= 70 -> Grade = 'C'
    ; Score >= 60 -> Grade = 'D'
    ; Grade = 'F').

%% --- Group 6: Mixed guard + output in body ---
:- dynamic safe_divide/3, factorial_base/2.
safe_divide(X, Y, Result) :- Y =\= 0, Result is X / Y.
safe_divide(_, 0, 0).
factorial_base(0, 1).
factorial_base(1, 1).

%% --- Group 7: Arity-1 predicates (boolean) ---
:- dynamic is_even/1, is_adult/1, is_vowel/1.
is_even(X) :- X mod 2 =:= 0.
is_adult(Age) :- Age >= 18.
is_vowel(a). is_vowel(e). is_vowel(i). is_vowel(o). is_vowel(u).

%% ========================================================
%% Test runner
%% ========================================================

try_compile(Pred/Arity, Target, Code, Status) :-
    (   catch(
            recursive_compiler:compile_recursive(Pred/Arity, [target(Target)], Code),
            _Error,
            fail
        )
    ->  Status = ok
    ;   Status = fail,
        Code = ""
    ).

check(Label, Code, Substring) :-
    (   sub_string(Code, _, _, _, Substring)
    ->  format('  PASS: ~w~n', [Label])
    ;   format('  MISS: ~w (no "~w")~n', [Label, Substring])
    ).

show_func(Code, FuncName) :-
    split_string(Code, "\n", "", Lines),
    format(atom(Prefix), "def ~w", [FuncName]),
    (   nth1(I, Lines, L), sub_string(L, _, _, _, Prefix)
    ->  Start is I, End is min(I + 8, 99999),
        forall((between(Start, End, J), nth1(J, Lines, LJ),
                (LJ \= "" ; J =:= Start)),
               (format('    ~w~n', [LJ])))
    ;   format('    (function ~w not found)~n', [FuncName])
    ).

%% --- Tests ---

test_swap :-
    writeln('=== swap/4 (pure fact, multi-output) ==='),
    try_compile(swap/4, python, Code, Status),
    (Status = ok -> show_func(Code, swap), check('has def', Code, "def swap") ; writeln('  SKIP: failed')).

test_min_max :-
    writeln('=== min_max/4 (multi-clause, multi-output from head) ==='),
    try_compile(min_max/4, python, Code, Status),
    (Status = ok -> show_func(Code, min_max), check('has def', Code, "def min_max") ; writeln('  SKIP: failed')).

test_celsius :-
    writeln('=== celsius_to_fahrenheit/2 (chained arithmetic) ==='),
    try_compile(celsius_to_fahrenheit/2, python, Code, Status),
    (Status = ok -> show_func(Code, celsius_to_fahrenheit), check('has * 9', Code, "* 9") ; writeln('  SKIP: failed')).

test_bmi :-
    writeln('=== bmi/3 (3-arg arithmetic) ==='),
    try_compile(bmi/3, python, Code, Status),
    (Status = ok -> show_func(Code, bmi), check('has def', Code, "def bmi") ; writeln('  SKIP: failed')).

test_safe_sqrt :-
    writeln('=== safe_sqrt/2 (guard + fallback) ==='),
    try_compile(safe_sqrt/2, python, Code, Status),
    (Status = ok -> show_func(Code, safe_sqrt), check('has sqrt', Code, "sqrt") ; writeln('  SKIP: failed')).

test_clamp_range :-
    writeln('=== clamp_range/4 (3-clause, head-arg output) ==='),
    try_compile(clamp_range/4, python, Code, Status),
    (Status = ok -> show_func(Code, clamp_range), check('has def', Code, "def clamp_range") ; writeln('  SKIP: failed')).

test_greet :-
    writeln('=== greet/2 (atom matching, string output) ==='),
    try_compile(greet/2, python, Code, Status),
    (Status = ok -> show_func(Code, greet),
        check('Hello', Code, "Hello, World!"),
        check('Farewell', Code, "Farewell!")
    ; writeln('  SKIP: failed')).

test_day_type :-
    writeln('=== day_type/2 (atom matching, atom output) ==='),
    try_compile(day_type/2, python, Code, Status),
    (Status = ok -> show_func(Code, day_type),
        check('weekend', Code, "weekend"),
        check('weekday', Code, "weekday")
    ; writeln('  SKIP: failed')).

test_tax_bracket :-
    writeln('=== tax_bracket/2 (nested if-then-else, 4 brackets) ==='),
    try_compile(tax_bracket/2, python, Code, Status),
    (Status = ok -> show_func(Code, tax_bracket),
        check('elif', Code, "elif"),
        check('0.4', Code, "0.4")
    ; writeln('  SKIP: failed')).

test_letter_grade :-
    writeln('=== letter_grade/2 (5-way nested if-then-else) ==='),
    try_compile(letter_grade/2, python, Code, Status),
    (Status = ok -> show_func(Code, letter_grade),
        check('elif', Code, "elif"),
        check('grade F', Code, "F")
    ; writeln('  SKIP: failed')).

test_safe_divide :-
    writeln('=== safe_divide/3 (guard + output, fallback clause) ==='),
    try_compile(safe_divide/3, python, Code, Status),
    (Status = ok -> show_func(Code, safe_divide),
        check('has def', Code, "def safe_divide")
    ; writeln('  SKIP: failed')).

test_factorial_base :-
    writeln('=== factorial_base/2 (pure fact multi-clause) ==='),
    try_compile(factorial_base/2, python, Code, Status),
    (Status = ok -> show_func(Code, factorial_base),
        check('return 1', Code, "return 1")
    ; writeln('  SKIP: failed')).

test_is_even :-
    writeln('=== is_even/1 (arity-1, mod guard) ==='),
    try_compile(is_even/1, python, Code, Status),
    (Status = ok -> show_func(Code, is_even),
        check('has arg1', Code, "def is_even(arg1)"),
        check('mod or %', Code, "%")
    ; writeln('  SKIP: failed')).

test_is_adult :-
    writeln('=== is_adult/1 (arity-1, comparison) ==='),
    try_compile(is_adult/1, python, Code, Status),
    (Status = ok -> show_func(Code, is_adult),
        check('has arg1', Code, "def is_adult(arg1)"),
        check('>= 18', Code, ">= 18")
    ; writeln('  SKIP: failed')).

test_is_vowel :-
    writeln('=== is_vowel/1 (arity-1, multi-clause fact matching) ==='),
    try_compile(is_vowel/1, python, Code, Status),
    (Status = ok -> show_func(Code, is_vowel),
        check('has def', Code, "def is_vowel(arg1)")
    ; writeln('  SKIP: failed')).

%% ========================================================
%% Group 8: Negation as failure / \+
%% ========================================================
:- dynamic is_odd/1, not_zero/1.
is_odd(X) :- X mod 2 =\= 0.
not_zero(X) :- X =\= 0.

test_is_odd :-
    writeln('=== is_odd/1 (=\\= guard) ==='),
    try_compile(is_odd/1, python, Code, Status),
    (Status = ok -> show_func(Code, is_odd), check('has !=', Code, "!=") ; writeln('  SKIP')).

test_not_zero :-
    writeln('=== not_zero/1 (simple !=) ==='),
    try_compile(not_zero/1, python, Code, Status),
    (Status = ok -> show_func(Code, not_zero), check('has !=', Code, "!= 0") ; writeln('  SKIP')).

%% ========================================================
%% Group 9: Multiple guards combined
%% ========================================================
:- dynamic in_range/3, triangle_type/4.
in_range(X, Lo, Hi) :- X >= Lo, X =< Hi.
triangle_type(A, B, C, equilateral) :- A =:= B, B =:= C.
triangle_type(A, B, C, isosceles) :- A =:= B ; B =:= C ; A =:= C.
triangle_type(_, _, _, scalene).

test_in_range :-
    writeln('=== in_range/3 (multiple guards, arity-3 boolean) ==='),
    try_compile(in_range/3, python, Code, Status),
    (Status = ok -> show_func(Code, in_range),
        check('has >= and <=', Code, ">="),
        check('has arg1', Code, "def in_range(arg1")
    ; writeln('  SKIP')).

test_triangle :-
    writeln('=== triangle_type/4 (multi-clause, guards + disjunction) ==='),
    try_compile(triangle_type/4, python, Code, Status),
    (Status = ok -> show_func(Code, triangle_type),
        check('equilateral', Code, "equilateral"),
        check('scalene', Code, "scalene")
    ; writeln('  SKIP')).

%% ========================================================
%% Group 10: Intermediate computations
%% ========================================================
:- dynamic hypotenuse/3, circle_area/2, discount_price/3.
hypotenuse(A, B, C) :- C is sqrt(A * A + B * B).
circle_area(R, Area) :- Area is 3.14159 * R * R.
discount_price(Price, Pct, Final) :- Final is Price * (1 - Pct / 100).

test_hypotenuse :-
    writeln('=== hypotenuse/3 (nested arithmetic with sqrt) ==='),
    try_compile(hypotenuse/3, python, Code, Status),
    (Status = ok -> show_func(Code, hypotenuse), check('sqrt', Code, "sqrt") ; writeln('  SKIP')).

test_circle_area :-
    writeln('=== circle_area/2 (float constant) ==='),
    try_compile(circle_area/2, python, Code, Status),
    (Status = ok -> show_func(Code, circle_area), check('3.14159', Code, "3.14159") ; writeln('  SKIP')).

test_discount :-
    writeln('=== discount_price/3 (compound arithmetic) ==='),
    try_compile(discount_price/3, python, Code, Status),
    (Status = ok -> show_func(Code, discount_price), check('has def', Code, "def discount_price") ; writeln('  SKIP')).

%% ========================================================
%% Group 11: Predicate with multiple returns in different clauses
%% ========================================================
:- dynamic http_status/2.
http_status(200, ok).
http_status(301, redirect).
http_status(404, not_found).
http_status(500, server_error).
http_status(_, unknown).

test_http_status :-
    writeln('=== http_status/2 (5-clause lookup table) ==='),
    try_compile(http_status/2, python, Code, Status),
    (Status = ok -> show_func(Code, http_status),
        check('200', Code, "200"),
        check('not_found', Code, "not_found"),
        check('unknown', Code, "unknown")
    ; writeln('  SKIP')).

run_tests :-
    test_swap,
    test_min_max,
    test_celsius,
    test_bmi,
    test_safe_sqrt,
    test_clamp_range,
    test_greet,
    test_day_type,
    test_tax_bracket,
    test_letter_grade,
    test_safe_divide,
    test_factorial_base,
    test_is_even,
    test_is_adult,
    test_is_vowel,
    %% Round 2b
    test_is_odd,
    test_not_zero,
    test_in_range,
    test_triangle,
    test_hypotenuse,
    test_circle_area,
    test_discount,
    test_http_status,
    nl, writeln('=== ALL 23 ROUND 2 TESTS DONE ===').
