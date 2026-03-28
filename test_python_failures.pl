:- encoding(utf8).
:- ['src/unifyweaver/init'].
:- use_module('src/unifyweaver/core/recursive_compiler').

%% Systematic failure detection — only predicates we expect SHOULD work
%% but currently DON'T.

try(Label, Pred/Arity) :-
    format('~w: ', [Label]),
    (   catch(
            recursive_compiler:compile_recursive(Pred/Arity, [target(python)], _Code),
            _Error, fail)
    ->  writeln(ok)
    ;   writeln('FAIL')
    ).

%% --- Patterns that should work but might not ---

%% Negation as failure
:- dynamic not_member/2.
not_member(_, []).
not_member(X, [H|T]) :- X \== H, not_member(X, T).

%% Accumulator pattern (non-recursive base)
:- dynamic sum_pair/3.
sum_pair(X, Y, S) :- S is X + Y.

%% Multiple guards in conjunction
:- dynamic valid_age/1.
valid_age(X) :- X >= 0, X =< 150.

%% Guard with arithmetic output
:- dynamic checked_double/2.
checked_double(X, Y) :- X > 0, Y is X * 2.
checked_double(X, 0) :- X =< 0.

%% Ternary with arithmetic
:- dynamic abs_custom/2.
abs_custom(X, X) :- X >= 0.
abs_custom(X, Y) :- X < 0, Y is -X.

%% Nested output (computed value used in guard)
:- dynamic classify_temp/2.
classify_temp(T, freezing) :- T =< 0.
classify_temp(T, cold) :- T > 0, T < 15.
classify_temp(T, warm) :- T >= 15, T < 30.
classify_temp(T, hot) :- T >= 30.

%% Mixed constant and variable in head
:- dynamic direction/2.
direction(0, zero).
direction(X, positive) :- X > 0.
direction(X, negative) :- X < 0.

%% Arity 4+ with guards
:- dynamic between_check/4.
between_check(X, Lo, Hi, in_range) :- X >= Lo, X =< Hi.
between_check(_, _, _, out_of_range).

%% String-like output
:- dynamic color_name/2.
color_name(1, red).
color_name(2, green).
color_name(3, blue).
color_name(_, unknown).

run_failures :-
    try('sum_pair/3', sum_pair/3),
    try('valid_age/1', valid_age/1),
    try('checked_double/2', checked_double/2),
    try('abs_custom/2', abs_custom/2),
    try('classify_temp/2', classify_temp/2),
    try('direction/2', direction/2),
    try('between_check/4', between_check/4),
    try('color_name/2', color_name/2),
    nl, writeln('=== FAILURE SCAN DONE ===').
