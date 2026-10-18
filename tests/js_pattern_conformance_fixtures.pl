:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% js_pattern_conformance_fixtures.pl
%
% Shared fixtures for the cross-target parity harness for the JavaScript
% *pattern* targets (test_js_pattern_cross_target_conformance.pl). One
% source of truth for the programs and their expected results, so every JS
% pattern backend (typescript, annotated_js, vanilla_js, clojurescript) is
% checked against the SAME spec rather than re-declaring fixtures per
% target. This is the JS-pattern analogue of wam_conformance_fixtures.pl
% (which serves the WAM backends): a safety net against one backend
% silently diverging from the shared Prolog semantics.
%
% ---------------------------------------------------------------------------
% CONTRACT DIFFERENCE FROM THE WAM FIXTURES
% ---------------------------------------------------------------------------
% The WAM fixtures use a BOOLEAN oracle: a ground query either holds or it
% does not, and every backend runs a 0-arity success/failure wrapper. The
% JS pattern targets are *functional* -- their generated code is a function
% that RETURNS a value (fib(10) -> 55). So the numeric fixtures here use a
% VALUE oracle instead:
%
%   js_conformance_query(Program, Inputs, Expected)
%     Inputs   - list of ground input terms passed as the function's args
%                (Prolog terms: ints, or an int-list for the fold case).
%     Expected - the value the function must return (numeric programs), or
%                `true`/`false` for the structural programs (see below).
%
% Expected results are hand-specified from standard Prolog semantics and
% cross-checked against a live Prolog oracle at test time by js_oracle/3
% (the `js_oracle_self_check` test in the harness runs every program's
% clauses and asserts the hand-specified Expected matches). That keeps the
% "Prolog-semantics oracle" honest and gives the suite a meaningful green
% test even when no JS toolchain is installed.
%
% ---------------------------------------------------------------------------
% PROGRAM FAMILIES (what a pattern target can actually compile)
% ---------------------------------------------------------------------------
% The pattern JS targets are NOT full WAM compilers. TypeScript's robust,
% runnable path is the recursion-pattern codegen (compile_recursion/3 with a
% pattern hint, and compile_module/3 for factorial), which emits clean,
% tsc-clean, node-runnable functions. So programs are tagged by family:
%
%   numeric    - a recursive numeric/list-fold function whose value can be
%                compiled by the recursion-pattern path and run under node.
%                This is the family the pattern targets exercise today.
%   structural - a classic list / head-shape predicate (member, append,
%                reverse, nested head). No pattern JS target compiles these
%                today; they are carried here so (a) the spec honours the
%                WAM-fixture philosophy (classics + a head-shape case) and
%                (b) the fuller annotated_js / vanilla_js targets, being
%                built in parallel, have a ready target to satisfy. The
%                pattern-target arms SKIP the structural family cleanly.
%
% A fixture program:
%   js_conformance_program(Name, Family, Pattern, FnName, ArgKind, Preds)
%     Name    - atom identifying the program.
%     Family  - numeric | structural (see above).
%     Pattern - codegen shape hint for the numeric family
%               (linear_recursion | tail_recursion | list_fold | factorial),
%               or `structural` for the structural family.
%     FnName  - the function name the generated code should define / export.
%     ArgKind - how the driver passes the input(s): num | numlist |
%               structural.
%     Preds   - Module:Name/Arity indicators of the Prolog oracle clauses
%               (asserted into `user` at load, below).

:- module(js_pattern_conformance_fixtures,
          [ js_conformance_program/6,
            js_conformance_query/3,
            js_oracle/3
          ]).

:- use_module(library(lists)).

% ============================================================
% Program clauses (the Prolog oracle, asserted into user:)
% ============================================================

% --- fib/2 — naive doubly-recursive Fibonacci (linear_recursion pattern) ---
:- dynamic user:cfib/2.
user:cfib(0, 0).
user:cfib(1, 1).
user:cfib(N, R) :- N > 1, N1 is N - 1, N2 is N - 2,
                   user:cfib(N1, R1), user:cfib(N2, R2),
                   R is R1 + R2.

% --- factorial/2 (factorial module pattern) ---
:- dynamic user:cfac/2.
user:cfac(0, 1).
user:cfac(N, R) :- N > 0, N1 is N - 1, user:cfac(N1, R1), R is N * R1.

% --- sum/2 — sum 1..N (tail_recursion pattern: the canned template folds
%     acc += n for n in N..1, i.e. the Nth triangular number). ---
:- dynamic user:csum/2.
user:csum(0, 0).
user:csum(N, R) :- N > 0, N1 is N - 1, user:csum(N1, R1), R is R1 + N.

% --- listsum/2 — sum of a list of numbers (list_fold pattern: reduce +). ---
:- dynamic user:clistsum/2.
user:clistsum([], 0).
user:clistsum([H|T], R) :- user:clistsum(T, R1), R is R1 + H.

% --- member/2 (structural: first-argument indexing + list matching) ---
:- dynamic user:cmem/2.
user:cmem(X, [X|_]).
user:cmem(X, [_|T]) :- user:cmem(X, T).

% --- append/3 (structural: list construction + recursion) ---
:- dynamic user:capp/3.
user:capp([], L, L).
user:capp([H|T], L, [H|R]) :- user:capp(T, L, R).

% --- reverse/2 via accumulator (structural) ---
:- dynamic user:crev_acc/3.
:- dynamic user:crev/2.
user:crev_acc([], A, A).
user:crev_acc([H|T], A, R) :- user:crev_acc(T, [H|A], R).
user:crev(L, R) :- user:crev_acc(L, [], R).

% --- head-shape: structure nested inside a cons head, discriminated on the
%     inner functor across clauses (the tokenizer/parser shape). ---
:- dynamic user:ckind/2.
user:ckind([cnum(V)|_], n(V)).
user:ckind([csym(V)|_], s(V)).
user:ckind([cwd(V)|_],  w(V)).

% ============================================================
% Fixture registry
% ============================================================

% Numeric family — the recursion-pattern path the pattern targets exercise.
js_conformance_program(fib,       numeric,    linear_recursion, fib,     num,     [user:cfib/2]).
js_conformance_program(factorial, numeric,    factorial,        fac,     num,     [user:cfac/2]).
js_conformance_program(sum,       numeric,    tail_recursion,   sum,     num,     [user:csum/2]).
js_conformance_program(listsum,   numeric,    list_fold,        listSum, numlist, [user:clistsum/2]).

% Structural family — classics + a head-shape case. Carried for the fuller
% annotated_js / vanilla_js targets; the pattern targets skip these.
js_conformance_program(member,    structural, structural,       cmem,    structural, [user:cmem/2]).
js_conformance_program(append,    structural, structural,       capp,    structural, [user:capp/3]).
js_conformance_program(reverse,   structural, structural,       crev,    structural, [user:crev/2, user:crev_acc/3]).
js_conformance_program(headshape, structural, structural,       ckind,   structural, [user:ckind/2]).

% ------------------------------------------------------------
% Queries — numeric family (VALUE oracle: Inputs -> returned value)
% ------------------------------------------------------------
js_conformance_query(fib, [0],  0).
js_conformance_query(fib, [1],  1).
js_conformance_query(fib, [7],  13).
js_conformance_query(fib, [10], 55).

js_conformance_query(factorial, [0], 1).
js_conformance_query(factorial, [1], 1).
js_conformance_query(factorial, [5], 120).
js_conformance_query(factorial, [6], 720).

js_conformance_query(sum, [0],  0).
js_conformance_query(sum, [5],  15).
js_conformance_query(sum, [10], 55).

js_conformance_query(listsum, [[1,2,3,4]],  10).
js_conformance_query(listsum, [[10,20,30]], 60).
js_conformance_query(listsum, [[]],         0).

% ------------------------------------------------------------
% Queries — structural family (BOOLEAN oracle: does the ground query hold?)
% Reserved for the fuller JS targets; pattern targets skip these.
% ------------------------------------------------------------
js_conformance_query(member, [a, [a,b,c]], true).
js_conformance_query(member, [c, [a,b,c]], true).
js_conformance_query(member, [z, [a,b,c]], false).

js_conformance_query(append, [[a,b], [c], [a,b,c]], true).
js_conformance_query(append, [[a],   [b], [b,a]],   false).

js_conformance_query(reverse, [[a,b,c], [c,b,a]], true).
js_conformance_query(reverse, [[a,b,c], [a,b,c]], false).

js_conformance_query(headshape, [[cnum(7)], n(7)], true).
js_conformance_query(headshape, [[csym(x)], s(x)], true).
js_conformance_query(headshape, [[csym(x)], n(x)], false).

% ============================================================
% Prolog oracle
% ============================================================

%% js_oracle(+Program, +Inputs, -Value)
%  Evaluate Program's Prolog clauses against Inputs and yield the oracle
%  Value the generated code must reproduce. For the numeric family the
%  predicate's last argument is the output, so Value is that binding. For
%  the structural family every argument is ground and Value is true/false
%  (whether the goal holds). Used by the harness's self-check test to prove
%  the hand-specified Expected values match standard Prolog semantics.
js_oracle(Program, Inputs, Value) :-
    js_conformance_program(Program, Family, _Pattern, _Fn, _ArgKind, Preds),
    Preds = [_Mod:Name/Arity | _],
    (   Family == numeric
    ->  append(Inputs, [Out], AllArgs),
        length(AllArgs, Arity),
        Goal =.. [Name | AllArgs],
        once(user:Goal),
        Value = Out
    ;   % structural: all inputs ground, boolean result
        length(Inputs, Arity),
        Goal =.. [Name | Inputs],
        ( user:Goal -> Value = true ; Value = false )
    ).
