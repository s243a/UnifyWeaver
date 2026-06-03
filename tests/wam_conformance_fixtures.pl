:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_conformance_fixtures.pl
%
% Shared classic-program fixtures for the cross-target WAM conformance
% harness (test_wam_cross_target_conformance.pl). One source of truth for
% the programs and their expected query results, so every WAM backend is
% checked against the SAME spec rather than re-declaring fixtures per
% target. This is what catches a backend that silently diverges from the
% others (e.g. the Haskell `Proceed` / WAT `allocate` first-arg-indexing
% bugs, where member/2 wrongly succeeded).
%
% DESIGN — coverage vs CI speed. The dominant cost in this harness is
% process startup (scalac/JVM/BEAM/node), not arithmetic. So fixtures
% favour cheap, high-coverage shapes over heavy recursion:
%   - `member` (a set operation) is high-level but far cheaper than a
%     generic recursive algorithm — the preferred everyday case.
%   - `builtins` packs the common arithmetic/comparison/unification
%     builtins into a handful of near-zero-compute queries.
%   - the recursive samples (fib, ack) are kept to SMALL inputs; the
%     harness compiles all predicates into ONE project per target (one
%     compile) and supports CONFORMANCE_SAMPLE for random query
%     subsetting, so a CI run stays fast while coverage accumulates
%     across runs.
%
% A fixture is:
%   conformance_program(Name, Preds)
%     Name   - atom identifying the program
%     Preds  - list of Module:Name/Arity indicators to compile (the
%              program clauses are asserted into `user` at load, below)
%   conformance_query(Name, PredKey, Args, Expected)
%     PredKey  - 'name/arity' atom (the queried predicate)
%     Args     - list of ground PROLOG TERMS (ints, atoms, atom-lists).
%                Kept as terms (not strings) so each adapter can render
%                them to its own driver contract, or synthesise a goal.
%     Expected - `true` or `false` (does the ground query hold?)
%
% Programs are intentionally self-contained (no list-library builtins
% beyond arithmetic) so every target compiles them identically.

:- module(wam_conformance_fixtures,
          [ conformance_program/2,
            conformance_query/4
          ]).

% ============================================================
% Program clauses (asserted into user:)
% ============================================================

% --- member/2 — first-argument indexing + list/structure matching ---
:- dynamic user:cmem/2.
user:cmem(X, [X|_]).
user:cmem(X, [_|T]) :- user:cmem(X, T).

% --- append/3 — list construction + recursion + backtracking ---
:- dynamic user:capp/3.
user:capp([], L, L).
user:capp([H|T], L, [H|R]) :- user:capp(T, L, R).

% --- list reverse via accumulator (linear) ---
:- dynamic user:crev_acc/3.
:- dynamic user:clist_reverse/2.
user:crev_acc([], A, A).
user:crev_acc([H|T], A, R) :- user:crev_acc(T, [H|A], R).
user:clist_reverse(L, R) :- user:crev_acc(L, [], R).

% --- Fibonacci (naïve doubly-recursive) — arithmetic + recursion ---
:- dynamic user:cfib/2.
user:cfib(0, 0).
user:cfib(1, 1).
user:cfib(N, R) :- N > 1, N1 is N - 1, N2 is N - 2,
                   user:cfib(N1, R1), user:cfib(N2, R2),
                   R is R1 + R2.

% --- Ackermann — recursion + arithmetic comparison. Kept to SMALL
%     inputs for CI speed (ack(3,3) is correct but ~2400 calls; we use
%     ack(2,_) ≈ a few dozen calls instead). ---
:- dynamic user:cack/3.
user:cack(0, N, R) :- R is N + 1.
user:cack(M, 0, R) :- M > 0, M1 is M - 1, user:cack(M1, 1, R).
user:cack(M, N, R) :- M > 0, N > 0,
                      M1 is M - 1, N1 is N - 1,
                      user:cack(M, N1, R1),
                      user:cack(M1, R1, R).

% --- builtins — packs the common arithmetic / comparison / unification
%     builtins into near-zero-compute queries. Broad coverage, cheap. ---
:- dynamic user:cbi_arith/1.
:- dynamic user:cbi_cmp/1.
:- dynamic user:cbi_eq/1.
% +, -, *, integer div, mod folded into one result: 5+6+12+3+2 = 28.
user:cbi_arith(R) :- A is 2 + 3, B is 10 - 4, C is 3 * 4,
                     D is 17 // 5, E is 17 mod 5,
                     R is A + B + C + D + E.
% the comparison family: >, <, >=, =<, =:=, =\= (1-arity so it succeeds
% only for N=5; kept off 0-arity, which trips a Scala-target codegen
% hang on comparison-only bodies).
user:cbi_cmp(N) :- N > 0, N < 10, N >= 1, N =< 9, N =:= 5, N =\= 4.
% term unification =/2.
user:cbi_eq(X) :- X = foo.

% ============================================================
% Fixture registry
% ============================================================

conformance_program(member,   [user:cmem/2]).
conformance_program(append,   [user:capp/3]).
conformance_program(reverse,  [user:crev_acc/3, user:clist_reverse/2]).
conformance_program(fib,      [user:cfib/2]).
conformance_program(ack,      [user:cack/3]).
conformance_program(builtins, [user:cbi_arith/1, user:cbi_cmp/1, user:cbi_eq/1]).

% member/2 — the regression that motivated the harness; the preferred
% cheap everyday case (set operation, first-arg indexing, backtracking).
conformance_query(member, 'cmem/2', [a, [a,b,c]], true).
conformance_query(member, 'cmem/2', [b, [a,b,c]], true).
conformance_query(member, 'cmem/2', [c, [a,b,c]], true).
conformance_query(member, 'cmem/2', [z, [a,b,c]], false).
conformance_query(member, 'cmem/2', [a, []],      false).

% append/3
conformance_query(append, 'capp/3', [[a,b], [c],   [a,b,c]], true).
conformance_query(append, 'capp/3', [[],    [a],   [a]],     true).
conformance_query(append, 'capp/3', [[a],   [b],   [a,b]],   true).
conformance_query(append, 'capp/3', [[a],   [b],   [b,a]],   false).

% reverse/2
conformance_query(reverse, 'clist_reverse/2', [[a,b,c], [c,b,a]], true).
conformance_query(reverse, 'clist_reverse/2', [[],      []],      true).
conformance_query(reverse, 'clist_reverse/2', [[a],     [a]],     true).
conformance_query(reverse, 'clist_reverse/2', [[a,b,c], [a,b,c]], false).

% fib/2 — one small recursive sample (fib(10)=55, ~177 calls).
conformance_query(fib, 'cfib/2', [0,  0],  true).
conformance_query(fib, 'cfib/2', [10, 55], true).
conformance_query(fib, 'cfib/2', [10, 54], false).

% ack/3 — small inputs only (ack(2,n)=2n+3).
conformance_query(ack, 'cack/3', [0, 5, 6], true).
conformance_query(ack, 'cack/3', [2, 3, 9], true).
conformance_query(ack, 'cack/3', [2, 3, 8], false).

% builtins — arithmetic, comparison, unification (all near-zero compute).
conformance_query(builtins, 'cbi_arith/1', [28],  true).
conformance_query(builtins, 'cbi_arith/1', [27],  false).
conformance_query(builtins, 'cbi_cmp/1',   [5],   true).
conformance_query(builtins, 'cbi_cmp/1',   [4],   false).
conformance_query(builtins, 'cbi_eq/1',    [foo], true).
conformance_query(builtins, 'cbi_eq/1',    [bar], false).
