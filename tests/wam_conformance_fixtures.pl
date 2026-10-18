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

% --- wide/10 — argument registers ABOVE A8 ---
%     Most WAM runtimes special-case a small window of argument
%     registers. A predicate of arity > 8 puts arguments outside that
%     window, and the first clause here fails on argument 9, so the
%     second clause is reached ONLY if A9 and A10 survive backtracking.
%     WAM-Go silently lost them: snapshotAllRegs saved Regs[0..7] and
%     skipped Regs[8..199] as "the X-register range", but A(N) maps to
%     Regs[N-1] and X starts at Regs[100]. Every classic program in this
%     file has arity <= 3, so nothing caught it.
:- dynamic user:cwide/10.
user:cwide(_,_,_,_,_,_,_,_, no, no).
user:cwide(A,B,C,D,E,F,G,H,I,S) :- S is A+B+C+D+E+F+G+H+I.

% --- nested structures inside a cons-cell head ---
%     `[tk(X)|R]` in a head argument makes the runtime destructure a
%     list AND a structure nested inside it, in that order. That is the
%     shape every tokenizer/parser uses, and it exercises two hazards
%     the flat classics never reach:
%       1. the nested get_structure must not clobber the enclosing list
%          context — the cons TAIL still has to be unified afterwards;
%       2. get_structure must dereference before choosing read vs write
%          mode, or an already-bound argument gets a fresh structure
%          built over the top of it.
%     ctail/3 pins the tail specifically; ckind/2 adds multi-clause
%     discrimination on the *inner* functor.
:- dynamic user:cnest/2.
:- dynamic user:ctail/3.
:- dynamic user:ckind/2.
user:cnest([tk(X)|_], X).
user:cnest([_|T], X) :- user:cnest(T, X).
user:ctail([tk(X)|R], X, R).
user:ckind([cnum(V)|_], n(V)).
user:ckind([csym(V)|_], s(V)).
user:ckind([cwd(V)|_],  w(V)).
% Same shape in WRITE mode. Every query in this harness is ground, so a
% head is normally matched, never built — which hides construction bugs
% completely. cbuild_chk/1 calls cbuild_one/2 with its second argument
% UNBOUND, so the runtime has to *build* `[tk(X)|[]]`, and then checks
% the result. WAM-C reserved the cons cells but allocated the nested
% tk/1 structure on top of them, so the tail slot was never written and
% every such query silently failed.
:- dynamic user:cbuild_one/2.
:- dynamic user:cbuild_chk/1.
% NOTE the `==/2`. This check used `=/2` originally, which made it blind
% to the very bug it was written for: if the head never binds the
% caller's L at all, `L = [tk(X)]` simply binds it on the spot and
% reports success. Only an identity comparison proves the head did the
% construction. WAM-Scala passed this query for months while
% `finalizeBuild` skipped the bind-through for A registers, so a clause
% head that builds an output argument never reached its caller — which
% turned `select_uw(H, [X|T], [X|T2])` into a non-terminating generator.
% Same trap `crepeat/1` documents below; same fix.
%
% These two live in their OWN program (`buildnest`), not in `nested`.
% They are the only write-mode queries in the suite, and a backend can
% match nested heads perfectly while failing to build them — Elixir and
% R do exactly that today. Sharing a program would let one xfail blanket
% the read-mode queries too, which is the mistake `repeatvar` was split
% out to avoid.
user:cbuild_one(X, [tk(X)|[]]).
user:cbuild_chk(X) :- user:cbuild_one(X, L), L == [tk(X)].
% A head with a REPEATED variable inside a structure. Matching
% `csame(p(X,X), X)` against a term whose slots are both unbound has to
% bind the second slot, not merely accept it. WAM-Rust's unify_value
% short-circuited on an unbound heap argument and returned success
% without binding, so the caller's term never became p(a,a) — the query
% failed while every individual step "succeeded".
%
% crepeat/1 has to use ==/2 rather than =/2: with the bug, T was left as
% p(_, a), and `T = p(a,a)` would happily bind the remaining slot and
% succeed. Only an identity comparison shows that the binding never
% happened. That does mean crepeat/1 depends on the backend's ==/2
% dereferencing correctly — which is why it lives in its own program.
:- dynamic user:csame/2.
:- dynamic user:crepeat/1.
user:csame(p(X, X), X).
user:crepeat(X) :- T = p(_, _), user:csame(T, X), T == p(X, X).

% --- empty-list identity ---
%     `[]` reached as a cons tail must unify with a literal `[]`. These
%     can be different objects inside a runtime (an interned atom, a
%     package-level constant, an empty list value), and if equality is
%     identity-based they silently differ. WAM-Go ended up with two
%     distinct `[]` atoms because the codegen registered its interned
%     atoms from a Go `init()`, which runs after package-level variable
%     initialisers had already cached a different one.
:- dynamic user:cnil_tail/2.
:- dynamic user:cone/2.
user:cnil_tail([_|T], T).
user:cone(X, [X|[]]).

% ============================================================
% Fixture registry
% ============================================================

conformance_program(member,   [user:cmem/2]).
conformance_program(append,   [user:capp/3]).
conformance_program(reverse,  [user:crev_acc/3, user:clist_reverse/2]).
conformance_program(fib,      [user:cfib/2]).
conformance_program(ack,      [user:cack/3]).
conformance_program(builtins, [user:cbi_arith/1, user:cbi_cmp/1, user:cbi_eq/1]).
conformance_program(wide,      [user:cwide/10]).
conformance_program(nested,    [user:cnest/2, user:ctail/3, user:ckind/2]).
conformance_program(buildnest, [user:cbuild_one/2, user:cbuild_chk/1]).
conformance_program(repeatvar, [user:csame/2, user:crepeat/1]).
conformance_program(emptylist, [user:cnil_tail/2, user:cone/2]).

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

% wide/10 — arguments above A8 must survive a failed first clause.
% Clause 1 fails on argument 9, so a runtime that drops A9/A10 at the
% choicepoint reports false here.
conformance_query(wide, 'cwide/10', [1,2,3,4,5,6,7,8,9,45], true).
conformance_query(wide, 'cwide/10', [1,2,3,4,5,6,7,8,9,44], false).
% ...and clause 1 itself must still match when it should.
conformance_query(wide, 'cwide/10', [0,0,0,0,0,0,0,0,no,no], true).

% nested — structure inside a cons head, and the cons tail after it.
conformance_query(nested, 'cnest/2', [[tk(a),tk(b)], a], true).
conformance_query(nested, 'cnest/2', [[tk(a),tk(b)], b], true).
conformance_query(nested, 'cnest/2', [[tk(a)],       z], false).
% ctail pins the tail: it is unified AFTER the nested get_structure.
conformance_query(nested, 'ctail/3', [[tk(a),tk(b)], a, [tk(b)]], true).
conformance_query(nested, 'ctail/3', [[tk(a)],       a, []],      true).
conformance_query(nested, 'ctail/3', [[tk(a),tk(b)], a, []],      false).
% ckind discriminates on the inner functor across clauses.
conformance_query(nested, 'ckind/2', [[cnum(7)], n(7)], true).
conformance_query(nested, 'ckind/2', [[csym(x)], s(x)], true).
conformance_query(nested, 'ckind/2', [[cwd(m)],  w(m)], true).
conformance_query(nested, 'ckind/2', [[csym(x)], n(x)], false).
% buildnest — write mode: build `[tk(X)|[]]` with an unbound output, then
% check it with ==/2. Its own program; see the fixture note above.
conformance_query(buildnest, 'cbuild_chk/1', [a], true).
conformance_query(buildnest, 'cbuild_chk/1', [b], true).

% repeatvar — a repeated variable inside a structure. Kept as its own
% program because detecting the bug needs ==/2 (see crepeat/1 below), so
% a backend can diverge here for a term-comparison reason that has
% nothing to do with the nested-head shapes above; a shared program
% would make one xfail blanket both.
conformance_query(repeatvar, 'csame/2',   [p(a,a), a], true).
conformance_query(repeatvar, 'csame/2',   [p(a,b), a], false).
conformance_query(repeatvar, 'crepeat/1', [a], true).

% emptylist — `[]` as a cons tail must equal a literal `[]`.
conformance_query(emptylist, 'cnil_tail/2', [[a],   []],  true).
conformance_query(emptylist, 'cnil_tail/2', [[a,b], [b]], true).
conformance_query(emptylist, 'cnil_tail/2', [[a],   [b]], false).
conformance_query(emptylist, 'cone/2',      [a, [a]],     true).
conformance_query(emptylist, 'cone/2',      [a, [a,b]],   false).
