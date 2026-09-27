:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% driver.pl -- shared corpus driver for the C++ WAM lane.
%
% The SAME predicates run on two engines:
%   * SWI-Prolog directly (the oracle), and
%   * the C++ WAM binary (resolver.pl + test_resolver.pl data + this file
%     compiled through wam_cpp_target.pl).
%
% emit_all/0 walks the 51-scenario contract corpus (corpus_case/4 +
% scenario_catalog/2 from test_resolver.pl), runs each query against the
% frozen resolver.pl, wraps the answer in ok(_)/fail, and prints one line
%   <CaseId> <canonical-term>
% per case.  ser/2 is a whitespace-free, operator-free canonical term
% writer using only builtins the C++ WAM runtime implements (functor/3,
% arg/3, atom_number/2, atomic_list_concat/2), so the two legs are
% byte-identical whenever the C++ WAM executes the resolver like SWI.
% Any per-line difference is a genuine C++-vs-SWI divergence.

% ---------------------------------------------------------------------------
% Per-query dispatch (mirrors dump_corpus.pl run_query/4, minus JSON).
% ---------------------------------------------------------------------------
result_term(resolve, Cat, Args, R) :-
    ( resolve(Cat, Args, Sel) -> R = ok(Sel) ; R = fail ).
result_term(resolve_layered, Cat, Args, R) :-
    ( resolve_layered(Cat, Args, Sel) -> R = ok(Sel) ; R = fail ).
result_term(explain_blocked, Cat, Args, R) :-
    ( explain_blocked_list(Cat, Args, L) -> R = ok(L) ; R = fail ).
result_term(layer_closure, Cat, Args, R) :-
    ( layer_closure(Cat, Args, L) -> R = ok(L) ; R = fail ).
result_term(removal_orphans, Cat, Args, R) :-
    ( removal_orphans(Cat, Args, O) -> R = ok(O) ; R = fail ).
result_term(safe_upgrade, Cat, [Pkg, Ver], R) :-
    ( safe_upgrade(Cat, Pkg, Ver, V) -> R = ok(V) ; R = fail ).
result_term(upgrade_set, Cat, [Pkg, Ver], R) :-
    ( upgrade_set_result(Cat, Pkg, Ver, U) -> R = ok(U) ; R = fail ).
result_term(freeze_audit, Cat, _, R) :-
    ( freeze_audit(Cat, A) -> R = ok(A) ; R = fail ).
result_term(dependents, Cat, Args, R) :-
    ( dependents(Cat, Args, D) -> R = ok(D) ; R = fail ).
result_term(dependents_installed, Cat, Args, R) :-
    ( dependents_installed(Cat, Args, D) -> R = ok(D) ; R = fail ).

case_result(Id, S) :-
    corpus_case(Id, CatName, Query, Args),
    scenario_catalog(CatName, Cat),
    ( result_term(Query, Cat, Args, R) -> true ; R = error ),
    ser(R, S).

% Failure-driven loop (NOT forall/2): on the C++ WAM, forall/2's
% double-negation scope mangles the shared Id/S bindings and writes the
% raw result term instead of the serialised atom.  The explicit
% backtracking loop below binds correctly on both engines.
emit_all :-
    (   corpus_case(Id, _, _, _),
        case_result(Id, S),
        write(Id), write(' '), write(S), nl,
        fail
    ;   true
    ).

% ---------------------------------------------------------------------------
% Minimal reproducer for the one blocker keeping this lane below 51/51.
%
% The C++ WAM (interpreter emit mode) mis-evaluates the FIRST if-then-else
% of resolver.pl's blocked_acc/5 when the enclosing clause carries several
% permanent (Y-register) variables AND a `\+`/negation sits in the ITE
% condition: the condition `base_ver(Cat,Name,BV), \+ satisfies(BV,C)`
% succeeds when run standalone (bug_repro_ok/0 below prints then=...), yet
% inside blocked_acc/5 the same condition takes the ELSE branch, so the
% blocked(...) fact is dropped and explain_blocked_list/3 returns [].
%
% SWI runs both predicates identically (both non-empty).  Divergence is a
% C++ WAM permanent-variable / negation codegen bug, not a resolver.pl or
% driver issue.  See cpp/README.md.
%
%   ./uwresolve 'bug_repro_ok/0'   -> then=v(1,0,0)   (condition true)
%   ./uwresolve 'bug_repro_bug/0'  -> after_ite1=[]    (same cond, ELSE)
bug_repro_ok :-
    scenario_catalog(blocked_base, Cat),
    ( base_ver(Cat, lib, BV), \+ satisfies(BV, gte(v(2,0,0)))
    -> write(then), write('='), write(BV) ; write(else) ), nl.

bug_repro_bug :-
    scenario_catalog(blocked_base, Cat),
    Name = lib, C = gte(v(2,0,0)), Seen = [], Acc0 = [],
    (   base_ver(Cat, Name, BV), \+ satisfies(BV, C)
    ->  Acc1 = [blocked(Name, needs(C), base_has(BV))|Acc0]
    ;   Acc1 = Acc0
    ),
    write(after_ite1), write('='), ser(Acc1, S1), write(S1), nl,
    ( seen_name(Seen, Name) -> true
    ; walk_pkg_for_blocked(Cat, Name, C, _Pkg, _Ver) -> true ; true ),
    _ = Acc1.

% ---------------------------------------------------------------------------
% Canonical term serialiser: whitespace-free, always-functional notation.
% Uses only builtins present in the C++ WAM runtime.
% ---------------------------------------------------------------------------
ser(T, S) :-
    ( var(T)      -> S = '_'
    ; integer(T)  -> atom_number(S, T)
    ; T == []     -> S = '[]'
    ; atom(T)     -> S = T
    ; compound(T) ->
        functor(T, F, N),
        ser_args(T, 1, N, Parts),
        atomic_list_concat(Parts, Inner),
        atomic_list_concat([F, '(', Inner, ')'], S)
    ; S = '?'
    ).

ser_args(_, I, N, []) :- I > N, !.
ser_args(T, I, N, Parts) :-
    arg(I, T, Ai),
    ser(Ai, A),
    ( I < N -> Parts = [A, ',' | Rest] ; Parts = [A | Rest] ),
    I1 is I + 1,
    ser_args(T, I1, N, Rest).
