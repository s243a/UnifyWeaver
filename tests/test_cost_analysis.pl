% test_cost_analysis.pl
%
% Tests for src/unifyweaver/core/cost_analysis.pl — the static goal/clause/
% predicate cost estimator. Each test defines predicates with a known cost
% shape and asserts the estimated tier, exercising: builtin weighting,
% conjunction/disjunction/if-then-else, predicate-call resolution, recursion
% detection (self- and mutual), and aggregate (findall/forall) unboundedness.

:- use_module('../src/unifyweaver/core/cost_analysis').
:- use_module(library(assoc)).

% ---- fixture program (asserted into `user`) --------------------------------
% Keep these heads distinct so current_predicate enumeration is clean.

:- dynamic ca_fact/1.
ca_fact(1). ca_fact(2). ca_fact(3).

% trivial: a single unification
ca_trivial(X) :- X = ok.

% cheap: a couple of arithmetic/compare goals
ca_cheap(X, Y) :- Y is X + 1, Y > 0.

% moderate: several builtins incl. list/atom ops
ca_moderate(L, A) :- length(L, N), N > 0, atom_concat(a, b, A), msort(L, _).

% calls another (non-recursive) predicate → cost composes
ca_caller(X) :- ca_cheap(X, _), ca_moderate([1,2], _).

% self-recursive → unbounded → tier recursive
ca_rec([]).
ca_rec([_|T]) :- ca_rec(T).

% mutual recursion → both unbounded
ca_even(0).
ca_even(N) :- N > 0, M is N - 1, ca_odd(M).
ca_odd(N) :- N > 0, M is N - 1, ca_even(M).

% non-recursive predicate that *calls* a recursive one → unbounded (propagates)
ca_uses_rec(L) :- ca_rec(L).

% aggregate over a cheap generator → unbounded (cardinality unknown)
ca_aggr(Xs) :- findall(X, ca_fact(X), Xs).

:- begin_tests(cost_analysis).

build(Model) :- build_cost_model(user, Model).

test(trivial_tier) :-
    build(M), predicate_cost_tier(ca_trivial/1, M, T),
    assertion(memberchk(T, [trivial, cheap])).

test(cheap_tier) :-
    build(M), predicate_cost_tier(ca_cheap/2, M, T),
    assertion(memberchk(T, [trivial, cheap])).

test(moderate_is_more_than_cheap) :-
    build(M),
    predicate_cost(ca_cheap/2, M, cost(Wc, bounded)),
    predicate_cost(ca_moderate/2, M, cost(Wm, bounded)),
    assertion(Wm > Wc).

test(caller_composes_callees) :-
    % caller cost should be >= the sum-ish of its callees (it calls both)
    build(M),
    predicate_cost(ca_caller/1, M, cost(Wcaller, bounded)),
    predicate_cost(ca_moderate/2, M, cost(Wm, bounded)),
    assertion(Wcaller > Wm).

test(self_recursive_is_recursive_tier) :-
    build(M), predicate_cost_tier(ca_rec/1, M, T),
    assertion(T == recursive).

test(self_recursive_is_unbounded) :-
    build(M), predicate_cost(ca_rec/1, M, cost(_, B)),
    assertion(B == unbounded).

test(mutual_recursion_both_unbounded) :-
    build(M),
    predicate_cost(ca_even/1, M, cost(_, Be)),
    predicate_cost(ca_odd/1, M, cost(_, Bo)),
    assertion(Be == unbounded), assertion(Bo == unbounded).

test(recursion_propagates_to_caller) :-
    % a non-recursive predicate that calls a recursive one is unbounded
    build(M), predicate_cost(ca_uses_rec/1, M, cost(_, B)),
    assertion(B == unbounded).

test(aggregate_is_unbounded) :-
    build(M), predicate_cost(ca_aggr/1, M, cost(_, B)),
    assertion(B == unbounded).

test(recursive_predicate_query) :-
    build(M),
    assertion(recursive_predicate(ca_rec/1, M)),
    assertion(recursive_predicate(ca_even/1, M)),
    assertion(\+ recursive_predicate(ca_cheap/2, M)).

% ---- goal_cost (arbitrary goals against the model) -------------------------

test(goal_cost_conjunction_adds) :-
    build(M),
    goal_cost((X = 1, Y is X + 1, Y > 0), M, cost(W, bounded)),
    assertion(W >= 4).   % =:1, is:2, >:2 (+ small)

test(goal_cost_findall_unbounded) :-
    build(M),
    goal_cost(findall(X, ca_fact(X), _L), M, cost(_, B)),
    assertion(B == unbounded).

test(goal_cost_ite_takes_branch_max) :-
    build(M),
    % then-branch heavier than else; ITE cost >= cond + max(then,else)
    goal_cost(( true -> msort([3,1], _) ; _ = a ), M, cost(Wite, _)),
    goal_cost(( true -> _ = a ; _ = b ), M, cost(Wcheap, _)),
    assertion(Wite > Wcheap).

test(builtin_override_is_respected) :-
    % overriding the cost table changes the estimate
    ( builtin_cost(my_special_op/0, _) -> true
    ; assertz(cost_analysis:builtin_cost(my_special_op/0, 999)) ),
    build(M),
    goal_cost(my_special_op, M, cost(W, _)),
    assertion(W >= 999),
    retractall(cost_analysis:builtin_cost(my_special_op/0, _)).

% ---- tier classification edge cases ----------------------------------------

test(cost_tier_bands) :-
    cost_tier(cost(0, bounded), trivial),
    cost_tier(cost(1, bounded), trivial),
    cost_tier(cost(3, bounded), cheap),
    cost_tier(cost(15, bounded), moderate),
    cost_tier(cost(100, bounded), expensive),
    cost_tier(cost(3, unbounded), recursive).

:- end_tests(cost_analysis).
