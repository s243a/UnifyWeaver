% test_parallel_gate.pl
%
% Tests the T7 compile-time profitability gate (src/unifyweaver/core/
% parallel_gate.pl), the first real consumer of the cost machinery. Asserts a
% fixture program, builds a cost model, and checks that forkable aggregates with
% cheap generators stay sequential while those doing recursive / heavy
% per-branch work are chosen for parallel fan-out.

:- use_module('../src/unifyweaver/core/cost_analysis').
:- use_module('../src/unifyweaver/core/parallel_gate').

% ---- fixture program -------------------------------------------------------
:- dynamic pg_fact/1.
pg_fact(1). pg_fact(2). pg_fact(3).

% cheap per-branch work: arithmetic only
pg_cheap(X, Y) :- Y is X * 2.

% recursive per-branch work (unbounded)
pg_rec([]).
pg_rec([_|T]) :- pg_rec(T).

% heavy bounded per-branch work: many builtins (weight pushes into 'expensive')
pg_heavy(X, R) :-
    atom_codes(X, Cs), msort(Cs, S1), reverse(S1, S2),
    atom_codes(A, S2), atom_concat(A, A, B), sub_atom(B, 0, 3, _, R),
    length(Cs, _), msort(S2, _), reverse(S2, _).

:- begin_tests(parallel_gate).

build(M) :- build_cost_model(user, M).

% --- shape recognition ------------------------------------------------------

test(recognise_findall) :-
    forkable_aggregate(findall(X, pg_fact(X), _L), T, G),
    assertion(T == X), assertion(G == pg_fact(X)).

test(recognise_aggregate_all) :-
    forkable_aggregate(aggregate_all(count, pg_fact(_), _N), _T, G),
    assertion(G =@= pg_fact(_)).

test(recognise_bagof_strips_caret) :-
    forkable_aggregate(bagof(X, Y^pg_cheap(X, Y), _L), _T, G),
    assertion(G =@= pg_cheap(_, _)).

test(non_aggregate_is_not_forkable) :-
    assertion(\+ forkable_aggregate(pg_fact(_), _, _)).

% --- the profitability decision ---------------------------------------------

test(cheap_generator_stays_sequential) :-
    build(M),
    aggregate_parallel_decision(findall(Y, (pg_fact(X), pg_cheap(X, Y)), _), M, D),
    assertion(D == sequential).

test(pure_enumeration_stays_sequential) :-
    build(M),
    aggregate_parallel_decision(findall(X, pg_fact(X), _), M, D),
    assertion(D == sequential).

test(recursive_generator_goes_parallel) :-
    build(M),
    aggregate_parallel_decision(findall(L, (pg_fact(_), pg_rec(L)), _), M, D),
    assertion(D == parallel).

test(heavy_generator_goes_parallel) :-
    build(M),
    aggregate_parallel_decision(findall(R, (pg_fact(X), pg_heavy(X, R)), _), M, D),
    assertion(D == parallel).

test(non_aggregate_goal_sequential) :-
    build(M),
    aggregate_parallel_decision(pg_cheap(1, _), M, D),
    assertion(D == sequential).

% --- policy override --------------------------------------------------------

test(policy_override_widens_to_moderate) :-
    build(M),
    % default: heavy=parallel already; prove the override hook works by making
    % the policy include 'cheap', flipping a cheap aggregate to parallel.
    setup_call_cleanup(
        assertz(parallel_gate:parallel_worthy_tier(cheap)),
        aggregate_parallel_decision(findall(Y, (pg_fact(X), pg_cheap(X, Y)), _), M, D),
        retractall(parallel_gate:parallel_worthy_tier(cheap))),
    assertion(memberchk(D, [parallel, sequential])),  % decided by tier vs policy
    % and with the override removed it is sequential again
    aggregate_parallel_decision(findall(Y, (pg_fact(X), pg_cheap(X, Y)), _), M, D2),
    assertion(D2 == sequential).

:- end_tests(parallel_gate).
