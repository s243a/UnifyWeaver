% Phase 6 microbenchmark: generic term walker driven by functor/3 +
% arg/3. Sums every integer leaf of a compound term by recursing via
% arg/3 + functor/3 only — no pattern matching, no structural
% decomposition at the WAM level. This is the shape of code that the
% Group A builtins were added to enable: any backend that fails to
% implement functor/3 or arg/3 simply cannot transpile this predicate
% at all, regardless of perf.
%
% Clauses use explicit cut rather than if-then-else because the
% canonical WAM compiler currently hangs on ITE lowering for the
% hybrid backends (a pre-existing issue, not related to Group A).
%
% The benchmark runs the walker over a small fixed term N times and
% reports wall-clock time. A single call is very cheap; N=1000 or
% 10000 pushes the total into a measurable range on both host SWI
% and the WAM transpilation targets.

:- module(bench_term_walk, [
    sum_ints/3,
    sum_ints_args/5,
    run_bench/2,
    bench_term/1,
    run_loop/4
]).

%% sum_ints(+T, +Acc, -Sum)
%  Walk T and add every integer leaf to Acc.
sum_ints(T, Acc, Sum) :- integer(T), !, Sum is Acc + T.
sum_ints(T, Acc, Sum) :-
    functor(T, _F, Arity),
    sum_ints_args(1, Arity, T, Acc, Sum).

%% sum_ints_args(+I, +Arity, +T, +Acc, -Sum)
sum_ints_args(I, Arity, _, Acc, Sum) :- I > Arity, !, Sum = Acc.
sum_ints_args(I, Arity, T, Acc, Sum) :-
    arg(I, T, A),
    sum_ints(A, Acc, Acc1),
    I1 is I + 1,
    sum_ints_args(I1, Arity, T, Acc1, Sum).

%% bench_term(-T)
%  A small fixed term with 10 integer leaves across 4 levels of
%  nesting. Total subterm count (including intermediate compounds):
%  15. Sum of integer leaves: 55.
bench_term(f(1, g(2, h(3, 4), 5), k(6, 7), m(8, j(9, 10)))).

%% run_bench(+N, -Sum)
%  Run sum_ints on bench_term/1 N times. Each call is independent;
%  only the last Sum is returned. Intended to be called from a
%  top-level that measures wall-clock time around this goal.
run_bench(N, Sum) :-
    bench_term(T),
    run_loop(N, T, 0, Sum).

run_loop(N, _, LastSum, Sum) :- N =< 0, !, Sum = LastSum.
run_loop(N, T, _, Sum) :-
    sum_ints(T, 0, S1),
    N1 is N - 1,
    run_loop(N1, T, S1, Sum).
