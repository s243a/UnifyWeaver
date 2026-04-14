%% SWI-Prolog baseline benchmark runner.
%% Runs the same workloads as the WAM-WAT benchmark suite and reports
%% ns/call for each, so the numbers are directly comparable.
%%
%% Usage (from project root):
%%   swipl examples/wam_term_builtins_bench/run_bench_swi.pl [iterations]

:- use_module(examples/wam_term_builtins_bench/bench_suite).
:- use_module(examples/wam_term_builtins_bench/bench_term_walk).

:- initialization(main, main).

bench(Name, Goal, Iters) :-
    %% Warmup
    (   between(1, 100, _), once(Goal), fail ; true ),
    %% Measure
    get_time(T0),
    (   between(1, Iters, _), once(Goal), fail ; true ),
    get_time(T1),
    DeltaS is T1 - T0,
    NsPerCall is (DeltaS * 1.0e9) / Iters,
    CallsPerSec is Iters / DeltaS,
    format('~w~35|~`.t~t~0f~10+~t~0f~12+~n',
           [Name, NsPerCall, CallsPerSec]).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [IterAtom|_], atom_number(IterAtom, Iters0)
    ->  Iters = Iters0
    ;   Iters = 100000
    ),
    format('SWI-Prolog Baseline (~w iterations per workload)~n', [Iters]),
    format('~`=t~70|~n'),
    format('~w~35|~`.t~t~w~10+~t~w~12+~n',
           ['Workload', 'ns/call', 'calls/s']),
    format('~`-t~70|~n'),

    bench(bench_true,          true, Iters),
    bench(bench_is_arith,      (_ is 1000 * 3 + 7), Iters),
    bench(bench_unify,         (X1 = foo(a,b,c), X1 = foo(a,b,c)), Iters),
    bench(bench_functor_read,  functor(foo(a,b,c), _, _), Iters),
    bench(bench_arg_read,      arg(2, foo(a,b,c), _), Iters),
    bench(bench_univ_decomp,   (foo(a,b) =.. _), Iters),
    bench(bench_copy_flat,     copy_term(foo(a,b), _), Iters),
    bench(bench_copy_nested,   copy_term(f(g(a),h(b)), _), Iters),
    bench(bench_sum_small,
          bench_term_walk:sum_ints(f(1,2,3), 0, _), Iters),
    bench(bench_sum_medium,
          bench_term_walk:sum_ints(f(1,g(2,3),4), 0, _), Iters),
    bench(bench_sum_big,
          bench_term_walk:sum_ints(
              f(1, g(2, h(3,4), 5), k(6,7), m(8, j(9,10))), 0, _), Iters),

    format('~`-t~70|~n'),
    halt.
