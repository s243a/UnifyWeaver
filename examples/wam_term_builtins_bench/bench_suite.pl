% WAM-WAT benchmark suite: multiple workloads exercising the Group A
% term inspection builtins + core WAM runtime. Each workload is a
% zero-argument predicate that does a fixed amount of work. The
% benchmark harness calls each export N times and reports ns/call.
%
% Workloads (in order of complexity):
%   bench_true          — baseline: just `true` (measures dispatch overhead)
%   bench_is_arith      — X is 1000 * 3 + 7 (arithmetic eval)
%   bench_unify         — X = foo(a,b,c), X = foo(a,b,c) (unify check)
%   bench_functor_read  — functor(foo(a,b,c), _, _) (read mode)
%   bench_arg_read      — arg(2, foo(a,b,c), _) (arg extraction)
%   bench_univ_decomp   — foo(a,b) =.. _ (decompose)
%   bench_copy_flat     — copy_term(foo(a,b), _) (shallow copy)
%   bench_copy_nested   — copy_term(f(g(a),h(b)), _) (deep copy)
%   bench_sum_small     — sum_ints(f(1,2,3), 0, _) (3-leaf walk)
%   bench_sum_medium    — sum_ints(f(1,g(2,3),4), 0, _) (5-leaf walk)
%   bench_sum_big       — sum_ints(f(1,g(2,h(3,4),5),k(6,7),m(8,j(9,10))),0,_)
%                         (10-leaf, 15-node deep walk)

:- module(bench_suite, [
    bench_true/0,
    bench_is_arith/0,
    bench_unify/0,
    bench_functor_read/0,
    bench_arg_read/0,
    bench_univ_decomp/0,
    bench_copy_flat/0,
    bench_copy_nested/0,
    bench_sum_small/0,
    bench_sum_medium/0,
    bench_sum_big/0,
    bench_term_depth/0,
    bench_fib10/0,
    fib/3,
    term_depth/2,
    term_depth_args/5
]).

:- use_module(bench_term_walk).

%% Baseline
bench_true :- true.

%% Arithmetic
bench_is_arith :- _ is 1000 * 3 + 7.

%% Unification
bench_unify :- X = foo(a, b, c), X = foo(a, b, c).

%% functor/3 read
bench_functor_read :- functor(foo(a, b, c), _, _).

%% arg/3
bench_arg_read :- arg(2, foo(a, b, c), _).

%% =../2 decompose
bench_univ_decomp :- foo(a, b) =.. _.

%% copy_term shallow
bench_copy_flat :- copy_term(foo(a, b), _).

%% copy_term nested
bench_copy_nested :- copy_term(f(g(a), h(b)), _).

%% sum_ints: small (3 leaves). Expected: 1+2+3 = 6.
bench_sum_small :- sum_ints(f(1, 2, 3), 0, 6).

%% sum_ints: medium (5 leaves). Expected: 1+2+3+4 = 10.
bench_sum_medium :- sum_ints(f(1, g(2, 3), 4), 0, 10).

%% sum_ints: big (10 leaves, 15 nodes). Expected: 1+2+...+10 = 55.
bench_sum_big :-
    sum_ints(f(1, g(2, h(3, 4), 5), k(6, 7), m(8, j(9, 10))), 0, 55).

%% --- Additional workloads ---

%% term_depth: max nesting depth of a compound term.
%% Exercises: functor/3, arg/3, is/2, recursive cross-pred calls.
term_depth(T, 0) :- integer(T), !.
term_depth(T, 0) :- atom(T), !.
term_depth(T, D) :-
    functor(T, _, Arity),
    term_depth_args(1, Arity, T, 0, MaxChild),
    D is MaxChild + 1.

term_depth_args(I, Arity, _, Max, Max) :- I > Arity, !.
term_depth_args(I, Arity, T, Acc, Max) :-
    arg(I, T, A),
    term_depth(A, AD),
    (   AD > Acc -> NewAcc = AD ; NewAcc = Acc ),
    I1 is I + 1,
    term_depth_args(I1, Arity, T, NewAcc, Max).

%% term_depth: f with nested g/h and m/j at depth 3. Expected: 3.
bench_term_depth :-
    term_depth(f(1, g(2, h(3, 4), 5), k(6, 7), m(8, j(9, 10))), 3).

%% fib: naive Fibonacci (exercises pure recursive arithmetic).
%% fib(N, Acc, Result) — accumulator-style to avoid stack overflow.
fib(N, _, 0) :- N =< 0, !.
fib(1, _, 1) :- !.
fib(N, _, Result) :-
    N1 is N - 1,
    fib(N1, 0, R1),
    N2 is N - 2,
    fib(N2, 0, R2),
    Result is R1 + R2.

%% fib10: fib(10) = 55 (the 10th Fibonacci number).
bench_fib10 :- fib(10, 0, 55).
