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
    bench_sum_big/0
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

%% sum_ints: small (3 leaves)
bench_sum_small :- sum_ints(f(1, 2, 3), 0, _).

%% sum_ints: medium (5 leaves)
bench_sum_medium :- sum_ints(f(1, g(2, 3), 4), 0, _).

%% sum_ints: big (10 leaves, 15 nodes)
bench_sum_big :-
    sum_ints(f(1, g(2, h(3, 4), 5), k(6, 7), m(8, j(9, 10))), 0, _).
