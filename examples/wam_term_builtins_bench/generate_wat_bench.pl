%% generate_wat_bench.pl
%%
%% Generates a WAM-WAT benchmark module containing all workloads from
%% bench_suite.pl + the underlying sum_ints/sum_ints_args predicates.
%% The generated .wat file exports one function per workload, each
%% taking no args and returning i32 (1 = success, 0 = fail).
%%
%% Usage (from project root):
%%   swipl examples/wam_term_builtins_bench/generate_wat_bench.pl
%%
%% Output: examples/wam_term_builtins_bench/bench_suite.wat
%%         (+ bench_suite.wasm after wat2wasm)

:- use_module('../../src/unifyweaver/targets/wam_wat_target').
:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module(bench_suite).
:- use_module(bench_term_walk).

:- initialization(main, main).

main :-
    Predicates = [
        bench_suite:bench_true/0,
        bench_suite:bench_is_arith/0,
        bench_suite:bench_unify/0,
        bench_suite:bench_functor_read/0,
        bench_suite:bench_arg_read/0,
        bench_suite:bench_univ_decomp/0,
        bench_suite:bench_copy_flat/0,
        bench_suite:bench_copy_nested/0,
        bench_suite:bench_sum_small/0,
        bench_suite:bench_sum_medium/0,
        bench_suite:bench_sum_big/0,
        bench_term_walk:sum_ints/3,
        bench_term_walk:sum_ints_args/5
    ],
    write_wam_wat_project(
        Predicates,
        [module_name(bench_suite)],
        'examples/wam_term_builtins_bench/bench_suite.wat'),
    halt.
