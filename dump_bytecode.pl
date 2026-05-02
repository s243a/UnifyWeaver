:- use_module('src/unifyweaver/targets/wam_target').
:- use_module('examples/wam_term_builtins_bench/bench_term_walk').
:- use_module('examples/wam_term_builtins_bench/bench_suite').

:- initialization(main, main).
main :-
    use_module('examples/wam_term_builtins_bench/bench_suite'),
    use_module('examples/wam_term_builtins_bench/bench_term_walk'),
    forall(member(P, [
        bench_suite:bench_sum_medium/0,
        bench_term_walk:sum_ints/3,
        bench_term_walk:sum_ints_args/5,
        bench_suite:term_depth/2,
        bench_suite:term_depth_args/5,
        bench_suite:fib/3,
        probe_bench:probe_td_atom_a_v1/0
    ]), (
        format('=== ~w ===~n', [P]),
        compile_predicate_to_wam(P, [], Code),
        format('~w~n~n', [Code])
    )),
    halt.
