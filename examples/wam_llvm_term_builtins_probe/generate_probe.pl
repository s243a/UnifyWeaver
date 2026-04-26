:- use_module('../../src/unifyweaver/targets/wam_llvm_target').
:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module('../wam_term_builtins_bench/bench_suite').
:- use_module('../wam_term_builtins_bench/bench_term_walk').
:- use_module('probe_bench').

:- initialization(main, main).

main :-
    Predicates = [
        probe_bench:probe_sum_small_0/0,
        probe_bench:probe_sum_small_1/0,
        probe_bench:probe_sum_small_2/0,
        probe_bench:probe_sum_small_3/0,
        probe_bench:probe_sum_small_4/0,
        probe_bench:probe_sum_small_5/0,
        probe_bench:probe_sum_small_6/0,
        probe_bench:probe_sum_small_7/0,
        probe_bench:probe_fib2_v0/0,
        probe_bench:probe_fib2_v1/0,
        probe_bench:probe_fib2_v2/0,
        probe_bench:probe_fib2_v3/0,
        probe_bench:probe_fib3_v0/0,
        probe_bench:probe_fib3_v1/0,
        probe_bench:probe_fib3_v2/0,
        probe_bench:probe_fib3_v3/0,
        probe_bench:probe_term_depth_v0/0,
        probe_bench:probe_term_depth_v1/0,
        probe_bench:probe_term_depth_v2/0,
        probe_bench:probe_term_depth_v3/0,
        probe_bench:probe_sum_leaf_v0/0,
        probe_bench:probe_sum_leaf_v1/0,
        probe_bench:probe_sum_leaf_v2/0,
        probe_bench:probe_sum_leaf_acc5_v5/0,
        probe_bench:probe_sum_leaf_acc5_v6/0,
        probe_bench:probe_sum_g1_v0/0,
        probe_bench:probe_sum_g1_v1/0,
        probe_bench:probe_sum_g1_v2/0,
        probe_bench:probe_sum_args_v0/0,
        probe_bench:probe_sum_args_v1/0,
        probe_bench:probe_sum_args_v2/0,
        probe_bench:probe_sum_args_v3/0,
        probe_bench:probe_sum_args_2_1_v0/0,
        probe_bench:probe_sum_args_2_1_v1/0,
        probe_bench:probe_sum_args_2_1_v2/0,
        probe_bench:probe_gt_2_1/0,
        probe_bench:probe_gt_1_1/0,
        probe_bench:probe_eq_1_1/0,
        probe_bench:probe_eq_1_2/0,
        probe_bench:probe_td_atom_a_v0/0,
        probe_bench:probe_td_atom_a_v1/0,
        probe_bench:probe_td_g1_v0/0,
        probe_bench:probe_td_g1_v1/0,
        probe_bench:probe_td_g1_v2/0,
        probe_bench:probe_td_g2_v0/0,
        probe_bench:probe_td_g2_v1/0,
        probe_bench:probe_td_g2_v2/0,
        probe_bench:probe_sum_med_v10/0,
        probe_bench:probe_sum_med_v9/0,
        probe_bench:probe_sum_med_v0/0,
        probe_bench:probe_tda_2_2_v0/0,
        probe_bench:probe_tda_2_2_v1/0,
        probe_bench:probe_tda_3_2_v0/0,
        probe_bench:probe_tda_3_2_v1/0,
        probe_bench:probe_smed_v0/0, probe_bench:probe_smed_v1/0,
        probe_bench:probe_smed_v2/0, probe_bench:probe_smed_v3/0,
        probe_bench:probe_smed_v4/0, probe_bench:probe_smed_v5/0,
        probe_bench:probe_smed_v6/0, probe_bench:probe_smed_v7/0,
        probe_bench:probe_smed_v8/0, probe_bench:probe_smed_v9/0,
        probe_bench:probe_smed_v10/0, probe_bench:probe_smed_v11/0,
        probe_bench:probe_smed_v12/0, probe_bench:probe_smed_v13/0,
        probe_bench:probe_smed_v14/0, probe_bench:probe_smed_v15/0,
        probe_bench:probe_sg23_v0/0, probe_bench:probe_sg23_v1/0,
        probe_bench:probe_sg23_v2/0, probe_bench:probe_sg23_v3/0,
        probe_bench:probe_sg23_v4/0, probe_bench:probe_sg23_v5/0,
        probe_bench:probe_sg23_v6/0,
        probe_bench:probe_smed/1,
        probe_bench:probe_sg23/1,
        probe_bench:probe_smed_v100/0, probe_bench:probe_smed_v1000/0,
        probe_bench:probe_smed_vneg1/0,
        probe_bench:probe_sfg_v0/0, probe_bench:probe_sfg_v3/0,
        probe_bench:probe_sfg_v5/0, probe_bench:probe_sfg_v7/0,
        probe_bench:probe_s1g_v0/0, probe_bench:probe_s1g_v3/0,
        probe_bench:probe_s1g_v5/0, probe_bench:probe_s1g_v6/0,
        probe_bench:probe_s1g_v7/0, probe_bench:probe_s1g_v8/0,
        probe_bench:probe_sfg/1,
        probe_bench:probe_s1g/1,
        probe_bench:probe_s1g1_v0/0, probe_bench:probe_s1g1_v1/0,
        probe_bench:probe_s1g1_v2/0, probe_bench:probe_s1g1_v3/0,
        probe_bench:probe_s1g1_v4/0,
        probe_bench:probe_sfg1_v0/0, probe_bench:probe_sfg1_v1/0,
        probe_bench:probe_sfg1_v2/0, probe_bench:probe_sfg1_v3/0,
        probe_bench:probe_s1g1/1, probe_bench:probe_sfg1/1,
        probe_bench:probe_sg_v0/0, probe_bench:probe_sg_v1/0,
        probe_bench:probe_sg_v2/0, probe_bench:probe_sg_v3/0,
        probe_bench:probe_sg/1,
        probe_bench:probe_fib/3,
        bench_term_walk:sum_ints/3,
        bench_term_walk:sum_ints_args/5,
        bench_suite:term_depth/2,
        bench_suite:term_depth_args/5
    ],
    LLPath = 'examples/wam_llvm_term_builtins_probe/probe_suite.ll',
    write_wam_llvm_project(
        Predicates,
        [ module_name(probe_suite)
        , target_triple('aarch64-unknown-linux-android')
        , target_datalayout('e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128')
        ],
        LLPath),
    append_wrappers(LLPath, Predicates),
    halt.

append_wrappers(LLPath, Predicates) :-
    findall(W, (
        member(_Mod:Pred/0, Predicates),
        atom_string(Pred, PS),
        format(atom(W),
'define i32 @run_~w() {
  %r = call i1 @~w()
  call void @wam_cleanup()
  %r32 = zext i1 %r to i32
  ret i32 %r32
}', [PS, PS])
    ), Wrappers),
    atomic_list_concat(Wrappers, '\n\n', WrappersStr),
    setup_call_cleanup(
        open(LLPath, append, Stream),
        format(Stream, "~n~n; === Probe wrappers ===~n~w~n", [WrappersStr]),
        close(Stream)
    ).
