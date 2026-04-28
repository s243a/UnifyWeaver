:- module(probe_bench, [
    probe_sum_small_0/0,
    probe_sum_small_1/0,
    probe_sum_small_2/0,
    probe_sum_small_3/0,
    probe_sum_small_4/0,
    probe_sum_small_5/0,
    probe_sum_small_6/0,
    probe_sum_small_7/0,
    probe_fib2/0,
    probe_fib2_v0/0,
    probe_fib2_v1/0,
    probe_fib2_v2/0,
    probe_fib2_v3/0,
    probe_fib3_v0/0,
    probe_fib3_v1/0,
    probe_fib3_v2/0,
    probe_fib3_v3/0,
    probe_term_depth_v0/0,
    probe_term_depth_v1/0,
    probe_term_depth_v2/0,
    probe_term_depth_v3/0,
    probe_fib/3,
    probe_sum_leaf_v0/0,
    probe_sum_leaf_v1/0,
    probe_sum_leaf_v2/0,
    probe_sum_leaf_acc5_v5/0,
    probe_sum_leaf_acc5_v6/0,
    probe_sum_g1_v0/0,
    probe_sum_g1_v1/0,
    probe_sum_g1_v2/0,
    probe_sum_args_v0/0,
    probe_sum_args_v1/0,
    probe_sum_args_v2/0,
    probe_sum_args_v3/0,
    probe_sum_args_2_1_v0/0,
    probe_sum_args_2_1_v1/0,
    probe_sum_args_2_1_v2/0,
    probe_gt_2_1/0,
    probe_gt_1_1/0,
    probe_eq_1_1/0,
    probe_eq_1_2/0,
    probe_td_atom_a_v0/0,
    probe_td_atom_a_v1/0,
    probe_td_g1_v0/0,
    probe_td_g1_v1/0,
    probe_td_g1_v2/0,
    probe_td_g2_v0/0,
    probe_td_g2_v1/0,
    probe_td_g2_v2/0,
    probe_sum_med_v10/0,
    probe_sum_med_v9/0,
    probe_sum_med_v0/0,
    probe_tda_2_2_v0/0,
    probe_tda_2_2_v1/0,
    probe_tda_3_2_v0/0,
    probe_tda_3_2_v1/0,
    probe_smed_v0/0, probe_smed_v1/0, probe_smed_v2/0, probe_smed_v3/0,
    probe_smed_v4/0, probe_smed_v5/0, probe_smed_v6/0, probe_smed_v7/0,
    probe_smed_v8/0, probe_smed_v9/0, probe_smed_v10/0, probe_smed_v11/0,
    probe_smed_v12/0, probe_smed_v13/0, probe_smed_v14/0, probe_smed_v15/0,
    probe_sg23_v0/0, probe_sg23_v1/0, probe_sg23_v2/0, probe_sg23_v3/0,
    probe_sg23_v4/0, probe_sg23_v5/0, probe_sg23_v6/0,
    probe_smed/1,
    probe_sg23/1,
    probe_smed_v100/0, probe_smed_v1000/0, probe_smed_vneg1/0,
    probe_sfg_v0/0, probe_sfg_v3/0, probe_sfg_v5/0, probe_sfg_v7/0,
    probe_s1g_v0/0, probe_s1g_v3/0, probe_s1g_v5/0,
    probe_s1g_v6/0, probe_s1g_v7/0, probe_s1g_v8/0,
    probe_sfg/1,
    probe_s1g/1,
    probe_s1g1_v0/0, probe_s1g1_v1/0, probe_s1g1_v2/0,
    probe_s1g1_v3/0, probe_s1g1_v4/0,
    probe_sfg1_v0/0, probe_sfg1_v1/0, probe_sfg1_v2/0, probe_sfg1_v3/0,
    probe_s1g1/1, probe_sfg1/1,
    probe_sg_v0/0, probe_sg_v1/0, probe_sg_v2/0, probe_sg_v3/0,
    probe_sg/1,
    dbg_sum/3, dbg_sum_args/5,
    probe_dbg_sum_fg/0, probe_dbg_sum_g/0,
    probe_dbg_sum_f12/0, probe_dbg_sum_f1g/0,
    probe_nested_build/0
]).
:- use_module('../wam_term_builtins_bench/bench_suite', [term_depth_args/5]).
:- use_module('../wam_term_builtins_bench/bench_term_walk').
:- use_module('../wam_term_builtins_bench/bench_suite', [term_depth/2]).

probe_sum_small_0 :- sum_ints(f(1, 2, 3), 0, 0).
probe_sum_small_1 :- sum_ints(f(1, 2, 3), 0, 1).
probe_sum_small_2 :- sum_ints(f(1, 2, 3), 0, 2).
probe_sum_small_3 :- sum_ints(f(1, 2, 3), 0, 3).
probe_sum_small_4 :- sum_ints(f(1, 2, 3), 0, 4).
probe_sum_small_5 :- sum_ints(f(1, 2, 3), 0, 5).
probe_sum_small_6 :- sum_ints(f(1, 2, 3), 0, 6).
probe_sum_small_7 :- sum_ints(f(1, 2, 3), 0, 7).

probe_fib(N, _, 0) :- N =< 0, !.
probe_fib(1, _, 1) :- !.
probe_fib(N, _, Result) :-
    N1 is N - 1,
    probe_fib(N1, 0, R1),
    N2 is N - 2,
    probe_fib(N2, 0, R2),
    Result is R1 + R2.

probe_fib2 :- probe_fib(2, 0, 1).
probe_fib2_v0 :- probe_fib(2, 0, 0).
probe_fib2_v1 :- probe_fib(2, 0, 1).
probe_fib2_v2 :- probe_fib(2, 0, 2).
probe_fib2_v3 :- probe_fib(2, 0, 3).

probe_fib3_v0 :- probe_fib(3, 0, 0).
probe_fib3_v1 :- probe_fib(3, 0, 1).
probe_fib3_v2 :- probe_fib(3, 0, 2).
probe_fib3_v3 :- probe_fib(3, 0, 3).

probe_term_depth_v0 :- term_depth(f(a, g(b, c)), 0).
probe_term_depth_v1 :- term_depth(f(a, g(b, c)), 1).
probe_term_depth_v2 :- term_depth(f(a, g(b, c)), 2).
probe_term_depth_v3 :- term_depth(f(a, g(b, c)), 3).

%% sum_ints with integer leaf — pure base case, no recursion.
probe_sum_leaf_v0 :- sum_ints(7, 0, 0).
probe_sum_leaf_v1 :- sum_ints(7, 0, 7).
probe_sum_leaf_v2 :- sum_ints(7, 0, 14).
probe_sum_leaf_acc5_v5 :- sum_ints(7, 5, 5).
probe_sum_leaf_acc5_v6 :- sum_ints(7, 5, 12).

%% sum_ints with one-leaf compound: g(1) — exercises one level of recursion.
probe_sum_g1_v0 :- sum_ints(g(1), 0, 0).
probe_sum_g1_v1 :- sum_ints(g(1), 0, 1).
probe_sum_g1_v2 :- sum_ints(g(1), 0, 2).

%% sum_ints_args directly with concrete args — bypass outer sum_ints.
%% sum_ints_args(1, 1, g(1), 0, Sum) should bind Sum=1.
probe_sum_args_v0 :- sum_ints_args(1, 1, g(1), 0, 0).
probe_sum_args_v1 :- sum_ints_args(1, 1, g(1), 0, 1).
probe_sum_args_v2 :- sum_ints_args(1, 1, g(1), 0, 2).
probe_sum_args_v3 :- sum_ints_args(1, 1, g(1), 0, 3).

%% sum_ints_args with I > Arity already true: clause 1 path only.
probe_sum_args_2_1_v0 :- sum_ints_args(2, 1, g(1), 1, 0).
probe_sum_args_2_1_v1 :- sum_ints_args(2, 1, g(1), 1, 1).
probe_sum_args_2_1_v2 :- sum_ints_args(2, 1, g(1), 1, 2).

%% raw builtin tests.
probe_gt_2_1 :- 2 > 1.
probe_gt_1_1 :- 1 > 1.
probe_eq_1_1 :- 1 = 1.
probe_eq_1_2 :- 1 = 2.

%% term_depth on simpler inputs.
probe_td_atom_a_v0 :- term_depth(a, 0).
probe_td_atom_a_v1 :- term_depth(a, 1).
probe_td_g1_v0 :- term_depth(g(a), 0).
probe_td_g1_v1 :- term_depth(g(a), 1).
probe_td_g1_v2 :- term_depth(g(a), 2).
probe_td_g2_v0 :- term_depth(g(a, b), 0).
probe_td_g2_v1 :- term_depth(g(a, b), 1).
probe_td_g2_v2 :- term_depth(g(a, b), 2).
%% sum_ints with deeper compound.
probe_sum_med_v10 :- sum_ints(f(1, g(2, 3), 4), 0, 10).
probe_sum_med_v9 :- sum_ints(f(1, g(2, 3), 4), 0, 9).
probe_sum_med_v0 :- sum_ints(f(1, g(2, 3), 4), 0, 0).

%% Direct probe of term_depth_args inner call
probe_tda_2_2_v0 :- term_depth_args(2, 2, g(a, b), 0, 0).
probe_tda_2_2_v1 :- term_depth_args(2, 2, g(a, b), 0, 1).
probe_tda_3_2_v0 :- term_depth_args(3, 2, g(a, b), 0, 0).
probe_tda_3_2_v1 :- term_depth_args(3, 2, g(a, b), 0, 1).

%% sweep sum_medium across all plausible values 0..15
probe_smed(V) :- sum_ints(f(1, g(2, 3), 4), 0, V).
probe_smed_v0  :- probe_smed(0).
probe_smed_v1  :- probe_smed(1).
probe_smed_v2  :- probe_smed(2).
probe_smed_v3  :- probe_smed(3).
probe_smed_v4  :- probe_smed(4).
probe_smed_v5  :- probe_smed(5).
probe_smed_v6  :- probe_smed(6).
probe_smed_v7  :- probe_smed(7).
probe_smed_v8  :- probe_smed(8).
probe_smed_v9  :- probe_smed(9).
probe_smed_v10 :- probe_smed(10).
probe_smed_v11 :- probe_smed(11).
probe_smed_v12 :- probe_smed(12).
probe_smed_v13 :- probe_smed(13).
probe_smed_v14 :- probe_smed(14).
probe_smed_v15 :- probe_smed(15).
probe_smed_v100 :- probe_smed(100).
probe_smed_v1000 :- probe_smed(1000).
probe_smed_vneg1 :- probe_smed(-1).

%% sum_ints on simpler nested compound: f(g(2,3))
probe_sfg(V) :- sum_ints(f(g(2, 3)), 0, V).
probe_sfg_v0 :- probe_sfg(0).
probe_sfg_v3 :- probe_sfg(3).
probe_sfg_v5 :- probe_sfg(5).
probe_sfg_v7 :- probe_sfg(7).

%% sum_ints with leading integer + nested compound
probe_s1g(V) :- sum_ints(f(1, g(2, 3)), 0, V).
probe_s1g_v0 :- probe_s1g(0).
probe_s1g_v3 :- probe_s1g(3).
probe_s1g_v5 :- probe_s1g(5).
probe_s1g_v6 :- probe_s1g(6).
probe_s1g_v7 :- probe_s1g(7).
probe_s1g_v8 :- probe_s1g(8).

%% sum_ints with single-arg nested compound: f(1, g(2))
probe_s1g1(V) :- sum_ints(f(1, g(2)), 0, V).
probe_s1g1_v0 :- probe_s1g1(0).
probe_s1g1_v1 :- probe_s1g1(1).
probe_s1g1_v2 :- probe_s1g1(2).
probe_s1g1_v3 :- probe_s1g1(3).
probe_s1g1_v4 :- probe_s1g1(4).

%% sum_ints with f(g(2)) — compound-only first arg
probe_sfg1(V) :- sum_ints(f(g(2)), 0, V).
probe_sfg1_v0 :- probe_sfg1(0).
probe_sfg1_v1 :- probe_sfg1(1).
probe_sfg1_v2 :- probe_sfg1(2).
probe_sfg1_v3 :- probe_sfg1(3).

%% sum_ints on g(2) directly — single-arg compound
probe_sg(V) :- sum_ints(g(2), 0, V).
probe_sg_v0 :- probe_sg(0).
probe_sg_v1 :- probe_sg(1).
probe_sg_v2 :- probe_sg(2).
probe_sg_v3 :- probe_sg(3).

%% Debug-instrumented sum_ints with prints between steps.
%% Prints intermediate accumulator values so we can see exactly where
%% the nested-compound recursion diverges from the expected trace.
dbg_sum(T, Acc, Sum) :-
    write(77000), nl,        %% entry print: every call to dbg_sum
    integer(T), !,
    write(77001), nl,        %% integer-clause path
    write(T), nl,
    Sum is Acc + T,
    write(77002), nl,
    write(Sum), nl.
dbg_sum(T, Acc, Sum) :-
    write(77003), nl,        %% compound-clause start
    %% probe: is T compound here? (compound/1 should succeed)
    (compound(T) -> write(77004) ; write(77005)), nl,
    %% probe: is T a Ref? var(T) returns true if it's an unbound Ref.
    (var(T) -> write(77006) ; write(77007)), nl,
    functor(T, _F, Arity),
    write(99001), nl,
    write(Arity), nl,
    dbg_sum_args(1, Arity, T, Acc, Sum),
    write(99002), nl,
    write(Sum), nl.

dbg_sum_args(I, Arity, _, Acc, Sum) :- I > Arity, !, Sum = Acc.
dbg_sum_args(I, Arity, T, Acc, Sum) :-
    arg(I, T, A),
    write(88001), nl,                              %% before call
    (var(A) -> write(88002) ; write(88003)), nl,   %% A bound or not?
    (compound(A) -> write(88004) ; write(88005)), nl, %% A compound?
    (integer(A) -> write(88006) ; write(88007)), nl,  %% A integer?
    dbg_sum(A, Acc, Acc1),
    write(99003), nl,
    write(Acc1), nl,
    I1 is I + 1,
    dbg_sum_args(I1, Arity, T, Acc1, Sum).

%% Run dbg_sum over f(g(2)) — should print intermediate acc values.
probe_dbg_sum_fg :- dbg_sum(f(g(2)), 0, _).
probe_dbg_sum_g :- dbg_sum(g(2), 0, _).
probe_dbg_sum_f12 :- dbg_sum(f(1, 2), 0, _).
probe_dbg_sum_f1g :- dbg_sum(f(1, g(2)), 0, _).

%% Test nested compound construction: build f(1, g(2,3)) and probe args.
probe_nested_build :-
    T = f(1, g(2, 3)),
    arg(1, T, A1),
    write(7001), nl,
    write(A1), nl,           % should be 1
    arg(2, T, A2),
    write(7002), nl,
    (compound(A2) -> write(7003) ; write(7004)), nl,    % should be compound
    (var(A2) -> write(7005) ; write(7006)), nl,         % should NOT be var
    arg(1, A2, A21),
    write(7007), nl,
    write(A21), nl.          % should be 2

%% sum_ints on g(2,3) directly to isolate the inner-compound case
probe_sg23(V) :- sum_ints(g(2, 3), 0, V).
probe_sg23_v0 :- probe_sg23(0).
probe_sg23_v1 :- probe_sg23(1).
probe_sg23_v2 :- probe_sg23(2).
probe_sg23_v3 :- probe_sg23(3).
probe_sg23_v4 :- probe_sg23(4).
probe_sg23_v5 :- probe_sg23(5).
probe_sg23_v6 :- probe_sg23(6).
