:- encoding(utf8).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_clojure_target').

:- dynamic user:wam_fact/1.
:- dynamic user:wam_foreign_pair/2.
:- dynamic user:wam_foreign_pair_query/1.
:- dynamic user:wam_foreign_stream_pair/2.
:- dynamic user:wam_foreign_stream_pair_query/1.
:- dynamic user:wam_execute_caller/1.
:- dynamic user:wam_call_caller/1.
:- dynamic user:wam_choice_fact/1.
:- dynamic user:wam_choice_caller/1.
:- dynamic user:wam_choice_or_z/1.
:- dynamic user:wam_bind_then_fact/1.
:- dynamic user:wam_bind_after_call/1.
:- dynamic user:wam_bind_before_execute/1.
:- dynamic user:wam_if_then_else/1.
:- dynamic user:wam_struct_fact/1.
:- dynamic user:wam_list_fact/1.
:- dynamic user:wam_use_struct/1.
:- dynamic user:wam_use_list/1.
:- dynamic user:wam_make_struct/1.
:- dynamic user:wam_make_list/1.
:- dynamic user:wam_double_struct_match/1.
:- dynamic user:wam_double_list_match/1.
:- dynamic user:wam_build_backtrack/1.
:- dynamic user:wam_env_build_backtrack/1.
:- dynamic user:wam_list_build_backtrack/1.
:- dynamic user:wam_env1/1.
:- dynamic user:wam_env2/1.
:- dynamic user:wam_env3/1.
:- dynamic user:wam_trail_choice/1.
:- dynamic user:wam_soft_cut_helper/1.
:- dynamic user:wam_soft_cut_outer_ok/1.
:- dynamic user:wam_cut_helper/1.
:- dynamic user:wam_hard_cut_outer_ok/1.
:- dynamic user:wam_not_b/1.
:- dynamic user:wam_fail_after_bind/1.
:- dynamic user:wam_atom_guard/1.
:- dynamic user:wam_integer_guard/1.
:- dynamic user:wam_number_guard/1.
:- dynamic user:wam_atomic_guard/1.
:- dynamic user:wam_nonvar_guard/1.
:- dynamic user:wam_unbound_arg/1.
:- dynamic user:wam_nonvar_unbound/1.
:- dynamic user:wam_var_guard/1.
:- dynamic user:wam_var_unbound/1.
:- dynamic user:wam_compound_guard/1.
:- dynamic user:wam_compound_unbound/1.
:- dynamic user:wam_callable_guard/1.
:- dynamic user:wam_callable_unbound/1.
:- dynamic user:wam_float_guard/1.
:- dynamic user:wam_float_unbound/1.
:- dynamic user:wam_is_list_guard/1.
:- dynamic user:wam_is_list_unbound/1.
:- dynamic user:wam_length_guard/2.
:- dynamic user:wam_length_bad_list/1.
:- dynamic user:wam_length_unbound_list/1.
:- dynamic user:wam_member_guard/2.
:- dynamic user:wam_member_backtrack_b/0.
:- dynamic user:wam_member_unbound_list/1.
:- dynamic user:wam_ground_guard/1.
:- dynamic user:wam_ground_unbound/1.
:- dynamic user:wam_ground_nested_unbound/1.
:- dynamic user:wam_arith_eq_42/1.
:- dynamic user:wam_arith_eq_float/1.
:- dynamic user:wam_arith_neq_42/1.
:- dynamic user:wam_arith_eq_unbound/1.
:- dynamic user:wam_arith_lt_42/1.
:- dynamic user:wam_arith_gt_42/1.
:- dynamic user:wam_arith_le_42/1.
:- dynamic user:wam_arith_ge_42/1.
:- dynamic user:wam_arith_lt_unbound/1.
:- dynamic user:wam_arith_is_inc_42/1.
:- dynamic user:wam_arith_is_combo_3_4/1.
:- dynamic user:wam_arith_is_neg_42/1.
:- dynamic user:wam_arith_is_div_5/1.
:- dynamic user:wam_arith_is_int_div_5/1.
:- dynamic user:wam_arith_is_mod_5/1.
:- dynamic user:wam_arith_is_unbound/1.

has(Code, Substr) :-
    once(sub_string(Code, _, _, _, Substr)).

user:wam_fact(a).
user:wam_foreign_pair_query(Y) :- user:wam_foreign_pair(a, Y).
user:wam_foreign_stream_pair_query(Y) :- user:wam_foreign_stream_pair(a, Y), Y = b.
user:wam_execute_caller(X) :- user:wam_fact(X).
user:wam_call_caller(X) :- user:wam_fact(X), user:wam_fact(X).
user:wam_choice_fact(a).
user:wam_choice_fact(b).
user:wam_choice_fact(c).
user:wam_choice_caller(X) :- user:wam_choice_fact(X).
user:wam_choice_or_z(X) :- user:wam_choice_fact(X).
user:wam_choice_or_z(z).
user:wam_bind_then_fact(X) :- Y = X, user:wam_fact(Y).
user:wam_bind_after_call(X) :- user:wam_fact(X), X = a.
user:wam_bind_before_execute(X) :- X = a, user:wam_fact(X).
user:wam_if_then_else(X) :- (X = a -> true ; X = b).
user:wam_struct_fact(f(a)).
user:wam_list_fact([a,b]).
user:wam_use_struct(X) :- user:wam_struct_fact(X).
user:wam_use_list(X) :- user:wam_list_fact(X).
user:wam_make_struct(X) :- X = f(a).
user:wam_make_list(X) :- X = [a,b].
user:wam_double_struct_match(X) :- user:wam_struct_fact(X), user:wam_struct_fact(X).
user:wam_double_list_match(X) :- user:wam_list_fact(X), user:wam_list_fact(X).
user:wam_build_backtrack(X) :- (X = f(a), fail ; X = f(b)).
user:wam_env_build_backtrack(X) :- (Y = f(a), fail ; Y = f(b)), X = Y.
user:wam_list_build_backtrack(X) :- (X = [a,b], fail ; X = [a,c]).
user:wam_env1(X) :- Y = X, Z = a, Y = Z.
user:wam_env2(X) :- user:wam_fact(X), Y = X, user:wam_fact(Y).
user:wam_env3(X) :- (X = a ; X = b), user:wam_fact(X).
user:wam_trail_choice(X) :- (Y = a ; Y = b), X = Y.
user:wam_soft_cut_helper(X) :- (X = a -> fail ; true).
user:wam_soft_cut_outer_ok(X) :- (user:wam_soft_cut_helper(Y), X = Y ; X = b).
user:wam_cut_helper(X) :- X = a, !, fail.
user:wam_hard_cut_outer_ok(X) :- (user:wam_cut_helper(Y), X = Y ; X = b).
user:wam_not_b(X) :- X \= b.
user:wam_fail_after_bind(X) :- X = a, fail.
user:wam_atom_guard(X) :- atom(X).
user:wam_integer_guard(X) :- integer(X).
user:wam_number_guard(X) :- number(X).
user:wam_atomic_guard(X) :- atomic(X).
user:wam_nonvar_guard(X) :- nonvar(X).
user:wam_unbound_arg(_).
user:wam_nonvar_unbound(_) :- user:wam_unbound_arg(Y), nonvar(Y).
user:wam_var_guard(X) :- var(X).
user:wam_var_unbound(_) :- user:wam_unbound_arg(Y), var(Y).
user:wam_compound_guard(X) :- compound(X).
user:wam_compound_unbound(_) :- user:wam_unbound_arg(Y), compound(Y).
user:wam_callable_guard(X) :- callable(X).
user:wam_callable_unbound(_) :- user:wam_unbound_arg(Y), callable(Y).
user:wam_float_guard(X) :- float(X).
user:wam_float_unbound(_) :- user:wam_unbound_arg(Y), float(Y).
user:wam_is_list_guard(X) :- is_list(X).
user:wam_is_list_unbound(_) :- user:wam_unbound_arg(Y), is_list(Y).
user:wam_length_guard(L, N) :- length(L, N).
user:wam_length_bad_list(N) :- length([a|b], N).
user:wam_length_unbound_list(N) :- user:wam_unbound_arg(Y), length(Y, N).
user:wam_member_guard(X, L) :- member(X, L).
user:wam_member_backtrack_b :- member(X, [a,b]), X = b.
user:wam_member_unbound_list(X) :- user:wam_unbound_arg(Y), member(X, Y).
user:wam_ground_guard(X) :- ground(X).
user:wam_ground_unbound(_) :- user:wam_unbound_arg(Y), ground(Y).
user:wam_ground_nested_unbound(_) :- user:wam_unbound_arg(Y), ground(f(Y)).
user:wam_arith_eq_42(X) :- X =:= 42.
user:wam_arith_eq_float(X) :- X =:= 3.5.
user:wam_arith_neq_42(X) :- X =\= 42.
user:wam_arith_eq_unbound(_) :- user:wam_unbound_arg(Y), Y =:= 42.
user:wam_arith_lt_42(X) :- X < 42.
user:wam_arith_gt_42(X) :- X > 42.
user:wam_arith_le_42(X) :- X =< 42.
user:wam_arith_ge_42(X) :- X >= 42.
user:wam_arith_lt_unbound(_) :- user:wam_unbound_arg(Y), Y < 42.
user:wam_arith_is_inc_42(M) :- M is 42 + 1.
user:wam_arith_is_combo_3_4(R) :- R is 3 + 4 * 2.
user:wam_arith_is_neg_42(M) :- M is -42.
user:wam_arith_is_div_5(M) :- M is 5 / 2.
user:wam_arith_is_int_div_5(M) :- M is 5 // 2.
user:wam_arith_is_mod_5(M) :- M is 5 mod 2.
user:wam_arith_is_unbound(_) :- user:wam_unbound_arg(Y), _ is Y + 1.

:- initialization(main, main).

main :-
    catch(run_smoke, Error, (print_message(error, Error), halt(1))),
    halt(0).

run_smoke :-
    unique_tmp_dir('tmp_wam_clojure_smoke', TmpDir),
    write_wam_clojure_project(
        [ user:wam_execute_caller/1,
          user:wam_call_caller/1,
          user:wam_fact/1,
          user:wam_foreign_pair_query/1,
          user:wam_foreign_pair/2,
          user:wam_foreign_stream_pair_query/1,
          user:wam_foreign_stream_pair/2,
          user:wam_choice_fact/1,
          user:wam_choice_caller/1,
          user:wam_choice_or_z/1,
          user:wam_bind_then_fact/1,
          user:wam_bind_after_call/1,
          user:wam_bind_before_execute/1,
          user:wam_if_then_else/1,
          user:wam_struct_fact/1,
          user:wam_list_fact/1,
          user:wam_use_struct/1,
          user:wam_use_list/1,
          user:wam_make_struct/1,
          user:wam_make_list/1,
          user:wam_double_struct_match/1,
          user:wam_double_list_match/1,
          user:wam_build_backtrack/1,
          user:wam_env_build_backtrack/1,
          user:wam_list_build_backtrack/1,
          user:wam_env1/1,
          user:wam_env2/1,
          user:wam_env3/1,
          user:wam_trail_choice/1,
          user:wam_soft_cut_helper/1,
          user:wam_soft_cut_outer_ok/1,
          user:wam_cut_helper/1,
          user:wam_hard_cut_outer_ok/1,
          user:wam_not_b/1,
          user:wam_fail_after_bind/1,
          user:wam_atom_guard/1,
          user:wam_integer_guard/1,
          user:wam_number_guard/1,
          user:wam_atomic_guard/1,
          user:wam_nonvar_guard/1,
          user:wam_unbound_arg/1,
          user:wam_nonvar_unbound/1,
          user:wam_var_guard/1,
          user:wam_var_unbound/1,
          user:wam_compound_guard/1,
          user:wam_compound_unbound/1,
          user:wam_callable_guard/1,
          user:wam_callable_unbound/1,
          user:wam_float_guard/1,
          user:wam_float_unbound/1,
          user:wam_is_list_guard/1,
          user:wam_is_list_unbound/1,
          user:wam_length_guard/2,
          user:wam_length_bad_list/1,
          user:wam_length_unbound_list/1,
          user:wam_member_guard/2,
          user:wam_member_backtrack_b/0,
          user:wam_member_unbound_list/1,
          user:wam_ground_guard/1,
          user:wam_ground_unbound/1,
          user:wam_ground_nested_unbound/1,
          user:wam_arith_eq_42/1,
          user:wam_arith_eq_float/1,
          user:wam_arith_neq_42/1,
          user:wam_arith_eq_unbound/1,
          user:wam_arith_lt_42/1,
          user:wam_arith_gt_42/1,
          user:wam_arith_le_42/1,
          user:wam_arith_ge_42/1,
          user:wam_arith_lt_unbound/1,
          user:wam_arith_is_inc_42/1,
          user:wam_arith_is_combo_3_4/1,
          user:wam_arith_is_neg_42/1,
          user:wam_arith_is_div_5/1,
          user:wam_arith_is_int_div_5/1,
          user:wam_arith_is_mod_5/1,
          user:wam_arith_is_unbound/1
        ],
        [ namespace('generated.wam_exec_test'),
          module_name('wam-clojure-exec-test'),
          foreign_predicates([wam_fact/1, wam_foreign_pair/2, wam_foreign_stream_pair/2]),
          clojure_foreign_handlers([
              handler(wam_fact/1, "(fn [args] (= (first args) \"a\"))"),
              handler(wam_foreign_pair/2, "(fn [args] (if (= (first args) \"a\") {:bindings {2 \"b\"}} false))"),
              handler(wam_foreign_stream_pair/2, "(fn [args] (if (= (first args) \"a\") {:solutions [{:bindings {2 \"a\"}} {:bindings {2 \"b\"}}]} false))")
          ])
        ],
        TmpDir),
    assert_lowered_read_unify_prefix_emitted(TmpDir),
    assert_lowered_env_prefix_emitted(TmpDir),
    assert_lowered_execute_emitted(TmpDir),
    assert_lowered_call_emitted(TmpDir),
    assert_lowered_cut_builtin_emitted(TmpDir),
    assert_lowered_not_unify_builtin_emitted(TmpDir),
    assert_lowered_fail_builtin_emitted(TmpDir),
    assert_lowered_atom_builtin_emitted(TmpDir),
    assert_lowered_integer_builtin_emitted(TmpDir),
    assert_lowered_number_builtin_emitted(TmpDir),
    assert_lowered_atomic_builtin_emitted(TmpDir),
    assert_lowered_nonvar_builtin_emitted(TmpDir),
    assert_lowered_var_builtin_emitted(TmpDir),
    assert_lowered_compound_builtin_emitted(TmpDir),
    assert_lowered_callable_builtin_emitted(TmpDir),
    assert_lowered_float_builtin_emitted(TmpDir),
    assert_lowered_is_list_builtin_emitted(TmpDir),
    assert_lowered_length_builtin_emitted(TmpDir),
    assert_lowered_member_builtin_emitted(TmpDir),
    assert_lowered_ground_builtin_emitted(TmpDir),
    assert_lowered_arithmetic_comparison_builtin_emitted(TmpDir),
    assert_multiclause_wrappers_runtime_mediated(TmpDir),
    smoke_cases(Cases),
    verify_outputs(TmpDir, Cases),
    delete_directory_and_contents(TmpDir),
    writeln('wam_clojure_runtime_smoke: ok').

smoke_cases([
    case('wam_execute_caller/1', 'a', "true"),
    case('wam_execute_caller/1', 'b', "false"),
    case('wam_call_caller/1', 'a', "true"),
    case('wam_call_caller/1', 'b', "false"),
    case('wam_foreign_pair_query/1', b, "true"),
    case('wam_foreign_pair_query/1', c, "false"),
    case('wam_foreign_stream_pair_query/1', a, "false"),
    case('wam_foreign_stream_pair_query/1', b, "true"),
    case('wam_foreign_stream_pair_query/1', c, "false"),
    case('wam_choice_caller/1', 'a', "true"),
    case('wam_choice_caller/1', 'b', "true"),
    case('wam_choice_caller/1', 'c', "true"),
    case('wam_choice_caller/1', 'd', "false"),
    case('wam_choice_or_z/1', 'z', "true"),
    case('wam_bind_then_fact/1', 'a', "true"),
    case('wam_bind_then_fact/1', 'b', "false"),
    case('wam_bind_after_call/1', 'a', "true"),
    case('wam_bind_after_call/1', 'b', "false"),
    case('wam_bind_before_execute/1', 'a', "true"),
    case('wam_bind_before_execute/1', 'b', "false"),
    case('wam_if_then_else/1', 'a', "true"),
    case('wam_if_then_else/1', 'b', "true"),
    case('wam_if_then_else/1', 'c', "false"),
    case('wam_use_struct/1', 'f(a)', "true"),
    case('wam_use_struct/1', 'f(b)', "false"),
    case('wam_use_list/1', '[a,b]', "true"),
    case('wam_use_list/1', '[a,c]', "false"),
    case('wam_double_struct_match/1', 'f(a)', "true"),
    case('wam_double_struct_match/1', 'f(b)', "false"),
    case('wam_double_list_match/1', '[a,b]', "true"),
    case('wam_double_list_match/1', '[a,c]', "false"),
    case('wam_make_struct/1', 'f(a)', "true"),
    case('wam_make_list/1', '[a,b]', "true"),
    case('wam_build_backtrack/1', 'f(a)', "false"),
    case('wam_build_backtrack/1', 'f(b)', "true"),
    case('wam_env_build_backtrack/1', 'f(a)', "false"),
    case('wam_env_build_backtrack/1', 'f(b)', "true"),
    case('wam_list_build_backtrack/1', '[a,b]', "false"),
    case('wam_list_build_backtrack/1', '[a,c]', "true"),
    case('wam_env1/1', a, "true"),
    case('wam_env1/1', b, "false"),
    case('wam_env2/1', a, "true"),
    case('wam_env2/1', b, "false"),
    case('wam_env3/1', a, "true"),
    case('wam_env3/1', b, "false"),
    case('wam_trail_choice/1', a, "true"),
    case('wam_trail_choice/1', b, "true"),
    case('wam_trail_choice/1', c, "false"),
    case('wam_soft_cut_outer_ok/1', b, "true"),
    case('wam_hard_cut_outer_ok/1', b, "true"),
    case('wam_not_b/1', a, "true"),
    case('wam_not_b/1', b, "false"),
    case('wam_fail_after_bind/1', a, "false"),
    case('wam_fail_after_bind/1', b, "false"),
    case('wam_atom_guard/1', a, "true"),
    case('wam_atom_guard/1', 'f(a)', "false"),
    case('wam_integer_guard/1', 42, "true"),
    case('wam_integer_guard/1', a, "false"),
    case('wam_number_guard/1', 42, "true"),
    case('wam_number_guard/1', a, "false"),
    case('wam_atomic_guard/1', a, "true"),
    case('wam_atomic_guard/1', 42, "true"),
    case('wam_atomic_guard/1', 'f(a)', "false"),
    case('wam_nonvar_guard/1', a, "true"),
    case('wam_nonvar_guard/1', 42, "true"),
    case('wam_nonvar_guard/1', 'f(a)', "true"),
    case('wam_nonvar_unbound/1', a, "false"),
    case('wam_var_guard/1', a, "false"),
    case('wam_var_guard/1', 42, "false"),
    case('wam_var_guard/1', 'f(a)', "false"),
    case('wam_var_unbound/1', a, "true"),
    case('wam_compound_guard/1', a, "false"),
    case('wam_compound_guard/1', 42, "false"),
    case('wam_compound_guard/1', 'f(a)', "true"),
    case('wam_compound_guard/1', '[a,b]', "true"),
    case('wam_compound_unbound/1', a, "false"),
    case('wam_callable_guard/1', a, "true"),
    case('wam_callable_guard/1', 42, "false"),
    case('wam_callable_guard/1', 'f(a)', "true"),
    case('wam_callable_guard/1', '[a,b]', "true"),
    case('wam_callable_unbound/1', a, "false"),
    case('wam_float_guard/1', 3.5, "true"),
    case('wam_float_guard/1', 42, "false"),
    case('wam_float_guard/1', a, "false"),
    case('wam_float_guard/1', 'f(a)', "false"),
    case('wam_float_unbound/1', a, "false"),
    case('wam_is_list_guard/1', '[]', "true"),
    case('wam_is_list_guard/1', '[a,b]', "true"),
    case('wam_is_list_guard/1', '[a|b]', "false"),
    case('wam_is_list_guard/1', a, "false"),
    case('wam_is_list_guard/1', 'f(a)', "false"),
    case('wam_is_list_unbound/1', a, "false"),
    case('wam_length_guard/2', args('[a,b]', 2), "true"),
    case('wam_length_guard/2', args('[a,b]', 1), "false"),
    case('wam_length_guard/2', args('[]', 0), "true"),
    case('wam_length_bad_list/1', 1, "false"),
    case('wam_length_unbound_list/1', 0, "false"),
    case('wam_member_guard/2', args(a, '[a,b]'), "true"),
    case('wam_member_guard/2', args(b, '[a,b]'), "true"),
    case('wam_member_guard/2', args(c, '[a,b]'), "false"),
    case('wam_member_backtrack_b/0', no_args, "true"),
    case('wam_member_unbound_list/1', a, "false"),
    case('wam_ground_guard/1', a, "true"),
    case('wam_ground_guard/1', 42, "true"),
    case('wam_ground_guard/1', 3.5, "true"),
    case('wam_ground_guard/1', 'f(a)', "true"),
    case('wam_ground_guard/1', '[a,b]', "true"),
    case('wam_ground_unbound/1', a, "false"),
    case('wam_ground_nested_unbound/1', a, "false"),
    case('wam_arith_eq_42/1', 42, "true"),
    case('wam_arith_eq_42/1', 3.5, "false"),
    case('wam_arith_eq_float/1', 3.5, "true"),
    case('wam_arith_neq_42/1', 3.5, "true"),
    case('wam_arith_neq_42/1', 42, "false"),
    case('wam_arith_eq_unbound/1', a, "false"),
    case('wam_arith_lt_42/1', 3.5, "true"),
    case('wam_arith_lt_42/1', 42, "false"),
    case('wam_arith_gt_42/1', 43, "true"),
    case('wam_arith_gt_42/1', 42, "false"),
    case('wam_arith_le_42/1', 42, "true"),
    case('wam_arith_le_42/1', 43, "false"),
    case('wam_arith_ge_42/1', 42, "true"),
    case('wam_arith_ge_42/1', 3.5, "false"),
    case('wam_arith_lt_unbound/1', a, "false"),
    case('wam_arith_is_inc_42/1', 43, "true"),
    case('wam_arith_is_inc_42/1', 42, "false"),
    case('wam_arith_is_combo_3_4/1', 11, "true"),
    case('wam_arith_is_neg_42/1', -42, "true"),
    case('wam_arith_is_div_5/1', 2.5, "true"),
    case('wam_arith_is_int_div_5/1', 2, "true"),
    case('wam_arith_is_mod_5/1', 1, "true"),
    case('wam_arith_is_unbound/1', a, "false")
]).

assert_lowered_read_unify_prefix_emitted(ProjectDir) :-
    directory_file_path(ProjectDir, 'src/generated/wam_exec_test/core.clj', CorePath),
    read_file_to_string(CorePath, CoreCode, []),
    has(CoreCode, "defn lowered-wam-struct-fact-1"),
    has(CoreCode, "defn lowered-wam-list-fact-1"),
    has(CoreCode, "runtime/enter-unify-mode"),
    has(CoreCode, "runtime/pop-unify-item"),
    has(CoreCode, "runtime/unify-values").

assert_lowered_env_prefix_emitted(ProjectDir) :-
    directory_file_path(ProjectDir, 'src/generated/wam_exec_test/core.clj', CorePath),
    read_file_to_string(CorePath, CoreCode, []),
    has(CoreCode, "defn lowered-wam-env1-1"),
    has(CoreCode, "update :env-stack conj {}"),
    has(CoreCode, "assoc :cut-bar"),
    has(CoreCode, "update :env-stack #(if (seq %) (pop %) %)"),
    has(CoreCode, "runtime/unify-values"),
    has(CoreCode, "runtime/succeed-state").

assert_lowered_execute_emitted(ProjectDir) :-
    directory_file_path(ProjectDir, 'src/generated/wam_exec_test/core.clj', CorePath),
    read_file_to_string(CorePath, CoreCode, []),
    has(CoreCode, "defn lowered-wam-execute-caller-1"),
    has(CoreCode, "if-let [target-pc"),
    has(CoreCode, "(get (:labels"),
    has(CoreCode, "\"wam_fact/1\""),
    has(CoreCode, ":pc target-pc").

assert_lowered_call_emitted(ProjectDir) :-
    directory_file_path(ProjectDir, 'src/generated/wam_exec_test/core.clj', CorePath),
    read_file_to_string(CorePath, CoreCode, []),
    has(CoreCode, "defn lowered-wam-call-caller-1"),
    has(CoreCode, "if-let [target-pc"),
    has(CoreCode, "(get (:labels"),
    has(CoreCode, "\"wam_fact/1\""),
    has(CoreCode, "update :stack conj (inc (:pc"),
    has(CoreCode, ":pc target-pc").

assert_lowered_cut_builtin_emitted(ProjectDir) :-
    directory_file_path(ProjectDir, 'src/generated/wam_exec_test/core.clj', CorePath),
    read_file_to_string(CorePath, CoreCode, []),
    has(CoreCode, "defn lowered-wam-cut-helper-1"),
    has(CoreCode, "update :choice-points"),
    has(CoreCode, "take (:cut-bar").

assert_lowered_not_unify_builtin_emitted(ProjectDir) :-
    directory_file_path(ProjectDir, 'src/generated/wam_exec_test/core.clj', CorePath),
    read_file_to_string(CorePath, CoreCode, []),
    has(CoreCode, "defn lowered-wam-not-b-1"),
    has(CoreCode, "runtime/unifiable?").

assert_lowered_fail_builtin_emitted(ProjectDir) :-
    directory_file_path(ProjectDir, 'src/generated/wam_exec_test/core.clj', CorePath),
    read_file_to_string(CorePath, CoreCode, []),
    has(CoreCode, "defn lowered-wam-fail-after-bind-1"),
    has(CoreCode, "runtime/backtrack").

assert_lowered_atom_builtin_emitted(ProjectDir) :-
    directory_file_path(ProjectDir, 'src/generated/wam_exec_test/core.clj', CorePath),
    read_file_to_string(CorePath, CoreCode, []),
    has(CoreCode, "defn lowered-wam-atom-guard-1"),
    has(CoreCode, "runtime/atom-term?").

assert_lowered_integer_builtin_emitted(ProjectDir) :-
    directory_file_path(ProjectDir, 'src/generated/wam_exec_test/core.clj', CorePath),
    read_file_to_string(CorePath, CoreCode, []),
    has(CoreCode, "defn lowered-wam-integer-guard-1"),
    has(CoreCode, "integer? value").

assert_lowered_number_builtin_emitted(ProjectDir) :-
    directory_file_path(ProjectDir, 'src/generated/wam_exec_test/core.clj', CorePath),
    read_file_to_string(CorePath, CoreCode, []),
    has(CoreCode, "defn lowered-wam-number-guard-1"),
    has(CoreCode, "number? value").

assert_lowered_atomic_builtin_emitted(ProjectDir) :-
    directory_file_path(ProjectDir, 'src/generated/wam_exec_test/core.clj', CorePath),
    read_file_to_string(CorePath, CoreCode, []),
    has(CoreCode, "defn lowered-wam-atomic-guard-1"),
    has(CoreCode, "runtime/atom-term? value"),
    has(CoreCode, "number? value").

assert_lowered_nonvar_builtin_emitted(ProjectDir) :-
    directory_file_path(ProjectDir, 'src/generated/wam_exec_test/core.clj', CorePath),
    read_file_to_string(CorePath, CoreCode, []),
    has(CoreCode, "defn lowered-wam-nonvar-guard-1"),
    has(CoreCode, "defn lowered-wam-nonvar-unbound-1"),
    has(CoreCode, "not= value ::lowered-unbound"),
    has(CoreCode, "not (runtime/logic-var? value)").

assert_lowered_var_builtin_emitted(ProjectDir) :-
    directory_file_path(ProjectDir, 'src/generated/wam_exec_test/core.clj', CorePath),
    read_file_to_string(CorePath, CoreCode, []),
    has(CoreCode, "defn lowered-wam-var-guard-1"),
    has(CoreCode, "defn lowered-wam-var-unbound-1"),
    has(CoreCode, "= value ::lowered-unbound"),
    has(CoreCode, "runtime/logic-var? value").

assert_lowered_compound_builtin_emitted(ProjectDir) :-
    directory_file_path(ProjectDir, 'src/generated/wam_exec_test/core.clj', CorePath),
    read_file_to_string(CorePath, CoreCode, []),
    has(CoreCode, "defn lowered-wam-compound-guard-1"),
    has(CoreCode, "defn lowered-wam-compound-unbound-1"),
    has(CoreCode, "runtime/structure-term? value").

assert_lowered_callable_builtin_emitted(ProjectDir) :-
    directory_file_path(ProjectDir, 'src/generated/wam_exec_test/core.clj', CorePath),
    read_file_to_string(CorePath, CoreCode, []),
    has(CoreCode, "defn lowered-wam-callable-guard-1"),
    has(CoreCode, "defn lowered-wam-callable-unbound-1"),
    has(CoreCode, "runtime/atom-term? value"),
    has(CoreCode, "runtime/structure-term? value").

assert_lowered_float_builtin_emitted(ProjectDir) :-
    directory_file_path(ProjectDir, 'src/generated/wam_exec_test/core.clj', CorePath),
    read_file_to_string(CorePath, CoreCode, []),
    has(CoreCode, "defn lowered-wam-float-guard-1"),
    has(CoreCode, "defn lowered-wam-float-unbound-1"),
    has(CoreCode, "float? value").

assert_lowered_is_list_builtin_emitted(ProjectDir) :-
    directory_file_path(ProjectDir, 'src/generated/wam_exec_test/core.clj', CorePath),
    read_file_to_string(CorePath, CoreCode, []),
    has(CoreCode, "defn lowered-wam-is-list-guard-1"),
    has(CoreCode, "defn lowered-wam-is-list-unbound-1"),
    has(CoreCode, "runtime/proper-list-term?").

assert_lowered_length_builtin_emitted(ProjectDir) :-
    directory_file_path(ProjectDir, 'src/generated/wam_exec_test/core.clj', CorePath),
    read_file_to_string(CorePath, CoreCode, []),
    has(CoreCode, "defn lowered-wam-length-guard-2"),
    has(CoreCode, "defn lowered-wam-length-bad-list-1"),
    has(CoreCode, "defn lowered-wam-length-unbound-list-1"),
    has(CoreCode, "runtime/proper-list-length").

assert_lowered_member_builtin_emitted(ProjectDir) :-
    directory_file_path(ProjectDir, 'src/generated/wam_exec_test/core.clj', CorePath),
    read_file_to_string(CorePath, CoreCode, []),
    has(CoreCode, "defn lowered-wam-member-guard-2"),
    has(CoreCode, "defn lowered-wam-member-backtrack-b-0"),
    has(CoreCode, "defn lowered-wam-member-unbound-list-1"),
    has(CoreCode, "runtime/apply-member-solution").

assert_lowered_ground_builtin_emitted(ProjectDir) :-
    directory_file_path(ProjectDir, 'src/generated/wam_exec_test/core.clj', CorePath),
    read_file_to_string(CorePath, CoreCode, []),
    has(CoreCode, "defn lowered-wam-ground-guard-1"),
    has(CoreCode, "defn lowered-wam-ground-unbound-1"),
    has(CoreCode, "defn lowered-wam-ground-nested-unbound-1"),
    has(CoreCode, "runtime/ground-term?").

assert_lowered_arithmetic_comparison_builtin_emitted(ProjectDir) :-
    directory_file_path(ProjectDir, 'src/generated/wam_exec_test/core.clj', CorePath),
    read_file_to_string(CorePath, CoreCode, []),
    has(CoreCode, "defn lowered-wam-arith-eq-42-1"),
    has(CoreCode, "defn lowered-wam-arith-eq-float-1"),
    has(CoreCode, "defn lowered-wam-arith-neq-42-1"),
    has(CoreCode, "defn lowered-wam-arith-eq-unbound-1"),
    has(CoreCode, "defn lowered-wam-arith-lt-42-1"),
    has(CoreCode, "defn lowered-wam-arith-gt-42-1"),
    has(CoreCode, "defn lowered-wam-arith-le-42-1"),
    has(CoreCode, "defn lowered-wam-arith-ge-42-1"),
    has(CoreCode, "defn lowered-wam-arith-lt-unbound-1"),
    has(CoreCode, "defn lowered-wam-arith-is-inc-42-1"),
    has(CoreCode, "defn lowered-wam-arith-is-combo-3-4-1"),
    has(CoreCode, "defn lowered-wam-arith-is-neg-42-1"),
    has(CoreCode, "defn lowered-wam-arith-is-div-5-1"),
    has(CoreCode, "defn lowered-wam-arith-is-int-div-5-1"),
    has(CoreCode, "defn lowered-wam-arith-is-mod-5-1"),
    has(CoreCode, "defn lowered-wam-arith-is-unbound-1"),
    has(CoreCode, "runtime/arithmetic-equal?"),
    has(CoreCode, "runtime/arithmetic-not-equal?"),
    has(CoreCode, "runtime/eval-arithmetic-term"),
    has(CoreCode, "runtime/arithmetic-less?"),
    has(CoreCode, "runtime/arithmetic-greater?"),
    has(CoreCode, "runtime/arithmetic-less-or-equal?"),
    has(CoreCode, "runtime/arithmetic-greater-or-equal?").

assert_multiclause_wrappers_runtime_mediated(ProjectDir) :-
    directory_file_path(ProjectDir, 'src/generated/wam_exec_test/core.clj', CorePath),
    read_file_to_string(CorePath, CoreCode, []),
    assert_empty_lowered_wrapper(CoreCode, "lowered-wam-choice-fact-1"),
    assert_empty_lowered_wrapper(CoreCode, "lowered-wam-choice-or-z-1"),
    assert_empty_lowered_wrapper(CoreCode, "lowered-wam-trail-choice-1").

assert_empty_lowered_wrapper(CoreCode, FuncName) :-
    (   lowered_wrapper_absent_or_empty(CoreCode, FuncName)
    ->  true
    ;   throw(error(multiclause_wrapper_not_runtime_mediated(FuncName), _))
    ).

lowered_wrapper_absent_or_empty(CoreCode, FuncName) :-
    format(string(Header), "(defn ~w [state]", [FuncName]),
    \+ has(CoreCode, Header),
    !.
lowered_wrapper_absent_or_empty(CoreCode, FuncName) :-
    empty_lowered_wrapper(CoreCode, FuncName).

empty_lowered_wrapper(CoreCode, FuncName) :-
    format(string(Expected), "(defn ~w [state]\n  state\n)", [FuncName]),
    has(CoreCode, Expected).

verify_outputs(ProjectDir, Cases) :-
    run_clojure_predicates_batch(ProjectDir, Cases, Actuals),
    verify_expected_outputs(Cases, Actuals).

verify_expected_outputs([], []).
verify_expected_outputs([case(PredKey, Arg, Expected)|RestCases], [Actual|RestActuals]) :-
    (   Actual == Expected
    ->  verify_expected_outputs(RestCases, RestActuals)
    ;   throw(error(assertion_error(PredKey, Arg, Expected, Actual), _))
    ).
verify_expected_outputs(Cases, Actuals) :-
    throw(error(batch_output_count_mismatch(Cases, Actuals), _)).

run_clojure_predicates_batch(ProjectDir, Cases, Outputs) :-
    find_clojure_classpath(ClassPath),
    cases_to_edn(Cases, Input),
    process_create(path(java),
                   ['-cp', ClassPath, 'clojure.main', '-m',
                    'generated.wam_exec_test.core', '--batch'],
                   [cwd(ProjectDir), stdin(pipe(In)), stdout(pipe(Out)), stderr(pipe(Err))]),
    write(In, Input),
    close(In),
    read_string(Out, _, OutStr0),
    read_string(Err, _, ErrStr),
    close(Out),
    close(Err),
    split_string(OutStr0, "\n", "\r\t ", OutputLines0),
    exclude(empty_string, OutputLines0, Outputs),
    (   ErrStr == ""
    ->  true
    ;   throw(error(java_batch_stderr(ErrStr), _))
    ).

cases_to_edn(Cases, Edn) :-
    maplist(case_to_edn, Cases, CaseEdns),
    atomic_list_concat(CaseEdns, '\n ', Body),
    format(string(Edn), "[~w]", [Body]).

case_to_edn(case(PredKey, Arg, _Expected), Edn) :-
    arg_to_edn_vector(Arg, ArgsEdn),
    format(string(Edn), "{:pred \"~w\" :args ~w}", [PredKey, ArgsEdn]).

arg_to_edn_vector(no_args, "[]") :- !.
arg_to_edn_vector(args(Arg1, Arg2), ArgsEdn) :-
    !,
    prolog_term_string_to_edn(Arg1, EdnArg1),
    prolog_term_string_to_edn(Arg2, EdnArg2),
    format(string(ArgsEdn), "[~w ~w]", [EdnArg1, EdnArg2]).
arg_to_edn_vector(Arg, ArgsEdn) :-
    prolog_term_string_to_edn(Arg, EdnArg),
    format(string(ArgsEdn), "[~w]", [EdnArg]).

empty_string("").

verify_output(ProjectDir, PredKey, Arg, Expected) :-
    run_clojure_predicate(ProjectDir, PredKey, Arg, Actual),
    (   Actual == Expected
    ->  true
    ;   throw(error(assertion_error(PredKey, Arg, Expected, Actual), _))
    ).

run_clojure_predicate(ProjectDir, PredKey, args(Arg1, Arg2), Output) :-
    find_clojure_classpath(ClassPath),
    prolog_term_string_to_edn(Arg1, EdnArg1),
    prolog_term_string_to_edn(Arg2, EdnArg2),
    process_create(path(java),
                   ['-cp', ClassPath, 'clojure.main', '-m',
                    'generated.wam_exec_test.core', PredKey, EdnArg1, EdnArg2],
                   [cwd(ProjectDir), stdout(pipe(Out)), stderr(pipe(Err))]),
    read_string(Out, _, OutStr0),
    read_string(Err, _, ErrStr),
    close(Out),
    close(Err),
    normalize_space(string(Output), OutStr0),
    (   ErrStr == ""
    ->  true
    ;   throw(error(java_stderr(PredKey, args(Arg1, Arg2), ErrStr), _))
    ).

run_clojure_predicate(ProjectDir, PredKey, no_args, Output) :-
    find_clojure_classpath(ClassPath),
    process_create(path(java),
                   ['-cp', ClassPath, 'clojure.main', '-m',
                    'generated.wam_exec_test.core', PredKey],
                   [cwd(ProjectDir), stdout(pipe(Out)), stderr(pipe(Err))]),
    read_string(Out, _, OutStr0),
    read_string(Err, _, ErrStr),
    close(Out),
    close(Err),
    normalize_space(string(Output), OutStr0),
    (   ErrStr == ""
    ->  true
    ;   throw(error(java_stderr(PredKey, no_args, ErrStr), _))
    ).

run_clojure_predicate(ProjectDir, PredKey, Arg, Output) :-
    find_clojure_classpath(ClassPath),
    prolog_term_string_to_edn(Arg, EdnArg),
    process_create(path(java),
                   ['-cp', ClassPath, 'clojure.main', '-m',
                    'generated.wam_exec_test.core', PredKey, EdnArg],
                   [cwd(ProjectDir), stdout(pipe(Out)), stderr(pipe(Err))]),
    read_string(Out, _, OutStr0),
    read_string(Err, _, ErrStr),
    close(Out),
    close(Err),
    normalize_space(string(Output), OutStr0),
    (   ErrStr == ""
    ->  true
    ;   throw(error(java_stderr(PredKey, Arg, ErrStr), _))
    ).

prolog_term_string_to_edn(a, "\"a\"") :- !.
prolog_term_string_to_edn(b, "\"b\"") :- !.
prolog_term_string_to_edn(c, "\"c\"") :- !.
prolog_term_string_to_edn(d, "\"d\"") :- !.
prolog_term_string_to_edn(z, "\"z\"") :- !.
prolog_term_string_to_edn('[]', "\"[]\"") :- !.
prolog_term_string_to_edn(-42, "-42") :- !.
prolog_term_string_to_edn(0, "0") :- !.
prolog_term_string_to_edn(1, "1") :- !.
prolog_term_string_to_edn(2, "2") :- !.
prolog_term_string_to_edn(2.5, "2.5") :- !.
prolog_term_string_to_edn(3.5, "3.5") :- !.
prolog_term_string_to_edn(11, "11") :- !.
prolog_term_string_to_edn(42, "42") :- !.
prolog_term_string_to_edn(43, "43") :- !.
prolog_term_string_to_edn("a", "\"a\"") :- !.
prolog_term_string_to_edn("b", "\"b\"") :- !.
prolog_term_string_to_edn("c", "\"c\"") :- !.
prolog_term_string_to_edn("d", "\"d\"") :- !.
prolog_term_string_to_edn("z", "\"z\"") :- !.
prolog_term_string_to_edn('f(a)', "{:tag :struct :functor \"f/1\" :args [\"a\"]}") :- !.
prolog_term_string_to_edn('f(b)', "{:tag :struct :functor \"f/1\" :args [\"b\"]}") :- !.
prolog_term_string_to_edn('[a,b]', "{:tag :struct :functor \"[|]/2\" :args [\"a\" {:tag :struct :functor \"[|]/2\" :args [\"b\" \"[]\"]}]}") :- !.
prolog_term_string_to_edn('[a,c]', "{:tag :struct :functor \"[|]/2\" :args [\"a\" {:tag :struct :functor \"[|]/2\" :args [\"c\" \"[]\"]}]}") :- !.
prolog_term_string_to_edn('[a|b]', "{:tag :struct :functor \"[|]/2\" :args [\"a\" \"b\"]}") :- !.
prolog_term_string_to_edn("f(a)", "{:tag :struct :functor \"f/1\" :args [\"a\"]}") :- !.
prolog_term_string_to_edn("f(b)", "{:tag :struct :functor \"f/1\" :args [\"b\"]}") :- !.
prolog_term_string_to_edn("[a,b]", "{:tag :struct :functor \"[|]/2\" :args [\"a\" {:tag :struct :functor \"[|]/2\" :args [\"b\" \"[]\"]}]}") :- !.
prolog_term_string_to_edn("[a,c]", "{:tag :struct :functor \"[|]/2\" :args [\"a\" {:tag :struct :functor \"[|]/2\" :args [\"c\" \"[]\"]}]}") :- !.
prolog_term_string_to_edn(Atom, Atom).

find_clojure_classpath(ClassPath) :-
    findall(Path,
        ( member(Path,
              [ '/data/data/com.termux/files/home/.m2/repository/org/clojure/clojure/1.11.1/clojure-1.11.1.jar',
                '/data/data/com.termux/files/home/.m2/repository/org/clojure/spec.alpha/0.3.218/spec.alpha-0.3.218.jar',
                '/data/data/com.termux/files/home/.m2/repository/org/clojure/core.specs.alpha/0.2.62/core.specs.alpha-0.2.62.jar'
              ]),
          exists_file(Path)
        ),
        JarPaths),
    JarPaths \= [],
    atomic_list_concat(['src'|JarPaths], :, ClassPath).

unique_tmp_dir(Prefix, TmpDir) :-
    get_time(T),
    Stamp is floor(T * 1000),
    format(atom(TmpDir), '~w_~w', [Prefix, Stamp]).
