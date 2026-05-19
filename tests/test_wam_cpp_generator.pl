:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_cpp_generator.pl — plunit tests for the hybrid C++ WAM target.
%
% Mirrors tests/test_wam_lua_generator.pl in structure, but scoped to the
% subset of behaviour the initial wam_cpp_target / wam_cpp_lowered_emitter
% pair guarantees: exports, registry wiring, project layout, lowerability
% checks, and lowered-function emission. End-to-end compile-and-run tests
% are gated on the presence of a host C++17 compiler (g++ / clang++).

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module('../src/unifyweaver/targets/wam_cpp_target').
:- use_module('../src/unifyweaver/targets/wam_cpp_lowered_emitter').
:- use_module('../src/unifyweaver/core/target_registry').

:- begin_tests(wam_cpp_generator).

:- dynamic user:wam_cpp_fact/1.
:- dynamic user:wam_cpp_choice/1.
:- dynamic user:wam_cpp_caller/1.
:- dynamic user:wam_cpp_rect/1.
:- dynamic user:wam_cpp_has_rect/0.
:- dynamic user:wam_cpp_has_rect_wrong/0.
:- dynamic user:wam_cpp_first/2.
:- dynamic user:wam_cpp_lst/1.
:- dynamic user:wam_cpp_add1/2.
:- dynamic user:wam_cpp_gt/2.
:- dynamic user:wam_cpp_test_arith/0.
:- dynamic user:wam_cpp_test_eq/0.
:- dynamic user:wam_cpp_test_neq/0.
:- dynamic user:wam_cpp_test_abs_neg/0.
:- dynamic user:wam_cpp_test_abs_float/0.
:- dynamic user:wam_cpp_test_sign_neg/0.
:- dynamic user:wam_cpp_test_sign_zero/0.
:- dynamic user:wam_cpp_test_sqrt/0.
:- dynamic user:wam_cpp_test_floor_neg/0.
:- dynamic user:wam_cpp_test_ceiling_neg/0.
:- dynamic user:wam_cpp_test_round/0.
:- dynamic user:wam_cpp_test_truncate_neg/0.
:- dynamic user:wam_cpp_test_unary_plus/0.
:- dynamic user:wam_cpp_test_bitnot/0.
:- dynamic user:wam_cpp_test_min_int/0.
:- dynamic user:wam_cpp_test_max_float/0.
:- dynamic user:wam_cpp_test_pow_star/0.
:- dynamic user:wam_cpp_test_pow_caret/0.
:- dynamic user:wam_cpp_test_pow_zero/0.
:- dynamic user:wam_cpp_test_gcd/0.
:- dynamic user:wam_cpp_test_gcd_neg/0.
:- dynamic user:wam_cpp_test_rem_neg/0.
:- dynamic user:wam_cpp_test_bitand/0.
:- dynamic user:wam_cpp_test_bitor/0.
:- dynamic user:wam_cpp_test_bitxor/0.
:- dynamic user:wam_cpp_test_shl/0.
:- dynamic user:wam_cpp_test_shr/0.
:- dynamic user:wam_cpp_test_arith_compose1/0.
:- dynamic user:wam_cpp_test_arith_compose2/0.
:- dynamic user:wam_cpp_test_pi/0.
:- dynamic user:wam_cpp_test_e/0.
:- dynamic user:wam_cpp_test_nan/0.
:- dynamic user:wam_cpp_test_sin_zero/0.
:- dynamic user:wam_cpp_test_cos_pi/0.
:- dynamic user:wam_cpp_test_tan_zero/0.
:- dynamic user:wam_cpp_test_asin_one/0.
:- dynamic user:wam_cpp_test_atan2_diag/0.
:- dynamic user:wam_cpp_test_cosh_zero/0.
:- dynamic user:wam_cpp_test_exp_zero/0.
:- dynamic user:wam_cpp_test_log_e/0.
:- dynamic user:wam_cpp_test_log_base/0.
:- dynamic user:wam_cpp_test_exp_log/0.
:- dynamic user:wam_cpp_test_pythag_one/0.
:- dynamic user:wam_cpp_is_atom/1.
:- dynamic user:wam_cpp_is_int/1.
:- dynamic user:wam_cpp_is_num/1.
:- dynamic user:wam_cpp_is_var/1.
:- dynamic user:wam_cpp_is_compound/1.
:- dynamic user:wam_cpp_test_nonvar/0.
:- dynamic user:wam_cpp_test_functor/0.
:- dynamic user:wam_cpp_test_arg1/0.
:- dynamic user:wam_cpp_test_arg_bad/0.
:- dynamic user:wam_cpp_test_univ_decompose/0.
:- dynamic user:wam_cpp_test_univ_compose/0.
:- dynamic user:wam_cpp_test_functor_atom/0.
:- dynamic user:wam_cpp_test_functor_int/0.
:- dynamic user:wam_cpp_test_functor_build/0.
:- dynamic user:wam_cpp_test_functor_build_atom/0.
:- dynamic user:wam_cpp_test_functor_build_int/0.
:- dynamic user:wam_cpp_test_arg_second/0.
:- dynamic user:wam_cpp_test_arg_outof/0.
:- dynamic user:wam_cpp_test_arg_zero_idx/0.
:- dynamic user:wam_cpp_test_arg_compound_val/0.
:- dynamic user:wam_cpp_test_univ_atom/0.
:- dynamic user:wam_cpp_test_univ_int/0.
:- dynamic user:wam_cpp_test_univ_compose_solo/0.
:- dynamic user:wam_cpp_test_copy_int/0.
:- dynamic user:wam_cpp_test_copy_nested/0.
:- dynamic user:wam_cpp_test_ground_compound/0.
:- dynamic user:wam_cpp_test_ground_unbound/0.
:- dynamic user:wam_cpp_test_ground_partial/0.
:- dynamic user:wam_cpp_test_ground_atom/0.
:- dynamic user:wam_cpp_test_ground_guard/0.
:- dynamic user:wam_cpp_test_ground_guard_not/0.
:- dynamic user:wam_cpp_test_unify/0.
:- dynamic user:wam_cpp_test_unify_fail/0.
:- dynamic user:wam_cpp_test_write/0.
:- dynamic user:wam_cpp_item/1.
:- dynamic user:wam_cpp_num/1.
:- dynamic user:wam_cpp_test_findall/0.
:- dynamic user:wam_cpp_test_findall_empty/0.
:- dynamic user:wam_cpp_test_findall_doubled/0.
:- dynamic user:wam_cpp_test_bagof/0.
:- dynamic user:wam_cpp_test_bagof_empty/0.
:- dynamic user:wam_cpp_test_setof/0.
:- dynamic user:wam_cpp_test_setof_empty/0.
:- dynamic user:wam_cpp_test_count/0.
:- dynamic user:wam_cpp_test_sum/0.
:- dynamic user:wam_cpp_test_min/0.
:- dynamic user:wam_cpp_test_max/0.
:- dynamic user:wam_cpp_test_set/0.
:- dynamic user:wam_cpp_h1/1.
:- dynamic user:wam_cpp_h2/1.
:- dynamic user:wam_cpp_two_helpers/0.
:- dynamic user:wam_cpp_two_helpers_swap/0.
:- dynamic user:wam_cpp_length_acc/3.
:- dynamic user:wam_cpp_list_length/2.
:- dynamic user:wam_cpp_test_len_empty/0.
:- dynamic user:wam_cpp_test_len_one/0.
:- dynamic user:wam_cpp_test_len_three/0.
:- dynamic user:wam_cpp_test_len_five/0.
% List & term builtins (member/2, length/2, copy_term/2):
:- dynamic user:wam_cpp_test_member_yes/0.
:- dynamic user:wam_cpp_test_member_no/0.
:- dynamic user:wam_cpp_test_member_first/0.
:- dynamic user:wam_cpp_test_length_three/0.
:- dynamic user:wam_cpp_test_length_zero/0.
:- dynamic user:wam_cpp_test_length_bad/0.
:- dynamic user:wam_cpp_test_copy_basic/0.
:- dynamic user:wam_cpp_test_copy_atom/0.
:- dynamic user:wam_cpp_test_tv_ground/0.
:- dynamic user:wam_cpp_test_tv_one/0.
:- dynamic user:wam_cpp_test_tv_shared/0.
:- dynamic user:wam_cpp_test_tv_nested/0.
:- dynamic user:wam_cpp_test_nv_shared/0.
:- dynamic user:wam_cpp_test_nv_start10/0.
:- dynamic user:wam_cpp_test_nv_ground/0.
:- dynamic user:wam_cpp_test_variant_yes/0.
:- dynamic user:wam_cpp_test_variant_share/0.
:- dynamic user:wam_cpp_test_variant_shape/0.
:- dynamic user:wam_cpp_test_variant_atom/0.
:- dynamic user:wam_cpp_test_variant_bind/0.
:- dynamic user:wam_cpp_test_variant_neq/0.
:- dynamic user:wam_cpp_test_uni_simple/0.
:- dynamic user:wam_cpp_test_uni_fail/0.
:- dynamic user:wam_cpp_test_uni_two/0.
:- dynamic user:wam_cpp_test_uni_ground/0.
:- dynamic user:wam_cpp_test_nth1_third/0.
:- dynamic user:wam_cpp_test_nth1_zero/0.
:- dynamic user:wam_cpp_test_nth1_outof/0.
:- dynamic user:wam_cpp_test_plus_xy/0.
:- dynamic user:wam_cpp_test_plus_xz/0.
:- dynamic user:wam_cpp_test_plus_yz/0.
:- dynamic user:wam_cpp_test_plus_check_no/0.
:- dynamic user:wam_cpp_test_del_one/0.
:- dynamic user:wam_cpp_test_del_none/0.
:- dynamic user:wam_cpp_test_del_all/0.
:- dynamic user:wam_cpp_test_sub_basic/0.
:- dynamic user:wam_cpp_test_sub_empty/0.
:- dynamic user:wam_cpp_test_sub_all/0.
:- dynamic user:wam_cpp_test_mb_atom_ok/0.
:- dynamic user:wam_cpp_test_mb_int_ok/0.
:- dynamic user:wam_cpp_test_mb_ground_ok/0.
:- dynamic user:wam_cpp_test_mb_callable_ok/0.
:- dynamic user:wam_cpp_test_mb_atom_throw/0.
:- dynamic user:wam_cpp_test_mb_int_throw/0.
:- dynamic user:wam_cpp_test_mb_ground_throw/0.
:- dynamic user:wam_cpp_test_string_chars_fwd/0.
:- dynamic user:wam_cpp_test_string_chars_rev/0.
:- dynamic user:wam_cpp_test_string_codes_fwd/0.
:- dynamic user:wam_cpp_test_string_code_first/0.
:- dynamic user:wam_cpp_test_string_code_last/0.
:- dynamic user:wam_cpp_test_string_code_oob/0.
:- dynamic user:wam_cpp_test_asinh_zero/0.
:- dynamic user:wam_cpp_test_acosh_one/0.
:- dynamic user:wam_cpp_test_atanh_zero/0.
:- dynamic user:wam_cpp_test_asinh_roundtrip/0.
:- dynamic user:wam_cpp_test_copysign_pos/0.
:- dynamic user:wam_cpp_test_copysign_neg/0.
:- dynamic user:wam_cpp_test_copysign_flip/0.
:- dynamic user:wam_cpp_test_popcount_zero/0.
:- dynamic user:wam_cpp_test_popcount_seven/0.
:- dynamic user:wam_cpp_test_popcount_ff/0.
:- dynamic user:wam_cpp_test_lsb_eight/0.
:- dynamic user:wam_cpp_test_lsb_36/0.
:- dynamic user:wam_cpp_test_msb_eight/0.
:- dynamic user:wam_cpp_test_msb_ff/0.
:- dynamic user:wam_cpp_test_dcg_greeting/0.
:- dynamic user:wam_cpp_test_dcg_greeting_no/0.
:- dynamic user:wam_cpp_test_dcg_yelp/0.
:- dynamic user:wam_cpp_test_dcg_optional_empty/0.
:- dynamic user:wam_cpp_test_dcg_optional_maybe/0.
:- dynamic user:wam_cpp_test_dcg_loop_empty/0.
:- dynamic user:wam_cpp_test_dcg_loop_three/0.
:- dynamic user:wam_cpp_test_dcg_phrase3/0.
:- dynamic user:wam_cpp_test_phrase_mq/0.
:- dynamic user:wam_cpp_test_call_mq/0.
:- dynamic user:wam_cpp_test_random_range/0.
:- dynamic user:wam_cpp_test_rb_in_range/0.
:- dynamic user:wam_cpp_test_rb_seeded_eq/0.
:- dynamic user:wam_cpp_test_rb_low_eq_high/0.
:- dynamic user:wam_cpp_test_rb_low_gt_high/0.
:- dynamic user:wam_cpp_test_rm_in_list/0.
:- dynamic user:wam_cpp_test_rm_empty/0.
:- dynamic user:wam_cpp_test_rp_length/0.
:- dynamic user:wam_cpp_test_rp_invariant/0.
:- dynamic user:wam_cpp_test_rp_empty/0.
:- dynamic user:wam_cpp_test_sort4_asc_dedup/0.
:- dynamic user:wam_cpp_test_sort4_asc_keep/0.
:- dynamic user:wam_cpp_test_sort4_desc_dedup/0.
:- dynamic user:wam_cpp_test_sort4_desc_keep/0.
:- dynamic user:wam_cpp_test_sort4_by_first/0.
:- dynamic user:wam_cpp_test_sort4_empty/0.
:- dynamic user:wam_cpp_test_predsort_asc/0.
:- dynamic user:wam_cpp_test_predsort_desc/0.
:- dynamic user:wam_cpp_test_predsort_by_key/0.
:- dynamic user:wam_cpp_test_predsort_empty/0.
:- dynamic user:wam_cpp_test_predsort_single/0.
:- dynamic user:wam_cpp_test_assertion_ok/0.
:- dynamic user:wam_cpp_test_assertion_fail/0.
:- dynamic user:wam_cpp_test_assertion_expr/0.
:- dynamic user:wam_cpp_test_assoc_empty/0.
:- dynamic user:wam_cpp_test_assoc_put_get/0.
:- dynamic user:wam_cpp_test_assoc_missing/0.
:- dynamic user:wam_cpp_test_assoc_overwrite/0.
:- dynamic user:wam_cpp_test_assoc_list_sorted/0.
:- dynamic user:wam_cpp_test_assoc_keys/0.
:- dynamic user:wam_cpp_test_assoc_values/0.
:- dynamic user:wam_cpp_test_assoc_compound_keys/0.
:- dynamic user:wam_cpp_assoc_depth/2.
:- dynamic user:wam_cpp_test_assoc_avl_balance_sorted/0.
:- dynamic user:wam_cpp_test_assoc_avl_balance_descending/0.
:- dynamic user:wam_cpp_test_assoc_avl_balance_zigzag/0.
:- dynamic user:wam_cpp_test_assoc_min/0.
:- dynamic user:wam_cpp_test_assoc_max/0.
:- dynamic user:wam_cpp_test_assoc_min_after_inserts/0.
:- dynamic user:wam_cpp_test_assoc_del_leaf/0.
:- dynamic user:wam_cpp_test_assoc_del_root/0.
:- dynamic user:wam_cpp_test_assoc_del_missing_fails/0.
:- dynamic user:wam_cpp_test_assoc_del_all_back_to_empty/0.
:- dynamic user:wam_cpp_test_assoc_del_rebalance_sorted/0.
:- dynamic user:wam_cpp_test_assoc_del_returns_value/0.
:- dynamic user:wam_cpp_test_assoc_del_min/0.
:- dynamic user:wam_cpp_test_assoc_del_max/0.
:- dynamic user:wam_cpp_test_assoc_del_min_empty_fails/0.
:- dynamic user:wam_cpp_test_assoc_del_max_empty_fails/0.
:- dynamic user:wam_cpp_test_assoc_pq_extract_min/0.
:- dynamic user:wam_cpp_test_assoc_get5_replace/0.
:- dynamic user:wam_cpp_test_assoc_get5_missing/0.
:- dynamic user:wam_cpp_test_assoc_get5_threads_old_value/0.
:- dynamic user:wam_cpp_assoc_double/2.
:- dynamic user:wam_cpp_test_assoc_map/0.
:- dynamic user:wam_cpp_test_assoc_map_empty/0.
:- dynamic user:wam_cpp_test_get_time_positive/0.
:- dynamic user:wam_cpp_test_stamp_utc/0.
:- dynamic user:wam_cpp_test_stamp_subsec/0.
:- dynamic user:wam_cpp_test_dt_roundtrip/0.
:- dynamic user:wam_cpp_test_date6_to_stamp/0.
:- dynamic user:wam_cpp_test_format_basic/0.
:- dynamic user:wam_cpp_test_format_year/0.
:- dynamic user:wam_cpp_test_format_from_date9/0.
:- dynamic user:wam_cpp_test_dcgcall_n0/0.
:- dynamic user:wam_cpp_test_dcgcall_n0_no/0.
:- dynamic user:wam_cpp_test_dcgcall_full/0.
:- dynamic user:wam_cpp_test_dcgcall_partial/0.
:- dynamic user:wam_cpp_test_dcgcall_partial_no/0.
:- dynamic user:wam_cpp_test_dcgcall_rep3/0.
:- dynamic user:wam_cpp_test_dcgcall_rep0/0.
:- dynamic user:wam_cpp_test_dcgcall_rep_no/0.
:- dynamic user:wam_cpp_test_pairs_keys/0.
:- dynamic user:wam_cpp_test_pairs_values/0.
:- dynamic user:wam_cpp_test_pairs_kv/0.
:- dynamic user:wam_cpp_test_take_some/0.
:- dynamic user:wam_cpp_test_take_overlong/0.
:- dynamic user:wam_cpp_test_take_zero/0.
:- dynamic user:wam_cpp_test_drop_some/0.
:- dynamic user:wam_cpp_test_drop_overlong/0.
:- dynamic user:wam_cpp_test_drop_zero/0.
:- dynamic user:wam_cpp_test_intersection/0.
:- dynamic user:wam_cpp_test_intersection_empty/0.
:- dynamic user:wam_cpp_test_union/0.
:- dynamic user:wam_cpp_test_permutation_check/0.
:- dynamic user:wam_cpp_test_permutation_no/0.
:- dynamic user:wam_cpp_test_idx_flat/0.
:- dynamic user:wam_cpp_test_idx_keep/0.
:- dynamic user:wam_cpp_test_idx_drop_first/0.
:- dynamic user:wam_cpp_test_idx_keep_first/0.
:- dynamic user:wam_cpp_test_idx_mixed/0.
:- dynamic user:wam_cpp_test_stream_roundtrip/0.
:- dynamic user:wam_cpp_test_stream_read_string/0.
:- dynamic user:wam_cpp_test_stream_append/0.
:- dynamic user:wam_cpp_test_stream_at_end/0.
:- dynamic user:wam_cpp_test_stream_missing_file/0.
:- dynamic user:wam_cpp_test_stream_close_unknown/0.
:- dynamic user:wam_cpp_test_intro_static_yes/0.
:- dynamic user:wam_cpp_test_intro_static_no/0.
:- dynamic user:wam_cpp_test_intro_arity_no/0.
:- dynamic user:wam_cpp_test_intro_builtin_yes/0.
:- dynamic user:wam_cpp_test_intro_prop_defined/0.
:- dynamic user:wam_cpp_test_intro_prop_static/0.
:- dynamic user:wam_cpp_test_intro_prop_dynamic/0.
:- dynamic user:wam_cpp_test_intro_prop_static_not_dynamic/0.
:- dynamic user:wam_cpp_test_intro_prop_count/0.
:- dynamic user:wam_cpp_test_intro_inst_throw/0.
:- dynamic user:wam_cpp_test_intro_indicator_throw/0.
:- dynamic user:wam_cpp_test_fs_exists_file_yes/0.
:- dynamic user:wam_cpp_test_fs_exists_file_no/0.
:- dynamic user:wam_cpp_test_fs_exists_dir_yes/0.
:- dynamic user:wam_cpp_test_fs_exists_dir_no/0.
:- dynamic user:wam_cpp_test_fs_make_check/0.
:- dynamic user:wam_cpp_test_fs_dir_files/0.
:- dynamic user:wam_cpp_test_fs_delete/0.
:- dynamic user:wam_cpp_test_fs_delete_missing/0.
:- dynamic user:wam_cpp_test_wos_stream/0.
:- dynamic user:wam_cpp_test_wos_format/0.
:- dynamic user:wam_cpp_test_clause_fact/0.
:- dynamic user:wam_cpp_test_clause_var/0.
:- dynamic user:wam_cpp_test_clause_enum/0.
:- dynamic user:wam_cpp_test_clause_rule_body/0.
:- dynamic user:wam_cpp_test_clause_missing/0.
:- dynamic user:wam_cpp_test_clause_unknown/0.
:- dynamic user:wam_cpp_test_clause_after_assertz/0.
:- dynamic user:wam_cpp_test_nested_dash/0.
:- dynamic user:wam_cpp_test_nested_f_g/0.
:- dynamic user:wam_cpp_test_nested_right/0.
:- dynamic user:wam_cpp_test_nested_constant/0.
:- dynamic user:wam_cpp_test_cp_check_static/0.
:- dynamic user:wam_cpp_test_cp_check_missing/0.
:- dynamic user:wam_cpp_test_cp_enum_arity/0.
:- dynamic user:wam_cpp_test_cp_enum_name_by_arity/0.
:- dynamic user:wam_cpp_test_cp_enum_all/0.
:- dynamic user:wam_cpp_test_cp_enum_none/0.
:- dynamic user:wam_cpp_test_cp_inst_throw/0.
:- dynamic user:wam_cpp_test_cp_indicator_throw/0.
:- dynamic user:wam_cpp_test_enum_member/0.

user:wam_cpp_test_member_yes   :- member(b, [a, b, c]).
user:wam_cpp_test_member_no    :- member(z, [a, b, c]).
user:wam_cpp_test_member_first :- member(a, [a, b, c]).
user:wam_cpp_test_length_three :- length([a, b, c], 3).
user:wam_cpp_test_length_zero  :- length([], 0).
user:wam_cpp_test_length_bad   :- length([a, b, c], 5).
user:wam_cpp_test_copy_basic   :- copy_term(foo(X, X, _Y), T), T = foo(A, A, _B).
user:wam_cpp_test_copy_atom    :- copy_term(hello, T), T = hello.
% term_variables/2: collect free vars in left-to-right first-occurrence
% order. Each shared variable appears once.
user:wam_cpp_test_tv_ground    :- term_variables(foo(1, 2, 3), L), L = [].
user:wam_cpp_test_tv_one       :- term_variables(foo(_X, 1), L), L = [_].
user:wam_cpp_test_tv_shared    :- term_variables(foo(X, _Y, X), L),
                                  L = [X, _], var(X).
user:wam_cpp_test_tv_nested    :- term_variables(p(q(_X, _Y), r(_Y, _Z)), L),
                                  length(L, 3).
% numbervars/3: bind free vars to $VAR(N) starting at Start.
user:wam_cpp_test_nv_shared    :-
    T = foo(X, Y, X),
    numbervars(T, 0, End),
    End = 2,
    T = foo('$VAR'(0), '$VAR'(1), '$VAR'(0)).
user:wam_cpp_test_nv_start10   :-
    T = pair(_, _),
    numbervars(T, 10, End),
    End = 12,
    T = pair('$VAR'(10), '$VAR'(11)).
user:wam_cpp_test_nv_ground    :- numbervars(hello, 0, 0).
% =@=/2 and \=@=/2: variant equivalence.
user:wam_cpp_test_variant_yes  :- foo(_, _, _) =@= foo(_, _, _).
user:wam_cpp_test_variant_share :- foo(X, _, X) =@= foo(A, _, A).
user:wam_cpp_test_variant_shape :- \+ (foo(_, _) =@= foo(_, _, _)).
user:wam_cpp_test_variant_atom :- \+ (foo(_) =@= bar(_)).
user:wam_cpp_test_variant_bind :- \+ (foo(a, X) =@= foo(X, a)).
user:wam_cpp_test_variant_neq  :- foo(_, _) \=@= foo(X, X).
% unifiable/3: non-binding unification + bindings list.
user:wam_cpp_test_uni_simple   :-
    unifiable(X, foo(a), B),
    B = [X = foo(a)],
    var(X).
user:wam_cpp_test_uni_fail     :- \+ unifiable(foo(a), foo(b), _).
user:wam_cpp_test_uni_two      :-
    unifiable(p(X, Y), p(1, 2), B),
    B = [X=1, Y=2],
    var(X), var(Y).
user:wam_cpp_test_uni_ground   :- unifiable(hello, hello, []).
% nth1/3 (1-indexed list access), plus/3 (bidirectional integer add),
% delete/3 (remove all == matches), subtract/3 (set difference),
% must_be/2 (type check that throws).
user:wam_cpp_test_nth1_third   :- nth1(3, [a, b, c], X), X = c.
user:wam_cpp_test_nth1_zero    :- \+ nth1(0, [a, b, c], _).
user:wam_cpp_test_nth1_outof   :- \+ nth1(5, [a, b, c], _).
user:wam_cpp_test_plus_xy      :- plus(2, 3, Z), Z = 5.
user:wam_cpp_test_plus_xz      :- plus(2, Y, 5), Y = 3.
user:wam_cpp_test_plus_yz      :- plus(X, 3, 5), X = 2.
user:wam_cpp_test_plus_check_no :- \+ plus(2, 3, 6).
user:wam_cpp_test_del_one      :- delete([a, b, c, b], b, R), R = [a, c].
user:wam_cpp_test_del_none     :- delete([a, b, c], z, R), R = [a, b, c].
user:wam_cpp_test_del_all      :- delete([a, a, a], a, R), R = [].
user:wam_cpp_test_sub_basic    :- subtract([1, 2, 3, 4], [2, 4], R), R = [1, 3].
user:wam_cpp_test_sub_empty    :- subtract([], [a, b], R), R = [].
user:wam_cpp_test_sub_all      :- subtract([1, 2, 3], [1, 2, 3], R), R = [].
user:wam_cpp_test_mb_atom_ok   :- must_be(atom, hello).
user:wam_cpp_test_mb_int_ok    :- must_be(integer, 42).
user:wam_cpp_test_mb_ground_ok :- must_be(ground, foo(1, 2)).
user:wam_cpp_test_mb_callable_ok :- must_be(callable, foo(1)).
user:wam_cpp_test_mb_atom_throw :-
    catch(must_be(atom, 5),
          error(type_error(atom, 5), _),
          true).
user:wam_cpp_test_mb_int_throw :-
    catch(must_be(integer, hello),
          error(type_error(integer, hello), _),
          true).
user:wam_cpp_test_mb_ground_throw :-
    catch(must_be(ground, foo(_, 2)),
          error(instantiation_error, _),
          true).
% String variant aliases: string_chars/string_codes route through the
% atom_chars/atom_codes path. string_code/3 is 1-based char-code access.
user:wam_cpp_test_string_chars_fwd  :- string_chars(hello, [h, e, l, l, o]).
user:wam_cpp_test_string_chars_rev  :- string_chars(A, [h, i]), A = hi.
user:wam_cpp_test_string_codes_fwd  :- string_codes(abc, [0'a, 0'b, 0'c]).
user:wam_cpp_test_string_code_first :- string_code(1, hello, 0'h).
user:wam_cpp_test_string_code_last  :- string_code(5, hello, 0'o).
user:wam_cpp_test_string_code_oob   :- \+ string_code(6, hello, _).
% Hyperbolic inverses + round-trip identities.
user:wam_cpp_test_asinh_zero        :- X is asinh(0), X =:= 0.0.
user:wam_cpp_test_acosh_one         :- X is acosh(1), X =:= 0.0.
user:wam_cpp_test_atanh_zero        :- X is atanh(0), X =:= 0.0.
user:wam_cpp_test_asinh_roundtrip   :- X is asinh(sinh(2)), X > 1.999, X < 2.001.
% copysign/2: magnitude of X with sign of Y; always Float.
user:wam_cpp_test_copysign_pos      :- X is copysign(5, 1), X = 5.0.
user:wam_cpp_test_copysign_neg      :- X is copysign(5, -3), X = -5.0.
user:wam_cpp_test_copysign_flip     :- X is copysign(-5, 3), X = 5.0.
% Integer bit counts: popcount, lsb, msb.
user:wam_cpp_test_popcount_zero     :- X is popcount(0), X = 0.
user:wam_cpp_test_popcount_seven    :- X is popcount(7), X = 3.
user:wam_cpp_test_popcount_ff       :- X is popcount(255), X = 8.
user:wam_cpp_test_lsb_eight         :- X is lsb(8), X = 3.
user:wam_cpp_test_lsb_36            :- X is lsb(36), X = 2.
user:wam_cpp_test_msb_eight         :- X is msb(8), X = 3.
user:wam_cpp_test_msb_ff            :- X is msb(255), X = 7.
% DCG support. SWI's term_expansion at load time rewrites these `-->`
% rules into normal Prolog clauses with two extra difference-list args
% (e.g. dcg_greeting/2). The phrase/2 and phrase/3 builtins then
% dispatch by appending [List, Rest] to the goal's args.
user:dcg_greeting --> [hello], [world].
user:dcg_yelp(X)  --> [shout, X].
user:dcg_optional --> [].
user:dcg_optional --> [maybe].
user:dcg_loop([])    --> [].
user:dcg_loop([X|T]) --> [X], dcg_loop(T).
user:wam_cpp_test_dcg_greeting    :- phrase(dcg_greeting, [hello, world]).
user:wam_cpp_test_dcg_greeting_no :- \+ phrase(dcg_greeting, [hello]).
user:wam_cpp_test_dcg_yelp        :- phrase(dcg_yelp(loud), [shout, loud]).
user:wam_cpp_test_dcg_optional_empty :- phrase(dcg_optional, []).
user:wam_cpp_test_dcg_optional_maybe :- phrase(dcg_optional, [maybe]).
user:wam_cpp_test_dcg_loop_empty  :- phrase(dcg_loop(L), []), L = [].
user:wam_cpp_test_dcg_loop_three  :- phrase(dcg_loop(L), [a, b, c]),
                                     L = [a, b, c].
user:wam_cpp_test_dcg_phrase3     :- phrase(dcg_greeting,
                                           [hello, world, extra], R),
                                     R = [extra].
% Module-qualified meta-call: phrase(user:Goal, ...) and call(user:Goal, ...).
% The dispatcher strips a leading Module: from the goal cell.
user:wam_cpp_test_phrase_mq       :- phrase(user:dcg_greeting, [hello, world]).
user:wam_cpp_test_call_mq         :-
    G = user:atom_concat(a, b),
    call(G, R),
    R = ab.
% Random library: random/1, random_between/3, random_member/2,
% random_permutation/2, set_random/1. Tests use set_random(seed(N)) for
% repeatability where reproducibility matters; others just check
% constraints (range, membership, length, permutation invariant).
user:wam_cpp_test_random_range    :- random(X), X >= 0.0, X < 1.0.
user:wam_cpp_test_rb_in_range     :-
    set_random(seed(42)),
    random_between(1, 10, X),
    X >= 1, X =< 10.
user:wam_cpp_test_rb_seeded_eq    :-
    set_random(seed(42)),
    random_between(1, 1000000, X),
    set_random(seed(42)),
    random_between(1, 1000000, Y),
    X = Y.
user:wam_cpp_test_rb_low_eq_high  :- random_between(5, 5, X), X = 5.
user:wam_cpp_test_rb_low_gt_high  :- \+ random_between(10, 5, _).
user:wam_cpp_test_rm_in_list      :-
    random_member(X, [a, b, c, d]),
    member(X, [a, b, c, d]).
user:wam_cpp_test_rm_empty        :- \+ random_member(_, []).
user:wam_cpp_test_rp_length       :-
    set_random(seed(1)),
    random_permutation([1, 2, 3, 4, 5], P),
    length(P, 5).
user:wam_cpp_test_rp_invariant    :-
    set_random(seed(1)),
    random_permutation([a, b, c], P),
    msort(P, [a, b, c]).
user:wam_cpp_test_rp_empty        :- random_permutation([], []).
% sort/4 (key + order options) and predsort/3 (custom comparator).
% sort/4 is a C++ builtin; predsort/3 lives as user-module Prolog
% asserted at module load (with wam_cpp_predsort_/5, wam_cpp_sort2/4,
% wam_cpp_predmerge/4, wam_cpp_predmerge_/7 helpers).
user:dcg_cmp_int(O, A, B) :- compare(O, A, B).
user:dcg_cmp_int_desc(O1, A, B) :-
    compare(O0, A, B),
    ( O0 == < -> O1 = >
    ; O0 == > -> O1 = <
    ; O1 = O0
    ).
user:dcg_cmp_by_first(O, A-_, B-_) :- compare(O, A, B).
user:wam_cpp_test_sort4_asc_dedup  :- sort(0, @<, [3, 1, 2, 1, 3], L),
                                      L = [1, 2, 3].
user:wam_cpp_test_sort4_asc_keep   :- sort(0, @=<, [3, 1, 2, 1, 3], L),
                                      L = [1, 1, 2, 3, 3].
user:wam_cpp_test_sort4_desc_dedup :- sort(0, @>, [3, 1, 2, 1, 3], L),
                                      L = [3, 2, 1].
user:wam_cpp_test_sort4_desc_keep  :- sort(0, @>=, [3, 1, 2, 1, 3], L),
                                      L = [3, 3, 2, 1, 1].
user:wam_cpp_test_sort4_by_first   :-
    sort(1, @<, [pair(b, 2), pair(a, 1), pair(c, 3), pair(a, 9)], L),
    L = [pair(a, 1), pair(b, 2), pair(c, 3)].
user:wam_cpp_test_sort4_empty      :- sort(0, @<, [], []).
user:wam_cpp_test_predsort_asc     :-
    predsort(dcg_cmp_int, [3, 1, 2, 1, 3], L), L = [1, 2, 3].
user:wam_cpp_test_predsort_desc    :-
    predsort(dcg_cmp_int_desc, [1, 3, 2, 5, 4], L), L = [5, 4, 3, 2, 1].
user:wam_cpp_test_predsort_by_key  :-
    predsort(dcg_cmp_by_first, [b-2, a-1, c-3, a-9], L),
    L = [a-1, b-2, c-3].
user:wam_cpp_test_predsort_empty   :- predsort(dcg_cmp_int, [], []).
user:wam_cpp_test_predsort_single  :- predsort(dcg_cmp_int, [42], [42]).
% assertion/1 (wam_cpp_target auto-asserts the user-module clause):
%   assertion(true) succeeds silently; assertion(false) / fail throws
%   error(assertion_failed, Goal). The include_stdlib option (also new)
%   makes write_wam_cpp_project auto-prepend the helpers.
user:wam_cpp_test_assertion_ok      :- assertion(true), assertion(1 =:= 1).
user:wam_cpp_test_assertion_fail    :-
    catch(assertion(fail),
          error(assertion_failed, fail),
          true).
user:wam_cpp_test_assertion_expr    :-
    catch(assertion(1 =:= 2),
          error(assertion_failed, _),
          true).
% assoc library (BST keyed by standard order, asserted as user-module
% Prolog and exposed via the stdlib `assoc` feature):
user:wam_cpp_test_assoc_empty         :- empty_assoc(t).
user:wam_cpp_test_assoc_put_get       :-
    empty_assoc(A0),
    put_assoc(name, A0, alice, A1),
    put_assoc(age, A1, 30, A2),
    get_assoc(name, A2, alice),
    get_assoc(age, A2, 30).
user:wam_cpp_test_assoc_missing       :-
    empty_assoc(A0),
    put_assoc(x, A0, 1, A1),
    \+ get_assoc(missing, A1, _).
user:wam_cpp_test_assoc_overwrite     :-
    empty_assoc(A0),
    put_assoc(k, A0, 1, A1),
    put_assoc(k, A1, 2, A2),
    get_assoc(k, A2, 2).
user:wam_cpp_test_assoc_list_sorted   :-
    list_to_assoc([c-3, a-1, b-2], A),
    assoc_to_list(A, [a-1, b-2, c-3]).
user:wam_cpp_test_assoc_keys          :-
    list_to_assoc([c-3, a-1, b-2], A),
    assoc_to_keys(A, [a, b, c]).
user:wam_cpp_test_assoc_values        :-
    list_to_assoc([c-3, a-1, b-2], A),
    assoc_to_values(A, [1, 2, 3]).
user:wam_cpp_test_assoc_compound_keys :-
    empty_assoc(A0),
    put_assoc(pt(1, 2), A0, north, A1),
    put_assoc(pt(3, 4), A1, south, A2),
    get_assoc(pt(1, 2), A2, north),
    get_assoc(pt(3, 4), A2, south).
% AVL balance: insert keys in ascending order — a plain BST would be
% a 16-deep chain, but the AVL must stay roughly log2(16) ≈ 4-5 deep.
% Tree-depth probe uses the t(K,V,B,L,R) node shape.
user:wam_cpp_assoc_depth(t, 0).
user:wam_cpp_assoc_depth(t(_, _, _, L, R), D) :-
    wam_cpp_assoc_depth(L, DL),
    wam_cpp_assoc_depth(R, DR),
    ( DL >= DR -> D is DL + 1 ; D is DR + 1 ).
user:wam_cpp_test_assoc_avl_balance_sorted :-
    empty_assoc(A0),
    put_assoc(1,  A0,  v1,  A1),
    put_assoc(2,  A1,  v2,  A2),
    put_assoc(3,  A2,  v3,  A3),
    put_assoc(4,  A3,  v4,  A4),
    put_assoc(5,  A4,  v5,  A5),
    put_assoc(6,  A5,  v6,  A6),
    put_assoc(7,  A6,  v7,  A7),
    put_assoc(8,  A7,  v8,  A8),
    put_assoc(9,  A8,  v9,  A9),
    put_assoc(10, A9,  v10, A10),
    put_assoc(11, A10, v11, A11),
    put_assoc(12, A11, v12, A12),
    put_assoc(13, A12, v13, A13),
    put_assoc(14, A13, v14, A14),
    put_assoc(15, A14, v15, A15),
    put_assoc(16, A15, v16, A16),
    wam_cpp_assoc_depth(A16, D),
    D =< 6,
    get_assoc(1,  A16, v1),
    get_assoc(8,  A16, v8),
    get_assoc(16, A16, v16),
    \+ get_assoc(0,  A16, _),
    \+ get_assoc(17, A16, _).
user:wam_cpp_test_assoc_avl_balance_descending :-
    % Mirror case — keys descending forces left-side rotations.
    empty_assoc(A0),
    put_assoc(16, A0,  v16, A1),
    put_assoc(15, A1,  v15, A2),
    put_assoc(14, A2,  v14, A3),
    put_assoc(13, A3,  v13, A4),
    put_assoc(12, A4,  v12, A5),
    put_assoc(11, A5,  v11, A6),
    put_assoc(10, A6,  v10, A7),
    put_assoc(9,  A7,  v9,  A8),
    put_assoc(8,  A8,  v8,  A9),
    put_assoc(7,  A9,  v7,  A10),
    put_assoc(6,  A10, v6,  A11),
    put_assoc(5,  A11, v5,  A12),
    put_assoc(4,  A12, v4,  A13),
    put_assoc(3,  A13, v3,  A14),
    put_assoc(2,  A14, v2,  A15),
    put_assoc(1,  A15, v1,  A16),
    wam_cpp_assoc_depth(A16, D),
    D =< 6,
    get_assoc(1,  A16, v1),
    get_assoc(16, A16, v16).
user:wam_cpp_test_assoc_avl_balance_zigzag :-
    % Insert in zigzag order — exercises both LR and RL rotations.
    empty_assoc(A0),
    put_assoc(8,  A0,  v8,  A1),
    put_assoc(4,  A1,  v4,  A2),
    put_assoc(12, A2,  v12, A3),
    put_assoc(6,  A3,  v6,  A4),
    put_assoc(10, A4,  v10, A5),
    put_assoc(2,  A5,  v2,  A6),
    put_assoc(14, A6,  v14, A7),
    put_assoc(5,  A7,  v5,  A8),
    put_assoc(7,  A8,  v7,  A9),
    put_assoc(9,  A9,  v9,  A10),
    put_assoc(11, A10, v11, A11),
    put_assoc(1,  A11, v1,  A12),
    put_assoc(3,  A12, v3,  A13),
    put_assoc(13, A13, v13, A14),
    put_assoc(15, A14, v15, A15),
    wam_cpp_assoc_depth(A15, D),
    D =< 5,
    get_assoc(7,  A15, v7),
    get_assoc(15, A15, v15),
    assoc_to_keys(A15, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).

% min_assoc / max_assoc / del_assoc — round out the assoc API.
user:wam_cpp_test_assoc_min :-
    empty_assoc(A0),
    put_assoc(5, A0, v5, A1),
    put_assoc(1, A1, v1, A2),
    put_assoc(9, A2, v9, A3),
    min_assoc(A3, 1, v1).
user:wam_cpp_test_assoc_max :-
    empty_assoc(A0),
    put_assoc(5, A0, v5, A1),
    put_assoc(1, A1, v1, A2),
    put_assoc(9, A2, v9, A3),
    max_assoc(A3, 9, v9).
user:wam_cpp_test_assoc_min_after_inserts :-
    % Inserting in zigzag order — min stays the leftmost regardless
    % of rotations.
    empty_assoc(A0),
    put_assoc(8,  A0, v8,  A1),
    put_assoc(4,  A1, v4,  A2),
    put_assoc(12, A2, v12, A3),
    put_assoc(2,  A3, v2,  A4),
    put_assoc(10, A4, v10, A5),
    put_assoc(1,  A5, v1,  A6),
    min_assoc(A6, 1, v1),
    max_assoc(A6, 12, v12).
user:wam_cpp_test_assoc_del_leaf :-
    % Build a tree with a leaf, delete it.
    empty_assoc(A0),
    put_assoc(2, A0, v2, A1),
    put_assoc(1, A1, v1, A2),
    put_assoc(3, A2, v3, A3),
    del_assoc(1, A3, v1, A4),
    \+ get_assoc(1, A4, _),
    get_assoc(2, A4, v2),
    get_assoc(3, A4, v3).
user:wam_cpp_test_assoc_del_root :-
    % Delete the root — exercises the "both children non-empty" path
    % that pulls the in-order successor.
    empty_assoc(A0),
    put_assoc(2, A0, v2, A1),
    put_assoc(1, A1, v1, A2),
    put_assoc(3, A2, v3, A3),
    del_assoc(2, A3, v2, A4),
    \+ get_assoc(2, A4, _),
    get_assoc(1, A4, v1),
    get_assoc(3, A4, v3).
user:wam_cpp_test_assoc_del_missing_fails :-
    empty_assoc(A0),
    put_assoc(a, A0, 1, A1),
    \+ del_assoc(z, A1, _, _).
user:wam_cpp_test_assoc_del_all_back_to_empty :-
    empty_assoc(A0),
    put_assoc(1, A0, v1, A1),
    put_assoc(2, A1, v2, A2),
    put_assoc(3, A2, v3, A3),
    put_assoc(4, A3, v4, A4),
    put_assoc(5, A4, v5, A5),
    del_assoc(3, A5, _, B1),
    del_assoc(1, B1, _, B2),
    del_assoc(5, B2, _, B3),
    del_assoc(2, B3, _, B4),
    del_assoc(4, B4, _, B5),
    B5 == t.
user:wam_cpp_test_assoc_del_rebalance_sorted :-
    % Insert 16 keys ascending (max-imbalance shape), then delete
    % the first half. Tree must stay balanced (depth bounded) and
    % retain the remaining keys.
    empty_assoc(A0),
    put_assoc(1,  A0,  _, A1),  put_assoc(2,  A1,  _, A2),
    put_assoc(3,  A2,  _, A3),  put_assoc(4,  A3,  _, A4),
    put_assoc(5,  A4,  _, A5),  put_assoc(6,  A5,  _, A6),
    put_assoc(7,  A6,  _, A7),  put_assoc(8,  A7,  _, A8),
    put_assoc(9,  A8,  _, A9),  put_assoc(10, A9,  _, A10),
    put_assoc(11, A10, _, A11), put_assoc(12, A11, _, A12),
    put_assoc(13, A12, _, A13), put_assoc(14, A13, _, A14),
    put_assoc(15, A14, _, A15), put_assoc(16, A15, _, A16),
    del_assoc(1, A16, _, B1),
    del_assoc(2, B1,  _, B2),
    del_assoc(3, B2,  _, B3),
    del_assoc(4, B3,  _, B4),
    del_assoc(5, B4,  _, B5),
    del_assoc(6, B5,  _, B6),
    del_assoc(7, B6,  _, B7),
    del_assoc(8, B7,  _, B8),
    wam_cpp_assoc_depth(B8, D),
    D =< 4,
    assoc_to_keys(B8, [9, 10, 11, 12, 13, 14, 15, 16]).
user:wam_cpp_test_assoc_del_returns_value :-
    % del_assoc/4 returns the deleted value as its 3rd arg.
    empty_assoc(A0),
    put_assoc(name,   A0, alice,  A1),
    put_assoc(age,    A1, 30,     A2),
    put_assoc(role,   A2, admin,  A3),
    del_assoc(age, A3, V, _),
    V = 30.

% del_min_assoc / del_max_assoc — extract the leftmost / rightmost
% (K, V) pair atomically and return the rebalanced tree.
user:wam_cpp_test_assoc_del_min :-
    empty_assoc(A0),
    put_assoc(3, A0, c, A1), put_assoc(1, A1, a, A2),
    put_assoc(5, A2, e, A3), put_assoc(2, A3, b, A4),
    del_min_assoc(A4, MK, MV, A5),
    MK = 1, MV = a,
    assoc_to_keys(A5, [2, 3, 5]).
user:wam_cpp_test_assoc_del_max :-
    empty_assoc(A0),
    put_assoc(3, A0, c, A1), put_assoc(1, A1, a, A2),
    put_assoc(5, A2, e, A3), put_assoc(2, A3, b, A4),
    del_max_assoc(A4, MK, MV, A5),
    MK = 5, MV = e,
    assoc_to_keys(A5, [1, 2, 3]).
user:wam_cpp_test_assoc_del_min_empty_fails :-
    empty_assoc(E), \+ del_min_assoc(E, _, _, _).
user:wam_cpp_test_assoc_del_max_empty_fails :-
    empty_assoc(E), \+ del_max_assoc(E, _, _, _).
user:wam_cpp_test_assoc_pq_extract_min :-
    % Priority-queue use: repeatedly extract the min. Ends in
    % the empty atom `t`.
    empty_assoc(A0),
    put_assoc(5, A0, e, A1), put_assoc(1, A1, a, A2),
    put_assoc(3, A2, c, A3), put_assoc(2, A3, b, A4),
    put_assoc(4, A4, d, A5),
    del_min_assoc(A5, 1, a, B1),
    del_min_assoc(B1, 2, b, B2),
    del_min_assoc(B2, 3, c, B3),
    del_min_assoc(B3, 4, d, B4),
    del_min_assoc(B4, 5, e, B5),
    B5 == t.

% get_assoc/5 — atomic test-and-set. Walks the tree once, binding
% the current value and producing a tree with the slot replaced.
user:wam_cpp_test_assoc_get5_replace :-
    empty_assoc(A0),
    put_assoc(a, A0, 1, A1),
    put_assoc(b, A1, 2, A2),
    put_assoc(c, A2, 3, A3),
    get_assoc(b, A3, 2, A4, 99),
    get_assoc(b, A4, 99),
    get_assoc(a, A4, 1),
    get_assoc(c, A4, 3).
user:wam_cpp_test_assoc_get5_missing :-
    empty_assoc(A0),
    put_assoc(a, A0, 1, A1),
    \+ get_assoc(z, A1, _, _, _).
user:wam_cpp_test_assoc_get5_threads_old_value :-
    % The 3rd-arg binding lets callers compute NewVal from OldVal in
    % a single pass.
    empty_assoc(A0),
    put_assoc(counter, A0, 5, A1),
    get_assoc(counter, A1, Old, A2, New),
    New is Old + 1,
    get_assoc(counter, A2, 6).

% map_assoc/3 — apply call(Goal, OldVal, NewVal) to every value.
user:wam_cpp_assoc_double(X, Y) :- Y is X * 2.
user:wam_cpp_test_assoc_map :-
    empty_assoc(A0),
    put_assoc(a, A0, 1, A1),
    put_assoc(b, A1, 2, A2),
    put_assoc(c, A2, 3, A3),
    put_assoc(d, A3, 4, A4),
    map_assoc(wam_cpp_assoc_double, A4, A5),
    get_assoc(a, A5, 2),
    get_assoc(b, A5, 4),
    get_assoc(c, A5, 6),
    get_assoc(d, A5, 8),
    assoc_to_keys(A5, [a, b, c, d]).
user:wam_cpp_test_assoc_map_empty :-
    empty_assoc(E),
    map_assoc(wam_cpp_assoc_double, E, R),
    R == t.
% Date/time: get_time/1 (Float seconds since epoch),
% stamp_date_time/3 (decompose + TZ), date_time_stamp/2 (compose),
% format_time/3 (strftime-style atom output). Tests use the canonical
% 2024-01-01 00:00:00 UTC stamp = 1704067200 as a reference point.
user:wam_cpp_test_get_time_positive :- get_time(T), T > 1700000000.0.
user:wam_cpp_test_stamp_utc         :-
    stamp_date_time(1704067200,
                    date(Y, Mo, D, H, Mi, S, _, TZ, _), 'UTC'),
    Y = 2024, Mo = 1, D = 1, H = 0, Mi = 0, S =:= 0.0, TZ = 'UTC'.
user:wam_cpp_test_stamp_subsec      :-
    stamp_date_time(1704067200.25, date(_, _, _, _, _, S, _, _, _), 'UTC'),
    S > 0.24, S < 0.26.
user:wam_cpp_test_dt_roundtrip      :-
    stamp_date_time(1704067200, DT, 'UTC'),
    date_time_stamp(DT, S),
    S =:= 1704067200.0.
user:wam_cpp_test_date6_to_stamp    :-
    date_time_stamp(date(2024, 1, 1, 12, 30, 45), S),
    S > 1700000000.0.
user:wam_cpp_test_format_basic      :-
    format_time(A, '%Y-%m-%d', 1704067200), A = '2024-01-01'.
user:wam_cpp_test_format_year       :-
    format_time(A, '%Y', 1704067200), A = '2024'.
user:wam_cpp_test_format_from_date9 :-
    stamp_date_time(1704067200, DT, 'UTC'),
    format_time(A, '%Y-%m-%d', DT), A = '2024-01-01'.
% DCG call//N (DCG-body meta-call) -- works via composition of SWI's
% --> term_expansion (which rewrites call(P) inside a DCG body to
% call(P, S0, S)) and the existing dispatch_call_meta. No new runtime
% code needed; this section documents and locks in the behavior.
user:dcg_emit_b --> [b].
user:dcg_emit_two(X, Y) --> [X], [Y].
user:dcg_emit_a --> [a].
user:dcg_with_call0 --> [a], call(user:dcg_emit_b), [c].
user:dcg_with_call_full --> [start], call(user:dcg_emit_two(x, y)), [end].
user:dcg_with_call_partial --> [start], call(user:dcg_emit_two(x), y), [end].
user:dcg_rep0(_, 0) --> [].
user:dcg_rep0(P, N) -->
    { N > 0, N1 is N - 1 }, call(P), user:dcg_rep0(P, N1).
user:wam_cpp_test_dcgcall_n0      :- phrase(dcg_with_call0, [a, b, c]).
user:wam_cpp_test_dcgcall_n0_no   :- \+ phrase(dcg_with_call0, [a, b, d]).
user:wam_cpp_test_dcgcall_full    :-
    phrase(dcg_with_call_full, [start, x, y, end]).
user:wam_cpp_test_dcgcall_partial :-
    phrase(dcg_with_call_partial, [start, x, y, end]).
user:wam_cpp_test_dcgcall_partial_no :-
    \+ phrase(dcg_with_call_partial, [start, x, z, end]).
user:wam_cpp_test_dcgcall_rep3    :-
    phrase(dcg_rep0(dcg_emit_a, 3), [a, a, a]).
user:wam_cpp_test_dcgcall_rep0    :-
    phrase(dcg_rep0(dcg_emit_a, 0), []).
user:wam_cpp_test_dcgcall_rep_no  :-
    \+ phrase(dcg_rep0(dcg_emit_a, 3), [a, a]).
% lists_extra stdlib feature: pairs_keys/values/keys_values, take/3,
% drop/3, intersection/3, union/3, permutation/2. All implemented as
% user-module Prolog asserted at wam_cpp_target.pl load and pulled
% in via include_stdlib(lists_extra) or include_stdlib(true).
user:wam_cpp_test_pairs_keys     :- pairs_keys([a-1, b-2, c-3], [a, b, c]).
user:wam_cpp_test_pairs_values   :- pairs_values([a-1, b-2, c-3], [1, 2, 3]).
user:wam_cpp_test_pairs_kv       :-
    pairs_keys_values([a-1, b-2], K, V), K = [a, b], V = [1, 2].
user:wam_cpp_test_take_some      :- take(3, [a, b, c, d, e], [a, b, c]).
user:wam_cpp_test_take_overlong  :- take(10, [a, b, c], [a, b, c]).
user:wam_cpp_test_take_zero      :- take(0, [a, b, c], []).
user:wam_cpp_test_drop_some      :- drop(2, [a, b, c, d, e], [c, d, e]).
user:wam_cpp_test_drop_overlong  :- drop(10, [a, b, c], []).
user:wam_cpp_test_drop_zero      :- drop(0, [a, b, c], [a, b, c]).
user:wam_cpp_test_intersection   :- intersection([1, 2, 3, 4], [3, 4, 5, 6], [3, 4]).
user:wam_cpp_test_intersection_empty :- intersection([1, 2], [3, 4], []).
user:wam_cpp_test_union          :- union([1, 2, 3], [3, 4, 5], [1, 2, 3, 4, 5]).
user:wam_cpp_test_permutation_check :- permutation([a, b, c], [c, a, b]).
user:wam_cpp_test_permutation_no    :- \+ permutation([a, b, c], [a, b, d]).
% Indexing-bug regression tests. Three-clause predicate with first
% arg [] / [H|T] / [H|T] used to lose backtracking when SwitchOnTerm
% jumped past the chain's entry TryMeElse. Both flat and recursive
% backtracking are exercised.
user:wam_cpp_idx_bar([], []).
user:wam_cpp_idx_bar([H|T], [H|R]) :- H > 10, wam_cpp_idx_bar(T, R).
user:wam_cpp_idx_bar([_|T], R) :- wam_cpp_idx_bar(T, R).
user:wam_cpp_test_idx_flat        :- wam_cpp_idx_bar([5], R), R = [].
user:wam_cpp_test_idx_keep        :- wam_cpp_idx_bar([15], R), R = [15].
user:wam_cpp_test_idx_drop_first  :- wam_cpp_idx_bar([5, 15], R), R = [15].
user:wam_cpp_test_idx_keep_first  :- wam_cpp_idx_bar([15, 5], R), R = [15].
user:wam_cpp_test_idx_mixed       :- wam_cpp_idx_bar([5, 15, 7, 20], R),
                                     R = [15, 20].
% Stream I/O: open/close + read_line_to_string + read_string +
% at_end_of_stream + write_to_stream/nl_to_stream. Each test
% creates a temp file in /tmp, exercises the path, then cleans up
% (we just leave the file -- /tmp is ephemeral).
user:wam_cpp_test_stream_roundtrip :-
    open('/tmp/wam_stream_t1.txt', write, W),
    write_to_stream(W, 'line one'), nl_to_stream(W),
    write_to_stream(W, 'line two'), nl_to_stream(W),
    close(W),
    open('/tmp/wam_stream_t1.txt', read, R),
    read_line_to_string(R, L1),
    read_line_to_string(R, L2),
    read_line_to_string(R, EOF),
    close(R),
    L1 = 'line one', L2 = 'line two', EOF = end_of_file.
user:wam_cpp_test_stream_read_string :-
    open('/tmp/wam_stream_t2.txt', write, W),
    write_to_stream(W, 'hello world'),
    close(W),
    open('/tmp/wam_stream_t2.txt', read, R),
    read_string(R, 5, N1, _, S1),
    read_string(R, 100, N2, _, S2),
    close(R),
    S1 = 'hello', N1 = 5,
    S2 = ' world', N2 = 6.
user:wam_cpp_test_stream_append :-
    open('/tmp/wam_stream_t3.txt', write, W1),
    write_to_stream(W1, 'first'), nl_to_stream(W1),
    close(W1),
    open('/tmp/wam_stream_t3.txt', append, W2),
    write_to_stream(W2, 'second'), nl_to_stream(W2),
    close(W2),
    open('/tmp/wam_stream_t3.txt', read, R),
    read_line_to_string(R, L1),
    read_line_to_string(R, L2),
    close(R),
    L1 = 'first', L2 = 'second'.
user:wam_cpp_test_stream_at_end :-
    open('/tmp/wam_stream_t4.txt', write, W),
    write_to_stream(W, 'one'), nl_to_stream(W),
    close(W),
    open('/tmp/wam_stream_t4.txt', read, R),
    \+ at_end_of_stream(R),
    read_line_to_string(R, _),
    at_end_of_stream(R),
    close(R).
user:wam_cpp_test_stream_missing_file :-
    catch(open('/tmp/wam_stream_does_not_exist.xyz', read, _),
          error(existence_error(source_sink, _), _),
          true).
user:wam_cpp_test_stream_close_unknown :-
    catch(close(no_such_handle),
          error(existence_error(stream, _), _),
          true).
% Introspection: current_predicate/1 (check mode) and
% predicate_property/2. Cover static lookups, dynamic-db lookups,
% preloaded-bytecode (e.g. append/3), property atoms (defined,
% static, dynamic), number_of_clauses/1, and the type/instantiation
% error paths.
:- dynamic user:wam_cpp_intro_dyn/1.
user:wam_cpp_intro_dyn(initial).
user:wam_cpp_intro_static(a).
user:wam_cpp_intro_static(b).
user:wam_cpp_test_intro_static_yes :-
    current_predicate(wam_cpp_intro_static/1).
user:wam_cpp_test_intro_static_no :-
    \+ current_predicate(wam_cpp_no_such_pred/3).
user:wam_cpp_test_intro_arity_no :-
    \+ current_predicate(wam_cpp_intro_static/2).
user:wam_cpp_test_intro_builtin_yes :-
    current_predicate(append/3).
user:wam_cpp_test_intro_prop_defined :-
    predicate_property(wam_cpp_intro_static(_), defined).
user:wam_cpp_test_intro_prop_static :-
    predicate_property(wam_cpp_intro_static(_), static).
user:wam_cpp_test_intro_prop_dynamic :-
    assertz(wam_cpp_intro_dyn(more)),
    predicate_property(wam_cpp_intro_dyn(_), dynamic).
user:wam_cpp_test_intro_prop_static_not_dynamic :-
    \+ predicate_property(wam_cpp_intro_static(_), dynamic).
user:wam_cpp_test_intro_prop_count :-
    retractall(wam_cpp_intro_dyn(_)),
    assertz(wam_cpp_intro_dyn(a)),
    assertz(wam_cpp_intro_dyn(b)),
    assertz(wam_cpp_intro_dyn(c)),
    predicate_property(wam_cpp_intro_dyn(_), number_of_clauses(N)),
    N = 3.
user:wam_cpp_test_intro_inst_throw :-
    catch(current_predicate(_),
          error(instantiation_error, _),
          true).
user:wam_cpp_test_intro_indicator_throw :-
    catch(current_predicate(foo),
          error(type_error(predicate_indicator, _), _),
          true).
% Filesystem helpers + with_output_to(stream(_), G). Each test
% sets up its own temp file/dir in /tmp. Tests are sequenced so
% they don't collide on shared paths.
user:wam_cpp_test_fs_exists_file_yes :- exists_file('/etc/hostname').
user:wam_cpp_test_fs_exists_file_no  :- \+ exists_file('/no/such/x.zzz').
user:wam_cpp_test_fs_exists_dir_yes  :- exists_directory('/tmp').
user:wam_cpp_test_fs_exists_dir_no   :- \+ exists_directory('/no/such/dir').
user:wam_cpp_test_fs_make_check :-
    catch(delete_file('/tmp/wam_e2e_makedir/.keep'), _, true),
    catch(delete_file('/tmp/wam_e2e_makedir'), _, true),
    make_directory('/tmp/wam_e2e_makedir'),
    exists_directory('/tmp/wam_e2e_makedir').
user:wam_cpp_test_fs_dir_files :-
    catch(make_directory('/tmp/wam_e2e_dir'), _, true),
    open('/tmp/wam_e2e_dir/file_a.txt', write, W1), close(W1),
    open('/tmp/wam_e2e_dir/file_b.txt', write, W2), close(W2),
    directory_files('/tmp/wam_e2e_dir', Files),
    member('file_a.txt', Files),
    member('file_b.txt', Files),
    member('.', Files),
    member('..', Files).
user:wam_cpp_test_fs_delete :-
    open('/tmp/wam_e2e_to_delete.txt', write, W), close(W),
    exists_file('/tmp/wam_e2e_to_delete.txt'),
    delete_file('/tmp/wam_e2e_to_delete.txt'),
    \+ exists_file('/tmp/wam_e2e_to_delete.txt').
user:wam_cpp_test_fs_delete_missing :-
    catch(delete_file('/no/such/file.zzz'),
          error(existence_error(source_sink, _), _),
          true).
user:wam_cpp_test_wos_stream :-
    open('/tmp/wam_e2e_wos.txt', write, S),
    with_output_to(stream(S), (write(hello), write(' '), write(world))),
    close(S),
    open('/tmp/wam_e2e_wos.txt', read, R),
    read_line_to_string(R, Line),
    close(R),
    Line = 'hello world'.
user:wam_cpp_test_wos_format :-
    open('/tmp/wam_e2e_wos2.txt', write, S),
    with_output_to(stream(S), format("~w-~w", [42, ok])),
    close(S),
    open('/tmp/wam_e2e_wos2.txt', read, R),
    read_line_to_string(R, Line),
    close(R),
    Line = '42-ok'.
% Runtime clause/2 over dynamic predicates. Iterates dynamic_db
% with backtracking via the shared RetractIterator + is_clause_only
% flag (no removal, unifies both Head and Body). Body is `true` for
% bare facts; `Body` from a `:-/2` stored term otherwise.
:- dynamic user:wam_cpp_clause_p/1.
:- dynamic user:wam_cpp_clause_q/2.
user:wam_cpp_clause_setup_p :-
    retractall(wam_cpp_clause_p(_)),
    assertz(wam_cpp_clause_p(1)),
    assertz(wam_cpp_clause_p(2)),
    assertz(wam_cpp_clause_p(3)).
user:wam_cpp_clause_setup_q :-
    retractall(wam_cpp_clause_q(_, _)),
    assertz(wam_cpp_clause_q(a, 10)),
    assertz((wam_cpp_clause_q(b, 20) :- write(side_b), nl)),
    assertz(wam_cpp_clause_q(c, 30)).
user:wam_cpp_test_clause_fact :-
    wam_cpp_clause_setup_p,
    clause(wam_cpp_clause_p(1), B), B = true.
user:wam_cpp_test_clause_var :-
    wam_cpp_clause_setup_p,
    clause(wam_cpp_clause_p(X), B), X = 1, B = true.
user:wam_cpp_test_clause_enum :-
    wam_cpp_clause_setup_p,
    findall(X, clause(wam_cpp_clause_p(X), true), L),
    L = [1, 2, 3].
user:wam_cpp_test_clause_rule_body :-
    wam_cpp_clause_setup_q,
    findall(triple(K, V, B), clause(wam_cpp_clause_q(K, V), B), L),
    L = [triple(a, 10, true),
         triple(b, 20, (write(side_b), nl)),
         triple(c, 30, true)].
user:wam_cpp_test_clause_missing :-
    wam_cpp_clause_setup_p,
    \+ clause(wam_cpp_clause_p(99), _).
user:wam_cpp_test_clause_unknown :-
    \+ clause(wam_cpp_no_such_dyn(_), _).
user:wam_cpp_test_clause_after_assertz :-
    wam_cpp_clause_setup_p,
    assertz(wam_cpp_clause_p(4)),
    findall(X, clause(wam_cpp_clause_p(X), _), L),
    L = [1, 2, 3, 4].
% findall with nested compound templates. Previously compile_compound_template
% threw on any compound arg, which got caught + swallowed by the per-
% predicate try-catch in compile_predicates_for_project, silently
% dropping the affected predicate from the output. The fix recurses
% into nested compounds using set_variable + put_structure on the
% sub-reg so the parent''s arg slot sees the sub-compound. Each test
% asserts ground facts and findalls them through a nested template.
:- dynamic user:wam_cpp_nest_emit/3.
user:wam_cpp_nest_setup :-
    retractall(wam_cpp_nest_emit(_, _, _)),
    assertz(wam_cpp_nest_emit(a, 10, true)),
    assertz(wam_cpp_nest_emit(b, 20, false)).
user:wam_cpp_test_nested_dash :-
    wam_cpp_nest_setup,
    findall(K-V-B, wam_cpp_nest_emit(K, V, B), L),
    L = [a-10-true, b-20-false].
user:wam_cpp_test_nested_f_g :-
    wam_cpp_nest_setup,
    findall(f(g(K, V), B), wam_cpp_nest_emit(K, V, B), L),
    L = [f(g(a, 10), true), f(g(b, 20), false)].
user:wam_cpp_test_nested_right :-
    wam_cpp_nest_setup,
    findall(g(K, f(V, B)), wam_cpp_nest_emit(K, V, B), L),
    L = [g(a, f(10, true)), g(b, f(20, false))].
user:wam_cpp_test_nested_constant :-
    wam_cpp_nest_setup,
    findall(g(K, f(V, marker)), wam_cpp_nest_emit(K, V, _), L),
    L = [g(a, f(10, marker)), g(b, f(20, marker))].
% Nondet current_predicate/1: enumeration via the CP-iterator path.
% PR #2277 added check-mode only; this PR moves the builtin from
% builtin() to the Call/Execute dispatch arms so partial-spec
% queries (Name/_, _/Arity, _) can iterate over labels + dynamic_db
% keys.
user:wam_cpp_cp_static_a(_).
user:wam_cpp_cp_static_b(_, _).
user:wam_cpp_test_cp_check_static :-
    current_predicate(wam_cpp_cp_static_a/1).
user:wam_cpp_test_cp_check_missing :-
    \+ current_predicate(wam_cpp_cp_no_such_pred/3).
user:wam_cpp_test_cp_enum_arity :-
    findall(A, current_predicate(wam_cpp_cp_static_a/A), L),
    L = [1].
user:wam_cpp_test_cp_enum_name_by_arity :-
    findall(N, current_predicate(N/0), L),
    member(wam_cpp_test_cp_enum_name_by_arity, L),
    member(wam_cpp_test_cp_check_static, L).
user:wam_cpp_test_cp_enum_all :-
    findall(N/A, current_predicate(N/A), L),
    member(append/3, L).
user:wam_cpp_test_cp_enum_none :-
    findall(N, current_predicate(N/999), L),
    L = [].
user:wam_cpp_test_cp_inst_throw :-
    catch(current_predicate(_),
          error(instantiation_error, _),
          true).
user:wam_cpp_test_cp_indicator_throw :-
    catch(current_predicate(foo),
          error(type_error(predicate_indicator, _), _),
          true).
user:wam_cpp_test_enum_member  :- findall(X, member(X, [a, b, c]), L),
                                  L = [a, b, c].

% Exception handling (catch/3 + throw/1) fixtures:
:- dynamic user:wam_cpp_test_catch_basic/0.
:- dynamic user:wam_cpp_test_catch_pass/0.
:- dynamic user:wam_cpp_test_catch_no_match/0.
:- dynamic user:wam_cpp_test_catch_nested/0.
:- dynamic user:wam_cpp_test_catch_fail/0.
:- dynamic user:wam_cpp_test_catch_compound/0.

user:wam_cpp_test_catch_basic     :- catch(throw(my_error), E, E = my_error).
user:wam_cpp_test_catch_pass      :- catch(true, _E, fail).
user:wam_cpp_test_catch_no_match  :- catch(throw(err_a), other, true).
user:wam_cpp_test_catch_nested    :- catch(catch(throw(inner), no_match, fail),
                                           E, E = inner).
user:wam_cpp_test_catch_fail      :- catch(fail, _, true).
user:wam_cpp_test_catch_compound  :- catch(throw(error(type_error, ctx)),
                                           error(Kind, _),
                                           Kind = type_error).

% List-builtin batch 2 (append/3, reverse/2, last/2, nth0/3):
:- dynamic user:wam_cpp_test_append_basic/0.
:- dynamic user:wam_cpp_test_append_empty_first/0.
:- dynamic user:wam_cpp_test_append_empty_second/0.
:- dynamic user:wam_cpp_test_reverse_basic/0.
:- dynamic user:wam_cpp_test_reverse_empty/0.
:- dynamic user:wam_cpp_test_reverse_singleton/0.
:- dynamic user:wam_cpp_test_last_basic/0.
:- dynamic user:wam_cpp_test_last_single/0.
:- dynamic user:wam_cpp_test_nth0_first/0.
:- dynamic user:wam_cpp_test_nth0_middle/0.
:- dynamic user:wam_cpp_test_nth0_last/0.

user:wam_cpp_test_append_basic         :- append([a,b], [c,d], L), L = [a,b,c,d].
user:wam_cpp_test_append_empty_first   :- append([], [a,b], [a,b]).
user:wam_cpp_test_append_empty_second  :- append([a,b], [], [a,b]).
user:wam_cpp_test_reverse_basic        :- reverse([a,b,c], R), R = [c,b,a].
user:wam_cpp_test_reverse_empty        :- reverse([], R), R = [].
user:wam_cpp_test_reverse_singleton    :- reverse([x], R), R = [x].
user:wam_cpp_test_last_basic           :- last([a,b,c], X), X = c.
user:wam_cpp_test_last_single          :- last([only], X), X = only.
user:wam_cpp_test_nth0_first           :- nth0(0, [a,b,c], X), X = a.
user:wam_cpp_test_nth0_middle          :- nth0(1, [a,b,c], X), X = b.
user:wam_cpp_test_nth0_last            :- nth0(2, [a,b,c], X), X = c.

% format/1 + format/2 fixtures. Each predicate prints something to
% stdout; the e2e test asserts exact captured output (printed text +
% the final true/false line).
:- dynamic user:wam_cpp_test_fmt1_noargs/0.
:- dynamic user:wam_cpp_test_fmt2_atoms/0.
:- dynamic user:wam_cpp_test_fmt2_ints/0.
:- dynamic user:wam_cpp_test_fmt2_compound/0.
:- dynamic user:wam_cpp_test_fmt2_tilde/0.
:- dynamic user:wam_cpp_test_fmt2_no_directives/0.

user:wam_cpp_test_fmt1_noargs       :- format('plain text~n').
user:wam_cpp_test_fmt2_atoms        :- format('a=~w b=~w~n', [hello, world]).
user:wam_cpp_test_fmt2_ints         :- format('~d + ~d = ~d~n', [1, 2, 3]).
user:wam_cpp_test_fmt2_compound     :- format('result: ~w~n', [foo(1, bar)]).
user:wam_cpp_test_fmt2_tilde        :- format('100~~~n', []).
user:wam_cpp_test_fmt2_no_directives :- format('hello world', []).

% ISO-error fixtures for is_iso/2 + is_lax/2. Some predicates expect
% to be flipped to ISO mode by the test''s write_wam_cpp_project
% options; others stay in lax mode to exercise the no-rewrite path.
:- dynamic user:wam_cpp_test_iso_is_type_error/0.
:- dynamic user:wam_cpp_test_iso_is_instantiation/0.
:- dynamic user:wam_cpp_test_iso_is_unmatched/0.
:- dynamic user:wam_cpp_test_lax_is_silent/0.
:- dynamic user:wam_cpp_test_explicit_lax_in_iso/0.
:- dynamic user:wam_cpp_test_iso_unbound_context/0.

user:wam_cpp_test_iso_is_type_error :-
    catch(X is foo, error(type_error(evaluable, _Culprit), _), X = X).

user:wam_cpp_test_iso_is_instantiation :-
    catch(Y is _Z + 1, error(instantiation_error, _), Y = Y).

% Unmatched catcher: ISO mode throws type_error, the catcher pattern
% doesn''t unify, so the throw propagates uncaught. Result: false.
user:wam_cpp_test_iso_is_unmatched :-
    catch(X is foo, error(some_other_kind, _), X = X).

% Lax mode: X is foo just fails. The (-> ; true) wrapper turns the
% failure into success so the predicate as a whole returns true —
% verifies that lax mode emits NO throw (otherwise the catch would
% need to recover).
user:wam_cpp_test_lax_is_silent :-
    (X is foo -> X = X ; true).

% Three-forms guarantee: explicit is_lax/2 inside an ISO-mode
% predicate must still fail silently. If the rewrite incorrectly
% touched the explicit form, this would throw and the (-> ; true)
% wrapper would still succeed — but stderr would have a "uncaught
% exception" trace and the catch above would catch type_error,
% breaking the (-> ; true) branch. Easier to test: just verify the
% (-> ; true) takes the false branch (returns true overall).
user:wam_cpp_test_explicit_lax_in_iso :-
    (is_lax(X, foo) -> X = X ; true).

% Verifies catch(_, error(Pattern, _), _) works even though Context
% is left unbound by throw_iso_error. Forward-looking test from
% SPECIFICATION §8.
user:wam_cpp_test_iso_unbound_context :-
    catch(X is foo,
          error(type_error(evaluable, Culprit), _Context),
          Culprit = foo/0).

% ISO sweep fixtures — arith compares + succ/2 + IEEE-754 float
% divide-by-zero. Each predicate tests one ISO/lax/explicit path.
:- dynamic user:wam_cpp_test_iso_gt_throws_inst/0.
:- dynamic user:wam_cpp_test_iso_lt_throws_type/0.
:- dynamic user:wam_cpp_test_iso_eq_throws_zero_div/0.
:- dynamic user:wam_cpp_test_lax_gt_silent_fail/0.
:- dynamic user:wam_cpp_test_explicit_lax_gt_in_iso/0.
:- dynamic user:wam_cpp_test_iso_succ_neg_throws/0.
:- dynamic user:wam_cpp_test_iso_succ_unbound_throws/0.
:- dynamic user:wam_cpp_test_iso_zero_div_throws/0.
:- dynamic user:wam_cpp_test_lax_float_div_zero_inf/0.
:- dynamic user:wam_cpp_test_lax_float_div_zero_nan/0.

user:wam_cpp_test_iso_gt_throws_inst :-
    % `_X > 5` with _X unbound → instantiation_error.
    catch(_X > 5, error(instantiation_error, _), true).

user:wam_cpp_test_iso_lt_throws_type :-
    % `foo < 5` → type_error(evaluable, foo/0).
    catch(foo < 5, error(type_error(evaluable, _), _), true).

user:wam_cpp_test_iso_eq_throws_zero_div :-
    % `1 / 0 =:= 0` → evaluation_error(zero_divisor) on int divide.
    catch(1 / 0 =:= 0,
          error(evaluation_error(zero_divisor), _),
          true).

user:wam_cpp_test_lax_gt_silent_fail :-
    % Lax: `_X > 5` with _X unbound → just fails.
    (_X > 5 -> true ; true).

user:wam_cpp_test_explicit_lax_gt_in_iso :-
    % Explicit `>_lax` inside an ISO-mode predicate must NOT throw —
    % three-forms guarantee end-to-end on a non-is/2 builtin.
    % Quoted because `>` is a Prolog operator and the unquoted form
    % `>_lax(_X, 5)` is a syntax error.
    ('>_lax'(_X, 5) -> true ; true).

user:wam_cpp_test_iso_succ_neg_throws :-
    % succ_iso(-1, _) → type_error(not_less_than_zero, -1).
    catch(succ(-1, _Y),
          error(type_error(not_less_than_zero, _), _),
          true).

user:wam_cpp_test_iso_succ_unbound_throws :-
    % succ_iso(_X, _Y) → instantiation_error.
    catch(succ(_X, _Y), error(instantiation_error, _), true).

user:wam_cpp_test_iso_zero_div_throws :-
    % Both int and float divide-by-zero throw under ISO.
    catch(_R is 1.0 / 0.0,
          error(evaluation_error(zero_divisor), _),
          true).

user:wam_cpp_test_lax_float_div_zero_inf :-
    % Lax: `R is 1.0 / 0.0` succeeds with R = inf. Verifies the
    % SPEC §6.1 IEEE-754 behavior change — previously this failed.
    R is 1.0 / 0.0,
    R > 1.0e308.

user:wam_cpp_test_lax_float_div_zero_nan :-
    % Lax: `R is 0.0 / 0.0` succeeds with NaN. IEEE 754 says
    % NaN =\= NaN is true (NaN is not equal to anything, including
    % itself), so this is the simplest self-checking NaN signature.
    R is 0.0 / 0.0,
    R =\= R.

% \+/1 and not/1 (negation as failure):
:- dynamic user:wam_cpp_test_not_fail/0.
:- dynamic user:wam_cpp_test_not_true/0.
:- dynamic user:wam_cpp_test_not_compound/0.
:- dynamic user:wam_cpp_test_not_alias_succeeds/0.
:- dynamic user:wam_cpp_test_not_alias_fails/0.
:- dynamic user:wam_cpp_test_not_nan_check/0.

user:wam_cpp_test_not_fail        :- \+ fail.
user:wam_cpp_test_not_true        :- \+ true.
% Conjunction-as-data exercises the goal-term-with-,/2 dispatch path
% (whose bug surfaced and was fixed in the ISO sweep). `X = a, X = b`
% cannot succeed, so its negation succeeds.
user:wam_cpp_test_not_compound    :- \+ (X = a, X = b).
user:wam_cpp_test_not_alias_succeeds :- not(fail).
user:wam_cpp_test_not_alias_fails    :- not(true).
% The original gap that motivated this PR — NaN self-check needs \+/1
% because NaN =:= NaN is false but \=== NaN at the structural level.
user:wam_cpp_test_not_nan_check   :- R is 0.0 / 0.0, \+ (R =:= R).

% call/N (meta-call):
:- dynamic user:wam_cpp_call_helper/1.
:- dynamic user:wam_cpp_test_call_atom/0.
:- dynamic user:wam_cpp_test_call_with_args/0.
:- dynamic user:wam_cpp_test_call_partial/0.
:- dynamic user:wam_cpp_test_call_compound_already/0.
:- dynamic user:wam_cpp_test_call_user_pred/0.

% Helper user predicate to dispatch indirectly.
user:wam_cpp_call_helper(hello).

% call(true) — 0-extra-args, atom goal. Tail-call path.
user:wam_cpp_test_call_atom :- call(true).

% call(=, X, 5) — 2 extras appended to atom functor. Mid-body Call
% path. Verifies the X gets bound to 5 by the dispatched =/2.
user:wam_cpp_test_call_with_args :- call(=, X, 5), X = 5.

% call(=(X), 7) — 1 extra appended to a 1-arg compound. The combined
% goal is =(X, 7), arity 2. Exercises the existing-args-plus-extras
% path of dispatch_call_meta.
user:wam_cpp_test_call_partial :- G = =(X), call(G, 7), X = 7.

% call(G) where G is already a full goal — no extras, compound goal.
user:wam_cpp_test_call_compound_already :-
    G = wam_cpp_call_helper(hello),
    call(G).

% call(F, X) dispatching to a USER predicate (not a builtin). Tests
% the user-label dispatch path inside invoke_goal_as_call.
user:wam_cpp_test_call_user_pred :-
    call(wam_cpp_call_helper, hello).

% maplist/2 + maplist/3 (helper-injected, built on call/N):
:- dynamic user:wam_cpp_double/2.
:- dynamic user:wam_cpp_positive/1.
:- dynamic user:wam_cpp_test_maplist2_all/0.
:- dynamic user:wam_cpp_test_maplist2_empty/0.
:- dynamic user:wam_cpp_test_maplist3_double/0.
:- dynamic user:wam_cpp_test_maplist3_check/0.
:- dynamic user:wam_cpp_test_findall_call/0.

user:wam_cpp_double(X, Y) :- Y is X * 2.
user:wam_cpp_positive(X) :- X > 0.

% Every element satisfies positive/1.
user:wam_cpp_test_maplist2_all :- maplist(wam_cpp_positive, [1, 2, 3]).
% Empty-list base case.
user:wam_cpp_test_maplist2_empty :- maplist(wam_cpp_positive, []).
% Higher-order list transformation: build the doubles list from
% [1,2,3]. Verifies maplist/3 + call/3 + user predicate compose.
user:wam_cpp_test_maplist3_double :-
    maplist(wam_cpp_double, [1, 2, 3], L),
    L = [2, 4, 6].
% Both lists ground — verifies P holds for each paired (X, Y).
user:wam_cpp_test_maplist3_check :-
    maplist(wam_cpp_double, [1, 2], [2, 4]).
% findall composes with call/N: collect the double of 1 into a
% single-element list. (Note: findall + member + call/N in
% conjunction has a separate latent bug unrelated to this PR — it
% hangs because of how findall''s aggregate frame interacts with
% member''s choice points; the simpler form here works.)
user:wam_cpp_test_findall_call :-
    findall(Y, call(wam_cpp_double, 1, Y), L),
    L = [2].

% findall + conjunction with backtracking. The previous PR (#2097)
% flagged this pattern as "hangs" — but that turned out to be a
% symptom of the PutStructure aliasing bug fixed in the same PR.
% Now that aliasing through A-regs is correct, these compose
% naturally. Tests here lock that in and exercise the higher-order
% patterns end-to-end.
:- dynamic user:wam_cpp_test_findall_member_arith/0.
:- dynamic user:wam_cpp_test_findall_member_user/0.
:- dynamic user:wam_cpp_test_findall_member_call/0.
:- dynamic user:wam_cpp_test_findall_filtered/0.
:- dynamic user:wam_cpp_test_findall_three_goals/0.

user:wam_cpp_test_findall_member_arith :-
    findall(Y, (member(X, [1, 2, 3]), Y is X * 2), L),
    L = [2, 4, 6].

user:wam_cpp_test_findall_member_user :-
    findall(Y, (member(X, [1, 2, 3]), wam_cpp_double(X, Y)), L),
    L = [2, 4, 6].

user:wam_cpp_test_findall_member_call :-
    findall(Y,
            (member(X, [1, 2, 3]), call(wam_cpp_double, X, Y)),
            L),
    L = [2, 4, 6].

user:wam_cpp_test_findall_filtered :-
    findall(X, (member(X, [1, 2, 3, 4, 5]), X > 2), L),
    L = [3, 4, 5].

user:wam_cpp_test_findall_three_goals :-
    findall(Z,
            (member(X, [1, 2, 3]), Y is X * 2, Z is Y + 10),
            L),
    L = [12, 14, 16].

% Nested findalls — the latent bug from PR #2098. The inner findall
% is NOT inlined by the WAM compiler (only the outermost
% findall/bagof/setof gets BeginAggregate-inlined); it''s emitted as
% a plain `call findall/3, 3`. Resolving this needed:
%   1. A meta-call findall/3 dispatcher (`dispatch_findall_call`)
%      that pushes an AggregateFrame and invokes the goal with cp =
%      findall_collect_pc.
%   2. A ConjFrame mechanism so the inner findall''s conjunction
%      goal-term (`,(member(...), X =< N)`) gets dispatched as G1
%      then G2 with proper backtracking through G1''s CPs.
:- dynamic user:wam_cpp_test_findall_nested/0.
:- dynamic user:wam_cpp_test_findall_nested_simple/0.
:- dynamic user:wam_cpp_test_findall_meta_no_conjunction/0.

% The original reproduction from PR #2098''s deferred list.
user:wam_cpp_test_findall_nested :-
    findall(L,
            (member(N, [2, 3]),
             findall(X, (member(X, [1, 2, 3, 4]), X =< N), L)),
            Ls),
    Ls = [[1, 2], [1, 2, 3]].

% Simpler nested case: inner findall without conjunction goal.
user:wam_cpp_test_findall_nested_simple :-
    findall(L,
            (member(N, [1, 2]),
             findall(N, member(_, [a, b, N]), L)),
            Ls),
    Ls = [[1, 1, 1], [2, 2, 2]].

% Meta-call findall/3 directly (no conjunction).
user:wam_cpp_test_findall_meta_no_conjunction :-
    % G is a single-goal compound — exercises dispatch_findall_call
    % WITHOUT the ConjFrame path.
    findall(X, member(X, [a, b, c]), L),
    L = [a, b, c].

% bagof/3 + setof/3 — share dispatch_aggregate_call with findall.
% bagof FAILS on empty acc; setof sorts + dedups via standard term
% order. Nested forms use the meta-call path (same as nested findall
% from PR #2099).
:- dynamic user:wam_cpp_test_bagof_basic/0.
:- dynamic user:wam_cpp_test_bagof_fails_empty/0.
:- dynamic user:wam_cpp_test_setof_basic/0.
:- dynamic user:wam_cpp_test_setof_dedups/0.
:- dynamic user:wam_cpp_test_setof_sorts_ints/0.
:- dynamic user:wam_cpp_test_bagof_nested/0.
:- dynamic user:wam_cpp_test_setof_nested/0.

user:wam_cpp_test_bagof_basic :-
    bagof(X, member(X, [a, b, c]), L), L = [a, b, c].

% bagof fails on empty acc (per ISO); the if-then-else wraps that
% so the predicate succeeds via the else branch. Regression guard
% for the backtrack-continue-on-Uninit fix in aggregate-finalise.
user:wam_cpp_test_bagof_fails_empty :-
    (bagof(_X, (member(_X, [a, b]), _X = z), _) -> true ; true).

user:wam_cpp_test_setof_basic :-
    setof(X, member(X, [c, a, b]), L), L = [a, b, c].

user:wam_cpp_test_setof_dedups :-
    setof(X, member(X, [c, a, b, a, c]), L), L = [a, b, c].

user:wam_cpp_test_setof_sorts_ints :-
    setof(X, member(X, [3, 1, 2, 1]), L), L = [1, 2, 3].

% Nested bagof — inner is non-inlined. Existential quantifier
% (N^Goal) exercises the ^/2 transparency path. Without the
% ^/2 handler, the body would fall through to builtin() and fail.
user:wam_cpp_test_bagof_nested :-
    bagof(L,
          N^(member(N, [2, 3]),
             bagof(X, (member(X, [1, 2, 3, 4]), X =< N), L)),
          Ls),
    Ls = [[1, 2], [1, 2, 3]].

% Nested setof — also exercises term_less recursion for the outer
% sort. List terms with the same functor ([|]/2) require
% args-level comparison to be ordered correctly.
user:wam_cpp_test_setof_nested :-
    setof(L,
          N^(member(N, [3, 2]),
             setof(X, (member(X, [3, 1, 2, 1]), X =< N), L)),
          Ls),
    Ls = [[1, 2], [1, 2, 3]].

% Disjunction goal-terms (;/2) as meta-call args. The WAM compiler
% inlines disjunction inside findall/bagof/setof body but builds it
% as a ;/2 compound when passed to catch/3, \+/1, call/1, or a
% non-inlined meta-aggregate (inner findall/bagof/setof).
:- dynamic user:wam_cpp_test_catch_disj/0.
:- dynamic user:wam_cpp_test_not_disj_both_fail/0.
:- dynamic user:wam_cpp_test_call_disj_first/0.
:- dynamic user:wam_cpp_test_call_disj_second/0.
:- dynamic user:wam_cpp_test_call_disj_first_fails/0.

% catch with disjunction inside — first alternative succeeds.
user:wam_cpp_test_catch_disj :-
    catch((true ; fail), _E, fail).

% \+ (fail ; fail) — both alternatives fail → \+ succeeds.
user:wam_cpp_test_not_disj_both_fail :-
    \+ (fail ; fail).

% call((G1 ; G2)) — G1 succeeds, X = 1.
user:wam_cpp_test_call_disj_first :-
    call((X = 1 ; X = 2)),
    X = 1.

% call((G1 ; G2)) — backtrack into G2 by requiring X = 2.
% Exercises the DisjAlt path: after G1 binds X=1, the X=2 check
% fails, backtrack pops the disjunction''s CP, DisjAlt dispatches
% G2 which binds X=2.
user:wam_cpp_test_call_disj_second :-
    call((X = 1 ; X = 2)),
    X = 2.

% call((fail ; G2)) — G1 fails immediately, G2 dispatched via
% DisjAlt. Tests the failure-on-first-arm path.
user:wam_cpp_test_call_disj_first_fails :-
    call((fail ; X = 7)),
    X = 7.

% If-then-else goal-terms `;(->(Cond, Then), Else)` — built by the
% WAM compiler when the user writes `(Cond -> Then ; Else)` as data
% (passed to catch/3, call/1, or inside a non-inlined meta-call).
% Distinct from plain disjunction because of cut semantics: when
% Cond succeeds, all of Cond''s choice points are committed away.
:- dynamic user:wam_cpp_test_ite_then_branch/0.
:- dynamic user:wam_cpp_test_ite_else_branch/0.
:- dynamic user:wam_cpp_test_ite_inside_findall/0.
:- dynamic user:wam_cpp_test_ite_cut_commits_cond/0.

% Cond=true → Then branch fires.
user:wam_cpp_test_ite_then_branch :-
    call((true -> X = 1 ; X = 2)),
    X = 1.

% Cond=fail → Else branch fires (via IfThenElse path).
user:wam_cpp_test_ite_else_branch :-
    call((fail -> X = 1 ; X = 2)),
    X = 2.

% findall + if-then-else as call''d goal-term — each iteration of
% the outer member picks Then or Else based on the condition.
user:wam_cpp_test_ite_inside_findall :-
    findall(Y, (member(X, [1, 2, 3, 4]),
                call((X > 2 -> Y = big ; Y = small))),
            L),
    L = [small, small, big, big].

% Cut semantics: Cond uses member/2 which has multiple solutions.
% After Cond commits with X=1, we don''t backtrack to try X=2.
% Y must be 1 — if cut weren''t happening, this would also succeed
% with Y=2 via re-dispatch of Cond.
user:wam_cpp_test_ite_cut_commits_cond :-
    call((member(X, [1, 2, 3]) -> Y = X ; Y = none)),
    Y = 1.

% Bare if-then goal-terms `(Cond -> Then)` — no Else. The WAM
% compiler builds them as `->/2` at the top level (not wrapped in
% ;/2). On Cond failure the whole construct fails. On Cond success
% Cond''s CPs are cut and Then runs.
:- dynamic user:wam_cpp_test_bif_then_runs/0.
:- dynamic user:wam_cpp_test_bif_cond_fail_propagates/0.
:- dynamic user:wam_cpp_test_bif_inside_catch/0.
:- dynamic user:wam_cpp_test_bif_not_when_cond_fails/0.
:- dynamic user:wam_cpp_test_bif_cut_commits/0.

% Cond succeeds → Then runs.
user:wam_cpp_test_bif_then_runs :-
    call((true -> X = 1)),
    X = 1.

% Cond fails → bare-if-then fails → outer if-then-else takes Else.
user:wam_cpp_test_bif_cond_fail_propagates :-
    (call((fail -> _Y = 1)) -> false ; true).

% Inside catch — Cond=true succeeds, Then binds X=7, catch passes
% through with no throw.
user:wam_cpp_test_bif_inside_catch :-
    catch((true -> X = 7), _, fail),
    X = 7.

% \+ (Cond fails → bif fails) → \+ succeeds.
user:wam_cpp_test_bif_not_when_cond_fails :-
    \+ call((fail -> _Y = 1)).

% Cut: Cond uses member/2 with 3 solutions. After committing to
% X=1, no backtrack to X=2/X=3.
user:wam_cpp_test_bif_cut_commits :-
    call((member(X, [1, 2, 3]) -> Y = X)),
    Y = 1.

% bagof/setof witness grouping. The WAM compiler inlines the
% OUTERMOST bagof/setof — there the goal is direct WAM, no term to
% walk for witnesses, so it behaves findall-like. For meta-call
% (non-inlined) dispatch, dispatch_aggregate_call walks the goal
% term to find free witnesses (vars in goal NOT in template and
% NOT under ^/2) and groups results by witness binding. First-
% group binding only for v1; backtracking through additional
% groups is a planned follow-up.
:- dynamic user:wam_cpp_parent_fixture/2.
:- dynamic user:wam_cpp_test_bagof_meta_groups_by_witness/0.
:- dynamic user:wam_cpp_test_bagof_meta_existential_no_grouping/0.
:- dynamic user:wam_cpp_test_setof_meta_groups_sorted/0.
:- dynamic user:wam_cpp_test_bagof_inlined_groups_by_witness/0.
:- dynamic user:wam_cpp_test_bagof_inlined_existential_flattens/0.
:- dynamic user:wam_cpp_test_setof_inlined_groups_sorted/0.

user:wam_cpp_parent_fixture(tom, bob).
user:wam_cpp_parent_fixture(tom, alice).
user:wam_cpp_parent_fixture(jane, carol).

% bagof via catch wrapper → meta-call dispatch → witness grouping.
% P is a free witness; first group is P=tom → L=[bob,alice].
% Verifies BOTH the result list AND the witness binding.
user:wam_cpp_test_bagof_meta_groups_by_witness :-
    catch(bagof(C, wam_cpp_parent_fixture(P, C), L), _, fail),
    L = [bob, alice],
    P = tom.

% Same goal but with ^/2 existential: P is suppressed → no
% witnesses → single flat group with all 3 children.
user:wam_cpp_test_bagof_meta_existential_no_grouping :-
    catch(bagof(C, P^wam_cpp_parent_fixture(P, C), L), _, fail),
    L = [bob, alice, carol].

% setof grouping — same witness logic as bagof but the group''s
% template list is sorted + dedup''d.
user:wam_cpp_test_setof_meta_groups_sorted :-
    catch(setof(C, wam_cpp_parent_fixture(P, C), L), _, fail),
    L = [alice, bob],
    P = tom.

% Inlined outer bagof — the WAM compiler now emits free-witness
% register info alongside BeginAggregate (4-arg form), so the inlined
% path does ISO witness grouping just like the meta-call path. First
% group is P=tom -> L=[bob, alice]; witness binds back to the caller.
user:wam_cpp_test_bagof_inlined_groups_by_witness :-
    bagof(C, wam_cpp_parent_fixture(P, C), L),
    L = [bob, alice],
    P = tom.

% Inlined bagof with caret existential — P is suppressed, so no
% witnesses are emitted (empty 4-arg). Falls back to single-group
% flat collection, matching the previous behaviour.
user:wam_cpp_test_bagof_inlined_existential_flattens :-
    bagof(C, P^wam_cpp_parent_fixture(P, C), L),
    L = [bob, alice, carol].

% Inlined setof grouping — first group sorted by term order.
user:wam_cpp_test_setof_inlined_groups_sorted :-
    setof(C, wam_cpp_parent_fixture(P, C), L),
    L = [alice, bob],
    P = tom.

% Bagof/setof GROUP BACKTRACKING — second-group binding works via
% the aggregate_next_group_pc CP machinery. Outer findall drives
% the inner bagof''s group iteration: each backtrack into bagof
% binds the next group.
:- dynamic user:wam_cpp_grandparent_fixture/2.
:- dynamic user:wam_cpp_test_findall_of_bagof_groups/0.
:- dynamic user:wam_cpp_test_findall_of_setof_groups/0.
:- dynamic user:wam_cpp_test_bagof_witness_pairs/0.
:- dynamic user:wam_cpp_test_bagof_single_group_unchanged/0.

% Extend the fixture for a more thorough multi-group test.
user:wam_cpp_grandparent_fixture(tom, bob).
user:wam_cpp_grandparent_fixture(tom, alice).
user:wam_cpp_grandparent_fixture(jane, carol).
user:wam_cpp_grandparent_fixture(jane, dave).

% Outer findall + inner bagof: groups iterate via the next-group
% CP. With 2 parents (tom, jane), each having 2 children, we get
% 2 groups.
user:wam_cpp_test_findall_of_bagof_groups :-
    findall(L,
            bagof(C, wam_cpp_grandparent_fixture(_P, C), L),
            Ls),
    Ls = [[bob, alice], [carol, dave]].

% Same with setof — each group sorted by term order.
user:wam_cpp_test_findall_of_setof_groups :-
    findall(L,
            setof(C, wam_cpp_grandparent_fixture(_P, C), L),
            Ls),
    Ls = [[alice, bob], [carol, dave]].

% Witness binding flows through to the outer findall: collect
% P-L pairs so we can verify the witness binding per group.
user:wam_cpp_test_bagof_witness_pairs :-
    findall(P-L,
            bagof(C, wam_cpp_grandparent_fixture(P, C), L),
            Pairs),
    Pairs = [tom-[bob, alice], jane-[carol, dave]].

% Single-group case — ^/2 existential, all 4 children in one bag.
% Regression guard that the existential path still works (no
% iterator pushed when there are no witnesses).
user:wam_cpp_test_bagof_single_group_unchanged :-
    catch(bagof(C, P^wam_cpp_grandparent_fixture(P, C), L),
          _, fail),
    L = [bob, alice, carol, dave].

% once/1 + forall/2 — desugared at the WAM compiler level into
% (G -> true) and \+ (G, \+ T) respectively. Inlined path uses the
% if-then-else / negation infrastructure; meta-call path (when once
% or forall appears inside a goal-term) is handled by the runtime''s
% invoke_goal_as_call dispatcher.
:- dynamic user:wam_cpp_test_once_first/0.
:- dynamic user:wam_cpp_test_once_inner_fail/0.
:- dynamic user:wam_cpp_test_once_no_backtrack/0.
:- dynamic user:wam_cpp_test_forall_all_pass/0.
:- dynamic user:wam_cpp_test_forall_some_fail/0.
:- dynamic user:wam_cpp_test_forall_empty/0.
:- dynamic user:wam_cpp_test_once_in_catch/0.
:- dynamic user:wam_cpp_test_forall_in_catch/0.
:- dynamic user:wam_cpp_test_findall_with_once/0.
:- dynamic user:wam_cpp_test_ite_in_findall_no_cut_outer/0.

% once succeeds with the first solution; subsequent solutions are
% inaccessible (no backtracking through the protected goal).
user:wam_cpp_test_once_first :-
    once(member(X, [a, b, c])),
    X = a.

% once fails when the inner goal has no solutions.
user:wam_cpp_test_once_inner_fail :-
    \+ once(fail).

% once commits to first solution — a follow-up `X = b` must fail
% because X was already bound to a.
user:wam_cpp_test_once_no_backtrack :-
    once(member(X, [a, b, c])),
    \+ X = b.

% forall succeeds when the test holds for every generator solution.
user:wam_cpp_test_forall_all_pass :-
    forall(member(_X, [1, 2, 3]), true).

% forall fails when at least one generator solution fails the test.
user:wam_cpp_test_forall_some_fail :-
    \+ forall(member(X, [1, 2, 3]), X = 1).

% forall over an empty generator succeeds trivially (vacuous truth).
user:wam_cpp_test_forall_empty :-
    forall(fail, fail).

% once inside catch — exercises the goal-term meta-call path
% (invoke_goal_as_call builds an IfThenFrame at runtime).
user:wam_cpp_test_once_in_catch :-
    catch(once(member(X, [a, b])), _, fail),
    X = a.

% forall inside catch — exercises the goal-term meta-call path
% (invoke_goal_as_call constructs \+ (G, \+ T) on the heap and
% dispatches via the negation builtin).
user:wam_cpp_test_forall_in_catch :-
    catch(forall(member(_X, [1, 2, 3]), true), _, fail).

% once inside findall — every member solution should be collected.
% Regression guard for the cut_ite fix: previously the once''s
% cut_ite would have dropped member''s iterator CP, causing only
% the first solution to be collected.
user:wam_cpp_test_findall_with_once :-
    findall(X, (member(X, [1, 2, 3]), once(true)), L),
    L = [1, 2, 3].

% Bare if-then-else inside findall — same cut_ite-preservation
% guarantee. Direct test of the fix without going through once.
user:wam_cpp_test_ite_in_findall_no_cut_outer :-
    findall(X, (member(X, [1, 2, 3]), (X > 0 -> true ; fail)), L),
    L = [1, 2, 3].

% format/1, /2, /3 — formatted output with ~-directives. /1 takes
% just a format string; /2 takes Format + Args; /3 takes Dest +
% Format + Args, where Dest selects user_output / user_error /
% atom(V) / string(V) / codes(V).
:- dynamic user:wam_cpp_test_format1/0.
:- dynamic user:wam_cpp_test_format2_w/0.
:- dynamic user:wam_cpp_test_format2_multi/0.
:- dynamic user:wam_cpp_test_format2_d/0.
:- dynamic user:wam_cpp_test_format2_a/0.
:- dynamic user:wam_cpp_test_format3_atom/0.
:- dynamic user:wam_cpp_test_format3_string/0.
:- dynamic user:wam_cpp_test_format3_codes/0.
:- dynamic user:wam_cpp_test_format3_chained/0.

user:wam_cpp_test_format1 :-
    format("plain text~n").

user:wam_cpp_test_format2_w :-
    format("X = ~w~n", [42]).

user:wam_cpp_test_format2_multi :-
    format("~w + ~w = ~w~n", [1, 2, 3]).

user:wam_cpp_test_format2_d :-
    format("~d~n", [42]).

user:wam_cpp_test_format2_a :-
    format("~a~n", [hello]).

user:wam_cpp_test_format3_atom :-
    format(atom(A), "X = ~w", [42]),
    A = 'X = 42'.

user:wam_cpp_test_format3_string :-
    format(string(S), "S = ~a", [hello]),
    S = 'S = hello'.

user:wam_cpp_test_format3_codes :-
    format(codes(C), "ab", []),
    C = [97, 98].

user:wam_cpp_test_format3_chained :-
    format(atom(A), "n=~d", [42]),
    A = 'n=42'.

% atom_codes/2, atom_chars/2, number_codes/2, atom_concat/3,
% atom_length/2, char_code/2 — bidirectional atom/number/string
% conversions. atom_codes / atom_chars / number_codes split on
% which side is bound; atom_concat is currently (+, +, ?) only;
% atom_length is +→-; char_code is bidirectional on single chars.
:- dynamic user:wam_cpp_test_atom_codes_fwd/0.
:- dynamic user:wam_cpp_test_atom_codes_rev/0.
:- dynamic user:wam_cpp_test_atom_chars_fwd/0.
:- dynamic user:wam_cpp_test_atom_chars_rev/0.
:- dynamic user:wam_cpp_test_number_codes_fwd/0.
:- dynamic user:wam_cpp_test_number_codes_rev_int/0.
:- dynamic user:wam_cpp_test_number_codes_neg/0.
:- dynamic user:wam_cpp_test_atom_concat/0.
:- dynamic user:wam_cpp_test_atom_concat_num/0.
:- dynamic user:wam_cpp_test_atom_length/0.
:- dynamic user:wam_cpp_test_atom_length_int/0.
:- dynamic user:wam_cpp_test_char_code_fwd/0.
:- dynamic user:wam_cpp_test_char_code_rev/0.
:- dynamic user:wam_cpp_test_format_atom_then_codes/0.

user:wam_cpp_test_atom_codes_fwd :-
    atom_codes(hi, C),
    C = [104, 105].

user:wam_cpp_test_atom_codes_rev :-
    atom_codes(A, [104, 105]),
    A = hi.

user:wam_cpp_test_atom_chars_fwd :-
    atom_chars(ab, Cs),
    Cs = [a, b].

user:wam_cpp_test_atom_chars_rev :-
    atom_chars(A, [a, b]),
    A = ab.

user:wam_cpp_test_number_codes_fwd :-
    number_codes(42, C),
    C = [52, 50].

user:wam_cpp_test_number_codes_rev_int :-
    number_codes(N, [52, 50]),
    N = 42.

user:wam_cpp_test_number_codes_neg :-
    number_codes(N, [45, 51]),
    N = -3.

user:wam_cpp_test_atom_concat :-
    atom_concat(foo, bar, R),
    R = foobar.

user:wam_cpp_test_atom_concat_num :-
    atom_concat(x, 42, R),
    R = 'x42'.

user:wam_cpp_test_atom_length :-
    atom_length(hello, L),
    L = 5.

user:wam_cpp_test_atom_length_int :-
    atom_length(12345, L),
    L = 5.

user:wam_cpp_test_char_code_fwd :-
    char_code(a, C),
    C = 97.

user:wam_cpp_test_char_code_rev :-
    char_code(Ch, 65),
    Ch = 'A'.

% Composition: build an atom with format/3 then take its length.
% Regression guard that format/3-built atoms behave like normal
% atoms — i.e. the Atom value is structurally identical.
user:wam_cpp_test_format_atom_then_codes :-
    format(atom(A), "~w~w", [hello, 42]),
    atom_length(A, 7),
    atom_codes(A, [104, 101, 108, 108, 111, 52, 50]).

% assertz/1, asserta/1, retract/1, retractall/1 — dynamic database
% manipulation. Facts only in this PR (rules deferred). Call/Execute
% dispatch through dynamic_db when no static label matches; each
% iteration unifies a fresh-renamed copy of the stored fact and
% pushes a CP (alt_pc=dynamic_next_clause_pc) when more remain.
:- dynamic user:wam_cpp_test_assertz_query/0.
:- dynamic user:wam_cpp_test_assertz_multi/0.
:- dynamic user:wam_cpp_test_asserta_order/0.
:- dynamic user:wam_cpp_test_retract/0.
:- dynamic user:wam_cpp_test_retract_var/0.
:- dynamic user:wam_cpp_test_retractall/0.
:- dynamic user:wam_cpp_test_retractall_empty/0.
:- dynamic user:wam_cpp_test_assertz_pair/0.
:- dynamic user:wam_cpp_test_dyn_backtrack/0.

user:wam_cpp_test_assertz_query :-
    assertz(wam_cpp_dyn_fact1(a)),
    wam_cpp_dyn_fact1(a).

user:wam_cpp_test_assertz_multi :-
    assertz(wam_cpp_dyn_item(1)),
    assertz(wam_cpp_dyn_item(2)),
    assertz(wam_cpp_dyn_item(3)),
    findall(X, wam_cpp_dyn_item(X), L),
    L = [1, 2, 3].

user:wam_cpp_test_asserta_order :-
    assertz(wam_cpp_dyn_thing(b)),
    asserta(wam_cpp_dyn_thing(a)),
    findall(X, wam_cpp_dyn_thing(X), L),
    L = [a, b].

user:wam_cpp_test_retract :-
    assertz(wam_cpp_dyn_p(1)),
    assertz(wam_cpp_dyn_p(2)),
    assertz(wam_cpp_dyn_p(3)),
    retract(wam_cpp_dyn_p(2)),
    findall(X, wam_cpp_dyn_p(X), L),
    L = [1, 3].

user:wam_cpp_test_retract_var :-
    assertz(wam_cpp_dyn_q(10)),
    retract(wam_cpp_dyn_q(X)),
    X = 10.

user:wam_cpp_test_retractall :-
    assertz(wam_cpp_dyn_r(1)),
    assertz(wam_cpp_dyn_r(2)),
    assertz(wam_cpp_dyn_r(3)),
    retractall(wam_cpp_dyn_r(_)),
    findall(X, wam_cpp_dyn_r(X), L),
    L = [].

% retractall always succeeds, even on never-asserted predicates.
user:wam_cpp_test_retractall_empty :-
    retractall(wam_cpp_dyn_absent(_)).

user:wam_cpp_test_assertz_pair :-
    assertz(wam_cpp_dyn_pair(a, 1)),
    assertz(wam_cpp_dyn_pair(b, 2)),
    findall(K-V, wam_cpp_dyn_pair(K, V), L),
    L = [a-1, b-2].

% Backtracking through dynamic clauses — the CP pushed by
% dynamic_try_next when more clauses remain drives findall''s
% next-solution loop. Filter via X > 1 to verify the iteration
% actually reaches subsequent clauses.
user:wam_cpp_test_dyn_backtrack :-
    assertz(wam_cpp_dyn_num(1)),
    assertz(wam_cpp_dyn_num(2)),
    assertz(wam_cpp_dyn_num(3)),
    findall(X, (wam_cpp_dyn_num(X), X > 1), L),
    L = [2, 3].

% sub_atom/5 — bidirectional substring search / extraction.
% Nondeterministic in modes where the substring position is not
% pre-determined; dispatch_sub_atom enumerates candidate (Before,
% Length) pairs filtered by which args are bound, then iterates
% via the SubAtomIterator + sub_atom_next_pc CP pattern.
:- dynamic user:wam_cpp_test_sub_atom_extract/0.
:- dynamic user:wam_cpp_test_sub_atom_after_computed/0.
:- dynamic user:wam_cpp_test_sub_atom_find_first/0.
:- dynamic user:wam_cpp_test_sub_atom_find_all/0.
:- dynamic user:wam_cpp_test_sub_atom_prefix/0.
:- dynamic user:wam_cpp_test_sub_atom_suffix/0.
:- dynamic user:wam_cpp_test_sub_atom_whole/0.
:- dynamic user:wam_cpp_test_sub_atom_empty/0.
:- dynamic user:wam_cpp_test_sub_atom_no_match/0.
:- dynamic user:wam_cpp_test_sub_atom_enum_all/0.

user:wam_cpp_test_sub_atom_extract :-
    sub_atom(hello, 1, 3, _, S),
    S = ell.

user:wam_cpp_test_sub_atom_after_computed :-
    sub_atom(hello, 1, 3, A, _),
    A = 1.

user:wam_cpp_test_sub_atom_find_first :-
    sub_atom(abcabc, B, L, _, b),
    B = 1, L = 1.

% Find all positions of "b" in "abcabc" — verifies that backtracking
% through the SubAtomIterator surfaces every match, not just the
% first one.
user:wam_cpp_test_sub_atom_find_all :-
    findall(B, sub_atom(abcabc, B, _, _, b), Bs),
    Bs = [1, 4].

user:wam_cpp_test_sub_atom_prefix :-
    sub_atom(hello, 0, 3, _, S),
    S = hel.

% Suffix via After=0.
user:wam_cpp_test_sub_atom_suffix :-
    sub_atom(hello, _, 3, 0, S),
    S = llo.

user:wam_cpp_test_sub_atom_whole :-
    sub_atom(abc, 0, 3, 0, S),
    S = abc.

% Empty substring is a valid match at any position; here we pin
% Before=0, Length=0 so the result is the empty atom.
user:wam_cpp_test_sub_atom_empty :-
    sub_atom(abc, 0, 0, 3, S),
    S = ''.

% No candidate (B, L) yields the requested Sub → fail.
user:wam_cpp_test_sub_atom_no_match :-
    \+ sub_atom(abc, _, _, _, xyz).

% Full enumeration of all substrings of "ab" (including empty).
% Order: row by Before, ascending Length within each row.
user:wam_cpp_test_sub_atom_enum_all :-
    findall(S, sub_atom(ab, _, _, _, S), L),
    L = ['', a, ab, '', b, ''].

% Rules in assertz/retract — extends #2136 (facts only). A rule is
% stored as a ":-/2"(Head, Body) compound; dynamic_try_next decomposes
% it on dispatch, unifies the head args with call args, and runs the
% body through a BodyFrame (flattened conjunction sequence). The
% BodyFrame mechanism replaces ConjFrame for rule bodies — ConjFrame
% nests on a shared conj_return_pc that fires the wrong frame for
% recursive predicates. ChoicePoints snapshot body_frames at push
% time so backtrack-into-a-goal''s-CP restores the surrounding
% rule body context.
:- dynamic user:wam_cpp_test_rule_simple/0.
:- dynamic user:wam_cpp_test_rule_conj/0.
:- dynamic user:wam_cpp_test_rule_mixed_facts_and_rule/0.
:- dynamic user:wam_cpp_test_rule_body_fails/0.
:- dynamic user:wam_cpp_test_rule_recursive/0.
:- dynamic user:wam_cpp_test_rule_backtrack_in_body/0.
:- dynamic user:wam_cpp_test_retract_rule/0.

user:wam_cpp_test_rule_simple :-
    assertz((wam_cpp_dyn_pos(X) :- X > 5)),
    wam_cpp_dyn_pos(10).

user:wam_cpp_test_rule_conj :-
    assertz((wam_cpp_dyn_dbl(X, Y) :- X > 0, Y is X * 2)),
    wam_cpp_dyn_dbl(3, Y),
    Y = 6.

% Mixed: 2 facts + 1 (non-recursive) rule, findall via the head.
% Note rule body is `true` to avoid infinite recursion through the
% predicate''s own clauses.
user:wam_cpp_test_rule_mixed_facts_and_rule :-
    assertz(wam_cpp_dyn_t(a, 1)),
    assertz(wam_cpp_dyn_t(b, 2)),
    assertz((wam_cpp_dyn_t(c, 99) :- true)),
    findall(K-V, wam_cpp_dyn_t(K, V), L),
    L = [a-1, b-2, c-99].

% Rule body fails on the call argument — the whole call fails.
user:wam_cpp_test_rule_body_fails :-
    assertz((wam_cpp_dyn_only_pos(X) :- X > 0)),
    \+ wam_cpp_dyn_only_pos(-3).

% Classic list-length recursion. Exercises BodyFrame chaining with
% nested recursive calls, and ChoicePoint''s saved_body_frames
% (since the recursive call''s outer body_frame must be restored
% when backtrack fires a CP from inside the recursion).
user:wam_cpp_test_rule_recursive :-
    assertz(wam_cpp_dyn_len([], 0)),
    assertz((wam_cpp_dyn_len([_|T], N) :- wam_cpp_dyn_len(T, M),
                                          N is M + 1)),
    wam_cpp_dyn_len([a, b, c], N),
    N = 3.

% Backtrack inside a rule body. A single clause whose body is
% nondeterministic (member/2 enumerates 3 solutions); findall over
% the rule collects all 3 — the body''s CPs must propagate up so
% findall''s force-backtrack re-enters them.
user:wam_cpp_test_rule_backtrack_in_body :-
    assertz((wam_cpp_dyn_pick(X) :- member(X, [1, 2, 3]))),
    findall(X, wam_cpp_dyn_pick(X), L),
    L = [1, 2, 3].

% retract a rule by its full ":-/2"(Head, Body) form — only the
% pattern-matching clause is removed; remaining clauses are intact.
user:wam_cpp_test_retract_rule :-
    assertz((wam_cpp_dyn_r(X) :- X > 5)),
    assertz((wam_cpp_dyn_r(X) :- X < 0)),
    retract((wam_cpp_dyn_r(X) :- X > 5)),
    \+ wam_cpp_dyn_r(10),
    wam_cpp_dyn_r(-1).

% Nondet retract/1 — completes the dynamic-db story. retract/1 now
% routes through the Call/Execute step arms (like findall) so its
% own CP iterator (RetractIterator) can drive backtracking through
% subsequent matches. Per ISO, retract is destructive: removal
% happens at the call (not at success of the whole query), and
% backtracking finds the next match in the post-removal database.
:- dynamic user:wam_cpp_test_retract_nondet_all/0.
:- dynamic user:wam_cpp_test_retract_nondet_pattern/0.
:- dynamic user:wam_cpp_test_retract_bind_via_pattern/0.
:- dynamic user:wam_cpp_test_retract_fail_no_match/0.
:- dynamic user:wam_cpp_test_retract_destructive/0.

% findall over retract surfaces every match in turn. Direct
% regression guard for the RetractIterator backtrack path.
user:wam_cpp_test_retract_nondet_all :-
    assertz(wam_cpp_nd_q(1)),
    assertz(wam_cpp_nd_q(2)),
    assertz(wam_cpp_nd_q(3)),
    findall(X, retract(wam_cpp_nd_q(X)), L),
    L = [1, 2, 3],
    findall(Y, wam_cpp_nd_q(Y), L2),
    L2 = [].

% Body-side filter: every clause matches the retract pattern (X
% unbound), so all 4 are retracted; findall only COLLECTS the
% ones whose body filter passes. Remaining db is empty.
user:wam_cpp_test_retract_nondet_pattern :-
    assertz(wam_cpp_nd_r(1)),
    assertz(wam_cpp_nd_r(2)),
    assertz(wam_cpp_nd_r(3)),
    assertz(wam_cpp_nd_r(4)),
    findall(X, (retract(wam_cpp_nd_r(X)), 0 is X mod 2), L),
    L = [2, 4],
    findall(Y, wam_cpp_nd_r(Y), Remaining),
    Remaining = [].

user:wam_cpp_test_retract_bind_via_pattern :-
    assertz(wam_cpp_nd_s(a, 10)),
    assertz(wam_cpp_nd_s(b, 20)),
    findall(K-V, retract(wam_cpp_nd_s(K, V)), L),
    L = [a-10, b-20].

% retract on never-asserted predicate fails (not throws).
user:wam_cpp_test_retract_fail_no_match :-
    \+ retract(wam_cpp_nd_absent(1)).

% Destructive: once retracted, gone — even on backtrack.
user:wam_cpp_test_retract_destructive :-
    assertz(wam_cpp_nd_t(1)),
    retract(wam_cpp_nd_t(1)),
    \+ wam_cpp_nd_t(1).

% nb_setval/2, nb_getval/2, b_setval/2, b_getval/2 — mutable globals.
% Stored in WamState''s nb_globals map. nb_setval REPLACES the
% CellPtr (non-backtrackable); b_setval mutates the existing cell
% via bind_cell so the trail can restore it on backtrack. Both
% setvals deep-copy the value so the stored term has fresh vars;
% getvals deep-copy on retrieval so repeated reads share structure
% but not bindings.
:- dynamic user:wam_cpp_test_nb_basic/0.
:- dynamic user:wam_cpp_test_nb_replace/0.
:- dynamic user:wam_cpp_test_nb_survives_backtrack/0.
:- dynamic user:wam_cpp_test_b_undone_on_backtrack/0.
:- dynamic user:wam_cpp_test_nb_unset_fails/0.
:- dynamic user:wam_cpp_test_nb_compound/0.
:- dynamic user:wam_cpp_test_nb_counter/0.

user:wam_cpp_test_nb_basic :-
    nb_setval(wam_cpp_nb_c1, 0),
    nb_getval(wam_cpp_nb_c1, V),
    V = 0.

user:wam_cpp_test_nb_replace :-
    nb_setval(wam_cpp_nb_c2, 1),
    nb_setval(wam_cpp_nb_c2, 2),
    nb_getval(wam_cpp_nb_c2, V),
    V = 2.

% nb_setval survives backtrack — the mutation inside the failing
% disjunct branch persists past the ;true continuation.
user:wam_cpp_test_nb_survives_backtrack :-
    nb_setval(wam_cpp_nb_c3, 10),
    (nb_setval(wam_cpp_nb_c3, 20), fail ; true),
    nb_getval(wam_cpp_nb_c3, V),
    V = 20.

% b_setval is undone on backtrack — same shape as above, but the
% inner b_setval gets rolled back when the disjunct fails.
user:wam_cpp_test_b_undone_on_backtrack :-
    nb_setval(wam_cpp_nb_c4, 10),
    (b_setval(wam_cpp_nb_c4, 20), fail ; true),
    nb_getval(wam_cpp_nb_c4, V),
    V = 10.

user:wam_cpp_test_nb_unset_fails :-
    \+ nb_getval(wam_cpp_nb_never_set, _).

user:wam_cpp_test_nb_compound :-
    nb_setval(wam_cpp_nb_c5, point(3, 4)),
    nb_getval(wam_cpp_nb_c5, V),
    V = point(3, 4).

% Counter pattern: increment-via-get-then-set. Confirms that
% repeated reads see the latest stored value (deep-copy-on-read
% doesn''t break the round-trip).
user:wam_cpp_test_nb_counter :-
    nb_setval(wam_cpp_nb_c6, 0),
    nb_getval(wam_cpp_nb_c6, V1), V2 is V1 + 1, nb_setval(wam_cpp_nb_c6, V2),
    nb_getval(wam_cpp_nb_c6, V3), V4 is V3 + 1, nb_setval(wam_cpp_nb_c6, V4),
    nb_getval(wam_cpp_nb_c6, Final),
    Final = 2.

% @</2, @=</2, @>/2, @>=/2, compare/3 — ISO §7.2 standard order of
% terms. Order: Variable @< Number @< Atom @< Compound. Numbers
% compare by value (equal-value int @< float tie-break). Atoms by
% codepoint. Compounds by arity, then name, then args lex.
:- dynamic user:wam_cpp_test_term_order_categories/0.
:- dynamic user:wam_cpp_test_term_order_numbers/0.
:- dynamic user:wam_cpp_test_term_order_atoms/0.
:- dynamic user:wam_cpp_test_term_order_arity/0.
:- dynamic user:wam_cpp_test_term_order_compound_name/0.
:- dynamic user:wam_cpp_test_term_order_compound_args/0.
:- dynamic user:wam_cpp_test_compare_lt/0.
:- dynamic user:wam_cpp_test_compare_eq/0.
:- dynamic user:wam_cpp_test_compare_gt/0.
:- dynamic user:wam_cpp_test_term_order_lte_eq/0.
:- dynamic user:wam_cpp_test_term_order_gte_eq/0.
:- dynamic user:wam_cpp_test_term_order_neg/0.

user:wam_cpp_test_term_order_categories :-
    X @< 1,
    1 @< foo,
    foo @< foo(1).

user:wam_cpp_test_term_order_numbers :-
    1 @< 2,
    1.5 @< 2.5,
    3 @< 3.5.

user:wam_cpp_test_term_order_atoms :-
    a @< b,
    abc @< abd,
    abc @< abcd.

% Standard order puts foo/1 @< foo/2 @< foo/3 — by arity first.
% String compare would put "foo/10" @< "foo/2" lexicographically,
% which is wrong; the ISO impl correctly orders by arity number.
user:wam_cpp_test_term_order_arity :-
    foo(1) @< foo(1, 2),
    foo(1, 2) @< foo(1, 2, 3).

user:wam_cpp_test_term_order_compound_name :-
    a(1) @< b(1),
    aa(1) @< ab(1).

user:wam_cpp_test_term_order_compound_args :-
    foo(1) @< foo(2),
    foo(a, 1) @< foo(a, 2).

user:wam_cpp_test_compare_lt :-
    compare(C, 1, 2), C = (<).

user:wam_cpp_test_compare_eq :-
    compare(C, foo, foo), C = (=).

user:wam_cpp_test_compare_gt :-
    compare(C, 5, 3), C = (>).

user:wam_cpp_test_term_order_lte_eq :-
    1 @=< 1.

user:wam_cpp_test_term_order_gte_eq :-
    foo @>= foo.

user:wam_cpp_test_term_order_neg :-
    \+ (2 @< 1),
    \+ (foo @< abc).

% char_type/2 + upcase_atom/2 + downcase_atom/2 — character
% classification and case conversion. char_type covers a useful
% subset of SWI''s patterns: alpha, alnum, digit, whitespace, space,
% punct, ascii, upper, lower (classifications); upper(Lower),
% lower(Upper), to_upper(U), to_lower(L) (case conversion);
% digit(Weight), code(Code) (bidirectional).
%
% Digit-char atoms like ''5'' are constructed via char_code/2 to
% side-step a WAM-text roundtrip quirk where digit-only atoms get
% re-parsed as integers — orthogonal bug, deferred.
:- dynamic user:wam_cpp_test_char_type_alpha/0.
:- dynamic user:wam_cpp_test_char_type_digit/0.
:- dynamic user:wam_cpp_test_char_type_whitespace/0.
:- dynamic user:wam_cpp_test_char_type_upper_arg/0.
:- dynamic user:wam_cpp_test_char_type_lower_arg/0.
:- dynamic user:wam_cpp_test_char_type_to_upper/0.
:- dynamic user:wam_cpp_test_char_type_to_lower/0.
:- dynamic user:wam_cpp_test_char_type_digit_weight/0.
:- dynamic user:wam_cpp_test_char_type_digit_reverse/0.
:- dynamic user:wam_cpp_test_char_type_code/0.
:- dynamic user:wam_cpp_test_upcase_atom/0.
:- dynamic user:wam_cpp_test_downcase_atom/0.

user:wam_cpp_test_char_type_alpha :-
    char_type(a, alpha),
    char_code(C, 0'1), \+ char_type(C, alpha).

user:wam_cpp_test_char_type_digit :-
    char_code(C, 0'5),
    char_type(C, digit).

user:wam_cpp_test_char_type_whitespace :-
    char_type(' ', whitespace).

user:wam_cpp_test_char_type_upper_arg :-
    char_type('A', upper(L)),
    L = a.

user:wam_cpp_test_char_type_lower_arg :-
    char_type(a, lower(U)),
    U = 'A'.

user:wam_cpp_test_char_type_to_upper :-
    char_type(a, to_upper(U)),
    U = 'A',
    char_type('B', to_upper(U2)),
    U2 = 'B'.

user:wam_cpp_test_char_type_to_lower :-
    char_type('A', to_lower(L)),
    L = a.

user:wam_cpp_test_char_type_digit_weight :-
    char_code(D, 0'7),
    char_type(D, digit(W)),
    W = 7.

user:wam_cpp_test_char_type_digit_reverse :-
    char_type(C, digit(3)),
    char_code(C, 0'3).

user:wam_cpp_test_char_type_code :-
    char_type('A', code(C)),
    C = 65,
    char_type(C2, code(97)),
    C2 = a.

user:wam_cpp_test_upcase_atom :-
    upcase_atom(hello, R),
    R = 'HELLO',
    upcase_atom('Already', R2),
    R2 = 'ALREADY'.

user:wam_cpp_test_downcase_atom :-
    downcase_atom('WORLD', R),
    R = world.

% numlist/3, sort/2, msort/2, select/3 — common list utilities.
% numlist/sort/msort are direct builtins (deterministic). select/3
% is helper-injected (nondet via try_me_else/trust_me) so
% findall/select enumerates every (Elem, Rest) pair.
:- dynamic user:wam_cpp_test_numlist/0.
:- dynamic user:wam_cpp_test_numlist_empty/0.
:- dynamic user:wam_cpp_test_numlist_single/0.
:- dynamic user:wam_cpp_test_sort/0.
:- dynamic user:wam_cpp_test_sort_mixed/0.
:- dynamic user:wam_cpp_test_sort_empty/0.
:- dynamic user:wam_cpp_test_msort/0.
:- dynamic user:wam_cpp_test_select_bound/0.
:- dynamic user:wam_cpp_test_select_all/0.
:- dynamic user:wam_cpp_test_select_missing/0.
:- dynamic user:wam_cpp_test_numlist_then_sort/0.

user:wam_cpp_test_numlist :-
    numlist(1, 5, L),
    L = [1, 2, 3, 4, 5].

user:wam_cpp_test_numlist_empty :-
    numlist(5, 3, L),
    L = [].

user:wam_cpp_test_numlist_single :-
    numlist(7, 7, L),
    L = [7].

user:wam_cpp_test_sort :-
    sort([3, 1, 2, 1, 3], L),
    L = [1, 2, 3].

% sort across categories — exercises standard_order_cmp.
user:wam_cpp_test_sort_mixed :-
    sort([foo, 1, bar, 2], L),
    L = [1, 2, bar, foo].

user:wam_cpp_test_sort_empty :-
    sort([], L),
    L = [].

% msort keeps duplicates and is stable.
user:wam_cpp_test_msort :-
    msort([3, 1, 2, 1, 3], L),
    L = [1, 1, 2, 3, 3].

user:wam_cpp_test_select_bound :-
    select(b, [a, b, c], R),
    R = [a, c].

% select/3 nondet via findall — every (Elem, Rest) pair surfaced.
user:wam_cpp_test_select_all :-
    findall(X-R, select(X, [a, b, c], R), L),
    L = [a-[b,c], b-[a,c], c-[a,b]].

user:wam_cpp_test_select_missing :-
    \+ select(z, [a, b, c], _).

% Composition guard: numlist builds the list, sort dedups it.
user:wam_cpp_test_numlist_then_sort :-
    numlist(1, 4, L),
    sort([3, 2, 1, 4, 2 | L], S),
    S = [1, 2, 3, 4].

% maplist/2..5 — apply a goal to each list element. Helper-injected
% as the standard recursive 2-clause Prolog definition; the body
% uses call/N to dispatch the goal with the threaded args. Variadic
% across 2..5 arities: maplist(G, L1, L2) calls G(Xi, Yi) for each
% (Xi, Yi) pair, etc.
:- dynamic user:wam_cpp_dyn_pos/1.
:- dynamic user:wam_cpp_dyn_pos2/1.
:- dynamic user:wam_cpp_test_maplist_2_check/0.
:- dynamic user:wam_cpp_test_maplist_2_fail/0.
:- dynamic user:wam_cpp_test_maplist_3_map/0.
:- dynamic user:wam_cpp_test_maplist_3_succ/0.
:- dynamic user:wam_cpp_test_maplist_3_empty/0.
:- dynamic user:wam_cpp_test_maplist_4/0.
:- dynamic user:wam_cpp_test_maplist_5/0.

user:wam_cpp_plus10(X, Y) :- Y is X + 10.
user:wam_cpp_add(X, Y, Z) :- Z is X + Y.
user:wam_cpp_add3(X, Y, Z, W) :- W is X + Y + Z.

user:wam_cpp_test_maplist_2_check :-
    assertz((wam_cpp_dyn_pos(X) :- X > 0)),
    maplist(wam_cpp_dyn_pos, [1, 2, 3]).

user:wam_cpp_test_maplist_2_fail :-
    assertz((wam_cpp_dyn_pos2(X) :- X > 0)),
    \+ maplist(wam_cpp_dyn_pos2, [1, -1, 3]).

user:wam_cpp_test_maplist_3_map :-
    maplist(wam_cpp_plus10, [1, 2, 3], L),
    L = [11, 12, 13].

user:wam_cpp_test_maplist_3_succ :-
    maplist(succ, [1, 2, 3], L),
    L = [2, 3, 4].

user:wam_cpp_test_maplist_3_empty :-
    maplist(wam_cpp_plus10, [], L),
    L = [].

user:wam_cpp_test_maplist_4 :-
    maplist(wam_cpp_add, [1, 2, 3], [10, 20, 30], L),
    L = [11, 22, 33].

user:wam_cpp_test_maplist_5 :-
    maplist(wam_cpp_add3, [1, 2], [10, 20], [100, 200], L),
    L = [111, 222].

% include/3, exclude/3, partition/4, foldl/4, foldl/5 — filter
% and fold meta-predicates. Helper-injected as 3-clause Prolog
% definitions (try_me_else/retry_me_else/trust_me); clause 2
% includes when the goal succeeds, clause 3 includes when \+ goal
% succeeds. Mutually exclusive — the extra try_me_else CP is
% harmless. foldl/4 and foldl/5 are 2-clause (base case + recursive).
:- dynamic user:wam_cpp_test_include/0.
:- dynamic user:wam_cpp_test_include_empty/0.
:- dynamic user:wam_cpp_test_exclude/0.
:- dynamic user:wam_cpp_test_partition/0.
:- dynamic user:wam_cpp_test_foldl4_sum/0.
:- dynamic user:wam_cpp_test_foldl4_empty/0.
:- dynamic user:wam_cpp_test_foldl5/0.

user:wam_cpp_gt0(X) :- X > 0.
user:wam_cpp_addc(X, Acc, Sum) :- Sum is Acc + X.
user:wam_cpp_join4(X, Y, Acc, Out) :- Out is Acc + X + Y.

user:wam_cpp_test_include :-
    include(wam_cpp_gt0, [1, -2, 3, -4, 5], L),
    L = [1, 3, 5].

user:wam_cpp_test_include_empty :-
    include(wam_cpp_gt0, [], L),
    L = [].

user:wam_cpp_test_exclude :-
    exclude(wam_cpp_gt0, [1, -2, 3, -4, 5], L),
    L = [-2, -4].

user:wam_cpp_test_partition :-
    partition(wam_cpp_gt0, [1, -2, 3, -4, 5], In, Ex),
    In = [1, 3, 5],
    Ex = [-2, -4].

% foldl/4 left fold: sum a list with accumulator.
user:wam_cpp_test_foldl4_sum :-
    foldl(wam_cpp_addc, [1, 2, 3, 4, 5], 0, Sum),
    Sum = 15.

user:wam_cpp_test_foldl4_empty :-
    foldl(wam_cpp_addc, [], 100, V),
    V = 100.

% foldl/5: parallel-list fold (X + Y added each step).
user:wam_cpp_test_foldl5 :-
    foldl(wam_cpp_join4, [1, 2, 3], [10, 20, 30], 0, Sum),
    Sum = 66.

% keysort/2 + pairs_keys/2 + pairs_values/2 + pairs_keys_values/3 —
% common idioms for sorting / decomposing Key-Value pair lists.
% keysort is a direct builtin using standard_order_cmp on the pair''s
% Key (args[0] of -/2). The pairs_* builtins are helper-injected
% recursive Prolog defs.
:- dynamic user:wam_cpp_test_keysort/0.
:- dynamic user:wam_cpp_test_keysort_stable/0.
:- dynamic user:wam_cpp_test_keysort_empty/0.
:- dynamic user:wam_cpp_test_pairs_keys/0.
:- dynamic user:wam_cpp_test_pairs_keys_empty/0.
:- dynamic user:wam_cpp_test_pairs_values/0.
:- dynamic user:wam_cpp_test_pairs_kv/0.
:- dynamic user:wam_cpp_test_keysort_then_values/0.

user:wam_cpp_test_keysort :-
    keysort([b-2, a-1, c-3], L),
    L = [a-1, b-2, c-3].

% Stable: equal keys preserve original Value order.
user:wam_cpp_test_keysort_stable :-
    keysort([b-1, a-2, b-3, a-4], L),
    L = [a-2, a-4, b-1, b-3].

user:wam_cpp_test_keysort_empty :-
    keysort([], L),
    L = [].

user:wam_cpp_test_pairs_keys :-
    pairs_keys([a-1, b-2, c-3], K),
    K = [a, b, c].

user:wam_cpp_test_pairs_keys_empty :-
    pairs_keys([], K),
    K = [].

user:wam_cpp_test_pairs_values :-
    pairs_values([a-1, b-2, c-3], V),
    V = [1, 2, 3].

user:wam_cpp_test_pairs_kv :-
    pairs_keys_values([a-1, b-2, c-3], K, V),
    K = [a, b, c],
    V = [1, 2, 3].

% Composition: sort by key, then pull values.
user:wam_cpp_test_keysort_then_values :-
    keysort([c-30, a-10, b-20], Sorted),
    pairs_values(Sorted, Vs),
    Vs = [10, 20, 30].

% WAM-text quote roundtrip for digit-only atoms — `''5''`, `''42''`,
% `''-3''` etc. Previously these were emitted unquoted in WAM text
% and the C++ value emitter''s cpp_value_literal re-parsed them as
% integers, so the runtime saw Integer where the source had Atom.
% Now quote_wam_constant prefixes a \\x01 atom-marker (inside the
% quotes) and cpp_value_literal recognises the marker, stripping it
% and emitting Value::Atom regardless of whether the content
% re-parses as a number.
:- dynamic user:wam_cpp_test_digit_atom_is_atom/0.
:- dynamic user:wam_cpp_test_digit_atom_not_integer/0.
:- dynamic user:wam_cpp_test_digit_atom_codes/0.
:- dynamic user:wam_cpp_test_digit_atom_length/0.
:- dynamic user:wam_cpp_test_digit_integer_unchanged/0.
:- dynamic user:wam_cpp_test_char_type_digit_direct/0.
:- dynamic user:wam_cpp_test_negative_atom/0.

user:wam_cpp_test_digit_atom_is_atom :-
    X = '5',
    atom(X).

user:wam_cpp_test_digit_atom_not_integer :-
    X = '5',
    \+ integer(X).

user:wam_cpp_test_digit_atom_codes :-
    % The atom ''5'' has codes [53] (ASCII ''5''). Pre-fix, it was
    % the integer 5 and atom_codes failed (or coerced).
    atom_codes('5', C),
    C = [53].

user:wam_cpp_test_digit_atom_length :-
    atom_length('5', L),
    L = 1.

% Integers (without quotes) stay as integers.
user:wam_cpp_test_digit_integer_unchanged :-
    X = 5,
    integer(X),
    \+ atom(X).

% Now that digit-atoms round-trip correctly, char_type accepts
% them directly (no char_code/2 workaround needed).
user:wam_cpp_test_char_type_digit_direct :-
    char_type('5', digit),
    char_type('7', digit(W)),
    W = 7.

% Negative-number-looking atoms.
user:wam_cpp_test_negative_atom :-
    X = '-3',
    atom(X),
    \+ integer(X),
    atom_length(X, 2).

% I/O polish — print/1, display/1, tab/1, write_canonical/1.
% writeln/1 already existed; just adding tests here. write/1 + nl/0
% are existing.
:- dynamic user:wam_cpp_test_print/0.
:- dynamic user:wam_cpp_test_display/0.
:- dynamic user:wam_cpp_test_tab/0.
:- dynamic user:wam_cpp_test_tab_zero/0.
:- dynamic user:wam_cpp_test_tab_neg/0.
:- dynamic user:wam_cpp_test_wc_simple/0.
:- dynamic user:wam_cpp_test_wc_quoted/0.
:- dynamic user:wam_cpp_test_wc_digit_atom/0.

user:wam_cpp_test_print :- print(hello), nl.
user:wam_cpp_test_display :- display(bar), nl.
user:wam_cpp_test_tab :- tab(3), write(x), nl.
user:wam_cpp_test_tab_zero :- tab(0), write(y), nl.

% Negative N → fail.
user:wam_cpp_test_tab_neg :- \+ tab(-1).

user:wam_cpp_test_wc_simple :- write_canonical(hello), nl.
user:wam_cpp_test_wc_quoted :- write_canonical('hello world'), nl.
% Digit-only atom — gets quoted in canonical form, distinguishing
% it from an integer.
user:wam_cpp_test_wc_digit_atom :- write_canonical('5'), nl.

% with_output_to/2 — capture write/format output into an atom,
% string, or codes list. Pushes an OutputCaptureFrame; while the
% frame is on top, all I/O builtins route through emit_output()
% which appends to the frame''s buffer instead of writing stdout.
% On goal success, OutputCaptureReturn pops the frame and unifies
% the buffer with the sink (atom/string/codes per the sink shape).
:- dynamic user:wam_cpp_test_wot_basic/0.
:- dynamic user:wam_cpp_test_wot_multi/0.
:- dynamic user:wam_cpp_test_wot_format/0.
:- dynamic user:wam_cpp_test_wot_string/0.
:- dynamic user:wam_cpp_test_wot_codes/0.
:- dynamic user:wam_cpp_test_wot_empty/0.
:- dynamic user:wam_cpp_test_wot_goal_fails/0.
:- dynamic user:wam_cpp_test_wot_tab/0.
:- dynamic user:wam_cpp_test_wot_nested/0.

user:wam_cpp_test_wot_basic :-
    with_output_to(atom(A), write(hello)),
    A = hello.

% Multi-write via conjunction.
user:wam_cpp_test_wot_multi :-
    with_output_to(atom(A), (write(foo), write(bar))),
    A = foobar.

user:wam_cpp_test_wot_format :-
    with_output_to(atom(A), format("X = ~w", [42])),
    A = 'X = 42'.

user:wam_cpp_test_wot_string :-
    with_output_to(string(S), write(test)),
    S = test.

user:wam_cpp_test_wot_codes :-
    with_output_to(codes(C), write(ab)),
    C = [97, 98].

user:wam_cpp_test_wot_empty :-
    with_output_to(atom(A), true),
    A = ''.

user:wam_cpp_test_wot_goal_fails :-
    \+ with_output_to(atom(_), fail).

% tab routes through emit_output too.
user:wam_cpp_test_wot_tab :-
    with_output_to(atom(A), (write(x), tab(3), write(y))),
    A = 'x   y'.

% Nested capture: inner frame intercepts only its own goal''s
% output; outer continues to capture the rest. Regression guard
% for the saved_cp handling via invoke_goal_as_call dispatch.
user:wam_cpp_test_wot_nested :-
    with_output_to(atom(Outer),
        (write(a),
         with_output_to(atom(Inner), write(b)),
         write(c))),
    Outer = ac,
    Inner = b.

% split_string/4 — separator-based splitting with optional pad-char
% stripping. Walks the input left-to-right, splitting on any char in
% SepChars; adjacent separators produce empty substrings. After
% splitting, each substring has leading/trailing chars in PadChars
% stripped. Empty SepChars = "no splits, just pad the whole input".
:- dynamic user:wam_cpp_test_split_simple/0.
:- dynamic user:wam_cpp_test_split_empty/0.
:- dynamic user:wam_cpp_test_split_single/0.
:- dynamic user:wam_cpp_test_split_double_sep/0.
:- dynamic user:wam_cpp_test_split_pad/0.
:- dynamic user:wam_cpp_test_split_sep_and_pad/0.
:- dynamic user:wam_cpp_test_split_multi_sep/0.
:- dynamic user:wam_cpp_test_split_pad_trailing/0.
:- dynamic user:wam_cpp_test_split_atom_input/0.

user:wam_cpp_test_split_simple :-
    split_string("a,b,c", ",", "", L),
    L = ["a","b","c"].

user:wam_cpp_test_split_empty :-
    split_string("", ",", "", L),
    L = [""].

user:wam_cpp_test_split_single :-
    split_string("hello", ",", "", L),
    L = ["hello"].

user:wam_cpp_test_split_double_sep :-
    split_string("a,,b", ",", "", L),
    L = ["a","","b"].

% No separators — just pad the whole input.
user:wam_cpp_test_split_pad :-
    split_string(" hello ", "", " ", L),
    L = ["hello"].

user:wam_cpp_test_split_sep_and_pad :-
    split_string("a , b , c", ",", " ", L),
    L = ["a","b","c"].

% Multiple separator chars in the set.
user:wam_cpp_test_split_multi_sep :-
    split_string("a,b;c,d", ",;", "", L),
    L = ["a","b","c","d"].

user:wam_cpp_test_split_pad_trailing :-
    split_string("abc   ", "", " ", L),
    L = ["abc"].

% Atom input works too (atom ≡ string in this runtime).
user:wam_cpp_test_split_atom_input :-
    split_string('a,b,c', ",", "", L),
    L = ["a","b","c"].

% term_to_atom/2 — bidirectional canonical-form serialisation.
% Forward (+Term, ?Atom): render Term and unify Atom. Reverse
% (-Term, +Atom): parse Atom via the canonical-form reader and
% unify Term. Variables in the parsed term are fresh.
:- dynamic user:wam_cpp_test_tta_fwd_atom/0.
:- dynamic user:wam_cpp_test_tta_fwd_int/0.
:- dynamic user:wam_cpp_test_tta_fwd_compound/0.
:- dynamic user:wam_cpp_test_tta_fwd_list/0.
:- dynamic user:wam_cpp_test_tta_rev_atom/0.
:- dynamic user:wam_cpp_test_tta_rev_int/0.
:- dynamic user:wam_cpp_test_tta_rev_neg/0.
:- dynamic user:wam_cpp_test_tta_rev_float/0.
:- dynamic user:wam_cpp_test_tta_rev_compound/0.
:- dynamic user:wam_cpp_test_tta_rev_list/0.
:- dynamic user:wam_cpp_test_tta_rev_nested/0.
:- dynamic user:wam_cpp_test_tta_roundtrip/0.

user:wam_cpp_test_tta_fwd_atom :-
    term_to_atom(hello, A),
    A = hello.

user:wam_cpp_test_tta_fwd_int :-
    term_to_atom(42, A),
    A = '42'.

user:wam_cpp_test_tta_fwd_compound :-
    term_to_atom(foo(1, bar), A),
    A = 'foo(1, bar)'.

user:wam_cpp_test_tta_fwd_list :-
    term_to_atom([a, b, c], A),
    A = '[a, b, c]'.

user:wam_cpp_test_tta_rev_atom :-
    term_to_atom(T, hello),
    T = hello.

user:wam_cpp_test_tta_rev_int :-
    term_to_atom(T, '42'),
    T = 42.

user:wam_cpp_test_tta_rev_neg :-
    term_to_atom(T, '-7'),
    T = -7.

user:wam_cpp_test_tta_rev_float :-
    term_to_atom(T, '3.14'),
    T = 3.14.

user:wam_cpp_test_tta_rev_compound :-
    term_to_atom(T, 'foo(1, bar)'),
    T = foo(1, bar).

user:wam_cpp_test_tta_rev_list :-
    term_to_atom(T, '[a, b, c]'),
    T = [a, b, c].

% Nested compound containing a list.
user:wam_cpp_test_tta_rev_nested :-
    term_to_atom(T, 'p(q(1), [2, 3])'),
    T = p(q(1), [2, 3]).

% Full round-trip: render then re-parse yields a structurally
% equivalent term.
user:wam_cpp_test_tta_roundtrip :-
    term_to_atom(foo(a, b, c), A),
    term_to_atom(T, A),
    T = foo(a, b, c).

% read/1 + read_term/1 — stdin term reading using the canonical-form
% parser from #2189. Terminator is `.` followed by whitespace or EOF.
% Pre-EOF empty input yields atom `end_of_file` (per ISO).
:- dynamic user:wam_cpp_test_read_atom/0.
:- dynamic user:wam_cpp_test_read_int/0.
:- dynamic user:wam_cpp_test_read_compound/0.
:- dynamic user:wam_cpp_test_read_list/0.
:- dynamic user:wam_cpp_test_read_eof/0.
:- dynamic user:wam_cpp_test_read_term_atom/0.

user:wam_cpp_test_read_atom :-
    read(T),
    T = hello.

user:wam_cpp_test_read_int :-
    read(T),
    T = 42.

user:wam_cpp_test_read_compound :-
    read(T),
    T = foo(1, bar).

user:wam_cpp_test_read_list :-
    read(T),
    T = [a, b, c].

user:wam_cpp_test_read_eof :-
    read(T),
    T = end_of_file.

user:wam_cpp_test_read_term_atom :-
    read_term(T),
    T = just_an_atom.

% get_char/1, get_code/1, peek_char/1, put_char/1, put_code/1 —
% single-char I/O. EOF behaviour: get_char/peek_char → atom
% end_of_file; get_code → -1. put_* routes through emit_output_char
% so with_output_to/2 captures the bytes.
:- dynamic user:wam_cpp_test_get_char/0.
:- dynamic user:wam_cpp_test_get_code/0.
:- dynamic user:wam_cpp_test_get_char_eof/0.
:- dynamic user:wam_cpp_test_get_code_eof/0.
:- dynamic user:wam_cpp_test_peek_char/0.
:- dynamic user:wam_cpp_test_put_char/0.
:- dynamic user:wam_cpp_test_put_code/0.
:- dynamic user:wam_cpp_test_put_in_capture/0.

user:wam_cpp_test_get_char :- get_char(C), C = h.

user:wam_cpp_test_get_code :- get_code(C), C = 104.

user:wam_cpp_test_get_char_eof :-
    get_char(C),
    C = end_of_file.

user:wam_cpp_test_get_code_eof :-
    get_code(C),
    C = -1.

% peek_char/1 doesn''t consume the byte — the subsequent get_char/1
% sees the same character.
user:wam_cpp_test_peek_char :-
    peek_char(C1),
    get_char(C2),
    C1 = a,
    C2 = a.

user:wam_cpp_test_put_char :-
    put_char(a), put_char(b), put_char(c), nl.

user:wam_cpp_test_put_code :-
    put_code(72), put_code(105), nl.  % "Hi\n"

% Capture put_char + put_code via with_output_to/2 — confirms
% they route through emit_output_char (not direct stdout).
user:wam_cpp_test_put_in_capture :-
    with_output_to(atom(A), (put_char(x), put_code(89))),
    A = 'xY'.

% atomic_list_concat/2, /3 — common join/split idiom.
% atom_string/2, string_concat/3, string_length/2 — atom ≡ string
% in this runtime; the string_* names are aliases.
% number_chars/2 — parallel to number_codes/2 with single-char atoms.
% atom_to_term/3 — atom_to_term(+Atom, -Term, -Bindings); Bindings is
% always [] since the parser doesn''t track source variable names.
:- dynamic user:wam_cpp_test_alc2/0.
:- dynamic user:wam_cpp_test_alc2_mixed/0.
:- dynamic user:wam_cpp_test_alc2_empty/0.
:- dynamic user:wam_cpp_test_alc3_join/0.
:- dynamic user:wam_cpp_test_alc3_split/0.
:- dynamic user:wam_cpp_test_alc3_split_multi/0.
:- dynamic user:wam_cpp_test_alc3_split_nosep/0.
:- dynamic user:wam_cpp_test_atom_string_fwd/0.
:- dynamic user:wam_cpp_test_atom_string_rev/0.
:- dynamic user:wam_cpp_test_string_length_alias/0.
:- dynamic user:wam_cpp_test_string_concat_alias/0.
:- dynamic user:wam_cpp_test_number_chars_fwd/0.
:- dynamic user:wam_cpp_test_number_chars_rev/0.
:- dynamic user:wam_cpp_test_atom_to_term/0.

user:wam_cpp_test_alc2 :-
    atomic_list_concat([a, b, c], R),
    R = abc.

user:wam_cpp_test_alc2_mixed :-
    atomic_list_concat([foo, 1, bar], R),
    R = 'foo1bar'.

user:wam_cpp_test_alc2_empty :-
    atomic_list_concat([], R),
    R = ''.

user:wam_cpp_test_alc3_join :-
    atomic_list_concat([a, b, c], '-', R),
    R = 'a-b-c'.

% /3 reverse mode: split.
user:wam_cpp_test_alc3_split :-
    atomic_list_concat(L, '-', 'a-b-c'),
    L = [a, b, c].

% Multi-char separator.
user:wam_cpp_test_alc3_split_multi :-
    atomic_list_concat(L, '::', 'foo::bar::baz'),
    L = [foo, bar, baz].

% No separator in input → single-element list.
user:wam_cpp_test_alc3_split_nosep :-
    atomic_list_concat(L, '-', 'nosep'),
    L = [nosep].

user:wam_cpp_test_atom_string_fwd :-
    atom_string(hello, S),
    S = hello.

user:wam_cpp_test_atom_string_rev :-
    atom_string(A, world),
    A = world.

user:wam_cpp_test_string_length_alias :-
    string_length(hello, N),
    N = 5.

user:wam_cpp_test_string_concat_alias :-
    string_concat(foo, bar, R),
    R = foobar.

user:wam_cpp_test_number_chars_fwd :-
    number_chars(42, C),
    C = ['4', '2'].

user:wam_cpp_test_number_chars_rev :-
    number_chars(N, ['4', '2']),
    N = 42.

user:wam_cpp_test_atom_to_term :-
    atom_to_term('foo(1, bar)', T, B),
    T = foo(1, bar),
    B = [].

% succ/2 + between/3 fixtures. succ is a direct bidirectional builtin;
% between is helper-injected and exercises the nondet path via findall.
:- dynamic user:wam_cpp_test_succ_fwd/0.
:- dynamic user:wam_cpp_test_succ_bwd/0.
:- dynamic user:wam_cpp_test_succ_zero/0.
:- dynamic user:wam_cpp_test_succ_neg_fail/0.
:- dynamic user:wam_cpp_test_succ_y_zero_fail/0.
:- dynamic user:wam_cpp_test_between_first/0.
:- dynamic user:wam_cpp_test_between_enum/0.
:- dynamic user:wam_cpp_test_between_singleton/0.
:- dynamic user:wam_cpp_test_between_empty/0.

user:wam_cpp_test_succ_fwd          :- succ(3, X), X = 4.
user:wam_cpp_test_succ_bwd          :- succ(X, 4), X = 3.
user:wam_cpp_test_succ_zero         :- succ(0, X), X = 1.
user:wam_cpp_test_succ_neg_fail     :- succ(-1, _).
user:wam_cpp_test_succ_y_zero_fail  :- succ(_, 0).
user:wam_cpp_test_between_first     :- between(1, 5, X), X = 1.
user:wam_cpp_test_between_enum      :- findall(X, between(1, 3, X), L), L = [1, 2, 3].
user:wam_cpp_test_between_singleton :- findall(X, between(5, 5, X), L), L = [5].
user:wam_cpp_test_between_empty     :- findall(X, between(5, 3, X), L), L = [].

% Indexing-instruction fixtures (switch_on_constant / switch_on_term):
:- dynamic user:wam_cpp_color/1.
:- dynamic user:wam_cpp_shape/2.
:- dynamic user:wam_cpp_mixed/1.
:- dynamic user:wam_cpp_listy/1.

user:wam_cpp_color(red).
user:wam_cpp_color(green).
user:wam_cpp_color(blue).
user:wam_cpp_shape(circle,   round).
user:wam_cpp_shape(square,   angular).
user:wam_cpp_shape(triangle, angular).
user:wam_cpp_mixed(a).
user:wam_cpp_mixed(1).
user:wam_cpp_mixed(foo(x)).
user:wam_cpp_listy([]).
user:wam_cpp_listy([_|_]).

% A2-dispatch fixture — multi-clause predicate where the dispatch
% constant lives in A2, not A1. The compiler emits
% switch_on_constant_a2 (a runtime no-op that historically was
% emitted as a C++ comment instead of a real push_back, leaving
% downstream labels off-by-one). The downstream predicate
% wam_cpp_after_a2/0 catches that regression: if its label PC is
% off, calling it crashes or jumps mid-predicate.
:- dynamic user:wam_cpp_a2dispatch/4.
:- dynamic user:wam_cpp_after_a2/0.
:- dynamic user:wam_cpp_test_a2_all_clauses/0.
:- dynamic user:wam_cpp_test_a2_downstream_label/0.
:- dynamic user:wam_cpp_a2tag/3.
:- dynamic user:wam_cpp_test_a2_direct_jump/0.
:- dynamic user:wam_cpp_test_a2_no_match_fails/0.
:- dynamic user:wam_cpp_test_a2_unbound_enumerates/0.
:- dynamic user:wam_cpp_test_a2_shared_key_backtracks/0.

user:wam_cpp_a2dispatch(B, same, K, t(same, B, K)).
user:wam_cpp_a2dispatch(=, grew, K, t(grew_eq, K)).
user:wam_cpp_a2dispatch(<, grew, K, t(grew_lt, K)).
user:wam_cpp_a2dispatch(>, grew, K, t(grew_gt, K)).

user:wam_cpp_after_a2 :- true.

user:wam_cpp_test_a2_all_clauses :-
    user:wam_cpp_a2dispatch(=, same, k1, T1), T1 = t(same, =, k1),
    user:wam_cpp_a2dispatch(=, grew, k2, T2), T2 = t(grew_eq, k2),
    user:wam_cpp_a2dispatch(<, grew, k3, T3), T3 = t(grew_lt, k3),
    user:wam_cpp_a2dispatch(>, grew, k4, T4), T4 = t(grew_gt, k4).

user:wam_cpp_test_a2_downstream_label :-
    user:wam_cpp_a2dispatch(=, same, k, _),
    user:wam_cpp_after_a2.

% Tag-style predicate where A1 is a free name and A2 is the constant
% the runtime indexes on. Distinct A2 values per clause exercise the
% O(1) direct-jump path of the new SwitchOnConstantA2 opcode; shared
% keys exercise the indexed_entry + retry chain.
user:wam_cpp_a2tag(_, ok,    info).
user:wam_cpp_a2tag(_, warn,  yellow).
user:wam_cpp_a2tag(_, error, red).
user:wam_cpp_a2tag(_, fatal, red).

user:wam_cpp_test_a2_direct_jump :-
    wam_cpp_a2tag(_, warn, yellow),
    wam_cpp_a2tag(any, error, red),
    wam_cpp_a2tag(thing, ok, info).
user:wam_cpp_test_a2_no_match_fails :-
    % Bound A2 with no matching clause must fail without enumerating.
    \+ wam_cpp_a2tag(_, no_such_tag, _).
user:wam_cpp_test_a2_unbound_enumerates :-
    % Unbound A2 → fall through to try_me_else chain; findall sees
    % all clauses.
    findall(T-C, wam_cpp_a2tag(_, T, C), L),
    L = [ok-info, warn-yellow, error-red, fatal-red].
user:wam_cpp_test_a2_shared_key_backtracks :-
    % A2=red appears in two clauses (error + fatal). Direct jump
    % lands on error; backtracking through the synthesized CP must
    % reach fatal.
    findall(T, wam_cpp_a2tag(_, T, red), L),
    L = [error, fatal].

% A2 = all-compound (different functors). Exercises switch_on_structure_a2.
% A1 is variable in every clause, so first-arg indexing is skipped.
:- dynamic user:wam_cpp_a2struct/2.
:- dynamic user:wam_cpp_test_a2_structure_direct/0.
:- dynamic user:wam_cpp_test_a2_structure_miss/0.
:- dynamic user:wam_cpp_test_a2_structure_unbound/0.

user:wam_cpp_a2struct(_, foo(red)).
user:wam_cpp_a2struct(_, bar(green, leaf)).
user:wam_cpp_a2struct(_, baz(blue, 1, ok)).

user:wam_cpp_test_a2_structure_direct :-
    wam_cpp_a2struct(any, foo(red)),
    wam_cpp_a2struct(any, bar(green, leaf)),
    wam_cpp_a2struct(any, baz(blue, 1, ok)).
user:wam_cpp_test_a2_structure_miss :-
    % qux/1 is not in the table; bound A2 with no functor match
    % fails fast at the switch.
    \+ wam_cpp_a2struct(any, qux(1)),
    % Wrong arity for foo also fails (foo/2 doesn't exist).
    \+ wam_cpp_a2struct(any, foo(1, 2)).
user:wam_cpp_test_a2_structure_unbound :-
    % Unbound A2 → switch falls through to chain; findall sees all.
    findall(F, ( wam_cpp_a2struct(_, S), functor(S, F, _) ), L),
    L = [foo, bar, baz].

% A2 = mixed atom + compound. Exercises switch_on_term_a2 — the most
% common shape (atom for "empty" plus compound for non-empty). Mirrors
% the assoc tree-recur predicate that's now indexed by this opcode.
:- dynamic user:wam_cpp_a2term/3.
:- dynamic user:wam_cpp_test_a2_term_atom_clause/0.
:- dynamic user:wam_cpp_test_a2_term_compound_clause/0.
:- dynamic user:wam_cpp_test_a2_term_list_clauses/0.
:- dynamic user:wam_cpp_test_a2_term_unbound/0.

user:wam_cpp_a2term(_, empty,     leaf).
user:wam_cpp_a2term(_, node(_, _), branch).
user:wam_cpp_a2term(_, [],        nil_list).
user:wam_cpp_a2term(_, [_|_],     cons_list).

user:wam_cpp_test_a2_term_atom_clause :-
    wam_cpp_a2term(any, empty, R), R = leaf.
user:wam_cpp_test_a2_term_compound_clause :-
    wam_cpp_a2term(any, node(1, 2), R), R = branch.
user:wam_cpp_test_a2_term_list_clauses :-
    wam_cpp_a2term(any, [],         R1), R1 = nil_list,
    wam_cpp_a2term(any, [a, b, c],  R2), R2 = cons_list.
user:wam_cpp_test_a2_term_unbound :-
    findall(R, wam_cpp_a2term(_, _, R), L),
    L = [leaf, branch, nil_list, cons_list].

% Mixed-mode A1 indexing — predicates with a trailing variable-A1
% clause acting as the catch-all default. Previously such predicates
% got NO A1 indexing (the var clause disabled the whole table). Now
% the indexed prefix (clauses before any variable-A1 clause) gets a
% switch_on_constant_fallthrough that jumps directly on a hit and
% falls through to the try_me_else chain on miss.
:- dynamic user:wam_cpp_mma_tag/2.
:- dynamic user:wam_cpp_test_mma_specific_hit/0.
:- dynamic user:wam_cpp_test_mma_unknown_falls_through/0.
:- dynamic user:wam_cpp_test_mma_specific_backtracks_to_default/0.
:- dynamic user:wam_cpp_test_mma_unknown_only_default/0.
:- dynamic user:wam_cpp_test_mma_all_unbound_enumerates/0.

user:wam_cpp_mma_tag(error, red).
user:wam_cpp_mma_tag(warn,  yellow).
user:wam_cpp_mma_tag(ok,    green).
user:wam_cpp_mma_tag(_,     gray).

user:wam_cpp_test_mma_specific_hit :-
    % Direct switch hit on each indexed clause.
    wam_cpp_mma_tag(error, red),
    wam_cpp_mma_tag(warn,  yellow),
    wam_cpp_mma_tag(ok,    green).
user:wam_cpp_test_mma_unknown_falls_through :-
    % A1 = something NOT in the switch table — must fall through
    % to the chain so the variable-A1 clause matches.
    wam_cpp_mma_tag(other, gray).
user:wam_cpp_test_mma_specific_backtracks_to_default :-
    % A1 = error matches clause 1 (red) AND the trailing var clause
    % (gray) on backtrack.
    findall(C, wam_cpp_mma_tag(error, C), L),
    L = [red, gray].
user:wam_cpp_test_mma_unknown_only_default :-
    % A1 = unknown matches ONLY the var clause.
    findall(C, wam_cpp_mma_tag(unknown, C), L),
    L = [gray].
user:wam_cpp_test_mma_all_unbound_enumerates :-
    % Unbound A1 enumerates every clause in source order.
    findall(T-C, wam_cpp_mma_tag(T, C), L),
    L = [error-red, warn-yellow, ok-green, _-gray].

% Same predicate shape with a variable clause in the MIDDLE — the
% indexed prefix is just the first clause (`a`). Everything after
% the var clause sits in the try_me_else chain and is only reached
% via fall-through.
:- dynamic user:wam_cpp_mma_mid/2.
:- dynamic user:wam_cpp_test_mma_mid_indexed/0.
:- dynamic user:wam_cpp_test_mma_mid_after_var/0.
:- dynamic user:wam_cpp_test_mma_mid_unbound/0.

user:wam_cpp_mma_mid(a, 1).
user:wam_cpp_mma_mid(_, 99).
user:wam_cpp_mma_mid(b, 2).
user:wam_cpp_mma_mid(c, 3).

user:wam_cpp_test_mma_mid_indexed :-
    % A1=a hits the indexed clause first, then the var clause on retry.
    findall(V, wam_cpp_mma_mid(a, V), L),
    L = [1, 99].
user:wam_cpp_test_mma_mid_after_var :-
    % A1=b falls through (no entry for b in the indexed prefix),
    % the chain walks all clauses and clauses 2+3 both match.
    findall(V, wam_cpp_mma_mid(b, V), L),
    L = [99, 2].
user:wam_cpp_test_mma_mid_unbound :-
    findall(K-V, wam_cpp_mma_mid(K, V), L),
    L = [a-1, _-99, b-2, c-3].

% Mixed-mode A2 indexing — mirror of the A1 case for predicates
% whose A1 is variable in every clause but whose A2 has a trailing
% variable catch-all. Previously the variable A2 disabled A2
% indexing entirely; now the prefix is dispatched via
% switch_on_constant_a2_fallthrough.
:- dynamic user:wam_cpp_mma2_tag/3.
:- dynamic user:wam_cpp_test_mma2_specific_hit/0.
:- dynamic user:wam_cpp_test_mma2_unknown_falls_through/0.
:- dynamic user:wam_cpp_test_mma2_specific_backtracks_to_default/0.
:- dynamic user:wam_cpp_test_mma2_unknown_only_default/0.
:- dynamic user:wam_cpp_test_mma2_all_unbound_enumerates/0.

user:wam_cpp_mma2_tag(_, error, red).
user:wam_cpp_mma2_tag(_, warn,  yellow).
user:wam_cpp_mma2_tag(_, ok,    green).
user:wam_cpp_mma2_tag(_, _,     gray).

user:wam_cpp_test_mma2_specific_hit :-
    wam_cpp_mma2_tag(any, error, red),
    wam_cpp_mma2_tag(any, warn,  yellow),
    wam_cpp_mma2_tag(any, ok,    green).
user:wam_cpp_test_mma2_unknown_falls_through :-
    % A2 = something NOT in the table — must fall through to the
    % chain so the variable-A2 clause matches.
    wam_cpp_mma2_tag(any, other, gray).
user:wam_cpp_test_mma2_specific_backtracks_to_default :-
    findall(C, wam_cpp_mma2_tag(any, error, C), L),
    L = [red, gray].
user:wam_cpp_test_mma2_unknown_only_default :-
    findall(C, wam_cpp_mma2_tag(any, unknown, C), L),
    L = [gray].
user:wam_cpp_test_mma2_all_unbound_enumerates :-
    findall(T-C, wam_cpp_mma2_tag(_, T, C), L),
    L = [error-red, warn-yellow, ok-green, _-gray].

% Variable A2 in the middle — indexed prefix is just `a`. After-var
% clauses are reachable only via fall-through.
:- dynamic user:wam_cpp_mma2_mid/3.
:- dynamic user:wam_cpp_test_mma2_mid_indexed/0.
:- dynamic user:wam_cpp_test_mma2_mid_after_var/0.

user:wam_cpp_mma2_mid(_, a, 1).
user:wam_cpp_mma2_mid(_, _, 99).
user:wam_cpp_mma2_mid(_, b, 2).
user:wam_cpp_mma2_mid(_, c, 3).

user:wam_cpp_test_mma2_mid_indexed :-
    findall(V, wam_cpp_mma2_mid(any, a, V), L),
    L = [1, 99].
user:wam_cpp_test_mma2_mid_after_var :-
    findall(V, wam_cpp_mma2_mid(any, b, V), L),
    L = [99, 2].

user:wam_cpp_test_write :- write(hello), nl.
% Y-reg isolation: both helpers use Y1/Y2 internally. Caller relies on
% preserved Y1 across the two calls.
user:wam_cpp_h1(X) :- user:wam_cpp_num(_), X = a.
user:wam_cpp_h2(Y) :- user:wam_cpp_num(_), Y = b.
user:wam_cpp_two_helpers      :- user:wam_cpp_h1(A), user:wam_cpp_h2(B), A = a, B = b.
user:wam_cpp_two_helpers_swap :- user:wam_cpp_h1(A), user:wam_cpp_h2(B), A = b, B = a.
% Tail-recursive list length — exercises cp threading + Y-reg framing
% across recursive calls.
user:wam_cpp_length_acc([], Acc, Acc).
user:wam_cpp_length_acc([_|T], Acc, N) :-
    Acc1 is Acc + 1,
    user:wam_cpp_length_acc(T, Acc1, N).
user:wam_cpp_list_length(L, N) :- user:wam_cpp_length_acc(L, 0, N).
user:wam_cpp_test_len_empty :- user:wam_cpp_list_length([], 0).
user:wam_cpp_test_len_one   :- user:wam_cpp_list_length([a], 1).
user:wam_cpp_test_len_three :- user:wam_cpp_list_length([a, b, c], 3).
user:wam_cpp_test_len_five  :- user:wam_cpp_list_length([a, b, c, d, e], 5).
user:wam_cpp_item(a). user:wam_cpp_item(b). user:wam_cpp_item(c).
user:wam_cpp_num(1).  user:wam_cpp_num(2).  user:wam_cpp_num(3). user:wam_cpp_num(2).
user:wam_cpp_test_findall         :- findall(X, user:wam_cpp_item(X), L), L = [a, b, c].
user:wam_cpp_test_findall_empty   :- findall(_, fail, L), L = [].
user:wam_cpp_test_findall_doubled :- findall(p(X, X), user:wam_cpp_item(X), L),
                                     L = [p(a, a), p(b, b), p(c, c)].
user:wam_cpp_test_bagof           :- bagof(X, user:wam_cpp_item(X), L), L = [a, b, c].
user:wam_cpp_test_bagof_empty     :- bagof(_, fail, _).
user:wam_cpp_test_setof           :- setof(X, user:wam_cpp_num(X), L), L = [1, 2, 3].
user:wam_cpp_test_setof_empty     :- setof(_, fail, _).
user:wam_cpp_test_count :- aggregate_all(count, user:wam_cpp_item(_), N), N = 3.
user:wam_cpp_test_sum   :- aggregate_all(sum(X),  user:wam_cpp_num(X), S), S = 8.
user:wam_cpp_test_min   :- aggregate_all(min(X),  user:wam_cpp_num(X), M), M = 1.
user:wam_cpp_test_max   :- aggregate_all(max(X),  user:wam_cpp_num(X), M), M = 3.
user:wam_cpp_test_set   :- aggregate_all(set(X),  user:wam_cpp_num(X), S), S = [1, 2, 3].

user:wam_cpp_fact(a).
user:wam_cpp_choice(a).
user:wam_cpp_choice(b).
user:wam_cpp_caller(X) :- user:wam_cpp_fact(X).
user:wam_cpp_rect(box(1, 2)).
user:wam_cpp_has_rect          :- user:wam_cpp_rect(box(1, 2)).
user:wam_cpp_has_rect_wrong    :- user:wam_cpp_rect(box(1, 3)).
user:wam_cpp_first(box(X, _), X).
user:wam_cpp_lst([a, b, c]).
% Arithmetic & comparison
user:wam_cpp_add1(X, Y)        :- Y is X + 1.
user:wam_cpp_gt(X, Y)          :- X > Y.
user:wam_cpp_test_arith        :- 6 is 2 + 4, 12 is 3 * 4, 5 is 10 / 2.
user:wam_cpp_test_eq           :- 5 =:= 2 + 3.
user:wam_cpp_test_neq          :- 5 =\= 6.
% Arithmetic function expansion -- abs/sign/sqrt/floor/ceiling/round/
% truncate/min/max/gcd/rem/** /^ -- and bit ops /\, \/, xor, >>, <<, \\.
user:wam_cpp_test_abs_neg      :- X is abs(-7), X = 7.
user:wam_cpp_test_abs_float    :- X is abs(-3.5), X = 3.5.
user:wam_cpp_test_sign_neg     :- X is sign(-9), X = -1.
user:wam_cpp_test_sign_zero    :- X is sign(0), X = 0.
user:wam_cpp_test_sqrt         :- X is sqrt(9), X = 3.0.
user:wam_cpp_test_floor_neg    :- X is floor(-3.2), X = -4.
user:wam_cpp_test_ceiling_neg  :- X is ceiling(-3.7), X = -3.
user:wam_cpp_test_round        :- X is round(2.5), X = 3.
user:wam_cpp_test_truncate_neg :- X is truncate(-3.7), X = -3.
user:wam_cpp_test_unary_plus   :- X is +(7), X = 7.
user:wam_cpp_test_bitnot       :- X is \(5), X = -6.
user:wam_cpp_test_min_int      :- X is min(3, 7), X = 3.
user:wam_cpp_test_max_float    :- X is max(2.5, 3.0), X = 3.0.
user:wam_cpp_test_pow_star     :- X is 2 ** 10, X = 1024.0.
user:wam_cpp_test_pow_caret    :- X is 2 ^ 10, X = 1024.
user:wam_cpp_test_pow_zero     :- X is 5 ^ 0, X = 1.
user:wam_cpp_test_gcd          :- X is gcd(12, 18), X = 6.
user:wam_cpp_test_gcd_neg      :- X is gcd(-12, 18), X = 6.
user:wam_cpp_test_rem_neg      :- X is rem(-7, 3), X = -1.
user:wam_cpp_test_bitand       :- X is 12 /\ 10, X = 8.
user:wam_cpp_test_bitor        :- X is 12 \/ 10, X = 14.
user:wam_cpp_test_bitxor       :- X is xor(12, 10), X = 6.
user:wam_cpp_test_shl          :- X is 1 << 5, X = 32.
user:wam_cpp_test_shr          :- X is 64 >> 3, X = 8.
user:wam_cpp_test_arith_compose1 :- X is max(abs(-5), abs(-3)), X = 5.
user:wam_cpp_test_arith_compose2 :- X is floor(sqrt(50)), X = 7.
% Transcendentals: trig (sin/cos/tan + inverses), hyperbolic, log/exp,
% constants (pi/e/inf/nan). All return Float.
user:wam_cpp_test_pi          :- X is pi, X > 3.14, X < 3.15.
user:wam_cpp_test_e           :- X is e, X > 2.71, X < 2.72.
user:wam_cpp_test_nan         :- X is nan, \+ (X =:= X).
user:wam_cpp_test_sin_zero    :- X is sin(0), X =:= 0.0.
user:wam_cpp_test_cos_pi      :- X is cos(pi), X > -1.001, X < -0.999.
user:wam_cpp_test_tan_zero    :- X is tan(0), X =:= 0.0.
user:wam_cpp_test_asin_one    :- X is asin(1), Y is pi/2, X > Y - 0.001, X < Y + 0.001.
user:wam_cpp_test_atan2_diag  :- X is atan2(1, 1), Y is pi/4, X > Y - 0.001, X < Y + 0.001.
user:wam_cpp_test_cosh_zero   :- X is cosh(0), X =:= 1.0.
user:wam_cpp_test_exp_zero    :- X is exp(0), X =:= 1.0.
user:wam_cpp_test_log_e       :- X is log(e), X > 0.999, X < 1.001.
user:wam_cpp_test_log_base    :- X is log(10, 1000), X > 2.999, X < 3.001.
user:wam_cpp_test_exp_log     :- X is exp(log(5)), X > 4.999, X < 5.001.
user:wam_cpp_test_pythag_one  :- X is sin(pi/4) ** 2 + cos(pi/4) ** 2,
                                 X > 0.999, X < 1.001.
% Type checks
user:wam_cpp_is_atom(X)        :- atom(X).
user:wam_cpp_is_int(X)         :- integer(X).
user:wam_cpp_is_num(X)         :- number(X).
user:wam_cpp_is_var(X)         :- var(X).
user:wam_cpp_is_compound(X)    :- compound(X).
user:wam_cpp_test_nonvar       :- X = foo, nonvar(X).
% Term inspection
user:wam_cpp_test_functor      :- functor(box(1, 2), box, 2).
user:wam_cpp_test_arg1         :- arg(1, box(a, b), a).
user:wam_cpp_test_arg_bad      :- arg(1, box(a, b), z).
user:wam_cpp_test_univ_decompose :- box(1, 2) =.. [box, 1, 2].
user:wam_cpp_test_univ_compose   :- T =.. [foo, a, b], T = foo(a, b).
% Term inspection -- expanded coverage for #1 (functor/arg/univ) + #2 (copy_term/ground).
user:wam_cpp_test_functor_atom       :- functor(hello, N, A), N == hello, A == 0.
user:wam_cpp_test_functor_int        :- functor(7, N, A), N == 7, A == 0.
user:wam_cpp_test_functor_build      :- functor(T, p, 3), T =.. [p, _, _, _].
user:wam_cpp_test_functor_build_atom :- functor(T, lone, 0), T == lone.
user:wam_cpp_test_functor_build_int  :- functor(T, 9, 0), T == 9.
user:wam_cpp_test_arg_second         :- arg(2, point(10, 20, 30), V), V == 20.
user:wam_cpp_test_arg_outof          :- \+ arg(5, point(10, 20), _).
user:wam_cpp_test_arg_zero_idx       :- \+ arg(0, point(10, 20), _).
user:wam_cpp_test_arg_compound_val   :- arg(1, wrap(inner(1, 2)), W), W = inner(1, 2).
user:wam_cpp_test_univ_atom          :- hello =.. [hello].
user:wam_cpp_test_univ_int           :- 42 =.. [42].
user:wam_cpp_test_univ_compose_solo  :- T =.. [lone], T == lone.
user:wam_cpp_test_copy_int           :- copy_term(42, T), T == 42.
user:wam_cpp_test_copy_nested        :- copy_term(f(g(X), X), f(g(A), A)).
user:wam_cpp_test_ground_compound    :- ground(foo(1, [2, 3], bar)).
user:wam_cpp_test_ground_unbound     :- \+ ground(_).
user:wam_cpp_test_ground_partial     :- \+ ground(foo(1, _, 3)).
user:wam_cpp_test_ground_atom        :- ground(hello).
user:wam_cpp_test_ground_guard       :- ( ground(foo(1, 2)) -> true ; fail ).
user:wam_cpp_test_ground_guard_not   :- ( ground(foo(_, 2)) -> fail ; true ).
% =/2 / \\=/2
user:wam_cpp_test_unify        :- X = foo, X = foo.
user:wam_cpp_test_unify_fail   :- foo \= foo.

% --------------------------------------------------------------------
% Module-level exports
% --------------------------------------------------------------------
test(exports) :-
    assertion(current_predicate(wam_cpp_target:write_wam_cpp_project/3)),
    assertion(current_predicate(wam_cpp_target:compile_wam_predicate_to_cpp/4)),
    assertion(current_predicate(wam_cpp_target:compile_wam_runtime_to_cpp/2)),
    assertion(current_predicate(wam_cpp_target:compile_wam_runtime_header_to_cpp/2)),
    assertion(current_predicate(wam_cpp_target:cpp_wam_resolve_emit_mode/2)),
    assertion(current_predicate(wam_cpp_target:escape_cpp_string/2)),
    assertion(current_predicate(wam_cpp_lowered_emitter:wam_cpp_lowerable/3)),
    assertion(current_predicate(wam_cpp_lowered_emitter:lower_predicate_to_cpp/4)),
    assertion(current_predicate(wam_cpp_lowered_emitter:cpp_lowered_func_name/2)).

% --------------------------------------------------------------------
% Registry wiring
% --------------------------------------------------------------------
test(registry) :-
    assertion(target_exists(wam_cpp)),
    assertion(target_family(wam_cpp, native)),
    assertion(target_module(wam_cpp, wam_cpp_target)).

% --------------------------------------------------------------------
% Emit-mode resolution
% --------------------------------------------------------------------
test(emit_mode_default) :-
    cpp_wam_resolve_emit_mode([], Mode),
    assertion(Mode == interpreter).

test(emit_mode_functions) :-
    cpp_wam_resolve_emit_mode([emit_mode(functions)], Mode),
    assertion(Mode == functions).

test(emit_mode_mixed) :-
    cpp_wam_resolve_emit_mode([emit_mode(mixed([foo/2,bar/3]))], Mode),
    assertion(Mode == mixed([foo/2, bar/3])).

test(emit_mode_invalid, [throws(error(domain_error(wam_cpp_emit_mode, garbage), _))]) :-
    cpp_wam_resolve_emit_mode([emit_mode(garbage)], _).

% --------------------------------------------------------------------
% Lowered function naming
% --------------------------------------------------------------------
test(lowered_func_name_simple) :-
    cpp_lowered_func_name(foo/2, Name),
    assertion(Name == 'lowered_foo_2').

test(lowered_func_name_sanitised) :-
    cpp_lowered_func_name('my-pred'/3, Name),
    assertion(Name == 'lowered_my_pred_3').

% --------------------------------------------------------------------
% Lowerability classification (operates on instruction lists directly)
% --------------------------------------------------------------------
test(lowerability_deterministic) :-
    Instrs = [get_constant("a", "A1"), proceed],
    wam_cpp_lowerable(wam_cpp_fact/1, Instrs, Reason),
    assertion(Reason == deterministic).

test(lowerability_multi_clause_1) :-
    Instrs = [try_me_else("L2"),
              get_constant("a", "A1"),
              proceed,
              trust_me,
              get_constant("b", "A1"),
              proceed],
    wam_cpp_lowerable(wam_cpp_choice/1, Instrs, Reason),
    assertion(Reason == multi_clause_1).

test(is_deterministic_helper) :-
    assertion(is_deterministic_pred_cpp([proceed])),
    assertion(\+ is_deterministic_pred_cpp([try_me_else("L"), proceed])).

% --------------------------------------------------------------------
% Lowered function emission
% --------------------------------------------------------------------
test(lower_predicate_emits_signature_and_proceed) :-
    Instrs = [get_constant("a", "A1"), proceed],
    lower_predicate_to_cpp(wam_cpp_fact/1, Instrs, [], Lines),
    atomic_list_concat(Lines, '\n', Code),
    assertion(sub_atom(Code, _, _, _, 'bool lowered_wam_cpp_fact_1(WamState* vm)')),
    assertion(sub_atom(Code, _, _, _, 'return true;')),
    assertion(sub_atom(Code, _, _, _, 'get_constant a, A1')).

test(lower_predicate_emits_unify_for_constants) :-
    Instrs = [get_constant("hello", "A1"), proceed],
    lower_predicate_to_cpp(test_const/1, Instrs, [], Lines),
    atomic_list_concat(Lines, '\n', Code),
    assertion(sub_atom(Code, _, _, _, 'Value::Atom("hello")')),
    assertion(sub_atom(Code, _, _, _, 'vm->trail_binding')),
    assertion(sub_atom(Code, _, _, _, 'return false;')).

test(lower_predicate_emits_call_dispatch) :-
    Instrs = [put_constant("a", "A1"), call("wam_cpp_fact/1", "1"), proceed],
    lower_predicate_to_cpp(wam_cpp_caller/1, Instrs, [], Lines),
    atomic_list_concat(Lines, '\n', Code),
    assertion(sub_atom(Code, _, _, _, 'vm->labels.find("wam_cpp_fact/1")')),
    assertion(sub_atom(Code, _, _, _, 'vm->run()')).

test(lower_predicate_routes_foreign_calls) :-
    Instrs = [put_constant("a", "A1"), call("edge/2", "2"), proceed],
    lower_predicate_to_cpp(uses_foreign/1, Instrs,
                           [foreign_pred_keys(["edge/2"])], Lines),
    atomic_list_concat(Lines, '\n', Code),
    assertion(sub_atom(Code, _, _, _, 'Instruction::CallForeign("edge/2", 2)')).

% --------------------------------------------------------------------
% Instruction literal emission (for the interpreter array)
% --------------------------------------------------------------------
test(instruction_literal_get_constant) :-
    wam_instruction_to_cpp_literal(get_constant("a", "A1"), Code),
    assertion(Code == 'Instruction::GetConstant(Value::Atom("a"), "A1")').

test(instruction_literal_proceed) :-
    wam_instruction_to_cpp_literal(proceed, Code),
    assertion(Code == 'Instruction::Proceed()').

test(instruction_literal_call) :-
    wam_instruction_to_cpp_literal(call("foo/2", "2"), Code),
    assertion(Code == 'Instruction::Call("foo/2", 2)').

% --------------------------------------------------------------------
% String escaping
% --------------------------------------------------------------------
test(escape_cpp_string_backslash) :-
    escape_cpp_string("a\\b", Out),
    assertion(Out == "a\\\\b").

test(escape_cpp_string_quote) :-
    escape_cpp_string("a\"b", Out),
    assertion(Out == "a\\\"b").

% --------------------------------------------------------------------
% Project layout
% --------------------------------------------------------------------
test(project_layout) :-
    unique_cpp_tmp_dir('tmp_cpp_layout', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_fact/1], [], TmpDir),
        ( directory_file_path(TmpDir, 'cpp/wam_runtime.h', Header),
          directory_file_path(TmpDir, 'cpp/wam_runtime.cpp', Runtime),
          directory_file_path(TmpDir, 'cpp/generated_program.cpp', Program),
          assertion(exists_file(Header)),
          assertion(exists_file(Runtime)),
          assertion(exists_file(Program))
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(project_runtime_header_content) :-
    unique_cpp_tmp_dir('tmp_cpp_header', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_fact/1], [], TmpDir),
        ( directory_file_path(TmpDir, 'cpp/wam_runtime.h', Header),
          read_file_to_string(Header, Code, []),
          assertion(sub_string(Code, _, _, _, 'namespace wam_cpp')),
          assertion(sub_string(Code, _, _, _, 'struct Value')),
          assertion(sub_string(Code, _, _, _, 'struct Instruction')),
          assertion(sub_string(Code, _, _, _, 'struct WamState')),
          assertion(sub_string(Code, _, _, _, 'unify(const Value&'))
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(project_program_includes_runtime) :-
    unique_cpp_tmp_dir('tmp_cpp_prog', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_fact/1], [], TmpDir),
        ( directory_file_path(TmpDir, 'cpp/generated_program.cpp', Program),
          read_file_to_string(Program, Code, []),
          assertion(sub_string(Code, _, _, _, '#include "wam_runtime.h"'))
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(lowered_functions_mode) :-
    unique_cpp_tmp_dir('tmp_cpp_lowered', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_fact/1],
            [emit_mode(functions)],
            TmpDir),
        ( directory_file_path(TmpDir, 'cpp/generated_program.cpp', Program),
          read_file_to_string(Program, Code, []),
          assertion(sub_string(Code, _, _, _, 'bool lowered_wam_cpp_fact_1(WamState* vm)'))
        ),
        delete_directory_and_contents(TmpDir)
    ).

% --------------------------------------------------------------------
% Optional: header compiles cleanly with a C++17 compiler if one
% is on PATH. Skipped silently otherwise — we don't want to gate
% Prolog-side CI on host toolchains.
% --------------------------------------------------------------------
test(cpp_compiler_smoke, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_smoke', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_fact/1],
                              [emit_mode(functions)], TmpDir),
        ( directory_file_path(TmpDir, 'cpp', CppDir),
          % Compile each .cpp separately (g++ disallows -o with -c and
          % multiple inputs). The generated program has no main(); -c
          % produces .o files and that is all we need to verify the
          % runtime + lowered code is syntactically valid.
          compile_one(CppDir, 'wam_runtime.cpp', 'wam_runtime.o', R1),
          assertion(R1 == exit(0)),
          compile_one(CppDir, 'generated_program.cpp', 'generated_program.o', R2),
          assertion(R2 == exit(0))
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% End-to-end: build a binary with main.cpp, run queries, check exit.
% ------------------------------------------------------------------

test(cpp_e2e_fact, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_fact', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_fact/1],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_fact/1', [a], true),
          run_query(BinPath, 'wam_cpp_fact/1', [b], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_choice_backtracking, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_choice', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_choice/1],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_choice/1', [a], true),
          % Clause 2 only reachable via backtracking — exercises the
          % choice point / trail / TrustMe path.
          run_query(BinPath, 'wam_cpp_choice/1', [b], true),
          run_query(BinPath, 'wam_cpp_choice/1', [c], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_caller, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_caller', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_caller/1, user:wam_cpp_fact/1],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          % caller(X) :- fact(X). Exercises Call dispatch + Proceed
          % through the labels table.
          run_query(BinPath, 'wam_cpp_caller/1', [a], true),
          run_query(BinPath, 'wam_cpp_caller/1', [b], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% Compound terms + lists: heap-resident structures via shared_ptr cells.
% Exercises Get/PutStructure + Get/PutList + Unify*/Set* + the CLI parser
% for compound and list syntax.
% ------------------------------------------------------------------

test(cpp_e2e_structure_head_match, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_struct', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_rect/1],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_rect/1', ['box(1,2)'], true),
          run_query(BinPath, 'wam_cpp_rect/1', ['box(1,3)'], false),
          run_query(BinPath, 'wam_cpp_rect/1', ['box(2,2)'], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_structure_build_and_match, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_build', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_has_rect/0, user:wam_cpp_has_rect_wrong/0,
             user:wam_cpp_rect/1],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          % has_rect builds box(1,2) and calls rect/1 — exercises
          % PutStructure + SetConstant + Execute.
          run_query(BinPath, 'wam_cpp_has_rect/0',       [], true),
          run_query(BinPath, 'wam_cpp_has_rect_wrong/0', [], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_structure_destructure, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_destr', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_first/2],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          % first(box(X, _), X). Pulls X out of the compound and unifies
          % with A2 — exercises UnifyVariable + GetValue across compounds.
          run_query(BinPath, 'wam_cpp_first/2', ['box(1,2)', '1'], true),
          run_query(BinPath, 'wam_cpp_first/2', ['box(7,8)', '7'], true),
          run_query(BinPath, 'wam_cpp_first/2', ['box(1,2)', '9'], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% Builtins: arithmetic, comparison, type checks, term inspection, =/2.
% ------------------------------------------------------------------

test(cpp_e2e_builtin_arithmetic, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_arith', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_add1/2, user:wam_cpp_test_arith/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_add1/2',     [5, 6], true),
          run_query(BinPath, 'wam_cpp_add1/2',     [5, 7], false),
          run_query(BinPath, 'wam_cpp_test_arith/0', [],  true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_arith_expansion, [condition(cpp_compiler_available)]) :-
    % Arithmetic function expansion: abs/sign/sqrt + flavors of int
    % rounding (truncate/floor/ceiling/round) + min/max/** /^/gcd/rem
    % + bitwise ops (/\, \/, xor, >>, <<, \\).
    unique_cpp_tmp_dir('tmp_cpp_e2e_arith_x', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_abs_neg/0,
                               user:wam_cpp_test_abs_float/0,
                               user:wam_cpp_test_sign_neg/0,
                               user:wam_cpp_test_sign_zero/0,
                               user:wam_cpp_test_sqrt/0,
                               user:wam_cpp_test_floor_neg/0,
                               user:wam_cpp_test_ceiling_neg/0,
                               user:wam_cpp_test_round/0,
                               user:wam_cpp_test_truncate_neg/0,
                               user:wam_cpp_test_unary_plus/0,
                               user:wam_cpp_test_bitnot/0,
                               user:wam_cpp_test_min_int/0,
                               user:wam_cpp_test_max_float/0,
                               user:wam_cpp_test_pow_star/0,
                               user:wam_cpp_test_pow_caret/0,
                               user:wam_cpp_test_pow_zero/0,
                               user:wam_cpp_test_gcd/0,
                               user:wam_cpp_test_gcd_neg/0,
                               user:wam_cpp_test_rem_neg/0,
                               user:wam_cpp_test_bitand/0,
                               user:wam_cpp_test_bitor/0,
                               user:wam_cpp_test_bitxor/0,
                               user:wam_cpp_test_shl/0,
                               user:wam_cpp_test_shr/0,
                               user:wam_cpp_test_arith_compose1/0,
                               user:wam_cpp_test_arith_compose2/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_abs_neg/0',         [], true),
          run_query(BinPath, 'wam_cpp_test_abs_float/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_sign_neg/0',        [], true),
          run_query(BinPath, 'wam_cpp_test_sign_zero/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_sqrt/0',            [], true),
          run_query(BinPath, 'wam_cpp_test_floor_neg/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_ceiling_neg/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_round/0',           [], true),
          run_query(BinPath, 'wam_cpp_test_truncate_neg/0',    [], true),
          run_query(BinPath, 'wam_cpp_test_unary_plus/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_bitnot/0',          [], true),
          run_query(BinPath, 'wam_cpp_test_min_int/0',         [], true),
          run_query(BinPath, 'wam_cpp_test_max_float/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_pow_star/0',        [], true),
          run_query(BinPath, 'wam_cpp_test_pow_caret/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_pow_zero/0',        [], true),
          run_query(BinPath, 'wam_cpp_test_gcd/0',             [], true),
          run_query(BinPath, 'wam_cpp_test_gcd_neg/0',         [], true),
          run_query(BinPath, 'wam_cpp_test_rem_neg/0',         [], true),
          run_query(BinPath, 'wam_cpp_test_bitand/0',          [], true),
          run_query(BinPath, 'wam_cpp_test_bitor/0',           [], true),
          run_query(BinPath, 'wam_cpp_test_bitxor/0',          [], true),
          run_query(BinPath, 'wam_cpp_test_shl/0',             [], true),
          run_query(BinPath, 'wam_cpp_test_shr/0',             [], true),
          run_query(BinPath, 'wam_cpp_test_arith_compose1/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_arith_compose2/0',  [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_arith_transcendentals, [condition(cpp_compiler_available)]) :-
    % Transcendentals: trig (sin/cos/tan + inverses), hyperbolic
    % (sinh/cosh/tanh), log/exp with base, constants pi/e/nan, plus a
    % Pythagorean identity check using `**`.
    unique_cpp_tmp_dir('tmp_cpp_e2e_trig', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_pi/0,
                               user:wam_cpp_test_e/0,
                               user:wam_cpp_test_nan/0,
                               user:wam_cpp_test_sin_zero/0,
                               user:wam_cpp_test_cos_pi/0,
                               user:wam_cpp_test_tan_zero/0,
                               user:wam_cpp_test_asin_one/0,
                               user:wam_cpp_test_atan2_diag/0,
                               user:wam_cpp_test_cosh_zero/0,
                               user:wam_cpp_test_exp_zero/0,
                               user:wam_cpp_test_log_e/0,
                               user:wam_cpp_test_log_base/0,
                               user:wam_cpp_test_exp_log/0,
                               user:wam_cpp_test_pythag_one/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_pi/0',           [], true),
          run_query(BinPath, 'wam_cpp_test_e/0',            [], true),
          run_query(BinPath, 'wam_cpp_test_nan/0',          [], true),
          run_query(BinPath, 'wam_cpp_test_sin_zero/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_cos_pi/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_tan_zero/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_asin_one/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_atan2_diag/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_cosh_zero/0',    [], true),
          run_query(BinPath, 'wam_cpp_test_exp_zero/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_log_e/0',        [], true),
          run_query(BinPath, 'wam_cpp_test_log_base/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_exp_log/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_pythag_one/0',   [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_builtin_comparison, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_cmp', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_gt/2,
                               user:wam_cpp_test_eq/0,
                               user:wam_cpp_test_neq/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_gt/2',     [5, 3], true),
          run_query(BinPath, 'wam_cpp_gt/2',     [3, 5], false),
          run_query(BinPath, 'wam_cpp_test_eq/0',  [],  true),
          run_query(BinPath, 'wam_cpp_test_neq/0', [],  true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_builtin_type_checks, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_types', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_is_atom/1, user:wam_cpp_is_int/1,
                               user:wam_cpp_is_num/1, user:wam_cpp_is_compound/1,
                               user:wam_cpp_test_nonvar/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_is_atom/1',  [foo],         true),
          run_query(BinPath, 'wam_cpp_is_atom/1',  [5],           false),
          run_query(BinPath, 'wam_cpp_is_int/1',   [5],           true),
          run_query(BinPath, 'wam_cpp_is_int/1',   [foo],         false),
          run_query(BinPath, 'wam_cpp_is_num/1',   [5],           true),
          run_query(BinPath, 'wam_cpp_is_num/1',   [foo],         false),
          run_query(BinPath, 'wam_cpp_is_compound/1', ['box(1,2)'], true),
          run_query(BinPath, 'wam_cpp_is_compound/1', [foo],        false),
          run_query(BinPath, 'wam_cpp_test_nonvar/0', [],            true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_builtin_term_inspection, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_term', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_functor/0,
                               user:wam_cpp_test_arg1/0,
                               user:wam_cpp_test_arg_bad/0,
                               user:wam_cpp_test_univ_decompose/0,
                               user:wam_cpp_test_univ_compose/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_functor/0',         [], true),
          run_query(BinPath, 'wam_cpp_test_arg1/0',            [], true),
          run_query(BinPath, 'wam_cpp_test_arg_bad/0',         [], false),
          run_query(BinPath, 'wam_cpp_test_univ_decompose/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_univ_compose/0',    [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_term_inspection_extended, [condition(cpp_compiler_available)]) :-
    % Expanded coverage for functor/3, arg/3, =../2, copy_term/2, ground/1
    % including edge cases (atom + integer + float Names, out-of-range
    % arg, atomic univ, deep-shared copy_term) and ground/1 used as an
    % if-then-else guard.
    unique_cpp_tmp_dir('tmp_cpp_e2e_term_ext', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_functor_atom/0,
                               user:wam_cpp_test_functor_int/0,
                               user:wam_cpp_test_functor_build/0,
                               user:wam_cpp_test_functor_build_atom/0,
                               user:wam_cpp_test_functor_build_int/0,
                               user:wam_cpp_test_arg_second/0,
                               user:wam_cpp_test_arg_outof/0,
                               user:wam_cpp_test_arg_zero_idx/0,
                               user:wam_cpp_test_arg_compound_val/0,
                               user:wam_cpp_test_univ_atom/0,
                               user:wam_cpp_test_univ_int/0,
                               user:wam_cpp_test_univ_compose_solo/0,
                               user:wam_cpp_test_copy_int/0,
                               user:wam_cpp_test_copy_nested/0,
                               user:wam_cpp_test_ground_compound/0,
                               user:wam_cpp_test_ground_unbound/0,
                               user:wam_cpp_test_ground_partial/0,
                               user:wam_cpp_test_ground_atom/0,
                               user:wam_cpp_test_ground_guard/0,
                               user:wam_cpp_test_ground_guard_not/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_functor_atom/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_functor_int/0',        [], true),
          run_query(BinPath, 'wam_cpp_test_functor_build/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_functor_build_atom/0', [], true),
          run_query(BinPath, 'wam_cpp_test_functor_build_int/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_arg_second/0',         [], true),
          run_query(BinPath, 'wam_cpp_test_arg_outof/0',          [], true),
          run_query(BinPath, 'wam_cpp_test_arg_zero_idx/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_arg_compound_val/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_univ_atom/0',          [], true),
          run_query(BinPath, 'wam_cpp_test_univ_int/0',           [], true),
          run_query(BinPath, 'wam_cpp_test_univ_compose_solo/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_copy_int/0',           [], true),
          run_query(BinPath, 'wam_cpp_test_copy_nested/0',        [], true),
          run_query(BinPath, 'wam_cpp_test_ground_compound/0',    [], true),
          run_query(BinPath, 'wam_cpp_test_ground_unbound/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_ground_partial/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_ground_atom/0',        [], true),
          run_query(BinPath, 'wam_cpp_test_ground_guard/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_ground_guard_not/0',   [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_term_variables, [condition(cpp_compiler_available)]) :-
    % term_variables/2, numbervars/3, =@=/2, \=@=/2, unifiable/3.
    % All operate by walking the term and using cell-pointer identity
    % to track which variables are which. unifiable/3 also relies on
    % the trail to capture-and-undo the unification result.
    unique_cpp_tmp_dir('tmp_cpp_e2e_tvars', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_tv_ground/0,
                               user:wam_cpp_test_tv_one/0,
                               user:wam_cpp_test_tv_shared/0,
                               user:wam_cpp_test_tv_nested/0,
                               user:wam_cpp_test_nv_shared/0,
                               user:wam_cpp_test_nv_start10/0,
                               user:wam_cpp_test_nv_ground/0,
                               user:wam_cpp_test_variant_yes/0,
                               user:wam_cpp_test_variant_share/0,
                               user:wam_cpp_test_variant_shape/0,
                               user:wam_cpp_test_variant_atom/0,
                               user:wam_cpp_test_variant_bind/0,
                               user:wam_cpp_test_variant_neq/0,
                               user:wam_cpp_test_uni_simple/0,
                               user:wam_cpp_test_uni_fail/0,
                               user:wam_cpp_test_uni_two/0,
                               user:wam_cpp_test_uni_ground/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_tv_ground/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_tv_one/0',        [], true),
          run_query(BinPath, 'wam_cpp_test_tv_shared/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_tv_nested/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_nv_shared/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_nv_start10/0',    [], true),
          run_query(BinPath, 'wam_cpp_test_nv_ground/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_variant_yes/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_variant_share/0', [], true),
          run_query(BinPath, 'wam_cpp_test_variant_shape/0', [], true),
          run_query(BinPath, 'wam_cpp_test_variant_atom/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_variant_bind/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_variant_neq/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_uni_simple/0',    [], true),
          run_query(BinPath, 'wam_cpp_test_uni_fail/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_uni_two/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_uni_ground/0',    [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_list_ops_and_guards, [condition(cpp_compiler_available)]) :-
    % nth1/3 (preloaded bytecode), plus/3 (bidirectional integer add),
    % delete/3 + subtract/3 (set-like ops using ==), must_be/2 (type
    % check with ISO error throws).
    unique_cpp_tmp_dir('tmp_cpp_e2e_lg', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_nth1_third/0,
                               user:wam_cpp_test_nth1_zero/0,
                               user:wam_cpp_test_nth1_outof/0,
                               user:wam_cpp_test_plus_xy/0,
                               user:wam_cpp_test_plus_xz/0,
                               user:wam_cpp_test_plus_yz/0,
                               user:wam_cpp_test_plus_check_no/0,
                               user:wam_cpp_test_del_one/0,
                               user:wam_cpp_test_del_none/0,
                               user:wam_cpp_test_del_all/0,
                               user:wam_cpp_test_sub_basic/0,
                               user:wam_cpp_test_sub_empty/0,
                               user:wam_cpp_test_sub_all/0,
                               user:wam_cpp_test_mb_atom_ok/0,
                               user:wam_cpp_test_mb_int_ok/0,
                               user:wam_cpp_test_mb_ground_ok/0,
                               user:wam_cpp_test_mb_callable_ok/0,
                               user:wam_cpp_test_mb_atom_throw/0,
                               user:wam_cpp_test_mb_int_throw/0,
                               user:wam_cpp_test_mb_ground_throw/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_nth1_third/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_nth1_zero/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_nth1_outof/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_plus_xy/0',        [], true),
          run_query(BinPath, 'wam_cpp_test_plus_xz/0',        [], true),
          run_query(BinPath, 'wam_cpp_test_plus_yz/0',        [], true),
          run_query(BinPath, 'wam_cpp_test_plus_check_no/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_del_one/0',        [], true),
          run_query(BinPath, 'wam_cpp_test_del_none/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_del_all/0',        [], true),
          run_query(BinPath, 'wam_cpp_test_sub_basic/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_sub_empty/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_sub_all/0',        [], true),
          run_query(BinPath, 'wam_cpp_test_mb_atom_ok/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_mb_int_ok/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_mb_ground_ok/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_mb_callable_ok/0', [], true),
          run_query(BinPath, 'wam_cpp_test_mb_atom_throw/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_mb_int_throw/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_mb_ground_throw/0',[], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_string_aliases_and_math, [condition(cpp_compiler_available)]) :-
    % string_chars/2 + string_codes/2 alias atom_chars/atom_codes;
    % string_code/3 is 1-based char-code access. Hyperbolic inverses
    % (asinh/acosh/atanh), copysign/2, and integer bit-count
    % functions (popcount/lsb/msb) round out the math tower.
    unique_cpp_tmp_dir('tmp_cpp_e2e_strm', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_string_chars_fwd/0,
                               user:wam_cpp_test_string_chars_rev/0,
                               user:wam_cpp_test_string_codes_fwd/0,
                               user:wam_cpp_test_string_code_first/0,
                               user:wam_cpp_test_string_code_last/0,
                               user:wam_cpp_test_string_code_oob/0,
                               user:wam_cpp_test_asinh_zero/0,
                               user:wam_cpp_test_acosh_one/0,
                               user:wam_cpp_test_atanh_zero/0,
                               user:wam_cpp_test_asinh_roundtrip/0,
                               user:wam_cpp_test_copysign_pos/0,
                               user:wam_cpp_test_copysign_neg/0,
                               user:wam_cpp_test_copysign_flip/0,
                               user:wam_cpp_test_popcount_zero/0,
                               user:wam_cpp_test_popcount_seven/0,
                               user:wam_cpp_test_popcount_ff/0,
                               user:wam_cpp_test_lsb_eight/0,
                               user:wam_cpp_test_lsb_36/0,
                               user:wam_cpp_test_msb_eight/0,
                               user:wam_cpp_test_msb_ff/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_string_chars_fwd/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_string_chars_rev/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_string_codes_fwd/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_string_code_first/0', [], true),
          run_query(BinPath, 'wam_cpp_test_string_code_last/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_string_code_oob/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_asinh_zero/0',        [], true),
          run_query(BinPath, 'wam_cpp_test_acosh_one/0',         [], true),
          run_query(BinPath, 'wam_cpp_test_atanh_zero/0',        [], true),
          run_query(BinPath, 'wam_cpp_test_asinh_roundtrip/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_copysign_pos/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_copysign_neg/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_copysign_flip/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_popcount_zero/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_popcount_seven/0',    [], true),
          run_query(BinPath, 'wam_cpp_test_popcount_ff/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_lsb_eight/0',         [], true),
          run_query(BinPath, 'wam_cpp_test_lsb_36/0',            [], true),
          run_query(BinPath, 'wam_cpp_test_msb_eight/0',         [], true),
          run_query(BinPath, 'wam_cpp_test_msb_ff/0',            [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_dcg, [condition(cpp_compiler_available)]) :-
    % DCG via phrase/2 + phrase/3. SWI's term_expansion handles the
    % --> rewrite at load time, so by the time we reach compile the
    % rules are already plain Prolog with two extra difference-list
    % args. The phrase/2 and phrase/3 dispatchers append [List, Rest]
    % onto the goal's args before invoking via invoke_goal_as_call.
    % Includes ground match, fail-on-shorter-input, parametric DCG
    % (yelp/1), composition (sentence calls greeting), nondet
    % (optional with two clauses), recursive DCG (loop), and the
    % phrase/3 form with non-empty leftover Rest.
    unique_cpp_tmp_dir('tmp_cpp_e2e_dcg', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_dcg_greeting/0,
                               user:wam_cpp_test_dcg_greeting_no/0,
                               user:wam_cpp_test_dcg_yelp/0,
                               user:wam_cpp_test_dcg_optional_empty/0,
                               user:wam_cpp_test_dcg_optional_maybe/0,
                               user:wam_cpp_test_dcg_loop_empty/0,
                               user:wam_cpp_test_dcg_loop_three/0,
                               user:wam_cpp_test_dcg_phrase3/0,
                               % DCG-expanded predicates the test calls.
                               user:dcg_greeting/2,
                               user:dcg_yelp/3,
                               user:dcg_optional/2,
                               user:dcg_loop/3],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_dcg_greeting/0',        [], true),
          run_query(BinPath, 'wam_cpp_test_dcg_greeting_no/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_dcg_yelp/0',            [], true),
          run_query(BinPath, 'wam_cpp_test_dcg_optional_empty/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_dcg_optional_maybe/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_dcg_loop_empty/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_dcg_loop_three/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_dcg_phrase3/0',         [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_meta_call_and_random, [condition(cpp_compiler_available)]) :-
    % Module-qualified meta-call polish (phrase(user:G, L) +
    % call(user:G, X)) plus the random library: random/1,
    % random_between/3, random_member/2, random_permutation/2,
    % set_random/1. Reproducibility tests use set_random(seed(N))
    % before twin calls and check the results match.
    unique_cpp_tmp_dir('tmp_cpp_e2e_mr', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_phrase_mq/0,
                               user:wam_cpp_test_call_mq/0,
                               user:wam_cpp_test_random_range/0,
                               user:wam_cpp_test_rb_in_range/0,
                               user:wam_cpp_test_rb_seeded_eq/0,
                               user:wam_cpp_test_rb_low_eq_high/0,
                               user:wam_cpp_test_rb_low_gt_high/0,
                               user:wam_cpp_test_rm_in_list/0,
                               user:wam_cpp_test_rm_empty/0,
                               user:wam_cpp_test_rp_length/0,
                               user:wam_cpp_test_rp_invariant/0,
                               user:wam_cpp_test_rp_empty/0,
                               % DCG-expanded predicate dcg_greeting/2 is
                               % already in user from the cpp_e2e_dcg
                               % bundle's defining clauses; we re-include
                               % it here for this bundle's link.
                               user:dcg_greeting/2],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_phrase_mq/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_call_mq/0',         [], true),
          run_query(BinPath, 'wam_cpp_test_random_range/0',    [], true),
          run_query(BinPath, 'wam_cpp_test_rb_in_range/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_rb_seeded_eq/0',    [], true),
          run_query(BinPath, 'wam_cpp_test_rb_low_eq_high/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_rb_low_gt_high/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_rm_in_list/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_rm_empty/0',        [], true),
          run_query(BinPath, 'wam_cpp_test_rp_length/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_rp_invariant/0',    [], true),
          run_query(BinPath, 'wam_cpp_test_rp_empty/0',        [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_sort_customization, [condition(cpp_compiler_available)]) :-
    % sort/4: Key (0 for whole-term, N for Nth compound arg) +
    % Order (@< / @=< / @> / @>=). @< and @> remove duplicate keys;
    % @=< and @>= keep them.
    % predsort/3: stdlib mergesort with user comparator that returns
    % <, =, or >. Equal pairs are deduped. Implemented as ordinary
    % user-module Prolog asserted at wam_cpp_target.pl load time so
    % the existing compile path handles it -- caller must include
    % the four helper predicates in addition to predsort/3 itself.
    unique_cpp_tmp_dir('tmp_cpp_e2e_sortp', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_sort4_asc_dedup/0,
                               user:wam_cpp_test_sort4_asc_keep/0,
                               user:wam_cpp_test_sort4_desc_dedup/0,
                               user:wam_cpp_test_sort4_desc_keep/0,
                               user:wam_cpp_test_sort4_by_first/0,
                               user:wam_cpp_test_sort4_empty/0,
                               user:wam_cpp_test_predsort_asc/0,
                               user:wam_cpp_test_predsort_desc/0,
                               user:wam_cpp_test_predsort_by_key/0,
                               user:wam_cpp_test_predsort_empty/0,
                               user:wam_cpp_test_predsort_single/0,
                               user:dcg_cmp_int/3,
                               user:dcg_cmp_int_desc/3,
                               user:dcg_cmp_by_first/3,
                               % predsort/3 stdlib helpers asserted by
                               % wam_cpp_target.pl at module load.
                               user:predsort/3,
                               user:wam_cpp_predsort_/5,
                               user:wam_cpp_sort2/4,
                               user:wam_cpp_predmerge/4,
                               user:wam_cpp_predmerge_/7],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_sort4_asc_dedup/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_sort4_asc_keep/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_sort4_desc_dedup/0', [], true),
          run_query(BinPath, 'wam_cpp_test_sort4_desc_keep/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_sort4_by_first/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_sort4_empty/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_predsort_asc/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_predsort_desc/0',    [], true),
          run_query(BinPath, 'wam_cpp_test_predsort_by_key/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_predsort_empty/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_predsort_single/0',  [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_stdlib_autoinclude_and_assertion,
     [condition(cpp_compiler_available)]) :-
    % include_stdlib(true) auto-prepends every registered stdlib
    % feature's helpers, so the caller's predicate list doesn't need
    % to spell out predsort/3's 4 helpers. assertion/1 is the new
    % stdlib feature delivered in this PR -- asserts (call(G) -> true
    % ; throw(error(assertion_failed, G))) and so a failed assertion
    % is observable via catch/3.
    unique_cpp_tmp_dir('tmp_cpp_e2e_stdlib', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_predsort_asc/0,
                               user:wam_cpp_test_assertion_ok/0,
                               user:wam_cpp_test_assertion_fail/0,
                               user:wam_cpp_test_assertion_expr/0,
                               user:dcg_cmp_int/3],
                              [emit_main(true), include_stdlib(true)],
                              TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_predsort_asc/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_assertion_ok/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_assertion_fail/0', [], true),
          run_query(BinPath, 'wam_cpp_test_assertion_expr/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_assoc_library, [condition(cpp_compiler_available)]) :-
    % assoc library: empty_assoc/1, put_assoc/4, get_assoc/3,
    % list_to_assoc/2, assoc_to_list/2, assoc_to_keys/2,
    % assoc_to_values/2. Backing rep is an unbalanced BST keyed by
    % standard order (compare/3). All clauses + helpers asserted in
    % the user module at wam_cpp_target.pl load time; pulled in by
    % include_stdlib(assoc). Tests cover empty + insert + lookup +
    % missing-key fail + overwrite + list round-trip (which
    % implicitly checks sorted in-order traversal) + compound keys.
    unique_cpp_tmp_dir('tmp_cpp_e2e_assoc', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_assoc_empty/0,
                               user:wam_cpp_test_assoc_put_get/0,
                               user:wam_cpp_test_assoc_missing/0,
                               user:wam_cpp_test_assoc_overwrite/0,
                               user:wam_cpp_test_assoc_list_sorted/0,
                               user:wam_cpp_test_assoc_keys/0,
                               user:wam_cpp_test_assoc_values/0,
                               user:wam_cpp_test_assoc_compound_keys/0,
                               user:wam_cpp_assoc_depth/2,
                               user:wam_cpp_test_assoc_avl_balance_sorted/0,
                               user:wam_cpp_test_assoc_avl_balance_descending/0,
                               user:wam_cpp_test_assoc_avl_balance_zigzag/0,
                               user:wam_cpp_test_assoc_min/0,
                               user:wam_cpp_test_assoc_max/0,
                               user:wam_cpp_test_assoc_min_after_inserts/0,
                               user:wam_cpp_test_assoc_del_leaf/0,
                               user:wam_cpp_test_assoc_del_root/0,
                               user:wam_cpp_test_assoc_del_missing_fails/0,
                               user:wam_cpp_test_assoc_del_all_back_to_empty/0,
                               user:wam_cpp_test_assoc_del_rebalance_sorted/0,
                               user:wam_cpp_test_assoc_del_returns_value/0,
                               user:wam_cpp_test_assoc_del_min/0,
                               user:wam_cpp_test_assoc_del_max/0,
                               user:wam_cpp_test_assoc_del_min_empty_fails/0,
                               user:wam_cpp_test_assoc_del_max_empty_fails/0,
                               user:wam_cpp_test_assoc_pq_extract_min/0,
                               user:wam_cpp_test_assoc_get5_replace/0,
                               user:wam_cpp_test_assoc_get5_missing/0,
                               user:wam_cpp_test_assoc_get5_threads_old_value/0,
                               user:wam_cpp_assoc_double/2,
                               user:wam_cpp_test_assoc_map/0,
                               user:wam_cpp_test_assoc_map_empty/0],
                              [emit_main(true), include_stdlib(assoc)],
                              TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_assoc_empty/0',                  [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_put_get/0',                [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_missing/0',                [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_overwrite/0',              [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_list_sorted/0',            [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_keys/0',                   [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_values/0',                 [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_compound_keys/0',          [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_avl_balance_sorted/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_avl_balance_descending/0', [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_avl_balance_zigzag/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_min/0',                    [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_max/0',                    [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_min_after_inserts/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_del_leaf/0',               [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_del_root/0',               [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_del_missing_fails/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_del_all_back_to_empty/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_del_rebalance_sorted/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_del_returns_value/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_del_min/0',                [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_del_max/0',                [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_del_min_empty_fails/0',    [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_del_max_empty_fails/0',    [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_pq_extract_min/0',         [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_get5_replace/0',           [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_get5_missing/0',           [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_get5_threads_old_value/0', [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_map/0',                    [], true),
          run_query(BinPath, 'wam_cpp_test_assoc_map_empty/0',              [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_date_time, [condition(cpp_compiler_available)]) :-
    % get_time/1 -> Float seconds since epoch (via chrono).
    % stamp_date_time/3 -> decompose to date/9 in UTC or local.
    % date_time_stamp/2 -> compose back (accepts date/9 or date/6).
    % format_time/3 -> strftime-style atom output (also accepts
    % pre-decomposed date/9 terms in the time arg).
    % Reference stamp: 1704067200 = 2024-01-01 00:00:00 UTC.
    unique_cpp_tmp_dir('tmp_cpp_e2e_dt', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_get_time_positive/0,
                               user:wam_cpp_test_stamp_utc/0,
                               user:wam_cpp_test_stamp_subsec/0,
                               user:wam_cpp_test_dt_roundtrip/0,
                               user:wam_cpp_test_date6_to_stamp/0,
                               user:wam_cpp_test_format_basic/0,
                               user:wam_cpp_test_format_year/0,
                               user:wam_cpp_test_format_from_date9/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_get_time_positive/0', [], true),
          run_query(BinPath, 'wam_cpp_test_stamp_utc/0',         [], true),
          run_query(BinPath, 'wam_cpp_test_stamp_subsec/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_dt_roundtrip/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_date6_to_stamp/0',    [], true),
          run_query(BinPath, 'wam_cpp_test_format_basic/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_format_year/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_format_from_date9/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_dcg_call, [condition(cpp_compiler_available)]) :-
    % DCG call//N (DCG-body meta-call). Works via composition of
    % SWI's --> term_expansion (which rewrites call(P) inside a DCG
    % body to call(P, S0, S)) and our existing dispatch_call_meta.
    % No new runtime code; these tests document and lock in the
    % behavior. Covers:
    %   call//1: nullary DCG via call(P).
    %   call//1 with full partial application: call(P(x, y)).
    %   call//2: call(P(x), y) -- one extra arg threaded through.
    %   Module-qualified DCG meta-call (via user: prefix already
    %     present on the rules, exercised by phrase indirection).
    %   Recursive higher-order DCG via call(P) inside rep0/2.
    unique_cpp_tmp_dir('tmp_cpp_e2e_dcgcall', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_dcgcall_n0/0,
                               user:wam_cpp_test_dcgcall_n0_no/0,
                               user:wam_cpp_test_dcgcall_full/0,
                               user:wam_cpp_test_dcgcall_partial/0,
                               user:wam_cpp_test_dcgcall_partial_no/0,
                               user:wam_cpp_test_dcgcall_rep3/0,
                               user:wam_cpp_test_dcgcall_rep0/0,
                               user:wam_cpp_test_dcgcall_rep_no/0,
                               user:dcg_emit_b/2,
                               user:dcg_emit_two/4,
                               user:dcg_emit_a/2,
                               user:dcg_with_call0/2,
                               user:dcg_with_call_full/2,
                               user:dcg_with_call_partial/2,
                               user:dcg_rep0/4],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_dcgcall_n0/0',         [], true),
          run_query(BinPath, 'wam_cpp_test_dcgcall_n0_no/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_dcgcall_full/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_dcgcall_partial/0',    [], true),
          run_query(BinPath, 'wam_cpp_test_dcgcall_partial_no/0', [], true),
          run_query(BinPath, 'wam_cpp_test_dcgcall_rep3/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_dcgcall_rep0/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_dcgcall_rep_no/0',     [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_lists_extra, [condition(cpp_compiler_available)]) :-
    % lists_extra stdlib: pairs_keys/2, pairs_values/2,
    % pairs_keys_values/3, take/3, drop/3, intersection/3, union/3,
    % permutation/2. All implemented as user-module Prolog asserted
    % at wam_cpp_target.pl load and auto-prepended via
    % include_stdlib(lists_extra).
    unique_cpp_tmp_dir('tmp_cpp_e2e_le', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_pairs_keys/0,
                               user:wam_cpp_test_pairs_values/0,
                               user:wam_cpp_test_pairs_kv/0,
                               user:wam_cpp_test_take_some/0,
                               user:wam_cpp_test_take_overlong/0,
                               user:wam_cpp_test_take_zero/0,
                               user:wam_cpp_test_drop_some/0,
                               user:wam_cpp_test_drop_overlong/0,
                               user:wam_cpp_test_drop_zero/0,
                               user:wam_cpp_test_intersection/0,
                               user:wam_cpp_test_intersection_empty/0,
                               user:wam_cpp_test_union/0,
                               user:wam_cpp_test_permutation_check/0,
                               user:wam_cpp_test_permutation_no/0],
                              [emit_main(true),
                               include_stdlib(lists_extra)],
                              TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_pairs_keys/0',         [], true),
          run_query(BinPath, 'wam_cpp_test_pairs_values/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_pairs_kv/0',           [], true),
          run_query(BinPath, 'wam_cpp_test_take_some/0',          [], true),
          run_query(BinPath, 'wam_cpp_test_take_overlong/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_take_zero/0',          [], true),
          run_query(BinPath, 'wam_cpp_test_drop_some/0',          [], true),
          run_query(BinPath, 'wam_cpp_test_drop_overlong/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_drop_zero/0',          [], true),
          run_query(BinPath, 'wam_cpp_test_intersection/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_intersection_empty/0', [], true),
          run_query(BinPath, 'wam_cpp_test_union/0',              [], true),
          run_query(BinPath, 'wam_cpp_test_permutation_check/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_permutation_no/0',     [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_indexing_backtrack, [condition(cpp_compiler_available)]) :-
    % Regression for the SwitchOnTerm-vs-CP-sub-chain bug uncovered
    % by PR #2264. Three-clause predicate with []/[H|T]/[H|T] first
    % args used to lose backtracking when the indexed jump for the
    % compound case skipped the chain's entry TryMeElse. The runtime
    % now flags indexed entries (SwitchOn{Term,Constant,Structure}
    % direct jumps) and the next RetryMeElse synthesizes a fresh CP
    % so the predicate level has its own backtrack handle.
    unique_cpp_tmp_dir('tmp_cpp_e2e_idx', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_idx_flat/0,
                               user:wam_cpp_test_idx_keep/0,
                               user:wam_cpp_test_idx_drop_first/0,
                               user:wam_cpp_test_idx_keep_first/0,
                               user:wam_cpp_test_idx_mixed/0,
                               user:wam_cpp_idx_bar/2],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_idx_flat/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_idx_keep/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_idx_drop_first/0', [], true),
          run_query(BinPath, 'wam_cpp_test_idx_keep_first/0', [], true),
          run_query(BinPath, 'wam_cpp_test_idx_mixed/0',      [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_stream_io, [condition(cpp_compiler_available)]) :-
    % Stream I/O foundation: open/3, close/1, read_line_to_string/2,
    % read_string/5, at_end_of_stream/1, write_to_stream/2,
    % nl_to_stream/1. Each test creates a temp file in /tmp and
    % exercises one path. Files are left in place; /tmp is ephemeral.
    unique_cpp_tmp_dir('tmp_cpp_e2e_stream', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_stream_roundtrip/0,
                               user:wam_cpp_test_stream_read_string/0,
                               user:wam_cpp_test_stream_append/0,
                               user:wam_cpp_test_stream_at_end/0,
                               user:wam_cpp_test_stream_missing_file/0,
                               user:wam_cpp_test_stream_close_unknown/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_stream_roundtrip/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_stream_read_string/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_stream_append/0',        [], true),
          run_query(BinPath, 'wam_cpp_test_stream_at_end/0',        [], true),
          run_query(BinPath, 'wam_cpp_test_stream_missing_file/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_stream_close_unknown/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_introspection, [condition(cpp_compiler_available)]) :-
    % current_predicate/1 (check mode) + predicate_property/2:
    % static / dynamic / defined / number_of_clauses(N), plus the
    % expected throws on missing instantiation and bad-shape PI.
    unique_cpp_tmp_dir('tmp_cpp_e2e_intro', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_intro_static_yes/0,
                               user:wam_cpp_test_intro_static_no/0,
                               user:wam_cpp_test_intro_arity_no/0,
                               user:wam_cpp_test_intro_builtin_yes/0,
                               user:wam_cpp_test_intro_prop_defined/0,
                               user:wam_cpp_test_intro_prop_static/0,
                               user:wam_cpp_test_intro_prop_dynamic/0,
                               user:wam_cpp_test_intro_prop_static_not_dynamic/0,
                               user:wam_cpp_test_intro_prop_count/0,
                               user:wam_cpp_test_intro_inst_throw/0,
                               user:wam_cpp_test_intro_indicator_throw/0,
                               user:wam_cpp_intro_static/1,
                               user:wam_cpp_intro_dyn/1],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_intro_static_yes/0',           [], true),
          run_query(BinPath, 'wam_cpp_test_intro_static_no/0',            [], true),
          run_query(BinPath, 'wam_cpp_test_intro_arity_no/0',             [], true),
          run_query(BinPath, 'wam_cpp_test_intro_builtin_yes/0',          [], true),
          run_query(BinPath, 'wam_cpp_test_intro_prop_defined/0',         [], true),
          run_query(BinPath, 'wam_cpp_test_intro_prop_static/0',          [], true),
          run_query(BinPath, 'wam_cpp_test_intro_prop_dynamic/0',         [], true),
          run_query(BinPath, 'wam_cpp_test_intro_prop_static_not_dynamic/0', [], true),
          run_query(BinPath, 'wam_cpp_test_intro_prop_count/0',           [], true),
          run_query(BinPath, 'wam_cpp_test_intro_inst_throw/0',           [], true),
          run_query(BinPath, 'wam_cpp_test_intro_indicator_throw/0',      [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_fs_and_wos, [condition(cpp_compiler_available)]) :-
    % Filesystem helpers (exists_file/1, exists_directory/1,
    % directory_files/2, make_directory/1, delete_file/1) plus
    % with_output_to(stream(_), Goal) which completes the stream PR
    % by routing the existing output-capture frame to a stream sink.
    unique_cpp_tmp_dir('tmp_cpp_e2e_fsws', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_fs_exists_file_yes/0,
                               user:wam_cpp_test_fs_exists_file_no/0,
                               user:wam_cpp_test_fs_exists_dir_yes/0,
                               user:wam_cpp_test_fs_exists_dir_no/0,
                               user:wam_cpp_test_fs_make_check/0,
                               user:wam_cpp_test_fs_dir_files/0,
                               user:wam_cpp_test_fs_delete/0,
                               user:wam_cpp_test_fs_delete_missing/0,
                               user:wam_cpp_test_wos_stream/0,
                               user:wam_cpp_test_wos_format/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_fs_exists_file_yes/0', [], true),
          run_query(BinPath, 'wam_cpp_test_fs_exists_file_no/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_fs_exists_dir_yes/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_fs_exists_dir_no/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_fs_make_check/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_fs_dir_files/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_fs_delete/0',          [], true),
          run_query(BinPath, 'wam_cpp_test_fs_delete_missing/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_wos_stream/0',         [], true),
          run_query(BinPath, 'wam_cpp_test_wos_format/0',         [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_clause_runtime, [condition(cpp_compiler_available)]) :-
    % Runtime clause/2 for dynamic predicates. Each test seeds the
    % dynamic database with assertz at startup (clause/2 only sees
    % runtime-asserted clauses, not the statically compiled ones)
    % then enumerates / matches. Covers facts, bound + var heads,
    % findall over the enumeration, rule clauses with bodies, miss,
    % unknown predicate, and assertz-then-enumerate.
    unique_cpp_tmp_dir('tmp_cpp_e2e_clause', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_clause_fact/0,
                               user:wam_cpp_test_clause_var/0,
                               user:wam_cpp_test_clause_enum/0,
                               user:wam_cpp_test_clause_rule_body/0,
                               user:wam_cpp_test_clause_missing/0,
                               user:wam_cpp_test_clause_unknown/0,
                               user:wam_cpp_test_clause_after_assertz/0,
                               user:wam_cpp_clause_setup_p/0,
                               user:wam_cpp_clause_setup_q/0,
                               user:wam_cpp_clause_p/1,
                               user:wam_cpp_clause_q/2],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_clause_fact/0',          [], true),
          run_query(BinPath, 'wam_cpp_test_clause_var/0',           [], true),
          run_query(BinPath, 'wam_cpp_test_clause_enum/0',          [], true),
          run_query(BinPath, 'wam_cpp_test_clause_rule_body/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_clause_missing/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_clause_unknown/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_clause_after_assertz/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_findall_nested_template, [condition(cpp_compiler_available)]) :-
    % Regression for the findall nested compound template bug
    % uncovered during PR #2286 development. The compile path used
    % to throw unsupported_template_arg on any compound arg of a
    % findall template, and the surrounding catch silently dropped
    % the predicate from the output. The fix in compile_compound_
    % template recurses into nested compounds using set_variable
    % at the parent level + put_structure on the sub-reg. Tests
    % cover left-nested (-/2-2x), right-nested, mixed-functor
    % nests, and templates containing a literal atom inside the
    % nested compound.
    unique_cpp_tmp_dir('tmp_cpp_e2e_nest', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_nested_dash/0,
                               user:wam_cpp_test_nested_f_g/0,
                               user:wam_cpp_test_nested_right/0,
                               user:wam_cpp_test_nested_constant/0,
                               user:wam_cpp_nest_setup/0,
                               user:wam_cpp_nest_emit/3],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_nested_dash/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_nested_f_g/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_nested_right/0',    [], true),
          run_query(BinPath, 'wam_cpp_test_nested_constant/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_current_predicate_nondet, [condition(cpp_compiler_available)]) :-
    % current_predicate/1 nondet enum via the CP-iterator path.
    % Re-uses the dispatch_current_predicate + current_pred_try_next
    % infrastructure introduced in this PR. Tests cover check mode
    % (parity with PR #2277), enum by partial spec (Name/_, _/Arity,
    % _/_), the empty-match case, and the two throw paths.
    unique_cpp_tmp_dir('tmp_cpp_e2e_cpnondet', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_cp_check_static/0,
                               user:wam_cpp_test_cp_check_missing/0,
                               user:wam_cpp_test_cp_enum_arity/0,
                               user:wam_cpp_test_cp_enum_name_by_arity/0,
                               user:wam_cpp_test_cp_enum_all/0,
                               user:wam_cpp_test_cp_enum_none/0,
                               user:wam_cpp_test_cp_inst_throw/0,
                               user:wam_cpp_test_cp_indicator_throw/0,
                               user:wam_cpp_cp_static_a/1,
                               user:wam_cpp_cp_static_b/2],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_cp_check_static/0',        [], true),
          run_query(BinPath, 'wam_cpp_test_cp_check_missing/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_cp_enum_arity/0',          [], true),
          run_query(BinPath, 'wam_cpp_test_cp_enum_name_by_arity/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_cp_enum_all/0',            [], true),
          run_query(BinPath, 'wam_cpp_test_cp_enum_none/0',           [], true),
          run_query(BinPath, 'wam_cpp_test_cp_inst_throw/0',          [], true),
          run_query(BinPath, 'wam_cpp_test_cp_indicator_throw/0',     [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_builtin_io, [condition(cpp_compiler_available)]) :-
    % write/1 + nl/0 should print "hello\n" before the driver prints
    % "true". Captures full stdout (not just the last line).
    unique_cpp_tmp_dir('tmp_cpp_e2e_io', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_write/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          process_create(BinPath, ['wam_cpp_test_write/0'],
                         [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, Output),
          close(Out),
          process_wait(PID, _),
          normalize_space(string(Trimmed), Output),
          assertion(Trimmed == "hello true")
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% findall/3 + aggregate_all/3 — exercises BeginAggregate / EndAggregate
% with all standard aggregate kinds (collect / count / sum / min / max / set).
% ------------------------------------------------------------------

test(cpp_e2e_findall, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_findall', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_item/1,
                               user:wam_cpp_test_findall/0,
                               user:wam_cpp_test_findall_empty/0,
                               user:wam_cpp_test_findall_doubled/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_findall/0',         [], true),
          run_query(BinPath, 'wam_cpp_test_findall_empty/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_findall_doubled/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_bagof_setof, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_bagof_setof', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_item/1, user:wam_cpp_num/1,
                               user:wam_cpp_test_bagof/0,
                               user:wam_cpp_test_bagof_empty/0,
                               user:wam_cpp_test_setof/0,
                               user:wam_cpp_test_setof_empty/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_bagof/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_bagof_empty/0', [], false),
          run_query(BinPath, 'wam_cpp_test_setof/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_setof_empty/0', [], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_aggregate_all, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_agg', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_item/1, user:wam_cpp_num/1,
                               user:wam_cpp_test_count/0, user:wam_cpp_test_sum/0,
                               user:wam_cpp_test_min/0,   user:wam_cpp_test_max/0,
                               user:wam_cpp_test_set/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_count/0', [], true),
          run_query(BinPath, 'wam_cpp_test_sum/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_min/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_max/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_set/0',   [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% Environment frames: Y-reg isolation across nested calls + cp threading
% through tail-recursive arithmetic. Both are correctness bugs that
% existed in #2036 and are fixed by this PR''s env-frame implementation
% (Allocate pushes a frame saving cp; Deallocate pops + restores;
% Y-reg lookup is scoped to the top frame).
% ------------------------------------------------------------------

test(cpp_e2e_yreg_isolation, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_yreg', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_num/1,
                               user:wam_cpp_h1/1, user:wam_cpp_h2/1,
                               user:wam_cpp_two_helpers/0,
                               user:wam_cpp_two_helpers_swap/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          % Both helpers use Y1/Y2 internally. The caller calls h1 then h2
          % and must NOT see h2''s Y1 stomp on h1''s result.
          run_query(BinPath, 'wam_cpp_two_helpers/0',      [], true),
          run_query(BinPath, 'wam_cpp_two_helpers_swap/0', [], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_recursive_arithmetic, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_recur', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_length_acc/3,
                               user:wam_cpp_list_length/2,
                               user:wam_cpp_test_len_empty/0,
                               user:wam_cpp_test_len_one/0,
                               user:wam_cpp_test_len_three/0,
                               user:wam_cpp_test_len_five/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          % Tail-recursive length with accumulator. Exercises:
          %   - cp threading through nested Call/Execute
          %   - Y-reg isolation across recursive frames
          %   - PutConstant allocating fresh cells (not mutating
          %     X-reg-aliased cells)
          run_query(BinPath, 'wam_cpp_test_len_empty/0', [], true),
          run_query(BinPath, 'wam_cpp_test_len_one/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_len_three/0', [], true),
          run_query(BinPath, 'wam_cpp_test_len_five/0',  [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_builtin_unification, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_unif', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_unify/0,
                               user:wam_cpp_test_unify_fail/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_unify/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_unify_fail/0', [], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_list_head_match, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_list', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_lst/1],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          % lst([a, b, c]). Exercises GetList + UnifyConstant +
          % UnifyVariable + GetStructure([|]/2) cell-by-cell.
          run_query(BinPath, 'wam_cpp_lst/1', ['[a,b,c]'], true),
          run_query(BinPath, 'wam_cpp_lst/1', ['[a,b]'],   false),
          run_query(BinPath, 'wam_cpp_lst/1', ['[a,b,d]'], false),
          run_query(BinPath, 'wam_cpp_lst/1', ['[]'],      false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% Indexing instructions: switch_on_constant (atoms / integers) +
% switch_on_term (typed dispatch with structure / list handling).
% Exercises constant-bound A1 dispatch (color, shape) and the
% combined type dispatch (mixed atom/int/struct/list).
% ------------------------------------------------------------------

% ------------------------------------------------------------------
% List & term builtins: member/2, length/2, copy_term/2. member and
% length are auto-injected as helper predicates (so they can backtrack
% naturally through their two clauses); copy_term is a direct builtin
% with structural deep-copy and shared-variable renaming.
% ------------------------------------------------------------------

test(cpp_e2e_member, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_member', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_member_yes/0,
                               user:wam_cpp_test_member_no/0,
                               user:wam_cpp_test_member_first/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_member_yes/0',   [], true),
          run_query(BinPath, 'wam_cpp_test_member_no/0',    [], false),
          run_query(BinPath, 'wam_cpp_test_member_first/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_length, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_length', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_length_three/0,
                               user:wam_cpp_test_length_zero/0,
                               user:wam_cpp_test_length_bad/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_length_three/0', [], true),
          run_query(BinPath, 'wam_cpp_test_length_zero/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_length_bad/0',   [], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_copy_term, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_copy', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_copy_basic/0,
                               user:wam_cpp_test_copy_atom/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          % copy_term(foo(X,X,Y), T) → T = foo(A,A,B) with A and B fresh.
          % The two X-positions in source must share a single fresh var
          % in the copy; Y becomes a different fresh var.
          run_query(BinPath, 'wam_cpp_test_copy_basic/0', [], true),
          run_query(BinPath, 'wam_cpp_test_copy_atom/0',  [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_member_enumeration, [condition(cpp_compiler_available)]) :-
    % findall enumerating through member is the full nondet test:
    % member must push a choice point on each match so the driver can
    % backtrack into it for the next solution.
    unique_cpp_tmp_dir('tmp_cpp_e2e_enum', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_enum_member/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_enum_member/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% Exception handling: catch/3 + throw/1. catch/3 pushes a side-stack
% CatcherFrame and dispatches the protected goal as a tail-call to an
% auto-injected CatchReturn instruction; throw/1 walks the catcher
% stack, unwinds VM state for each frame, and invokes the first
% matching frame''s recovery goal. Uncaught throws print to stderr and
% return false; backtrack() pops catcher frames whose protected goal
% exhausted solutions without throwing.
% ------------------------------------------------------------------

test(cpp_e2e_catch_caught, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_catch_caught', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_catch_basic/0,
                               user:wam_cpp_test_catch_pass/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_catch_basic/0', [], true),
          run_query(BinPath, 'wam_cpp_test_catch_pass/0',  [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_catch_uncaught, [condition(cpp_compiler_available)]) :-
    % An uncaught throw walks past all frames and exits with false,
    % printing "uncaught exception: <term>" to stderr (which run_query
    % discards). The query result on stdout is "false".
    unique_cpp_tmp_dir('tmp_cpp_e2e_catch_uncaught', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_catch_no_match/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_catch_no_match/0', [], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_catch_nested, [condition(cpp_compiler_available)]) :-
    % Inner catcher doesn''t unify with the thrown term, so its frame
    % is popped and the throw walk continues to the outer catcher,
    % which matches and runs its recovery.
    unique_cpp_tmp_dir('tmp_cpp_e2e_catch_nested', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_catch_nested/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_catch_nested/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_catch_fail_propagates, [condition(cpp_compiler_available)]) :-
    % catch(fail, _, true) — the goal fails without throwing, so the
    % failure propagates past catch (recovery is NOT invoked).
    % backtrack() pops the catcher frame when CPs run below its base.
    unique_cpp_tmp_dir('tmp_cpp_e2e_catch_fail', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_catch_fail/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_catch_fail/0', [], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_catch_compound_pattern, [condition(cpp_compiler_available)]) :-
    % error(type_error, ctx) thrown; catcher pattern error(Kind, _)
    % unifies, binding Kind. Recovery confirms binding.
    unique_cpp_tmp_dir('tmp_cpp_e2e_catch_cmpd', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_catch_compound/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_catch_compound/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% List-builtin batch 2: append/3, reverse/2, last/2, nth0/3. All four
% are auto-injected helper predicates (same mechanism as member/2 +
% length/2 from the prior batch). reverse/2 dispatches to a helper
% reverse_acc/3 (also injected) for tail-recursive accumulator form.
% nth0/3 exercises the helper path with arithmetic builtins (>/2 and
% is/2) in the recursive clause.
% ------------------------------------------------------------------

test(cpp_e2e_append, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_append', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_append_basic/0,
                               user:wam_cpp_test_append_empty_first/0,
                               user:wam_cpp_test_append_empty_second/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_append_basic/0',        [], true),
          run_query(BinPath, 'wam_cpp_test_append_empty_first/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_append_empty_second/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_reverse, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_reverse', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_reverse_basic/0,
                               user:wam_cpp_test_reverse_empty/0,
                               user:wam_cpp_test_reverse_singleton/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_reverse_basic/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_reverse_empty/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_reverse_singleton/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_last, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_last', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_last_basic/0,
                               user:wam_cpp_test_last_single/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_last_basic/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_last_single/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_nth0, [condition(cpp_compiler_available)]) :-
    % nth0''s recursive clause uses >/2 and is/2 — verifies that the
    % helper-injection path interoperates with arithmetic builtins.
    unique_cpp_tmp_dir('tmp_cpp_e2e_nth0', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_nth0_first/0,
                               user:wam_cpp_test_nth0_middle/0,
                               user:wam_cpp_test_nth0_last/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_nth0_first/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_nth0_middle/0', [], true),
          run_query(BinPath, 'wam_cpp_test_nth0_last/0',   [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% format/1 + format/2: tilde-directive formatted printing to stdout.
% Compiled as `execute format/N`, which now falls back to builtin()
% in step()''s Execute/Call arms when no user label matches. Asserts
% exact captured stdout to verify directive expansion is correct.
% ------------------------------------------------------------------

test(cpp_e2e_format_noargs, [condition(cpp_compiler_available)]) :-
    % format/1 takes only a format string. Exercises the 1-arity
    % dispatch path (no args list to walk).
    unique_cpp_tmp_dir('tmp_cpp_e2e_fmt1', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_fmt1_noargs/0,
                               user:wam_cpp_test_fmt2_no_directives/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_stdout(BinPath, 'wam_cpp_test_fmt1_noargs/0', [],
                           true, "plain text\n"),
          run_query_stdout(BinPath, 'wam_cpp_test_fmt2_no_directives/0', [],
                           true, "hello world")
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_format_atoms_and_ints, [condition(cpp_compiler_available)]) :-
    % ~w (write) on atoms and ~d (integer) directives.
    unique_cpp_tmp_dir('tmp_cpp_e2e_fmt2_ai', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_fmt2_atoms/0,
                               user:wam_cpp_test_fmt2_ints/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_stdout(BinPath, 'wam_cpp_test_fmt2_atoms/0', [],
                           true, "a=hello b=world\n"),
          run_query_stdout(BinPath, 'wam_cpp_test_fmt2_ints/0', [],
                           true, "1 + 2 = 3\n")
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_format_compound, [condition(cpp_compiler_available)]) :-
    % ~w on a compound term goes through render(), exercising the
    % full Value printer.
    unique_cpp_tmp_dir('tmp_cpp_e2e_fmt2_cmpd', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_fmt2_compound/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_stdout(BinPath, 'wam_cpp_test_fmt2_compound/0', [],
                           true, "result: foo(1, bar)\n")
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_format_tilde_escape, [condition(cpp_compiler_available)]) :-
    % ~~ emits a literal tilde. The format string ''100~~~n'' contains
    % "100" + "~~" (literal tilde) + "~n" (newline) = "100~\n".
    unique_cpp_tmp_dir('tmp_cpp_e2e_fmt2_tilde', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_fmt2_tilde/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_stdout(BinPath, 'wam_cpp_test_fmt2_tilde/0', [],
                           true, "100~\n")
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% ISO arithmetic — first ISO-aware builtin (is_iso/2 + is_lax/2).
% Each test flips the relevant test predicate to ISO mode via the
% inline `iso_errors(PI, true)` option; the explicit-lax test ALSO
% flips its enclosing predicate to ISO mode and verifies the
% explicit is_lax/2 call site survives the rewrite (three-forms
% guarantee from WAM_CPP_ISO_ERRORS_PHILOSOPHY §3.3).
% ------------------------------------------------------------------

test(cpp_e2e_iso_is_throws_type_error, [condition(cpp_compiler_available)]) :-
    % ISO mode + non-evaluable atom → catcher with
    % error(type_error(evaluable, _), _) matches; recovery runs.
    unique_cpp_tmp_dir('tmp_cpp_e2e_iso_is_type', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_iso_is_type_error/0],
                              [emit_main(true),
                               iso_errors(wam_cpp_test_iso_is_type_error/0,
                                          true)],
                              TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_iso_is_type_error/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_iso_is_throws_instantiation, [condition(cpp_compiler_available)]) :-
    % ISO mode + RHS contains unbound → catcher with
    % error(instantiation_error, _) matches.
    unique_cpp_tmp_dir('tmp_cpp_e2e_iso_is_inst', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_iso_is_instantiation/0],
                              [emit_main(true),
                               iso_errors(wam_cpp_test_iso_is_instantiation/0,
                                          true)],
                              TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_iso_is_instantiation/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_iso_is_unmatched_propagates,
     [condition(cpp_compiler_available)]) :-
    % ISO mode throws type_error, but the catcher pattern is
    % error(some_other_kind, _) which doesn''t unify. Throw walks
    % past the catcher → uncaught → false on stdout, "uncaught
    % exception" diagnostic on stderr (which run_query discards).
    unique_cpp_tmp_dir('tmp_cpp_e2e_iso_is_unmatched', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_iso_is_unmatched/0],
                              [emit_main(true),
                               iso_errors(wam_cpp_test_iso_is_unmatched/0,
                                          true)],
                              TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_iso_is_unmatched/0', [], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_lax_is_silent_fail, [condition(cpp_compiler_available)]) :-
    % Default-mode predicate: X is foo silently fails. The
    % (-> ; true) wraps that into success → true.
    unique_cpp_tmp_dir('tmp_cpp_e2e_lax_is', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_lax_is_silent/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_lax_is_silent/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_explicit_lax_in_iso_predicate,
     [condition(cpp_compiler_available)]) :-
    % Three-forms guarantee: predicate is flipped to ISO mode, but
    % uses is_lax(X, foo) directly. The rewrite must NOT touch the
    % explicit lax key; behavior stays lax (silent fail wrapped
    % into success by -> ; true). If the rewrite incorrectly
    % converted is_lax/2 → is_iso/2, throw would propagate and the
    % program would not return true cleanly.
    unique_cpp_tmp_dir('tmp_cpp_e2e_explicit_lax_in_iso', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_explicit_lax_in_iso/0],
                              [emit_main(true),
                               iso_errors(wam_cpp_test_explicit_lax_in_iso/0,
                                          true)],
                              TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_explicit_lax_in_iso/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_iso_unbound_context, [condition(cpp_compiler_available)]) :-
    % Catcher pattern is error(type_error(evaluable, Culprit), _) —
    % the Context slot is bound to a fresh unbound, but the catch
    % still succeeds because unification with an unbound is
    % unconditional. Recovery verifies Culprit binds to foo/0 (per
    % SPEC §6 culprit-shape rule). Regression guard against the
    % decision to leave Context unbound for v1.
    unique_cpp_tmp_dir('tmp_cpp_e2e_iso_unbound_ctx', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_iso_unbound_context/0],
                              [emit_main(true),
                               iso_errors(wam_cpp_test_iso_unbound_context/0,
                                          true)],
                              TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_iso_unbound_context/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% ISO sweep — arithmetic comparisons + succ + IEEE-754 lax floats
% (PR #3 of the ISO series). Builds on the is_iso/2 + is_lax/2
% infrastructure from #2084. Verifies:
%   - >_iso/2, <_iso/2, =:=_iso/2 throw the right errors.
%   - >_lax/2 inside an ISO-mode predicate survives the rewrite.
%   - succ_iso/2 throws type_error / instantiation_error per §6.
%   - Lax float divide-by-zero produces inf / NaN per §6.1.
%   - ISO float and integer divide-by-zero both throw
%     evaluation_error(zero_divisor).
% ------------------------------------------------------------------

test(cpp_e2e_iso_compare_throws_inst, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_iso_gt_inst', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_iso_gt_throws_inst/0],
                              [emit_main(true),
                               iso_errors(wam_cpp_test_iso_gt_throws_inst/0,
                                          true)],
                              TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_iso_gt_throws_inst/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_iso_compare_throws_type, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_iso_lt_type', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_iso_lt_throws_type/0],
                              [emit_main(true),
                               iso_errors(wam_cpp_test_iso_lt_throws_type/0,
                                          true)],
                              TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_iso_lt_throws_type/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_iso_compare_throws_zero_div,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_iso_eq_zerodiv', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_iso_eq_throws_zero_div/0],
                              [emit_main(true),
                               iso_errors(wam_cpp_test_iso_eq_throws_zero_div/0,
                                          true)],
                              TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_iso_eq_throws_zero_div/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_lax_compare_silent_fail, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_lax_gt', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_lax_gt_silent_fail/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_lax_gt_silent_fail/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_explicit_lax_compare_in_iso,
     [condition(cpp_compiler_available)]) :-
    % Three-forms guarantee for arith compares: explicit `>_lax/2`
    % inside an ISO-mode predicate must survive the rewrite (silent
    % fail), not be upgraded to `>_iso/2` (which would throw).
    unique_cpp_tmp_dir('tmp_cpp_e2e_explicit_lax_gt', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_explicit_lax_gt_in_iso/0],
                              [emit_main(true),
                               iso_errors(wam_cpp_test_explicit_lax_gt_in_iso/0,
                                          true)],
                              TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_explicit_lax_gt_in_iso/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_iso_succ_negative_throws,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_iso_succ_neg', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_iso_succ_neg_throws/0],
                              [emit_main(true),
                               iso_errors(wam_cpp_test_iso_succ_neg_throws/0,
                                          true)],
                              TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_iso_succ_neg_throws/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_iso_succ_unbound_throws,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_iso_succ_unbound', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_iso_succ_unbound_throws/0],
                              [emit_main(true),
                               iso_errors(wam_cpp_test_iso_succ_unbound_throws/0,
                                          true)],
                              TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_iso_succ_unbound_throws/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_iso_float_div_zero_throws,
     [condition(cpp_compiler_available)]) :-
    % ISO mode catches float divide-by-zero too (lax would silently
    % succeed with inf — see the next test for that side).
    unique_cpp_tmp_dir('tmp_cpp_e2e_iso_zero_div', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_iso_zero_div_throws/0],
                              [emit_main(true),
                               iso_errors(wam_cpp_test_iso_zero_div_throws/0,
                                          true)],
                              TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_iso_zero_div_throws/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_lax_float_div_zero_inf,
     [condition(cpp_compiler_available)]) :-
    % SPEC §6.1 lax behavior change: float divide-by-zero now
    % produces inf instead of failing. Verifies R > 1e308 to avoid
    % depending on Value::Float''s text rendering of "inf".
    unique_cpp_tmp_dir('tmp_cpp_e2e_lax_inf', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_lax_float_div_zero_inf/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_lax_float_div_zero_inf/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_lax_float_div_zero_nan,
     [condition(cpp_compiler_available)]) :-
    % SPEC §6.1: 0.0 / 0.0 produces NaN. Verified via NaN \=:= NaN
    % which is the IEEE-754 self-comparison signature.
    unique_cpp_tmp_dir('tmp_cpp_e2e_lax_nan', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_lax_float_div_zero_nan/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_lax_float_div_zero_nan/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% \+/1 and not/1 — negation as failure. Implemented via a
% NegationFrame side stack symmetric to the catcher_frames machinery:
% the protected goal is dispatched with cp set to a synthetic
% NegationReturn instruction; if the goal succeeds (lands on
% NegationReturn) the negation fails; if the goal fails (CPs drain
% to the frame''s base) backtrack() pops the frame and the negation
% succeeds at the saved continuation.
% ------------------------------------------------------------------

test(cpp_e2e_not_fail, [condition(cpp_compiler_available)]) :-
    % `\+ fail` — goal fails, negation succeeds → true.
    unique_cpp_tmp_dir('tmp_cpp_e2e_not_fail', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_not_fail/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_not_fail/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_not_true_fails, [condition(cpp_compiler_available)]) :-
    % `\+ true` — goal succeeds, negation fails → false.
    unique_cpp_tmp_dir('tmp_cpp_e2e_not_true', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_not_true/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_not_true/0', [], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_not_compound_conjunction,
     [condition(cpp_compiler_available)]) :-
    % `\+ (X = a, X = b)` — conjunction-as-data goes through
    % put_structure ,/2 → invoke_goal_as_call dispatches it. The
    % conjunction cannot succeed (X can''t be both a and b), so the
    % negation succeeds. Regression guard against the ,/2-tokenizer
    % issue from PR #2084 surfacing on the negation path too.
    unique_cpp_tmp_dir('tmp_cpp_e2e_not_compound', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_not_compound/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_not_compound/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_not_alias, [condition(cpp_compiler_available)]) :-
    % not/1 is an alias for \+/1. Test both success (not(fail)) and
    % failure (not(true)) sides to verify the dispatch covers both
    % keys equivalently.
    unique_cpp_tmp_dir('tmp_cpp_e2e_not_alias', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_not_alias_succeeds/0,
                               user:wam_cpp_test_not_alias_fails/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_not_alias_succeeds/0', [], true),
          run_query(BinPath,
                    'wam_cpp_test_not_alias_fails/0', [], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_not_nan_check, [condition(cpp_compiler_available)]) :-
    % The gap that motivated this PR: NaN self-check via \+ (=:=).
    % R is 0.0/0.0 → NaN; NaN =:= NaN is false (per IEEE 754); \+
    % flips that to true. Verifies the negation goal-dispatch path
    % AND the IEEE-754 lax float-divide path interoperate.
    unique_cpp_tmp_dir('tmp_cpp_e2e_not_nan', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_not_nan_check/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_not_nan_check/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% call/N — meta-call. `call(Goal)` dispatches Goal as a goal;
% `call(Goal, X1, ..., XK)` appends X1..XK to Goal''s existing args
% and dispatches the resulting goal. The WAM compiler emits this as
% `execute call/N` (tail) or `call call/N, N` (non-tail), with Goal
% in A1 and the extras in A2..AN. dispatch_call_meta builds the
% combined goal term and routes through invoke_goal_as_call (the
% same path catch/3 and \+/1 use).
% ------------------------------------------------------------------

test(cpp_e2e_call_atom, [condition(cpp_compiler_available)]) :-
    % call(true) — tail-call dispatch path through the Execute arm
    % (where instr.n is 0 since Execute instructions don''t carry
    % arity; dispatch_call_meta parses arity from the op-name
    % suffix). Regression guard for that specific Execute/Call
    % asymmetry.
    unique_cpp_tmp_dir('tmp_cpp_e2e_call_atom', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_call_atom/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_call_atom/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_call_with_args, [condition(cpp_compiler_available)]) :-
    % call(=, X, 5) builds =(X, 5) and dispatches.
    unique_cpp_tmp_dir('tmp_cpp_e2e_call_args', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_call_with_args/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_call_with_args/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_call_partial, [condition(cpp_compiler_available)]) :-
    % G = =(X), call(G, 7) — extras append to existing compound args.
    unique_cpp_tmp_dir('tmp_cpp_e2e_call_partial', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_call_partial/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_call_partial/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_call_compound_already,
     [condition(cpp_compiler_available)]) :-
    % G = full_goal, call(G) — no extras, already-complete compound.
    % Tests the call/1 path with a compound argument.
    unique_cpp_tmp_dir('tmp_cpp_e2e_call_compound', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_call_compound_already/0,
                               user:wam_cpp_call_helper/1],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_call_compound_already/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_call_user_pred, [condition(cpp_compiler_available)]) :-
    % call(F, X) dispatching to a USER predicate (not a builtin).
    % Tests the user-label dispatch path inside invoke_goal_as_call.
    unique_cpp_tmp_dir('tmp_cpp_e2e_call_user', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_call_user_pred/0,
                               user:wam_cpp_call_helper/1],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_call_user_pred/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% maplist/2 + maplist/3 — higher-order list mapping. Helper-injected
% (per WAM_ITEMS_API §6) on top of call/N. maplist/2 is "predicate
% holds for every element"; maplist/3 is "P transforms each X to Y".
% The maplist/3 + call/3 + user-predicate composition is the key
% demonstration that higher-order programming works end-to-end on
% the C++ target.
%
% Also tests `findall + call/N` composition — a common idiom for
% "collect transformed values."
% ------------------------------------------------------------------

test(cpp_e2e_maplist2_all, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_ml2_all', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_maplist2_all/0,
                               user:wam_cpp_positive/1],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_maplist2_all/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_maplist2_empty, [condition(cpp_compiler_available)]) :-
    % maplist/2 base case — empty list succeeds trivially.
    unique_cpp_tmp_dir('tmp_cpp_e2e_ml2_empty', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_maplist2_empty/0,
                               user:wam_cpp_positive/1],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_maplist2_empty/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_maplist3_double, [condition(cpp_compiler_available)]) :-
    % The key test: maplist/3 + call/3 + a USER predicate.
    % maplist(double, [1,2,3], L) walks the input list, calling
    % double/2 via call/3 on each element. The bug that broke
    % this before the PutStructure-fresh-cell fix: when
    % invoke_goal_as_call set A1 from the goal''s args and the
    % goal body did PutStructure into A2 (e.g. */2 for Y is X*2),
    % the existing-cell-bind optimisation in begin_write wrote
    % into a cell still aliased with A1. Fix: PutStructure and
    % PutList now always allocate fresh.
    unique_cpp_tmp_dir('tmp_cpp_e2e_ml3_double', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_maplist3_double/0,
                               user:wam_cpp_double/2],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_maplist3_double/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_maplist3_check, [condition(cpp_compiler_available)]) :-
    % Both lists ground — maplist/3 in checking mode: verifies
    % double(X, Y) holds for each paired (X, Y).
    unique_cpp_tmp_dir('tmp_cpp_e2e_ml3_check', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_maplist3_check/0,
                               user:wam_cpp_double/2],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_maplist3_check/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_findall_call_compose,
     [condition(cpp_compiler_available)]) :-
    % findall with call/N inside its goal. Verifies the aggregate
    % frame + meta-call composition.
    unique_cpp_tmp_dir('tmp_cpp_e2e_findall_call', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_findall_call/0,
                               user:wam_cpp_double/2],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_findall_call/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% findall + conjunction with backtracking. PR #2097 flagged these
% as "hangs" — turned out to be the PutStructure aliasing bug fixed
% in the same PR. Now that aliasing through A-regs allocates fresh,
% these compose naturally. The tests here lock that in.
% ------------------------------------------------------------------

test(cpp_e2e_findall_member_arith,
     [condition(cpp_compiler_available)]) :-
    % findall(Y, (member(X, [...]), Y is X * 2), L) — member
    % backtracks, Y gets computed each time, all Ys collected.
    % Was the original reproduction in the maplist PR''s "deferred"
    % list.
    unique_cpp_tmp_dir('tmp_cpp_e2e_fa_arith', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_findall_member_arith/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_findall_member_arith/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_findall_member_user_pred,
     [condition(cpp_compiler_available)]) :-
    % findall(Y, (member(X, [...]), double(X, Y)), L) — same shape
    % but dispatches a user predicate as the second conjunct.
    % Exercises the user-label dispatch + aggregate-frame
    % collection together.
    unique_cpp_tmp_dir('tmp_cpp_e2e_fa_user', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_findall_member_user/0,
                               user:wam_cpp_double/2],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_findall_member_user/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_findall_member_call_meta,
     [condition(cpp_compiler_available)]) :-
    % The full higher-order pipeline: findall + member + call/N
    % dispatching to a user predicate. Three meta-machineries
    % stacked — aggregate frame, member''s choice-point retry, and
    % call/N''s goal-term dispatch.
    unique_cpp_tmp_dir('tmp_cpp_e2e_fa_call', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_findall_member_call/0,
                               user:wam_cpp_double/2],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_findall_member_call/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_findall_filtered, [condition(cpp_compiler_available)]) :-
    % findall(X, (member(X, [...]), X > 2), L) — backtracking
    % through a filter condition. Tests that the aggregate frame
    % correctly handles per-solution success/failure.
    unique_cpp_tmp_dir('tmp_cpp_e2e_fa_filtered', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_findall_filtered/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_findall_filtered/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_findall_three_goal_conjunction,
     [condition(cpp_compiler_available)]) :-
    % Three goals chained: member backtracks, arith computes,
    % arith computes again. Verifies the aggregate frame handles
    % an N-goal conjunction (not just 2).
    unique_cpp_tmp_dir('tmp_cpp_e2e_fa_three', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_findall_three_goals/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_findall_three_goals/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% Nested findalls — the inner findall isn''t inlined, so a meta-call
% findall/3 dispatcher + ConjFrame mechanism for ,/2 goal-terms is
% needed. Resolved the latent bug deferred from PR #2098.
% ------------------------------------------------------------------

test(cpp_e2e_findall_meta_no_conjunction,
     [condition(cpp_compiler_available)]) :-
    % Single-goal meta findall/3 (no conjunction). Exercises
    % dispatch_findall_call without the ConjFrame path. Simplest
    % nested-findall case.
    unique_cpp_tmp_dir('tmp_cpp_e2e_fa_meta_simple', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_findall_meta_no_conjunction/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_findall_meta_no_conjunction/0',
                    [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_findall_nested_simple,
     [condition(cpp_compiler_available)]) :-
    % Outer findall, inner findall WITHOUT a conjunction in its goal
    % (just a single member/2 call). Tests dispatch_findall_call
    % alone — the ConjFrame path is exercised via the next test.
    unique_cpp_tmp_dir('tmp_cpp_e2e_fa_nested_simple', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_findall_nested_simple/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_findall_nested_simple/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_findall_nested, [condition(cpp_compiler_available)]) :-
    % The full original reproduction:
    %   findall(L,
    %           (member(N, [2, 3]),
    %            findall(X, (member(X, [1, 2, 3, 4]), X =< N), L)),
    %           Ls)
    %   → Ls = [[1, 2], [1, 2, 3]]
    %
    % Exercises:
    %   - Outer findall (inlined BeginAggregate/EndAggregate).
    %   - Inner findall (meta-call via dispatch_findall_call).
    %   - Inner goal is a conjunction (,/2 goal-term) → ConjFrame
    %     dispatch with G1=member, G2=(X =< N).
    %   - G1 has multiple solutions; each one re-dispatches G2 via
    %     the ConjFrame staying on the stack across backtracks.
    %   - Inner aggregate finalises, binds L, outer''s EndAggregate
    %     collects.
    unique_cpp_tmp_dir('tmp_cpp_e2e_fa_nested', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_findall_nested/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_findall_nested/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% bagof/3 + setof/3 — share dispatch_aggregate_call with findall.
% bagof fails on empty; setof sorts + dedups via standard term order
% (term_less in finalize_aggregate, with recursive args comparison
% so list compounds with the same functor get ordered correctly).
% Nested forms exercise the meta-call path and the ^/2 transparency
% handler.
% ------------------------------------------------------------------

test(cpp_e2e_bagof_basic, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_bagof_basic', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_bagof_basic/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_bagof_basic/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_bagof_fails_empty,
     [condition(cpp_compiler_available)]) :-
    % bagof returns failure (per ISO) when the goal has no
    % solutions. The if-then-else wraps that — predicate succeeds
    % via the else branch. Regression guard for the
    % backtrack-continue-on-Uninit fix.
    unique_cpp_tmp_dir('tmp_cpp_e2e_bagof_empty', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_bagof_fails_empty/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_bagof_fails_empty/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_setof_basic, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_setof_basic', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_setof_basic/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_setof_basic/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_setof_dedups, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_setof_dedups', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_setof_dedups/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_setof_dedups/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_setof_sorts_ints,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_setof_sorts', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_setof_sorts_ints/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_setof_sorts_ints/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_bagof_nested, [condition(cpp_compiler_available)]) :-
    % Nested bagof with existential quantifier (N^Goal).
    % Exercises:
    %   - Outer inlined bagof.
    %   - Inner non-inlined bagof via dispatch_aggregate_call("bagof").
    %   - ^/2 transparency in both invoke_goal_as_call AND the
    %     Call step arm (the WAM emits `call ^/2, 2` for the
    %     existential quantifier).
    %   - ConjFrame dispatch for the inner''s ,(member, X =< N) goal.
    unique_cpp_tmp_dir('tmp_cpp_e2e_bagof_nested', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_bagof_nested/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_bagof_nested/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_setof_nested, [condition(cpp_compiler_available)]) :-
    % Nested setof. The outer setof''s sort sees list-shaped
    % compounds [1,2] and [1,2,3] (both functor [|]/2/2). The
    % term_less helper''s recursive args comparison is what makes
    % the lexicographic ordering work: regression guard for the
    % "compound sort by functor only" bug fixed in this PR.
    unique_cpp_tmp_dir('tmp_cpp_e2e_setof_nested', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_setof_nested/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_setof_nested/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% Disjunction goal-terms (;/2) — handled by invoke_goal_as_call by
% pushing a CP whose alt_pc = disj_alt_pc and a paired DisjFrame
% carrying G2 + after_pc. G1 dispatched normally; on G1 exhaustion
% backtrack reaches the CP, DisjAlt pops both CP and DisjFrame, then
% dispatches G2.
% ------------------------------------------------------------------

test(cpp_e2e_catch_disjunction,
     [condition(cpp_compiler_available)]) :-
    % catch((true ; fail), _, _) — the catch goal is a disjunction
    % term. G1=true succeeds; the catch succeeds with no throw and
    % the recovery never runs.
    unique_cpp_tmp_dir('tmp_cpp_e2e_catch_disj', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_catch_disj/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_catch_disj/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_negation_disjunction_both_fail,
     [condition(cpp_compiler_available)]) :-
    % \+ (fail ; fail) — both alternatives fail → \+ succeeds.
    % Exercises the DisjAlt path: G1=fail fails immediately,
    % DisjAlt fires, G2=fail also fails, full disjunction fails,
    % which is what \+ needs to succeed.
    unique_cpp_tmp_dir('tmp_cpp_e2e_neg_disj', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_not_disj_both_fail/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_not_disj_both_fail/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_call_disjunction_first,
     [condition(cpp_compiler_available)]) :-
    % call((X = 1 ; X = 2)) with subsequent X = 1 — first
    % alternative''s binding sticks (no backtrack needed).
    unique_cpp_tmp_dir('tmp_cpp_e2e_call_disj_first', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_call_disj_first/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_call_disj_first/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_call_disjunction_second,
     [condition(cpp_compiler_available)]) :-
    % call((X = 1 ; X = 2)) with subsequent X = 2 — forces backtrack
    % into G2 via DisjAlt. The critical regression guard for the
    % CP-paired-with-DisjFrame mechanism.
    unique_cpp_tmp_dir('tmp_cpp_e2e_call_disj_second', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_call_disj_second/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_call_disj_second/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_call_disjunction_first_fails,
     [condition(cpp_compiler_available)]) :-
    % call((fail ; X = 7)) — G1 fails immediately, DisjAlt
    % dispatches G2 which binds X=7. Tests the
    % immediate-failure-of-G1 path.
    unique_cpp_tmp_dir('tmp_cpp_e2e_call_disj_g1fail', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_call_disj_first_fails/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_call_disj_first_fails/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% If-then-else goal-terms `(Cond -> Then ; Else)` passed to a
% meta-call. The WAM compiler builds them as `;(->(Cond, Then), Else)`.
% invoke_goal_as_call peeks at the first arg of ;/2 — if it''s ->/2,
% routes to an IfThenFrame + paired CP with cut-on-Cond-success
% semantics. Otherwise falls through to plain disjunction.
% ------------------------------------------------------------------

test(cpp_e2e_ite_then_branch,
     [condition(cpp_compiler_available)]) :-
    % Cond=true → IfThenCommit fires → Then branch dispatched.
    unique_cpp_tmp_dir('tmp_cpp_e2e_ite_then', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_ite_then_branch/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_ite_then_branch/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_ite_else_branch,
     [condition(cpp_compiler_available)]) :-
    % Cond=fail → backtrack to our CP → IfThenElse fires → Else
    % branch dispatched. Regression guard for the trust_me-style
    % pop in IfThenElse (without it the alt_pc fires repeatedly
    % into an empty if_then_frames, causing infinite recurse).
    unique_cpp_tmp_dir('tmp_cpp_e2e_ite_else', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_ite_else_branch/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_ite_else_branch/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_ite_inside_findall,
     [condition(cpp_compiler_available)]) :-
    % findall + if-then-else as a call''d goal. Each iteration of
    % member dispatches the if-then-else; for X∈{1,2} the Else
    % branch fires (Y=small), for X∈{3,4} the Then branch fires
    % (Y=big). Exercises if-then-else inside an aggregate context
    % with backtracking.
    unique_cpp_tmp_dir('tmp_cpp_e2e_ite_findall', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_ite_inside_findall/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_ite_inside_findall/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_ite_cut_commits_cond,
     [condition(cpp_compiler_available)]) :-
    % Cut semantics: `call((member(X, [1, 2, 3]) -> Y = X ; ...))`
    % must commit to X=1 (Cond''s first solution) and NOT
    % backtrack-retry for X=2 / X=3 later. IfThenCommit drops the
    % CPs from Cond''s dispatch back to base_cp_count, achieving
    % the cut. Critical regression guard.
    unique_cpp_tmp_dir('tmp_cpp_e2e_ite_cut', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_ite_cut_commits_cond/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_ite_cut_commits_cond/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% Bare (Cond -> Then) goal-terms — no Else. The WAM compiler builds
% these as ->/2 at the top level (not wrapped in ;/2). Reuses the
% IfThenFrame machinery with else_goal = null; on Cond failure the
% IfThenElse op propagates failure instead of dispatching Else.
% ------------------------------------------------------------------

test(cpp_e2e_bif_then_runs, [condition(cpp_compiler_available)]) :-
    % Cond=true → IfThenCommit → Then runs.
    unique_cpp_tmp_dir('tmp_cpp_e2e_bif_then', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_bif_then_runs/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_bif_then_runs/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_bif_cond_fail_propagates,
     [condition(cpp_compiler_available)]) :-
    % Cond=fail → IfThenElse fires; else_goal is null → propagate
    % failure. Wrapped in if-then-else so the outer test succeeds
    % via the else branch.
    unique_cpp_tmp_dir('tmp_cpp_e2e_bif_fail', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_bif_cond_fail_propagates/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_bif_cond_fail_propagates/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_bif_inside_catch,
     [condition(cpp_compiler_available)]) :-
    % Bare (Cond -> Then) inside catch — the catch protected goal
    % is a ->/2 term, dispatched via invoke_goal_as_call.
    unique_cpp_tmp_dir('tmp_cpp_e2e_bif_catch', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_bif_inside_catch/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_bif_inside_catch/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_bif_not_when_cond_fails,
     [condition(cpp_compiler_available)]) :-
    % \+ (bif with Cond=fail). The bif fails → \+ succeeds.
    % Tests that bare-if-then''s failure propagates correctly
    % through the negation layer.
    unique_cpp_tmp_dir('tmp_cpp_e2e_bif_not', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_bif_not_when_cond_fails/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_bif_not_when_cond_fails/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_bif_cut_commits,
     [condition(cpp_compiler_available)]) :-
    % Cut semantics for bare-if-then. member/2 in Cond has 3
    % solutions; we commit to X=1 and don''t retry. Verifies
    % IfThenCommit''s CP-trimming applies to the bare form too.
    unique_cpp_tmp_dir('tmp_cpp_e2e_bif_cut', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_bif_cut_commits/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_bif_cut_commits/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% bagof/setof witness grouping. Meta-call dispatch walks the goal
% term to find free witnesses (unbound vars in Goal that aren''t in
% Template and aren''t under ^/2). Results are grouped by witness
% binding; first group''s template list is bound to the result and
% the witness cells are bound to that group''s witness values.
% Backtracking through additional groups is a planned follow-up.
% ------------------------------------------------------------------

test(cpp_e2e_bagof_meta_groups_by_witness,
     [condition(cpp_compiler_available)]) :-
    % The key test: bagof(C, parent(P, C), L) via meta-call
    % dispatch. P is a free witness. First group is P=tom →
    % L=[bob, alice]. Verifies BOTH the list shape AND that P
    % gets bound to tom (witness binding back to caller).
    unique_cpp_tmp_dir('tmp_cpp_e2e_bagof_grp', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_parent_fixture/2,
             user:wam_cpp_test_bagof_meta_groups_by_witness/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_bagof_meta_groups_by_witness/0',
                    [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_bagof_meta_existential_no_grouping,
     [condition(cpp_compiler_available)]) :-
    % ^/2 existentially quantifies P → no witnesses → all 3
    % children flatten into one group.
    unique_cpp_tmp_dir('tmp_cpp_e2e_bagof_excl', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_parent_fixture/2,
             user:wam_cpp_test_bagof_meta_existential_no_grouping/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_bagof_meta_existential_no_grouping/0',
                    [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_setof_meta_groups_sorted,
     [condition(cpp_compiler_available)]) :-
    % setof: same witness grouping as bagof, but the per-group
    % template list is sorted (and dedup''d via term_less).
    unique_cpp_tmp_dir('tmp_cpp_e2e_setof_grp', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_parent_fixture/2,
             user:wam_cpp_test_setof_meta_groups_sorted/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_setof_meta_groups_sorted/0',
                    [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_bagof_inlined_groups_by_witness,
     [condition(cpp_compiler_available)]) :-
    % Inlined outer bagof — the WAM compiler emits direct
    % BeginAggregate/EndAggregate with the 4-arg form carrying free-
    % witness register info. The runtime resolves them lazily at
    % EndAggregate (witness Y-regs get allocated INSIDE the aggregate
    % body) and snapshots witness values parallel to acc, so the
    % finaliser groups by witness equality. First group: P=tom,
    % L=[bob, alice]; witness binding flows back to the caller.
    unique_cpp_tmp_dir('tmp_cpp_e2e_bagof_inl_grp', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_parent_fixture/2,
             user:wam_cpp_test_bagof_inlined_groups_by_witness/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_bagof_inlined_groups_by_witness/0',
                    [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_bagof_inlined_existential_flattens,
     [condition(cpp_compiler_available)]) :-
    % ^/2 existential on the inlined path → no witnesses emitted
    % (4th begin_aggregate arg is empty). The frame''s witness_regs
    % stays empty, so EndAggregate does the simple non-grouping path
    % and aggregate-finalise builds one flat group. Regression guard
    % that the existential-quantified inlined path keeps working.
    unique_cpp_tmp_dir('tmp_cpp_e2e_bagof_inl_ex', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_parent_fixture/2,
             user:wam_cpp_test_bagof_inlined_existential_flattens/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_bagof_inlined_existential_flattens/0',
                    [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_setof_inlined_groups_sorted,
     [condition(cpp_compiler_available)]) :-
    % setof on the inlined path: same witness grouping as bagof,
    % per-group template list sorted via term_less. First group:
    % P=tom, L=[alice, bob].
    unique_cpp_tmp_dir('tmp_cpp_e2e_setof_inl_grp', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_parent_fixture/2,
             user:wam_cpp_test_setof_inlined_groups_sorted/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_setof_inlined_groups_sorted/0',
                    [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% bagof/setof group backtracking — completes the witness-grouping
% story from PR #2108. The aggregate-finalise path now pushes an
% AggregateGroupIterator and a CP whose alt_pc = aggregate_next_group_pc.
% On backtrack into the aggregate, AggregateNextGroup pops the next
% group from the iterator and binds it; if more remain, pushes
% another CP. This makes `findall(L, bagof(C, parent(P, C), L), Ls)`
% return the FULL list of groups, not just the first.
% ------------------------------------------------------------------

test(cpp_e2e_findall_of_bagof_groups,
     [condition(cpp_compiler_available)]) :-
    % `findall(L, bagof(C, parent(_P, C), L), Ls)` — outer findall
    % drives inner bagof''s group iteration. Two parents → two
    % groups. The first group is bound on the initial bagof
    % success; backtrack triggers the next-group CP which binds
    % the second group.
    unique_cpp_tmp_dir('tmp_cpp_e2e_fa_bagof_grp', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_grandparent_fixture/2,
             user:wam_cpp_test_findall_of_bagof_groups/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_findall_of_bagof_groups/0',
                    [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_findall_of_setof_groups,
     [condition(cpp_compiler_available)]) :-
    % Same shape with setof — each group is sorted via term_less.
    unique_cpp_tmp_dir('tmp_cpp_e2e_fa_setof_grp', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_grandparent_fixture/2,
             user:wam_cpp_test_findall_of_setof_groups/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_findall_of_setof_groups/0',
                    [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_bagof_witness_pairs,
     [condition(cpp_compiler_available)]) :-
    % `findall(P-L, bagof(C, parent(P, C), L), Pairs)` — verifies
    % that the witness binding flows back to the outer findall.
    % First iteration: P=tom, L=[bob,alice]. Second: P=jane,
    % L=[carol,dave]. Critical regression guard for the
    % per-group witness rebinding via the CP machinery.
    unique_cpp_tmp_dir('tmp_cpp_e2e_bagof_pairs', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_grandparent_fixture/2,
             user:wam_cpp_test_bagof_witness_pairs/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_bagof_witness_pairs/0',
                    [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_bagof_single_group_unchanged,
     [condition(cpp_compiler_available)]) :-
    % ^/2 existential — no witnesses, no iterator, no extra CP.
    % Regression guard that the existential-quantified path still
    % works as before (single-group binding, no backtrack
    % artifacts).
    unique_cpp_tmp_dir('tmp_cpp_e2e_bagof_single', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_grandparent_fixture/2,
             user:wam_cpp_test_bagof_single_group_unchanged/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_bagof_single_group_unchanged/0',
                    [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% once/1 + forall/2 — desugared at the WAM compile level to
% (G -> true) and \+ (G, \+ T) respectively. Inlined goals reuse
% the if-then-else / negation infrastructure; goal-term position
% (catch, call, nested aggregates) is handled by the runtime''s
% invoke_goal_as_call dispatcher.
% Includes a fix to CutIte that drops only the top CP and any CPs
% Cond pushed above it, restoring cut_barrier from the saved value;
% the previous behaviour cut all CPs above a global barrier, which
% could swallow an enclosing aggregate''s generator CP.
% ------------------------------------------------------------------

test(cpp_e2e_once_first, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_once_first', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_once_first/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_once_first/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_once_inner_fail, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_once_fail', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_once_inner_fail/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_once_inner_fail/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_once_no_backtrack, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_once_nobt', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_once_no_backtrack/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_once_no_backtrack/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_forall_all_pass, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_forall_all', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_forall_all_pass/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_forall_all_pass/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_forall_some_fail, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_forall_fail', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_forall_some_fail/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_forall_some_fail/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_forall_empty, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_forall_empty', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_forall_empty/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_forall_empty/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_once_in_catch, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_once_catch', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_once_in_catch/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_once_in_catch/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_forall_in_catch, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_forall_catch', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_forall_in_catch/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_forall_in_catch/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_findall_with_once, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_fa_once', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_findall_with_once/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_findall_with_once/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_ite_in_findall_no_cut_outer,
     [condition(cpp_compiler_available)]) :-
    % Direct regression guard for the CutIte fix: a bare if-then-else
    % inside findall must not swallow the outer findall''s generator
    % CPs. Without the fix, this returns [1] instead of [1, 2, 3].
    unique_cpp_tmp_dir('tmp_cpp_e2e_ite_in_fa', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_ite_in_findall_no_cut_outer/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_ite_in_findall_no_cut_outer/0',
                    [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% format/1, /2, /3 — formatted output. /1 and /2 print to stdout;
% /3 dispatches on a destination argument that selects user_output /
% user_error / atom(V) / string(V) / codes(V). The string-building
% variants (atom/string/codes) unify their argument with the
% rendered output, enabling in-process string construction.
% ------------------------------------------------------------------

test(cpp_e2e_format1, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_format1', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_format1/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_stdout(BinPath, 'wam_cpp_test_format1/0', [],
                           true, "plain text\n")
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_format2_w, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_format2_w', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_format2_w/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_stdout(BinPath, 'wam_cpp_test_format2_w/0', [],
                           true, "X = 42\n")
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_format2_multi, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_format2_multi', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_format2_multi/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_stdout(BinPath, 'wam_cpp_test_format2_multi/0', [],
                           true, "1 + 2 = 3\n")
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_format2_d, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_format2_d', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_format2_d/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_stdout(BinPath, 'wam_cpp_test_format2_d/0', [],
                           true, "42\n")
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_format2_a, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_format2_a', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_format2_a/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_stdout(BinPath, 'wam_cpp_test_format2_a/0', [],
                           true, "hello\n")
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_format3_atom, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_format3_atom', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_format3_atom/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_format3_atom/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_format3_string, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_format3_string', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_format3_string/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_format3_string/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_format3_codes, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_format3_codes', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_format3_codes/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_format3_codes/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_format3_chained, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_format3_chained', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_format3_chained/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_format3_chained/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% atom_codes / atom_chars / number_codes / atom_concat / atom_length
% / char_code — atom/string/number conversions.
% ------------------------------------------------------------------

test(cpp_e2e_atom_codes_fwd, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_atom_codes_fwd', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_atom_codes_fwd/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_atom_codes_fwd/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_atom_codes_rev, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_atom_codes_rev', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_atom_codes_rev/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_atom_codes_rev/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_atom_chars_fwd, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_atom_chars_fwd', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_atom_chars_fwd/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_atom_chars_fwd/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_atom_chars_rev, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_atom_chars_rev', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_atom_chars_rev/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_atom_chars_rev/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_number_codes_fwd, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_number_codes_fwd', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_number_codes_fwd/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_number_codes_fwd/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_number_codes_rev_int,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_number_codes_rev_int', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_number_codes_rev_int/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_number_codes_rev_int/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_number_codes_neg, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_number_codes_neg', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_number_codes_neg/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_number_codes_neg/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_atom_concat, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_atom_concat', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_atom_concat/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_atom_concat/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_atom_concat_num, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_atom_concat_num', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_atom_concat_num/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_atom_concat_num/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_atom_length, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_atom_length', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_atom_length/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_atom_length/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_atom_length_int, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_atom_length_int', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_atom_length_int/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_atom_length_int/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_char_code_fwd, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_char_code_fwd', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_char_code_fwd/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_char_code_fwd/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_char_code_rev, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_char_code_rev', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_char_code_rev/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_char_code_rev/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_format_atom_then_codes,
     [condition(cpp_compiler_available)]) :-
    % Composition guard: an atom built by format/3 should be
    % indistinguishable from one written as a literal — same
    % length, same code list.
    unique_cpp_tmp_dir('tmp_cpp_e2e_fmt_atom_codes', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_format_atom_then_codes/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_format_atom_then_codes/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% assertz/asserta/retract/retractall — dynamic database manipulation
% for FACTS. Rules (Head :- Body) are rejected by the builtin and
% deferred to a follow-up PR. Each test asserts into a uniquely-
% named predicate so tests are independent across runs.
% ------------------------------------------------------------------

test(cpp_e2e_assertz_query, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_assertz_query', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_assertz_query/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_assertz_query/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_assertz_multi, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_assertz_multi', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_assertz_multi/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_assertz_multi/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_asserta_order, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_asserta_order', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_asserta_order/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_asserta_order/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_retract, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_retract', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_retract/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_retract/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_retract_var, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_retract_var', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_retract_var/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_retract_var/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_retractall, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_retractall', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_retractall/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_retractall/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_retractall_empty, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_retractall_empty', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_retractall_empty/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_retractall_empty/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_assertz_pair, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_assertz_pair', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_assertz_pair/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_assertz_pair/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_dyn_backtrack, [condition(cpp_compiler_available)]) :-
    % Backtracking through dynamic clauses: findall driver explores
    % all 3 wam_cpp_dyn_num/1 facts via the CP pushed by
    % dynamic_try_next. Filtering with X > 1 confirms iteration
    % actually reaches the later clauses, not just the first.
    unique_cpp_tmp_dir('tmp_cpp_e2e_dyn_bt', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_dyn_backtrack/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_dyn_backtrack/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% sub_atom/5 — substring search and decomposition. Both deterministic
% (extraction with known Before/Length) and nondeterministic (find Sub
% within Atom, or full enumeration) modes are exercised below.
% ------------------------------------------------------------------

test(cpp_e2e_sub_atom_extract, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_sub_atom_extract', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_sub_atom_extract/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_sub_atom_extract/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_sub_atom_after_computed,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_sub_atom_aft', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_sub_atom_after_computed/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_sub_atom_after_computed/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_sub_atom_find_first,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_sub_atom_fst', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_sub_atom_find_first/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_sub_atom_find_first/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_sub_atom_find_all,
     [condition(cpp_compiler_available)]) :-
    % Multi-match enumeration: findall over sub_atom backtracks
    % through every (Before, Length) where Sub matches. Direct
    % regression guard for the SubAtomIterator CP pattern.
    unique_cpp_tmp_dir('tmp_cpp_e2e_sub_atom_all', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_sub_atom_find_all/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_sub_atom_find_all/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_sub_atom_prefix, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_sub_atom_pre', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_sub_atom_prefix/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_sub_atom_prefix/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_sub_atom_suffix, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_sub_atom_suf', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_sub_atom_suffix/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_sub_atom_suffix/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_sub_atom_whole, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_sub_atom_whole', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_sub_atom_whole/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_sub_atom_whole/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_sub_atom_empty, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_sub_atom_empty', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_sub_atom_empty/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_sub_atom_empty/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_sub_atom_no_match,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_sub_atom_nm', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_sub_atom_no_match/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_sub_atom_no_match/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_sub_atom_enum_all,
     [condition(cpp_compiler_available)]) :-
    % Full enumeration: all 6 substrings of "ab" (length 0..2 at each
    % valid position). Verifies the iterator visits every candidate.
    unique_cpp_tmp_dir('tmp_cpp_e2e_sub_atom_enum', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_sub_atom_enum_all/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_sub_atom_enum_all/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% Rules in assertz/retract — completes the dynamic-database story
% from #2136. Rules are stored as ":-/2"(Head, Body) compounds and
% dispatched by flattening Body into a sequential goal list (a
% BodyFrame), with ChoicePoints snapshotting body_frames so that
% backtrack into any goal''s CP restores the correct rule context.
% ------------------------------------------------------------------

test(cpp_e2e_rule_simple, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_rule_simple', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_rule_simple/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_rule_simple/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_rule_conj, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_rule_conj', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_rule_conj/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_rule_conj/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_rule_mixed_facts_and_rule,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_rule_mixed', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_rule_mixed_facts_and_rule/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_rule_mixed_facts_and_rule/0',
                    [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_rule_body_fails, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_rule_body_fails', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_rule_body_fails/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_rule_body_fails/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_rule_recursive, [condition(cpp_compiler_available)]) :-
    % Recursion regression guard: 3-deep list-length needs both
    % BodyFrame (so nested rule-body conjunctions don''t loop on
    % shared conj_return_pc) and ChoicePoint::saved_body_frames
    % (so backtrack from a deeper level''s body CP restores the
    % outer rule''s body_frame context).
    unique_cpp_tmp_dir('tmp_cpp_e2e_rule_recursive', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_rule_recursive/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_rule_recursive/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_rule_backtrack_in_body,
     [condition(cpp_compiler_available)]) :-
    % Body of an asserted rule is itself nondet (member/2).
    % findall must re-enter the body''s member CP after the first
    % solution; that''s the path that exercises body_frames
    % restoration via the CP.
    unique_cpp_tmp_dir('tmp_cpp_e2e_rule_bt', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_rule_backtrack_in_body/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_rule_backtrack_in_body/0',
                    [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_retract_rule, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_retract_rule', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_retract_rule/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_retract_rule/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% Nondeterministic retract/1 — backtracks through subsequent matches.
% Each successful retract is destructive per ISO; the database stays
% modified across backtrack.
% ------------------------------------------------------------------

test(cpp_e2e_retract_nondet_all,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_retract_nd_all', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_retract_nondet_all/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_retract_nondet_all/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_retract_nondet_pattern,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_retract_nd_pat', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_retract_nondet_pattern/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_retract_nondet_pattern/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_retract_bind_via_pattern,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_retract_bind', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_retract_bind_via_pattern/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_retract_bind_via_pattern/0',
                    [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_retract_fail_no_match,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_retract_no_match', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_retract_fail_no_match/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_retract_fail_no_match/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_retract_destructive,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_retract_dest', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_retract_destructive/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_retract_destructive/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% Mutable globals — nb_setval/getval (non-backtrackable) and
% b_setval/getval (trail-tracked). Stored in WamState::nb_globals.
% ------------------------------------------------------------------

test(cpp_e2e_nb_basic, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_nb_basic', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_nb_basic/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_nb_basic/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_nb_replace, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_nb_replace', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_nb_replace/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_nb_replace/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_nb_survives_backtrack,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_nb_bt', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_nb_survives_backtrack/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_nb_survives_backtrack/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_b_undone_on_backtrack,
     [condition(cpp_compiler_available)]) :-
    % Direct regression guard for b_setval''s trail integration.
    unique_cpp_tmp_dir('tmp_cpp_e2e_b_bt', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_b_undone_on_backtrack/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_b_undone_on_backtrack/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_nb_unset_fails, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_nb_unset', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_nb_unset_fails/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_nb_unset_fails/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_nb_compound, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_nb_compound', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_nb_compound/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_nb_compound/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_nb_counter, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_nb_counter', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_nb_counter/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_nb_counter/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% @</2, @=</2, @>/2, @>=/2, compare/3 — standard order of terms.
% ------------------------------------------------------------------

test(cpp_e2e_term_order_categories,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_to_cat', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_term_order_categories/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_term_order_categories/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_term_order_numbers,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_to_num', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_term_order_numbers/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_term_order_numbers/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_term_order_atoms,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_to_atoms', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_term_order_atoms/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_term_order_atoms/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_term_order_arity,
     [condition(cpp_compiler_available)]) :-
    % Regression guard: arity-first comparison (not "Name/Arity"
    % string lex). foo/2 @< foo/10 even though "foo/10" @< "foo/2"
    % alphabetically.
    unique_cpp_tmp_dir('tmp_cpp_e2e_to_arity', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_term_order_arity/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_term_order_arity/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_term_order_compound_name,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_to_cname', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_term_order_compound_name/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_term_order_compound_name/0',
                    [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_term_order_compound_args,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_to_cargs', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_term_order_compound_args/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_term_order_compound_args/0',
                    [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_compare_lt, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_compare_lt', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_compare_lt/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_compare_lt/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_compare_eq, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_compare_eq', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_compare_eq/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_compare_eq/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_compare_gt, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_compare_gt', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_compare_gt/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_compare_gt/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_term_order_lte_eq,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_to_lte', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_term_order_lte_eq/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_term_order_lte_eq/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_term_order_gte_eq,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_to_gte', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_term_order_gte_eq/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_term_order_gte_eq/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_term_order_neg, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_to_neg', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_term_order_neg/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_term_order_neg/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% char_type/2 + upcase_atom/2 + downcase_atom/2 — character
% classification and whole-atom case conversion.
% ------------------------------------------------------------------

test(cpp_e2e_char_type_alpha, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_ct_alpha', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_char_type_alpha/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_char_type_alpha/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_char_type_digit, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_ct_digit', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_char_type_digit/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_char_type_digit/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_char_type_whitespace,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_ct_ws', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_char_type_whitespace/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_char_type_whitespace/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_char_type_upper_arg,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_ct_upper', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_char_type_upper_arg/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_char_type_upper_arg/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_char_type_lower_arg,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_ct_lower', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_char_type_lower_arg/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_char_type_lower_arg/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_char_type_to_upper,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_ct_to_up', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_char_type_to_upper/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_char_type_to_upper/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_char_type_to_lower,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_ct_to_lo', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_char_type_to_lower/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_char_type_to_lower/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_char_type_digit_weight,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_ct_dw', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_char_type_digit_weight/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_char_type_digit_weight/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_char_type_digit_reverse,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_ct_dr', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_char_type_digit_reverse/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_char_type_digit_reverse/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_char_type_code, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_ct_code', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_char_type_code/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_char_type_code/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_upcase_atom, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_upcase', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_upcase_atom/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_upcase_atom/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_downcase_atom, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_downcase', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_downcase_atom/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_downcase_atom/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% numlist/3, sort/2, msort/2, select/3 — list utilities.
% ------------------------------------------------------------------

test(cpp_e2e_numlist, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_numlist', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_numlist/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_numlist/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_numlist_empty, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_numlist_empty', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_numlist_empty/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_numlist_empty/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_numlist_single, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_numlist_single', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_numlist_single/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_numlist_single/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_sort, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_sort', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_sort/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_sort/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_sort_mixed, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_sort_mixed', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_sort_mixed/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_sort_mixed/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_sort_empty, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_sort_empty', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_sort_empty/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_sort_empty/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_msort, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_msort', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_msort/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_msort/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_select_bound, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_select_bound', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_select_bound/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_select_bound/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_select_all, [condition(cpp_compiler_available)]) :-
    % select/3 nondet — findall enumerates every (Elem, Rest) pair.
    unique_cpp_tmp_dir('tmp_cpp_e2e_select_all', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_select_all/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_select_all/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_select_missing, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_select_missing', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_select_missing/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_select_missing/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_numlist_then_sort,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_nlsort', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_numlist_then_sort/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_numlist_then_sort/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% maplist/2..5 — apply a goal to each list element.
% ------------------------------------------------------------------

test(cpp_e2e_maplist_2_check,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_ml2_chk', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_maplist_2_check/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_maplist_2_check/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_maplist_2_fail,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_ml2_fail', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_maplist_2_fail/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_maplist_2_fail/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_maplist_3_map,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_ml3_map', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_plus10/2,
                               user:wam_cpp_test_maplist_3_map/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_maplist_3_map/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_maplist_3_succ,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_ml3_succ', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_maplist_3_succ/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_maplist_3_succ/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_maplist_3_empty,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_ml3_emp', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_plus10/2,
                               user:wam_cpp_test_maplist_3_empty/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_maplist_3_empty/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_maplist_4, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_ml4', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_add/3,
                               user:wam_cpp_test_maplist_4/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_maplist_4/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_maplist_5, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_ml5', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_add3/4,
                               user:wam_cpp_test_maplist_5/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_maplist_5/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% include/3, exclude/3, partition/4, foldl/4, foldl/5 — filter +
% fold meta-predicates.
% ------------------------------------------------------------------

test(cpp_e2e_include, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_include', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_gt0/1,
                               user:wam_cpp_test_include/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_include/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_include_empty, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_inc_empty', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_gt0/1,
                               user:wam_cpp_test_include_empty/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_include_empty/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_exclude, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_exclude', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_gt0/1,
                               user:wam_cpp_test_exclude/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_exclude/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_partition, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_partition', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_gt0/1,
                               user:wam_cpp_test_partition/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_partition/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_foldl4_sum, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_foldl4_sum', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_addc/3,
                               user:wam_cpp_test_foldl4_sum/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_foldl4_sum/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_foldl4_empty, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_foldl4_empty', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_addc/3,
                               user:wam_cpp_test_foldl4_empty/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_foldl4_empty/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_foldl5, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_foldl5', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_join4/4,
                               user:wam_cpp_test_foldl5/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_foldl5/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% keysort/2 + pairs_keys/2 + pairs_values/2 + pairs_keys_values/3 —
% Key-Value pair list utilities.
% ------------------------------------------------------------------

test(cpp_e2e_keysort, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_keysort', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_keysort/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_keysort/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_keysort_stable,
     [condition(cpp_compiler_available)]) :-
    % Stability regression guard — keys b/a appear twice; their
    % original Value-order must be preserved on key ties.
    unique_cpp_tmp_dir('tmp_cpp_e2e_ks_stable', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_keysort_stable/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_keysort_stable/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_keysort_empty,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_ks_empty', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_keysort_empty/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_keysort_empty/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_pairs_keys, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_pairs_keys', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_pairs_keys/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_pairs_keys/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_pairs_keys_empty,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_pk_empty', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_pairs_keys_empty/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_pairs_keys_empty/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_pairs_values, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_pairs_values', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_pairs_values/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_pairs_values/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_pairs_kv, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_pairs_kv', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_pairs_kv/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_pairs_kv/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_keysort_then_values,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_ks_then', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_keysort_then_values/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_keysort_then_values/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% WAM-text quote roundtrip for digit-only atoms — regression guard
% for the atom-marker convention added by quote_wam_constant +
% cpp_value_literal.
% ------------------------------------------------------------------

test(cpp_e2e_digit_atom_is_atom,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_da_atom', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_digit_atom_is_atom/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_digit_atom_is_atom/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_digit_atom_not_integer,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_da_nint', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_digit_atom_not_integer/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_digit_atom_not_integer/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_digit_atom_codes,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_da_codes', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_digit_atom_codes/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_digit_atom_codes/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_digit_atom_length,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_da_len', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_digit_atom_length/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_digit_atom_length/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_digit_integer_unchanged,
     [condition(cpp_compiler_available)]) :-
    % Sanity: unquoted integer 5 stays integer, doesn''t become atom.
    unique_cpp_tmp_dir('tmp_cpp_e2e_di_int', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_digit_integer_unchanged/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_digit_integer_unchanged/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_char_type_digit_direct,
     [condition(cpp_compiler_available)]) :-
    % char_type with literal digit-atom now works directly (no
    % char_code/2 workaround needed).
    unique_cpp_tmp_dir('tmp_cpp_e2e_ctd', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_char_type_digit_direct/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_char_type_digit_direct/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_negative_atom,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_neg_atom', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_negative_atom/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_negative_atom/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% I/O polish — print/1, display/1, tab/1, write_canonical/1.
% Stdout-printing tests use run_query_stdout to assert exact output.
% ------------------------------------------------------------------

test(cpp_e2e_print, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_print', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_print/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_stdout(BinPath, 'wam_cpp_test_print/0', [],
                           true, "hello\n")
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_display, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_display', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_display/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_stdout(BinPath, 'wam_cpp_test_display/0', [],
                           true, "bar\n")
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_tab, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_tab', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_tab/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_stdout(BinPath, 'wam_cpp_test_tab/0', [],
                           true, "   x\n")
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_tab_zero, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_tab_zero', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_tab_zero/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_stdout(BinPath, 'wam_cpp_test_tab_zero/0', [],
                           true, "y\n")
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_tab_neg, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_tab_neg', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_tab_neg/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_tab_neg/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_wc_simple, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_wc_simple', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_wc_simple/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_stdout(BinPath, 'wam_cpp_test_wc_simple/0', [],
                           true, "hello\n")
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_wc_quoted, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_wc_quoted', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_wc_quoted/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_stdout(BinPath, 'wam_cpp_test_wc_quoted/0', [],
                           true, "'hello world'\n")
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_wc_digit_atom,
     [condition(cpp_compiler_available)]) :-
    % Digit-only atoms get quoted in canonical form, distinguishing
    % them from the integer with the same textual form.
    unique_cpp_tmp_dir('tmp_cpp_e2e_wc_digit', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_wc_digit_atom/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_stdout(BinPath, 'wam_cpp_test_wc_digit_atom/0',
                           [], true, "'5'\n")
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% with_output_to/2 — capture I/O into an atom/string/codes.
% ------------------------------------------------------------------

test(cpp_e2e_wot_basic, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_wot_basic', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_wot_basic/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_wot_basic/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_wot_multi, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_wot_multi', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_wot_multi/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_wot_multi/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_wot_format, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_wot_format', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_wot_format/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_wot_format/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_wot_string, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_wot_string', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_wot_string/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_wot_string/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_wot_codes, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_wot_codes', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_wot_codes/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_wot_codes/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_wot_empty, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_wot_empty', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_wot_empty/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_wot_empty/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_wot_goal_fails,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_wot_fails', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_wot_goal_fails/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_wot_goal_fails/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_wot_tab, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_wot_tab', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_wot_tab/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_wot_tab/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_wot_nested, [condition(cpp_compiler_available)]) :-
    % Nested capture regression guard: ensures the inner frame''s
    % saved_cp uses after_pc from invoke_goal_as_call (not pc + 1
    % which is meaningless when the builtin is dispatched as a
    % goal-term).
    unique_cpp_tmp_dir('tmp_cpp_e2e_wot_nested', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_wot_nested/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_wot_nested/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% split_string/4 — separator + pad string splitting.
% ------------------------------------------------------------------

test(cpp_e2e_split_simple, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_split_simple', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_split_simple/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_split_simple/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_split_empty, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_split_empty', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_split_empty/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_split_empty/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_split_single, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_split_single', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_split_single/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_split_single/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_split_double_sep,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_split_dsep', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_split_double_sep/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_split_double_sep/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_split_pad, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_split_pad', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_split_pad/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_split_pad/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_split_sep_and_pad,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_split_sp', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_split_sep_and_pad/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_split_sep_and_pad/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_split_multi_sep,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_split_msep', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_split_multi_sep/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_split_multi_sep/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_split_pad_trailing,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_split_padt', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_split_pad_trailing/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_split_pad_trailing/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_split_atom_input,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_split_atom', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_split_atom_input/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_split_atom_input/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% term_to_atom/2 — bidirectional canonical-form term ↔ atom.
% Forward via render(), reverse via the canonical-form term parser.
% ------------------------------------------------------------------

test(cpp_e2e_tta_fwd_atom, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_tta_fwd_atom', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_tta_fwd_atom/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_tta_fwd_atom/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_tta_fwd_int, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_tta_fwd_int', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_tta_fwd_int/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_tta_fwd_int/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_tta_fwd_compound,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_tta_fwd_compound', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_tta_fwd_compound/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_tta_fwd_compound/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_tta_fwd_list, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_tta_fwd_list', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_tta_fwd_list/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_tta_fwd_list/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_tta_rev_atom, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_tta_rev_atom', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_tta_rev_atom/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_tta_rev_atom/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_tta_rev_int, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_tta_rev_int', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_tta_rev_int/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_tta_rev_int/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_tta_rev_neg, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_tta_rev_neg', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_tta_rev_neg/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_tta_rev_neg/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_tta_rev_float, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_tta_rev_float', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_tta_rev_float/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_tta_rev_float/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_tta_rev_compound,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_tta_rev_compound', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_tta_rev_compound/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_tta_rev_compound/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_tta_rev_list, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_tta_rev_list', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_tta_rev_list/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_tta_rev_list/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_tta_rev_nested,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_tta_rev_nested', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_tta_rev_nested/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_tta_rev_nested/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_tta_roundtrip,
     [condition(cpp_compiler_available)]) :-
    % Full render-then-parse round-trip — the parser accepts what
    % render produces.
    unique_cpp_tmp_dir('tmp_cpp_e2e_tta_rt', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_tta_roundtrip/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_tta_roundtrip/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% read/1 + read_term/1 — stdin term reading. Tests pipe input to
% the child via run_query_with_stdin, which writes the supplied
% text + closes the child''s stdin before reading stdout.
% ------------------------------------------------------------------

test(cpp_e2e_read_atom, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_read_atom', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_read_atom/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_with_stdin(BinPath, 'wam_cpp_test_read_atom/0',
                               [], "hello.\n", true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_read_int, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_read_int', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_read_int/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_with_stdin(BinPath, 'wam_cpp_test_read_int/0',
                               [], "42.\n", true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_read_compound, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_read_compound', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_read_compound/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_with_stdin(BinPath,
                               'wam_cpp_test_read_compound/0',
                               [], "foo(1, bar).\n", true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_read_list, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_read_list', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_read_list/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_with_stdin(BinPath, 'wam_cpp_test_read_list/0',
                               [], "[a, b, c].\n", true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_read_eof, [condition(cpp_compiler_available)]) :-
    % Empty stdin → atom end_of_file.
    unique_cpp_tmp_dir('tmp_cpp_e2e_read_eof', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_read_eof/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_with_stdin(BinPath, 'wam_cpp_test_read_eof/0',
                               [], "", true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_read_term_atom,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_read_term_atom', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_read_term_atom/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_with_stdin(BinPath,
                               'wam_cpp_test_read_term_atom/0',
                               [], "just_an_atom.\n", true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% get_char/1, get_code/1, peek_char/1, put_char/1, put_code/1 —
% single-char I/O. Input tests pipe one char via stdin; output
% tests assert exact stdout bytes.
% ------------------------------------------------------------------

test(cpp_e2e_get_char, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_get_char', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_get_char/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_with_stdin(BinPath, 'wam_cpp_test_get_char/0',
                               [], "h", true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_get_code, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_get_code', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_get_code/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_with_stdin(BinPath, 'wam_cpp_test_get_code/0',
                               [], "h", true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_get_char_eof,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_get_char_eof', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_get_char_eof/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_with_stdin(BinPath,
                               'wam_cpp_test_get_char_eof/0',
                               [], "", true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_get_code_eof,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_get_code_eof', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_get_code_eof/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_with_stdin(BinPath,
                               'wam_cpp_test_get_code_eof/0',
                               [], "", true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_peek_char, [condition(cpp_compiler_available)]) :-
    % peek_char/1 doesn''t consume — subsequent get_char/1 sees the
    % same byte. One char of stdin satisfies both reads.
    unique_cpp_tmp_dir('tmp_cpp_e2e_peek_char', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_peek_char/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_with_stdin(BinPath, 'wam_cpp_test_peek_char/0',
                               [], "a", true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_put_char, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_put_char', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_put_char/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_stdout(BinPath, 'wam_cpp_test_put_char/0', [],
                           true, "abc\n")
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_put_code, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_put_code', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_put_code/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query_stdout(BinPath, 'wam_cpp_test_put_code/0', [],
                           true, "Hi\n")
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_put_in_capture,
     [condition(cpp_compiler_available)]) :-
    % Capture regression guard — put_char/put_code route through
    % emit_output_char so with_output_to/2 sees them.
    unique_cpp_tmp_dir('tmp_cpp_e2e_put_capture', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_put_in_capture/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_put_in_capture/0',
                    [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% atomic_list_concat/2-3 + small atom/string/number conversions:
% atom_string/2, string_concat/3, string_length/2, number_chars/2,
% atom_to_term/3.
% ------------------------------------------------------------------

test(cpp_e2e_alc2, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_alc2', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_alc2/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_alc2/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_alc2_mixed, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_alc2_mixed', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_alc2_mixed/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_alc2_mixed/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_alc2_empty, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_alc2_empty', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_alc2_empty/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_alc2_empty/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_alc3_join, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_alc3_join', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_alc3_join/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_alc3_join/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_alc3_split, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_alc3_split', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_alc3_split/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_alc3_split/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_alc3_split_multi,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_alc3_split_multi', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_alc3_split_multi/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_alc3_split_multi/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_alc3_split_nosep,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_alc3_split_nosep', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_alc3_split_nosep/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_alc3_split_nosep/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_atom_string_fwd,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_atom_string_fwd', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_atom_string_fwd/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_atom_string_fwd/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_atom_string_rev,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_atom_string_rev', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_atom_string_rev/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_atom_string_rev/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_string_length_alias,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_string_length', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_string_length_alias/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_string_length_alias/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_string_concat_alias,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_string_concat', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project(
            [user:wam_cpp_test_string_concat_alias/0],
            [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_string_concat_alias/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_number_chars_fwd,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_number_chars_fwd', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_number_chars_fwd/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_number_chars_fwd/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_number_chars_rev,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_number_chars_rev', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_number_chars_rev/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath,
                    'wam_cpp_test_number_chars_rev/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_atom_to_term, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_atom_to_term', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_atom_to_term/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_atom_to_term/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% ------------------------------------------------------------------
% Arithmetic builtins: succ/2 (direct bidirectional) and between/3
% (helper-injected, nondet via the standard two-clause definition).
% ------------------------------------------------------------------

test(cpp_e2e_succ, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_succ', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_succ_fwd/0,
                               user:wam_cpp_test_succ_bwd/0,
                               user:wam_cpp_test_succ_zero/0,
                               user:wam_cpp_test_succ_neg_fail/0,
                               user:wam_cpp_test_succ_y_zero_fail/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_succ_fwd/0',         [], true),
          run_query(BinPath, 'wam_cpp_test_succ_bwd/0',         [], true),
          run_query(BinPath, 'wam_cpp_test_succ_zero/0',        [], true),
          % succ(-1, _) and succ(_, 0) both fail per ISO domain.
          run_query(BinPath, 'wam_cpp_test_succ_neg_fail/0',    [], false),
          run_query(BinPath, 'wam_cpp_test_succ_y_zero_fail/0', [], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_between, [condition(cpp_compiler_available)]) :-
    % between/3 is helper-injected. The enum case exercises the full
    % nondet path: findall drives backtracking through both clauses.
    unique_cpp_tmp_dir('tmp_cpp_e2e_between', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_test_between_first/0,
                               user:wam_cpp_test_between_enum/0,
                               user:wam_cpp_test_between_singleton/0,
                               user:wam_cpp_test_between_empty/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_between_first/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_between_enum/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_between_singleton/0', [], true),
          run_query(BinPath, 'wam_cpp_test_between_empty/0',     [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_switch_on_constant, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_swc', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_color/1, user:wam_cpp_shape/2],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          % First clause via "default" fall-through.
          run_query(BinPath, 'wam_cpp_color/1', [red],    true),
          % Later clauses reached via direct switch jump (bypassing
          % try_me_else; verifies the retry_me_else no-op fix).
          run_query(BinPath, 'wam_cpp_color/1', [green],  true),
          run_query(BinPath, 'wam_cpp_color/1', [blue],   true),
          % Bound non-key: switch returns false directly.
          run_query(BinPath, 'wam_cpp_color/1', [orange], false),
          run_query(BinPath, 'wam_cpp_shape/2', [circle,   round],   true),
          run_query(BinPath, 'wam_cpp_shape/2', [square,   angular], true),
          run_query(BinPath, 'wam_cpp_shape/2', [triangle, angular], true),
          run_query(BinPath, 'wam_cpp_shape/2', [circle,   angular], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(cpp_e2e_switch_on_term, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_swt', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_mixed/1, user:wam_cpp_listy/1],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          % Mixed clauses: atom, integer, structure — switch_on_term
          % dispatches by type.
          run_query(BinPath, 'wam_cpp_mixed/1', [a],          true),
          run_query(BinPath, 'wam_cpp_mixed/1', ['1'],        true),
          run_query(BinPath, 'wam_cpp_mixed/1', ['foo(x)'],   true),
          run_query(BinPath, 'wam_cpp_mixed/1', [b],          false),
          run_query(BinPath, 'wam_cpp_mixed/1', ['bar(x)'],   false),
          % List dispatch: [] takes the constant table, [_|_] takes
          % the list-pc path.
          run_query(BinPath, 'wam_cpp_listy/1', ['[]'],       true),
          run_query(BinPath, 'wam_cpp_listy/1', ['[a,b]'],    true),
          run_query(BinPath, 'wam_cpp_listy/1', [foo],        false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% Regression for the switch_on_constant_a2 off-by-one bug. The WAM-asm
% layer emits switch_on_constant_a2 at the head of any multi-clause
% predicate that dispatches on A2 (rather than A1). Before the fix the
% C++ emitter rendered it as a /comment/, but the PC-counter still
% advanced — so every label after the first such predicate pointed
% one instruction past its real entry. The downstream check
% (wam_cpp_test_a2_downstream_label) catches that: if
% wam_cpp_after_a2's label is wrong, the call lands inside the prior
% predicate's frame and either crashes or returns false.
test(cpp_e2e_switch_on_constant_a2, [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_swa2', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_a2dispatch/4,
                               user:wam_cpp_after_a2/0,
                               user:wam_cpp_test_a2_all_clauses/0,
                               user:wam_cpp_test_a2_downstream_label/0,
                               user:wam_cpp_a2tag/3,
                               user:wam_cpp_test_a2_direct_jump/0,
                               user:wam_cpp_test_a2_no_match_fails/0,
                               user:wam_cpp_test_a2_unbound_enumerates/0,
                               user:wam_cpp_test_a2_shared_key_backtracks/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_a2_all_clauses/0',           [], true),
          run_query(BinPath, 'wam_cpp_test_a2_downstream_label/0',      [], true),
          run_query(BinPath, 'wam_cpp_test_a2_direct_jump/0',           [], true),
          run_query(BinPath, 'wam_cpp_test_a2_no_match_fails/0',        [], true),
          run_query(BinPath, 'wam_cpp_test_a2_unbound_enumerates/0',    [], true),
          run_query(BinPath, 'wam_cpp_test_a2_shared_key_backtracks/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% Mixed-mode A1 indexing — predicates with variable-A1 clauses no
% longer disable A1 indexing entirely. Clauses up to (but not
% including) the first variable-A1 clause get a
% switch_on_constant_fallthrough table; the bound-but-unmatched
% case falls through to the try_me_else chain so trailing variable
% clauses still match.
test(cpp_e2e_mixed_mode_a1_indexing,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_mma', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_mma_tag/2,
                               user:wam_cpp_test_mma_specific_hit/0,
                               user:wam_cpp_test_mma_unknown_falls_through/0,
                               user:wam_cpp_test_mma_specific_backtracks_to_default/0,
                               user:wam_cpp_test_mma_unknown_only_default/0,
                               user:wam_cpp_test_mma_all_unbound_enumerates/0,
                               user:wam_cpp_mma_mid/2,
                               user:wam_cpp_test_mma_mid_indexed/0,
                               user:wam_cpp_test_mma_mid_after_var/0,
                               user:wam_cpp_test_mma_mid_unbound/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_mma_specific_hit/0',                    [], true),
          run_query(BinPath, 'wam_cpp_test_mma_unknown_falls_through/0',           [], true),
          run_query(BinPath, 'wam_cpp_test_mma_specific_backtracks_to_default/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_mma_unknown_only_default/0',            [], true),
          run_query(BinPath, 'wam_cpp_test_mma_all_unbound_enumerates/0',          [], true),
          run_query(BinPath, 'wam_cpp_test_mma_mid_indexed/0',                     [], true),
          run_query(BinPath, 'wam_cpp_test_mma_mid_after_var/0',                   [], true),
          run_query(BinPath, 'wam_cpp_test_mma_mid_unbound/0',                     [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% Mixed-mode A2 indexing — same lift as the A1 case, but applied
% to A2 when A1 is variable in every clause. The indexed A2 prefix
% (clauses before the first variable-A2 clause) gets
% switch_on_constant_a2_fallthrough; bound-but-unmatched A2 falls
% through to the try_me_else chain so trailing variable clauses
% still match.
test(cpp_e2e_mixed_mode_a2_indexing,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_mma2', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_mma2_tag/3,
                               user:wam_cpp_test_mma2_specific_hit/0,
                               user:wam_cpp_test_mma2_unknown_falls_through/0,
                               user:wam_cpp_test_mma2_specific_backtracks_to_default/0,
                               user:wam_cpp_test_mma2_unknown_only_default/0,
                               user:wam_cpp_test_mma2_all_unbound_enumerates/0,
                               user:wam_cpp_mma2_mid/3,
                               user:wam_cpp_test_mma2_mid_indexed/0,
                               user:wam_cpp_test_mma2_mid_after_var/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_mma2_specific_hit/0',                    [], true),
          run_query(BinPath, 'wam_cpp_test_mma2_unknown_falls_through/0',           [], true),
          run_query(BinPath, 'wam_cpp_test_mma2_specific_backtracks_to_default/0',  [], true),
          run_query(BinPath, 'wam_cpp_test_mma2_unknown_only_default/0',            [], true),
          run_query(BinPath, 'wam_cpp_test_mma2_all_unbound_enumerates/0',          [], true),
          run_query(BinPath, 'wam_cpp_test_mma2_mid_indexed/0',                     [], true),
          run_query(BinPath, 'wam_cpp_test_mma2_mid_after_var/0',                   [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

% A2 structure + term indexing. Predicates whose A1 is variable in
% every clause currently get NO A1 indexing (correct: a variable
% head matches anything). Previously they also got no A2 indexing
% unless A2 was all atomic constants. With switch_on_structure_a2
% and switch_on_term_a2, multi-clause predicates with compound or
% mixed A2 also get O(1) dispatch.
test(cpp_e2e_switch_on_structure_and_term_a2,
     [condition(cpp_compiler_available)]) :-
    unique_cpp_tmp_dir('tmp_cpp_e2e_sta2', TmpDir),
    setup_call_cleanup(
        write_wam_cpp_project([user:wam_cpp_a2struct/2,
                               user:wam_cpp_test_a2_structure_direct/0,
                               user:wam_cpp_test_a2_structure_miss/0,
                               user:wam_cpp_test_a2_structure_unbound/0,
                               user:wam_cpp_a2term/3,
                               user:wam_cpp_test_a2_term_atom_clause/0,
                               user:wam_cpp_test_a2_term_compound_clause/0,
                               user:wam_cpp_test_a2_term_list_clauses/0,
                               user:wam_cpp_test_a2_term_unbound/0],
                              [emit_main(true)], TmpDir),
        ( build_e2e_binary(TmpDir, BinPath),
          run_query(BinPath, 'wam_cpp_test_a2_structure_direct/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_a2_structure_miss/0',       [], true),
          run_query(BinPath, 'wam_cpp_test_a2_structure_unbound/0',    [], true),
          run_query(BinPath, 'wam_cpp_test_a2_term_atom_clause/0',     [], true),
          run_query(BinPath, 'wam_cpp_test_a2_term_compound_clause/0', [], true),
          run_query(BinPath, 'wam_cpp_test_a2_term_list_clauses/0',    [], true),
          run_query(BinPath, 'wam_cpp_test_a2_term_unbound/0',         [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

compile_one(CppDir, Src, Obj, Status) :-
    directory_file_path(CppDir, Src, SrcPath),
    directory_file_path(CppDir, Obj, ObjPath),
    process_create(path('g++'),
                   ['-std=c++17', '-c', '-o', ObjPath, SrcPath],
                   [stderr(null), process(PID)]),
    process_wait(PID, Status).

build_e2e_binary(TmpDir, BinPath) :-
    directory_file_path(TmpDir, 'cpp', CppDir),
    directory_file_path(CppDir, 'wam_runtime.cpp', Rt),
    directory_file_path(CppDir, 'generated_program.cpp', Prog),
    directory_file_path(CppDir, 'main.cpp', Main),
    directory_file_path(CppDir, 'cpp_test', BinPath),
    process_create(path('g++'),
                   ['-std=c++17', '-O0', '-o', BinPath, Rt, Prog, Main],
                   [stderr(null), process(PID)]),
    process_wait(PID, Status),
    assertion(Status == exit(0)).

run_query(BinPath, PredKey, Args, Expected) :-
    maplist(atom_string, Args, ArgStrs),
    process_create(BinPath, [PredKey|ArgStrs],
                   [stdout(pipe(Out)), stderr(null), process(PID)]),
    read_string(Out, _, Output),
    close(Out),
    process_wait(PID, _),
    normalize_space(string(Trimmed), Output),
    expected_str(Expected, ExpStr),
    assertion(Trimmed == ExpStr).

expected_str(true,  "true").
expected_str(false, "false").

% run_query_stdout(+BinPath, +PredKey, +Args, +Status, +ExpPrintedOut)
%  Captures full stdout (printed bytes from the predicate body, then
%  the trailing "true\n"/"false\n" line emitted by main) and asserts
%  exact equality with ExpPrintedOut ++ Status + "\n".
run_query_stdout(BinPath, PredKey, Args, Status, ExpPrint) :-
    maplist(atom_string, Args, ArgStrs),
    process_create(BinPath, [PredKey|ArgStrs],
                   [stdout(pipe(Out)), stderr(null), process(PID)]),
    read_string(Out, _, Output),
    close(Out),
    process_wait(PID, _),
    expected_str(Status, StatusStr),
    string_concat(ExpPrint, StatusStr, ExpWithStatus),
    string_concat(ExpWithStatus, "\n", Expected),
    assertion(Output == Expected).

% run_query_with_stdin(+BinPath, +PredKey, +Args, +StdinText, +Expected)
%  Pipes StdinText to the child''s stdin, then runs the same shape
%  of assertion as run_query (trimmed stdout matches "true"/"false").
%  Used for read_term/1 etc. tests that consume input.
run_query_with_stdin(BinPath, PredKey, Args, StdinText, Expected) :-
    maplist(atom_string, Args, ArgStrs),
    process_create(BinPath, [PredKey|ArgStrs],
                   [stdin(pipe(In)), stdout(pipe(Out)),
                    stderr(null), process(PID)]),
    write(In, StdinText),
    close(In),
    read_string(Out, _, Output),
    close(Out),
    process_wait(PID, _),
    normalize_space(string(Trimmed), Output),
    expected_str(Expected, ExpStr),
    assertion(Trimmed == ExpStr).

% ------------------------------------------------------------------
% ISO error configuration — plumbing PR tests. The key swap tables
% are intentionally empty in this PR so the rewrite is a no-op; the
% tests here exercise the config loader, the mode resolver, the
% inline-wins precedence, and the multi-module warning emission.
% Behavior-changing tests (cpp_e2e_iso_* / cpp_e2e_lax_* /
% cpp_e2e_explicit_*) land with the first ISO builtin.
% ------------------------------------------------------------------

test(iso_errors_config_loader_basic) :-
    iso_errors_temp_config_file(Path, [
        'iso_errors_default(true).',
        'iso_errors_override(legacy_lookup/3, false).',
        'iso_errors_override(unsafe_div/3, false).',
        'iso_errors_override(experimental:my_pred/2, true).',
        'some_future_fact(hello).'
    ]),
    setup_call_cleanup(
        true,
        ( wam_cpp_target:iso_errors_load_config(Path, Config),
          assertion(Config == iso_config(true,
              [legacy_lookup/3-false,
               unsafe_div/3-false,
               (experimental:my_pred/2)-true])),
          % mode_for resolution, including bare-PI cross-module match.
          wam_cpp_target:iso_errors_mode_for(Config,
              user:legacy_lookup/3, M1),
          assertion(M1 == false),
          wam_cpp_target:iso_errors_mode_for(Config,
              user:never_listed/2, M2),
          assertion(M2 == true),                  % falls back to default
          wam_cpp_target:iso_errors_mode_for(Config,
              experimental:my_pred/2, M3),
          assertion(M3 == true),
          wam_cpp_target:iso_errors_mode_for(Config,
              other_mod:my_pred/2, M4),
          assertion(M4 == true)                   % only experimental: matches; default wins
        ),
        delete_file(Path)
    ).

test(iso_errors_inline_wins_over_file) :-
    iso_errors_temp_config_file(Path, [
        'iso_errors_default(false).',
        'iso_errors_override(legacy_lookup/3, false).'
    ]),
    setup_call_cleanup(
        true,
        ( % File says false; inline says true. Inline wins.
          wam_cpp_target:iso_errors_resolve_options(
              [iso_errors_config(Path),
               iso_errors(true),
               iso_errors(legacy_lookup/3, true)],
              Config),
          wam_cpp_target:iso_errors_mode_for(Config,
              user:legacy_lookup/3, M1),
          assertion(M1 == true),
          wam_cpp_target:iso_errors_mode_for(Config,
              user:never_listed/2, M2),
          assertion(M2 == true)                   % inline default wins too
        ),
        delete_file(Path)
    ).

test(iso_errors_multi_module_warning) :-
    % Capture user_error output via with_output_to. Verify the
    % warning fires when a bare override matches predicates from
    % two different modules in the input list.
    Config = iso_config(false, [safe_div/2-false]),
    Predicates = [mod_a:safe_div/2, mod_b:safe_div/2, mod_c:other/3],
    with_output_to(string(Captured),
        % stderr is the actual target — redirect via user_error.
        ( current_output(Curr),
          set_stream(Curr, alias(user_error)),
          wam_cpp_target:iso_errors_warn_multi_module(Config, Predicates),
          set_stream(user_error, alias(user_error))
        )),
    assertion(sub_string(Captured, _, _, _,
        "matches 2 predicates")),
    assertion(sub_string(Captured, _, _, _, "mod_a")),
    assertion(sub_string(Captured, _, _, _, "mod_b")).

test(iso_errors_audit_structure) :-
    % With empty key tables, every site is `default` with no flip.
    % Verifies the audit machinery walks predicates + reports the
    % expected record shape, even without behavior changes yet.
    Options = [iso_errors(test_audit_pred/0, true)],
    wam_cpp_target:wam_cpp_iso_audit(
        [user:wam_cpp_test_audit_pred/0],
        Options,
        Audit),
    % One predicate in input → one audit record.
    assertion(Audit = [audit(user:wam_cpp_test_audit_pred/0, _Mode, _Sites)]).

:- dynamic user:wam_cpp_test_audit_pred/0.
user:wam_cpp_test_audit_pred :- X is 1 + 2, X = 3.

% Helper: write a list of lines to a temp file, return its path.
iso_errors_temp_config_file(Path, Lines) :-
    get_time(T), N is round(T * 1000),
    format(atom(Path), '/tmp/iso_cfg_~w.pl', [N]),
    setup_call_cleanup(
        open(Path, write, Out),
        forall(member(L, Lines), format(Out, '~w~n', [L])),
        close(Out)).

% Unit tests for compile_predicates_for_project''s on_compile_error
% diagnostic policy. The handler logs / re-throws / drops based on
% Policy. Verified by calling handle_compile_error/3 directly and
% inspecting the failure / exception path.

test(compile_error_throw_policy_rethrows) :-
    catch(
        wam_cpp_target:handle_compile_error(throw, foo/1,
                                            error(test_marker, _)),
        Caught,
        Caught = error(test_marker, _)
    ),
    assertion(nonvar(Caught)).

test(compile_error_warn_policy_fails) :-
    \+ wam_cpp_target:handle_compile_error(warn, foo/1,
                                          error(test_marker, _)).

test(compile_error_skip_policy_fails) :-
    \+ wam_cpp_target:handle_compile_error(skip, foo/1,
                                          error(test_marker, _)).

test(compile_error_default_is_warn) :-
    % Anything that isn''t throw/skip falls through to the warn
    % handler. Using `unknown_policy` here exercises that branch.
    \+ wam_cpp_target:handle_compile_error(unknown_policy, foo/1,
                                          error(test_marker, _)).

:- end_tests(wam_cpp_generator).

% --------------------------------------------------------------------
% Helpers
% --------------------------------------------------------------------

unique_cpp_tmp_dir(Prefix, Dir) :-
    get_time(T), N is round(T * 1000),
    format(atom(Dir), 'tests/~w_~w', [Prefix, N]).

cpp_compiler_available :-
    catch(
        ( process_create(path('g++'), ['--version'],
                         [stdout(null), stderr(null), process(PID)]),
          process_wait(PID, exit(0))
        ),
        _,
        fail).
