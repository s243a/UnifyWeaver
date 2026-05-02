:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_wam_scala_runtime_smoke.pl
%
% Runtime smoke tests for the WAM Scala hybrid target.
% Uses the scalac compiler to compile the generated project and
% the scala runner to execute it, verifying correctness of the
% stepping interpreter.
%
% Tests are gated on scalac availability to match the Clojure target
% pattern. Set SCALA_SMOKE_TESTS=1 or install scalac to enable.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_scala_target').

% ============================================================
% Test predicates
% ============================================================

:- dynamic user:wam_fact/1.
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
:- dynamic user:wam_use_struct/1.
:- dynamic user:wam_make_struct/1.
:- dynamic user:wam_env1/1.
:- dynamic user:wam_env2/1.
:- dynamic user:wam_trail_choice/1.
:- dynamic user:wam_cut_first/1.
:- dynamic user:wam_cut_caller/1.
:- dynamic user:wam_list_match/1.
:- dynamic user:wam_use_list/1.
:- dynamic user:wam_make_list/1.
:- dynamic user:wam_foreign_yes/1.
:- dynamic user:wam_foreign_yes_caller/1.
:- dynamic user:wam_foreign_pair/2.
:- dynamic user:wam_foreign_pair_query/1.
:- dynamic user:wam_foreign_multi/2.
:- dynamic user:wam_foreign_multi_query/1.
:- dynamic user:wam_inc/2.
:- dynamic user:wam_double/2.
:- dynamic user:wam_combo/3.
:- dynamic user:wam_neg/2.
:- dynamic user:wam_lt/2.
:- dynamic user:wam_gt/2.
:- dynamic user:wam_geq/2.
:- dynamic user:wam_leq/2.
:- dynamic user:wam_eq_arith/2.
:- dynamic user:wam_neq_arith/2.
:- dynamic user:wam_cut_obs/1.
:- dynamic user:wam_always_fail/1.
:- dynamic user:wam_dummy_a/1.
:- dynamic user:wam_neg_dummy/1.
:- dynamic user:wam_float_mul/2.
:- dynamic user:wam_float_div/2.
:- dynamic user:wam_float_lit/1.
:- dynamic user:wam_float_cmp_lt/2.
:- dynamic user:wam_int_div_true/2.
:- dynamic user:wam_pair_inline/2.
:- dynamic user:wam_pair_sidecar/2.
:- dynamic user:wam_pair_file/2.
:- dynamic user:wam_call_dummy/1.
:- dynamic user:wam_meta_dummy/1.
:- dynamic user:wam_member_q/2.
:- dynamic user:wam_member_or_abc/1.
:- dynamic user:wam_length_q/2.
:- dynamic user:wam_append_q/3.
:- dynamic user:wam_findall_dummy/1.
:- dynamic user:wam_findall_simple/1.
:- dynamic user:wam_findall_template/1.
:- dynamic user:wam_findall_empty/1.
:- dynamic user:wam_call2/2.
:- dynamic user:wam_pred1/2.
:- dynamic user:wam_atom_codes_q/2.
:- dynamic user:wam_atom_length_q/2.
:- dynamic user:wam_append_split_q/2.
:- dynamic user:wam_length_gen_q/2.
:- dynamic user:wam_var_q/1.
:- dynamic user:wam_nonvar_q/1.
:- dynamic user:wam_atom_q/1.
:- dynamic user:wam_number_q/1.
:- dynamic user:wam_is_list_q/1.
:- dynamic user:wam_ground_q/1.
:- dynamic user:wam_atomic_q/1.
:- dynamic user:wam_copy_term_q/2.
:- dynamic user:wam_sort_q/2.
:- dynamic user:wam_msort_q/2.
:- dynamic user:wam_bagof_q/1.
:- dynamic user:wam_setof_q/1.
:- dynamic user:wam_setof_dups_dummy/1.
:- dynamic user:wam_setof_dups_q/1.
:- dynamic user:wam_bagof_empty_q/1.
:- dynamic user:wam_var_check_then_bind/1.
:- dynamic user:wam_dup_first_arg/2.
:- dynamic user:wam_dup_first_q/2.
:- dynamic user:wam_between_q/3.
:- dynamic user:wam_between_collect/3.
:- dynamic user:wam_format_w/1.
:- dynamic user:wam_format_combo/2.
:- dynamic user:wam_format_dn/1.

user:wam_fact(a).
user:wam_execute_caller(X)    :- user:wam_fact(X).
user:wam_call_caller(X)       :- user:wam_fact(X), user:wam_fact(X).
user:wam_choice_fact(a).
user:wam_choice_fact(b).
user:wam_choice_fact(c).
user:wam_choice_caller(X)     :- user:wam_choice_fact(X).
user:wam_choice_or_z(X)       :- user:wam_choice_fact(X).
user:wam_choice_or_z(z).
user:wam_bind_then_fact(X)    :- Y = X, user:wam_fact(Y).
user:wam_bind_after_call(X)   :- user:wam_fact(X), X = a.
user:wam_bind_before_execute(X) :- X = a, user:wam_fact(X).
user:wam_if_then_else(X)      :- (X = a -> true ; X = b).
user:wam_struct_fact(f(a)).
user:wam_use_struct(X)        :- user:wam_struct_fact(X).
user:wam_make_struct(X)       :- X = f(a).
user:wam_env1(X)              :- Y = X, Z = a, Y = Z.
user:wam_env2(X)              :- user:wam_fact(X), Y = X, user:wam_fact(Y).
user:wam_trail_choice(X)      :- (Y = a ; Y = b), X = Y.
% Hard cut: first clause commits on a, second clause handles b.
user:wam_cut_first(a) :- !.
user:wam_cut_first(b).
% Cut inside an inner predicate, then call from an outer caller, to
% exercise the cutBar interaction across allocate/deallocate frames.
user:wam_cut_caller(X) :- user:wam_cut_first(X).
% List match exercises get_list / unify_constant / unify_variable
% and the get_structure [|]/2 nested case.
user:wam_list_match([a, b]).
user:wam_use_list(L)   :- user:wam_list_match(L).
% List build exercises put_list / set_constant / set_variable and
% put_structure [|]/2.
user:wam_make_list(X)  :- X = [a, b].

% Foreign predicate stubs. When `foreign_predicates([P/A,...])` is
% passed to the project writer, the WAM body is replaced with a single
% CallForeign instruction; these Prolog bodies exist only so the
% predicates are declared.
user:wam_foreign_yes(_).
user:wam_foreign_yes_caller(X)  :- user:wam_foreign_yes(X).
user:wam_foreign_pair(_, _).
user:wam_foreign_pair_query(Y)  :- user:wam_foreign_pair(a, Y).
user:wam_foreign_multi(_, _).
user:wam_foreign_multi_query(Y) :- user:wam_foreign_multi(a, Y).
% wam_cut_obs: Y is fresh in the body; foreign yields Y=a then Y=b on
% backtrack. The cut after the foreign call must drop the foreign CP so
% that a later unify failure cannot redo the foreign and try the second
% solution. With correct cut: wam_cut_obs(b) → false. Without: would be true.
user:wam_cut_obs(X) :- user:wam_foreign_multi(a, Y), !, Y = X.

% Arithmetic: WAM emits builtin_call is/2, =:=/2, </2, >/2, =</2, >=/2,
% =\=/2 and builds expression terms via put_structure +/-/*/'/' .
user:wam_inc(N, M)        :- M is N + 1.
user:wam_double(N, M)     :- M is N * 2.
user:wam_combo(A, B, R)   :- R is A + B * 2.
user:wam_neg(N, M)        :- M is -N.
user:wam_lt(X, Y)         :- X < Y.
user:wam_gt(X, Y)         :- X > Y.
user:wam_geq(X, Y)        :- X >= Y.
user:wam_leq(X, Y)        :- X =< Y.
user:wam_eq_arith(X, Y)   :- X =:= Y.
user:wam_neq_arith(X, Y)  :- X =\= Y.

% --- fail/0 and \+/1 ---
user:wam_always_fail(_X) :- fail.
user:wam_dummy_a(a).
% Negation-as-failure: succeeds only when the inner goal fails.
user:wam_neg_dummy(X) :- \+ user:wam_dummy_a(X).

% --- Float arithmetic ---
user:wam_float_mul(X, Y)    :- Y is X * 2.5.
user:wam_float_div(X, Y)    :- Y is X / 2.0.
user:wam_float_lit(Y)       :- Y is 3.14.
user:wam_float_cmp_lt(X, Y) :- X < Y.
% True division of two integers: 5/2 -> 2.5, not 2 (per SWI semantics).
user:wam_int_div_true(X, Y) :- Y is X / 2.

% --- S7 fact backend seam ---
% Same relation under two backends. The inline form has WAM-compiled
% facts; the sidecar form passes the same tuples via scala_fact_sources
% so the codegen synthesises a ForeignHandler that enumerates them.
% Both forms must give identical answers for every query.
user:wam_pair_inline(a, b).
user:wam_pair_inline(b, c).
user:wam_pair_inline(c, d).
user:wam_pair_sidecar(_, _).
user:wam_pair_file(_, _).

% --- call/1 meta-call ---
user:wam_call_dummy(a).
user:wam_call_dummy(b).
user:wam_meta_dummy(X) :- call(user:wam_call_dummy(X)).

% --- List builtins ---
user:wam_member_q(X, L)     :- member(X, L).
user:wam_member_or_abc(X)   :- member(X, [a, b, c]).
user:wam_length_q(L, N)     :- length(L, N).
user:wam_append_q(A, B, C)  :- append(A, B, C).

% --- findall/3 ---
user:wam_findall_dummy(a).
user:wam_findall_dummy(b).
user:wam_findall_simple(L)    :- findall(X, user:wam_findall_dummy(X), L).
user:wam_findall_template(L)  :- findall(p(X), user:wam_findall_dummy(X), L).
user:wam_findall_empty(L)     :- findall(X, user:wam_no_such_pred(X), L).

% --- call/N ---
user:wam_pred1(X, Y) :- Y = X.
% call(F, X) — F is the goal-atom; X is the additional arg.
% Calling wam_call2(wam_pred1, 5) should bind X=5 by calling wam_pred1(5,_)
user:wam_call2(F, X) :- call(F, X, X).

% --- atom_codes/2 ---
user:wam_atom_codes_q(A, Cs) :- atom_codes(A, Cs).
% --- atom_length/2 ---
user:wam_atom_length_q(A, N) :- atom_length(A, N).

% --- append/3 split mode and length/2 generative ---
user:wam_append_split_q(A, B) :- append(A, B, [a,b,c]).
user:wam_length_gen_q(L, N)   :- length(L, N).

% --- Type-check builtins ---
user:wam_var_q(X)        :- var(X).
user:wam_nonvar_q(X)     :- nonvar(X).
user:wam_atom_q(X)       :- atom(X).
user:wam_number_q(X)     :- number(X).
user:wam_is_list_q(X)    :- is_list(X).
user:wam_ground_q(X)     :- ground(X).
user:wam_atomic_q(X)     :- atomic(X).

% A predicate that exercises var/1 on a fresh variable: succeeds because
% Y is unbound when var(Y) runs.
user:wam_var_check_then_bind(X) :- var(Y), Y = X.

% --- copy_term/2 ---
user:wam_copy_term_q(A, B) :- copy_term(A, B).

% --- Sorting ---
user:wam_sort_q(L, S)  :- sort(L, S).
user:wam_msort_q(L, S) :- msort(L, S).

% --- bagof/3, setof/3 ---
user:wam_bagof_q(L)            :- bagof(X, user:wam_findall_dummy(X), L).
user:wam_setof_q(L)            :- setof(X, user:wam_findall_dummy(X), L).
user:wam_bagof_empty_q(L)      :- bagof(X, user:wam_no_such_pred_2(X), L).
user:wam_setof_dups_dummy(a).
user:wam_setof_dups_dummy(b).
user:wam_setof_dups_dummy(a).
user:wam_setof_dups_dummy(c).
user:wam_setof_dups_dummy(b).
user:wam_setof_dups_q(L) :- setof(X, user:wam_setof_dups_dummy(X), L).

% Two clauses with the SAME first argument — exercises the WAM
% compiler's first-arg indexing for the multi-match case. The
% switch_on_constant emitted for this predicate has `a` mapped to BOTH
% a default case (clause 1) AND an L_n case (clause 3). The runtime
% must enumerate both via the try_me_else chain, not jump directly to
% one and skip the other.
user:wam_dup_first_arg(a, b).
user:wam_dup_first_arg(c, d).
user:wam_dup_first_arg(a, e).
user:wam_dup_first_q(X, Y) :- user:wam_dup_first_arg(X, Y).

% --- between/3 ---
user:wam_between_q(L, H, X)         :- between(L, H, X).
user:wam_between_collect(L, H, R)   :- findall(X, between(L, H, X), R).

% --- format/2 ---
user:wam_format_w(X)            :- format("v=~w", [X]).
user:wam_format_combo(X, Y)     :- format("~a is ~w!", [X, Y]).
user:wam_format_dn(X)           :- format("n=~d~n", [X]).

% ============================================================
% Condition: only run if scalac is available
% ============================================================

%% scala_available/0 — true if scalac is on PATH
scala_available :-
    (   getenv('SCALA_SMOKE_TESTS', "1") -> true
    ;   catch(
            process_create(path(scalac), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            _,
            fail
        ),
        process_wait(Pid, exit(0))
    ).

% ============================================================
% Tests (gated on scala_available)
% ============================================================

:- begin_tests(wam_scala_runtime_smoke,
               [ condition(scala_available) ]).

test(fact_match) :-
    with_scala_project(
        [user:wam_fact/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_fact/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_fact/1', 'b', "false")
        )).

test(execute_caller) :-
    with_scala_project(
        [user:wam_execute_caller/1, user:wam_fact/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_execute_caller/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_execute_caller/1', 'b', "false")
        )).

test(call_caller) :-
    with_scala_project(
        [user:wam_call_caller/1, user:wam_fact/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_call_caller/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_call_caller/1', 'b', "false")
        )).

test(multi_clause_choice) :-
    with_scala_project(
        [user:wam_choice_fact/1, user:wam_choice_caller/1, user:wam_choice_or_z/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_choice_caller/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_choice_caller/1', 'b', "true"),
            verify_scala(TmpDir, 'wam_choice_caller/1', 'c', "true"),
            verify_scala(TmpDir, 'wam_choice_caller/1', 'd', "false"),
            verify_scala(TmpDir, 'wam_choice_or_z/1', 'z', "true")
        )).

test(binding_variants) :-
    with_scala_project(
        [user:wam_fact/1,
         user:wam_bind_then_fact/1,
         user:wam_bind_after_call/1,
         user:wam_bind_before_execute/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_bind_then_fact/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_bind_then_fact/1', 'b', "false"),
            verify_scala(TmpDir, 'wam_bind_after_call/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_bind_after_call/1', 'b', "false"),
            verify_scala(TmpDir, 'wam_bind_before_execute/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_bind_before_execute/1', 'b', "false")
        )).

test(if_then_else) :-
    with_scala_project(
        [user:wam_if_then_else/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_if_then_else/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_if_then_else/1', 'b', "true"),
            verify_scala(TmpDir, 'wam_if_then_else/1', 'c', "false")
        )).

test(structure_matching) :-
    with_scala_project(
        [user:wam_struct_fact/1, user:wam_use_struct/1, user:wam_make_struct/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_use_struct/1', 'f(a)', "true"),
            verify_scala(TmpDir, 'wam_use_struct/1', 'f(b)', "false"),
            verify_scala(TmpDir, 'wam_make_struct/1', 'f(a)', "true")
        )).

test(environment_frames) :-
    with_scala_project(
        [user:wam_fact/1, user:wam_env1/1, user:wam_env2/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_env1/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_env1/1', 'b', "false"),
            verify_scala(TmpDir, 'wam_env2/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_env2/1', 'b', "false")
        )).

test(trail_and_backtrack) :-
    with_scala_project(
        [user:wam_trail_choice/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_trail_choice/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_trail_choice/1', 'b', "true"),
            verify_scala(TmpDir, 'wam_trail_choice/1', 'c', "false")
        )).

% --- S3+S4 lockdown coverage ---

test(hard_cut) :-
    with_scala_project(
        [user:wam_cut_first/1, user:wam_cut_caller/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_cut_first/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_cut_first/1', 'b', "true"),
            verify_scala(TmpDir, 'wam_cut_first/1', 'c', "false"),
            verify_scala(TmpDir, 'wam_cut_caller/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_cut_caller/1', 'b', "true"),
            verify_scala(TmpDir, 'wam_cut_caller/1', 'c', "false")
        )).

test(list_match) :-
    with_scala_project(
        [user:wam_list_match/1, user:wam_use_list/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_use_list/1',   '[a,b]', "true"),
            verify_scala(TmpDir, 'wam_use_list/1',   '[a,c]', "false"),
            verify_scala(TmpDir, 'wam_list_match/1', '[a,b]', "true"),
            verify_scala(TmpDir, 'wam_list_match/1', '[a,b,c]', "false")
        )).

test(list_build) :-
    with_scala_project(
        [user:wam_make_list/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_make_list/1', '[a,b]', "true"),
            verify_scala(TmpDir, 'wam_make_list/1', '[a,c]', "false")
        )).

% --- Phase S5: foreign predicate integration ---

test(foreign_boolean) :-
    foreign_yes_handler(YesHandler),
    with_scala_project(
        [user:wam_foreign_yes/1, user:wam_foreign_yes_caller/1],
        [ foreign_predicates([wam_foreign_yes/1]),
          intern_atoms([a, b]),
          scala_foreign_handlers([handler(wam_foreign_yes/1, YesHandler)]) ],
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_foreign_yes_caller/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_foreign_yes_caller/1', 'b', "false")
        )).

test(foreign_binding) :-
    foreign_pair_handler(PairHandler),
    with_scala_project(
        [user:wam_foreign_pair/2, user:wam_foreign_pair_query/1],
        [ foreign_predicates([wam_foreign_pair/2]),
          intern_atoms([a, b, c]),
          scala_foreign_handlers([handler(wam_foreign_pair/2, PairHandler)]) ],
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_foreign_pair_query/1', 'b', "true"),
            verify_scala(TmpDir, 'wam_foreign_pair_query/1', 'c', "false")
        )).

test(foreign_multi) :-
    foreign_multi_handler(MultiHandler),
    with_scala_project(
        [user:wam_foreign_multi/2, user:wam_foreign_multi_query/1],
        [ foreign_predicates([wam_foreign_multi/2]),
          intern_atoms([a, b, c]),
          scala_foreign_handlers([handler(wam_foreign_multi/2, MultiHandler)]) ],
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_foreign_multi_query/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_foreign_multi_query/1', 'b', "true"),
            verify_scala(TmpDir, 'wam_foreign_multi_query/1', 'c', "false")
        )).

% S6: cut after a multi-solution foreign must drop the foreign choice
% point so a later unify failure cannot redo the foreign.
% Discriminating case: wam_cut_obs(b) with cut → false; without cut → true.
test(cut_after_foreign_multi) :-
    foreign_multi_handler(MultiHandler),
    with_scala_project(
        [user:wam_foreign_multi/2, user:wam_cut_obs/1],
        [ foreign_predicates([wam_foreign_multi/2]),
          intern_atoms([a, b, c]),
          scala_foreign_handlers([handler(wam_foreign_multi/2, MultiHandler)]) ],
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_cut_obs/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_cut_obs/1', 'b', "false"),
            verify_scala(TmpDir, 'wam_cut_obs/1', 'c', "false")
        )).

% --- Arithmetic builtins ---

test(arith_is) :-
    with_scala_project(
        [user:wam_inc/2, user:wam_double/2, user:wam_combo/3, user:wam_neg/2],
        _Opts,
        TmpDir,
        (
            verify_scala_args(TmpDir, 'wam_inc/2',    ['5', '6'],     "true"),
            verify_scala_args(TmpDir, 'wam_inc/2',    ['5', '7'],     "false"),
            verify_scala_args(TmpDir, 'wam_double/2', ['7', '14'],    "true"),
            verify_scala_args(TmpDir, 'wam_double/2', ['7', '15'],    "false"),
            verify_scala_args(TmpDir, 'wam_combo/3',  ['3','4','11'], "true"),
            verify_scala_args(TmpDir, 'wam_combo/3',  ['3','4','12'], "false"),
            verify_scala_args(TmpDir, 'wam_neg/2',    ['5', '-5'],    "true"),
            verify_scala_args(TmpDir, 'wam_neg/2',    ['5', '5'],     "false")
        )).

test(arith_compare) :-
    with_scala_project(
        [user:wam_lt/2, user:wam_gt/2, user:wam_geq/2, user:wam_leq/2,
         user:wam_eq_arith/2, user:wam_neq_arith/2],
        _Opts,
        TmpDir,
        (
            verify_scala_args(TmpDir, 'wam_lt/2',        ['1', '2'], "true"),
            verify_scala_args(TmpDir, 'wam_lt/2',        ['2', '1'], "false"),
            verify_scala_args(TmpDir, 'wam_lt/2',        ['2', '2'], "false"),
            verify_scala_args(TmpDir, 'wam_gt/2',        ['2', '1'], "true"),
            verify_scala_args(TmpDir, 'wam_gt/2',        ['1', '2'], "false"),
            verify_scala_args(TmpDir, 'wam_geq/2',       ['2', '2'], "true"),
            verify_scala_args(TmpDir, 'wam_geq/2',       ['1', '2'], "false"),
            verify_scala_args(TmpDir, 'wam_leq/2',       ['2', '2'], "true"),
            verify_scala_args(TmpDir, 'wam_leq/2',       ['3', '2'], "false"),
            verify_scala_args(TmpDir, 'wam_eq_arith/2',  ['3', '3'], "true"),
            verify_scala_args(TmpDir, 'wam_eq_arith/2',  ['3', '4'], "false"),
            verify_scala_args(TmpDir, 'wam_neq_arith/2', ['3', '4'], "true"),
            verify_scala_args(TmpDir, 'wam_neq_arith/2', ['3', '3'], "false")
        )).

% --- fail/0 builtin ---
test(builtin_fail) :-
    with_scala_project(
        [user:wam_always_fail/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_always_fail/1', 'a', "false"),
            verify_scala(TmpDir, 'wam_always_fail/1', 'b', "false")
        )).

% --- \+/1 negation-as-failure ---
% wam_neg_dummy(a) → false (wam_dummy_a(a) succeeds, so \+ fails)
% wam_neg_dummy(b) → true  (wam_dummy_a(b) fails,    so \+ succeeds)
test(builtin_negation) :-
    with_scala_project(
        [user:wam_dummy_a/1, user:wam_neg_dummy/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_neg_dummy/1', 'a', "false"),
            verify_scala(TmpDir, 'wam_neg_dummy/1', 'b', "true"),
            verify_scala(TmpDir, 'wam_neg_dummy/1', 'c', "true")
        )).

% --- Float arithmetic ---
test(arith_float) :-
    with_scala_project(
        [user:wam_float_mul/2, user:wam_float_div/2,
         user:wam_float_lit/1, user:wam_float_cmp_lt/2,
         user:wam_int_div_true/2],
        _Opts,
        TmpDir,
        (
            verify_scala_args(TmpDir, 'wam_float_mul/2',    ['4', '10.0'],   "true"),
            verify_scala_args(TmpDir, 'wam_float_mul/2',    ['4', '11.0'],   "false"),
            verify_scala_args(TmpDir, 'wam_float_div/2',    ['9', '4.5'],    "true"),
            verify_scala_args(TmpDir, 'wam_float_div/2',    ['9', '4.0'],    "false"),
            verify_scala_args(TmpDir, 'wam_float_lit/1',    ['3.14'],        "true"),
            verify_scala_args(TmpDir, 'wam_float_lit/1',    ['3.15'],        "false"),
            verify_scala_args(TmpDir, 'wam_float_cmp_lt/2', ['1.5', '2.5'],  "true"),
            verify_scala_args(TmpDir, 'wam_float_cmp_lt/2', ['2.5', '1.5'],  "false"),
            % Mixed Int/Float comparison should promote.
            verify_scala_args(TmpDir, 'wam_float_cmp_lt/2', ['1', '2.5'],    "true"),
            verify_scala_args(TmpDir, 'wam_float_cmp_lt/2', ['2.5', '3'],    "true"),
            % True division of integers yields a Float.
            verify_scala_args(TmpDir, 'wam_int_div_true/2', ['5', '2.5'],    "true"),
            verify_scala_args(TmpDir, 'wam_int_div_true/2', ['5', '2'],      "false")
        )).

% --- S7 fact backend seam: parity inline-vs-sidecar ---
% Both versions answer the same set of queries identically.
test(fact_source_inline_vs_sidecar) :-
    InlineQueries =
        [['a','b']-true, ['a','c']-false, ['b','c']-true,
         ['c','d']-true, ['x','y']-false],
    % Inline backend: WAM-compiled facts.
    with_scala_project(
        [user:wam_pair_inline/2],
        _OptsInline,
        TmpDir1,
        forall(member(Args-Expected, InlineQueries),
               ( (Expected == true -> S = "true" ; S = "false"),
                 verify_scala_args(TmpDir1, 'wam_pair_inline/2', Args, S)
               ))),
    % Sidecar backend: same tuples, declared via scala_fact_sources.
    with_scala_project(
        [user:wam_pair_sidecar/2],
        [ scala_fact_sources(
              [source(wam_pair_sidecar/2, [[a,b], [b,c], [c,d]])]) ],
        TmpDir2,
        forall(member(Args-Expected, InlineQueries),
               ( (Expected == true -> S = "true" ; S = "false"),
                 verify_scala_args(TmpDir2, 'wam_pair_sidecar/2', Args, S)
               ))).

% --- File-backed fact source ---
% Same parity check, but the tuples come from a CSV file at runtime.
test(fact_source_file_backed) :-
    write_facts_csv('_tmp_facts.csv', [[a,b], [b,c], [c,d]]),
    absolute_file_name('_tmp_facts.csv', AbsPath),
    Queries =
        [['a','b']-true, ['a','c']-false, ['b','c']-true,
         ['c','d']-true, ['x','y']-false],
    setup_call_cleanup(
        true,
        with_scala_project(
            [user:wam_pair_file/2],
            % Atoms in the CSV are read at runtime, so they need to be
            % declared up-front for parseFactArg to find them in the
            % runtime intern table. Otherwise every atom collapses to id
            % -1 and the test can't discriminate matches from mismatches.
            [ intern_atoms([a, b, c, d, x, y]),
              scala_fact_sources(
                  [source(wam_pair_file/2, file(AbsPath))]) ],
            TmpDir,
            forall(member(Args-Expected, Queries),
                   ( (Expected == true -> S = "true" ; S = "false"),
                     verify_scala_args(TmpDir, 'wam_pair_file/2', Args, S)
                   ))),
        catch(delete_file('_tmp_facts.csv'), _, true)).

% --- call/1 meta-call ---
test(call_meta) :-
    with_scala_project(
        [user:wam_call_dummy/1, user:wam_meta_dummy/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_meta_dummy/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_meta_dummy/1', 'b', "true"),
            verify_scala(TmpDir, 'wam_meta_dummy/1', 'c', "false")
        )).

% --- List builtins ---
test(builtin_member) :-
    with_scala_project(
        [user:wam_member_q/2, user:wam_member_or_abc/1],
        [ intern_atoms([a, b, c, d]) ],
        TmpDir,
        (
            verify_scala_args(TmpDir, 'wam_member_q/2', ['a', '[a,b,c]'], "true"),
            verify_scala_args(TmpDir, 'wam_member_q/2', ['b', '[a,b,c]'], "true"),
            verify_scala_args(TmpDir, 'wam_member_q/2', ['c', '[a,b,c]'], "true"),
            verify_scala_args(TmpDir, 'wam_member_q/2', ['d', '[a,b,c]'], "false"),
            verify_scala_args(TmpDir, 'wam_member_q/2', ['a', '[]'],      "false"),
            verify_scala(TmpDir, 'wam_member_or_abc/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_member_or_abc/1', 'b', "true"),
            verify_scala(TmpDir, 'wam_member_or_abc/1', 'c', "true"),
            verify_scala(TmpDir, 'wam_member_or_abc/1', 'd', "false")
        )).

test(builtin_length) :-
    with_scala_project(
        [user:wam_length_q/2],
        _Opts,
        TmpDir,
        (
            verify_scala_args(TmpDir, 'wam_length_q/2', ['[]',      '0'], "true"),
            verify_scala_args(TmpDir, 'wam_length_q/2', ['[a]',     '1'], "true"),
            verify_scala_args(TmpDir, 'wam_length_q/2', ['[a,b,c]', '3'], "true"),
            verify_scala_args(TmpDir, 'wam_length_q/2', ['[a,b]',   '3'], "false")
        )).

test(builtin_append) :-
    with_scala_project(
        [user:wam_append_q/3],
        [ intern_atoms([a, b, c]) ],
        TmpDir,
        (
            verify_scala_args(TmpDir, 'wam_append_q/3',
                              ['[a]', '[b,c]', '[a,b,c]'], "true"),
            verify_scala_args(TmpDir, 'wam_append_q/3',
                              ['[]', '[a,b]', '[a,b]'], "true"),
            verify_scala_args(TmpDir, 'wam_append_q/3',
                              ['[a,b]', '[]', '[a,b]'], "true"),
            verify_scala_args(TmpDir, 'wam_append_q/3',
                              ['[a]', '[b]', '[a,c]'], "false")
        )).

% --- findall/3 ---
test(findall_simple) :-
    with_scala_project(
        [user:wam_findall_dummy/1, user:wam_findall_simple/1],
        _Opts,
        TmpDir,
        (
            % Two clauses: wam_findall_dummy(a) and (b). Bag is [a, b].
            verify_scala(TmpDir, 'wam_findall_simple/1', '[a,b]', "true"),
            verify_scala(TmpDir, 'wam_findall_simple/1', '[b,a]', "false"),
            verify_scala(TmpDir, 'wam_findall_simple/1', '[]',    "false"),
            verify_scala(TmpDir, 'wam_findall_simple/1', '[a]',   "false")
        )).

test(findall_template) :-
    with_scala_project(
        [user:wam_findall_dummy/1, user:wam_findall_template/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_findall_template/1', '[p(a),p(b)]', "true"),
            verify_scala(TmpDir, 'wam_findall_template/1', '[p(a)]',      "false"),
            verify_scala(TmpDir, 'wam_findall_template/1', '[]',          "false")
        )).

test(findall_empty) :-
    with_scala_project(
        [user:wam_findall_empty/1],
        _Opts,
        TmpDir,
        (
            % No clauses for the inner goal → bag is [].
            verify_scala(TmpDir, 'wam_findall_empty/1', '[]',  "true"),
            verify_scala(TmpDir, 'wam_findall_empty/1', '[a]', "false")
        )).

% --- call/N ---
test(call_n_arity_3) :-
    with_scala_project(
        [user:wam_pred1/2, user:wam_call2/2],
        [ intern_atoms([wam_pred1, foo]) ],
        TmpDir,
        (
            % wam_call2(wam_pred1, X) calls wam_pred1(X, X) — succeeds for
            % any X (X = X by unification).
            verify_scala_args(TmpDir, 'wam_call2/2', ['wam_pred1', 'foo'], "true"),
            % If the goal-atom doesn't resolve, expect false.
            verify_scala_args(TmpDir, 'wam_call2/2', ['no_such_pred', 'foo'], "false")
        )).

% --- atom_codes/2 ---
test(builtin_atom_codes) :-
    with_scala_project(
        [user:wam_atom_codes_q/2],
        [ intern_atoms([abc, hi]) ],
        TmpDir,
        (
            % "abc" -> [97, 98, 99]
            verify_scala_args(TmpDir, 'wam_atom_codes_q/2',
                              ['abc', '[97,98,99]'], "true"),
            verify_scala_args(TmpDir, 'wam_atom_codes_q/2',
                              ['hi',  '[104,105]'], "true"),
            verify_scala_args(TmpDir, 'wam_atom_codes_q/2',
                              ['abc', '[97,98]'], "false")
        )).

% --- atom_length/2 ---
test(builtin_atom_length) :-
    with_scala_project(
        [user:wam_atom_length_q/2],
        [ intern_atoms([abc, hi, '']) ],
        TmpDir,
        (
            verify_scala_args(TmpDir, 'wam_atom_length_q/2', ['abc', '3'], "true"),
            verify_scala_args(TmpDir, 'wam_atom_length_q/2', ['hi',  '2'], "true"),
            verify_scala_args(TmpDir, 'wam_atom_length_q/2', ['abc', '4'], "false")
        )).

% --- append/3 split mode ---
test(builtin_append_split) :-
    with_scala_project(
        [user:wam_append_split_q/2],
        [ intern_atoms([a, b, c]) ],
        TmpDir,
        (
            % Splits of [a,b,c]: ([], [a,b,c]) ([a], [b,c]) ([a,b], [c]) ([a,b,c], [])
            verify_scala_args(TmpDir, 'wam_append_split_q/2', ['[]',     '[a,b,c]'], "true"),
            verify_scala_args(TmpDir, 'wam_append_split_q/2', ['[a]',    '[b,c]'],   "true"),
            verify_scala_args(TmpDir, 'wam_append_split_q/2', ['[a,b]',  '[c]'],     "true"),
            verify_scala_args(TmpDir, 'wam_append_split_q/2', ['[a,b,c]','[]'],      "true"),
            % Not a valid split:
            verify_scala_args(TmpDir, 'wam_append_split_q/2', ['[a]',    '[c]'],     "false")
        )).

% --- length/2 generative mode ---
% length(L, N) with L unbound and N ground → builds list of N fresh vars.
% A list of N fresh vars unifies with any concrete list of length N.
test(builtin_length_generative) :-
    with_scala_project(
        [user:wam_length_gen_q/2],
        [ intern_atoms([x, y, z]) ],
        TmpDir,
        (
            % L=[x,y,z], N=3 → ground both → unify length 3 with 3 → true
            verify_scala_args(TmpDir, 'wam_length_gen_q/2', ['[x,y,z]', '3'], "true"),
            % L=[x,y], N=3 → 2 != 3 → false
            verify_scala_args(TmpDir, 'wam_length_gen_q/2', ['[x,y]',   '3'], "false"),
            % We can't easily exercise pure generative mode from CLI
            % (would need an unbound L), but the round-trip with a
            % ground list of matching length confirms the deterministic
            % path; the generative branch is exercised when the WAM
            % runtime sees an unbound first arg with a bound length.
            verify_scala_args(TmpDir, 'wam_length_gen_q/2', ['[]',      '0'], "true")
        )).

% --- Type-check builtins ---
test(builtin_var_nonvar) :-
    with_scala_project(
        [user:wam_var_q/1, user:wam_nonvar_q/1, user:wam_var_check_then_bind/1],
        [ intern_atoms([a, b]) ],
        TmpDir,
        (
            % var(a) — atom, not unbound — fails.
            verify_scala(TmpDir, 'wam_var_q/1',    'a', "false"),
            verify_scala(TmpDir, 'wam_nonvar_q/1', 'a', "true"),
            % var(Y) inside a body where Y is locally fresh — succeeds.
            verify_scala(TmpDir, 'wam_var_check_then_bind/1', 'a', "true")
        )).

test(builtin_atom_number) :-
    with_scala_project(
        [user:wam_atom_q/1, user:wam_number_q/1, user:wam_atomic_q/1],
        [ intern_atoms([a, b]) ],
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_atom_q/1',    'a',    "true"),
            verify_scala(TmpDir, 'wam_atom_q/1',    '5',    "false"),
            verify_scala(TmpDir, 'wam_atom_q/1',    'f(a)', "false"),
            verify_scala(TmpDir, 'wam_number_q/1',  '5',    "true"),
            verify_scala(TmpDir, 'wam_number_q/1',  '3.14', "true"),
            verify_scala(TmpDir, 'wam_number_q/1',  'a',    "false"),
            verify_scala(TmpDir, 'wam_atomic_q/1',  'a',    "true"),
            verify_scala(TmpDir, 'wam_atomic_q/1',  '5',    "true"),
            verify_scala(TmpDir, 'wam_atomic_q/1',  'f(a)', "false")
        )).

test(builtin_is_list_ground) :-
    with_scala_project(
        [user:wam_is_list_q/1, user:wam_ground_q/1],
        [ intern_atoms([a, b]) ],
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_is_list_q/1', '[a,b]', "true"),
            verify_scala(TmpDir, 'wam_is_list_q/1', '[]',    "true"),
            verify_scala(TmpDir, 'wam_is_list_q/1', 'a',     "false"),
            verify_scala(TmpDir, 'wam_ground_q/1',  'a',     "true"),
            verify_scala(TmpDir, 'wam_ground_q/1',  'f(a)',  "true"),
            verify_scala(TmpDir, 'wam_ground_q/1',  '[a,b]', "true")
        )).

% --- copy_term/2 ---
% Ground terms copy to themselves; copy_term(a, a) succeeds.
test(builtin_copy_term) :-
    with_scala_project(
        [user:wam_copy_term_q/2],
        [ intern_atoms([a, b, foo]) ],
        TmpDir,
        (
            verify_scala_args(TmpDir, 'wam_copy_term_q/2', ['a', 'a'],         "true"),
            verify_scala_args(TmpDir, 'wam_copy_term_q/2', ['a', 'b'],         "false"),
            verify_scala_args(TmpDir, 'wam_copy_term_q/2', ['foo(a)','foo(a)'], "true"),
            verify_scala_args(TmpDir, 'wam_copy_term_q/2', ['foo(a)','foo(b)'], "false")
        )).

% --- Sorting ---
test(builtin_sort) :-
    with_scala_project(
        [user:wam_sort_q/2, user:wam_msort_q/2],
        [ intern_atoms([a, b, c, d]) ],
        TmpDir,
        (
            % sort/2: dedup and order
            verify_scala_args(TmpDir, 'wam_sort_q/2',  ['[c,a,b]',     '[a,b,c]'], "true"),
            verify_scala_args(TmpDir, 'wam_sort_q/2',  ['[c,a,b,a]',   '[a,b,c]'], "true"),
            verify_scala_args(TmpDir, 'wam_sort_q/2',  ['[3,1,2]',     '[1,2,3]'], "true"),
            verify_scala_args(TmpDir, 'wam_sort_q/2',  ['[]',          '[]'],      "true"),
            verify_scala_args(TmpDir, 'wam_sort_q/2',  ['[c,a,b]',     '[a,c,b]'], "false"),
            % msort/2: keep duplicates
            verify_scala_args(TmpDir, 'wam_msort_q/2', ['[c,a,b,a]', '[a,a,b,c]'], "true"),
            verify_scala_args(TmpDir, 'wam_msort_q/2', ['[c,a,b,a]', '[a,b,c]'],   "false")
        )).

% --- bagof/3, setof/3 ---
test(builtin_bagof) :-
    with_scala_project(
        [user:wam_findall_dummy/1, user:wam_bagof_q/1, user:wam_bagof_empty_q/1],
        _Opts,
        TmpDir,
        (
            % wam_findall_dummy(a). wam_findall_dummy(b). → bagof returns [a,b]
            verify_scala(TmpDir, 'wam_bagof_q/1', '[a,b]', "true"),
            verify_scala(TmpDir, 'wam_bagof_q/1', '[]',    "false"),
            % bagof with no solutions FAILS (unlike findall which returns [])
            verify_scala(TmpDir, 'wam_bagof_empty_q/1', '[]', "false"),
            verify_scala(TmpDir, 'wam_bagof_empty_q/1', '_',  "false")
        )).

test(builtin_setof) :-
    with_scala_project(
        [user:wam_findall_dummy/1, user:wam_setof_q/1,
         user:wam_setof_dups_dummy/1, user:wam_setof_dups_q/1],
        _Opts,
        TmpDir,
        (
            % Two clauses, both unique values → set is [a,b]
            verify_scala(TmpDir, 'wam_setof_q/1', '[a,b]', "true"),
            verify_scala(TmpDir, 'wam_setof_q/1', '[a]',   "false"),
            % wam_setof_dups_dummy has clauses for a, b, a, c, b → set is [a,b,c]
            verify_scala(TmpDir, 'wam_setof_dups_q/1', '[a,b,c]', "true"),
            verify_scala(TmpDir, 'wam_setof_dups_q/1', '[a,a,b,b,c]', "false")
        )).

% switch_on_constant with multi-match (two clauses sharing the same
% first arg constant): the runtime must enumerate every matching
% clause via the try_me_else chain, not jump directly to one and skip
% the rest. Without the multi-match fall-through, wam_dup_first_q(a,e)
% would be unreachable. Regression test for that case.
test(builtin_switch_dup_first_arg) :-
    with_scala_project(
        [user:wam_dup_first_arg/2, user:wam_dup_first_q/2],
        [ intern_atoms([a, b, c, d, e]) ],
        TmpDir,
        (
            verify_scala_args(TmpDir, 'wam_dup_first_q/2', ['a','b'], "true"),
            verify_scala_args(TmpDir, 'wam_dup_first_q/2', ['a','e'], "true"),
            verify_scala_args(TmpDir, 'wam_dup_first_q/2', ['c','d'], "true"),
            verify_scala_args(TmpDir, 'wam_dup_first_q/2', ['a','x'], "false"),
            verify_scala_args(TmpDir, 'wam_dup_first_q/2', ['b','b'], "false")
        )).

% --- between/3 ---
%   bound mode: succeeds iff Low =< X =< High
%   unbound + findall: enumerates Low..High via the multi-solution path
test(builtin_between) :-
    with_scala_project(
        [user:wam_between_q/3, user:wam_between_collect/3],
        _Opts,
        TmpDir,
        (
            verify_scala_args(TmpDir, 'wam_between_q/3', ['1','5','3'], "true"),
            verify_scala_args(TmpDir, 'wam_between_q/3', ['1','5','1'], "true"),
            verify_scala_args(TmpDir, 'wam_between_q/3', ['1','5','5'], "true"),
            verify_scala_args(TmpDir, 'wam_between_q/3', ['1','5','0'], "false"),
            verify_scala_args(TmpDir, 'wam_between_q/3', ['1','5','6'], "false"),
            verify_scala_args(TmpDir, 'wam_between_q/3', ['1','5','-1'], "false"),
            % findall + between exercises the multi-solution path.
            verify_scala_args(TmpDir, 'wam_between_collect/3',
                              ['1','5','[1,2,3,4,5]'], "true"),
            verify_scala_args(TmpDir, 'wam_between_collect/3',
                              ['3','3','[3]'],         "true"),
            verify_scala_args(TmpDir, 'wam_between_collect/3',
                              ['1','3','[1,2,3,4]'],   "false")
        )).

% format/2 prints to stdout AND returns true. The generated main
% prints the run result on its own line, so the captured stdout
% (after normalize_space collapses newline → space) is the format
% output adjacent to the literal "true"/"false". The expected
% strings below reflect that.
test(builtin_format) :-
    with_scala_project(
        [user:wam_format_w/1, user:wam_format_combo/2, user:wam_format_dn/1],
        [ intern_atoms([hello, world]) ],
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_format_w/1', '5', "v=5true"),
            verify_scala_args(TmpDir, 'wam_format_combo/2',
                              ['hello', 'world'], "hello is world!true"),
            % ~n inserts a newline, which normalize_space turns into a
            % single space between the format output and "true".
            verify_scala(TmpDir, 'wam_format_dn/1', '42', "n=42 true")
        )).

% Multi-query stream mode: run a batch of queries in a single JVM
% invocation by pointing --queries at a file. Output is one
% true/false line per query.
test(query_stream_runner) :-
    with_scala_project(
        [user:wam_fact/1],
        _Opts,
        TmpDir,
        (
            absolute_file_name(TmpDir, AbsTmp),
            directory_file_path(AbsTmp, 'queries.txt', QueriesPath),
            setup_call_cleanup(
                open(QueriesPath, write, Stream),
                ( format(Stream, 'wam_fact/1 a~n', []),
                  format(Stream, 'wam_fact/1 b~n', []),
                  format(Stream, '# comment line — ignored~n', []),
                  format(Stream, '~n', []),                    % blank line — also ignored
                  format(Stream, 'wam_fact/1 a~n', [])
                ),
                close(Stream)),
            run_scala_query_stream(TmpDir, QueriesPath, Output),
            % normalize_space collapses each line's newline into a
            % single space, so three queries → "true false true".
            (   Output == "true false true"
            ->  true
            ;   throw(error(query_stream_mismatch(Output), _))
            )
        )).

:- end_tests(wam_scala_runtime_smoke).

% ============================================================
% Test fixture helpers
% ============================================================

%% with_scala_project(+Preds, +Opts, -TmpDir, :Goal)
%  Generates a Scala project, compiles it, runs Goal (which may call
%  verify_scala/4), then cleans up. TmpDir is bound for Goal's use.
with_scala_project(Preds, ExtraOpts0, TmpDir, Goal) :-
    (   var(ExtraOpts0) -> ExtraOpts = [] ; ExtraOpts = ExtraOpts0 ),
    unique_scala_tmp_dir('tmp_scala_smoke', TmpDir),
    BaseOpts = [ package('generated.wam_scala_smoke.core'),
                 runtime_package('generated.wam_scala_smoke.core'),
                 module_name('wam-scala-smoke') ],
    append(ExtraOpts, BaseOpts, AllOpts),
    write_wam_scala_project(Preds, AllOpts, TmpDir),
    compile_scala_project(TmpDir),
    setup_call_cleanup(
        true,
        call(Goal),
        delete_directory_and_contents(TmpDir)).

%% compile_scala_project(+ProjectDir) is det.
%  Compiles all generated .scala files with scalac into ProjectDir/classes/.
compile_scala_project(ProjectDir) :-
    absolute_file_name(ProjectDir, AbsProjectDir),
    directory_file_path(AbsProjectDir, 'classes', ClassDir),
    make_directory_path(ClassDir),
    find_scala_sources(AbsProjectDir, Sources),
    Sources \= [],
    process_create(path(scalac),
                   ['-d', ClassDir | Sources],
                   [cwd(AbsProjectDir),
                    stdout(pipe(Out)), stderr(pipe(Err)),
                    process(Pid)]),
    read_string(Out, _, _OutStr),
    read_string(Err, _, ErrStr),
    close(Out),
    close(Err),
    process_wait(Pid, exit(ExitCode)),
    (   ExitCode =:= 0
    ->  true
    ;   throw(error(scala_compile_failed(ExitCode, ErrStr), _))
    ).

%% find_scala_sources(+AbsProjectDir, -Sources)
%  Returns list of absolute paths of all .scala files under ProjectDir/src/.
find_scala_sources(AbsProjectDir, Sources) :-
    directory_file_path(AbsProjectDir, 'src', SrcDir),
    findall(F,
        ( directory_member(SrcDir, RelF,
              [extensions([scala]), recursive(true)]),
          directory_file_path(SrcDir, RelF, F)
        ),
        Sources).

%% verify_scala(+ProjectDir, +PredKey, +Arg, +Expected)
%  Single-arg form for predicates of arity 1.
verify_scala(ProjectDir, PredKey, Arg, Expected) :-
    verify_scala_args(ProjectDir, PredKey, [Arg], Expected).

%% verify_scala_args(+ProjectDir, +PredKey, +Args, +Expected)
%  Multi-arg form: Args is a list of atoms/strings to pass on the CLI.
verify_scala_args(ProjectDir, PredKey, Args, Expected) :-
    run_scala_predicate_args(ProjectDir, PredKey, Args, Actual),
    (   Actual == Expected
    ->  true
    ;   throw(error(assertion_error(PredKey, Args, Expected, Actual), _))
    ).

run_scala_predicate_args(ProjectDir, PredKey, Args, Output) :-
    absolute_file_name(ProjectDir, AbsProjectDir),
    directory_file_path(AbsProjectDir, 'classes', ClassDir),
    atom_string(PredKey, PredStr),
    maplist([A, S]>>atom_string(A, S), Args, ArgStrs),
    append(['-classpath', ClassDir,
            'generated.wam_scala_smoke.core.GeneratedProgram',
            PredStr], ArgStrs, ProcArgs),
    process_create(path(scala), ProcArgs,
                   [cwd(AbsProjectDir),
                    stdout(pipe(Out)), stderr(pipe(Err)),
                    process(Pid)]),
    read_string(Out, _, OutStr0),
    read_string(Err, _, ErrStr),
    close(Out),
    close(Err),
    process_wait(Pid, exit(ExitCode)),
    normalize_space(string(Output), OutStr0),
    (   ExitCode =:= 0
    ->  true
    ;   throw(error(scala_run_failed(ExitCode, PredKey, Args, ErrStr), _))
    ).

unique_scala_tmp_dir(Prefix, TmpDir) :-
    get_time(T),
    Stamp is floor(T * 1000),
    format(atom(TmpDir), '~w_~w', [Prefix, Stamp]).

%% run_scala_query_stream(+ProjectDir, +QueriesPath, -Output)
%  Invokes the generated program in --queries mode and captures
%  normalized stdout. Each line of the queries file is one CLI-style
%  query; the program prints true/false per query.
run_scala_query_stream(ProjectDir, QueriesPath, Output) :-
    absolute_file_name(ProjectDir, AbsTmp),
    directory_file_path(AbsTmp, 'classes', ClassDir),
    atom_string(QueriesPath, QPathStr),
    process_create(path(scala),
        ['-classpath', ClassDir,
         'generated.wam_scala_smoke.core.GeneratedProgram',
         '--queries', QPathStr],
        [cwd(AbsTmp), stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
    read_string(Out, _, OutStr),
    read_string(Err, _, _ErrStr),
    close(Out),
    close(Err),
    process_wait(Pid, exit(_)),
    normalize_space(string(Output), OutStr).

%% write_facts_csv(+Path, +Tuples) is det.
%  Writes each tuple as a comma-separated line to Path. Used by the
%  file-backed fact source test.
write_facts_csv(Path, Tuples) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        forall(member(Tuple, Tuples),
               ( atomic_list_concat(Tuple, ',', Line),
                 format(Stream, '~w~n', [Line]) )),
        close(Stream)).

% ============================================================
% Foreign handler bodies (Scala source)
% ============================================================
% Each handler is a Scala expression of type `ForeignHandler` that
% gets injected into the generated `foreignHandlers` Map literal.
% Handlers run inside the GeneratedProgram object so they can
% reference `internTable` directly.

foreign_yes_handler(Code) :-
    Code = "new ForeignHandler {\n      def apply(args: Array[WamTerm]): ForeignResult = {\n        val aId = internTable.stringToId.getOrElse(\"a\", -1)\n        args(0) match {\n          case Atom(id) if id == aId => ForeignSucceed\n          case _                     => ForeignFail\n        }\n      }\n    }".

foreign_pair_handler(Code) :-
    Code = "new ForeignHandler {\n      def apply(args: Array[WamTerm]): ForeignResult = {\n        val aId = internTable.stringToId.getOrElse(\"a\", -1)\n        val bId = internTable.stringToId.getOrElse(\"b\", -1)\n        args(0) match {\n          case Atom(id) if id == aId => ForeignBindings(Map(2 -> Atom(bId)))\n          case _                     => ForeignFail\n        }\n      }\n    }".

foreign_multi_handler(Code) :-
    Code = "new ForeignHandler {\n      def apply(args: Array[WamTerm]): ForeignResult = {\n        val aId = internTable.stringToId.getOrElse(\"a\", -1)\n        val bId = internTable.stringToId.getOrElse(\"b\", -1)\n        args(0) match {\n          case Atom(id) if id == aId =>\n            ForeignMulti(Seq(\n              Map(2 -> Atom(aId)),\n              Map(2 -> Atom(bId))\n            ))\n          case _ => ForeignFail\n        }\n      }\n    }".
