:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_wam_rust_bugfix_batch_1.pl
%
% SWI-Prolog comparison harness for the Rust WAM bug-fix batch 1
% (docs/reports/wam_rust_bugfix_batch_1.md, ledger rows D140 onward). Each
% program is a small reproduction of one defect; the item tag names the
% report section:
%
%   B1  names containing `\` or `"` emitted into Rust string literals
%   B2  ignore/1
%   B3  aggregate_all/3 inside a findall/3 goal (and other inner positions)
%   B4  dynamic rule bodies resume on redo instead of re-running
%   B5  lowered ITE: a failing call in the condition leaves its frame behind
%   B6  `_XT<N>` aggregate template scratch registers (slot 0 = A1)
%
% The harness is the one of tests/test_wam_rust_baseline_fixes.pl: one source
% of truth (bb_clause/2), asserted into user: for SWI and handed to
% write_wam_rust_project/3; a failure-driven driver
%     bb_drv_<name> :- Setup, ( Query, write(Out), nl, fail ; true ).
% is compiled with the program and run by its WAM label in
% emit_mode(interpreter) and emit_mode(functions); its stdout and success
% flag must equal SWI's. When emit_mode(functions) lowered the query
% predicate, its Rust function is also called directly and compared with
% SWI's first solution (the lowered tier is first-solution by contract), on
% the default build and on the non-interned
% `--no-default-features --features decorate_sort` build.
%
%   LANG=C.UTF-8 swipl -q -g run_tests -t halt tests/test_wam_rust_bugfix_batch_1.pl
%
% (A UTF-8 locale is needed: the generator reads templates that contain
% non-ASCII text.)

:- module(test_wam_rust_bugfix_batch_1,
          [test_wam_rust_bugfix_batch_1/0]).

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(filesex), [make_directory_path/1,
                                 delete_directory_and_contents/1]).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module('../src/unifyweaver/targets/wam_rust_target',
              [write_wam_rust_project/3]).

% ---------------------------------------------------------------------
% Programs. bb_program(Name, Item, Query, Out, Setup)
%   Query/Out : the query whose solutions are printed (write(Out)).
%   Setup     : goal run once by the driver before the query loop.
% Clauses are listed under one program name each (bb_clause/2) and all of
% them are compiled into every crate.
% ---------------------------------------------------------------------

:- discontiguous bb_program/5, bb_clause/2.
:- dynamic bb_dynamic/1.
:- discontiguous bb_dynamic/1.

% --- B1: names with `\` or `"` in Rust string literals ----------------
%
% A `\+` term built in a clause body (put_structure \+/1), meta-called.
bb_program(esc_naf_no,  'B1', esc_naf(a, R), R, true).
bb_program(esc_naf_yes, 'B1', esc_naf(b, R), R, true).
bb_clause(esc_naf_no, (esc_naf(X, R) :- G = (\+ esc_q(X)), ( call(G) -> R = yes ; R = no ))).
bb_clause(esc_naf_no, esc_q(a)).
% `\=` built in a body and meta-called, after a choice point.
bb_program(esc_neq, 'B1', esc_neq(X, R), [X, R], true).
bb_clause(esc_neq, (esc_neq(X, R) :- member(X, [a, b]), G = (X \= a), ( call(G) -> R = ne ; R = eq ))).
% Functor names with a backslash or a double quote in clause heads
% (get_structure), read back with functor/3 (write/1 prints operators in
% canonical form in this runtime, so the terms themselves are not printed).
bb_program(esc_functor, 'B1', esc_functor(N), N, true).
bb_clause(esc_functor, (esc_functor(N) :- esc_ops(T), functor(T, N, _))).
bb_clause(esc_functor, esc_ops(\+ a)).
bb_clause(esc_functor, esc_ops(a \= b)).
bb_clause(esc_functor, esc_ops(1 =\= 2)).
bb_clause(esc_functor, esc_ops(a \== b)).
bb_clause(esc_functor, esc_ops('q"r'(1))).
% Lowered code: put_structure and get_structure of `\+/1` in deterministic
% predicates, and atom constants containing `\` and `"`.
bb_program(esc_head,  'B1', esc_mk(R), R, true).
bb_clause(esc_head, (esc_mk(R) :- esc_unwrap(\+ k, R))).
bb_clause(esc_head, esc_unwrap(\+ X, X)).
bb_program(esc_atoms_then, 'B1', esc_atoms('a\\b', R), R, true).
bb_program(esc_atoms_else, 'B1', esc_atoms(x, R), R, true).
bb_clause(esc_atoms_then, (esc_atoms(X, R) :- ( X == 'a\\b' -> R = 'say "hi"' ; R = 'c\\d' ))).
bb_program(esc_quote_const, 'B1', esc_qc(R), R, true).
bb_clause(esc_quote_const, (esc_qc(R) :- esc_qv(V), R = V)).
bb_clause(esc_quote_const, esc_qv('x"y')).
% A predicate whose name contains a backslash (its label, its dispatch key).
bb_program(esc_pred_name, 'B1', esc_callp(R), R, true).
bb_clause(esc_pred_name, (esc_callp(R) :- 'esc\\pred'(R))).
bb_clause(esc_pred_name, 'esc\\pred'(ok)).

% --- B2: ignore/1 ----------------------------------------------------
%
% A failing goal: ignore/1 succeeds.
bb_program(ign_fail, 'B2', ign_fail(R), R, true).
bb_clause(ign_fail, (ign_fail(R) :- ignore(ign_none), R = ok)).
bb_clause(ign_fail, (ign_none :- fail)).
% First solution only.
bb_program(ign_first, 'B2', ign_first(X), X, true).
bb_clause(ign_first, (ign_first(X) :- ignore(ign_d(X)))).
bb_clause(ign_first, ign_d(1)).
bb_clause(ign_first, ign_d(2)).
% The bindings of a goal that fails are undone.
bb_program(ign_undo, 'B2', ign_undo(R), R, true).
bb_clause(ign_undo, (ign_undo(R) :- ignore(ign_bind_fail(X)), ( var(X) -> R = unbound ; R = bound(X) ))).
bb_clause(ign_undo, (ign_bind_fail(X) :- X = 1, fail)).
% The goal is an opaque cut scope: the caller's choice points survive.
bb_program(ign_cut, 'B2', ign_cut(X), X, true).
bb_clause(ign_cut, (ign_cut(X) :- ign_d(X), ignore(!))).
% Inside a findall/3 goal, meta-called, and as the last goal of a clause.
bb_program(ign_findall, 'B2', ign_findall(L), L, true).
bb_clause(ign_findall, (ign_findall(L) :- findall(p(X, Y), (ign_d(X), ignore(ign_e(X, Y)), ( var(Y) -> Y = none ; true )), L))).
bb_clause(ign_findall, ign_e(2, two)).
bb_program(ign_meta, 'B2', ign_meta(R), R, true).
bb_clause(ign_meta, (ign_meta(R) :- G = ignore(ign_none), call(G), R = ok)).
bb_program(ign_tail, 'B2', ign_tail(X), X, true).
bb_clause(ign_tail, (ign_tail(X) :- X = t, ignore(ign_d(_)))).
% Side effects of the goal happen once; a ball thrown inside propagates.
bb_program(ign_write, 'B2', ign_write, done, true).
bb_clause(ign_write, (ign_write :- ignore((write(a), fail)))).
bb_program(ign_throw, 'B2', ign_throw(R), R, true).
bb_clause(ign_throw, (ign_throw(R) :- catch(ignore(throw(oops)), E, R = caught(E)))).

% --- B3: aggregate_all/3 in inner goal positions ---------------------
%
% The R0b case (nested_agg_inline): aggregate_all/3 inside a findall/3 goal.
bb_program(agg_in_findall, 'B3', agg_in_findall(L), L, true).
bb_clause(agg_in_findall, (agg_in_findall(L) :- findall(c(K, C), (member(K, [p, q, r]), aggregate_all(count, agg_e(K, _), C)), L))).
bb_clause(agg_in_findall, agg_e(p, 2)).
bb_clause(agg_in_findall, agg_e(p, 1)).
bb_clause(agg_in_findall, agg_e(p, 2)).
bb_clause(agg_in_findall, agg_e(q, 3)).
bb_clause(agg_in_findall, (agg_none :- fail)).
bb_clause(agg_in_findall, (agg_none2(_) :- fail)).
% Every template.
bb_program(agg_templates, 'B3', agg_templates(L), L, true).
bb_clause(agg_templates, (agg_templates(L) :- findall(r(K, S, Mx, Mn, B, St), (member(K, [p, q]), aggregate_all(sum(V), agg_e(K, V), S), aggregate_all(max(V), agg_e(K, V), Mx), aggregate_all(min(V), agg_e(K, V), Mn), aggregate_all(bag(V), agg_e(K, V), B), aggregate_all(set(V), agg_e(K, V), St)), L))).
bb_program(agg_witness, 'B3', agg_witness(L), L, true).
bb_clause(agg_witness, (agg_witness(L) :- findall(W-M, (member(K, [p, q]), aggregate_all(max(V, w(K, V)), agg_e(K, V), W), aggregate_all(min(V, w(K, V)), agg_e(K, V), M)), L0), agg_pairs(L0, L))).
bb_clause(agg_witness, agg_pairs([], [])).
bb_clause(agg_witness, (agg_pairs([A-B|T], [A, B|R]) :- agg_pairs(T, R))).
% sum/max/min evaluate their expression.
bb_program(agg_expr, 'B3', agg_expr(L), L, true).
bb_clause(agg_expr, (agg_expr(L) :- findall(x(K, S, Mx), (member(K, [p, q]), aggregate_all(sum(V * 2), agg_e(K, V), S), aggregate_all(max(V + 1), agg_e(K, V), Mx)), L))).
% Empty goals: count 0, sum 0, bag [], set []; max of nothing fails.
bb_program(agg_empty, 'B3', agg_empty(L), L, true).
bb_clause(agg_empty, (agg_empty(L) :- findall(e(C, S, B, St), (member(_, [x]), aggregate_all(count, agg_none, C), aggregate_all(sum(V), agg_none2(V), S), aggregate_all(bag(V), agg_none2(V), B), aggregate_all(set(V), agg_none2(V), St)), L))).
bb_program(agg_empty_max, 'B3', agg_empty_max(L), L, true).
bb_clause(agg_empty_max, (agg_empty_max(L) :- findall(M, (member(_, [x]), aggregate_all(max(V), agg_none2(V), M)), L))).
% The same templates as clause-body goals (the inline begin/end_aggregate
% path): bag/set of nothing is [], max/min of nothing fails.
bb_program(agg_inline_empty, 'B3', agg_inline_empty(B, S), [B, S], true).
bb_clause(agg_inline_empty, (agg_inline_empty(B, S) :- aggregate_all(bag(V), agg_none2(V), B), aggregate_all(set(V), agg_none2(V), S))).
bb_program(agg_inline_max_empty, 'B3', agg_inline_max_empty(M), M, true).
bb_clause(agg_inline_max_empty, (agg_inline_max_empty(M) :- aggregate_all(max(V), agg_none2(V), M))).
bb_program(agg_inline_min_empty, 'B3', agg_inline_min_empty(M), M, true).
bb_clause(agg_inline_min_empty, (agg_inline_min_empty(M) :- aggregate_all(min(V), agg_none2(V), M))).
% Control constructs in the goal: a nondeterministic left conjunct, a cut,
% a disjunction, \+, an if-then-else.
bb_program(agg_control, 'B3', agg_control(L), L, true).
bb_clause(agg_control, (agg_control(L) :- findall(r(A, B, C, D, E), (member(_, [x]),
     aggregate_all(count, (agg_e(_, V), V > 1), A),
     aggregate_all(bag(V), (agg_e(p, V), !), B),
     aggregate_all(bag(V), (agg_e(q, V) ; V = z), C),
     aggregate_all(bag(K), (member(K, [p, q, r]), \+ agg_e(K, _)), D),
     aggregate_all(bag(T), (member(K, [p, r]), ( agg_e(K, _) -> T = y(K) ; T = n(K) )), E)), L))).
% In an if-then-else condition and branch, and meta-called.
bb_program(agg_ite, 'B3', agg_ite(K, R), [K, R], true).
bb_clause(agg_ite, (agg_ite(K, R) :- member(K, [p, r]), ( aggregate_all(count, agg_e(K, _), C), C > 0 -> R = some(C) ; aggregate_all(count, agg_e(_, _), N), R = none(N) ))).
bb_program(agg_meta, 'B3', agg_meta(C), C, true).
bb_clause(agg_meta, (agg_meta(C) :- G = aggregate_all(count, agg_e(_, _), C), call(G))).
bb_program(findall_meta, 'B3', findall_meta(L), L, true).
bb_clause(findall_meta, (findall_meta(L) :- G = findall(V, agg_e(p, V), L), call(G))).
% Side effects of the goal persist; a ball thrown inside propagates; every
% solution gets fresh variables.
bb_program(agg_assert, 'B3', agg_assert(L), L, true).
bb_clause(agg_assert, (agg_assert(L) :- findall(C, (member(_, [x]), aggregate_all(count, (member(X, [a, b]), assertz(agg_log(X))), C)), Cs), findall(X, agg_log(X), Xs), L = r(Cs, Xs))).
bb_program(agg_throw, 'B3', agg_throw(R), R, true).
bb_clause(agg_throw, (agg_throw(R) :- catch(findall(C, (member(_, [x]), aggregate_all(count, throw(boom), C)), _), E, true), R = caught(E))).
bb_program(agg_fresh, 'B3', agg_fresh(R), R, true).
bb_clause(agg_fresh, (agg_fresh(R) :- findall(B, (member(_, [x]), aggregate_all(bag(f(_)), member(_, [1, 2]), B)), [[f(P), f(Q)]]), ( P == Q -> R = shared ; R = distinct ))).

% --- B4: dynamic rule bodies resume on redo ---------------------------
%
% A side effect in a dynamic rule body happens once per solution (the R0b
% case: SWI prints `ab`; the body used to be re-run from its start on every
% redo, printing `aabab`).
bb_program(dyn_write, 'B4', dyn_w(X), X,
           assertz((dyn_w(Y) :- member(Y, [a, b]), write(Y)))).
% Reading lines through a dynamic rule loses no input (R0b: Rust read l1, l3).
bb_program(dyn_read, 'B4',
           ( stream_open(F, S), ( dyn_rl(S, L), write(L), nl, fail ; true ), read_line(S, A), stream_close(S) ),
           after(A),
           ( bb_lines_file(F), assertz((dyn_rl(S0, L0) :- member(_, [1, 2]), read_line(S0, L0))) )).
bb_clause(dyn_read, bb_lines_file('output/rust_wam_bugfix_batch_1_lines.txt')).
% A compiled nondeterministic predicate in the body, alone and in a
% conjunction.
bb_program(dyn_compiled, 'B4', dyn_c(X), X, assertz((dyn_c(Y) :- dyn_p(Y)))).
bb_clause(dyn_compiled, dyn_p(1)).
bb_clause(dyn_compiled, dyn_p(2)).
bb_clause(dyn_compiled, dyn_p(3)).
bb_program(dyn_conj, 'B4', dyn_cj(X, Y), [X, Y],
           assertz((dyn_cj(A, B) :- dyn_p(A), dyn_p(B), A < B))).
% A cut in the body prunes the body's alternatives to its left and the
% predicate's later clauses.
bb_program(dyn_cut, 'B4', dyn_k(X), X,
           ( assertz((dyn_k(Y) :- member(Y, [a, b, c]), Y \== a, !)), assertz(dyn_k(z)) )).
bb_program(dyn_cut_fail, 'B4', dyn_kf(X), X,
           ( assertz((dyn_kf(Y) :- member(Y, [a, b]), !, fail)), assertz(dyn_kf(z)) )).
% A recursive dynamic predicate.
bb_program(dyn_rec, 'B4', dyn_path(a, X), X,
           ( assertz(dyn_edge(a, b)), assertz(dyn_edge(b, c)), assertz(dyn_edge(c, d)),
             assertz((dyn_path(P, Q) :- dyn_edge(P, Q))),
             assertz((dyn_path(P, Q) :- dyn_edge(P, M), dyn_path(M, Q))) )).
% If-then-else, disjunction and \+ in the body.
bb_program(dyn_ctl, 'B4', dyn_ctl(X, R), [X, R],
           assertz((dyn_ctl(Y, T) :- ( Y = 1 ; Y = 2 ; Y = 3 ), \+ Y = 2, ( Y > 1 -> T = big ; T = small )))).
% A ball thrown in the body propagates.
bb_program(dyn_throw, 'B4', catch(dyn_t(_), E, true), E,
           assertz((dyn_t(Y) :- member(Y, [1, 2]), Y > 1, throw(dboom(Y))))).
% All solutions under findall/3, with a side effect per solution.
bb_program(dyn_findall, 'B4', findall(X, dyn_m(X), L), L,
           assertz((dyn_m(Y) :- member(Y, [1, 2, 3]), write(Y)))).
% The body sees the database the caller's continuation changed between
% solutions; asserts in the body persist.
bb_program(dyn_db, 'B4', ( dyn_q(X), assertz(dyn_seen(2)) ), X,
           ( assertz(dyn_seen(0)), assertz((dyn_q(Y) :- member(Y, [1, 2, 3]), \+ dyn_seen(Y), assertz(dyn_log(Y)))) )).
bb_program(dyn_db_log, 'B4', findall(Y, dyn_log(Y), L), L,
           ( assertz(dyn_seen(0)), assertz((dyn_q(Y) :- member(Y, [1, 2, 3]), \+ dyn_seen(Y), assertz(dyn_log(Y)))),
             ( dyn_q(Z), assertz(dyn_seen(2)), Z > 5 ; true ) )).

% --- B5: lowered ITE condition with a failing call --------------------
%
% The condition calls a predicate that allocates an environment and fails;
% the else branch reads the caller's permanent variables (the R0b case
% naf_temp: SWI got(a,b), lowered entry no solution).
bb_program(lite_naf, 'B5', lite_naf(a, b, R), R, true).
bb_clause(lite_naf, (lite_naf(X, Y, R) :- \+ lite_clob(X), R = got(X, Y))).
bb_clause(lite_naf, (lite_clob(X) :- lite_w(k1, k2, X), lite_w(X, k3, k4), fail)).
bb_clause(lite_naf, lite_w(_, _, _)).
bb_program(lite_ite, 'B5', lite_ite(a, b, R), R, true).
bb_clause(lite_ite, (lite_ite(X, Y, R) :- ( lite_deep(X, Z) -> R = yes(Z) ; R = no(X, Y) ))).
bb_clause(lite_ite, (lite_deep(X, Z) :- lite_w(X, Z, k5), lite_v(Z, W), W == q)).
bb_clause(lite_ite, lite_v(_, p)).
% A callee that succeeds (and leaves its choice points) before a later goal
% of the condition fails; then a nested if-then-else whose condition fails
% inside the else branch.
bb_program(lite_part, 'B5', lite_part(a, R), R, true).
bb_clause(lite_part, (lite_part(X, R) :- ( lite_two(X, Z), Z > 1 -> R = big(Z) ; R = small(X) ))).
bb_clause(lite_part, (lite_two(X, Z) :- lite_w(X, Z, k6), Z = 1)).
bb_clause(lite_part, lite_two(b, 0)).
bb_program(lite_nested, 'B5', lite_nested(a, b, R), R, true).
bb_clause(lite_nested, (lite_nested(X, Y, R) :- ( lite_clob(X) -> R = one ; ( lite_deep(Y, _) -> R = two ; R = three(X, Y) ) ))).

% --- B6: `_XT<N>` template scratch registers --------------------------
%
% A findall/bagof/setof/aggregate_all template with a compound argument is
% built with `set_variable _XT<N>` / `put_structure F, _XT<N>`; the runtime
% decoded `_XT<N>` to flat slot 0, the slot of A1, which also holds the
% template's root, so the aggregate collected the last sub-compound.
bb_program(xt_nested, 'B6', xt_nested(L), L, true).
bb_clause(xt_nested, (xt_nested(L) :- findall(f(g(X)), member(X, [1, 2]), L))).
bb_program(xt_two, 'B6', xt_two(L), L, true).
bb_clause(xt_two, (xt_two(L) :- findall(f(g(X), h(Y)), member(p(X, Y), [p(1, a), p(2, b)]), L))).
bb_program(xt_deep, 'B6', xt_deep(L), L, true).
bb_clause(xt_deep, (xt_deep(L) :- findall(f(g(h(X)), k), member(X, [1, 2]), L))).
bb_program(xt_list, 'B6', xt_list(L), L, true).
bb_clause(xt_list, (xt_list(L) :- findall([X, Y], member(p(X, Y), [p(1, a), p(2, b)]), L))).
bb_program(xt_arg_live, 'B6', xt_arg_live(z, L), L, true).
bb_clause(xt_arg_live, (xt_arg_live(A, L) :- findall(f(g(X), A), member(X, [1, 2]), L))).
bb_program(xt_bag, 'B6', xt_bag(L, S), [L, S], true).
bb_clause(xt_bag, (xt_bag(L, S) :- aggregate_all(bag(f(g(X))), member(X, [2, 1]), L), aggregate_all(set(f(g(X))), member(X, [2, 1, 2]), S))).
bb_program(xt_setof, 'B6', xt_setof(L, B), [L, B], true).
bb_clause(xt_setof, (xt_setof(L, B) :- setof(f(g(X)), member(X, [2, 1, 2]), L), bagof(f(g(X)), member(X, [2, 1]), B))).
bb_program(xt_inner, 'B6', xt_inner(L), L, true).
bb_clause(xt_inner, (xt_inner(L) :- findall(f(g(X), Ys), (member(X, [1, 2]), findall(h(k(Y)), member(Y, [X, 3]), Ys)), L))).
bb_program(xt_bound_var, 'B6', xt_bound_var(L), L, true).
bb_clause(xt_bound_var, (xt_bound_var(L) :- X = a, findall(X, member(X, [a, b]), L))).
bb_program(xt_count, 'B6', xt_count(N, L), [N, L], true).
bb_clause(xt_count, (xt_count(N, L) :- aggregate_all(count, member(_, [a, b]), N), findall(f(g(N)), member(_, [x]), L))).

bb_dynamic(dyn_w/1).
bb_dynamic(dyn_rl/2).
bb_dynamic(dyn_c/1).
bb_dynamic(dyn_cj/2).
bb_dynamic(dyn_k/1).
bb_dynamic(dyn_kf/1).
bb_dynamic(dyn_edge/2).
bb_dynamic(dyn_path/2).
bb_dynamic(dyn_ctl/2).
bb_dynamic(dyn_t/1).
bb_dynamic(dyn_m/1).
bb_dynamic(dyn_q/1).
bb_dynamic(dyn_seen/1).
bb_dynamic(dyn_log/1).

%% bb_swi_shim(?Clause)
%  SWI-only definitions of the Rust runtime's line-reader builtins (as in
%  tests/test_wam_rust_stream_swi_parity.pl); asserted for the SWI oracle,
%  never compiled for Rust.
bb_swi_shim((stream_open(P, S) :- open(P, read, S))).
bb_swi_shim((read_line(S, L) :- read_line_to_string(S, L0), ( L0 == end_of_file -> L = end_of_file ; atom_string(L, L0) ))).
bb_swi_shim((stream_close(S) :- close(S))).

%% bb_dynamic(?PI)
%  Predicates the drivers create at run time with assertz/1. Declared dynamic
%  in user: for SWI; not compiled for Rust (the runtime's dynamic database
%  serves them).

bb_dynamic(agg_log/1).

%% bb_lowered_out_of_scope(?Name, ?Why)
%  Programs whose DIRECT lowered entry is wrong for a reason outside this
%  batch. Their interpreted entries in emit_mode(functions) are still
%  checked. A lowered `call`/`execute` of a runtime builtin that has no label
%  (call/N here) returns false instead of running it (noted in
%  tests/test_wam_rust_baseline_fixes.pl).
bb_lowered_out_of_scope(esc_naf_no,  lowered_call_of_builtin_fails).
bb_lowered_out_of_scope(esc_naf_yes, lowered_call_of_builtin_fails).
bb_lowered_out_of_scope(esc_neq,     lowered_call_of_builtin_fails).
% ignore/1 is such a builtin: every B2 query predicate the emitter lowers
% reaches it through a lowered `call ignore/1`.
bb_lowered_out_of_scope(Name, lowered_call_of_builtin_fails) :-
    bb_program(Name, 'B2', _, _, _).
% The B3 query predicates the emitter lowers contain a findall/3 or an
% aggregate_all/3 (the lowered emitter drops begin_aggregate/end_aggregate,
% noted by the R-1 harness) or reach aggregate_all/3 through a lowered call.
bb_lowered_out_of_scope(Name, lowered_aggregates_unsupported) :-
    bb_program(Name, 'B3', _, _, _).
% A dynamic predicate has no label either: a lowered call of one fails.
bb_lowered_out_of_scope(Name, lowered_call_of_unlabelled_predicate_fails) :-
    bb_program(Name, 'B4', _, _, _).
% Every B6 program is an aggregate (dropped by the lowered emitter).
bb_lowered_out_of_scope(Name, lowered_aggregates_unsupported) :-
    bb_program(Name, 'B6', _, _, _).

%% bb_known_divergence(?Mode, ?Name, ?Why)
%  Programs whose interpreted entry still differs from SWI-Prolog for a
%  reason outside this batch. Not compared with SWI; test
%  known_divergences_still_present asserts that each still differs, so the
%  fix that removes one must delete its line.
%
%  The shared compiler (wam_target.pl compile_aggregate_all/6 and
%  init_template_leaf_vars/4) initialises every template variable with
%  `put_variable R, R` before begin_aggregate, also a variable that is
%  already bound outside the aggregate (a head argument, the result of an
%  earlier goal), which loses its binding: findall(f(g(X), A), ..., L) with
%  A = z collects f(g(1), _), and findall(X, member(X, [a, b]), L) with X = a
%  gives [a, b] (SWI [a]). Found while testing B6; it is not the `_XT<N>`
%  aliasing and needs a change to the shared compiler.
bb_known_divergence(interpreter, xt_arg_live, template_var_bound_outside_is_reinitialised).
bb_known_divergence(functions,   xt_arg_live, template_var_bound_outside_is_reinitialised).
bb_known_divergence(interpreter, xt_bound_var, template_var_bound_outside_is_reinitialised).
bb_known_divergence(functions,   xt_bound_var, template_var_bound_outside_is_reinitialised).
bb_known_divergence(interpreter, xt_count,    template_var_bound_outside_is_reinitialised).
bb_known_divergence(functions,   xt_count,    template_var_bound_outside_is_reinitialised).

%% bb_lowered_required(?Name)
%  Programs whose query predicate MUST be lowered in functions mode, so the
%  lowered check cannot silently become a skip.
bb_lowered_required(esc_head).
bb_lowered_required(esc_atoms_then).
bb_lowered_required(esc_quote_const).
bb_lowered_required(esc_pred_name).
bb_lowered_required(lite_naf).
bb_lowered_required(lite_ite).
bb_lowered_required(lite_part).
bb_lowered_required(lite_nested).

% ---------------------------------------------------------------------
% Installation
% ---------------------------------------------------------------------

bb_driver_name(Name, D) :- atom_concat(bb_drv_, Name, D).

bb_clause_head(Clause, F/A) :-
    ( Clause = (H :- _) -> true ; H = Clause ),
    functor(H, F, A).

bb_program_pis(PIs) :-
    findall(PI, (bb_clause(_, C), bb_clause_head(C, PI)), PIs0),
    findall(D/0, (bb_program(N, _, _, _, _), bb_driver_name(N, D)), PIs1),
    append(PIs0, PIs1, PIs2),
    sort(PIs2, PIs).

install_bb_programs :-
    bb_program_pis(PIs),
    forall(member(F/A, PIs),
           ( functor(H, F, A), catch(retractall(user:H), _, true) )),
    forall(bb_dynamic(F/A),
           ( functor(H, F, A),
             catch(retractall(user:H), _, true),
             dynamic(user:F/A) )),
    forall(bb_clause(_, C), assertz(user:C)),
    forall(bb_swi_shim((H :- _)),
           ( functor(H, F, A), functor(G, F, A), retractall(user:G) )),
    forall(bb_swi_shim(C), assertz(user:C)),
    write_bb_lines_file,
    forall(bb_program(N, _, Q, Out, Setup),
           ( bb_driver_name(N, D),
             assertz(user:(D :- Setup, ( Q, write(Out), nl, fail ; true ))) )).

%% write_bb_lines_file
%  The input file of dyn_read (both SWI and the Rust binaries read it).
write_bb_lines_file :-
    user:bb_lines_file(F),
    file_directory_name(F, Dir),
    make_directory_path(Dir),
    setup_call_cleanup(open(F, write, S),
                       format(S, "l1~nl2~nl3~nl4~n", []),
                       close(S)).

bb_rust_preds(Preds) :-
    bb_program_pis(PIs),
    findall(user:PI, member(PI, PIs), Preds).

% ---------------------------------------------------------------------
% SWI oracle
% ---------------------------------------------------------------------

bb_reset_dynamic :-
    forall(bb_dynamic(F/A),
           ( functor(H, F, A), catch(retractall(user:H), _, true) )).

%% swi_all(+Name, -Out, -Ok)
swi_all(Name, Out, Ok) :-
    bb_reset_dynamic,
    bb_driver_name(Name, D),
    with_output_to(string(Raw),
                   ( catch(user:D, _, fail) -> Ok = true ; Ok = false )),
    normalize_output(Raw, Out).

%% swi_once(+Name, -Out)
swi_once(Name, Out) :-
    bb_reset_dynamic,
    bb_program(Name, _, Q, O, Setup),
    with_output_to(string(Raw),
                   catch(( call(user:Setup), call(user:Q) -> write(O), nl ; true ),
                         _, true)),
    normalize_output(Raw, Out).

normalize_output(Raw, Out) :-
    split_string(Raw, "\n", "\r", Lines0),
    exclude(==(""), Lines0, Lines),
    atomic_list_concat(Lines, "\n", Joined),
    split_string(Joined, " \t", "", Parts),
    atomic_list_concat(Parts, '', Out).

% ---------------------------------------------------------------------
% Rust builds
% ---------------------------------------------------------------------

cargo_available :-
    catch(
        ( process_create(path(cargo), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

%% bb_mode(?Mode, ?EmitOpts, ?CargoFeatureArgs, ?Dir)
%  functions_nodef shares the functions crate (same generated source) and
%  builds it into a second target dir without default features.
bb_mode(interpreter,     [emit_mode(interpreter)], '',
        'output/rust_wam_bugfix_batch_1_interpreter').
bb_mode(functions,       [emit_mode(functions)],   '',
        'output/rust_wam_bugfix_batch_1_functions').
bb_mode(functions_nodef, [emit_mode(functions)],
        '--no-default-features --features decorate_sort',
        'output/rust_wam_bugfix_batch_1_functions').

bb_target_dir(functions_nodef, 'target_nodef') :- !.
bb_target_dir(_, 'target').

:- dynamic bb_built/1.
:- dynamic bb_generated/1.

bb_generate(Dir, EmitOpts) :-
    (   bb_generated(Dir)
    ->  true
    ;   ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
        make_directory_path(Dir),
        install_bb_programs,
        bb_rust_preds(Preds),
        append([module_name('bbprobe'), wam_fallback(true)], EmitOpts, Opts),
        write_wam_rust_project(Preds, Opts, Dir),
        write_bb_driver(Dir),
        assertz(bb_generated(Dir))
    ).

%% bb_ensure_built(+Mode)
bb_ensure_built(Mode) :-
    (   bb_built(Mode)
    ->  true
    ;   bb_mode(Mode, EmitOpts, Feats, Dir),
        bb_generate(Dir, EmitOpts),
        bb_target_dir(Mode, TD),
        format(atom(Cmd),
               'cd ~w && CARGO_TARGET_DIR=~w cargo build ~w --bin bb_probe 2>&1',
               [Dir, TD, Feats]),
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(std), process(Pid)]),
        read_string(Out, _, OutStr), close(Out),
        process_wait(Pid, Status),
        (   Status == exit(0)
        ->  assertz(bb_built(Mode))
        ;   format(user_error, "~n[cargo build output, ~w]~n~w~n", [Mode, OutStr]),
            throw(bb_build_failed(Mode, Status))
        )
    ).

bb_bin(Mode, Bin) :-
    bb_mode(Mode, _, _, Dir),
    bb_target_dir(Mode, TD),
    format(atom(Bin), '~w/~w/debug/bb_probe', [Dir, TD]).

%% lowered_entry(+Dir, +Name, -Fn)
%  Fn is the lowered function's Rust name when the functions-mode emitter
%  lowered the query predicate of program Name.
lowered_entry(Dir, Name, Fn) :-
    bb_program(Name, _, Q, _, _),
    functor(Q, F, A),
    atomic_list_concat([Dir, '/src/lib.rs'], LibRs),
    exists_file(LibRs),
    read_file_to_string(LibRs, S, []),
    format(atom(Key), '~w_~w', [F, A]),
    format(atom(Fn), 'lowered_~w', [Key]),
    format(atom(Sym), 'pub fn ~w(', [Fn]),
    sub_string(S, _, _, _, Sym).

write_bb_driver(Dir) :-
    atomic_list_concat([Dir, '/src/bin/bb_probe'], BinDir),
    make_directory_path(BinDir),
    atomic_list_concat([BinDir, '/main.rs'], Path),
    bb_driver_source(Head, Tail),
    findall(Arm,
            ( bb_program(N, _, _, _, _),
              lowered_entry(Dir, N, Fn),
              format(atom(Arm),
                     '            "~w" => run_lowered(&mut vm, bbprobe::~w, &rest),~n',
                     [Fn, Fn]) ),
            Arms0),
    sort(Arms0, Arms),
    atomic_list_concat(Arms, ArmsText),
    setup_call_cleanup(open(Path, write, S),
                       format(S, '~w~w~w', [Head, ArmsText, Tail]),
                       close(S)).

bb_driver_source(
"// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
//
// bb_probe: run a 0-arity driver predicate by its WAM label (program output
// comes from the compiled write/1), or call a LOWERED function directly with
// the query arguments (`--lowered fn arg..`, `_` = the output variable).
#![allow(unused_imports, dead_code, unused_mut, unused_variables)]
use bbprobe::state::WamState;
use bbprobe::value::Value;
use bbprobe::{setup_foreign_predicates, shared_wam_program};

const OUT: &str = \"_bb_out\";

fn parse_arg(s: &str) -> Value {
    if s == \"_\" {
        Value::Unbound(OUT.into())
    } else if let Ok(n) = s.parse::<i64>() {
        Value::Integer(n)
    } else if s == \"[]\" {
        Value::Atom(\"[]\".into())
    } else {
        Value::Atom(s.into())
    }
}

fn run_lowered(vm: &mut WamState, f: fn(&mut WamState) -> bool, args: &[String]) {
    vm.reset_query();
    vm.cp = 0;
    for (i, a) in args.iter().enumerate() {
        vm.set_reg(&format!(\"A{}\", i + 1), parse_arg(a));
    }
    let out_reg = args.iter().position(|a| a == \"_\").map(|i| format!(\"A{}\", i + 1));
    if f(vm) {
        match out_reg {
            None => println!(\"yes\"),
            Some(r) => {
                let bound = vm.deref_heap(&Value::Unbound(OUT.into()));
                let out = if bound.is_unbound() {
                    vm.get_reg(&r).map(|v| vm.deref_heap(&v)).unwrap_or(bound)
                } else {
                    bound
                };
                println!(\"{}\", out);
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (code, labels) = shared_wam_program();
    let mut vm = WamState::new(code, labels);
    setup_foreign_predicates(&mut vm);

    if args.get(1).map(|s| s.as_str()) == Some(\"--lowered\") {
        let key = args.get(2).cloned().unwrap_or_default();
        let rest: Vec<String> = args.iter().skip(3).cloned().collect();
        match key.as_str() {
",
"            other => { eprintln!(\"NO_LOWERED {}\", other); std::process::exit(3); }
        }
        return;
    }

    let key = match args.get(1) {
        Some(k) => k.clone(),
        None => { eprintln!(\"usage: bb_probe <pred/arity>\"); std::process::exit(2); }
    };
    let target = match vm.labels.get(&key) {
        Some(pc) => *pc,
        None => { eprintln!(\"NO_LABEL {}\", key); std::process::exit(3); }
    };
    vm.reset_query();
    vm.cp = 0;
    vm.pc = target;
    let ok = vm.run();
    eprintln!(\"run={}\", ok);
}
").

%% bb_run(+Mode, +Args, -Out, -Err)
bb_run(Mode, Args, Out, Err) :-
    bb_run(Mode, Args, [], Out, Err).

%% bb_run(+Mode, +Args, +StdinText, -Out, -Err)
%  Each run is capped at 120 s (a program that loops on unfixed code reports
%  a crash instead of hanging the suite).
bb_run(Mode, Args, Stdin, Out, Err) :-
    bb_bin(Mode, Bin),
    process_create(path(timeout), ['120', Bin|Args],
                   [stdin(pipe(I)), stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    ( Stdin == [] -> true ; format(I, '~w', [Stdin]) ),
    close(I),
    read_string(O, _, S1), read_string(E, _, S2),
    close(O), close(E),
    process_wait(Pid, _),
    normalize_output(S1, Out),
    Err = S2.

%% rust_all(+Mode, +Name, -Out, -Ok)
rust_all(Mode, Name, Out, Ok) :-
    bb_driver_name(Name, D),
    format(atom(Key), '~w/0', [D]),
    bb_run(Mode, [Key], Out, Err),
    (   sub_string(Err, _, _, _, "run=true") -> Ok = true
    ;   sub_string(Err, _, _, _, "run=false") -> Ok = false
    ;   Ok = crashed(Err)
    ).

%% rust_once(+Mode, +Name, -Out) is semidet.
%  Fails when the query predicate was not lowered.
rust_once(Mode, Name, Out) :-
    bb_mode(Mode, _, _, Dir),
    lowered_entry(Dir, Name, Fn),
    bb_program(Name, _, Q, O, _),
    Q =.. [_|QArgs],
    maplist(bb_arg_text(O), QArgs, ArgTexts),
    bb_run(Mode, ['--lowered', Fn|ArgTexts], Out, _).

bb_arg_text(O, A, '_') :- A == O, !.
bb_arg_text(_, A, T) :- format(atom(T), '~w', [A]).

% ---------------------------------------------------------------------
% Checks
% ---------------------------------------------------------------------

bb_all_mode(interpreter).
bb_all_mode(functions).

check_all(Mode, Name) :-
    bb_known_divergence(Mode, Name, Why),
    !,
    format(user_error, '  [~w] ~w: known divergence, not compared (~w)~n',
           [Mode, Name, Why]).
check_all(Mode, Name) :-
    bb_ensure_built(Mode),
    swi_all(Name, SOut, SOk),
    rust_all(Mode, Name, ROut, ROk),
    (   SOut == ROut, SOk == ROk
    ->  true
    ;   format(user_error,
               'BUGFIX-1 DIVERGENCE [~w] ~w~n  swi : ~q run=~w~n  rust: ~q run=~w~n',
               [Mode, Name, SOut, SOk, ROut, ROk]),
        fail
    ).

check_once(Mode, Name) :-
    bb_ensure_built(Mode),
    (   bb_lowered_out_of_scope(Name, Why)
    ->  format(user_error, '  [~w] ~w: lowered entry not checked (~w)~n',
               [Mode, Name, Why])
    ;   rust_once(Mode, Name, ROut)
    ->  swi_once(Name, SOut),
        (   SOut == ROut
        ->  true
        ;   format(user_error,
                   'BUGFIX-1 LOWERED DIVERGENCE [~w] ~w~n  swi once: ~q~n  rust    : ~q~n',
                   [Mode, Name, SOut, ROut]),
            fail
        )
    ;   format(user_error, '  [~w] ~w: query predicate not lowered (skipped)~n',
               [Mode, Name])
    ).

test_wam_rust_bugfix_batch_1 :-
    run_tests(wam_rust_bugfix_batch_1).

:- begin_tests(wam_rust_bugfix_batch_1, [condition(cargo_available),
                                        setup(install_bb_programs)]).

test(all_solutions_match_swi,
     [forall(( bb_all_mode(Mode), bb_program(Name, _, _, _, _) ))]) :-
    once(check_all(Mode, Name)).

test(lowered_first_solution_matches_swi,
     [forall(( member(Mode, [functions, functions_nodef]),
               bb_program(Name, _, _, _, _) ))]) :-
    once(check_once(Mode, Name)).

test(known_divergences_still_present,
     [forall(bb_known_divergence(Mode, Name, _))]) :-
    bb_ensure_built(Mode),
    swi_all(Name, SOut, SOk),
    rust_all(Mode, Name, ROut, ROk),
    \+ ( SOut == ROut, SOk == ROk ).

test(lowered_entries_present, [forall(bb_lowered_required(Name))]) :-
    bb_ensure_built(functions),
    bb_mode(functions, _, _, Dir),
    once(lowered_entry(Dir, Name, _)).

:- end_tests(wam_rust_bugfix_batch_1).
