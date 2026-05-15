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
:- dynamic user:wam_cpp_test_enum_member/0.

user:wam_cpp_test_member_yes   :- member(b, [a, b, c]).
user:wam_cpp_test_member_no    :- member(z, [a, b, c]).
user:wam_cpp_test_member_first :- member(a, [a, b, c]).
user:wam_cpp_test_length_three :- length([a, b, c], 3).
user:wam_cpp_test_length_zero  :- length([], 0).
user:wam_cpp_test_length_bad   :- length([a, b, c], 5).
user:wam_cpp_test_copy_basic   :- copy_term(foo(X, X, _Y), T), T = foo(A, A, _B).
user:wam_cpp_test_copy_atom    :- copy_term(hello, T), T = hello.
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
    % frame + meta-call composition. (The richer
    % "findall + member + call/N" form has a separate latent bug
    % involving aggregate-frame + choice-point interaction with
    % conjunctions — not addressed in this PR.)
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
