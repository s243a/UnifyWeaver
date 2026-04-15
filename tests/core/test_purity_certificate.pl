:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_purity_certificate.pl — Phase P1 tests for the shared purity
% certificate module. Covers:
%   - Certificate shape + builtin producers
%   - User annotations (parallel/order_independent/parallel_safe)
%   - Blacklist verdicts with correct reason atoms
%   - Determinism (repeated calls return bit-identical certs)
%   - Termination on deeply-nested bodies
%   - Back-compat equivalence with clause_body_analysis:is_pure_goal/1
%     and is_order_independent/2
%
% Run: swipl -q -g "consult('tests/core/test_purity_certificate.pl'), \
%       run_tests(purity_certificate), halt(0)"

:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/core/purity_certificate').
:- use_module('../../src/unifyweaver/core/clause_body_analysis').

:- begin_tests(purity_certificate).

% ----------------------------------------------------------------------------
% Certificate shape
% ----------------------------------------------------------------------------

test(goal_pure_blacklist_clean) :-
    analyze_goal_purity(member(_, [1,2,3]), Cert),
    Cert = purity_cert(pure, analyzed(blacklist), Conf, Reasons),
    assertion(Conf >= 0.8),
    assertion(memberchk(blacklist_clean, Reasons)).

test(goal_impure_io_write) :-
    analyze_goal_purity(write(hello), Cert),
    Cert = purity_cert(impure(Rs), analyzed(blacklist), _, Reasons),
    assertion(memberchk(io_ops, Rs)),
    assertion(memberchk(io_ops, Reasons)).

test(goal_impure_io_format) :-
    analyze_goal_purity(format('~w', [x]), Cert),
    Cert = purity_cert(impure([io_ops]), _, _, _).

test(goal_impure_database) :-
    analyze_goal_purity(assertz(foo(1)), Cert),
    Cert = purity_cert(impure([database_mods]), _, _, _).

test(goal_impure_retract) :-
    analyze_goal_purity(retractall(foo(_)), Cert),
    Cert = purity_cert(impure([database_mods]), _, _, _).

test(goal_impure_global_state) :-
    analyze_goal_purity(nb_setval(x, 1), Cert),
    Cert = purity_cert(impure([global_state]), _, _, _).

test(goal_impure_domain) :-
    analyze_goal_purity(send_message(a, b), Cert),
    Cert = purity_cert(impure([domain_specific]), _, _, _).

% ----------------------------------------------------------------------------
% User annotations
% ----------------------------------------------------------------------------

test(pred_declared_parallel,
     [setup(assertz(clause_body_analysis:order_independent(user:p1/2))),
      cleanup(retract(clause_body_analysis:order_independent(user:p1/2)))]) :-
    analyze_predicate_purity(p1/2, Cert),
    Cert = purity_cert(pure, declared, 1.0, Reasons),
    assertion(memberchk(declared_by_user, Reasons)).

test(pred_declared_with_module,
     [setup(assertz(clause_body_analysis:order_independent(user:p2/3))),
      cleanup(retract(clause_body_analysis:order_independent(user:p2/3)))]) :-
    analyze_predicate_purity(user:p2/3, Cert),
    Cert = purity_cert(pure, declared, 1.0, _).

test(pred_parallel_safe_hook,
     [setup(assertz(clause_body_analysis:parallel_safe(custom_goal/1))),
      cleanup(retract(clause_body_analysis:parallel_safe(custom_goal/1)))]) :-
    analyze_goal_purity(custom_goal(x), Cert),
    Cert = purity_cert(pure, declared, 1.0, Reasons),
    assertion(memberchk(parallel_safe, Reasons)).

% ----------------------------------------------------------------------------
% Predicate-level clause walk
% ----------------------------------------------------------------------------

test(pred_clause_walk_pure,
     [setup(assertz((user:pure_pred(X, Y) :- member(X, [1,2,3]), Y is X + 1))),
      cleanup(retractall(user:pure_pred(_, _)))]) :-
    analyze_predicate_purity(pure_pred/2, Cert),
    Cert = purity_cert(pure, _, _, _).

test(pred_clause_walk_impure,
     [setup(assertz((user:impure_pred(X) :- write(X), nl))),
      cleanup(retractall(user:impure_pred(_)))]) :-
    analyze_predicate_purity(impure_pred/1, Cert),
    Cert = purity_cert(impure(Rs), _, _, _),
    assertion(memberchk(io_ops, Rs)).

test(pred_no_clauses_unknown) :-
    analyze_predicate_purity(does_not_exist_xyz/3, Cert),
    Cert = purity_cert(unknown, _, _, _).

% ----------------------------------------------------------------------------
% Convenience predicates
% ----------------------------------------------------------------------------

test(is_certified_pure_declared,
     [setup(assertz(clause_body_analysis:order_independent(user:q1/0))),
      cleanup(retract(clause_body_analysis:order_independent(user:q1/0)))]) :-
    assertion(is_certified_pure(q1/0)).

test(is_certified_impure_has_reasons,
     [setup(assertz((user:bad_pred :- format('~w', [x])))),
      cleanup(retractall(user:bad_pred))]) :-
    is_certified_impure(bad_pred/0, Reasons),
    assertion(memberchk(io_ops, Reasons)).

test(purity_confidence_declared_is_one,
     [setup(assertz(clause_body_analysis:order_independent(user:q2/0))),
      cleanup(retract(clause_body_analysis:order_independent(user:q2/0)))]) :-
    purity_confidence(q2/0, Conf),
    assertion(Conf =:= 1.0).

% ----------------------------------------------------------------------------
% Determinism
% ----------------------------------------------------------------------------

test(determinism_goal) :-
    analyze_goal_purity(write(hello), C1),
    analyze_goal_purity(write(hello), C2),
    analyze_goal_purity(write(hello), C3),
    assertion(C1 == C2),
    assertion(C2 == C3).

test(determinism_predicate,
     [setup(assertz(clause_body_analysis:order_independent(user:det_pred/1))),
      cleanup(retract(clause_body_analysis:order_independent(user:det_pred/1)))]) :-
    analyze_predicate_purity(det_pred/1, C1),
    analyze_predicate_purity(det_pred/1, C2),
    assertion(C1 == C2).

% ----------------------------------------------------------------------------
% Termination on deep goals
% ----------------------------------------------------------------------------

test(termination_deep_conjunction,
     [setup(build_deep_conjunction(500, Body),
            assertz((user:deep_pred :- Body))),
      cleanup(retractall(user:deep_pred))]) :-
    analyze_predicate_purity(deep_pred/0, Cert),
    Cert = purity_cert(pure, _, _, _).

% Helper: produces (member(X1,[1]), member(X2,[1]), …, member(XN,[1]))
build_deep_conjunction(1, member(_, [1])) :- !.
build_deep_conjunction(N, (member(_, [1]), Rest)) :-
    N > 1,
    N1 is N - 1,
    build_deep_conjunction(N1, Rest).

% ----------------------------------------------------------------------------
% Merge semantics
% ----------------------------------------------------------------------------

test(merge_empty) :-
    merge_certificates([], purity_cert(unknown, _, _, _)).

test(merge_single_passthrough) :-
    Cert = purity_cert(pure, declared, 1.0, [x]),
    merge_certificates([Cert], Merged),
    assertion(Merged == Cert).

test(merge_all_pure_picks_max_confidence) :-
    C1 = purity_cert(pure, analyzed(blacklist), 0.8, [a]),
    C2 = purity_cert(pure, declared, 1.0, [b]),
    merge_certificates([C1, C2], Merged),
    Merged = purity_cert(pure, declared, Conf, _),
    assertion(Conf =:= 1.0).

test(merge_any_impure_wins) :-
    C1 = purity_cert(pure, declared, 1.0, [a]),
    C2 = purity_cert(impure([io_ops]), analyzed(blacklist), 0.95, [io_ops]),
    merge_certificates([C1, C2], Merged),
    Merged = purity_cert(impure([io_ops]), _, _, _).

test(merge_pure_plus_unknown_high_conf_yields_pure) :-
    C1 = purity_cert(pure, declared, 1.0, [a]),
    C2 = purity_cert(unknown, inferred, 0.3, [b]),
    merge_certificates([C1, C2], Merged),
    Merged = purity_cert(pure, _, _, _).

test(merge_pure_plus_unknown_low_conf_yields_unknown) :-
    C1 = purity_cert(pure, analyzed(blacklist), 0.8, [a]),
    C2 = purity_cert(unknown, inferred, 0.5, [b]),
    merge_certificates([C1, C2], Merged),
    Merged = purity_cert(unknown, _, _, _).

% ----------------------------------------------------------------------------
% Back-compat equivalence with legacy clause_body_analysis predicates
% ----------------------------------------------------------------------------

test(backcompat_is_pure_goal_member) :-
    assertion(clause_body_analysis:is_pure_goal(member(_, [1]))).

test(backcompat_is_pure_goal_write_fails) :-
    assertion(\+ clause_body_analysis:is_pure_goal(write(hi))).

test(backcompat_is_pure_goal_assertz_fails) :-
    assertion(\+ clause_body_analysis:is_pure_goal(assertz(foo))).

test(backcompat_is_order_independent_declared,
     [setup(assertz(clause_body_analysis:order_independent(user:bcpred/1))),
      cleanup(retract(clause_body_analysis:order_independent(user:bcpred/1)))]) :-
    clause_body_analysis:is_order_independent(bcpred/1, R),
    assertion(R == declared).

test(backcompat_is_order_independent_proven,
     [setup(assertz((user:bcpred2(X,Y) :- Y is X + 1))),
      cleanup(retractall(user:bcpred2(_,_)))]) :-
    clause_body_analysis:is_order_independent(bcpred2/2, R),
    assertion(R == proven([pure_goals])).

test(backcompat_is_order_independent_impure_fails,
     [setup(assertz((user:bcpred3(X) :- write(X)))),
      cleanup(retractall(user:bcpred3(_)))]) :-
    assertion(\+ clause_body_analysis:is_order_independent(bcpred3/1, _)).

% ----------------------------------------------------------------------------
% Producer registry
% ----------------------------------------------------------------------------

test(registered_producers_has_builtins) :-
    registered_producers(Specs),
    assertion(member(producer_spec(user_annotations, 100, _), Specs)),
    assertion(member(producer_spec(blacklist, 50, _), Specs)).

test(user_annotations_outranks_blacklist,
     % Declared pure wins even over an impure-looking body, because
     % user_annotations has higher priority than blacklist.
     [setup((
        assertz(clause_body_analysis:order_independent(user:override_pred/1)),
        assertz((user:override_pred(X) :- write(X)))  % impure body
      )),
      cleanup((
        retract(clause_body_analysis:order_independent(user:override_pred/1)),
        retractall(user:override_pred(_))
      ))]) :-
    analyze_predicate_purity(override_pred/1, Cert),
    Cert = purity_cert(pure, declared, 1.0, _).

:- end_tests(purity_certificate).
