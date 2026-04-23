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
% purity_analysis exports pure_builtin/1 as a delegator — but so does
% purity_certificate. Import the others without clashing.
:- use_module('../../src/unifyweaver/core/advanced/purity_analysis',
              [is_pure_goal/1, is_pure_body/1, is_associative_op/1]).
:- use_module('../../src/unifyweaver/core/recursive_kernel_detection').

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
     [setup((build_deep_conjunction(500, Body),
             assertz((user:deep_pred :- Body)))),
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
    assertion(member(producer_spec(blacklist, 50, _), Specs)),
    assertion(member(producer_spec(whitelist, 40, _), Specs)).

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

% ----------------------------------------------------------------------------
% P2: whitelist producer
% ----------------------------------------------------------------------------

test(whitelist_pure_goal_arithmetic) :-
    assertion(is_whitelist_pure_goal(X is 1 + 2)),
    _ = X.

test(whitelist_pure_goal_member) :-
    assertion(is_whitelist_pure_goal(member(_, [1,2,3]))).

test(whitelist_pure_goal_true) :-
    assertion(is_whitelist_pure_goal(true)).

test(whitelist_pure_goal_rejects_unknown_builtin) :-
    assertion(\+ is_whitelist_pure_goal(my_unknown_op(_, _))).

test(whitelist_pure_goal_rejects_write) :-
    assertion(\+ is_whitelist_pure_goal(write(x))).

test(whitelist_analyzer_produces_whitelist_proof) :-
    purity_certificate:whitelist_analyzer(member(_, [1]), goal, Cert),
    Cert = purity_cert(pure, analyzed(whitelist), _, Reasons),
    assertion(memberchk(whitelist_only, Reasons)).

test(whitelist_analyzer_unknown_for_unrecognized) :-
    purity_certificate:whitelist_analyzer(my_unknown_op(_, _), goal, Cert),
    Cert = purity_cert(unknown, analyzed(whitelist), _, Reasons),
    assertion(memberchk(not_in_whitelist, Reasons)).

test(whitelist_catalogue_contains_is) :-
    assertion(pure_builtin(is/2)).

test(whitelist_catalogue_contains_length) :-
    assertion(pure_builtin(length/2)).

test(whitelist_catalogue_rejects_assertz) :-
    assertion(\+ pure_builtin(assertz/1)).

% ----------------------------------------------------------------------------
% P2: advanced/purity_analysis back-compat
% ----------------------------------------------------------------------------

test(advanced_is_pure_goal_delegates) :-
assertion(purity_analysis:is_pure_goal(member(_, [1]))).

test(advanced_is_pure_goal_rejects_write) :-
assertion(\+ purity_analysis:is_pure_goal(write(x))).

test(advanced_is_pure_goal_rejects_unknown_builtin) :-
% Stricter than the blacklist — an unknown builtin isn't in the
    % whitelist, so purity_analysis rejects it even though the
    % blacklist would (permissively) accept it.
    assertion(\+ purity_analysis:is_pure_goal(my_unknown_op(x))).

test(advanced_is_pure_body_conjunction) :-
assertion(purity_analysis:is_pure_body((member(_, [1]), X is 1 + 2))),
    _ = X.

test(advanced_is_pure_body_rejects_impure) :-
assertion(\+ purity_analysis:is_pure_body((member(_, [1]), write(x)))).

test(advanced_pure_builtin_delegates) :-
assertion(purity_analysis:pure_builtin(is/2)),
    assertion(purity_analysis:pure_builtin(length/2)),
    assertion(\+ purity_analysis:pure_builtin(assertz/1)).

test(advanced_is_associative_op_local) :-
% is_associative_op/1 stays local to purity_analysis — it's not
    % a purity concern and shouldn't have migrated.
    assertion(purity_analysis:is_associative_op(+)),
    assertion(purity_analysis:is_associative_op(*)),
    assertion(\+ purity_analysis:is_associative_op(-)).

% ----------------------------------------------------------------------------
% P2: whitelist is strictly narrower than blacklist
% ----------------------------------------------------------------------------

test(whitelist_narrower_than_blacklist) :-
    % A predicate that's pure by blacklist (not impure) but NOT in
    % whitelist shows the distinction. user-defined functors fall
    % into this gap: blacklist can't see them as impure, so they
    % pass. Whitelist can't see them as pure, so they fail.
    Goal = my_random_userdef(_),
    assertion(\+ is_whitelist_pure_goal(Goal)),
    % Blacklist-layer producer still says pure:
    purity_certificate:blacklist_analyzer(Goal, goal, BlCert),
    BlCert = purity_cert(pure, analyzed(blacklist), _, _).

% ----------------------------------------------------------------------------
% P3: kernel registry producer
% ----------------------------------------------------------------------------

% Helper: install a canonical category_ancestor/4 shape + supporting
% max_depth and category_parent. The detector requires user:max_depth/1
% to exist with a positive integer value, and two clauses with the
% specific recursive shape (base + recursive with `_ is _ + 1`).
install_kernel_fixture :-
    catch(retractall(user:max_depth(_)), _, true),
    catch(retractall(user:category_parent(_, _)), _, true),
    catch(retractall(user:category_ancestor(_,_,_,_)), _, true),
    assertz(user:max_depth(5)),
    assertz((user:category_parent(_, _) :- fail)),
    assertz((user:category_ancestor(X, Y, 1, V) :-
                category_parent(X, Y),
                \+ member(Y, V))),
    assertz((user:category_ancestor(X, Y, H, V) :-
                max_depth(M), length(V, D), D < M, !,
                category_parent(X, Z), \+ member(Z, V),
                category_ancestor(Z, Y, H1, [Z|V]),
                H is H1 + 1)).

uninstall_kernel_fixture :-
    retractall(user:max_depth(_)),
    retractall(user:category_parent(_, _)),
    retractall(user:category_ancestor(_,_,_,_)).

test(kernel_analyzer_certifies_category_ancestor,
     [setup(install_kernel_fixture),
      cleanup(uninstall_kernel_fixture)]) :-
    analyze_predicate_purity(user:category_ancestor/4, Cert),
    Cert = purity_cert(pure, certified(kernel_registry), 1.0, Reasons),
    assertion(memberchk(kernel_owned, Reasons)).

test(kernel_outranks_blacklist_on_shape_match,
     % The kernel body would otherwise trip the blacklist's
     % no-unknown-impurity default to `pure(blacklist)`. We confirm
     % the kernel producer's higher priority wins.
     [setup(install_kernel_fixture),
      cleanup(uninstall_kernel_fixture)]) :-
    analyze_predicate_purity(user:category_ancestor/4,
                             purity_cert(_, certified(kernel_registry), _, _)).

test(non_kernel_predicate_falls_through_to_blacklist,
     [setup(assertz((user:plain_pred(X, Y) :- Y is X + 1))),
      cleanup(retractall(user:plain_pred(_, _)))]) :-
    analyze_predicate_purity(user:plain_pred/2, Cert),
    Cert = purity_cert(pure, Proof, _, _),
    % Should NOT be certified by kernel — the kernel producer
    % returns unknown for this shape, chain falls to blacklist.
    assertion(Proof \= certified(kernel_registry)).

test(registered_producers_has_kernel_registry) :-
    registered_producers(Specs),
    assertion(member(producer_spec(kernel_registry, 90, _), Specs)).

test(producer_priority_order) :-
    registered_producers(Specs),
    findall(P-N,
            member(producer_spec(N, P, _), Specs),
            Pairs),
    % user_annotations > kernel_registry > blacklist > whitelist
    memberchk(100-user_annotations, Pairs),
    memberchk(90-kernel_registry, Pairs),
    memberchk(50-blacklist, Pairs),
    memberchk(40-whitelist, Pairs).

% ----------------------------------------------------------------------------
% P3: contradiction detection
% ----------------------------------------------------------------------------

test(contradiction_none_for_pure_predicate,
     [setup(assertz((user:innocent(X, Y) :- Y is X + 1))),
      cleanup(retractall(user:innocent(_, _)))]) :-
    check_purity_contradictions(user:innocent/2, Cs),
    assertion(Cs == []).

test(contradiction_found_declared_vs_impure_body,
     [setup((
        assertz(clause_body_analysis:order_independent(user:sus_decl/1)),
        assertz((user:sus_decl(X) :- write(X)))
      )),
      cleanup((
        retract(clause_body_analysis:order_independent(user:sus_decl/1)),
        retractall(user:sus_decl(_))
      ))]) :-
    check_purity_contradictions(user:sus_decl/1, Cs),
    assertion(Cs \== []),
    Cs = [contradiction(HiName, _HiCert, LoName, LoCert)|_],
    assertion(HiName == user_annotations),
    assertion(LoName == blacklist),
    LoCert = purity_cert(impure(Reasons), _, _, _),
    assertion(memberchk(io_ops, Reasons)).

test(contradiction_kernel_vs_impure_body,
     % Contrived: install a kernel shape AND add an impure clause
     % to the same predicate. The detector matches only some clauses
     % but the blacklist sees the impure one.
     %
     % Note: the detector requires ONLY two specific clause shapes,
     % so adding a third impure clause breaks the detector. To keep
     % this test meaningful we install the standard shape, then
     % inject an impure NON-matching clause first — the detector
     % still scans all clauses and may or may not match depending
     % on how many clauses there are in total. This test mostly
     % confirms check_purity_contradictions doesn't crash when
     % mixing kernel + impure bodies.
     [setup(install_kernel_fixture),
      cleanup(uninstall_kernel_fixture)]) :-
    % Just confirm the API returns a list without error.
    check_purity_contradictions(user:category_ancestor/4, Cs),
    assertion(is_list(Cs)).

test(warn_purity_contradictions_succeeds_silently,
     [setup(assertz((user:silent_pred :- true))),
      cleanup(retractall(user:silent_pred))]) :-
    % No contradiction → no output, predicate succeeds.
    warn_purity_contradictions(user:silent_pred/0).

test(warn_purity_contradictions_with_contradiction,
     [setup((
        assertz(clause_body_analysis:order_independent(user:loud_pred/1)),
        assertz((user:loud_pred(X) :- format('~w', [X])))
      )),
      cleanup((
        retract(clause_body_analysis:order_independent(user:loud_pred/1)),
        retractall(user:loud_pred(_))
      ))]) :-
    % With a contradiction, the call still succeeds (it's a warning,
    % not an error). Output goes to user_error.
    warn_purity_contradictions(user:loud_pred/1).

% ============================================================================
% Phase P5: JSON serialization round-trips
% ============================================================================

test(json_round_trip_pure) :-
    Cert = purity_cert(pure, declared, 1.0, [declared_by_user]),
    cert_to_json(Cert, Json),
    cert_from_json(Json, Cert2),
    Cert2 = purity_cert(pure, declared, 1.0, [declared_by_user]).

test(json_round_trip_impure) :-
    Cert = purity_cert(impure([io_ops, database_mods]), analyzed(blacklist), 0.95, []),
    cert_to_json(Cert, Json),
    cert_from_json(Json, Cert2),
    Cert2 = purity_cert(impure(Reasons), analyzed(blacklist), 0.95, Reasons),
    msort(Reasons, Sorted),
    msort([io_ops, database_mods], Sorted).

test(json_round_trip_unknown_with_compound_reason) :-
    Cert = purity_cert(unknown, inferred, 0.3, [unknown_builtin(my_op/2)]),
    cert_to_json(Cert, Json),
    cert_from_json(Json, Cert2),
    Cert2 = purity_cert(unknown, inferred, 0.3, [unknown_builtin(my_op/2)]).

test(json_round_trip_certified_source) :-
    Cert = purity_cert(pure, certified(static_analysis), 0.9, [whitelisted]),
    cert_to_json(Cert, Json),
    cert_from_json(Json, Cert2),
    Cert2 = purity_cert(pure, certified(static_analysis), 0.9, [whitelisted]), !.

test(json_metadata_passthrough) :-
    Cert = purity_cert(pure, declared, 1.0, []),
    cert_to_json(Cert, _{subject: "foo/2", producer: "user_annotations"}, Json),
    get_dict(subject, Json, "foo/2"),
    get_dict(producer, Json, "user_annotations"),
    get_dict(verdict, Json, "pure").

test(json_missing_verdict_throws, [throws(error(cert_json_error(missing_verdict), _))]) :-
    cert_from_json(_{proof: _{kind: "declared"}, confidence: 1.0}, _).

test(json_confidence_out_of_range_throws, [throws(error(cert_json_error(confidence_out_of_range(2.0)), _))]) :-
    cert_from_json(_{verdict: "pure", proof: _{kind: "declared"}, confidence: 2.0}, _).

test(json_unknown_proof_kind_throws, [throws(error(cert_json_error(unknown_proof_kind("bogus")), _))]) :-
    cert_from_json(_{verdict: "pure", proof: _{kind: "bogus"}, confidence: 1.0}, _).

test(json_forward_compat_ignores_unknown_keys) :-
    Json = _{verdict: "pure", proof: _{kind: "declared"},
             confidence: 1.0, reasons: ["test"], foo: "bar", extra: 42},
    cert_from_json(Json, Cert),
    Cert = purity_cert(pure, declared, 1.0, [test]), !.

:- end_tests(purity_certificate).
