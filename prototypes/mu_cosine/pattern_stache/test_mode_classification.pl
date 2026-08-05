:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_mode_classification.pl - the mode table of
% DESUGARING DESIGN_desugaring_to_prolog_goals.md §3, transcribed into
% unit assertions against the LIVE elaborator.
%
% Run from this directory:
%   swipl -g run_tests -t halt test_mode_classification.pl
%
% ============================================================
% THE TABLE (DESIGN_desugaring_to_prolog_goals.md §3, cited so a
% table edit and a behavior edit must meet in review):
%
%   | goal                                   | instantiation      | discharged      |
%   |----------------------------------------|--------------------|-----------------|
%   | non_amplifying(product)                | ground, closed     | elaboration     |
%   | has_type(ptree(pt), substrate(pt))     | ground             | elaboration     |
%   | has_type(X, substrate(C)), X free      | under-instantiated | RESIDUATES      |
%   | in_support(decay, D), needs data       | needs data         | runtime         |
%
% The precondition the elaborator report named — "these tests become
% trustworthy only when the constraint registry (ruling 5) seals which
% predicates are closed" — holds: ruling 5(b) shipped as the generated,
% hash-checked mirror, so "closed" is now a property of a sealed
% artifact, not of prose.
%
% TWO RECORDED DIVERGENCES between the table and the live v1/v1.1
% elaborator, both scope facts, not bugs:
%   - non_amplifying/1: the table says "discharged at elaboration,
%     always" — but non_amplifying facts are NOT in the sealed v0.4
%     registry, and §6.1 forbids closed-predicate claims without
%     registry backing.  Under ruling 4(a) the live elaborator refuses
%     it fail-closed (unknown_constraint).  The table's row unblocks
%     when the parked registry-derivation work lands (out of scope
%     here, tracked in the AST lane).
%   - in_support/2 ("runtime"): runtime-phase constraint classes are
%     checker-dispatch territory (consumer 2's phase protocol), not
%     the v1 elaborator; under ruling 4(a) the live elaborator refuses
%     it fail-closed as well.
% Each divergence is asserted BELOW as the live behavior, so the day
% either construct is admitted, these tests fail and force this header
% and the table to be reconciled in the same review.
%
% Classification vocabulary (the elaborator's observable outcomes):
%   DISCHARGE  the goal is consumed; it does not appear in the result
%   RESIDUATE  the goal survives into pattern(_, Store)
%   ERROR      a named pe_elaborate error at elaboration time

:- use_module(pe_elaborate, [elaborate/3]).
:- use_module(library(plunit)).
:- use_module(library(lists)).

:- begin_tests(mode_classification).

% -- ground closed check (table row 2): DISCHARGE --
test(ground_true_check_discharges) :-
    elaborate(fs, [has_type(t1, substrate(pearltrees))], ground(fs)).

% ...and the fail-closed refinement of "discharged": a ground check
% that is FALSE is an error, never a silent discharge or residual.
test(ground_false_check_errors,
     error(pe_elaborate(constraint_failed(_, _)))) :-
    elaborate(fs, [has_type(t1, substrate(haiku))], _).

% -- ground binding (§12: bindings vanish at elaboration): DISCHARGE --
test(ground_binding_discharges) :-
    elaborate(lca_frac(C), [C = simplemind], ground(lca_frac(simplemind))).

% -- binding with nonground value: RESIDUATE until its value grounds
%    (and DISCHARGE in a later pass when it does) --
test(nonground_value_binding_residuates_then_discharges) :-
    elaborate(lca_frac(C), [C = D, D = simplemind],
              ground(lca_frac(simplemind))).

% -- under-instantiated has_type (table row 3): RESIDUATE --
test(under_instantiated_has_type_residuates) :-
    elaborate(lca_frac(C), [has_type(_X, substrate(C))],
              pattern(_, [has_type(_, substrate(_))])).

% ...including a free subject over a ground, SATISFIABLE type side
% (v1.1: eagerly validated, unchanged when legal).
test(free_subject_ground_type_residuates_when_legal) :-
    elaborate(fs, [has_type(_X, substrate(pearltrees))],
              pattern(fs, [has_type(_, substrate(pearltrees))])).

% ...but ground-and-false in the ground portion is an ERROR, not a
% dormant residual (v1.1, ruled on PR #4095).
test(ground_and_false_portion_errors,
     error(pe_elaborate(constraint_unsatisfiable(_, _)))) :-
    elaborate(fs, [has_type(_X, substrate(frobnicate))], _).

% -- table row 1, RECORDED DIVERGENCE: non_amplifying(product) --
% The table says "discharged at elaboration, always"; the live
% elaborator refuses it because the sealed registry carries no
% non_amplifying facts (ruling 4(a) scope; §6.1 closure discipline).
% When the registry-derivation work admits it, THIS test fails and
% the header above must be reconciled with the table in one review.
test(non_amplifying_currently_refused_fail_closed,
     error(pe_elaborate(unknown_constraint(non_amplifying(product), _)))) :-
    elaborate(fs, [non_amplifying(product)], _).

% -- table row 4, RECORDED DIVERGENCE: in_support(decay, D) --
% "Runtime" is a checker-dispatch phase (consumer 2), outside the v1
% elaborator's ruled scope; refused fail-closed for now.
test(in_support_currently_refused_fail_closed,
     error(pe_elaborate(unknown_constraint(in_support(decay, _), _)))) :-
    elaborate(fs, [in_support(decay, _D)], _).

% -- the mode split is per-goal, not per-store: one store can
%    discharge, residuate, and carry bindings simultaneously --
test(mixed_store_classifies_per_goal) :-
    elaborate(product(hop_decay(C, gamma(0.6)), lca_frac(C)),
              [ has_type(t1, substrate(pearltrees)),   % ground true: discharge
                C = simplemind,                        % ground binding: discharge
                has_type(_Y, judge(_J))                % under-instantiated: residuate
              ],
              pattern(T, Store)),
    T == product(hop_decay(simplemind, gamma(0.6)), lca_frac(simplemind)),
    Store = [has_type(_, judge(_))].

:- end_tests(mode_classification).
