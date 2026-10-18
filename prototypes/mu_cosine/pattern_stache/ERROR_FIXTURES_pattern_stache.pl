% SPDX-License-Identifier: MIT OR Apache-2.0
%
% ERROR_FIXTURES_pattern_stache.pl - SEALED golden fixtures for the
% lane's typed diagnostics (desugaring doc §7; oracle row closed per
% DESIGN_prolog_elaborator.md §4 "typed diagnostics").
%
% One fact per error class, mapping the class label to the EXACT term
% thrown (inside the error/2 wrapper), with variables in thrown terms
% frozen as '$VAR'(N) in first-occurrence order — tests raise each
% error live, apply numbervars/3 to a copy, and require ==.
%
% FORMAT CHOICE, recorded: this is a Prolog data file read with
% read_term/2 (never consulted as code), NOT JSON.  The fixtures ARE
% Prolog terms; a JSON encoding would need a bespoke term<->JSON
% convention (atom-vs-string ambiguity, operator spelling, quoting)
% whose silent-misparse risk is the exact hazard class this lane
% documents.  read_term with the standard operator table reads the
% terms exactly.
%
% SEALING, recorded: the raising test (test_error_fixtures.pl) embeds
% this file's sha256 as a frozen constant and verifies it before
% comparing anything — so a fixture edit and a test edit must meet in
% the same review, the same double-lock as the mode-classification
% suite's table citation.  Same lifecycle as every golden in the lane:
% sealed once stable, re-sealed only deliberately.
%
% These goldens pin OUR error terms.  They are deliberately NOT
% compared against vNext's Python error strings — that coupling is
% named forbidden in the design note's oracle inventory.
%
% Class coverage note: pe_elaborate's binding_variable_already_bound
% is a defensive clause that validation makes unreachable (duplicates
% are refused up front); it has no fixture because no input raises it.

error_fixtures_version(1).

% -- pe_where: the where-form's fail-closed classes --
error_fixture(w_not_where,
    pe_where(not_a_where_term(lca_frac(simplemind)))).
error_fixture(w_bad_list,
    pe_where(bad_binding_list(oops))).
error_fixture(w_bad_binding,
    pe_where(bad_binding(simplemind))).
error_fixture(w_dup,
    pe_where(duplicate_binding_for_one_variable)).
error_fixture(w_dead,
    pe_where(dead_binding('$VAR'(0)=fs))).
error_fixture(w_pin_pos,
    pe_where(binding_reaches_pin_channel(pin_name))).
error_fixture(w_illegal,
    pe_where(illegal_binding_value(foo, at(kwarg(margin, t))))).
error_fixture(w_unbound,
    pe_where(unbound_after_elaboration(product(hop_decay(simplemind, gamma(0.6)), lca_frac('$VAR'(0)))))).

% -- pe_elaborate: the elaborator taxonomy, incl. v1.1 --
error_fixture(e_bad_pairs,
    pe_elaborate(bad_goal_pairs([not_a_pair]))).
error_fixture(e_dead,
    pe_elaborate(dead_binding('$VAR'(0)=simplemind))).
error_fixture(e_unknown,
    pe_elaborate(unknown_constraint(frobnicate(x), origin(none)))).
error_fixture(e_failed,
    pe_elaborate(constraint_failed(has_type(t1, substrate(haiku)), origin('surface:t1::substrate')))).
error_fixture(e_unsat,
    pe_elaborate(constraint_unsatisfiable(has_type('$VAR'(0), substrate(frobnicate)), origin(none)))).
error_fixture(e_pin_pos,
    pe_elaborate(binding_rejected(binding_reaches_pin_channel(pin_name), origin(none)))).
error_fixture(e_pin_val,
    pe_elaborate(binding_rejected(illegal_binding_value(pin(fs, 'run/1'), at(arg(lca_frac, 1))), origin(none)))).
