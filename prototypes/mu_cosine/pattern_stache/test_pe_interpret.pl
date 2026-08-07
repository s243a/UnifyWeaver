:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_pe_interpret.pl - §7 of DESIGN_process_expression_patterns.md
% (ruling 4(b)) against the checked-in relation.
%
% Run from this directory:
%   swipl -g run_tests -t halt test_pe_interpret.pl
%
% Organised by spec section, with the REFUSALS treated as first-class
% as the happy paths — §7.3's whole content is what must be rejected,
% so a suite that only tested acceptance would be testing the easy
% half.  Both §7.3 worked examples appear verbatim as tests.

:- use_module(pe_interpret).
:- use_module(pe_registry_mirror,
              [pe_walk_shape/2, pe_walk_shape_kind/1, pe_weight_value/1]).
:- use_module(library(plunit)).
:- use_module(library(lists)).

subject(principal_tree(pearltrees)::substrate(pearltrees)).

% File scope, not inside a unit: three units use it.
structural_options([ estimand(structural_score),
                     floor(real_value("0.02")),
                     gamma(real_value("0.6")),
                     distance(hops),
                     shared_prefix(lca),
                     normalization(none),
                     argument_roles(ordered),
                     support(all_pairs),
                     unreachable(zero),
                     numeric(exact_decimal) ]).


%% ============================================
%% §7.1 The dispatcher pre-pass
%% ============================================

:- begin_tests(dispatch_prepass).

% "explicit estimand=... goes directly to the matching clauses"
test(explicit_estimand_passes_through) :-
    subject(S),
    dispatch(lineage_op(S, [estimand(hop_decay)]), Outcome),
    Outcome = explicit([lineage_op(S2, [estimand(hop_decay)])]),
    S2 == S.

% "absent estimand plus a content-bound family_spec expands to the
% finite, fully defaulted explicit requests listed by that
% specification"
test(family_spec_expands_to_finite_requests) :-
    subject(S),
    dispatch(lineage_op(S, [family_spec(lineage_interpretations_v1)]), Outcome),
    Outcome = family(lineage_interpretations_v1, Digest, Requests),
    atom(Digest),
    atom_length(Digest, 64),                   % sha256 hex
    Requests = [lineage_op(_, [estimand(hop_decay)])].

% The expansion drops family_spec itself: it has been consumed, and
% leaving it in would make the request fail consume_only downstream.
test(family_spec_field_is_consumed_by_expansion) :-
    subject(S),
    dispatch(lineage_op(S, [family_spec(lineage_interpretations_v1)]),
             family(_, _, [lineage_op(_, Opts)])),
    \+ memberchk(family_spec(_), Opts).

% "absent estimand and absent family_spec is underconstrained, not an
% implicit enumeration" — the refusal is the point.
test(absent_both_is_underconstrained) :-
    subject(S),
    dispatch(lineage_op(S, []), underconstrained(no_estimand_and_no_family_spec)).

% ...and it stays a refusal all the way out through resolve/3, rather
% than degrading into "here are all the interpretations".
test(underconstrained_survives_to_resolve) :-
    subject(S),
    resolve(lineage_op(S, []), unique, underconstrained(_)),
    resolve(lineage_op(S, []), all, underconstrained(_)).

test(unknown_family_spec_is_underconstrained) :-
    subject(S),
    dispatch(lineage_op(S, [family_spec(no_such_spec_v9)]),
             underconstrained(unknown_family_spec(no_such_spec_v9))).

test(duplicate_family_spec_refused) :-
    subject(S),
    dispatch(lineage_op(S, [family_spec(lineage_interpretations_v1),
                            family_spec(lineage_interpretations_v1)]),
             underconstrained(duplicate_family_spec(2))).

% The family-spec content digest is stable across calls (it is a
% content hash of the ruleset file, not a per-call value).
test(family_spec_digest_is_stable) :-
    family_spec_digest(_, D1),
    family_spec_digest(_, D2),
    D1 == D2.

:- end_tests(dispatch_prepass).

%% ============================================
%% §7.1 interpretation/3 — the two illustrative rules
%% ============================================

:- begin_tests(interpretation_rules).

% hop_decay with everything defaulted: the four defaults are exactly
% the spec's (real_value("0.85"), int_value("1"), ancestor,
% unbounded_depth).
test(hop_decay_all_defaults) :-
    subject(S),
    interpretation(lineage_op(S, [estimand(hop_decay)]), Sem, Rule),
    Rule == lineage_as_hop_decay_v1,
    Sem = semantic_request(hop_decay_targets(Subj, Params), RepOpts),
    Subj == principal_tree(pearltrees),
    Params == [decay(real_value("0.85")), hop_origin(int_value("1")),
               direction(ancestor), depth(unbounded_depth)],
    RepOpts == [].

% Supplied values override defaults, one at a time.
test(hop_decay_supplied_values_win) :-
    subject(S),
    interpretation(lineage_op(S, [estimand(hop_decay),
                                  decay(real_value("0.5")),
                                  direction(descendant)]),
                   semantic_request(hop_decay_targets(_, Params), _), _),
    memberchk(decay(real_value("0.5")), Params),
    memberchk(direction(descendant), Params),
    memberchk(hop_origin(int_value("1")), Params).   % still defaulted

% extract_options TRANSFERS impl rather than discarding it — the
% representation options carry it to §7.2.
test(impl_is_transferred_not_discarded) :-
    subject(S),
    interpretation(lineage_op(S, [estimand(hop_decay), impl(atom(graph_walk))]),
                   semantic_request(_, RepOpts), _),
    RepOpts == [impl(atom(graph_walk))].

% structural_score: nine REQUIRED fields, no defaults.
test(structural_score_with_all_required) :-
    subject(S),
    structural_options(Opts),
    interpretation(lineage_op(S, Opts), Sem, Rule),
    Rule == lineage_as_structural_score_v1,
    Sem = semantic_request(structural_lineage_score(_, Params), _),
    length(Params, 9).

% ...and missing even one required field rejects the candidate.
% (Drops `floor`, keeping the estimand, so the rule is selected and
% then fails on required_options rather than never being reached.)
test(structural_score_missing_required_rejected) :-
    subject(S),
    structural_options(Full),
    selectchk(floor(_), Full, Missing),
    \+ interpretation(lineage_op(S, Missing), _, _).

% The two rules are mutually exclusive on estimand: exactly one fires.
test(estimand_selects_exactly_one_rule) :-
    subject(S),
    findall(R, interpretation(lineage_op(S, [estimand(hop_decay)]), _, R), Rs),
    Rs == [lineage_as_hop_decay_v1].

:- end_tests(interpretation_rules).

%% ============================================
%% §7.3 Every option is consumed — THE REFUSALS
%% ============================================
%
% Both worked examples from §7.3 appear here verbatim.

:- begin_tests(option_consumption_refusals).

% §7.3 example 1, verbatim:
%   lineage_op(principal_tree(pearltrees), estimand='structural_score',
%              decay=0.85)
%   # Error: decay is not consumed by structural_score.
test(spec_example_unconsumed_field_rejected) :-
    subject(S),
    structural_options(Base),
    append(Base, [decay(real_value("0.85"))], Opts),
    \+ interpretation(lineage_op(S, Opts), _, _),
    resolve(lineage_op(S, Opts), unique, no_candidates).

% §7.3 example 2, verbatim:
%   lineage_op(..., estimand='hop_decay', decay=0.85, decay=0.5)
%   # Error: duplicate field.
test(spec_example_duplicate_field_rejected) :-
    subject(S),
    Opts = [estimand(hop_decay), decay(real_value("0.85")), decay(real_value("0.5"))],
    \+ interpretation(lineage_op(S, Opts), _, _),
    resolve(lineage_op(S, Opts), unique, no_candidates).

% A duplicate of the SAME value is still a duplicate — "exactly once"
% is about occurrences, not about disagreement.
test(duplicate_with_identical_value_still_rejected) :-
    subject(S),
    Opts = [estimand(hop_decay), decay(real_value("0.85")), decay(real_value("0.85"))],
    \+ interpretation(lineage_op(S, Opts), _, _).

% Unknown field.
test(unknown_field_rejected) :-
    subject(S),
    \+ interpretation(lineage_op(S, [estimand(hop_decay), frobnicate(1)]), _, _).

% Misspelled field: `decy` is not `decay`, and the rule must not
% silently default the real one and carry the typo along.
test(misspelled_field_rejected) :-
    subject(S),
    \+ interpretation(lineage_op(S, [estimand(hop_decay), decy(real_value("0.5"))]), _, _).

% Inapplicable field: hop_decay's own fields offered to
% structural_score.
test(inapplicable_field_rejected) :-
    subject(S),
    structural_options(Base),
    append(Base, [hop_origin(int_value("1"))], Opts),
    \+ interpretation(lineage_op(S, Opts), _, _).

% A duplicate estimand rejects rather than letting one occurrence pick
% the rule.
test(duplicate_estimand_rejected) :-
    subject(S),
    \+ interpretation(lineage_op(S, [estimand(hop_decay), estimand(structural_score)]), _, _).

:- end_tests(option_consumption_refusals).

%% ============================================
%% §7.1 helpers, directly
%% ============================================

:- begin_tests(option_helpers).

test(option_exact_requires_the_value) :-
    option_exact(estimand(hop_decay), [estimand(hop_decay)]),
    \+ option_exact(estimand(hop_decay), [estimand(structural_score)]).

test(option_exact_rejects_duplicates) :-
    \+ option_exact(estimand(hop_decay), [estimand(hop_decay), estimand(hop_decay)]).

test(option_or_default_three_ways) :-
    option_or_default(decay, [], d, V1), V1 == d,
    option_or_default(decay, [decay(x)], d, V2), V2 == x,
    \+ option_or_default(decay, [decay(x), decay(y)], d, _).

test(required_options_binds_and_demands) :-
    required_options([a(1), b(2)], [a(A), b(B)]),
    A == 1, B == 2,
    \+ required_options([a(1)], [a(_), b(_)]).

% "extract_options/3 transfers, rather than discards, representation
% fields" — and the result does not depend on supplied order.
test(extract_options_transfers_and_is_order_stable) :-
    extract_options([impl(x), decay(1)], [impl], T1), T1 == [impl(x)],
    extract_options([decay(1), impl(x)], [impl], T2), T2 == [impl(x)],
    extract_options([decay(1)], [impl], T3), T3 == [].

% consume_only is an EXACT partition: not a subset check either way.
test(consume_only_exact_partition) :-
    consume_only([estimand(e), impl(i)], [estimand], [impl(i)]),
    % transferred option not actually supplied
    \+ consume_only([estimand(e)], [estimand], [impl(i)]),
    % supplied option in neither part
    \+ consume_only([estimand(e), stray(1)], [estimand], []),
    % a key claimed by BOTH parts is not "exactly once"
    \+ consume_only([impl(i)], [impl], [impl(i)]).

:- end_tests(option_helpers).

%% ============================================
%% §7.2 representation/4
%% ============================================

:- begin_tests(representation_rules).

% "Absence of impl leaves all otherwise compatible implementations as
% candidates" — all three of the spec's example forms.
test(absent_impl_leaves_all_candidates) :-
    subject(S),
    candidates(lineage_op(S, [estimand(hop_decay)]), Groups),
    length(Groups, 3).

% "...it never creates an implicit deployable default": three
% candidates means `unique` FAILS rather than one being chosen.
test(absent_impl_creates_no_deployable_default) :-
    subject(S),
    resolve(lineage_op(S, [estimand(hop_decay)]), unique, not_unique(3)).

% A supplied impl selects exactly one.
test(supplied_impl_selects_one, [forall(member(I-Form,
        [ graph_walk-graph_walk_hop_decay,
          materialized_table-materialized_hop_decay_table,
          compiled_lookup-compiled_hop_decay_lookup ]))]) :-
    subject(S),
    resolve(lineage_op(S, [estimand(hop_decay), impl(atom(I))]), unique,
            selected(Precise, Rules)),
    compound_name_arity(Precise, Form, 1),
    length(Rules, 2).

% "Unknown impl yields no_candidates, and unique therefore fails."
test(unknown_impl_yields_no_candidates) :-
    subject(S),
    resolve(lineage_op(S, [estimand(hop_decay), impl(atom(no_such_impl))]),
            unique, no_candidates).

% impl is consumed exactly once here: a duplicate impl rejects.
test(duplicate_impl_rejected) :-
    subject(S),
    resolve(lineage_op(S, [estimand(hop_decay),
                           impl(atom(graph_walk)), impl(atom(graph_walk))]),
            unique, no_candidates).

% An unrecognised representation constraint rejects rather than riding
% along unconsumed.
test(unknown_representation_constraint_rejected) :-
    \+ representation(hop_decay_targets(x, []), [frobnicate(1)], _, _).

% Bare and atom(_) are the same spelling of an impl; a str/1 is not,
% because impl is enumerated configuration rather than free text.
test(impl_spelling_bare_or_atom_but_not_str) :-
    representation(hop_decay_targets(x, []), [impl(graph_walk)], _, _),
    representation(hop_decay_targets(x, []), [impl(atom(graph_walk))], _, _),
    \+ representation(hop_decay_targets(x, []), [impl(str("graph_walk"))], _, _).

% structural_score has no representation rule yet, so it resolves to
% no_candidates rather than to something invented.
test(structural_score_has_no_representation_yet) :-
    subject(S),
    structural_options(Opts),
    resolve(lineage_op(S, Opts), unique, no_candidates).

:- end_tests(representation_rules).

%% ============================================
%% §8.1 collection / §8.2 unique
%% ============================================

:- begin_tests(collection_and_unique).

% "group identical candidate ASTs and retain every sorted deriving
% rule ID" — two rule ids per candidate here (one interpretation, one
% representation), sorted.
test(groups_retain_sorted_deriving_rule_ids) :-
    subject(S),
    candidates(lineage_op(S, [estimand(hop_decay), impl(atom(graph_walk))]),
               [candidate(_, Rules)]),
    Rules == [hop_decay_as_graph_walk_v1, lineage_as_hop_decay_v1],
    msort(Rules, Rules).

% Group ordering is deterministic across runs.
test(group_order_is_deterministic) :-
    subject(S),
    candidates(lineage_op(S, [estimand(hop_decay)]), G1),
    candidates(lineage_op(S, [estimand(hop_decay)]), G2),
    G1 == G2.

% §8.2: unique succeeds on exactly one, and the two failure modes are
% distinguishable rather than both being bare failure.
test(unique_distinguishes_zero_from_many) :-
    subject(S),
    resolve(lineage_op(S, [estimand(hop_decay), impl(atom(nope))]), unique, no_candidates),
    resolve(lineage_op(S, [estimand(hop_decay)]), unique, not_unique(3)).

% The family-spec path reaches candidates too (the whole point of the
% expansion), and its requests are fully defaulted.
test(family_expansion_produces_candidates) :-
    subject(S),
    candidates(lineage_op(S, [family_spec(lineage_interpretations_v1)]), Groups),
    length(Groups, 3).

:- end_tests(collection_and_unique).

%% ============================================
%% Receipts — partial by construction
%% ============================================

:- begin_tests(receipts).

% §7.1: "The family-spec content digest ... appears in the receipt."
test(family_spec_digest_appears_in_receipt) :-
    subject(S),
    resolution_receipt(lineage_op(S, [family_spec(lineage_interpretations_v1)]),
                       all, R),
    D = R.dispatch,
    D.kind == family_spec,
    D.family_spec == lineage_interpretations_v1,
    atom_length(D.family_spec_sha256, 64).

% The receipt ENUMERATES what it omits, so a consumer sees the gap
% rather than receiving something that looks complete.
test(receipt_enumerates_its_omissions) :-
    subject(S),
    resolution_receipt(lineage_op(S, [estimand(hop_decay), impl(atom(graph_walk))]),
                       unique, R),
    memberchk(typed_ast_sha256, R.omitted_requires_frozen_canonical_encoding),
    memberchk(candidate_set_sha256, R.omitted_requires_frozen_canonical_encoding),
    memberchk(resolver_version, R.omitted_artifact_absent).

% No identity digest is present under any key: the ladder stops before
% the factory stage, and this is the test that says so.
test(receipt_mints_no_identity) :-
    subject(S),
    resolution_receipt(lineage_op(S, [estimand(hop_decay), impl(atom(graph_walk))]),
                       unique, R),
    \+ get_dict(typed_ast_sha256, R, _),
    \+ get_dict(candidate_set_sha256, R, _),
    \+ get_dict(source_ast_sha256, R, _),
    \+ get_dict(selected_typed_ast_sha256, R, _).

% An underconstrained dispatch still yields a receipt, carrying the
% refusal rather than an empty candidate list.
test(underconstrained_receipt_carries_the_reason) :-
    subject(S),
    resolution_receipt(lineage_op(S, []), unique, R),
    D = R.dispatch,
    D.kind == underconstrained,
    R.outcome = underconstrained(_).

:- end_tests(receipts).

%% ============================================
%% Rider: declared walk shapes reach the mirror
%% ============================================

:- begin_tests(walk_shape_rider).

% The generator now emits the DECLARED shape, so a consumer can read
% it instead of inferring it from the walk's name.
test(declared_walk_shapes_present) :-
    pe_walk_shape(sibling, palindromic),
    pe_walk_shape(cousin, palindromic).

test(shape_vocabulary_includes_unused_value) :-
    pe_walk_shape_kind(palindromic),
    pe_walk_shape_kind(non_palindromic).      % declared, not yet used

test(weight_vocabulary_present) :-
    pe_weight_value(uniform),
    pe_weight_value(idf_node_size).

% Every declared walk's shape is a declared shape kind — the check
% that makes reading the declaration meaningful.
test(every_walk_shape_is_a_declared_kind) :-
    forall(pe_walk_shape(_, Shape), pe_walk_shape_kind(Shape)).

:- end_tests(walk_shape_rider).
