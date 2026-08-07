% SPDX-License-Identifier: MIT OR Apache-2.0
%
% FAMILY_SPECS_pattern_stache.pl - family specifications for the §7.1
% dispatcher pre-pass.  READ AS DATA (read_term/2), never consulted as
% code.
%
% §7.1 of DESIGN_process_expression_patterns.md requires that an
% absent estimand plus a content-bound `family_spec` expand to "the
% finite, fully defaulted explicit requests listed by that
% specification", and that "the family-spec content digest is a
% transitive ruleset dependency and appears in the receipt".
%
% ============================================================
% THIS FILE IS A PROTOTYPE STAND-IN, AND SAYS SO
% ============================================================
% The spec NAMES `lineage_interpretations_v1` but never lists its
% contents, so the contents below are this prototype's construction,
% not a transcription.  They are marked stand-in wherever they are
% consumed, and the underdetermination is recorded in the report
% rather than silently resolved.
%
% ============================================================
% A TENSION §7.1 LEAVES OPEN (recorded, not ruled)
% ============================================================
% "Fully defaulted explicit requests" and §7.1's own
% `required_options` rule pull against each other: an interpretation
% whose fields are REQUIRED (as `lineage_as_structural_score_v1`'s
% nine are) cannot be fully defaulted, so a family spec can only
% expand to it by SUPPLYING those values itself.  Both readings are
% available and the spec does not choose:
%   (a) a family spec lists only interpretations that are fully
%       defaultable — structural_score is then not family-expandable;
%   (b) a family spec may carry explicit values for required fields,
%       making the request "fully defaulted" in the sense of
%       "nothing left to supply".
% This file exercises BOTH so neither is foreclosed: the `v1` spec
% takes reading (a), the `v1_with_values` spec takes reading (b).
% Whichever the owner rules, one of these becomes wrong and the tests
% that pin it will say so.

family_spec_version(1).

%% family_spec(SpecId, RequestOptionsList)
%  Each element is a fully explicit option list — an estimand plus
%  whatever that estimand's rule needs beyond its defaults.

% Reading (a): only the fully-defaultable interpretation.
family_spec(lineage_interpretations_v1,
            [ [estimand(hop_decay)] ]).

% Reading (b): required fields supplied by the spec itself.
family_spec(lineage_interpretations_v1_with_values,
            [ [estimand(hop_decay)],
              [ estimand(structural_score),
                floor(real_value("0.02")),
                gamma(real_value("0.6")),
                distance(hops),
                shared_prefix(lca),
                normalization(none),
                argument_roles(ordered),
                support(all_pairs),
                unreachable(zero),
                numeric(exact_decimal)
              ] ]).
