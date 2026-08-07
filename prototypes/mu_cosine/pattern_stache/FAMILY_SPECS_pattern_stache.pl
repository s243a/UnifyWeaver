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
% RULED: WHAT "FULLY DEFAULTED" MEANS
% ============================================================
% This file previously carried TWO specs to exercise two available
% readings of "fully defaulted", because §7.1 chose neither and the
% lane does not rule on the spec's behalf.  The reading is now RULED:
%
%   "Fully defaulted" means FULLY GROUNDED BY THE SPEC'S OWN CONTENT,
%   not by registry defaults alone.  A family-spec entry must expand
%   to a COMPLETE explicit request.
%
% Two consequences, both load-bearing:
%
%   - An estimand whose fields all carry defaults may be listed BARE.
%     `hop_decay` is such an estimand, and is listed bare below.
%   - An estimand with REQUIRED fields — `structural_score`'s nine —
%     may appear in a spec ONLY with every required value carried
%     explicitly in the spec entry itself, so the content digest
%     covers them.
%
% This keeps the expansion finite and total: `required_options` can
% never fire against a spec-expanded request, because a spec entry
% that would trip it is malformed.  `test_pe_interpret.pl` enforces
% exactly that as a ruleset well-formedness property, so a future
% entry that violates the ruling fails a test rather than producing a
% quietly empty candidate set.
%
% Consequence for `lineage_interpretations_v1`, ruled explicitly: it
% contains EXACTLY the bare `hop_decay` entry.  `structural_score`
% enters only via a future spec VERSION that grounds all nine fields —
% which is a new content digest, as it should be, since a spec that
% expands to different requests is a different ruleset dependency.
%
% ============================================================
% STILL A PROTOTYPE STAND-IN
% ============================================================
% §7.1 names `lineage_interpretations_v1` but never lists its
% contents, so the *entry* below remains this prototype's
% construction.  What is no longer open is the RULE the entry has to
% satisfy.

family_spec_version(1).

%% family_spec(SpecId, RequestOptionsList)
%  Each element is a complete explicit option list: an estimand plus
%  every value that estimand's rule cannot default for itself.

% hop_decay defaults all four of its fields, so the bare entry is
% already a complete request.
family_spec(lineage_interpretations_v1,
            [ [estimand(hop_decay)] ]).
