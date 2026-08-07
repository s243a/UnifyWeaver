:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% pe_interpret.pl - interpretation/3 and representation/4:
% §7 of DESIGN_process_expression_patterns.md, made into a
% checked-in Prolog module (ruling 4(b)).
%
% The spec says of its own §7.1 clauses: "These clauses illustrate the
% relation; they are not claims about a checked-in Prolog module."
% This module is that claim.  Where §7 determines behaviour it is
% transcribed; where §7 underdetermines it, the gap is NAMED here and
% recorded in REPORT_pattern_stache_interpretation.md rather than
% ruled from inside this lane.
%
% ============================================================
% WHAT THIS DOES NOT DO
% ============================================================
% Nothing here mints an identity.  The §0 ladder runs
%   ground -> semantically explicit -> representation-resolved
%          -> factory-verified
% and this module stops at the third rung: it produces candidate
% precise ASTs and a PARTIAL receipt.  It does not compute
% `typed_ast_sha256`, `candidate_set_sha256`, `source_ast_sha256`, or
% `selected_typed_ast_sha256` — every one of those requires a frozen
% canonical encoding for semantic-request ASTs, and no such encoding
% exists (see the receipt section, and the report's underdetermination
% list).  The one digest this module DOES compute is the family-spec
% content digest, which §7.1 requires in the receipt and which is a
% RULESET-DEPENDENCY digest — the same category as the registry
% mirror's source hash, not a process identity.
%
% ============================================================
% PURITY (§7.3)
% ============================================================
% "Rules may use only a deterministic pure allowlist.  They may not
% depend on I/O, dynamic database state, clocks, random global state,
% cuts, clause order, or unbounded search."
%
% Accordingly: interpretation/3 and representation/4 clauses contain
% NO cuts and do not depend on their own order — every derivation is
% collected and the resolver groups them (§8.1's "setof is useful
% notation but is not the normative algorithm").  Helper predicates
% use if-then-else for determinism, which commits within the helper
% and never across rule selection; that is the reading of "may not
% depend on cuts" this module takes, and it is stated so a reviewer
% can disagree with it explicitly.

:- module(pe_interpret, [
    % §7.1 dispatcher pre-pass
    dispatch/2,                  % +Abstract, -Outcome
    % §7.1 / §7.2 relations
    interpretation/3,            % ?Abstract, ?Semantic, ?RuleId
    representation/4,            % ?Semantic, ?Constraints, ?Precise, ?RuleId
    % §7.1 option helpers (exported for testing; they are the spec's)
    option_exact/2,
    option_or_default/4,
    required_options/2,
    extract_options/3,
    consume_only/3,
    % §8.1 collection / §8.2 unique
    resolve/3,                   % +Abstract, +Selector, -Outcome
    candidates/2,                % +Abstract, -Groups
    % receipts (partial by construction — see header)
    resolution_receipt/3,        % +Abstract, +Selector, -Receipt
    family_spec_digest/2,        % -SpecId is unused; -HexDigest
    resource_cap/1,
    % the spec spells subjects `S::substrate(C)`; exporting the
    % operator keeps that spelling available to consumers rather than
    % making each one redeclare it
    op(200, xfx, ::)
]).

:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(sha)).

% The spec spells subjects `S::substrate(C)`.  A local operator keeps
% that spelling rather than paraphrasing it.
:- op(200, xfx, ::).

%% resource_cap(-N)
%  §8.1: "collect all derivations under a finite resource cap" and
%  "exhausting the resource cap is an error, not a partial candidate
%  set."  Refuse above it; never truncate.
resource_cap(1000).

%% ============================================
%% §7.1 THE DISPATCHER PRE-PASS
%% ============================================
%
% "Before interpretation/3 runs, the dispatcher handles absent
% estimands:
%   - explicit estimand=... goes directly to the matching clauses;
%   - absent estimand plus a content-bound family_spec expands to the
%     finite, fully defaulted explicit requests listed by that
%     specification; and
%   - absent estimand and absent family_spec is underconstrained, not
%     an implicit enumeration."

%% dispatch(+Abstract, -Outcome)
%  Outcome is one of:
%    explicit(Requests)              - Requests is a list of Abstracts
%    family(SpecId, Digest, Requests)
%    underconstrained(Reason)
dispatch(lineage_op(Subject, Options), Outcome) :-
    option_count(estimand, Options, NEst),
    option_count(family_spec, Options, NFam),
    dispatch_(NEst, NFam, Subject, Options, Outcome).

% explicit estimand: straight through, one request, unchanged.
dispatch_(NEst, _, Subject, Options, explicit([lineage_op(Subject, Options)])) :-
    NEst >= 1.
% absent estimand, one family_spec: expand.
dispatch_(0, 1, Subject, Options, Outcome) :-
    option_or_fail(family_spec, Options, SpecTerm),
    spec_id(SpecTerm, SpecId),
    (   family_spec_requests(SpecId, RequestOptionLists)
    ->  family_spec_digest(_, Digest),
        maplist(expand_request(Subject, Options), RequestOptionLists, Requests),
        Outcome = family(SpecId, Digest, Requests)
    ;   Outcome = underconstrained(unknown_family_spec(SpecId))
    ).
% absent both: refuse.  NOT an implicit enumeration.
dispatch_(0, 0, _, _, underconstrained(no_estimand_and_no_family_spec)).
% more than one family_spec is a duplicate field (§7.3).
dispatch_(0, NFam, _, _, underconstrained(duplicate_family_spec(NFam))) :-
    NFam >= 2.

% The expanded request carries the spec's explicit options and drops
% the family_spec field itself (it has been consumed by expansion).
expand_request(Subject, Options, RequestOptions, lineage_op(Subject, Merged)) :-
    exclude(has_key(family_spec), Options, Rest),
    append(Rest, RequestOptions, Merged).

spec_id(family_spec(Id), Id).

%% ============================================
%% §7.1 OPTION HELPERS
%% ============================================
%
% Options are a LIST OF OCCURRENCES — unary Key(Value) terms, possibly
% with repeated keys.  §7.1 speaks of "the supplied option
% occurrences" and §7.3 rejects duplicates, so the list (not a map) is
% the faithful representation: a map would make duplicates
% unrepresentable and silently satisfy a rule the spec says must fail.

option_key(Opt, Key) :-
    compound(Opt),
    compound_name_arity(Opt, Key, 1).

has_key(Key, Opt) :- option_key(Opt, Key).

option_count(Key, Options, N) :-
    include(has_key(Key), Options, Matching),
    length(Matching, N).

option_or_fail(Key, Options, Opt) :-
    include(has_key(Key), Options, [Opt]).

%% option_exact(+OptionTerm, +Options)
%  The option occurs, with exactly this value, exactly once.  Two
%  occurrences of the key make the rule inapplicable rather than
%  letting one silently win.
option_exact(OptionTerm, Options) :-
    option_key(OptionTerm, Key),
    include(has_key(Key), Options, [Found]),
    Found == OptionTerm.

%% option_or_default(+Key, +Options, +Default, -Value)
%  Exactly one occurrence -> its value; none -> Default; two or more
%  -> fail (a duplicate has no single value to take).
option_or_default(Key, Options, Default, Value) :-
    include(has_key(Key), Options, Matching),
    (   Matching == []
    ->  Value = Default
    ;   Matching = [One],
        arg(1, One, Value)
    ).

%% required_options(+Options, +Required)
%  Each Key(Var) in Required must occur exactly once in Options; the
%  variable binds to the supplied value.  No defaults: absence is
%  rejection.
required_options(_, []).
required_options(Options, [Req|Rest]) :-
    option_key(Req, Key),
    include(has_key(Key), Options, [Found]),
    arg(1, Found, Value),
    arg(1, Req, Value),
    required_options(Options, Rest).

%% extract_options(+Options, +Keys, -Transferred)
%  TRANSFERS, rather than discards, the representation fields (§7.1's
%  own wording).  Sorted by key so the result does not depend on the
%  order options were supplied in.
extract_options(Options, Keys, Transferred) :-
    include(key_in(Keys), Options, Selected),
    msort(Selected, Transferred).

key_in(Keys, Opt) :-
    option_key(Opt, Key),
    memberchk(Key, Keys).

%% consume_only(+Options, +SemanticKeys, +Transferred)
%  §7.1: "verifies that the semantic keys and transferred
%  representation options form an exact partition of the supplied
%  option occurrences."  §7.3: "Unknown, duplicate, misspelled, or
%  inapplicable fields reject the candidate."
%
%  Four conditions, each corresponding to one word of §7.3:
%    - no key occurs twice                       (duplicate)
%    - every transferred option was supplied     (partition soundness)
%    - semantic and transferred keys are disjoint (exactly once)
%    - every supplied key is in one part or the other
%      (unknown / misspelled / inapplicable)
consume_only(Options, SemanticKeys, Transferred) :-
    maplist(option_key, Options, SuppliedKeys),
    no_duplicate_keys(SuppliedKeys),
    maplist(option_key, Transferred, TKeys),
    forall(member(K, TKeys), memberchk(K, SuppliedKeys)),
    forall(member(K, TKeys), \+ memberchk(K, SemanticKeys)),
    forall(member(K, SuppliedKeys),
           (   memberchk(K, SemanticKeys)
           ;   memberchk(K, TKeys)
           )).

no_duplicate_keys(Keys) :-
    msort(Keys, Sorted),
    sort(Keys, Unique),
    length(Sorted, N),
    length(Unique, N).

%% ============================================
%% §7.1 INTERPRETATION — the two illustrative rules
%% ============================================
%
% Transcribed from the spec.  No cuts; order-independent.

interpretation(
    lineage_op(S::substrate(_C), Options),
    semantic_request(
        hop_decay_targets(
            S,
            [ decay(D),
              hop_origin(H),
              direction(Dir),
              depth(Limit)
            ]),
        RepresentationOptions),
    lineage_as_hop_decay_v1
) :-
    option_exact(estimand(hop_decay), Options),
    option_or_default(decay, Options, real_value("0.85"), D),
    option_or_default(hop_origin, Options, int_value("1"), H),
    option_or_default(direction, Options, ancestor, Dir),
    option_or_default(depth, Options, unbounded_depth, Limit),
    extract_options(Options, [impl], RepresentationOptions),
    consume_only(
        Options,
        [estimand, decay, hop_origin, direction, depth],
        RepresentationOptions).

interpretation(
    lineage_op(S::substrate(_C), Options),
    semantic_request(
        structural_lineage_score(
            S,
            [ floor(Floor),
              gamma(Gamma),
              distance(Distance),
              shared_prefix(Prefix),
              normalization(Norm),
              argument_roles(Roles),
              support(Support),
              unreachable(Unreachable),
              numeric(Numeric)
            ]),
        RepresentationOptions),
    lineage_as_structural_score_v1
) :-
    option_exact(estimand(structural_score), Options),
    required_options(
        Options,
        [ floor(Floor),
          gamma(Gamma),
          distance(Distance),
          shared_prefix(Prefix),
          normalization(Norm),
          argument_roles(Roles),
          support(Support),
          unreachable(Unreachable),
          numeric(Numeric)
        ]),
    extract_options(Options, [impl], RepresentationOptions),
    consume_only(
        Options,
        [estimand, floor, gamma, distance, shared_prefix,
         normalization, argument_roles, support, unreachable, numeric],
        RepresentationOptions).

%% ============================================
%% §7.2 REPRESENTATION
%% ============================================
%
% "representation(Semantic, RepresentationConstraints, Precise,
% RuleId).  The second argument is exactly
% SemanticGroundAST.representation_constraints.  `impl` is consumed
% here, exactly once.  Absence of `impl` leaves all otherwise
% compatible implementations as candidates; it never creates an
% implicit deployable default.  Unknown `impl` yields `no_candidates`,
% and `unique` therefore fails."
%
% The three precise forms are the spec's own examples.

representation(
    hop_decay_targets(S, Params),
    Constraints,
    graph_walk_hop_decay(hop_decay_targets(S, Params)),
    hop_decay_as_graph_walk_v1
) :-
    impl_admits(Constraints, graph_walk).

representation(
    hop_decay_targets(S, Params),
    Constraints,
    materialized_hop_decay_table(hop_decay_targets(S, Params)),
    hop_decay_as_materialized_table_v1
) :-
    impl_admits(Constraints, materialized_table).

representation(
    hop_decay_targets(S, Params),
    Constraints,
    compiled_hop_decay_lookup(hop_decay_targets(S, Params)),
    hop_decay_as_compiled_lookup_v1
) :-
    impl_admits(Constraints, compiled_lookup).

%% impl_admits(+Constraints, +ThisImpl)
%  Consumes `impl` exactly once.  Absent impl admits every compatible
%  implementation — deliberately NOT a default, which is why the
%  resolver can then return several candidates and `unique` fails
%  rather than one being silently picked.  A supplied impl admits only
%  its own rule, so an UNKNOWN impl admits none and the resolver
%  reports no_candidates.
impl_admits(Constraints, ThisImpl) :-
    consume_representation(Constraints, [impl]),
    include(has_key(impl), Constraints, Matching),
    (   Matching == []
    ->  true                        % all compatible candidates
    ;   Matching = [impl(Value)],
        impl_value(Value, ThisImpl)
    ).

% The Prolog adapter distinguishes ref/1, atom/1, str/1 (§7.1); an
% impl arrives as one of those or bare.  Bare and atom(_) are accepted
% spellings of the same thing; str/1 is not, because `impl` is
% enumerated configuration, not free text.
impl_value(atom(V), V).
impl_value(V, V) :- atom(V).

%% consume_representation(+Constraints, +AllowedKeys)
%  Every representation constraint must be consumed here; an
%  unrecognised one rejects the candidate rather than riding along.
consume_representation(Constraints, AllowedKeys) :-
    maplist(option_key, Constraints, Keys),
    no_duplicate_keys(Keys),
    forall(member(K, Keys), memberchk(K, AllowedKeys)).

%% ============================================
%% §8.1 CANDIDATE COLLECTION, §8.2 unique
%% ============================================

%% candidates(+Abstract, -Groups)
%  Groups is a list of candidate(Precise, SortedRuleIds) — §8.1 step
%  6: "group identical candidate ASTs and retain every sorted deriving
%  rule ID".  Sorted by standard order of the ground candidate term.
%
%  UNDERDETERMINED, recorded not ruled: §8.1 step 7 says "sort the
%  groups by canonical typed-AST bytes", and step 8 hashes that
%  ordered structure.  No frozen canonical encoding exists for these
%  semantic-request ASTs, so this uses standard term order as a
%  STAND-IN and computes NO hash.  The ordering is therefore stable
%  and deterministic but is not the spec's ordering, and nothing
%  downstream may treat it as one.
candidates(Abstract, Groups) :-
    dispatch(Abstract, Outcome),
    candidates_for(Outcome, Groups).

candidates_for(underconstrained(Reason), underconstrained(Reason)).
candidates_for(explicit(Requests), Groups) :-
    derive_groups(Requests, Groups).
candidates_for(family(_SpecId, _Digest, Requests), Groups) :-
    derive_groups(Requests, Groups).

derive_groups(Requests, Groups) :-
    findall(Precise-RuleIds,
            ( member(Req, Requests),
              interpretation(Req, semantic_request(Sem, Constraints), IRule),
              representation(Sem, Constraints, Precise, RRule),
              RuleIds = [IRule, RRule]
            ),
            Derivations),
    length(Derivations, N),
    resource_cap(Cap),
    (   N > Cap
    ->  throw(error(pe_interpret(resource_cap_exceeded(Cap, N)), _))
    ;   true
    ),
    (   Derivations == []
    ->  Groups = no_candidates
    ;   group_derivations(Derivations, Groups)
    ).

% Group identical ASTs, union their deriving rule ids (sorted), then
% order the groups.
group_derivations(Derivations, Groups) :-
    msort(Derivations, Sorted),
    group_runs(Sorted, Groups).

group_runs([], []).
group_runs([Ast-Rules|Rest], [candidate(Ast, AllRules)|Groups]) :-
    same_ast_prefix(Ast, Rest, MoreRules, Tail),
    append(Rules, MoreRules, Combined),
    sort(Combined, AllRules),
    group_runs(Tail, Groups).

same_ast_prefix(Ast, [A2-R2|Rest], Rules, Tail) :-
    A2 == Ast,
    same_ast_prefix(Ast, Rest, More, Tail),
    append(R2, More, Rules).
same_ast_prefix(Ast, [A2-R2|Rest], [], [A2-R2|Rest]) :-
    A2 \== Ast.
same_ast_prefix(_, [], [], []).

%% resolve(+Abstract, +Selector, -Outcome)
%  §8.2: "`unique` succeeds only when the normalized candidate set
%  contains exactly one AST.  Zero and multiple candidates fail."
%  Failure here is reported as a named outcome rather than a silent
%  `fail`, so a caller can tell "no candidates" from "several".
resolve(Abstract, Selector, Outcome) :-
    candidates(Abstract, Groups),
    resolve_(Groups, Selector, Outcome).

resolve_(underconstrained(R), _, underconstrained(R)).
resolve_(no_candidates, _, no_candidates).
resolve_(Groups, unique, Outcome) :-
    is_list(Groups),
    (   Groups = [candidate(Ast, Rules)]
    ->  Outcome = selected(Ast, Rules)
    ;   length(Groups, N),
        Outcome = not_unique(N)
    ).
resolve_(Groups, all, candidates(Groups)) :-
    is_list(Groups).

%% ============================================
%% FAMILY SPECS (read as data) AND THEIR DIGEST
%% ============================================

:- dynamic family_spec_cache/2.
:- dynamic family_spec_digest_cache/1.

family_spec_file(Path) :-
    module_property(pe_interpret, file(Here)),
    file_directory_name(Here, Dir),
    atomic_list_concat([Dir, '/FAMILY_SPECS_pattern_stache.pl'], Path).

load_family_specs :-
    (   family_spec_digest_cache(_)
    ->  true
    ;   family_spec_file(Path),
        read_file_to_string(Path, S, [encoding(octet)]),
        sha_hash(S, H, [algorithm(sha256), encoding(octet)]),
        hash_atom(H, Digest),
        assertz(family_spec_digest_cache(Digest)),
        setup_call_cleanup(
            open(Path, read, Stream, [encoding(utf8)]),
            read_specs(Stream),
            close(Stream))
    ).

read_specs(Stream) :-
    read_term(Stream, T, []),
    (   T == end_of_file
    ->  true
    ;   (   T = family_spec(Id, Requests)
        ->  assertz(family_spec_cache(Id, Requests))
        ;   true
        ),
        read_specs(Stream)
    ).

family_spec_requests(SpecId, Requests) :-
    load_family_specs,
    family_spec_cache(SpecId, Requests).

%% family_spec_digest(-SpecId, -HexDigest)
%  §7.1: "The family-spec content digest is a transitive ruleset
%  dependency and appears in the receipt."  This is a RULESET
%  dependency digest — the same category as the registry mirror's
%  source hash — and explicitly NOT a process identity.  SpecId is
%  unbound: the digest covers the whole ruleset file, which is what
%  "transitive ruleset dependency" means.
family_spec_digest(_SpecId, Digest) :-
    load_family_specs,
    family_spec_digest_cache(Digest).

%% ============================================
%% RECEIPTS — PARTIAL BY CONSTRUCTION
%% ============================================
%
% §9 specifies a ResolutionReceipt with ~15 version+sha256 pairs and
% four typed-AST digests.  This module can honestly fill only a
% subset, and REFUSES to invent the rest:
%
%   filled    stage, ruleset digest (this file's own bytes),
%             family_spec digest when the dispatcher expanded one,
%             candidates with deriving_rule_ids, selector
%   OMITTED   source_ast_sha256, typed_ast_sha256,
%             candidate_set_sha256, selected_typed_ast_sha256 — all
%             require a frozen canonical encoding for semantic-request
%             ASTs, which does not exist (§15 assigns identifiers
%             "only when their checked-in content is frozen")
%   OMITTED   type_system / resolver / prolog_adapter /
%             prolog_engine / prolog_conformance_profile /
%             pure_allowlist / resource_policy versions and hashes —
%             none of these artifacts exists in this lane
%
% The omissions are ENUMERATED in the receipt itself, so a consumer
% sees what is missing instead of receiving a receipt that looks
% complete.  A receipt with invented digests would be worse than no
% receipt: it would mint identities the ladder says this stage must
% not mint.
resolution_receipt(Abstract, Selector, Receipt) :-
    resolve(Abstract, Selector, Outcome),
    dispatch(Abstract, Dispatch),
    ruleset_digest(RulesetDigest),
    dispatch_receipt_fields(Dispatch, DispatchFields),
    Receipt = receipt{
        schema: 'pattern_stache.interpretation-receipt.partial.v0',
        stage: interpretation_and_representation,
        ruleset_sha256: RulesetDigest,
        dispatch: DispatchFields,
        outcome: Outcome,
        selector: Selector,
        omitted_requires_frozen_canonical_encoding:
            [ source_ast_sha256, typed_ast_sha256,
              candidate_set_sha256, selected_typed_ast_sha256 ],
        omitted_artifact_absent:
            [ type_system_version, resolver_version,
              prolog_adapter_version, prolog_engine_id,
              prolog_conformance_profile_version,
              pure_allowlist_version, resource_policy_version ]
    }.

dispatch_receipt_fields(family(SpecId, Digest, _),
                        dispatch{ kind: family_spec,
                                  family_spec: SpecId,
                                  family_spec_sha256: Digest }).
dispatch_receipt_fields(explicit(_), dispatch{ kind: explicit_estimand }).
dispatch_receipt_fields(underconstrained(R),
                        dispatch{ kind: underconstrained, reason: R }).

%% ruleset_digest(-Hex)
%  This module's own bytes: the ruleset whose rules derived the
%  candidates.  Again a dependency digest, not an identity.
ruleset_digest(Digest) :-
    module_property(pe_interpret, file(Path)),
    read_file_to_string(Path, S, [encoding(octet)]),
    sha_hash(S, H, [algorithm(sha256), encoding(octet)]),
    hash_atom(H, Digest).
