:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% purity_certificate.pl — Shared purity / order-independence verdicts
%
% Target-agnostic module that produces certificates of the form
%
%     purity_cert(Verdict, Proof, Confidence, Reasons)
%
% where:
%   Verdict    = pure | impure(ReasonList) | unknown
%   Proof      = declared | analyzed(Strategy) | certified(Source) | inferred
%   Confidence = float in [0.0, 1.0]
%   Reasons    = list of atom   (culprits for impure, evidence for pure)
%
% See docs/design/PURITY_CERTIFICATE_{PROPOSAL,SPECIFICATION,IMPLEMENTATION_PLAN}.md
% for the full design. This file is the Phase P1 deliverable:
%   - Certificate shape and public API.
%   - Two producers: user annotations (highest priority) and
%     builtin blacklist.
%   - Registration API for future producers (kernel registry in P3,
%     whitelist absorption in P2).
%   - Merge API for multi-producer resolution.
%
% Intentionally NOT in Phase P1:
%   - Whitelist strategy absorption (Phase P2).
%   - Kernel registry producer (Phase P3).
%   - Haskell WAM ParTryMeElse wiring (Phase P4).
%   - JSON serialization (Phase P5).
%
% Back-compat: clause_body_analysis:is_pure_goal/1 and
% is_order_independent/2 become thin wrappers that call this module
% internally. External callers see identical behavior.

:- module(purity_certificate, [
    % Core data shape — exported as a documentation marker, not a
    % predicate (purity_cert/4 terms are constructed positionally).
    purity_cert_shape/0,

    % Primary producer API
    analyze_goal_purity/2,            % +Goal, -Cert
    analyze_predicate_purity/2,       % +PredIndicator, -Cert
    analyze_clause_purity/2,          % +Head-Body, -Cert

    % Convenience predicates
    is_certified_pure/1,              % +PredIndicator
    is_certified_impure/2,            % +PredIndicator, -Reasons
    purity_confidence/2,              % +PredIndicator, -Confidence

    % Multi-producer resolution
    merge_certificates/2,             % +CertList, -Merged

    % Producer registration (extension point for P2, P3)
    register_purity_producer/2,       % +ProducerSpec, +Priority
    registered_producers/1,           % -ProducerSpecs (sorted by priority desc)

    % Built-in impurity catalogue — exposed so other modules can extend
    impurity_class/2,                 % ?Goal, ?ReasonAtom

    % Strict whitelist (for tail-recursion transformation and other
    % consumers that refuse to trust the permissive blacklist).
    pure_builtin/1,                   % ?Functor/Arity
    is_whitelist_pure_goal/1,         % +Goal

    % Contradiction detection — runs every producer and reports any
    % disagreement between a higher-priority `pure` verdict and a
    % lower-priority `impure` verdict. Pure (no side effects by
    % default); callers decide how loud to be about findings.
    check_purity_contradictions/2,    % +PredIndicator, -Contradictions
    warn_purity_contradictions/1,     % +PredIndicator — prints to user_error

    % Phase P5: JSON serialization for cross-language interop
    cert_to_json/2,                   % +Cert, -JsonDict
    cert_to_json/3,                   % +Cert, +Metadata, -JsonDict
    cert_from_json/2                  % +JsonDict, -Cert
]).

:- use_module(library(lists)).

% ============================================================================
% CERTIFICATE SHAPE (documentation marker)
% ============================================================================

%% purity_cert_shape
%  Always true. Documents the compound term shape used throughout:
%
%      purity_cert(Verdict, Proof, Confidence, Reasons)
%
%  Defining a predicate for "this shape exists" lets ?- help/1 find
%  the documentation even though the term is constructed positionally.
purity_cert_shape.

% ============================================================================
% PRODUCER REGISTRY
% ============================================================================

:- dynamic producer_registration/3.
% producer_registration(Name, Priority, Analyzer)
%   Name     — atom identifying the producer
%   Priority — integer; higher = checked first
%   Analyzer — pred:(+Subject, +SubjectKind, -Cert)
%     SubjectKind = goal | clause | predicate

%% register_purity_producer(+ProducerSpec, +Priority)
%  Register a producer. Idempotent — re-registering with the same
%  Name replaces the old entry.
%
%  ProducerSpec = purity_producer(Name, Analyzer, SupportedKinds)
register_purity_producer(purity_producer(Name, Analyzer, _SupportedKinds), Priority) :-
    retractall(producer_registration(Name, _, _)),
    assertz(producer_registration(Name, Priority, Analyzer)).

%% registered_producers(-Specs)
%  Return registered producers sorted by priority descending.
registered_producers(Specs) :-
    findall(Priority-Name-Analyzer,
            producer_registration(Name, Priority, Analyzer),
            Raw),
    sort(0, @>=, Raw, Sorted),
    maplist([P-N-A, producer_spec(N, P, A)]>>true, Sorted, Specs).

% ============================================================================
% PUBLIC API
% ============================================================================

%% analyze_goal_purity(+Goal, -Cert)
%  Certify a single goal. Walks producers in priority order. First
%  producer that returns a non-unknown verdict wins. If none match,
%  returns unknown.
analyze_goal_purity(Goal, Cert) :-
    registered_producers(Specs),
    first_matching_cert(Specs, Goal, goal, Cert0),
    !,
    Cert = Cert0.
analyze_goal_purity(_, purity_cert(unknown, inferred, 0.0, [no_producer_matched])).

%% analyze_predicate_purity(+PredIndicator, -Cert)
%  Certify a predicate by analyzing its clauses. PredIndicator is
%  Module:Name/Arity or bare Name/Arity (user: is assumed).
%
%  Resolution order (per spec §2.1):
%    1. User annotation (:- parallel/1 or :- order_independent/1) → declared
%    2. (Phase P3) Kernel registry membership → certified(kernel_registry)
%    3. Clause-body analysis over every clause → analyzed(clause_walk)
%    4. Otherwise → unknown
analyze_predicate_purity(PredIndicator, Cert) :-
    normalize_pred_indicator(PredIndicator, Normalized),
    registered_producers(Specs),
    first_matching_cert(Specs, Normalized, predicate, Cert0),
    !,
    Cert = Cert0.
analyze_predicate_purity(_, purity_cert(unknown, inferred, 0.0, [no_producer_matched])).

%% analyze_clause_purity(+Head-Body, -Cert)
%  Certify a single clause. Body-level — walks goals, aggregates
%  verdicts. Used internally by clause_walk_analyzer.
analyze_clause_purity(Head-Body, Cert) :-
    normalize_body_goals(Body, Goals),
    maplist(analyze_goal_purity, Goals, GoalCerts),
    % If any goal is impure, clause is impure.
    ( memberchk(purity_cert(impure(_), _, _, _), GoalCerts)
    -> aggregate_impure_reasons(GoalCerts, Reasons),
       Cert = purity_cert(impure(Reasons), analyzed(clause_walk), 0.9, Reasons)
    ; memberchk(purity_cert(unknown, _, _, _), GoalCerts)
    -> collect_unknown_reasons(GoalCerts, Reasons),
       Cert = purity_cert(unknown, analyzed(clause_walk), 0.5, Reasons)
    ; Cert = purity_cert(pure, analyzed(clause_walk), 0.9, [blacklist_clean])
    ),
    _ = Head. % suppress unused-var warning; Head kept for future extensions

% ----- first_matching_cert -----
first_matching_cert([producer_spec(_Name, _Pri, Analyzer)|Rest], Subject, Kind, Cert) :-
    ( catch(call(Analyzer, Subject, Kind, Try), _, fail),
      Try \= purity_cert(unknown, _, _, _)
    -> Cert = Try
    ;  first_matching_cert(Rest, Subject, Kind, Cert)
    ).
first_matching_cert([], _, _, purity_cert(unknown, inferred, 0.0, [no_producer_matched])).

% ============================================================================
% CONVENIENCE PREDICATES
% ============================================================================

is_certified_pure(PredIndicator) :-
    analyze_predicate_purity(PredIndicator,
                             purity_cert(pure, _, _, _)).

is_certified_impure(PredIndicator, Reasons) :-
    analyze_predicate_purity(PredIndicator,
                             purity_cert(impure(Reasons), _, _, _)).

purity_confidence(PredIndicator, Confidence) :-
    analyze_predicate_purity(PredIndicator,
                             purity_cert(_, _, Confidence, _)).

% ============================================================================
% CONTRADICTION DETECTION
% ============================================================================

%% check_purity_contradictions(+PredIndicator, -Contradictions)
%  Runs every registered producer (not just first-match) and reports
%  disagreement. A contradiction is emitted when a higher-priority
%  producer says `pure` but a lower-priority producer finds `impure`.
%
%  Why this exists: the default `analyze_predicate_purity/2` picks
%  the highest-priority non-unknown verdict, which means a user
%  annotation (priority 100) can mask an impure body that the
%  blacklist (priority 50) would flag. This predicate surfaces the
%  discrepancy so callers can warn, error, or log — without the
%  module itself deciding which is "right" (the user may have good
%  reason to assert purity despite an impure-looking body, e.g.
%  memoization via assertz that's semantically transparent).
%
%  Contradictions are reported as terms of the form:
%      contradiction(HighSource, HighVerdict, LowSource, LowVerdict)
%  where HighSource / LowSource are producer names and HighVerdict /
%  LowVerdict are the full purity_cert terms from each.
%
%  No side effects — this predicate is pure. Callers that want a
%  warning printed should use warn_purity_contradictions/1.
check_purity_contradictions(PredIndicator, Contradictions) :-
    normalize_pred_indicator(PredIndicator, Normalized),
    registered_producers(Specs),
    producer_verdicts(Specs, Normalized, predicate, Verdicts),
    % A contradiction is a higher-priority pure paired with a
    % lower-priority impure. Walk the priority-ordered list and
    % collect such pairs.
    find_contradictions(Verdicts, Contradictions).

producer_verdicts([], _, _, []).
producer_verdicts([producer_spec(Name, _Pri, Analyzer)|Rest],
                  Subject, Kind,
                  [verdict(Name, Cert)|RestOut]) :-
    ( catch(call(Analyzer, Subject, Kind, Cert), _, fail)
    -> true
    ; Cert = purity_cert(unknown, inferred, 0.0, [producer_errored])
    ),
    producer_verdicts(Rest, Subject, Kind, RestOut).

% find_contradictions: for each pair of verdicts where the higher-priority
% one is `pure` and the lower-priority one is `impure`, emit a contradiction.
find_contradictions([], []).
find_contradictions([verdict(HiName, HiCert)|Rest], Out) :-
    HiCert = purity_cert(pure, _, _, _),
    !,
    findall(contradiction(HiName, HiCert, LoName, LoCert),
            ( member(verdict(LoName, LoCert), Rest),
              LoCert = purity_cert(impure(_), _, _, _)
            ),
            Here),
    find_contradictions(Rest, More),
    append(Here, More, Out).
find_contradictions([_|Rest], Out) :-
    find_contradictions(Rest, Out).

%% warn_purity_contradictions(+PredIndicator)
%  Opt-in convenience over check_purity_contradictions/2 — prints each
%  contradiction to user_error. Always succeeds; prints nothing if
%  there are no contradictions.
%
%  Suitable for compiler pipelines that want to surface user-declared
%  purity that looks wrong under static analysis. Unlike a hard
%  failure, this only warns — the user's declaration is still
%  trusted by analyze_predicate_purity/2.
warn_purity_contradictions(PredIndicator) :-
    check_purity_contradictions(PredIndicator, Contradictions),
    forall(member(C, Contradictions),
           print_contradiction(PredIndicator, C)).

print_contradiction(Pred,
        contradiction(HiName, _HiCert, LoName,
                      purity_cert(impure(Reasons), _, _, _))) :-
    format(user_error,
           "~N[purity_certificate] ~w: ~w says pure but ~w finds impure: ~w~n",
           [Pred, HiName, LoName, Reasons]).

% ============================================================================
% MERGE
% ============================================================================

%% merge_certificates(+Certs, -Merged)
%  Combine multiple certificates for the same subject.
%  Resolution (per spec §2.4):
%    - Any impure → impure (short-circuit; conservative).
%    - All pure → pure with max(Confidence) and merged Reasons.
%    - Mixed pure + unknown → pure if any producer has Confidence >= 0.9,
%      else unknown.
merge_certificates([], purity_cert(unknown, inferred, 0.0, [empty_merge])) :- !.
merge_certificates([Single], Single) :- !.
merge_certificates(Certs, Merged) :-
    % Any impure wins (short-circuit).
    ( member(purity_cert(impure(Rs), P, C, R), Certs)
    -> Merged = purity_cert(impure(Rs), P, C, R)
    ; pures_and_unknowns(Certs, Pures, Unknowns),
      ( Pures = [_|_]
      -> max_pure_cert(Pures, HighPure),
         HighPure = purity_cert(pure, HP, HC, _),
         collect_reasons(Pures, PureReasons),
         ( Unknowns == []
         -> Merged = purity_cert(pure, HP, HC, PureReasons)
         ;  HC >= 0.9
         -> Merged = purity_cert(pure, HP, HC, PureReasons)
         ;  Merged = purity_cert(unknown, inferred, HC, PureReasons)
         )
      ;  Unknowns = [purity_cert(unknown, UP, UC, UR)|_]
      -> Merged = purity_cert(unknown, UP, UC, UR)
      ;  Merged = purity_cert(unknown, inferred, 0.0, [merge_fallthrough])
      )
    ),
    !.

pures_and_unknowns([], [], []).
pures_and_unknowns([C|Rest], [C|PR], UR) :-
    C = purity_cert(pure, _, _, _), !,
    pures_and_unknowns(Rest, PR, UR).
pures_and_unknowns([C|Rest], PR, [C|UR]) :-
    C = purity_cert(unknown, _, _, _), !,
    pures_and_unknowns(Rest, PR, UR).
pures_and_unknowns([_|Rest], PR, UR) :-
    pures_and_unknowns(Rest, PR, UR).

max_pure_cert([C], C) :- !.
max_pure_cert([C1|Rest], Best) :-
    max_pure_cert(Rest, Best0),
    C1 = purity_cert(pure, _, C1Conf, _),
    Best0 = purity_cert(pure, _, Best0Conf, _),
    ( C1Conf >= Best0Conf -> Best = C1 ; Best = Best0 ),
    !.

collect_reasons(Certs, Reasons) :-
    findall(R,
            ( member(purity_cert(_, _, _, Rs), Certs),
              member(R, Rs)
            ),
            All),
    sort(All, Reasons).

% ============================================================================
% PRODUCER 1: USER ANNOTATIONS
% ============================================================================

%% user_annotation_analyzer(+Subject, +Kind, -Cert)
%  Consults clause_body_analysis:order_independent/1 and
%  parallel_safe/1 dynamic facts (and their user:-prefixed variants).
user_annotation_analyzer(Pred/Arity, predicate, Cert) :-
    nonvar(Pred), integer(Arity),
    !,
    user_annotation_analyzer(user:Pred/Arity, predicate, Cert).
user_annotation_analyzer(Module:Pred/Arity, predicate, Cert) :-
    nonvar(Module),
    !,
    ( has_annotation(Module:Pred/Arity, parallel)
    -> Cert = purity_cert(pure, declared, 1.0,
                          [declared_by_user, parallel])
    ; has_annotation(Module:Pred/Arity, order_independent)
    -> Cert = purity_cert(pure, declared, 1.0,
                          [declared_by_user, order_independent])
    ; Cert = purity_cert(unknown, inferred, 0.0, [no_annotation])
    ).
user_annotation_analyzer(Goal, goal, Cert) :-
    callable(Goal),
    !,
    functor(Goal, F, A),
    ( has_parallel_safe(F/A)
    -> Cert = purity_cert(pure, declared, 1.0,
                          [declared_by_user, parallel_safe])
    ; Cert = purity_cert(unknown, inferred, 0.0, [no_annotation])
    ).
user_annotation_analyzer(_, _, purity_cert(unknown, inferred, 0.0, [no_annotation])).

has_annotation(Module:Pred/Arity, parallel) :-
    catch(clause_body_analysis:order_independent(Module:Pred/Arity), _, fail), !.
has_annotation(_:Pred/Arity, parallel) :-
    catch(clause_body_analysis:order_independent(Pred/Arity), _, fail).
has_annotation(Module:Pred/Arity, order_independent) :-
    catch(clause_body_analysis:order_independent(Module:Pred/Arity), _, fail), !.
has_annotation(_:Pred/Arity, order_independent) :-
    catch(clause_body_analysis:order_independent(Pred/Arity), _, fail).

has_parallel_safe(F/A) :-
    catch(clause_body_analysis:parallel_safe(F/A), _, fail).

% ============================================================================
% PRODUCER 2: BUILTIN BLACKLIST
% ============================================================================

%% blacklist_analyzer(+Subject, +Kind, -Cert)
%  Decides purity by checking for known-impure builtins. Matches the
%  existing clause_body_analysis:is_impure_builtin/1 catalogue.
blacklist_analyzer(Goal, goal, Cert) :-
    callable(Goal),
    !,
    ( impurity_class(Goal, ReasonAtom)
    -> Cert = purity_cert(impure([ReasonAtom]), analyzed(blacklist), 0.95,
                          [ReasonAtom])
    ; Cert = purity_cert(pure, analyzed(blacklist), 0.9, [blacklist_clean])
    ).
blacklist_analyzer(Pred/Arity, predicate, Cert) :-
    nonvar(Pred), integer(Arity),
    !,
    blacklist_analyzer(user:Pred/Arity, predicate, Cert).
blacklist_analyzer(Module:Pred/Arity, predicate, Cert) :-
    nonvar(Module),
    !,
    functor(Head, Pred, Arity),
    findall(Head-Body,
            catch(clause(Module:Head, Body), _, fail),
            Clauses),
    ( Clauses == []
    -> Cert = purity_cert(unknown, analyzed(blacklist), 0.3,
                          [no_clauses_found])
    ;  maplist(analyze_clause_purity, Clauses, ClauseCerts),
       merge_certificates(ClauseCerts, Cert)
    ).
blacklist_analyzer(_, _, purity_cert(unknown, inferred, 0.0, [unsupported_subject])).

%% impurity_class(?Goal, ?Reason)
%  Catalog of impure builtins with their reason atoms. Mirrors
%  clause_body_analysis:is_impure_builtin/1 but adds reason tags
%  (io_ops, database_mods, global_state, domain_specific) so consumers
%  know *why* a predicate is impure, not just *that* it is.
% I/O Impure
impurity_class(write(_), io_ops).
impurity_class(nl, io_ops).
impurity_class(format(_,_), io_ops).
impurity_class(read(_), io_ops).
impurity_class(read_term(_,_), io_ops).
impurity_class(get_char(_), io_ops).
impurity_class(get_code(_), io_ops).
impurity_class(peek_char(_), io_ops).
impurity_class(peek_code(_), io_ops).
% Database Impure
impurity_class(assert(_), database_mods).
impurity_class(asserta(_), database_mods).
impurity_class(assertz(_), database_mods).
impurity_class(retract(_), database_mods).
impurity_class(retractall(_), database_mods).
% Global Variable Impure
impurity_class(nb_setval(_,_), global_state).
impurity_class(b_setval(_,_), global_state).
% Target-specific / Domain-specific
impurity_class(send_message(_,_), domain_specific).
impurity_class(succ_or_zero(_,_), domain_specific).

% ============================================================================
% PRODUCER 3: KERNEL REGISTRY (P3)
% ============================================================================

%% kernel_analyzer(+Subject, +Kind, -Cert)
%  Decides purity by asking recursive_kernel_detection whether the
%  predicate matches one of the hand-coded native kernel shapes
%  (category_ancestor, transitive_closure2, transitive_distance3,
%  weighted_shortest_path3, transitive_parent_distance4,
%  astar_shortest_path4, transitive_step_parent_distance5, …).
%
%  Rationale: kernels are audited-by-construction. The kernel author
%  has hand-written native code that implements the predicate with
%  no side effects, so when the detector matches the predicate's
%  clause pattern, we're entitled to a very high-confidence pure
%  verdict — higher than the syntactic blacklist, weaker only than
%  an explicit user annotation.
%
%  Registered at priority 90 (above blacklist 50, below user
%  annotations 100). This lets kernel verdicts win when the blacklist
%  would otherwise say `unknown` for a non-standard functor inside
%  the kernel's body (e.g., the kernel calls some user-defined helper
%  that the blacklist can't classify).
%
%  NOTE: invoking the detector requires actually loading the user's
%  clauses. For predicates with no clauses we return `unknown` rather
%  than false — a predicate the compiler hasn't yet seen isn't
%  provably a kernel *or* provably not one.
kernel_analyzer(Pred/Arity, predicate, Cert) :-
    nonvar(Pred), integer(Arity),
    !,
    kernel_analyzer(user:Pred/Arity, predicate, Cert).
kernel_analyzer(Module:Pred/Arity, predicate, Cert) :-
    nonvar(Module),
    !,
    functor(Head, Pred, Arity),
    findall(Head-StrippedBody,
            ( catch(clause(Module:Head, Body), _, fail),
              strip_body_modules(Body, StrippedBody)
            ),
            Clauses),
    ( Clauses == []
    -> Cert = purity_cert(unknown, inferred, 0.0, [not_a_kernel])
    ;  catch(
           recursive_kernel_detection:detect_recursive_kernel(
               Pred, Arity, Clauses, _Kernel),
           _, fail)
    -> Cert = purity_cert(pure, certified(kernel_registry), 1.0,
                          [kernel_owned])
    ;  Cert = purity_cert(unknown, inferred, 0.0, [not_a_kernel])
    ).
kernel_analyzer(_, _, purity_cert(unknown, inferred, 0.0,
                                  [kernel_unsupported_subject])).

% ============================================================================
% PRODUCER 4: STRICT BUILTIN WHITELIST
% ============================================================================

%% whitelist_analyzer(+Subject, +Kind, -Cert)
%  Strictly whitelist-based — a goal is pure only if its functor
%  appears in the pure_builtin/1 catalogue below. Returns `pure` with
%  Proof = analyzed(whitelist) when the match succeeds, else `unknown`.
%
%  Rationale: this is the analyzer tail-recursion transformation
%  wants to consult, because it refuses to transform anything it
%  can't PROVE is pure. The blacklist is permissive (absence of
%  evidence); the whitelist is strict (positive evidence only).
%
%  Registered at priority 40 — lower than the blacklist's 50, so in
%  the default first-match chain the blacklist wins for goals it can
%  decide. Consumers that specifically want whitelist semantics
%  should use is_whitelist_pure_goal/1 or call whitelist_analyzer/3
%  directly rather than relying on the chained analyze_goal_purity/2.
whitelist_analyzer(Goal, goal, Cert) :-
    callable(Goal),
    !,
    ( is_whitelist_pure_goal(Goal)
    -> Cert = purity_cert(pure, analyzed(whitelist), 0.9,
                          [whitelist_only])
    ; Cert = purity_cert(unknown, analyzed(whitelist), 0.3,
                         [not_in_whitelist])
    ).
whitelist_analyzer(_, _, purity_cert(unknown, inferred, 0.0,
                                     [whitelist_unsupported_subject])).

%% is_whitelist_pure_goal(+Goal)
%  True iff Goal's functor is in the strict whitelist. Exported for
%  consumers (advanced/purity_analysis.pl) that want a direct boolean
%  check without threading through the certificate producer chain.
is_whitelist_pure_goal(true) :- !.
is_whitelist_pure_goal(Goal) :-
    callable(Goal),
    functor(Goal, F, A),
    pure_builtin(F/A).

%% pure_builtin(?Functor/Arity)
%  Strict whitelist of pure built-in predicates. Absorbed from
%  src/unifyweaver/core/advanced/purity_analysis.pl so both the
%  tail-recursion transformation and the certificate module consult
%  a single source of truth.

% Arithmetic
pure_builtin(is/2).
pure_builtin(succ/2).
pure_builtin(plus/3).

% Comparison
pure_builtin((>)/2).
pure_builtin((<)/2).
pure_builtin((>=)/2).
pure_builtin((=<)/2).
pure_builtin((=:=)/2).
pure_builtin((=\=)/2).

% Unification
pure_builtin((=)/2).
pure_builtin((\=)/2).

% Type checks
pure_builtin(number/1).
pure_builtin(integer/1).
pure_builtin(float/1).
pure_builtin(atom/1).
pure_builtin(compound/1).
pure_builtin(is_list/1).
pure_builtin(var/1).
pure_builtin(nonvar/1).
pure_builtin(ground/1).

% Term manipulation
pure_builtin(functor/3).
pure_builtin(arg/3).
pure_builtin((=..)/2).
pure_builtin(copy_term/2).

% List operations (pure, though member/2 is nondeterministic)
pure_builtin(length/2).
pure_builtin(append/3).
pure_builtin(msort/2).
pure_builtin(sort/2).
pure_builtin(nth0/3).
pure_builtin(nth1/3).
pure_builtin(last/2).
pure_builtin(member/2).
pure_builtin(between/3).
pure_builtin(reverse/2).

% Control (pure forms)
pure_builtin(true/0).

% ============================================================================
% HELPERS
% ============================================================================

%% normalize_pred_indicator(+In, -Out)
%  Out is always Module:Pred/Arity (user: inserted if absent).
normalize_pred_indicator(Module:Pred/Arity, Module:Pred/Arity) :- !.
normalize_pred_indicator(Pred/Arity, user:Pred/Arity).

%% strip_body_modules(+Body, -Stripped)
%  Peel off any M:Goal wrappers at the top level and inside
%  conjunctions/disjunctions so downstream pattern matchers see the
%  raw structure. assertz/1 from a non-user module tags the body with
%  the calling module, which trips detectors that pattern-match on
%  `(Goal, Rest)` expecting a bare conjunction.
strip_body_modules(Var, Var) :- var(Var), !.
strip_body_modules(_M:Body, Stripped) :-
    !,
    strip_body_modules(Body, Stripped).
strip_body_modules((A, B), (SA, SB)) :-
    !,
    strip_body_modules(A, SA),
    strip_body_modules(B, SB).
strip_body_modules((A ; B), (SA ; SB)) :-
    !,
    strip_body_modules(A, SA),
    strip_body_modules(B, SB).
strip_body_modules((A -> B), (SA -> SB)) :-
    !,
    strip_body_modules(A, SA),
    strip_body_modules(B, SB).
strip_body_modules(\+ A, \+ SA) :-
    !,
    strip_body_modules(A, SA).
strip_body_modules(Goal, Goal).

%% normalize_body_goals(+Body, -Goals)
%  Flatten a clause body conjunction into a list of goals.
%  Duplicates clause_body_analysis:normalize_goals/2 to avoid a
%  circular dependency (clause_body_analysis becomes a caller of this
%  module, not the other way round).
normalize_body_goals(true, []) :- !.
normalize_body_goals((Left, Right), Goals) :-
    !,
    normalize_body_goals(Left, L),
    normalize_body_goals(Right, R),
    append(L, R, Goals).
normalize_body_goals(_Module:Goal, Goals) :- !,
    normalize_body_goals(Goal, Goals).
normalize_body_goals(Goal, [Goal]).

aggregate_impure_reasons(Certs, Reasons) :-
    findall(R,
            ( member(purity_cert(impure(Rs), _, _, _), Certs),
              member(R, Rs)
            ),
            All),
    sort(All, Reasons).

collect_unknown_reasons(Certs, Reasons) :-
    findall(R,
            ( member(purity_cert(unknown, _, _, Rs), Certs),
              member(R, Rs)
            ),
            All),
    sort(All, Reasons).

% ============================================================================
% BOOTSTRAP — register built-in producers
% ============================================================================

:- register_purity_producer(
       purity_producer(user_annotations,
                       purity_certificate:user_annotation_analyzer,
                       [goal, predicate]),
       100).
% ============================================================================
% PHASE P5: JSON SERIALIZATION
% ============================================================================

%% cert_to_json(+Cert, -JsonDict) is det.
%  Serialize a purity certificate to a SWI-Prolog dict suitable for
%  json_write_dict/2. Convenience wrapper with empty metadata.
cert_to_json(Cert, Json) :-
    cert_to_json(Cert, _{}, Json).

%% cert_to_json(+Cert, +Metadata, -JsonDict) is det.
%  Serialize with optional metadata (subject, producer, producer_version).
%  Metadata is a dict; unknown keys are passed through to the output.
cert_to_json(purity_cert(Verdict, Proof, Confidence, Reasons), Metadata, Json) :-
    verdict_to_json(Verdict, VerdictStr, ExtraReasons),
    proof_to_json(Proof, ProofDict),
    append(ExtraReasons, Reasons, AllReasons),
    maplist(reason_to_string, AllReasons, ReasonStrs),
    BaseDict = _{verdict: VerdictStr, proof: ProofDict,
                 confidence: Confidence, reasons: ReasonStrs},
    put_dict(Metadata, BaseDict, Json).

verdict_to_json(pure, "pure", []).
verdict_to_json(impure(ReasonList), "impure", ReasonList) :-
    is_list(ReasonList), !.
verdict_to_json(impure(_), "impure", []).
verdict_to_json(unknown, "unknown", []).

proof_to_json(declared, _{kind: "declared"}).
proof_to_json(analyzed(Strategy), _{kind: "analyzed", strategy: StratStr}) :-
    term_to_atom(Strategy, StratStr).
proof_to_json(certified(Source), _{kind: "certified", source: SrcStr}) :-
    term_to_atom(Source, SrcStr).
proof_to_json(inferred, _{kind: "inferred"}).

reason_to_string(R, S) :-
    (   atom(R) -> atom_string(R, S)
    ;   term_to_atom(R, A), atom_string(A, S)
    ).

%% cert_from_json(+JsonDict, -Cert) is det.
%  Deserialize a JSON dict to a purity_cert/4 term. Unknown keys in
%  the dict are silently ignored (forward-compat per spec §3.2).
cert_from_json(Json, purity_cert(Verdict, Proof, Confidence, Reasons)) :-
    (   get_dict(verdict, Json, VerdictStr) -> true
    ;   throw(error(cert_json_error(missing_verdict), _))
    ),
    (   get_dict(confidence, Json, Confidence) -> true
    ;   throw(error(cert_json_error(missing_confidence), _))
    ),
    (   number(Confidence), Confidence >= 0.0, Confidence =< 1.0 -> true
    ;   throw(error(cert_json_error(confidence_out_of_range(Confidence)), _))
    ),
    (   get_dict(proof, Json, ProofDict) -> true
    ;   throw(error(cert_json_error(missing_proof), _))
    ),
    json_to_verdict(VerdictStr, Json, Verdict),
    json_to_proof(ProofDict, Proof),
    json_to_reasons(Json, Reasons).

json_to_verdict("pure", _, pure).
json_to_verdict("impure", Json, impure(Reasons)) :-
    json_to_reasons(Json, Reasons).
json_to_verdict("unknown", _, unknown).
json_to_verdict(Other, _, _) :-
    throw(error(cert_json_error(unknown_verdict(Other)), _)).

json_to_proof(ProofDict, Proof) :-
    get_dict(kind, ProofDict, Kind),
    json_to_proof_(Kind, ProofDict, Proof).
json_to_proof_(Kind, _, _) :-
    var(Kind),
    throw(error(cert_json_error(missing_proof_kind), _)).
json_to_proof_("declared", _, declared).
json_to_proof_("analyzed", Dict, analyzed(Strategy)) :-
    (   get_dict(strategy, Dict, StratStr)
    ->  term_to_atom(Strategy, StratStr)
    ;   Strategy = unknown
    ).
json_to_proof_("certified", Dict, certified(Source)) :-
    (   get_dict(source, Dict, SrcStr)
    ->  term_to_atom(Source, SrcStr)
    ;   Source = unknown
    ).
json_to_proof_("inferred", _, inferred).
json_to_proof_(Other, _, _) :-
    \+ var(Other),
    Other \= "declared", Other \= "analyzed",
    Other \= "certified", Other \= "inferred",
    throw(error(cert_json_error(unknown_proof_kind(Other)), _)).

json_to_reasons(Json, Reasons) :-
    (   get_dict(reasons, Json, RList), is_list(RList)
    ->  maplist(string_to_reason, RList, Reasons)
    ;   Reasons = []
    ).

string_to_reason(S, R) :-
    (   atom_string(A, S), catch(term_to_atom(R, A), _, fail) -> true
    ;   atom_string(R, S)
    ).

:- register_purity_producer(
       purity_producer(kernel_registry,
                       purity_certificate:kernel_analyzer,
                       [predicate]),
       90).
:- register_purity_producer(
       purity_producer(blacklist,
                       purity_certificate:blacklist_analyzer,
                       [goal, clause, predicate]),
       50).
:- register_purity_producer(
       purity_producer(whitelist,
                       purity_certificate:whitelist_analyzer,
                       [goal]),
       40).
