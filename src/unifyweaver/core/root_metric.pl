:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% root_metric.pl — Target-agnostic recognition of root-anchored graph metrics.
%
% A root-anchored metric is a per-node value defined by a recurrence pinned at
% a root: value(Root) = boundary; value(Node) = AGGREGATE over parents of
% COMBINE(value(parent), edge_step). Minimum/maximum distance to root and the
% additive "effective distance" / flux are the three canonical instances; they
% differ only in the (aggregate, combine) semiring.
%
% This module is the Phase 0 spec surface from
% docs/design/ROOT_ANCHORED_METRICS_{PHILOSOPHY,SPECIFICATION,IMPLEMENTATION_PLAN}.md:
% it parses + validates + normalises the `root_metric/2` directive and exposes
% the normalised spec + the semiring table for targets to consume. It does NOT
% emit code (same analysis-produces-terms / targets-consume-terms pattern as
% demand_analysis.pl and purity_certificate.pl), so it is inert until a target
% reads the registered specs.
%
% Usage as a directive in user code:
%
%   :- use_module(library(root_metric)).
%   :- root_metric(min_dist_to_root/3,
%          [ edge(parent/2, up), boundary('Root', 0),
%            combine(succ), aggregate(min),
%            max_depth(10), cycles(bounded), materialize(ingest) ]).
%
% Defaults: cycles(bounded), materialize(kernel), max_depth(10). `combine(succ)`
% normalises to `combine(plus(1))`. Invalid directives throw
% root_metric_error(Reason) so a bad spec fails loudly at load time.

:- module(root_metric, [
    root_metric/2,                 % +Name/Arity, +Opts  (directive: validate + register)
    normalize_root_metric/3,       % +Name/Arity, +Opts, -Spec  (throws on invalid)
    validate_root_metric/2,        % +Name/Arity, +Opts         (semidet, no throw)
    registered_root_metric/2,      % ?Name/Arity, ?Spec         (dynamic store)
    root_metric_name/2,            % +Spec, -Name/Arity
    root_metric_edge/3,            % +Spec, -EdgePred/2, -Direction
    root_metric_boundary/3,        % +Spec, -RootTerm, -V0
    root_metric_combine/2,         % +Spec, -Combine            (normalised: plus(K)|scale(D))
    root_metric_aggregate/2,       % +Spec, -Aggregate          (min|max|sum)
    root_metric_max_depth/2,       % +Spec, -Depth
    root_metric_cycles/2,          % +Spec, -Policy
    root_metric_materialize/2,     % +Spec, -Where
    root_metric_semiring/3,        % +Aggregate, +Combine, -semiring(Op,Combine,Annihilator,Identity)
    combine_class/2,               % +Combine, -Class           (additive|multiplicative)
    root_metric_preset/2,          % ?PresetName, ?AlgorithmOpts
    default_root_metric/4          % +EdgePred/2, +Direction, +RootTerm, -Spec  (min distance)
]).

:- dynamic registered_root_metric/2.

%% ========================================================================
%% Directive entry point: validate + normalise + register.
%% ========================================================================

root_metric(NameArity, Opts) :-
    normalize_root_metric(NameArity, Opts, Spec),
    % Re-registering the same name replaces the prior spec (idempotent reload).
    retractall(registered_root_metric(NameArity, _)),
    assertz(registered_root_metric(NameArity, Spec)).

%% ========================================================================
%% Validation + normalisation.
%% ========================================================================

%% validate_root_metric(+NameArity, +Opts) is semidet.
%  True iff the directive would normalise without error. Never throws.
validate_root_metric(NameArity, Opts) :-
    catch(normalize_root_metric(NameArity, Opts, _), _, fail).

%% normalize_root_metric(+NameArity, +Opts, -Spec) is det.
%  Spec = root_metric_spec(Name/Arity, NormOpts) with defaults filled and
%  combine(succ) rewritten to combine(plus(1)). Throws root_metric_error/1.
normalize_root_metric(NameArity, Opts0, root_metric_spec(NameArity, NormOpts)) :-
    must_name_arity(NameArity),
    must_be_opt_list(Opts0),
    expand_preset(Opts0, Opts),
    % required + defaulted fields
    req_edge(Opts, Edge),
    req_boundary(Opts, Boundary),
    req_combine(Opts, Combine),
    req_aggregate(Opts, Aggregate),
    opt_default(max_depth(D), Opts, max_depth(10)), check_max_depth(D),
    opt_default(cycles(Cyc), Opts, cycles(bounded)), check_cycles(Cyc),
    opt_default(materialize(M), Opts, materialize(kernel)), check_materialize(M),
    % cross-field: the (aggregate, combine) pair must name a known semiring
    ( root_metric_semiring(Aggregate, Combine, _)
    -> true
    ;  throw(root_metric_error(unsupported_pair(Aggregate, Combine)))
    ),
    NormOpts = [ Edge, Boundary, combine(Combine), aggregate(Aggregate),
                 max_depth(D), cycles(Cyc), materialize(M) ].

must_name_arity(Name/Arity) :-
    ( atom(Name), integer(Arity), Arity >= 1
    -> true
    ;  throw(root_metric_error(bad_name_arity(Name/Arity))) ).
must_name_arity(Other) :-
    \+ Other = _/_,
    throw(root_metric_error(bad_name_arity(Other))).

must_be_opt_list(Opts) :-
    ( is_list(Opts) -> true ; throw(root_metric_error(opts_not_a_list(Opts))) ).

req_edge(Opts, edge(P/2, Dir)) :-
    ( memberchk(edge(P/2, Dir), Opts)
    -> ( atom(P), member(Dir, [up, down])
       -> true ; throw(root_metric_error(bad_edge(edge(P/2, Dir)))) )
    ;  throw(root_metric_error(missing(edge))) ).

req_boundary(Opts, boundary(Root, V0)) :-
    ( memberchk(boundary(Root, V0), Opts)
    -> ( number(V0) -> true ; throw(root_metric_error(bad_boundary_value(V0))) )
    ;  throw(root_metric_error(missing(boundary))) ).

%% combine: succ normalises to plus(1); plus(K)/scale(D) validated.
req_combine(Opts, Norm) :-
    ( memberchk(combine(C), Opts) -> true ; throw(root_metric_error(missing(combine))) ),
    normalize_combine(C, Norm).

normalize_combine(succ, plus(1)) :- !.
normalize_combine(plus(K), plus(K)) :- !,
    ( number(K) -> true ; throw(root_metric_error(bad_combine(plus(K)))) ).
normalize_combine(scale(D), scale(D)) :- !,
    ( number(D), D > 0, D =< 1 -> true ; throw(root_metric_error(bad_combine(scale(D)))) ).
normalize_combine(C, _) :- throw(root_metric_error(bad_combine(C))).

req_aggregate(Opts, A) :-
    ( memberchk(aggregate(A), Opts) -> true ; throw(root_metric_error(missing(aggregate))) ),
    ( member(A, [min, max, sum]) -> true ; throw(root_metric_error(bad_aggregate(A))) ).

check_max_depth(D) :-
    ( integer(D), D >= 0 -> true ; throw(root_metric_error(bad_max_depth(D))) ).
check_cycles(C) :-
    ( member(C, [bounded, scc, ignore, visited]) -> true ; throw(root_metric_error(bad_cycles(C))) ).
check_materialize(M) :-
    ( member(M, [kernel, ingest]) -> true ; throw(root_metric_error(bad_materialize(M))) ).

%% opt_default(+TemplateWithVar, +Opts, +DefaultTerm): bind from Opts or default.
opt_default(Template, Opts, _Default) :- memberchk(Template, Opts), !.
opt_default(Template, _Opts, Default) :- Template = Default.

%% ========================================================================
%% Presets — name the *algorithm* shape so a user declares only the
%% graph-specific edge + boundary. `min_dist` is the default metric.
%% A directive may carry `preset(Name)`; explicit opts override the preset.
%% ========================================================================

root_metric_preset(min_dist,  [ combine(succ),       aggregate(min), max_depth(10) ]).
root_metric_preset(max_dist,  [ combine(succ),       aggregate(max), max_depth(10) ]).
root_metric_preset(effective, [ combine(scale(0.2)), aggregate(sum), max_depth(10) ]).

%% default_root_metric(+EdgePred/2, +Direction, +RootTerm, -Spec)
%  The canonical zero-thought metric: minimum distance to root over the given
%  edge. Lets a caller get a usable spec without writing the full option list.
default_root_metric(EdgePred/2, Direction, RootTerm, Spec) :-
    normalize_root_metric(min_dist_to_root/3,
        [ edge(EdgePred/2, Direction), boundary(RootTerm, 0), preset(min_dist) ],
        Spec).

%% expand_preset(+Opts, -Expanded): if a preset(Name) is present, splice in its
%  fields, with any explicit option of the same functor/arity taking precedence.
expand_preset(Opts, Expanded) :-
    ( select(preset(P), Opts, Rest)
    -> ( root_metric_preset(P, Base) -> true ; throw(root_metric_error(bad_preset(P))) ),
       findall(B, (member(B, Base), \+ opt_present(B, Rest)), Adds),
       append(Rest, Adds, Expanded)
    ;  Expanded = Opts ).

opt_present(Opt, List) :- functor(Opt, F, A), functor(Probe, F, A), memberchk(Probe, List).

%% ========================================================================
%% Semiring table — the (aggregate, combine-class) pairs we support.
%% semiring(Op, Combine, Annihilator, Identity).
%% ========================================================================

combine_class(plus(_), additive).
combine_class(scale(_), multiplicative).

root_metric_semiring(min, plus(_),  semiring(min, plus,  inf,    0)).
root_metric_semiring(max, plus(_),  semiring(max, plus,  neg_inf, 0)).
root_metric_semiring(sum, scale(_), semiring(sum, scale, 0,      1)).

%% ========================================================================
%% Accessors over a normalised Spec.
%% ========================================================================

root_metric_name(root_metric_spec(NameArity, _), NameArity).
root_metric_edge(root_metric_spec(_, Opts), P/2, Dir) :- memberchk(edge(P/2, Dir), Opts).
root_metric_boundary(root_metric_spec(_, Opts), Root, V0) :- memberchk(boundary(Root, V0), Opts).
root_metric_combine(root_metric_spec(_, Opts), C) :- memberchk(combine(C), Opts).
root_metric_aggregate(root_metric_spec(_, Opts), A) :- memberchk(aggregate(A), Opts).
root_metric_max_depth(root_metric_spec(_, Opts), D) :- memberchk(max_depth(D), Opts).
root_metric_cycles(root_metric_spec(_, Opts), C) :- memberchk(cycles(C), Opts).
root_metric_materialize(root_metric_spec(_, Opts), M) :- memberchk(materialize(M), Opts).
