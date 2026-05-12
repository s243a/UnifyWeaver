:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% cost_function.pl — Target-agnostic registry + validator for the
% pluggable cost-function strategy slot.
%
% Cost functions score nodes for relevance during warm-build's tree
% construction (see SCAN_STRATEGY_SPECIFICATION.md). Three are
% registered:
%
%   - hop_distance         — BFS distance from the nearest endpoint;
%                            cheap, no per-graph tuning, real
%                            implementation in P1.
%   - flux                 — branching-factor-aware decay from
%                            endpoints with separate parent/child
%                            legs; panic stub in P1, real in P3+.
%   - semantic_similarity  — dot-product against a query embedding;
%                            panic stub in P1, requires embeddings to
%                            be useful.
%
% Each cost function has a parameter schema and a defaults list. The
% target adapter (e.g. wam_haskell_target.pl) consults this module to
% validate the workload's tree_cost_function/2 declaration, fill
% defaults, and then emits target-specific code (the Haskell CostFn
% record for the WAM-Haskell target; future targets will emit their
% own equivalents).
%
% This module is target-agnostic. It does NOT emit code — that's the
% target adapter's job.

:- module(cost_function, [
    %% Registry
    cost_function_name/1,                  % ?Name
    cost_function_param_schema/2,          % +Name, -Schema

    %% Validation
    validate_cost_function/1,              % +Term — throws on invalid
    is_cost_function_term/1,               % +Term — semidet

    %% Defaults
    cost_function_default_params/2,        % +Name, -DefaultParams
    cost_function_with_defaults/2          % +Term0, -Term1
]).

:- use_module(library(error), [must_be/2]).
:- use_module(library(lists)).

%% =====================================================================
%% Registry
%% =====================================================================
%
% Each schema entry is `param_spec(Key, Type, Optionality)` where
%   - Key is the param-term functor (atom)
%   - Type is integer, positive_integer, float, or atom
%   - Optionality is `required` or `default(Value)`
%
% Adding a new cost function: append a new cost_function_entry/2 clause
% with its param specs. Adding a new param to an existing function:
% extend its schema list.

cost_function_entry(hop_distance, [
    param_spec(max_hops, positive_integer, default(5))
]).

cost_function_entry(flux, [
    param_spec(iterations,   positive_integer, default(1)),
    param_spec(parent_decay, float,            default(0.5)),
    param_spec(child_decay,  float,            default(0.3)),
    param_spec(flux_merge,   atom,             default(sum))
]).

cost_function_entry(semantic_similarity, [
    param_spec(dim,            positive_integer, default(128)),
    param_spec(embedding_path, atom,             required)
]).

%% =====================================================================
%! cost_function_name(?Name) is nondet.
%% =====================================================================

cost_function_name(Name) :-
    cost_function_entry(Name, _).

%% =====================================================================
%! cost_function_param_schema(+Name, -Schema) is det.
%% =====================================================================

cost_function_param_schema(Name, Schema) :-
    must_be(atom, Name),
    (   cost_function_entry(Name, Schema)
    ->  true
    ;   throw(error(domain_error(cost_function_name, Name),
                    context(cost_function_param_schema/2,
                            'no such cost function registered')))
    ).

%% =====================================================================
%! is_cost_function_term(+Term) is semidet.
%
%  Shape check only — does NOT validate parameter contents.
%% =====================================================================

is_cost_function_term(tree_cost_function(Name, Params)) :-
    atom(Name),
    is_list(Params),
    cost_function_entry(Name, _).

%% =====================================================================
%! validate_cost_function(+Term) is det.
%
%  Throws if Term is not a valid `tree_cost_function(Name, Params)`:
%
%    - Term must have the shape tree_cost_function(Name, Params)
%    - Name must be a registered cost function
%    - Params must be a list
%    - Each schema entry marked `required` must be present in Params
%    - Each provided param must match its schema type
%
%  Unknown param keys (not in the schema) are silently accepted —
%  future-compat for newer-param-on-older-codegen flow.
%% =====================================================================

validate_cost_function(Term) :-
    (   nonvar(Term),
        Term = tree_cost_function(Name, Params)
    ->  true
    ;   throw(error(domain_error(cost_function_term, Term),
                    context(validate_cost_function/1,
                            'expected tree_cost_function(Name, Params)')))
    ),
    must_be(atom, Name),
    must_be(list, Params),
    cost_function_param_schema(Name, Schema),
    validate_required_params(Schema, Params, Name),
    validate_param_types(Schema, Params, Name).

validate_required_params([], _, _).
validate_required_params([param_spec(Key, _Type, required) | Rest], Params, Name) :-
    !,
    (   provided_param_key(Key, Params)
    ->  validate_required_params(Rest, Params, Name)
    ;   throw(error(domain_error(required_param, Key),
                    context(validate_cost_function/1,
                            cost_function_missing_required(Name, Key))))
    ).
validate_required_params([_ | Rest], Params, Name) :-
    validate_required_params(Rest, Params, Name).

provided_param_key(Key, Params) :-
    member(Term, Params),
    nonvar(Term),
    functor(Term, Key, 1).

validate_param_types([], _, _).
validate_param_types([param_spec(Key, Type, _) | Rest], Params, Name) :-
    Probe =.. [Key, Value],
    (   member(Probe, Params)
    ->  validate_param_type(Type, Value, Key, Name)
    ;   true
    ),
    validate_param_types(Rest, Params, Name).

validate_param_type(integer,          V, _, _) :- integer(V), !.
validate_param_type(positive_integer, V, _, _) :- integer(V), V > 0, !.
validate_param_type(float,            V, _, _) :- number(V), !.
validate_param_type(atom,             V, _, _) :- atom(V), !.
validate_param_type(Type, Value, Key, Name) :-
    throw(error(type_error(Type, Value),
                context(validate_cost_function/1,
                        cost_function_param(Name, Key)))).

%% =====================================================================
%! cost_function_default_params(+Name, -Defaults) is det.
%
%  Default-bearing params from Name's schema as an option list.
%  Required-without-default entries are omitted.
%% =====================================================================

cost_function_default_params(Name, Defaults) :-
    cost_function_param_schema(Name, Schema),
    findall(Term,
            (member(param_spec(Key, _Type, default(Value)), Schema),
             Term =.. [Key, Value]),
            Defaults).

%% =====================================================================
%! cost_function_with_defaults(+Term0, -Term1) is det.
%
%  Fill in missing optional parameters with their defaults. Caller-
%  provided params are kept (no override). Validates Term0 first.
%% =====================================================================

cost_function_with_defaults(tree_cost_function(Name, ProvidedParams),
                            tree_cost_function(Name, FullParams)) :-
    validate_cost_function(tree_cost_function(Name, ProvidedParams)),
    cost_function_default_params(Name, Defaults),
    %% Caller-provided params win — they come first in the concat.
    append(ProvidedParams, Defaults, Combined),
    %% De-duplicate by functor name so the output doesn't carry both.
    dedupe_by_functor(Combined, FullParams).

dedupe_by_functor([], []).
dedupe_by_functor([T | Rest], [T | Out]) :-
    functor(T, K, A),
    exclude(same_functor(K, A), Rest, Filtered),
    dedupe_by_functor(Filtered, Out).

same_functor(K, A, T) :-
    functor(T, K, A).
