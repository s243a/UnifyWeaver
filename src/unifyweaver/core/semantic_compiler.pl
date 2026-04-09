:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025-2026 John William Creighton (@s243a)
%
% semantic_compiler.pl - Generic Semantic Search Compilation Interface
%
% This module provides a unified way to define and compile semantic search
% predicates across multiple targets with support for fallback providers.

:- module(semantic_compiler, [
    declare_semantic_provider/2,     % +Predicate, +Options
    get_semantic_provider/3,         % +Target, +Predicate, -ProviderInfo
    is_semantic_predicate/1,         % +Goal
    compile_semantic_call/4,         % +Target, +Goal, +VarMap, -Code

    % Options-aware compilation (semantic_search/4)
    extract_search_options/2,        % +Goal, -SearchOptions
    merge_provider_options/3,        % +ProviderInfo, +SearchOptions, -Merged

    % Edge weight precomputation (semantic → distance kernel bridge)
    compile_edge_weights/4,          % +Target, +EdgePred, +Model, -Code

    % Fuzzy logic compilation
    is_fuzzy_predicate/1,            % +Goal
    compile_fuzzy_call/3             % +Target, +Goal, -Code
]).

:- use_module(library(lists)).
:- use_module(library(option)).

% ============================================================================
% DYNAMIC STORAGE
% ============================================================================

:- dynamic stored_semantic_provider/2. % stored_semantic_provider(Predicate, Options)
:- dynamic user_providers_loaded/0.

% ============================================================================
% CORE API
% ============================================================================

%% load_user_providers
%
%  Discover and register semantic providers declared in the user module
%  via semantic_provider/2. Runs at most once per session.
%
load_user_providers :-
    user_providers_loaded, !.
load_user_providers :-
    (   current_predicate(user:semantic_provider/2)
    ->  forall(user:semantic_provider(Pred/Arity, Options),
               declare_semantic_provider(Pred/Arity, Options))
    ;   true
    ),
    assertz(user_providers_loaded).

%% declare_semantic_provider(+Predicate, +Options)
%
%  Register a semantic search provider configuration.
%
%  Options:
%    - targets(TargetList) where TargetList is a list of:
%        target(Name, [provider(P), model(M), device(D), ...])
%    - fallback(ProviderName)
%
declare_semantic_provider(Pred/Arity, Options) :-
    atom(Pred), integer(Arity),
    retractall(stored_semantic_provider(Pred/Arity, _)),
    assertz(stored_semantic_provider(Pred/Arity, Options)).

%% get_semantic_provider(+Target, +Predicate, -ProviderInfo)
%
%  Find the best provider for a target.
%
get_semantic_provider(Target, Pred/Arity, ProviderInfo) :-
    load_user_providers,
    stored_semantic_provider(Pred/Arity, Options),
    member(targets(Targets), Options),
    member(target(Target, ProviderInfo), Targets),
    !.
get_semantic_provider(_Target, Pred/Arity, ProviderInfo) :-
    stored_semantic_provider(Pred/Arity, Options),
    member(fallback(ProviderInfo), Options),
    !.

%% is_semantic_predicate(+Goal)
%
%  True if the goal is registered as a semantic predicate.
%
is_semantic_predicate(Goal) :-
    load_user_providers,
    functor(Goal, Pred, Arity),
    stored_semantic_provider(Pred/Arity, _).

%% compile_semantic_call(+Target, +Goal, +VarMap, -Code)
%
%  Dispatch compilation to target-specific semantic handler.
%  Supports both Pred/3 and Pred/4 (with inline options).
%  For Pred/4, options are merged with provider config before dispatch.
%
compile_semantic_call(Target, Goal, VarMap, Code) :-
    functor(Goal, Pred, Arity),
    % Try exact arity first, then fall back to base arity (e.g., /4 → /3)
    (   get_semantic_provider(Target, Pred/Arity, BaseProvider)
    ->  true
    ;   Arity > 3,
        BaseArity is Arity - 1,
        get_semantic_provider(Target, Pred/BaseArity, BaseProvider)
    ),
    % Merge inline options from Goal (if present) with provider config
    extract_search_options(Goal, SearchOpts),
    merge_provider_options(BaseProvider, SearchOpts, MergedProvider),
    semantic_dispatch(Target, Goal, MergedProvider, VarMap, Code).

%% extract_search_options(+Goal, -Options)
%
%  Extract inline search options from the Goal.
%  For Pred/4 goals, the 4th argument is the options list.
%  For Pred/3 goals (or when 4th arg is unbound), returns [].
%
extract_search_options(Goal, Options) :-
    Goal =.. [_ | Args],
    length(Args, Len),
    Len >= 4,
    nth1(4, Args, Opts),
    is_list(Opts),
    !,
    Options = Opts.
extract_search_options(_, []).

%% merge_provider_options(+ProviderInfo, +SearchOptions, -Merged)
%
%  Merge inline search options with the provider configuration.
%  Inline options override provider defaults.
%  Supported inline options:
%    threshold(T)  — minimum similarity score for results
%    model(M)      — override the provider model name
%    index(Path)   — override the vector database path
%    top_k(K)      — override result count
%
merge_provider_options(ProviderInfo, [], ProviderInfo) :- !.
merge_provider_options(ProviderInfo, SearchOpts, Merged) :-
    % Override model if specified in search options
    (   member(model(M), SearchOpts)
    ->  select(model(_), ProviderInfo, P1),
        Merged0 = [model(M) | P1]
    ;   Merged0 = ProviderInfo
    ),
    % Add threshold if specified (not typically in provider config)
    (   member(threshold(T), SearchOpts)
    ->  Merged1 = [threshold(T) | Merged0]
    ;   Merged1 = Merged0
    ),
    % Add index path if specified
    (   member(index(Path), SearchOpts)
    ->  Merged = [index(Path) | Merged1]
    ;   Merged = Merged1
    ).

% ============================================================================
% TARGET DISPATCH (Multifile hooks)
% ============================================================================

:- multifile semantic_dispatch/5.
% semantic_dispatch(+Target, +Goal, +ProviderInfo, +VarMap, -Code)

% ============================================================================
% SEMANTIC EDGE WEIGHT PRECOMPUTATION
% ============================================================================

%% compile_edge_weights(+Target, +EdgePred, +EmbeddingModel, -Code)
%
%  Generate target-language code to precompute semantic edge weights.
%  For each edge in EdgePred/2, computes:
%    weight = 1 - cosine_similarity(embedding(From), embedding(To))
%  and stores as weighted edge facts for the shortest path kernel.
%
%  This bridges the semantic interface (embeddings) with the distance
%  kernel (weighted edges) — edge weights are semantic distances.
%
:- multifile compile_edge_weights/4.
% compile_edge_weights(+Target, +EdgePred, +EmbeddingModel, -Code)

%% Python edge weight precomputation
compile_edge_weights(python, EdgePred, Model, Code) :-
    format(string(Code),
'# Precompute semantic edge weights from embeddings
from sentence_transformers import SentenceTransformer
import numpy as np

_model = SentenceTransformer("~w")

def precompute_edge_weights(edges):
    """Compute semantic distance for each edge: 1 - cosine_sim(emb(a), emb(b))"""
    nodes = list(set([n for e in edges for n in e]))
    embeddings = {n: _model.encode(n, convert_to_numpy=True) for n in nodes}
    weights = {}
    for a, b in edges:
        sim = np.dot(embeddings[a], embeddings[b]) / (
            np.linalg.norm(embeddings[a]) * np.linalg.norm(embeddings[b]) + 1e-9)
        weights[(a, b)] = 1.0 - float(sim)
    return weights

_edge_weights = precompute_edge_weights(~w_edges)
', [Model, EdgePred]).

%% Go edge weight precomputation
compile_edge_weights(go, _EdgePred, Model, Code) :-
    format(string(Code),
'\t// Precompute semantic edge weights from embeddings
\temb, err := embedder.NewHugotEmbedder("models/~w-onnx", "~w")
\tif err != nil { log.Fatal(err) }
\tdefer emb.Close()
\t
\tfunc precomputeEdgeWeights(emb *embedder.HugotEmbedder, edges [][2]string) map[[2]string]float64 {
\t\tembCache := make(map[string][]float64)
\t\tweights := make(map[[2]string]float64)
\t\tfor _, edge := range edges {
\t\t\tfor _, node := range edge {
\t\t\t\tif _, ok := embCache[node]; !ok {
\t\t\t\t\tv, _ := emb.Embed(node)
\t\t\t\t\tembCache[node] = v
\t\t\t\t}
\t\t\t}
\t\t\tsim := cosineSim(embCache[edge[0]], embCache[edge[1]])
\t\t\tweights[edge] = 1.0 - sim
\t\t}
\t\treturn weights
\t}
', [Model, Model]).

% ============================================================================
% FUZZY LOGIC COMPILATION
% ============================================================================

%% is_fuzzy_predicate(+Goal)
%
%  True if the goal is a fuzzy logic operation that can be compiled
%  via fuzzy_dispatch/4.
%
is_fuzzy_predicate(Goal) :-
    functor(Goal, Name, _),
    fuzzy_op(Name).

fuzzy_op(f_and).
fuzzy_op(f_or).
fuzzy_op(f_dist_or).
fuzzy_op(f_union).
fuzzy_op(f_not).
fuzzy_op(eval_fuzzy_expr).
fuzzy_op(blend_scores).
fuzzy_op(multiply_scores).
fuzzy_op(top_k).
fuzzy_op(apply_filter).
fuzzy_op(apply_boost).

%% compile_fuzzy_call(+Target, +Goal, -Code)
%
%  Dispatch compilation of a fuzzy logic goal to the target-specific handler.
%
compile_fuzzy_call(Target, Goal, Code) :-
    fuzzy_dispatch(Target, Goal, Code).

:- multifile fuzzy_dispatch/3.
% fuzzy_dispatch(+Target, +Goal, -Code)
