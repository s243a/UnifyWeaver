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
    compile_semantic_call/4          % +Target, +Goal, +VarMap, -Code
]).

:- use_module(library(lists)).
:- use_module(library(option)).

% ============================================================================
% DYNAMIC STORAGE
% ============================================================================

:- dynamic stored_semantic_provider/2. % stored_semantic_provider(Predicate, Options)

% ============================================================================
% CORE API
% ============================================================================

%% load_user_providers
%
%  Discover and register semantic providers declared in the user module
%  via semantic_provider/2.
%
load_user_providers :-
    (   current_predicate(user:semantic_provider/2)
    ->  forall(user:semantic_provider(Pred/Arity, Options),
               declare_semantic_provider(Pred/Arity, Options))
    ;   true
    ).

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
%
compile_semantic_call(Target, Goal, VarMap, Code) :-
    functor(Goal, Pred, Arity),
    get_semantic_provider(Target, Pred/Arity, ProviderInfo),
    semantic_dispatch(Target, Goal, ProviderInfo, VarMap, Code).

% ============================================================================
% TARGET DISPATCH (Multifile hooks)
% ============================================================================

:- multifile semantic_dispatch/5.
% semantic_dispatch(+Target, +Goal, +ProviderInfo, +VarMap, -Code)

% Default fallbacks for common targets if no specific dispatch is defined
semantic_dispatch(go, semantic_search(Query, TopK, _Results), Provider, VarMap, Code) :-
    option(provider(hugot), Provider),
    option(model(Model), Provider, 'all-MiniLM-L6-v2'),
    option(device(Device), Provider, auto),
    % Lookup variable name in VarMap if it's a variable
    (   member(Query=QueryVar, VarMap) -> QueryExpr = QueryVar ; QueryExpr = Query ),
    (   member(TopK=TopKVar, VarMap) -> TopKExpr = TopKVar ; TopKExpr = TopK ),

    % Device initialization for Go (hugot)
    (   Device == gpu
    ->  DeviceInit = '\temb, err := embedder.NewHugotEmbedder("models/~w-onnx", "~w", embedder.WithGPU())'
    ;   Device == cpu
    ->  DeviceInit = '\temb, err := embedder.NewHugotEmbedder("models/~w-onnx", "~w", embedder.WithCPU())'
    ;   DeviceInit = '\temb, err := embedder.NewHugotEmbedder("models/~w-onnx", "~w") // Auto device'
    ),

    format(string(Code), '
\t// Initialize hugot embedder with model ~w
~w
\tif err != nil { 
\t\tlog.Printf("Warning: GPU initialization failed, falling back to CPU: %%v", err)
\t\temb, err = embedder.NewHugotEmbedder("models/~w-onnx", "~w", embedder.WithCPU())
\t\tif err != nil { log.Fatal(err) }
\t}
\tdefer emb.Close()

\t// Embed query: ~w
\tqVec, err := emb.Embed(~w)
\tif err != nil { log.Fatal(err) }
\t
\t// Search top ~w results
\tresults, err := search.Search(store, qVec, ~w)
', [Model, DeviceInit, Model, Model, Model, Model, QueryExpr, QueryExpr, TopKExpr, TopKExpr]).

semantic_dispatch(python, semantic_search(Query, TopK, _Results), Provider, VarMap, Code) :-
    option(provider(transformers), Provider),
    option(model(Model), Provider, 'all-MiniLM-L6-v2'),
    option(device(Device), Provider, auto),
    (   member(Query=QueryVar, VarMap) -> QueryExpr = QueryVar ; QueryExpr = Query ),
    (   member(TopK=TopKVar, VarMap) -> TopKExpr = TopKVar ; TopKExpr = TopK ),

    % Device initialization for Python (transformers)
    (   Device == gpu
    ->  DeviceStr = "cuda"
    ;   Device == cpu
    ->  DeviceStr = "cpu"
    ;   DeviceStr = "None" % auto
    ),

    format(string(Code), '
import torch
from sentence_transformers import SentenceTransformer

# Initialize model with device: ~w and fallback
device = "~w"
if device == "cuda" and not torch.cuda.is_available():
    print("Warning: CUDA not available, falling back to CPU")
    device = "cpu"

model = SentenceTransformer("~w", device=device)

# Embed query: ~w (TopK: ~w)
query_emb = model.encode(~w, convert_to_numpy=True)
results = searcher.search(query_emb, top_k=~w)
', [Device, DeviceStr, Model, QueryExpr, TopKExpr, QueryExpr, TopKExpr]).

semantic_dispatch(rust, semantic_search(Query, TopK, _Results), Provider, VarMap, Code) :-
    option(provider(candle), Provider),
    option(model(Model), Provider, 'all-MiniLM-L6-v2'),
    (   member(Query=QueryVar, VarMap) -> QueryExpr = QueryVar ; QueryExpr = Query ),
    (   member(TopK=TopKVar, VarMap) -> TopKExpr = TopKVar ; TopKExpr = TopK ),
    format(string(Code), '
    // Initialize candle searcher with model ~w
    let searcher = PtSearcher::new("data.redb", "~w")?;
    let results = searcher.text_search("~w", ~w)?;
    println!("{}", serde_json::to_string_pretty(&results)?);
', [Model, Model, QueryExpr, TopKExpr]).

semantic_dispatch(rust, semantic_search(Query, TopK, _Results), Provider, VarMap, Code) :-
    option(provider(onnx), Provider),
    option(model(Model), Provider, 'all-MiniLM-L6-v2'),
    (   member(Query=QueryVar, VarMap) -> QueryExpr = QueryVar ; QueryExpr = Query ),
    (   member(TopK=TopKVar, VarMap) -> TopKExpr = TopKVar ; TopKExpr = TopK ),
    format(string(Code), '
    // Initialize ONNX searcher with model ~w
    let searcher = OnnxSearcher::new("data.redb", "models/~w-onnx")?;
    let results = searcher.text_search("~w", ~w)?;
', [Model, Model, QueryExpr, TopKExpr]).

semantic_dispatch(csharp, semantic_search(Query, TopK, _Results), Provider, VarMap, Code) :-
    option(provider(onnx), Provider),
    option(model(Model), Provider, 'all-MiniLM-L6-v2'),
    (   member(Query=QueryVar, VarMap) -> QueryExpr = QueryVar ; QueryExpr = Query ),
    (   member(TopK=TopKVar, VarMap) -> TopKExpr = TopKVar ; TopKExpr = TopK ),
    format(string(Code), '
    // Initialize C# ONNX Runtime searcher
    var searcher = new OnnxVectorSearch("data.db", "models/~w-onnx");
    var results = searcher.Search("~w", ~w);
', [Model, QueryExpr, TopKExpr]).
