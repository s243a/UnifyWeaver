:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% recursive_kernel_detection.pl — Shared recursive kernel detection
%
% Target-independent clause-pattern analysis that identifies known
% recursive predicate shapes (e.g., transitive closure with cycle
% detection, depth-bounded ancestor search). The detection results
% are consumed by target-specific emitters (Rust, Haskell, LLVM, etc.)
% to generate specialized native code instead of generic WAM dispatch.
%
% Factored from wam_rust_target.pl / rust_target.pl so the same
% detectors serve all targets. See:
%   - docs/design/WAM_HASKELL_LOWERED_BACKGROUND.md §Path 3
%   - docs/proposals/WAM_FOREIGN_DISPATCH_RETURN_TYPE.md
%
% Usage:
%   :- use_module('recursive_kernel_detection').
%   detect_recursive_kernel(category_ancestor, 4, Clauses, Kernel).

:- module(recursive_kernel_detection, [
    detect_recursive_kernel/4,       % +Pred, +Arity, +Clauses, -Kernel
    kernel_metadata/4,               % +Kernel, -NativeKind, -ResultLayout, -ResultMode
    kernel_config/2,                 % +Kernel, -ConfigOps
    kernel_register_layout/2,        % +KernelKind, -RegisterSpecs
    kernel_native_call/2,            % +KernelKind, -CallSpec
    kernel_template_file/2,          % +KernelKind, -FileName
    registered_kernel/2              % -KernelKind, -Arity
]).

:- use_module(library(lists)).

%% =====================================================================
%% Detection pipeline
%% =====================================================================

%% detect_recursive_kernel(+Pred, +Arity, +Clauses, -Kernel)
%  Try each registered detector against the user's clause list. On
%  success, Kernel is a term:
%      recursive_kernel(KernelKind, Pred/Arity, ConfigOps)
%  where KernelKind is an atom naming the kernel shape, and ConfigOps
%  is a list of target-neutral configuration terms extracted from the
%  clauses (e.g., max_depth(10) for category_ancestor).
%
%  Clauses is a list of Head-Body pairs as returned by
%  findall(Head-Body, user:clause(Head, Body), Clauses).
detect_recursive_kernel(Pred, Arity, Clauses, Kernel) :-
    kernel_detector(KernelKind, Detector),
    call(Detector, Pred, Arity, Clauses, Kernel),
    Kernel = recursive_kernel(KernelKind, _, _),
    !.  % commit to first matching kernel

%% kernel_metadata(+Kernel, -NativeKind, -ResultLayout, -ResultMode)
%  Extract the three orthogonal properties a target emitter needs to
%  generate native code for the detected kernel.
%
%  - NativeKind: atom used as a dispatch key (e.g., category_ancestor)
%  - ResultLayout: tuple(N) — how many output slots per solution
%  - ResultMode: stream | deterministic | deterministic_collection
kernel_metadata(recursive_kernel(Kind, _, _), NativeKind, ResultLayout, ResultMode) :-
    kernel_native_kind(Kind, NativeKind),
    kernel_result_layout(Kind, ResultLayout),
    kernel_result_mode(Kind, ResultMode).

%% kernel_config(+Kernel, -ConfigOps)
%  Extract the target-neutral configuration from a detected kernel.
kernel_config(recursive_kernel(_, _, ConfigOps), ConfigOps).

%% registered_kernel(-KernelKind, -Arity)
%  Enumerate all registered kernel kinds and their expected arities.
%  Useful for documentation and auto-discovery.
registered_kernel(KernelKind, Arity) :-
    kernel_detector(KernelKind, _),
    kernel_expected_arity(KernelKind, Arity).

%% =====================================================================
%% Kernel registry — add new kernels by adding clauses here
%% =====================================================================

kernel_detector(category_ancestor, detect_category_ancestor).
kernel_detector(transitive_closure2, detect_transitive_closure).
kernel_detector(transitive_distance3, detect_transitive_distance).
kernel_detector(weighted_shortest_path3, detect_weighted_shortest_path).
kernel_detector(transitive_parent_distance4, detect_transitive_parent_distance).
kernel_detector(astar_shortest_path4, detect_astar_shortest_path).
kernel_detector(transitive_step_parent_distance5, detect_transitive_step_parent_distance).

%% =====================================================================
%% Kernel metadata
%% =====================================================================

kernel_native_kind(category_ancestor, category_ancestor).
kernel_native_kind(transitive_closure2, transitive_closure2).
kernel_native_kind(transitive_distance3, transitive_distance3).
kernel_native_kind(weighted_shortest_path3, weighted_shortest_path3).
kernel_native_kind(transitive_parent_distance4, transitive_parent_distance4).
kernel_native_kind(astar_shortest_path4, astar_shortest_path4).
kernel_native_kind(transitive_step_parent_distance5, transitive_step_parent_distance5).

kernel_result_layout(category_ancestor, tuple(1)).
kernel_result_layout(transitive_closure2, tuple(1)).
kernel_result_layout(transitive_distance3, tuple(2)).  % (target, distance)
kernel_result_layout(weighted_shortest_path3, tuple(2)).  % (target, weight)
kernel_result_layout(transitive_parent_distance4, tuple(3)).  % (target, parent, distance)
kernel_result_layout(astar_shortest_path4, tuple(1)).  % single distance (goal-directed)
kernel_result_layout(transitive_step_parent_distance5, tuple(4)).  % (target, step, parent, distance)

kernel_result_mode(category_ancestor, stream).
kernel_result_mode(transitive_closure2, stream).
kernel_result_mode(transitive_distance3, stream).
kernel_result_mode(weighted_shortest_path3, stream).
kernel_result_mode(transitive_parent_distance4, stream).
kernel_result_mode(astar_shortest_path4, stream).  % stream of at most 1
kernel_result_mode(transitive_step_parent_distance5, stream).

kernel_expected_arity(category_ancestor, 4).
kernel_expected_arity(transitive_closure2, 2).
kernel_expected_arity(transitive_distance3, 3).
kernel_expected_arity(weighted_shortest_path3, 3).
kernel_expected_arity(transitive_parent_distance4, 4).
kernel_expected_arity(astar_shortest_path4, 4).
kernel_expected_arity(transitive_step_parent_distance5, 5).

%% =====================================================================
%% Register layout — describes input/output registers for each kernel
%%
%% Each register spec is one of:
%%   input(RegN, atom)       — read register N, expect Atom, extract string
%%   input(RegN, vlist_atoms) — read register N, expect VList, extract [String]
%%   output(RegN, integer)   — bind result to register N as Integer
%%   output(RegN, atom)      — bind result to register N as Atom
%% =====================================================================

kernel_register_layout(category_ancestor, [
    input(1, atom),          % cat
    input(2, atom),          % root
    output(3, integer),      % hops (result)
    input(4, vlist_atoms)    % visited
]).

kernel_register_layout(transitive_closure2, [
    input(1, atom),          % source node
    output(2, atom)          % target node (result)
]).

kernel_register_layout(transitive_distance3, [
    input(1, atom),          % source node
    output(2, atom),         % target node (result, part 1 of tuple)
    output(3, integer)       % distance (result, part 2 of tuple)
]).

kernel_register_layout(weighted_shortest_path3, [
    input(1, atom),          % source node
    output(2, atom),         % target node (result)
    output(3, float)         % shortest-path weight (result)
]).

kernel_register_layout(transitive_parent_distance4, [
    input(1, atom),          % source node
    output(2, atom),         % target node
    output(3, atom),         % parent on shortest path
    output(4, integer)       % distance
]).

kernel_register_layout(astar_shortest_path4, [
    input(1, atom),          % source node
    input(2, atom),          % target node (must be bound — goal-directed)
    input(3, integer),       % dimensionality (Minkowski exponent)
    output(4, float)         % shortest-path distance
]).

kernel_register_layout(transitive_step_parent_distance5, [
    input(1, atom),          % source node
    output(2, atom),         % target node
    output(3, atom),         % first hop (step) from source
    output(4, atom),         % immediate parent of target
    output(5, integer)       % distance
]).

%% =====================================================================
%% Native function call spec — how to assemble arguments for the kernel
%%
%% CallSpec = call(FuncName, ArgSpecs)
%% ArgSpecs is an ordered list of:
%%   config_facts(KeyAtom)        — Map.lookup KeyAtom (wcForeignFacts ctx)
%%   config_facts_from(ConfigKey) — like config_facts but KeyAtom is read
%%                                  from the kernel's config at generation time
%%   reg(RegN)                    — the extracted value from register N
%%   config_int(KeyAtom, Default) — integer from wcForeignConfig
%%   derived(length, RegN)        — length of extracted list from register N
%% =====================================================================

kernel_native_call(category_ancestor,
    call(nativeKernel_category_ancestor, [
        config_facts_from(edge_pred),     % parents map (edge pred name from detection)
        reg(1),                           % catS
        reg(2),                           % rootS
        config_int(max_depth, 10),        % maxD
        derived(length, 4),               % length visitedStrs
        reg(4)                            % visitedStrs
    ])).

kernel_native_call(transitive_closure2,
    call(nativeKernel_transitive_closure, [
        config_facts_from(edge_pred),     % edges map (edge pred name from detection)
        reg(1)                            % source node
    ])).

kernel_native_call(transitive_distance3,
    call(nativeKernel_transitive_distance, [
        config_facts_from(edge_pred),     % edges map
        reg(1)                            % source node
    ])).

kernel_native_call(weighted_shortest_path3,
    call(nativeKernel_weighted_shortest_path, [
        config_weighted_facts_from(edge_pred),  % weighted edges: IntMap [(Int, Double)]
        reg(1)                                  % source node
    ])).

kernel_native_call(transitive_parent_distance4,
    call(nativeKernel_transitive_parent_distance, [
        config_facts_from(edge_pred),     % edges map
        reg(1)                            % source node
    ])).

kernel_native_call(astar_shortest_path4,
    call(nativeKernel_astar_shortest_path, [
        config_weighted_facts_from(edge_pred),       % weighted edges
        config_weighted_facts_from(direct_dist_pred), % heuristic oracle
        reg(1),                                      % source (interned)
        reg(2),                                      % target (interned)
        reg(3)                                       % dimensionality (integer)
    ])).

kernel_native_call(transitive_step_parent_distance5,
    call(nativeKernel_transitive_step_parent_distance, [
        config_facts_from(edge_pred),     % edges map
        reg(1)                            % source node
    ])).

%% =====================================================================
%% Template file mapping — Mustache template for each kernel kind
%% =====================================================================

kernel_template_file(category_ancestor, 'kernel_category_ancestor.hs.mustache').
kernel_template_file(transitive_closure2, 'kernel_transitive_closure.hs.mustache').
kernel_template_file(transitive_distance3, 'kernel_transitive_distance.hs.mustache').
kernel_template_file(weighted_shortest_path3, 'kernel_weighted_shortest_path.hs.mustache').
kernel_template_file(transitive_parent_distance4, 'kernel_transitive_parent_distance.hs.mustache').
kernel_template_file(astar_shortest_path4, 'kernel_astar_shortest_path.hs.mustache').
kernel_template_file(transitive_step_parent_distance5, 'kernel_transitive_step_parent_distance.hs.mustache').

%% =====================================================================
%% Detector: category_ancestor
%%
%% Matches the shape:
%%   category_ancestor(Cat, Parent, 1, Visited) :-
%%       category_parent(Cat, Parent),
%%       \+ member(Parent, Visited).
%%   category_ancestor(Cat, Ancestor, Hops, Visited) :-
%%       max_depth(MaxD), length(Visited, D), D < MaxD, !,
%%       category_parent(Cat, Mid),
%%       \+ member(Mid, Visited),
%%       category_ancestor(Mid, Ancestor, H1, [Mid|Visited]),
%%       Hops is H1 + 1.
%%
%% Extracted config: max_depth(N).
%% =====================================================================

detect_category_ancestor(category_ancestor, 4, Clauses, Kernel) :-
    % Must have at least two clauses
    Clauses = [_|[_|_]],
    % Find a base clause with \+ member and a binary call (the edge pred)
    member(BaseHead-BaseBody, Clauses),
    BaseBody \= true,
    sub_term(\+ member(_, _), BaseBody),
    % Extract the edge predicate: the first binary call in the base clause
    % that isn't member/2 and shares the first arg with the head.
    BaseHead =.. [_, BaseCat|_],
    find_edge_pred(BaseBody, BaseCat, EdgePred),
    % Find a recursive clause with \+ member and Hops is _ + 1
    member(_-RecBody, Clauses),
    RecBody \= true,
    RecBody \= BaseBody,
    sub_term(\+ member(_, _), RecBody),
    sub_term((_ is _ + 1), RecBody),
    % Extract max_depth from the user's asserted facts
    current_predicate(user:max_depth/1),
    user:max_depth(MaxDepth),
    integer(MaxDepth),
    MaxDepth > 0,
    Kernel = recursive_kernel(category_ancestor, category_ancestor/4,
                              [max_depth(MaxDepth), edge_pred(EdgePred/2)]).

%% find_edge_pred(+Body, +HeadArg1, -EdgePredName)
%  Find the binary predicate call in Body whose first arg unifies with
%  HeadArg1 (the "category" argument). This is the edge predicate.
find_edge_pred((A, _), HeadArg, EdgePred) :-
    find_edge_pred(A, HeadArg, EdgePred), !.
find_edge_pred((_, B), HeadArg, EdgePred) :-
    find_edge_pred(B, HeadArg, EdgePred), !.
find_edge_pred(Goal, HeadArg, EdgePred) :-
    Goal \= (_ , _),
    Goal \= (\+ _),
    Goal =.. [EdgePred, Arg1, _],
    EdgePred \= member,
    Arg1 == HeadArg.

%% =====================================================================
%% Detector: transitive_closure (binary edge relation)
%%
%% Matches the shape:
%%   closure(X, Y) :- edge(X, Y).
%%   closure(X, Y) :- edge(X, Z), closure(Z, Y).
%%
%% Extracted config: edge_pred(EdgePred/2).
%% =====================================================================

detect_transitive_closure(Pred, 2, Clauses, Kernel) :-
    member(BaseHead-BaseBody, Clauses),
    member(RecHead-RecBody, Clauses),
    BaseHead =.. [Pred, BaseStart, BaseTarget],
    RecHead =.. [Pred, RecStart, RecTarget],
    % Base: edge(X, Y)
    BaseBody =.. [EdgePred, BaseArg1, BaseArg2],
    BaseArg1 == BaseStart,
    BaseArg2 == BaseTarget,
    % Recursive: edge(X, Z), closure(Z, Y)
    RecBody = (EdgeGoal, RecGoal),
    EdgeGoal =.. [EdgePred, EdgeArg1, EdgeArg2],
    RecGoal =.. [Pred, RecMid, RecGoalTarget],
    EdgeArg1 == RecStart,
    RecGoalTarget == RecTarget,
    EdgeArg2 == RecMid,
    Kernel = recursive_kernel(transitive_closure2, Pred/2,
                              [edge_pred(EdgePred/2)]).

%% =====================================================================
%% Detector: transitive_distance (BFS with distance)
%%
%% Matches the shape:
%%   tdist(X, Y, 1) :- edge(X, Y).
%%   tdist(X, Y, D) :- edge(X, Z), tdist(Z, Y, D1), D is D1 + 1.
%%
%% Or with the 1 spelled out as:
%%   tdist(X, Y, D) :- D = 1, edge(X, Y).
%%
%% Extracted config: edge_pred(EdgePred/2).
%% =====================================================================

detect_transitive_distance(Pred, 3, Clauses, Kernel) :-
    member(BaseHead-BaseBody, Clauses),
    member(RecHead-RecBody, Clauses),
    BaseBody \= RecBody,
    BaseHead =.. [Pred, BaseStart, BaseTarget, BaseDist],
    RecHead =.. [Pred, RecStart, RecTarget, RecDist],
    % Base clause: either `tdist(X,Y,1) :- edge(X,Y)` or with D=1 in body
    find_edge_in_body(BaseBody, BaseStart, BaseTarget, EdgePred),
    % Check base binds distance to 1 (either literally or in head)
    (   BaseDist == 1
    ;   find_unify_one(BaseBody, BaseDist)
    ),
    % Recursive clause: find edge(RecStart, Mid), recursive call, D is D1+1
    find_edge_from(RecBody, RecStart, EdgePred, Mid),
    find_recursive_call(RecBody, Pred, Mid, RecTarget, D1),
    find_is_plus_one(RecBody, RecDist, D1),
    Kernel = recursive_kernel(transitive_distance3, Pred/3,
                              [edge_pred(EdgePred/2)]).

%% find_edge_in_body(+Body, +Start, +Target, -EdgePred)
%  Find `edge(Start, Target)` call somewhere in the body (base clause).
find_edge_in_body((A, _), Start, Target, EdgePred) :-
    find_edge_in_body(A, Start, Target, EdgePred), !.
find_edge_in_body((_, B), Start, Target, EdgePred) :-
    find_edge_in_body(B, Start, Target, EdgePred), !.
find_edge_in_body(Goal, Start, Target, EdgePred) :-
    Goal \= (_, _),
    Goal \= (_ = _),
    Goal \= (_ is _),
    compound(Goal),
    Goal =.. [EdgePred, Arg1, Arg2],
    EdgePred \= member,
    Arg1 == Start,
    Arg2 == Target.

%% find_edge_from(+Body, +Start, -EdgePred, -Mid)
%  Find `EdgePred(Start, Mid)` call — EdgePred name bound from caller.
find_edge_from((A, _), Start, EdgePred, Mid) :-
    find_edge_from(A, Start, EdgePred, Mid), !.
find_edge_from((_, B), Start, EdgePred, Mid) :-
    find_edge_from(B, Start, EdgePred, Mid), !.
find_edge_from(Goal, Start, EdgePred, Mid) :-
    Goal \= (_, _),
    compound(Goal),
    Goal =.. [EdgePred, Arg1, Arg2],
    Arg1 == Start,
    Mid = Arg2.

%% find_recursive_call(+Body, +Pred, +Start, +Target, -Dist)
%  Find `Pred(Start, Target, Dist)` call.
find_recursive_call((A, _), Pred, Start, Target, Dist) :-
    find_recursive_call(A, Pred, Start, Target, Dist), !.
find_recursive_call((_, B), Pred, Start, Target, Dist) :-
    find_recursive_call(B, Pred, Start, Target, Dist), !.
find_recursive_call(Goal, Pred, Start, Target, Dist) :-
    Goal \= (_, _),
    compound(Goal),
    Goal =.. [Pred, A1, A2, A3],
    A1 == Start,
    A2 == Target,
    Dist = A3.

%% find_is_plus_one(+Body, +Target, +Source)
%  Find `Target is Source + 1`.
find_is_plus_one((A, _), T, S) :- find_is_plus_one(A, T, S), !.
find_is_plus_one((_, B), T, S) :- find_is_plus_one(B, T, S), !.
find_is_plus_one((T is Expr), T0, S) :-
    T == T0,
    Expr = S0 + 1,
    S0 == S.

%% find_unify_one(+Body, +Var)
%  Find `Var = 1` or `Var is 1`.
find_unify_one((A, _), V) :- find_unify_one(A, V), !.
find_unify_one((_, B), V) :- find_unify_one(B, V), !.
find_unify_one((V1 = 1), V) :- V1 == V.
find_unify_one((V1 is 1), V) :- V1 == V.

%% =====================================================================
%% Detector: weighted_shortest_path (Dijkstra over weighted edges)
%%
%% Matches the shape:
%%   wsp(X, Y, W) :- weighted_edge(X, Y, W).
%%   wsp(X, Y, TotalW) :-
%%       weighted_edge(X, Mid, W1),
%%       wsp(Mid, Y, RestW),
%%       TotalW is RestW + W1.
%%
%% The edge predicate is ternary: edge_pred(From, To, Weight).
%%
%% Extracted config: edge_pred(EdgePred/3).
%% =====================================================================

detect_weighted_shortest_path(Pred, 3, Clauses, Kernel) :-
    member(BaseHead-BaseBody, Clauses),
    member(RecHead-RecBody, Clauses),
    BaseBody \= RecBody,
    BaseHead =.. [Pred, BaseStart, BaseTarget, BaseW],
    RecHead =.. [Pred, RecStart, RecTarget, RecTotalW],
    % Base: wsp(X, Y, W) :- edge(X, Y, W)
    find_weighted_edge(BaseBody, BaseStart, BaseTarget, BaseW, EdgePred),
    % Recursive: edge(X, Mid, W1), wsp(Mid, Y, RestW), TotalW is RestW + W1
    find_weighted_edge_from(RecBody, RecStart, EdgePred, Mid, W1),
    find_wsp_recursive_call(RecBody, Pred, Mid, RecTarget, RestW),
    find_is_sum(RecBody, RecTotalW, RestW, W1),
    Kernel = recursive_kernel(weighted_shortest_path3, Pred/3,
                              [edge_pred(EdgePred/3)]).

%% find_weighted_edge(+Body, +Start, +Target, +Weight, -EdgePred)
%  Find `edge(Start, Target, Weight)` call.
find_weighted_edge((A, _), Start, Target, W, EdgePred) :-
    find_weighted_edge(A, Start, Target, W, EdgePred), !.
find_weighted_edge((_, B), Start, Target, W, EdgePred) :-
    find_weighted_edge(B, Start, Target, W, EdgePred), !.
find_weighted_edge(Goal, Start, Target, W, EdgePred) :-
    Goal \= (_, _),
    Goal \= (_ is _),
    compound(Goal),
    Goal =.. [EdgePred, Arg1, Arg2, Arg3],
    EdgePred \= member,
    Arg1 == Start,
    Arg2 == Target,
    Arg3 == W.

%% find_weighted_edge_from(+Body, +Start, +EdgePred, -Mid, -Weight)
%  Find `EdgePred(Start, Mid, Weight)` call with EdgePred name known.
find_weighted_edge_from((A, _), Start, EdgePred, Mid, W) :-
    find_weighted_edge_from(A, Start, EdgePred, Mid, W), !.
find_weighted_edge_from((_, B), Start, EdgePred, Mid, W) :-
    find_weighted_edge_from(B, Start, EdgePred, Mid, W), !.
find_weighted_edge_from(Goal, Start, EdgePred, Mid, W) :-
    Goal \= (_, _),
    compound(Goal),
    Goal =.. [EdgePred, Arg1, Arg2, Arg3],
    Arg1 == Start,
    Mid = Arg2,
    W = Arg3.

%% find_wsp_recursive_call(+Body, +Pred, +Start, +Target, -RestWeight)
find_wsp_recursive_call((A, _), Pred, Start, Target, RestW) :-
    find_wsp_recursive_call(A, Pred, Start, Target, RestW), !.
find_wsp_recursive_call((_, B), Pred, Start, Target, RestW) :-
    find_wsp_recursive_call(B, Pred, Start, Target, RestW), !.
find_wsp_recursive_call(Goal, Pred, Start, Target, RestW) :-
    Goal \= (_, _),
    compound(Goal),
    Goal =.. [Pred, A1, A2, A3],
    A1 == Start,
    A2 == Target,
    RestW = A3.

%% find_is_sum(+Body, +TotalW, +RestW, +EdgeW)
%  Find `TotalW is RestW + EdgeW` or `TotalW is EdgeW + RestW`.
find_is_sum((A, _), T, R, E) :- find_is_sum(A, T, R, E), !.
find_is_sum((_, B), T, R, E) :- find_is_sum(B, T, R, E), !.
find_is_sum((T is Expr), T0, R, E) :-
    T == T0,
    (   Expr = R0 + E0, R0 == R, E0 == E
    ;   Expr = E0 + R0, R0 == R, E0 == E
    ).

%% =====================================================================
%% Detector: transitive_parent_distance (BFS + parent + distance)
%%
%% Matches the shape:
%%   pd(X, Y, X, 1) :- edge(X, Y).
%%   pd(X, Y, P, D) :-
%%       edge(X, Mid),
%%       pd(Mid, Y, P, D1),
%%       D is D1 + 1.
%%
%% Base case: parent is the source itself (distance 1 direct edge).
%% Recursive: parent comes from the recursive subquery.
%%
%% Extracted config: edge_pred(EdgePred/2).
%% =====================================================================

detect_transitive_parent_distance(Pred, 4, Clauses, Kernel) :-
    member(BaseHead-BaseBody, Clauses),
    member(RecHead-RecBody, Clauses),
    BaseBody \= RecBody,
    BaseHead =.. [Pred, BaseStart, BaseTarget, BaseParent, BaseDist],
    RecHead =.. [Pred, RecStart, RecTarget, RecParent, RecDist],
    % Base: pd(X, Y, X, 1) :- edge(X, Y).  -- parent == source, dist == 1
    BaseParent == BaseStart,
    (   BaseDist == 1 ; find_unify_one(BaseBody, BaseDist) ),
    find_edge_in_body(BaseBody, BaseStart, BaseTarget, EdgePred),
    % Recursive: edge(X, Mid), pd(Mid, Y, P, D1), D is D1 + 1
    find_edge_from(RecBody, RecStart, EdgePred, Mid),
    find_pd_recursive_call(RecBody, Pred, Mid, RecTarget, RecParent, D1),
    find_is_plus_one(RecBody, RecDist, D1),
    Kernel = recursive_kernel(transitive_parent_distance4, Pred/4,
                              [edge_pred(EdgePred/2)]).

%% find_pd_recursive_call(+Body, +Pred, +Start, +Target, +Parent, -Dist)
%  Find `Pred(Start, Target, Parent, Dist)` call.
find_pd_recursive_call((A, _), Pred, Start, Target, Parent, Dist) :-
    find_pd_recursive_call(A, Pred, Start, Target, Parent, Dist), !.
find_pd_recursive_call((_, B), Pred, Start, Target, Parent, Dist) :-
    find_pd_recursive_call(B, Pred, Start, Target, Parent, Dist), !.
find_pd_recursive_call(Goal, Pred, Start, Target, Parent, Dist) :-
    Goal \= (_, _),
    compound(Goal),
    Goal =.. [Pred, A1, A2, A3, A4],
    A1 == Start,
    A2 == Target,
    A3 == Parent,
    Dist = A4.

%% =====================================================================
%% Detector: astar_shortest_path (A* with heuristic)
%%
%% Matches the shape (4-arity weighted shortest path with Dim passthrough):
%%   astar(X, Y, _Dim, W) :- weighted_edge(X, Y, W).
%%   astar(X, Y, Dim, TotalW) :-
%%       weighted_edge(X, Mid, W1),
%%       astar(Mid, Y, Dim, RestW),
%%       TotalW is RestW + W1.
%%
%% Extracted config:
%%   - edge_pred(EdgePred/3)             from clauses
%%   - direct_dist_pred(DistPred/3)      from user-asserted fact (optional)
%%   - dimensionality(Dim)               from user-asserted fact (default 5)
%%
%% If no direct_dist_pred is asserted, astar degenerates to Dijkstra
%% (heuristic = 0) — still different from wsp because it is goal-directed
%% (takes a Target input) and returns a single distance.
%% =====================================================================

detect_astar_shortest_path(Pred, 4, Clauses, Kernel) :-
    member(BaseHead-BaseBody, Clauses),
    member(RecHead-RecBody, Clauses),
    BaseBody \= RecBody,
    BaseHead =.. [Pred, BaseStart, BaseTarget, _BaseDim, BaseW],
    RecHead =.. [Pred, RecStart, RecTarget, RecDim, RecTotalW],
    % Base: astar(X, Y, _, W) :- edge(X, Y, W)
    find_weighted_edge(BaseBody, BaseStart, BaseTarget, BaseW, EdgePred),
    % Recursive: edge(X, Mid, W1), astar(Mid, Y, Dim, RestW), TotalW is RestW + W1
    find_weighted_edge_from(RecBody, RecStart, EdgePred, Mid, W1),
    find_astar_recursive_call(RecBody, Pred, Mid, RecTarget, RecDim, RestW),
    find_is_sum(RecBody, RecTotalW, RestW, W1),
    % Extract optional direct_dist_pred from user facts
    (   current_predicate(user:direct_dist_pred/1),
        user:direct_dist_pred(DistPredSpec)
    ->  DirectDistConf = direct_dist_pred(DistPredSpec)
    ;   DirectDistConf = direct_dist_pred(EdgePred/3)  % fallback = edges
    ),
    % Extract optional dimensionality from user facts
    (   current_predicate(user:dimensionality/1),
        user:dimensionality(Dim),
        integer(Dim)
    ->  true
    ;   Dim = 5
    ),
    Kernel = recursive_kernel(astar_shortest_path4, Pred/4,
                              [edge_pred(EdgePred/3),
                               DirectDistConf,
                               dimensionality(Dim)]).

%% find_astar_recursive_call(+Body, +Pred, +Start, +Target, +Dim, -RestW)
%  Find `Pred(Start, Target, Dim, RestW)` recursive call.
find_astar_recursive_call((A, _), Pred, Start, Target, Dim, RestW) :-
    find_astar_recursive_call(A, Pred, Start, Target, Dim, RestW), !.
find_astar_recursive_call((_, B), Pred, Start, Target, Dim, RestW) :-
    find_astar_recursive_call(B, Pred, Start, Target, Dim, RestW), !.
find_astar_recursive_call(Goal, Pred, Start, Target, Dim, RestW) :-
    Goal \= (_, _),
    compound(Goal),
    Goal =.. [Pred, A1, A2, A3, A4],
    A1 == Start,
    A2 == Target,
    A3 == Dim,
    RestW = A4.

%% =====================================================================
%% Detector: transitive_step_parent_distance
%%   (BFS + first hop + immediate parent + distance)
%%
%% Matches the shape:
%%   tspd(X, Y, Y, X, 1) :- edge(X, Y).
%%       % direct edge: step is the target, parent is the source
%%   tspd(X, Y, Mid, P, D) :-
%%       edge(X, Mid),
%%       tspd(Mid, Y, _IgnoredStep, P, D1),
%%       D is D1 + 1.
%%       % step is the first edge target; parent comes from recursion
%%
%% Extracted config: edge_pred(EdgePred/2).
%% =====================================================================

detect_transitive_step_parent_distance(Pred, 5, Clauses, Kernel) :-
    member(BaseHead-BaseBody, Clauses),
    member(RecHead-RecBody, Clauses),
    BaseBody \= RecBody,
    BaseHead =.. [Pred, BaseStart, BaseTarget, BaseStep, BaseParent, BaseDist],
    RecHead =.. [Pred, RecStart, RecTarget, RecStep, RecParent, RecDist],
    % Base: direct edge with step==target, parent==source, dist==1
    BaseStep == BaseTarget,
    BaseParent == BaseStart,
    (   BaseDist == 1 ; find_unify_one(BaseBody, BaseDist) ),
    find_edge_in_body(BaseBody, BaseStart, BaseTarget, EdgePred),
    % Recursive: edge(X, Mid), tspd(Mid, Y, _, P, D1), D is D1 + 1
    find_edge_from(RecBody, RecStart, EdgePred, Mid),
    RecStep == Mid,  % step in head is the immediate edge target
    find_tspd_recursive_call(RecBody, Pred, Mid, RecTarget, RecParent, D1),
    find_is_plus_one(RecBody, RecDist, D1),
    Kernel = recursive_kernel(transitive_step_parent_distance5, Pred/5,
                              [edge_pred(EdgePred/2)]).

%% find_tspd_recursive_call(+Body, +Pred, +Start, +Target, +Parent, -Dist)
%  Find `Pred(Start, Target, _IgnoredStep, Parent, Dist)` call.
find_tspd_recursive_call((A, _), Pred, Start, Target, Parent, Dist) :-
    find_tspd_recursive_call(A, Pred, Start, Target, Parent, Dist), !.
find_tspd_recursive_call((_, B), Pred, Start, Target, Parent, Dist) :-
    find_tspd_recursive_call(B, Pred, Start, Target, Parent, Dist), !.
find_tspd_recursive_call(Goal, Pred, Start, Target, Parent, Dist) :-
    Goal \= (_, _),
    compound(Goal),
    Goal =.. [Pred, A1, A2, _IgnoredA3, A4, A5],
    A1 == Start,
    A2 == Target,
    A4 == Parent,
    Dist = A5.

%% =====================================================================
%% Helper: sub_term/2 — check if a term appears anywhere in a body
%% =====================================================================

sub_term(Pattern, Term) :-
    subsumes_term(Pattern, Term), !.
sub_term(Pattern, Term) :-
    compound(Term),
    Term =.. [_|Args],
    member(Arg, Args),
    sub_term(Pattern, Arg).
