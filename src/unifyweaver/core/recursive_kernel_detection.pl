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

%% =====================================================================
%% Kernel metadata
%% =====================================================================

kernel_native_kind(category_ancestor, category_ancestor).
kernel_native_kind(transitive_closure2, transitive_closure2).

kernel_result_layout(category_ancestor, tuple(1)).
kernel_result_layout(transitive_closure2, tuple(1)).

kernel_result_mode(category_ancestor, stream).
kernel_result_mode(transitive_closure2, stream).

kernel_expected_arity(category_ancestor, 4).
kernel_expected_arity(transitive_closure2, 2).

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
    % Find a base clause with \+ member and Hops=1
    member(_-BaseBody, Clauses),
    BaseBody \= true,
    sub_term(\+ member(_, _), BaseBody),
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
                              [max_depth(MaxDepth)]).

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
%% Helper: sub_term/2 — check if a term appears anywhere in a body
%% =====================================================================

sub_term(Pattern, Term) :-
    subsumes_term(Pattern, Term), !.
sub_term(Pattern, Term) :-
    compound(Term),
    Term =.. [_|Args],
    member(Arg, Args),
    sub_term(Pattern, Arg).
