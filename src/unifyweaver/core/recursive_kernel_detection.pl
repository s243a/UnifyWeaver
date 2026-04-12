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

%% =====================================================================
%% Template file mapping — Mustache template for each kernel kind
%% =====================================================================

kernel_template_file(category_ancestor, 'kernel_category_ancestor.hs.mustache').
kernel_template_file(transitive_closure2, 'kernel_transitive_closure.hs.mustache').

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
%% Helper: sub_term/2 — check if a term appears anywhere in a body
%% =====================================================================

sub_term(Pattern, Term) :-
    subsumes_term(Pattern, Term), !.
sub_term(Pattern, Term) :-
    compound(Term),
    Term =.. [_|Args],
    member(Arg, Args),
    sub_term(Pattern, Arg).
