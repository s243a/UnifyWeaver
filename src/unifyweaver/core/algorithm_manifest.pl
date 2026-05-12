:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% algorithm_manifest.pl — Workload-author manifest abstraction.
%
% Separates the *algorithm* (the logical/declarative specification of
% what is being computed) from the *optimization manifest* (the
% operational specification of how to compile it: cost-model knobs,
% cache mode, scan strategy, demand-filter spec, etc.).
%
% Algorithm here means the Datalog/SQL sense of "a query" — the
% declarative *what*, not the algorithms-textbook sense of "a
% step-by-step procedure". The optimization manifest *completes* the
% definition by saying how the algorithm should be compiled.
%
% This module is target-agnostic. The manifest entries flow into the
% existing option-list plumbing that resolvers in
% src/unifyweaver/core/cost_model.pl and target-specific code
% (e.g. wam_haskell_target.pl) already consume.
%
% See docs/design/ALGORITHM_MANIFEST_SPECIFICATION.md for the full
% spec.

:- module(algorithm_manifest, [
    %% User-facing directives (workload files call these via `:-`)
    decl_algorithm/2,                       % +Name, +AlgorithmOpts
    decl_algorithm_optimization/2,          % +Name, +OptList

    %% Codegen-side API (target adapters call these)
    load_algorithm_manifest/2,              % +Options0, -Options
    merge_caller_and_manifest_options/3,    % +Caller, +Manifest, -Merged

    %% Accessors
    manifest_algorithm/2,                   % -Name, -AlgorithmOpts
    manifest_optimization_options/2,        % +Name, -ConcatenatedOpts

    %% Test isolation
    reset_manifest/0
]).

:- use_module(library(error), [must_be/2, domain_error/2]).
:- use_module(library(lists)).
:- use_module(library(option), [option/3]).

%% =====================================================================
%% Internal registry
%% =====================================================================
%
% These are private dynamic predicates that hold the asserted state.
% Workloads don't touch them directly — they call decl_algorithm/2
% and decl_algorithm_optimization/2 as directives, which assert into
% the registry.

:- dynamic registered_algorithm/2.
:- dynamic registered_optimization/2.

%% =====================================================================
%! decl_algorithm(+Name, +AlgorithmOpts) is det.
%
%  Declare an algorithm.
%
%  Workload usage:
%
%      :- use_module(library(algorithm_manifest)).
%      :- decl_algorithm(effective_distance, [
%             kernel(category_ancestor/4),
%             seeds(article/1),
%             roots(root_category/1),
%             max_depth(10),
%             dimension_n(5)
%         ]).
%
%  At most one decl_algorithm/2 per Name across all loaded files. A
%  second declaration with the same Name throws an error rather than
%  silently picking one (defensive: keeps the manifest unambiguous).
%
%  Validation of AlgorithmOpts beyond list-shape is deferred to the
%  target adapter, which knows what option keys make sense for its
%  codegen. The manifest module itself doesn't validate semantically.
%% =====================================================================

decl_algorithm(Name, Opts) :-
    must_be(atom, Name),
    must_be(list, Opts),
    (   registered_algorithm(Name, _)
    ->  throw(error(
            domain_error(unique_algorithm_decl, Name),
            context(decl_algorithm/2,
                    'algorithm already declared (only one decl_algorithm/2 per Name is allowed)')))
    ;   assertz(registered_algorithm(Name, Opts))
    ).

%% =====================================================================
%! decl_algorithm_optimization(+Name, +OptList) is det.
%
%  Declare an optimization manifest entry for an algorithm.
%
%  Multiple decl_algorithm_optimization/2 facts per Name are allowed
%  and recommended — workload authors can split optimizations by
%  concern (cost-model knobs, cache tier, scan strategy, demand filter,
%  etc.) into separate declarations for readability.
%
%  Their option lists are concatenated in declaration order using
%  append/2 (shallow concat; NOT flatten/2, which would destructure
%  nested option terms whose values are themselves lists).
%
%  When the merged list is later read via SWI-Prolog's option/3
%  accessor, the first occurrence of any key wins. That's option/3's
%  documented behaviour for repeated keys; the manifest itself does
%  no de-duplication at concat time.
%% =====================================================================

decl_algorithm_optimization(Name, Opts) :-
    must_be(atom, Name),
    must_be(list, Opts),
    assertz(registered_optimization(Name, Opts)).

%% =====================================================================
%! load_algorithm_manifest(+Options0, -Options) is det.
%
%  Merge the declared algorithm + optimization manifest into Options0.
%  Caller-provided options win on key conflict (i.e. they appear
%  earlier in the resulting list, so option/3's first-match read
%  semantics prefer them).
%
%  Idempotent: a sentinel `algorithm_manifest_loaded(true)` marks an
%  already-merged option list; calling load_algorithm_manifest/2 a
%  second time on the same list is a no-op. This lets target
%  adapters call it from multiple entry points (e.g.
%  write_wam_haskell_project/3 and compile_wam_runtime_to_haskell/3,
%  the latter of which is also a nested call from the former)
%  without double-merging.
%
%  When no decl_algorithm/2 is in scope but decl_algorithm_optimization/2
%  facts exist, those facts are orphan and ignored with a stderr
%  warning. (We can't merge optimizations without an algorithm to
%  attach them to.)
%
%  When no manifest data is present at all, returns Options0 unchanged
%  (with the sentinel added).
%% =====================================================================

load_algorithm_manifest(Options0, Options) :-
    must_be(list, Options0),
    (   memberchk(algorithm_manifest_loaded(true), Options0)
    ->  Options = Options0
    ;   compute_merged_options(Options0, MergedNoSentinel),
        Options = [algorithm_manifest_loaded(true) | MergedNoSentinel]
    ).

compute_merged_options(Options0, Merged) :-
    (   registered_algorithm(Name, _AlgOpts)
    ->  manifest_optimization_options(Name, ManifestOpts),
        merge_caller_and_manifest_options(Options0, ManifestOpts, Merged)
    ;   warn_orphan_optimizations,
        Merged = Options0
    ).

%% =====================================================================
%! merge_caller_and_manifest_options(+Caller, +Manifest, -Merged) is det.
%
%  Concatenate caller options with manifest options.
%
%  Implementation: `append(Caller, Manifest, Merged)`. Caller appears
%  first, so when downstream code reads via option/3 (which returns
%  the first matching key), caller values win on conflict.
%
%  This is intentionally not the SWI-Prolog library predicate
%  `merge_options/3` — that one has different conventions around
%  argument order and `option/3`'s default-handling semantics. We
%  pick a clear name to avoid confusion at call sites.
%% =====================================================================

merge_caller_and_manifest_options(Caller, Manifest, Merged) :-
    must_be(list, Caller),
    must_be(list, Manifest),
    append(Caller, Manifest, Merged).

%% =====================================================================
%! manifest_algorithm(-Name, -AlgorithmOpts) is semidet.
%
%  Read the currently declared algorithm. Fails if none is declared.
%  Useful for target adapters that want to consult algorithm-level
%  metadata (e.g. `kernel(P/N)` to know which predicate is the
%  recursive kernel).
%% =====================================================================

manifest_algorithm(Name, Opts) :-
    registered_algorithm(Name, Opts).

%% =====================================================================
%! manifest_optimization_options(+Name, -ConcatenatedOpts) is det.
%
%  Read all optimization options declared for Name, concatenated in
%  declaration order. Returns the empty list if no optimizations are
%  declared (this is not an error — an algorithm with no manifest
%  optimizations runs with caller-only options).
%% =====================================================================

manifest_optimization_options(Name, ConcatenatedOpts) :-
    must_be(atom, Name),
    findall(OptList, registered_optimization(Name, OptList), AllOptLists),
    append(AllOptLists, ConcatenatedOpts).

%% =====================================================================
%! reset_manifest is det.
%
%  Clear all registered manifest state. Used by tests for isolation
%  between runs. Production code should not call this.
%% =====================================================================

reset_manifest :-
    retractall(registered_algorithm(_, _)),
    retractall(registered_optimization(_, _)).

%% =====================================================================
%% Internals
%% =====================================================================

warn_orphan_optimizations :-
    findall(Name, registered_optimization(Name, _), Names),
    sort(Names, Unique),
    forall(member(Name, Unique),
           format(user_error,
                  '[algorithm_manifest] warning: decl_algorithm_optimization(~w, _) declared but no matching decl_algorithm(~w, _); optimization ignored~n',
                  [Name, Name])).
