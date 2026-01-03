:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% hasher.pl - Predicate content hashing for incremental compilation
%
% Computes stable hashes of predicate definitions to detect changes.
% Used by the incremental compiler to determine if recompilation is needed.
%
% Usage:
%   ?- hash_predicate(foo/2, Hash).
%   Hash = 12345678.
%
%   ?- hash_predicate_with_deps(foo/2, CombinedHash).
%   CombinedHash = 87654321.

:- module(hasher, [
    hash_predicate/2,              % +Pred/Arity, -Hash
    hash_predicate_with_deps/2,    % +Pred/Arity, -CombinedHash
    hash_predicate_with_options/3, % +Pred/Arity, +Options, -Hash
    hash_clauses/2,                % +Clauses, -Hash
    normalize_term/2,              % +Term, -NormalizedTerm
    test_hasher/0
]).

:- use_module(library(lists)).

% Import call_graph for dependency tracking
:- use_module('../core/advanced/call_graph').

% ============================================================================
% PREDICATE HASHING
% ============================================================================

%% hash_predicate(+Pred/Arity, -Hash) is det.
%
% Compute a hash of a predicate's clause definitions.
% The hash changes when:
%   - Clauses are added, removed, or modified
%   - Clause order changes
%
% The hash is stable across sessions (same clauses = same hash).
%
hash_predicate(Pred/Arity, Hash) :-
    functor(Head, Pred, Arity),
    findall(NormalizedClause,
        (   clause(Head, Body),
            normalize_clause(Head, Body, NormalizedClause)
        ),
        Clauses),
    hash_clauses(Clauses, Hash).

%% hash_predicate_with_deps(+Pred/Arity, -CombinedHash) is det.
%
% Compute a hash that includes both the predicate and all its dependencies.
% This provides transitive change detection - if any dependency changes,
% the combined hash changes.
%
hash_predicate_with_deps(Pred/Arity, CombinedHash) :-
    hash_predicate(Pred/Arity, SelfHash),
    get_all_dependencies(Pred/Arity, Deps),
    findall(DepHash,
        (   member(Dep, Deps),
            hash_predicate(Dep, DepHash)
        ),
        DepHashes),
    sort([SelfHash|DepHashes], SortedHashes),
    term_hash(SortedHashes, CombinedHash).

%% hash_predicate_with_options(+Pred/Arity, +Options, -Hash) is det.
%
% Compute a hash that includes compilation options.
% Different options should produce different hashes to avoid
% serving incorrectly cached code.
%
hash_predicate_with_options(Pred/Arity, Options, Hash) :-
    hash_predicate(Pred/Arity, PredHash),
    % Normalize options for consistent hashing
    sort(Options, SortedOptions),
    term_hash([PredHash, SortedOptions], Hash).

% ============================================================================
% CLAUSE NORMALIZATION
% ============================================================================

%% normalize_clause(+Head, +Body, -NormalizedClause) is det.
%
% Normalize a clause for stable hashing.
% Variables are renamed to canonical forms (V1, V2, ...).
%
normalize_clause(Head, Body, normalized(NormHead, NormBody)) :-
    copy_term((Head, Body), (HeadCopy, BodyCopy)),
    numbervars((HeadCopy, BodyCopy), 0, _),
    NormHead = HeadCopy,
    NormBody = BodyCopy.

%% normalize_term(+Term, -NormalizedTerm) is det.
%
% Normalize a term by renaming variables to canonical form.
%
normalize_term(Term, NormalizedTerm) :-
    copy_term(Term, TermCopy),
    numbervars(TermCopy, 0, _),
    NormalizedTerm = TermCopy.

% ============================================================================
% HASH COMPUTATION
% ============================================================================

%% hash_clauses(+Clauses, -Hash) is det.
%
% Compute a hash from a list of normalized clauses.
% Uses term_hash/2 which is fast and deterministic.
%
hash_clauses(Clauses, Hash) :-
    term_hash(Clauses, Hash).

% ============================================================================
% DEPENDENCY HELPERS
% ============================================================================

%% get_all_dependencies(+Pred/Arity, -AllDeps) is det.
%
% Get all transitive dependencies of a predicate.
%
get_all_dependencies(Pred/Arity, AllDeps) :-
    get_all_deps_acc([Pred/Arity], [], AllDeps).

get_all_deps_acc([], Acc, Acc).
get_all_deps_acc([Pred|Queue], Visited, AllDeps) :-
    (   member(Pred, Visited)
    ->  get_all_deps_acc(Queue, Visited, AllDeps)
    ;   (   catch(call_graph:get_dependencies(Pred, DirectDeps), _, DirectDeps = [])
        ->  true
        ;   DirectDeps = []
        ),
        subtract(DirectDeps, Visited, NewDeps),
        append(Queue, NewDeps, NewQueue),
        get_all_deps_acc(NewQueue, [Pred|Visited], AllDeps)
    ).

% ============================================================================
% TESTS
% ============================================================================

test_hasher :-
    writeln('=== HASHER TESTS ==='),

    % Setup test predicates
    setup_test_predicates,

    % Test 1: Hash stability
    test_hash_stability,

    % Test 2: Hash changes on modification
    test_hash_change_detection,

    % Test 3: Normalization
    test_normalization,

    % Test 4: Options affect hash
    test_options_hash,

    % Cleanup
    cleanup_test_predicates,

    writeln('=== ALL HASHER TESTS PASSED ===').

setup_test_predicates :-
    % Clear any existing test predicates
    catch(abolish(user:test_hash_pred/2), _, true),
    catch(abolish(user:test_hash_helper/1), _, true),

    % Define test predicates
    assertz(user:test_hash_pred(a, 1)),
    assertz(user:test_hash_pred(b, 2)),
    assertz(user:(test_hash_pred(X, Y) :- test_hash_helper(X), Y is X + 1)),
    assertz(user:test_hash_helper(10)).

cleanup_test_predicates :-
    catch(abolish(user:test_hash_pred/2), _, true),
    catch(abolish(user:test_hash_helper/1), _, true).

test_hash_stability :-
    write('  Testing hash stability... '),
    hash_predicate(test_hash_pred/2, H1),
    hash_predicate(test_hash_pred/2, H2),
    (   H1 == H2
    ->  writeln('PASS')
    ;   format('FAIL: ~w \\= ~w~n', [H1, H2]),
        fail
    ).

test_hash_change_detection :-
    write('  Testing hash change detection... '),
    hash_predicate(test_hash_pred/2, H1),
    % Add a new clause
    assertz(user:test_hash_pred(c, 3)),
    hash_predicate(test_hash_pred/2, H2),
    % Remove the clause
    retract(user:test_hash_pred(c, 3)),
    hash_predicate(test_hash_pred/2, H3),
    (   H1 \== H2, H1 == H3
    ->  writeln('PASS')
    ;   format('FAIL: H1=~w, H2=~w, H3=~w~n', [H1, H2, H3]),
        fail
    ).

test_normalization :-
    write('  Testing variable normalization... '),
    % Two terms with different variable names should normalize the same
    normalize_term(foo(X, _, X), N1),
    normalize_term(foo(A, _, A), N2),
    (   N1 == N2
    ->  writeln('PASS')
    ;   format('FAIL: ~w \\= ~w~n', [N1, N2]),
        fail
    ).

test_options_hash :-
    write('  Testing options affect hash... '),
    hash_predicate_with_options(test_hash_pred/2, [], H1),
    hash_predicate_with_options(test_hash_pred/2, [unique(true)], H2),
    hash_predicate_with_options(test_hash_pred/2, [unique(true)], H3),
    (   H1 \== H2, H2 == H3
    ->  writeln('PASS')
    ;   format('FAIL: H1=~w, H2=~w, H3=~w~n', [H1, H2, H3]),
        fail
    ).
