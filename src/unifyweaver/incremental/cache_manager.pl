:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% cache_manager.pl - Compilation cache for incremental compilation
%
% Stores and retrieves compiled code indexed by predicate, target, and hash.
% Supports cache invalidation when dependencies change.
%
% Usage:
%   % Store compiled code
%   ?- store_cached(foo/2, bash, 12345678, "compiled code...").
%   true.
%
%   % Retrieve cached code (if hash matches)
%   ?- get_cached(foo/2, bash, 12345678, Code).
%   Code = "compiled code...".
%
%   % Invalidate when dependency changes
%   ?- invalidate_cache(foo/2, bash).
%   true.

:- module(cache_manager, [
    get_cached/4,               % +Pred/Arity, +Target, +CurrentHash, -Code
    store_cached/4,             % +Pred/Arity, +Target, +Hash, +Code
    invalidate_cache/2,         % +Pred/Arity, +Target
    invalidate_all_targets/1,   % +Pred/Arity
    invalidate_dependents/2,    % +Pred/Arity, +Target
    clear_cache/0,              % Clear all cached entries
    clear_cache_target/1,       % +Target - Clear cache for specific target
    cache_stats/1,              % -Stats - Get cache statistics
    cache_entries/1,            % -Entries - List all cache entries
    is_cached/3,                % +Pred/Arity, +Target, +Hash
    compilation_cache/5,        % Dynamic: Pred/Arity, Target, Hash, Code, Timestamp
    test_cache_manager/0
]).

:- use_module(library(lists)).

% Import call_graph for dependency tracking
:- use_module('../core/advanced/call_graph', [
    get_dependencies/2,
    get_transitive_dependents/2
]).

% ============================================================================
% CACHE STORAGE
% ============================================================================

%% compilation_cache(Pred/Arity, Target, Hash, Code, Timestamp)
%
% Dynamic predicate storing cached compilations.
%   - Pred/Arity: The predicate being cached
%   - Target: Target language (bash, go, rust, csharp, powershell, sql)
%   - Hash: Content hash at time of compilation
%   - Code: The compiled code (string or term)
%   - Timestamp: Unix timestamp when cached
%
:- dynamic compilation_cache/5.

% ============================================================================
% CACHE RETRIEVAL
% ============================================================================

%% get_cached(+Pred/Arity, +Target, +CurrentHash, -Code) is semidet.
%
% Retrieve cached code if the current hash matches the cached hash.
% Fails if no cache entry exists or hash doesn't match.
%
get_cached(Pred/Arity, Target, CurrentHash, Code) :-
    compilation_cache(Pred/Arity, Target, CurrentHash, Code, _).

%% is_cached(+Pred/Arity, +Target, +Hash) is semidet.
%
% Check if a predicate is cached with the given hash.
%
is_cached(Pred/Arity, Target, Hash) :-
    compilation_cache(Pred/Arity, Target, Hash, _, _).

% ============================================================================
% CACHE STORAGE
% ============================================================================

%% store_cached(+Pred/Arity, +Target, +Hash, +Code) is det.
%
% Store compiled code in the cache.
% Replaces any existing cache entry for this predicate/target combination.
%
store_cached(Pred/Arity, Target, Hash, Code) :-
    get_time(Timestamp),
    % Remove any existing entry for this predicate/target
    retractall(compilation_cache(Pred/Arity, Target, _, _, _)),
    % Store new entry
    assertz(compilation_cache(Pred/Arity, Target, Hash, Code, Timestamp)).

% ============================================================================
% CACHE INVALIDATION
% ============================================================================

%% invalidate_cache(+Pred/Arity, +Target) is det.
%
% Remove the cached entry for a specific predicate and target.
% Succeeds even if no entry exists.
%
invalidate_cache(Pred/Arity, Target) :-
    retractall(compilation_cache(Pred/Arity, Target, _, _, _)).

%% invalidate_all_targets(+Pred/Arity) is det.
%
% Remove cached entries for a predicate across all targets.
%
invalidate_all_targets(Pred/Arity) :-
    retractall(compilation_cache(Pred/Arity, _, _, _, _)).

%% invalidate_dependents(+Pred/Arity, +Target) is det.
%
% Invalidate cache entries for all predicates that depend on this one.
% Uses the call graph to find transitive dependents (reverse graph traversal).
%
invalidate_dependents(Pred/Arity, Target) :-
    % Use call_graph's transitive dependents (includes the predicate itself)
    catch(
        get_transitive_dependents(Pred/Arity, AllDependents),
        _,
        AllDependents = []
    ),
    % Remove the predicate itself from dependents list
    exclude(=(Pred/Arity), AllDependents, Dependents),
    forall(
        member(Dep, Dependents),
        invalidate_cache(Dep, Target)
    ).

% ============================================================================
% CACHE MANAGEMENT
% ============================================================================

%% clear_cache is det.
%
% Remove all cached entries.
%
clear_cache :-
    retractall(compilation_cache(_, _, _, _, _)).

%% clear_cache_target(+Target) is det.
%
% Remove all cached entries for a specific target.
%
clear_cache_target(Target) :-
    retractall(compilation_cache(_, Target, _, _, _)).

%% cache_stats(-Stats) is det.
%
% Get statistics about the cache.
% Stats is a list of key-value pairs:
%   - total_entries: Number of cached entries
%   - by_target: List of target-count pairs
%   - oldest_entry: Timestamp of oldest entry (or none)
%   - newest_entry: Timestamp of newest entry (or none)
%
cache_stats(Stats) :-
    findall(Target-Pred/Arity-Timestamp,
        compilation_cache(Pred/Arity, Target, _, _, Timestamp),
        Entries),
    length(Entries, TotalEntries),

    % Count by target
    findall(Target-Count,
        (   member(Target, [bash, go, rust, csharp, powershell, sql]),
            findall(P, compilation_cache(P, Target, _, _, _), Ps),
            length(Ps, Count),
            Count > 0
        ),
        ByTarget),

    % Find oldest and newest
    (   Entries = []
    ->  OldestEntry = none, NewestEntry = none
    ;   findall(T, member(_-_-T, Entries), Timestamps),
        min_list(Timestamps, OldestEntry),
        max_list(Timestamps, NewestEntry)
    ),

    Stats = [
        total_entries(TotalEntries),
        by_target(ByTarget),
        oldest_entry(OldestEntry),
        newest_entry(NewestEntry)
    ].

%% cache_entries(-Entries) is det.
%
% List all cache entries with their metadata.
% Each entry is: entry(Pred/Arity, Target, Hash, Timestamp)
%
cache_entries(Entries) :-
    findall(
        entry(Pred/Arity, Target, Hash, Timestamp),
        compilation_cache(Pred/Arity, Target, Hash, _, Timestamp),
        Entries
    ).

% ============================================================================
% CACHE AGING (Optional)
% ============================================================================

%% evict_old_entries(+MaxAgeSeconds) is det.
%
% Remove cache entries older than MaxAgeSeconds.
% Useful for keeping cache size manageable.
%
evict_old_entries(MaxAgeSeconds) :-
    get_time(Now),
    Cutoff is Now - MaxAgeSeconds,
    findall(Pred/Arity-Target,
        (   compilation_cache(Pred/Arity, Target, _, _, Timestamp),
            Timestamp < Cutoff
        ),
        ToEvict),
    forall(
        member(Pred/Arity-Target, ToEvict),
        invalidate_cache(Pred/Arity, Target)
    ).

% ============================================================================
% TESTS
% ============================================================================

test_cache_manager :-
    writeln('=== CACHE MANAGER TESTS ==='),

    % Setup: clear any existing cache
    clear_cache,

    % Test 1: Store and retrieve
    test_store_and_retrieve,

    % Test 2: Hash mismatch returns no result
    test_hash_mismatch,

    % Test 3: Invalidation
    test_invalidation,

    % Test 4: Cache stats
    test_cache_stats,

    % Test 5: Clear by target
    test_clear_by_target,

    % Cleanup
    clear_cache,

    writeln('=== ALL CACHE MANAGER TESTS PASSED ===').

test_store_and_retrieve :-
    write('  Testing store and retrieve... '),
    clear_cache,

    % Store a cache entry
    store_cached(test_pred/2, bash, 12345, "echo test"),

    % Retrieve with correct hash
    (   get_cached(test_pred/2, bash, 12345, Code),
        Code == "echo test"
    ->  writeln('PASS')
    ;   writeln('FAIL: Could not retrieve cached code'),
        fail
    ).

test_hash_mismatch :-
    write('  Testing hash mismatch... '),
    clear_cache,

    % Store with one hash
    store_cached(test_pred/2, bash, 12345, "echo test"),

    % Try to retrieve with different hash
    (   get_cached(test_pred/2, bash, 99999, _)
    ->  writeln('FAIL: Should not have returned code for wrong hash'),
        fail
    ;   writeln('PASS')
    ).

test_invalidation :-
    write('  Testing invalidation... '),
    clear_cache,

    % Store entries
    store_cached(pred_a/1, bash, 111, "code_a"),
    store_cached(pred_b/2, bash, 222, "code_b"),
    store_cached(pred_a/1, go, 333, "code_a_go"),

    % Invalidate one entry
    invalidate_cache(pred_a/1, bash),

    % Check pred_a/1 bash is gone but others remain
    (   \+ is_cached(pred_a/1, bash, 111),
        is_cached(pred_b/2, bash, 222),
        is_cached(pred_a/1, go, 333)
    ->  writeln('PASS')
    ;   writeln('FAIL: Invalidation did not work correctly'),
        fail
    ).

test_cache_stats :-
    write('  Testing cache stats... '),
    clear_cache,

    % Store some entries
    store_cached(stat_pred1/1, bash, 100, "code1"),
    store_cached(stat_pred2/2, bash, 200, "code2"),
    store_cached(stat_pred3/1, go, 300, "code3"),

    % Get stats
    cache_stats(Stats),

    % Verify total count
    (   member(total_entries(3), Stats)
    ->  writeln('PASS')
    ;   format('FAIL: Expected 3 entries, got ~w~n', [Stats]),
        fail
    ).

test_clear_by_target :-
    write('  Testing clear by target... '),
    clear_cache,

    % Store entries for different targets
    store_cached(clear_pred/1, bash, 100, "bash_code"),
    store_cached(clear_pred/1, go, 200, "go_code"),
    store_cached(clear_pred/2, bash, 300, "bash_code2"),

    % Clear only bash
    clear_cache_target(bash),

    % Verify bash entries gone, go remains
    (   \+ is_cached(clear_pred/1, bash, 100),
        \+ is_cached(clear_pred/2, bash, 300),
        is_cached(clear_pred/1, go, 200)
    ->  writeln('PASS')
    ;   writeln('FAIL: Clear by target did not work correctly'),
        fail
    ).
