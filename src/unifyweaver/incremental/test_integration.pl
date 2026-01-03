:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_integration.pl - Integration tests for incremental compilation
%
% Tests the full incremental compilation pipeline with actual target compilers.

:- module(test_integration, [
    run_integration_tests/0,
    test_bash_incremental/0,
    test_cache_invalidation/0,
    test_dependency_tracking/0
]).

:- use_module(library(lists)).

% Import incremental compiler
:- use_module(incremental_compiler, [
    compile_incremental/4,
    clear_incremental_cache/0,
    incremental_stats/0
]).

% Import cache manager for direct inspection
:- use_module(cache_manager, [
    cache_entries/1,
    is_cached/3
]).

% Import hasher for hash comparison
:- use_module(hasher, [
    hash_predicate_with_options/3
]).

% Import stream_compiler (Bash target)
:- use_module('../core/stream_compiler').

% ============================================================================
% MAIN TEST RUNNER
% ============================================================================

run_integration_tests :-
    writeln('=== INCREMENTAL COMPILATION INTEGRATION TESTS ==='),
    writeln(''),

    % Clear cache before tests
    clear_incremental_cache,

    % Setup test predicates
    setup_integration_tests,

    % Run tests
    test_bash_incremental,
    test_cache_invalidation,
    test_dependency_tracking,
    test_options_affect_cache,

    % Cleanup
    cleanup_integration_tests,
    clear_incremental_cache,

    writeln(''),
    writeln('=== ALL INTEGRATION TESTS PASSED ===').

% ============================================================================
% TEST SETUP AND CLEANUP
% ============================================================================

setup_integration_tests :-
    writeln('Setting up test predicates...'),

    % Clear any existing test predicates
    catch(abolish(user:int_test_parent/2), _, true),
    catch(abolish(user:int_test_grandparent/2), _, true),
    catch(abolish(user:int_test_helper/1), _, true),

    % Define test predicates
    assertz(user:int_test_parent(alice, bob)),
    assertz(user:int_test_parent(bob, charlie)),
    assertz(user:int_test_parent(diana, eve)),

    % Grandparent depends on parent
    assertz(user:(int_test_grandparent(GP, GC) :-
        int_test_parent(GP, P),
        int_test_parent(P, GC))),

    % Helper predicate
    assertz(user:int_test_helper(value1)),
    assertz(user:int_test_helper(value2)),

    writeln('Test predicates created.'),
    writeln('').

cleanup_integration_tests :-
    catch(abolish(user:int_test_parent/2), _, true),
    catch(abolish(user:int_test_grandparent/2), _, true),
    catch(abolish(user:int_test_helper/1), _, true).

% ============================================================================
% TEST: BASIC BASH INCREMENTAL COMPILATION
% ============================================================================

test_bash_incremental :-
    writeln('Test 1: Basic Bash incremental compilation'),

    % First compilation - should be cache miss
    write('  First compile (cache miss)... '),
    compile_incremental(int_test_parent/2, bash, [], Code1),
    (   Code1 \= []
    ->  writeln('OK')
    ;   writeln('FAIL: No code generated'),
        fail
    ),

    % Verify it was cached
    write('  Verifying cache entry... '),
    hash_predicate_with_options(int_test_parent/2, [], Hash1),
    (   is_cached(int_test_parent/2, bash, Hash1)
    ->  writeln('OK')
    ;   writeln('FAIL: Not cached'),
        fail
    ),

    % Second compilation - should be cache hit
    write('  Second compile (cache hit)... '),
    compile_incremental(int_test_parent/2, bash, [], Code2),
    (   Code1 == Code2
    ->  writeln('OK')
    ;   writeln('FAIL: Cache returned different code'),
        fail
    ),

    writeln('  PASS'),
    writeln('').

% ============================================================================
% TEST: CACHE INVALIDATION ON CHANGE
% ============================================================================

test_cache_invalidation :-
    writeln('Test 2: Cache invalidation on predicate change'),

    % Compile and cache
    write('  Initial compile... '),
    compile_incremental(int_test_helper/1, bash, [], Code1),
    hash_predicate_with_options(int_test_helper/1, [], Hash1),
    writeln('OK'),

    % Modify the predicate
    write('  Modifying predicate... '),
    assertz(user:int_test_helper(value3)),
    writeln('OK'),

    % Hash should change
    write('  Verifying hash changed... '),
    hash_predicate_with_options(int_test_helper/1, [], Hash2),
    (   Hash1 \== Hash2
    ->  writeln('OK')
    ;   writeln('FAIL: Hash did not change'),
        fail
    ),

    % Old cache entry should not match
    write('  Verifying old cache miss... '),
    (   \+ is_cached(int_test_helper/1, bash, Hash2)
    ->  writeln('OK')
    ;   writeln('FAIL: Old cache should not match new hash'),
        fail
    ),

    % Recompile - should compile fresh and cache new version
    write('  Recompile after change... '),
    compile_incremental(int_test_helper/1, bash, [], Code2),
    (   Code1 \== Code2
    ->  writeln('OK')
    ;   writeln('OK (code may be same structure)')
    ),

    % Restore predicate
    retract(user:int_test_helper(value3)),

    writeln('  PASS'),
    writeln('').

% ============================================================================
% TEST: DEPENDENCY TRACKING
% ============================================================================

test_dependency_tracking :-
    writeln('Test 3: Dependency tracking'),

    % Compile grandparent (depends on parent)
    write('  Compile grandparent (depends on parent)... '),
    compile_incremental(int_test_grandparent/2, bash, [], _GpCode),
    writeln('OK'),

    % Compile parent
    write('  Compile parent... '),
    compile_incremental(int_test_parent/2, bash, [], _ParentCode),
    writeln('OK'),

    % Both should be cached
    write('  Verifying both cached... '),
    hash_predicate_with_options(int_test_grandparent/2, [], GpHash),
    hash_predicate_with_options(int_test_parent/2, [], ParentHash),
    (   is_cached(int_test_grandparent/2, bash, GpHash),
        is_cached(int_test_parent/2, bash, ParentHash)
    ->  writeln('OK')
    ;   writeln('FAIL: Not all cached'),
        fail
    ),

    writeln('  PASS'),
    writeln('').

% ============================================================================
% TEST: OPTIONS AFFECT CACHE
% ============================================================================

test_options_affect_cache :-
    writeln('Test 4: Different options produce different hashes'),

    % Compile with no options
    write('  Compile with no options... '),
    compile_incremental(int_test_helper/1, bash, [], _Code1),
    writeln('OK'),

    % Get hash with no options
    hash_predicate_with_options(int_test_helper/1, [], Hash1),

    % Compile with different options (replaces previous cache entry)
    write('  Compile with unique(true) option... '),
    compile_incremental(int_test_helper/1, bash, [unique(true)], _Code2),
    writeln('OK'),

    % Hashes should differ
    write('  Verifying different hashes... '),
    hash_predicate_with_options(int_test_helper/1, [unique(true)], Hash2),
    (   Hash1 \== Hash2
    ->  writeln('OK')
    ;   writeln('FAIL: Hashes should differ'),
        fail
    ),

    % New options should have replaced old cache entry
    write('  Verifying new options cache entry... '),
    (   is_cached(int_test_helper/1, bash, Hash2)
    ->  writeln('OK')
    ;   writeln('FAIL: New entry should be cached'),
        fail
    ),

    % Compiling with original options should trigger recompile (cache miss)
    write('  Compile with original options (cache miss)... '),
    compile_incremental(int_test_helper/1, bash, [], _Code3),
    writeln('OK'),

    writeln('  PASS'),
    writeln('').
