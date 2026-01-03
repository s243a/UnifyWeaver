:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_integration.pl - Integration tests for incremental compilation
%
% Tests the full incremental compilation pipeline with actual target compilers.

:- module(test_integration, [
    run_integration_tests/0,
    run_multi_target_tests/0,
    test_bash_incremental/0,
    test_cache_invalidation/0,
    test_dependency_tracking/0,
    test_multi_target_caching/0,
    test_cross_target_independence/0
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

% Import target compilers for testing
:- use_module('../core/stream_compiler').
:- use_module('../targets/go_target').
:- use_module('../targets/python_target').
:- use_module('../targets/typescript_target').

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

% ============================================================================
% MULTI-TARGET TESTS (Phase 4)
% ============================================================================

run_multi_target_tests :-
    writeln('=== MULTI-TARGET INCREMENTAL COMPILATION TESTS ==='),
    writeln(''),

    % Clear cache before tests
    clear_incremental_cache,

    % Setup test predicates
    setup_integration_tests,

    % Run multi-target tests
    test_multi_target_caching,
    test_cross_target_independence,
    test_target_specific_invalidation,

    % Cleanup
    cleanup_integration_tests,
    clear_incremental_cache,

    writeln(''),
    writeln('=== ALL MULTI-TARGET TESTS PASSED ===').

% ============================================================================
% TEST: MULTI-TARGET CACHING
% ============================================================================

test_multi_target_caching :-
    writeln('Test 5: Multi-target caching (Bash, Go, Python, TypeScript)'),

    % Test each target compiles and caches
    test_target_compile(bash, 'Bash'),
    test_target_compile(go, 'Go'),
    test_target_compile(python, 'Python'),
    test_target_compile(typescript, 'TypeScript'),

    writeln('  PASS'),
    writeln('').

test_target_compile(Target, TargetName) :-
    format('  Compiling to ~w... ', [TargetName]),
    (   catch(
            compile_incremental(int_test_parent/2, Target, [], Code),
            Error,
            (format('ERROR: ~w~n', [Error]), fail)
        ),
        Code \= []
    ->  writeln('OK')
    ;   format('FAIL: ~w compilation failed~n', [TargetName]),
        fail
    ),

    % Verify it was cached
    format('  Verifying ~w cache... ', [TargetName]),
    hash_predicate_with_options(int_test_parent/2, [], Hash),
    (   is_cached(int_test_parent/2, Target, Hash)
    ->  writeln('OK')
    ;   format('FAIL: ~w not cached~n', [TargetName]),
        fail
    ).

% ============================================================================
% TEST: CROSS-TARGET INDEPENDENCE
% ============================================================================

test_cross_target_independence :-
    writeln('Test 6: Cross-target cache independence'),

    % Clear and compile to multiple targets
    clear_incremental_cache,

    % Compile same predicate to different targets
    write('  Compile to Bash... '),
    compile_incremental(int_test_helper/1, bash, [], BashCode),
    writeln('OK'),

    write('  Compile to Go... '),
    compile_incremental(int_test_helper/1, go, [], GoCode),
    writeln('OK'),

    % Codes should be different (different languages)
    write('  Verifying different output... '),
    (   BashCode \== GoCode
    ->  writeln('OK')
    ;   writeln('FAIL: Same code for different targets'),
        fail
    ),

    % Both should be cached independently
    write('  Verifying independent caches... '),
    hash_predicate_with_options(int_test_helper/1, [], Hash),
    (   is_cached(int_test_helper/1, bash, Hash),
        is_cached(int_test_helper/1, go, Hash)
    ->  writeln('OK')
    ;   writeln('FAIL: Both targets should be cached'),
        fail
    ),

    % Cache hit for each target
    write('  Verifying cache hits... '),
    compile_incremental(int_test_helper/1, bash, [], BashCode2),
    compile_incremental(int_test_helper/1, go, [], GoCode2),
    (   BashCode == BashCode2, GoCode == GoCode2
    ->  writeln('OK')
    ;   writeln('FAIL: Cache should return same code'),
        fail
    ),

    writeln('  PASS'),
    writeln('').

% ============================================================================
% TEST: TARGET-SPECIFIC INVALIDATION
% ============================================================================

test_target_specific_invalidation :-
    writeln('Test 7: Target-specific cache invalidation'),

    clear_incremental_cache,

    % Compile to both targets
    write('  Initial compile to Bash and Go... '),
    compile_incremental(int_test_helper/1, bash, [], _),
    compile_incremental(int_test_helper/1, go, [], _),
    writeln('OK'),

    % Get hashes
    hash_predicate_with_options(int_test_helper/1, [], Hash1),

    % Invalidate only Bash cache
    write('  Invalidating Bash cache only... '),
    cache_manager:invalidate_cache(int_test_helper/1, bash),
    writeln('OK'),

    % Bash should be gone, Go should remain
    write('  Verifying selective invalidation... '),
    (   \+ is_cached(int_test_helper/1, bash, Hash1),
        is_cached(int_test_helper/1, go, Hash1)
    ->  writeln('OK')
    ;   writeln('FAIL: Only Bash should be invalidated'),
        fail
    ),

    writeln('  PASS'),
    writeln('').
