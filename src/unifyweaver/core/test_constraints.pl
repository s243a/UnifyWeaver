% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% test_constraints.pl - Test constraint system with various scenarios

:- module(test_constraints, [
    test_constraints/0
]).

:- use_module(constraint_analyzer).
:- use_module(stream_compiler).

test_constraints :-
    format('~n=== Testing Constraint System Integration ===~n'),

    % Setup: Define test data and predicates
    setup_test_data,

    % Test 1: Default behavior (unique + unordered = sort -u)
    format('~n--- Test 1: Default constraints (sort -u) ---~n'),
    test_default_constraints,

    % Test 2: Ordered deduplication (unique + ordered = hash)
    format('~n--- Test 2: Ordered constraints (hash dedup) ---~n'),
    test_ordered_constraints,

    % Test 3: No deduplication (unique=false)
    format('~n--- Test 3: No deduplication ---~n'),
    test_no_deduplication,

    % Test 4: Runtime override
    format('~n--- Test 4: Runtime option override ---~n'),
    test_runtime_override,

    % Cleanup
    cleanup_test_data,

    format('~n=== All constraint integration tests passed! ===~n').

% Setup test data
setup_test_data :-
    % Clear any existing definitions (catch errors if they don't exist)
    catch(abolish(user:test_fact/1), _, true),
    catch(abolish(user:test_default/2), _, true),
    catch(abolish(user:test_ordered/2), _, true),
    catch(abolish(user:test_no_dedup/2), _, true),
    catch(abolish(user:test_override/2), _, true),

    % Define test facts in user module
    assertz(user:(test_fact(a))),
    assertz(user:(test_fact(b))),
    assertz(user:(test_fact(c))),

    % Define test rules in user module
    assertz(user:(test_default(X, Y) :- test_fact(X), test_fact(Y))),
    assertz(user:(test_ordered(X, Y) :- test_fact(X), test_fact(Y))),
    assertz(user:(test_no_dedup(X, Y) :- test_fact(X), test_fact(Y))),
    assertz(user:(test_override(X, Y) :- test_fact(X), test_fact(Y))).

% Test 1: Default behavior (should use sort -u)
test_default_constraints :-
    % No explicit declaration - should use defaults: unique=true, unordered=true
    compile_predicate(test_default/2, [], Code),

    % Verify it uses sort -u
    (   sub_string(Code, _, _, _, "sort -u") ->
        format('  ✓ Uses sort -u (as expected)~n')
    ;   format('  ✗ FAILED: Expected sort -u~n'),
        fail
    ),

    % Verify constraints
    get_constraints(test_default/2, Constraints),
    format('  Constraints: ~w~n', [Constraints]),
    (   Constraints = [unique(true), unordered(true)] ->
        format('  ✓ Correct default constraints~n')
    ;   format('  ✗ FAILED: Wrong constraints~n'),
        fail
    ).

% Test 2: Ordered deduplication (should use hash)
test_ordered_constraints :-
    % Declare as unique but ordered
    declare_constraint(test_ordered/2, [unique, ordered]),
    compile_predicate(test_ordered/2, [], Code),

    % Verify it uses hash-based dedup
    (   sub_string(Code, _, _, _, "declare -A seen") ->
        format('  ✓ Uses hash-based dedup (as expected)~n')
    ;   format('  ✗ FAILED: Expected hash-based dedup~n'),
        fail
    ),

    % Verify it preserves order comment
    (   sub_string(Code, _, _, _, "preserves order") ->
        format('  ✓ Has "preserves order" comment~n')
    ;   format('  ✗ FAILED: Missing "preserves order" comment~n'),
        fail
    ),

    % Verify constraints
    get_constraints(test_ordered/2, Constraints),
    format('  Constraints: ~w~n', [Constraints]),
    (   member(unordered(false), Constraints) ->
        format('  ✓ Correct ordered constraint~n')
    ;   format('  ✗ FAILED: Wrong constraints~n'),
        fail
    ).

% Test 3: No deduplication
test_no_deduplication :-
    % Declare as non-unique
    declare_constraint(test_no_dedup/2, [unique(false)]),
    compile_predicate(test_no_dedup/2, [], Code),

    % Verify it has no deduplication
    (   \+ sub_string(Code, _, _, _, "sort -u"),
        \+ sub_string(Code, _, _, _, "declare -A seen") ->
        format('  ✓ No deduplication (as expected)~n')
    ;   format('  ✗ FAILED: Should not have deduplication~n'),
        fail
    ),

    % Verify no dedup comment
    (   sub_string(Code, _, _, _, "no deduplication") ->
        format('  ✓ Has "no deduplication" comment~n')
    ;   format('  ✗ FAILED: Missing "no deduplication" comment~n'),
        fail
    ).

% Test 4: Runtime option override
test_runtime_override :-
    % Declare as default (unordered)
    declare_constraint(test_override/2, [unique, unordered]),

    % But override at runtime to be ordered
    compile_predicate(test_override/2, [unordered(false)], Code),

    % Should use hash (runtime overrides declaration)
    (   sub_string(Code, _, _, _, "declare -A seen") ->
        format('  ✓ Runtime option overrides declaration~n')
    ;   format('  ✗ FAILED: Runtime option should override~n'),
        fail
    ).

% Cleanup test data
cleanup_test_data :-
    catch(abolish(user:test_fact/1), _, true),
    catch(abolish(user:test_default/2), _, true),
    catch(abolish(user:test_ordered/2), _, true),
    catch(abolish(user:test_no_dedup/2), _, true),
    catch(abolish(user:test_override/2), _, true),
    clear_constraints(test_default/2),
    clear_constraints(test_ordered/2),
    clear_constraints(test_no_dedup/2),
    clear_constraints(test_override/2).
