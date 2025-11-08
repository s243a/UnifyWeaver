:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_pattern_detection_issues.pl - Verify pattern detection issues from POST_RELEASE_TODO

:- use_module('../src/unifyweaver/core/recursive_compiler').
:- use_module('../src/unifyweaver/core/advanced/advanced_recursive_compiler').

main :-
    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  Pattern Detection Issues Verification                ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n~n', []),

    test_list_length_pattern,
    test_descendant_pattern,

    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  Issue Verification Complete                          ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n', []),
    halt(0).

main :-
    format('~n[✗] Tests failed~n', []),
    halt(1).

%% Issue #1: list_length/2 linear recursion detection
test_list_length_pattern :-
    format('~n[Issue #1] list_length/2 linear recursion detection~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Define list_length
    catch(abolish(list_length/2), _, true),
    assertz((list_length([], 0))),
    assertz((list_length([_|T], N) :- list_length(T, N1), N is N1 + 1)),

    format('Predicate defined:~n', []),
    format('  list_length([], 0).~n', []),
    format('  list_length([_|T], N) :- list_length(T, N1), N is N1 + 1.~n~n', []),

    % Try to compile with advanced recursion
    format('Attempting advanced recursive compilation...~n', []),
    (   compile_advanced_recursive(list_length/2, [], _Code)
    ->  format('  ✓ UNEXPECTED: Compilation succeeded~n', []),
        format('  Issue #1 appears to be FIXED~n', [])
    ;   format('  ✗ CONFIRMED: Advanced pattern detection failed~n', []),
        format('  Issue #1 is CONFIRMED - list_length not detected as linear recursion~n', [])
    ),

    % Clean up
    catch(abolish(list_length/2), _, true),
    format('[Issue #1] Verified~n', []),
    !.

%% Issue #2: descendant/2 pattern detection
test_descendant_pattern :-
    format('~n[Issue #2] descendant/2 pattern detection~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Define parent facts
    catch(abolish(parent/2), _, true),
    assertz(parent(alice, bob)),
    assertz(parent(bob, charlie)),

    % Define descendant (reverse of ancestor)
    catch(abolish(descendant/2), _, true),
    assertz((descendant(X, Y) :- parent(Y, X))),
    assertz((descendant(X, Z) :- parent(Y, X), descendant(Y, Z))),

    format('Predicates defined:~n', []),
    format('  parent(alice, bob).~n', []),
    format('  parent(bob, charlie).~n', []),
    format('  descendant(X, Y) :- parent(Y, X).~n', []),
    format('  descendant(X, Z) :- parent(Y, X), descendant(Y, Z).~n~n', []),

    % Try basic recursive compilation
    format('Attempting basic recursive compilation...~n', []),
    (   compile_recursive(descendant/2, [], Code)
    ->  format('  ✓ Basic compilation succeeded~n', []),
        % Check if it's using BFS (transitive closure optimization)
        (   sub_atom(Code, _, _, _, 'Work queue')
        ->  format('  ✓ GOOD: Uses BFS optimization (transitive closure detected)~n', []),
            format('  Issue #2 appears to be FIXED~n', [])
        ;   format('  ✗ CONFIRMED: No BFS optimization found~n', []),
            format('  Issue #2 is CONFIRMED - descendant not detected as transitive closure~n', [])
        )
    ;   format('  ✗ Basic compilation failed~n', []),
        format('  Issue #2 is CONFIRMED - descendant compilation fails~n', [])
    ),

    % Try advanced compilation
    format('~nAttempting advanced recursive compilation...~n', []),
    (   compile_advanced_recursive(descendant/2, [], _Code2)
    ->  format('  ✓ Advanced compilation succeeded~n', [])
    ;   format('  ✗ Advanced pattern detection failed (expected - it should use basic BFS)~n', [])
    ),

    % Clean up
    catch(abolish(parent/2), _, true),
    catch(abolish(descendant/2), _, true),
    format('[Issue #2] Verified~n', []),
    !.

:- initialization(main, main).
