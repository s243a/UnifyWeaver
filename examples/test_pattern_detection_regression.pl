:- encoding(utf8).
% test_pattern_detection_regression.pl - Regression tests for pattern detection
%
% Verifies that pattern detection continues working correctly after fixes.
% Tests the issues that were resolved in Priority 1 (Issues #1 and #2).

:- use_module('../src/unifyweaver/core/recursive_compiler').
:- use_module('../src/unifyweaver/core/advanced/advanced_recursive_compiler').
:- use_module('../src/unifyweaver/core/advanced/pattern_matchers').

main :-
    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  Pattern Detection Regression Tests                   ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n~n', []),

    test_list_length_pattern,
    test_descendant_classification,
    test_factorial_linear_recursion,

    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  All Regression Tests Passed ✓                         ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n', []),
    halt(0).

main :-
    format('~n[✗] Regression tests failed~n', []),
    halt(1).

%% Test 1: list_length/2 pattern detection
test_list_length_pattern :-
    format('~n[Test 1] list_length/2 linear recursion detection~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Define list_length
    catch(abolish(list_length/2), _, true),
    assertz((list_length([], 0))),
    assertz((list_length([_|T], N) :- list_length(T, N1), N is N1 + 1)),

    % Test pattern detection
    (   is_linear_recursive_streamable(list_length/2)
    ->  format('  ✓ list_length/2 detected as linear recursion~n', [])
    ;   format('  ✗ FAIL: list_length/2 not detected as linear recursion~n', []),
        fail
    ),

    % Test compilation
    (   compile_advanced_recursive(list_length/2, [], Code)
    ->  format('  ✓ list_length/2 compiles successfully~n', []),
        % Verify memoization is present
        (   sub_string(Code, _, _, _, 'memo')
        ->  format('  ✓ Generated code includes memoization~n', [])
        ;   format('  ⚠ Warning: No memoization found in generated code~n', [])
        )
    ;   format('  ✗ FAIL: list_length/2 compilation failed~n', []),
        fail
    ),

    % Clean up
    catch(abolish(list_length/2), _, true),
    format('[✓] Test 1 Passed~n', []),
    !.

%% Test 2: descendant/2 transitive closure detection
test_descendant_classification :-
    format('~n[Test 2] descendant/2 transitive closure detection~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Define parent facts
    catch(abolish(parent/2), _, true),
    catch(abolish(descendant/2), _, true),
    assertz(parent(alice, bob)),
    assertz(parent(bob, charlie)),
    assertz(parent(charlie, diana)),

    % Define descendant
    assertz((descendant(X, Y) :- parent(Y, X))),
    assertz((descendant(X, Z) :- parent(Y, X), descendant(Y, Z))),

    % Test compilation with BFS optimization
    (   compile_recursive(descendant/2, [], Code)
    ->  format('  ✓ descendant/2 compiles successfully~n', []),
        % Verify BFS optimization (work queue)
        (   sub_string(Code, _, _, _, 'queue')
        ->  format('  ✓ Generated code uses BFS optimization (queue found)~n', [])
        ;   format('  ⚠ Warning: No queue found in generated code~n', [])
        ),
        % Verify visited tracking
        (   sub_string(Code, _, _, _, 'visited')
        ->  format('  ✓ Generated code includes visited tracking~n', [])
        ;   format('  ⚠ Warning: No visited tracking found~n', [])
        )
    ;   format('  ✗ FAIL: descendant/2 compilation failed~n', []),
        fail
    ),

    % Clean up
    catch(abolish(parent/2), _, true),
    catch(abolish(descendant/2), _, true),
    format('[✓] Test 2 Passed~n', []),
    !.

%% Test 3: factorial/2 linear recursion with execution
test_factorial_linear_recursion :-
    format('~n[Test 3] factorial/2 linear recursion compilation and execution~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Define factorial
    catch(abolish(factorial/2), _, true),
    assertz((factorial(0, 1))),
    assertz((factorial(N, F) :- N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1)),

    % Test compilation
    (   compile_advanced_recursive(factorial/2, [], Code)
    ->  format('  ✓ factorial/2 compiles successfully~n', []),

        % Write and test execution
        open('/tmp/test_factorial_regression.sh', write, Stream),
        write(Stream, Code),
        close(Stream),

        % Test execution
        catch(
            process_create('/bin/bash', ['-c', 'source /tmp/test_factorial_regression.sh && factorial 5 ""'],
                [stdout(pipe(Output))]),
            Error,
            (format('  ✗ Execution error: ~w~n', [Error]), fail)
        ),

        (   Output \= pipe(_) ->
            read_string(Output, _, Result),
            close(Output),
            % Check for correct result: factorial(5) = 120
            (   sub_string(Result, _, _, _, '120')
            ->  format('  ✓ Execution produces correct result (5! = 120)~n', [])
            ;   format('  ✗ FAIL: Incorrect result: ~w~n', [Result]),
                fail
            )
        ;   format('  ⚠ Could not read execution output~n', [])
        )
    ;   format('  ✗ FAIL: factorial/2 compilation failed~n', []),
        fail
    ),

    % Clean up
    catch(abolish(factorial/2), _, true),
    catch(delete_file('/tmp/test_factorial_regression.sh'), _, true),
    format('[✓] Test 3 Passed~n', []),
    !.

:- initialization(main, main).
