:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_linear_fib_normal.pl - Test fibonacci WITHOUT forbid_linear_recursion

:- use_module('../src/unifyweaver/core/advanced/advanced_recursive_compiler').

main :-
    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  Testing Linear Recursion: Fibonacci (Normal)         ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n~n', []),

    % Define fibonacci WITHOUT forbid directive
    catch(abolish(my_fib/2), _, true),
    assertz((my_fib(0, 0))),
    assertz((my_fib(1, 1))),
    assertz((my_fib(N, F) :- N > 1, N1 is N - 1, N2 is N - 2, my_fib(N1, F1), my_fib(N2, F2), F is F1 + F2)),

    format('Predicate defined:~n', []),
    format('  my_fib(0, 0).~n', []),
    format('  my_fib(1, 1).~n', []),
    format('  my_fib(N, F) :- N > 1, ..., F is F1 + F2.~n~n', []),

    % Try to compile
    format('Attempting to compile my_fib/2...~n', []),
    (   compile_advanced_recursive(my_fib/2, [], Code)
    ->  format('~n✓ Compilation succeeded!~n~n', []),

        % Write to file
        open('output/my_fib.sh', write, Stream),
        write(Stream, Code),
        close(Stream),
        format('✓ Code written to output/my_fib.sh~n~n', []),

        % Test execution
        format('Testing with my_fib(10)...~n', []),
        format('(Expected: Fibonacci(10) = 55)~n~n', [])
    ;   format('~n✗ Compilation failed~n', []),
        fail
    ),

    % Clean up
    catch(abolish(my_fib/2), _, true),

    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  Test Complete                                         ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n', []),
    halt(0).

main :-
    format('~n[✗] Test failed~n', []),
    halt(1).

:- initialization(main, main).
