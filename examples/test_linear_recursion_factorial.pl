:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_linear_recursion_factorial.pl - Test factorial linear recursion compilation

:- use_module('../src/unifyweaver/core/advanced/advanced_recursive_compiler').

main :-
    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  Testing Linear Recursion: Factorial                  ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n~n', []),

    % Define factorial
    catch(abolish(factorial/2), _, true),
    assertz((factorial(0, 1))),
    assertz((factorial(N, F) :- N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1)),

    format('Predicate defined:~n', []),
    format('  factorial(0, 1).~n', []),
    format('  factorial(N, F) :- N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1.~n~n', []),

    % Try to compile
    format('Attempting to compile factorial/2...~n', []),
    (   compile_advanced_recursive(factorial/2, [], Code)
    ->  format('~n✓ Compilation succeeded!~n~n', []),
        format('Generated code length: ~w characters~n~n', [string_length(Code, _)]),

        % Write to file
        open('output/factorial.sh', write, Stream),
        write(Stream, Code),
        close(Stream),
        format('✓ Code written to output/factorial.sh~n~n', []),

        % Try to execute it
        format('Testing execution with factorial(5)...~n', []),
        catch(
            process_create('/bin/bash', ['output/factorial.sh', '5', ''], [stdout(pipe(Output))]),
            Error,
            (format('Execution error: ~w~n', [Error]), fail)
        ),
        (   Output \= pipe(_) ->
            read_string(Output, _, Result),
            close(Output),
            format('Result: ~w~n', [Result])
        ;   format('Could not read output~n', [])
        )
    ;   format('~n✗ Compilation failed~n', []),
        fail
    ),

    % Clean up
    catch(abolish(factorial/2), _, true),

    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  Test Complete                                         ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n', []),
    halt(0).

main :-
    format('~n[✗] Test failed~n', []),
    halt(1).

:- initialization(main, main).
