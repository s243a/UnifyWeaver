:- module(test_compiler_driver, [
    test_compiler_driver/0
]).

:- use_module(library(compiler_driver)).
:- use_module(library(lists)).
:- use_module(test_data). % Load the test data

% --- Test Runner ---

test_compiler_driver :-
    writeln('--- Testing Recursive Compiler Driver ---'),
    
    % 1. Compile the top-level predicate
    compile(ancestor/2, [output_dir('output/core_tests')], GeneratedScripts),
    
    % 2. Verify the generated scripts
    (   subset(['output/core_tests/parent.sh', 'output/core_tests/ancestor.sh'], GeneratedScripts) ->
        writeln('  ✅ PASS: Correct scripts were generated.')
    ;   writeln('  ❌ FAIL: Incorrect scripts were generated.'),
        halt(1)
    ),
    
    % 3. Generate a test runner script
    generate_test_runner(GeneratedScripts, 'output/core_tests/test_runner.sh'),
    
    % 4. Run the test runner
    writeln('--- Running Generated Test Runner ---'),
    shell('bash output/core_tests/test_runner.sh', ExitCode),
    (   ExitCode == 0 ->
        writeln('  ✅ PASS: Test runner executed successfully.')
    ;   writeln('  ❌ FAIL: Test runner failed.'),
        halt(1)
    ),
    
    writeln('--- Recursive Compiler Test Complete ---').

generate_test_runner(GeneratedScripts, TestRunnerPath) :-
    open(TestRunnerPath, write, Stream),
    write(Stream, '#!/bin/bash\n'),
    write(Stream, 'set -e\n'), % Exit on error
    
    % Source all the generated scripts
    forall(member(Script, GeneratedScripts),
           format(Stream, 'source ~w\n', [Script])),
    
    % Add test cases
    write(Stream, '\n'),
    write(Stream, 'echo "--- Running Tests ---"\n'),
    write(Stream, 'ancestor a c >/dev/null && echo "  ✅ PASS: ancestor a c" || (echo "  ❌ FAIL: ancestor a c"; exit 1)\n'),
    write(Stream, '! ancestor d a >/dev/null && echo "  ✅ PASS: ! ancestor d a" || (echo "  ❌ FAIL: ! ancestor d a"; exit 1)\n'),
    
    close(Stream).

% Helper to check if a list is a subset of another
subset([], _).
subset([H|T], List) :-
    member(H, List),
    subset(T, List).
