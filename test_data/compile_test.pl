#!/usr/bin/env swipl
%% compile_test.pl - Test compilation of semantic source

:- initialization(main, main).

:- use_module('../examples/semantic_source_test').
:- use_module('../src/unifyweaver/core/dynamic_source_compiler').

main(_Argv) :-
    writeln('=== Compiling Semantic Source Test ==='),
    nl,

    % Get the source configuration
    writeln('Source: test_papers/2'),
    writeln('Backend: python_onnx'),
    writeln('Target: bash'),
    nl,

    % Compile the source
    writeln('Compiling...'),
    (   compile_dynamic_source(test_papers/2, [target(bash)], BashCode)
    ->  writeln('✓ Compilation successful'),
        nl,

        % Save to file
        OutputFile = 'test_papers.sh',
        open(OutputFile, write, Stream),
        write(Stream, BashCode),
        close(Stream),

        writeln('Generated code saved to: test_papers.sh'),
        nl,

        % Show first few lines
        writeln('First 20 lines of generated code:'),
        writeln('---'),
        split_string(BashCode, "\n", "", Lines),
        take_n(20, Lines, FirstLines),
        maplist(writeln, FirstLines),
        writeln('...'),
        nl,

        writeln('✓ Test compilation complete')
    ;   writeln('✗ Compilation failed'),
        halt(1)
    ),

    halt(0).

take_n(0, _, []) :- !.
take_n(_, [], []) :- !.
take_n(N, [H|T], [H|Rest]) :-
    N > 0,
    N1 is N - 1,
    take_n(N1, T, Rest).
