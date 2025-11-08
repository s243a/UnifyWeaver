:- module(test_perl_service, [run_tests/0, run_tests/1]).

% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_perl_service.pl - Unit tests for the perl_service module.

:- asserta(user:file_search_path(library, 'src/unifyweaver/core')).
:- asserta(user:file_search_path(library, 'tests/core')).

:- use_module(library(plunit)).
:- use_module(library(perl_service)).

:- begin_tests(perl_service).

test(inline_perl_call) :-
    PerlCode = 'print join("-", @ARGV);',
    Args = ['a', 'b', 'c'],
    InputVar = 'SomeData',
    generate_inline_perl_call(PerlCode, Args, InputVar, BashCode),
    
    % Write the bash script to a temporary file
    tmp_file(test_perl, TmpFile),
    setup_call_cleanup(
        open(TmpFile, write, Stream),
        format(Stream, '#!/bin/bash~n~w', [BashCode]),
        close(Stream)
    ),

    % Make it executable
    process_create(path(chmod), ['+x', TmpFile], []),

    % Run the script and capture output
    setup_call_cleanup(
        process_create(path(bash), [TmpFile], [stdout(pipe(Out))]),
        read_string(Out, _, Output),
        close(Out)
    ),

    % Verify the output
    Output = "a-b-c".

:- end_tests(perl_service).
