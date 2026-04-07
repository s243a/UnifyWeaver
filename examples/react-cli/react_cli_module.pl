% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% react_cli_module.pl - React CLI App Generator
%
% Generates the React CLI app using the declarative react_generator.

:- module(react_cli_module, [generate_all/0]).

:- use_module('../../src/unifyweaver/glue/react_generator').

generate_all :-
    format('Generating React CLI files...~n~n'),
    
    % Define the CLI component
    declare_ui_component(react_cli, [
        type(form),
        title("UnifyWeaver CLI"),
        description("Browser-based command line interface"),
        inputs([
            input(path, string, "Working Directory", [default("/home/user/projects")]),
            input(command, string, "Command", [placeholder("Enter shell command...")])
        ]),
        operations([
            operation(execute, '/execute', "Run Command"),
            operation(list, '/ls', "List Directory")
        ])
    ]),
    
    % Generate the React app code (only include the declared component)
    generate_react_app(react_cli, [components([react_cli])], AppCode),
    make_directory_path('generated/src'),
    open('generated/src/App.tsx', write, S),
    write(S, AppCode),
    close(S),

    format('  Created: generated/src/App.tsx~n'),
    format('Done! Compare with prototype/ and run "npm run dev" to test.~n').
