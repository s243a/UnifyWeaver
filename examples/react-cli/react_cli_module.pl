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
        type(file_browser)
    ]),
    
    % Generate the React app code exactly as the prototype
    generate_react_component(react_cli, AppCode),
    open('src/App.tsx', write, S),
    write(S, AppCode),
    close(S),
    
    format('  Created: src/App.tsx~n'),
    format('Done! Run "npm run dev" to test.~n').
