% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% react_http_cli_module.pl - React HTTP CLI App Generator
%
% Generates the React HTTP CLI app using both react_generator 
% and html_interface_generator.

:- module(react_http_cli_module, [generate_all/0]).

:- use_module('../../src/unifyweaver/glue/react_generator').
:- use_module('../../src/unifyweaver/ui/html_interface_generator').
:- use_module('../../src/unifyweaver/ui/http_cli_ui').

generate_all :-
    format('Generating React HTTP CLI files...~n~n'),
    
    % 1. Generate React component via react_generator
    declare_ui_component(react_http_cli, [
        type(http_cli)
    ]),
    generate_react_component(react_http_cli, AppCode),
    open('src/App.tsx', write, S),
    write(S, AppCode),
    close(S),
    format('  Created: src/App.tsx~n'),

    format('Done! Run "npm run dev" to test.~n').
