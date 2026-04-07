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
        type(form),
        title("UnifyWeaver HTTP CLI"),
        description("Remote HTTP-based command line interface"),
        inputs([
            input(endpoint, string, "API Endpoint", [default("https://localhost:3001")]),
            input(command, string, "Command", [placeholder("Enter remote command...")])
        ]),
        operations([
            operation(send, '/api/run', "Execute Remote")
        ])
    ]),
    generate_react_app(react_http_cli, AppCode),
    open('src/App.tsx', write, S),
    write(S, AppCode),
    close(S),
    format('  Created: src/App.tsx~n'),
    
    % 2. Generate HTML interface via html_interface_generator
    http_cli_interface(Spec),
    http_cli_theme(Theme),
    generate_html_interface(Spec, Theme, HTML),
    open('index.html', write, H),
    write(H, HTML),
    close(H),
    format('  Created: index.html~n'),
    
    format('Done! Run "npm run dev" to test.~n').
