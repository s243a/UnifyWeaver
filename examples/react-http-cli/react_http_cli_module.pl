% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% react_http_cli_module.pl - React HTTP CLI App Generator
%
% Generates the React HTTP CLI app from declarative Prolog specifications
% using the react_generator framework. API configuration, authentication,
% tab layout, and browse roots are all defined as Prolog terms.

:- module(react_http_cli_module, [generate_all/0]).

:- use_module('../../src/unifyweaver/glue/react_generator').

generate_all :-
    format('Generating React HTTP CLI files...~n~n'),

    % Define the HTTP CLI component with full specification.
    % - auth/2 defines login fields with default values
    % - tabs/1 defines the tab bar (id, label, icon, options)
    % - browse_roots/1 defines the filesystem root selector options
    declare_ui_component(react_http_cli, [
        type(http_cli),
        title('UnifyWeaver CLI'),
        api_base('https://localhost:3001'),
        auth(required, [
            field(email, 'Email', 'shell@local'),
            field(password, 'Password', 'shell')
        ]),
        tabs([
            tab(browse, 'Browse', '📁', []),
            tab(upload, 'Upload', '📤', []),
            tab(cat, 'Cat', '📄', []),
            tab(shell, 'Shell', '🔐', [highlight])
        ]),
        browse_roots([
            root(sandbox, 'Sandbox'),
            root(project, 'Project'),
            root(home, 'Home')
        ]),
        default_root(sandbox),
        features([browse, upload, cat, shell, download, view_contents, set_working_dir])
    ]),

    % Generate the React app code from the above specification
    generate_react_component(react_http_cli, AppCode),
    open('src/App.tsx', write, S),
    write(S, AppCode),
    close(S),

    format('  Created: src/App.tsx~n'),
    format('Done! Run "npm run dev" to test.~n').
