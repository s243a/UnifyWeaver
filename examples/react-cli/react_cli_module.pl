% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% react_cli_module.pl - React CLI App Generator
%
% Generates the React CLI app from declarative Prolog specifications
% using the react_generator framework. The mock filesystem, initial
% state, and features are all defined here as Prolog terms.

:- module(react_cli_module, [generate_all/0]).

:- use_module('../../src/unifyweaver/glue/react_generator').

generate_all :-
    format('Generating React CLI files...~n~n'),

    % Define the file browser component with full specification.
    % The mock_fs/1 entries define the demo filesystem hierarchy;
    % the generator converts these Prolog terms into a JavaScript
    % Record<string, FileEntry[]> object.
    declare_ui_component(react_cli, [
        type(file_browser),
        initial_path('/home/user/projects'),
        root_path('/home/user'),
        mock_fs([
            fs_dir('/home/user', [
                entry(projects, directory, 4096),
                entry(documents, directory, 4096),
                entry('.bashrc', file, 220)
            ]),
            fs_dir('/home/user/projects', [
                entry(src, directory, 4096),
                entry(docs, directory, 4096),
                entry('package.json', file, 1240),
                entry('README.md', file, 3500),
                entry('tsconfig.json', file, 562),
                entry('vite.config.ts', file, 180)
            ]),
            fs_dir('/home/user/projects/src', [
                entry('App.tsx', file, 2400),
                entry('main.tsx', file, 150),
                entry('index.css', file, 800)
            ]),
            fs_dir('/home/user/projects/docs', [
                entry('GUIDE.md', file, 5200),
                entry('API.md', file, 8900)
            ]),
            fs_dir('/home/user/documents', [
                entry('notes.txt', file, 450),
                entry('todo.md', file, 120)
            ])
        ]),
        features([search, download, view_contents, set_working_dir, notifications])
    ]),

    % Generate the React app code from the above specification
    generate_react_component(react_cli, AppCode),
    open('src/App.tsx', write, S),
    write(S, AppCode),
    close(S),

    format('  Created: src/App.tsx~n'),
    format('Done! Run "npm run dev" to test.~n').
