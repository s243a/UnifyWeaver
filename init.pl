:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% init.pl - Minimal UnifyWeaver environment initialization
% This file sets up the Prolog environment for running UnifyWeaver scripts
%
% Usage:
%   swipl -f init.pl                    # Interactive session
%   swipl -f init.pl -g "goal, halt"    # Run goal and exit

:- dynamic user:file_search_path/2.
:- dynamic user:library_directory/1.
:- dynamic user:unifyweaver_root/1.

%% unifyweaver_init
%  Initialize the UnifyWeaver environment by setting up search paths
%  This predicate determines the project root from init.pl's location
%  and sets up all necessary paths relative to it
unifyweaver_init :-
    % Get the directory where init.pl is located (project root)
    prolog_load_context(directory, ProjectRoot),

    % Store project root for reference
    retractall(user:unifyweaver_root(_)),
    assertz(user:unifyweaver_root(ProjectRoot)),

    % Build paths relative to project root
    atom_concat(ProjectRoot, '/src', SrcDir),
    atom_concat(ProjectRoot, '/src/unifyweaver', UnifyweaverDir),

    % Set up file search paths
    % This allows use_module(library(unifyweaver/...))
    asserta(user:file_search_path(unifyweaver, UnifyweaverDir)),

    % This allows use_module(library(...)) for modules in src/
    asserta(user:library_directory(SrcDir)),

    % Confirmation message
    format('[UnifyWeaver] Environment initialized~n', []),
    format('  Project root: ~w~n', [ProjectRoot]),
    format('  Source directory: ~w~n', [SrcDir]),
    format('  UnifyWeaver modules: ~w~n', [UnifyweaverDir]).

% Run initialization when this file is loaded
:- initialization(unifyweaver_init, now).
