% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

% This file initializes the UnifyWeaver educational environment.
% It should be loaded before running any of the examples.

:- dynamic user:file_search_path/2.
:- dynamic user:unifyweaver_root/1.

% Load necessary library for path manipulation
:- use_module(library(filesex)).

unifyweaver_education_init :-
    prolog_load_context(directory, Here),
    directory_file_path(Here, '..', ProjDir),
    retractall(user:unifyweaver_root(_)),
    assertz(user:unifyweaver_root(ProjDir)),
    directory_file_path(ProjDir, 'src', AbsSrcDir),
    directory_file_path(AbsSrcDir, 'unifyweaver', AbsUnifyweaverDir),
    asserta(user:file_search_path(unifyweaver, AbsUnifyweaverDir)),
    format('[UnifyWeaver] Educational environment initialized.~n', []),
    format('  Project root: ~w~n', [ProjDir]),
    format('  Absolute path to src: ~w~n', [AbsSrcDir]),
    format('  \'unifyweaver\' library alias configured for: ~w~n', [AbsUnifyweaverDir]),
    format('You can now use use_module(unifyweaver(...)) to load modules.~n').

:- initialization(unifyweaver_education_init, now).
