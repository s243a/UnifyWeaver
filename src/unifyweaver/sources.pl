:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% sources.pl - Public interface for data sources
% Provides the source/3 predicate that users call in their code

:- module(sources, [source/3]).

:- use_module('core/dynamic_source_compiler').

%% source(+Type, +Name, +Options)
%  Public interface for defining a data source
%  Registers the source for later compilation
%
%  Example:
%    :- source(csv, users, [csv_file('data.csv'), has_header(true)]).
%
source(Type, Name, Options) :-
    % Determine arity from options or use defaults
    determine_arity(Options, Arity),
    
    % Register this as a dynamic source
    register_dynamic_source(Name/Arity, Type, Options),
    
    format('Defined source: ~w/~w using ~w~n', [Name, Arity, Type]).

%% determine_arity(+Options, -Arity)
%  Determine predicate arity from options
determine_arity(Options, Arity) :-
    % Check if arity is explicitly specified
    (   member(arity(Arity), Options)
    ->  true
    % Check if columns are specified
    ;   member(columns(Cols), Options)
    ->  length(Cols, Arity)
    % For CSV with header detection
    ;   member(has_header(true), Options),
        member(csv_file(File), Options)
    ->  detect_csv_arity(File, Options, Arity)
    % Default arity
    ;   Arity = 2
    ).

%% detect_csv_arity(+File, +Options, -Arity)
%  Detect arity from CSV file header
detect_csv_arity(File, Options, Arity) :-
    catch(
        (   open(File, read, Stream),
            read_line_to_string(Stream, FirstLine),
            close(Stream),
            % Get delimiter
            (   member(delimiter(Delim), Options)
            ->  true
            ;   Delim = ','
            ),
            % Split and count columns
            split_string(FirstLine, Delim, ' "', Columns),
            length(Columns, Arity)
        ),
        _Error,
        Arity = 2  % Default on error
    ).
