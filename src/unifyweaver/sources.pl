:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% sources.pl - Public interface for data sources
% Provides the source/3 predicate that users call in their code

:- module(sources, [source/3]).

:- use_module(library(option)).
:- use_module('core/dynamic_source_compiler').

%% source(+Type, +Name, +Options)
%  Public interface for defining a data source
%  Registers the source for later compilation
%
%  Example:
%    :- source(csv, users, [csv_file('data.csv'), has_header(true)]).
%
source(Type, Name, Options0) :-
    augment_source_options(Type, Options0, Options),
    % Determine arity from options or use defaults
    determine_arity(Options, Arity),
    validate_source_options(Type, Options, Arity),
    % Register this as a dynamic source
    register_dynamic_source(Name/Arity, Type, Options),
    format('Defined source: ~w/~w using ~w~n', [Name, Arity, Type]).

%% determine_arity(+Options, -Arity)
%  Determine predicate arity from options
determine_arity(Options, Arity) :-
    member(schema(_), Options),
    !,
    Arity = 1.
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

%% augment_source_options(+Type, +Options, -Augmented)
augment_source_options(Type, Options, Augmented) :-
    (   Type = csv
    ->  augment_csv_options(Options, Augmented)
    ;   Type = json
    ->  augment_json_options(Options, Augmented)
    ;   Augmented = Options
    ).

augment_csv_options(Options0, Options) :-
    option(delimiter(Delimiter), Options0, ','),
    ensure_option(field_separator(Delimiter), Options0, Options1),
    ensure_option(record_separator(line_feed), Options1, Options2),
    ensure_option(record_format(text_line), Options2, Options3),
    ensure_option(quote_style(double_quote), Options3, Options4),
    (   option(has_header(true), Options0),
        \+ option(skip_lines(_), Options0)
    ->  ensure_option(skip_lines(1), Options4, Options5)
    ;   Options5 = Options4
    ),
    (   option(csv_file(File), Options0)
    ->  absolute_file_name(File, Abs),
        ensure_option(input(file(Abs)), Options5, Options6)
    ;   Options6 = Options5
    ),
    Options = Options6.

augment_json_options(Options0, Options) :-
    ensure_option(record_format(json), Options0, Options1),
    ensure_option(record_separator(line_feed), Options1, Options2),
    (   option(type_hint(TypeHint), Options0)
    ->  ensure_option(type_hint(TypeHint), Options2, Options3)
    ;   Options3 = Options2
    ),
    (   option(return_object(Return), Options0)
    ->  ensure_option(return_object(Return), Options3, Options4)
    ;   Options4 = Options3
    ),
    (   option(json_file(File), Options0)
    ->  absolute_file_name(File, Abs),
        ensure_option(input(file(Abs)), Options4, Options5)
    ;   Options5 = Options4
    ),
    (   option(schema(Schema), Options0)
    ->  ensure_option(schema(Schema), Options5, Options6),
        ensure_option(return_object(true), Options6, Options7)
    ;   Options7 = Options5
    ),
    Options = Options7.

ensure_option(Term, Options0, Options) :-
    functor(Term, Name, Arity),
    (   member(Existing, Options0),
        functor(Existing, Name, Arity)
    ->  Options = Options0
    ;   Options = [Term|Options0]
    ).

validate_source_options(json, Options, Arity) :-
    !,
    validate_json_source_options(Options, Arity).
validate_source_options(_, _, _).

validate_json_source_options(Options, Arity) :-
    (   option(schema(Schema), Options)
    ->  validate_json_schema(Options, Schema, Arity)
    ;   option(return_object(true), Options)
    ->  validate_json_return_object(Options, Arity)
    ;   validate_json_columns(Options, Arity)
    ).

validate_json_return_object(Options, Arity) :-
    (   option(type_hint(Type), Options),
        Type \= ''
    ->  true
    ;   throw(error(domain_error(json_type_hint, Options), _))
    ),
    (   Arity =:= 1
    ->  true
    ;   throw(error(domain_error(json_return_object_arity, Arity), _))
    ),
    (   option(columns(Cols), Options),
        Cols \= []
    ->  throw(error(domain_error(json_return_object_columns, Cols), _))
    ;   true
    ).

validate_json_columns(Options, Arity) :-
    (   option(columns(Columns), Options),
        is_list(Columns),
        Columns \= []
    ->  validate_column_entries(Columns),
        length(Columns, Count),
        (   Count =:= Arity
        ->  true
        ;   throw(error(domain_error(json_columns_arity, json{columns:Columns, arity:Arity}), _))
        )
    ;   throw(error(domain_error(json_columns, Options), _))
    ).

validate_json_schema(Options, Schema, Arity) :-
    (   Arity =:= 1
    ->  true
    ;   throw(error(domain_error(json_schema_arity, Arity), _))
    ),
    (   member(arity(Declared), Options),
        Declared \= 1
    ->  throw(error(domain_error(json_schema_arity, Declared), _))
    ;   true
    ),
    validate_schema_fields(Schema),
    (   option(columns(_), Options)
    ->  throw(error(domain_error(json_schema_columns_conflict, Options), _))
    ;   true
    ).

validate_schema_fields(Schema) :-
    (   is_list(Schema)
    ->  maplist(validate_schema_field, Schema)
    ;   throw(error(domain_error(json_schema, Schema), _))
    ).

validate_schema_field(field(Name, Path, Type)) :-
    (   atom(Name)
    ->  true
    ;   string(Name)
    ->  true
    ;   throw(error(domain_error(json_schema_field_name, Name), _))
    ),
    (   atom(Path)
    ->  true
    ;   string(Path)
    ->  true
    ;   throw(error(domain_error(json_schema_field_path, Path), _))
    ),
    (   validate_schema_type(Type)
    ->  true
    ;   throw(error(domain_error(json_schema_field_type, Type), _))
    ).
validate_schema_field(Term) :-
    throw(error(domain_error(json_schema_field, Term), _)).

validate_schema_type(string).
validate_schema_type(integer).
validate_schema_type(long).
validate_schema_type(float).
validate_schema_type(double).
validate_schema_type(number).
validate_schema_type(boolean).
validate_schema_type(json).

validate_column_entries(Columns) :-
    maplist(validate_column_entry, Columns).

validate_column_entry(Column) :-
    (   atom(Column)
    ->  atom_string(Column, String)
    ;   string(Column)
    ->  String = Column
    ;   throw(error(domain_error(json_column_entry, Column), _))
    ),
    (   String == ""
    ->  throw(error(domain_error(json_column_entry, Column), _))
    ;   true
    ).
