:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% dynamic_source_compiler.pl - Core infrastructure for dynamic data sources
% Provides plugin registration and compilation dispatch

:- module(dynamic_source_compiler, [
    register_source_type/2,         % +Type, +Module
    compile_dynamic_source/3,       % +Pred/Arity, +Options, -BashCode
    is_dynamic_source/1,            % +Pred/Arity
    list_source_types/1,            % -Types
    register_dynamic_source/3,      % +Pred/Arity, +SourceSpec, +Options
    dynamic_source_def/3,           % +Pred/Arity, -Type, -Config (multifile)
    dynamic_source_metadata/2,      % +Pred/Arity, -Metadata dict
    test_dynamic_source_compiler/0
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module(firewall).

%% ============================================
%% DYNAMIC PREDICATES
%% ============================================

:- dynamic source_type_registry/2.  % source_type_registry(Type, Module)
:- dynamic dynamic_source_def/3.     % dynamic_source_def(Pred/Arity, Type, Config)
:- multifile dynamic_source_def/3.   % Allow plugins to add facts
:- dynamic dynamic_source_metadata/2. % dynamic_source_metadata(Pred/Arity, MetaDict)

%% ============================================
%% SOURCE TYPE REGISTRATION
%% ============================================

%% register_source_type(+Type, +Module)
%  Register a source plugin module
%  Called by plugins during initialization
register_source_type(Type, Module) :-
    (   source_type_registry(Type, ExistingModule)
    ->  (   ExistingModule = Module
        ->  true  % Already registered by same module
        ;   format('Warning: Source type ~w already registered by ~w, overriding with ~w~n',
                   [Type, ExistingModule, Module]),
            retract(source_type_registry(Type, ExistingModule)),
            assertz(source_type_registry(Type, Module))
        )
    ;   assertz(source_type_registry(Type, Module)),
        format('Registered source type: ~w -> ~w~n', [Type, Module])
    ).

%% list_source_types(-Types)
%  Get list of all registered source types
list_source_types(Types) :-
    findall(Type, source_type_registry(Type, _), Types).

%% get_source_module(+Type, -Module)
%  Get module for a registered source type
get_source_module(Type, Module) :-
    source_type_registry(Type, Module),
    !.
get_source_module(Type, _) :-
    format('Error: Source type ~w not registered~n', [Type]),
    fail.

%% ============================================
%% DYNAMIC SOURCE DEFINITION
%% ============================================

%% register_dynamic_source(+Pred/Arity, +SourceSpec, +Options)
%  Register a predicate as using a dynamic source
%  SourceSpec is either:
%    - Type (atom) - e.g., awk
%    - type(Type, Config) - e.g., type(awk, [command='awk ...'])
register_dynamic_source(Pred/Arity, SourceSpec, Options) :-
    % Parse SourceSpec
    (   SourceSpec = type(Type, Config)
    ->  true
    ;   Type = SourceSpec,
        Config = []
    ),

    % Merge config with options
    append(Config, Options, MergedConfig),

    % Enforce firewall policy at declaration time
    enforce_firewall_policy(Pred/Arity, MergedConfig),

    % Store definition
    retractall(dynamic_source_def(Pred/Arity, _, _)),
    assertz(dynamic_source_def(Pred/Arity, Type, MergedConfig)),

    % Store normalized IO metadata
    extract_io_metadata(Pred/Arity, MergedConfig, Metadata),
    retractall(dynamic_source_metadata(Pred/Arity, _)),
    assertz(dynamic_source_metadata(Pred/Arity, Metadata)),

    format('Registered dynamic source: ~w/~w using ~w~n', [Pred, Arity, Type]).

%% is_dynamic_source(+Pred/Arity)
%  Check if predicate is registered as dynamic source
is_dynamic_source(Pred/Arity) :-
    dynamic_source_def(Pred/Arity, _, _).

%% ============================================
%% COMPILATION
%% ============================================

%% compile_dynamic_source(+Pred/Arity, +Options, -BashCode)
%  Compile a dynamic source predicate to bash code
compile_dynamic_source(Pred/Arity, RuntimeOptions, BashCode) :-
    % Get source definition
    dynamic_source_def(Pred/Arity, Type, Config),

    % Get source plugin module
    get_source_module(Type, Module),

    % Merge runtime options with stored config
    append(RuntimeOptions, Config, AllOptions),

    % Re-validate against firewall in case runtime options add new capabilities
    enforce_firewall_policy(Pred/Arity, AllOptions),

    format('Compiling dynamic source: ~w/~w using ~w~n', [Pred, Arity, Type]),

    % Call plugin's compile predicate
    % Plugin interface: compile_source(+Pred/Arity, +Config, +Options, -BashCode)
    Module:compile_source(Pred/Arity, Config, AllOptions, BashCode).

%% ============================================
%% FIREWALL ENFORCEMENT
%% ============================================

enforce_firewall_policy(PredIndicator, Options) :-
    firewall:get_firewall_policy(PredIndicator, Firewall),
    resolve_target(Options, Target),
    firewall:enforce_firewall(PredIndicator, Target, Options, Firewall).

resolve_target(Options, Target) :-
    (   member(target(TargetOption), Options)
    ->  Target = TargetOption
    ;   Target = bash
    ).

%% ============================================
%% TESTS
%% ============================================

test_dynamic_source_compiler :-
    writeln('=== Testing Dynamic Source Compiler ==='),

    % Test 1: Register source type
    write('Test 1 - Register source type: '),
    register_source_type(test_source, test_module),
    (   source_type_registry(test_source, test_module)
    ->  writeln('PASS')
    ;   writeln('FAIL')
    ),

    % Test 2: List source types
    write('Test 2 - List source types: '),
    list_source_types(Types),
    (   member(test_source, Types)
    ->  writeln('PASS')
    ;   writeln('FAIL')
    ),

    % Test 3: Register dynamic source
    write('Test 3 - Register dynamic source: '),
    register_dynamic_source(my_data/2, test_source,
        [ option1=value1,
          record_separator(nul),
          field_separator(','),
          input(file('data.csv'))
        ]),
    (   is_dynamic_source(my_data/2)
    ->  writeln('PASS')
    ;   writeln('FAIL')
    ),

    % Test 4: Check source definition
    write('Test 4 - Check source definition: '),
    (   dynamic_source_def(my_data/2, test_source, Config),
        member(option1=value1, Config)
    ->  writeln('PASS')
    ;   writeln('FAIL')
    ),

    write('Test 5 - Metadata extraction: '),
    (   dynamic_source_metadata(my_data/2, Meta),
        Meta.record_separator == nul,
        Meta.field_separator == ',',
        Meta.input == file('data.csv')
    ->  writeln('PASS')
    ;   writeln('FAIL')
    ),

    % Clean up
    retractall(source_type_registry(test_source, _)),
    retractall(dynamic_source_def(my_data/2, _, _)),
    retractall(dynamic_source_metadata(my_data/2, _)),

    writeln('=== Dynamic Source Compiler Tests Complete ===').

%% extract_io_metadata(+Options, -Metadata:dict)
%  Normalizes commonly-used IO descriptors so downstream targets
%  understand how records are streamed into the predicate.
extract_io_metadata(Pred/Arity, Options, Meta) :-
    option(record_separator(RawRecordSep), Options, nul),
    normalize_record_separator(RawRecordSep, RecordSep),
    option(field_separator(RawFieldSep), Options, ':'),
    normalize_field_separator(RawFieldSep, FieldSep),
    option(record_format(RawFormat), Options, text),
    normalize_record_format(RawFormat, RecordFormat),
    option(input(RawInput), Options, stdin),
    normalize_input(RawInput, Input),
    option(skip_lines(SkipRows), Options, 0),
    option(quote_style(RawQuote), Options, none),
    normalize_quote_style(RawQuote, QuoteStyle),
    (   option(columns(RawColumns), Options)
    ->  ColumnsInput = RawColumns
    ;   ColumnsInput = []
    ),
    normalize_columns(ColumnsInput, Columns, ColumnSelectors),
    option(type_hint(TypeHint0), Options, none),
    normalize_type_hint(TypeHint0, TypeHintRaw),
    option(return_object(ReturnObject0), Options, false),
    normalize_boolean(ReturnObject0, ReturnObject),
    (   option(schema(SchemaRaw), Options)
    ->  schema_record_name(Pred/Arity, Options, RecordType),
        normalize_schema_structure(SchemaRaw, RecordType, SchemaFields, SchemaRecords),
        format(atom(SchemaFullName), 'UnifyWeaver.Generated.~w', [RecordType])
    ;   SchemaFields = [],
        SchemaRecords = [],
        RecordType = none,
        SchemaFullName = none
    ),
    (   TypeHintRaw = none,
        SchemaFullName \= none
    ->  TypeHint = SchemaFullName
    ;   TypeHint = TypeHintRaw
    ),
    Meta = metadata{
        record_separator:RecordSep,
        field_separator:FieldSep,
        record_format:RecordFormat,
        input:Input,
        skip_rows:SkipRows,
        quote_style:QuoteStyle,
        columns:Columns,
        column_selectors:ColumnSelectors,
        type_hint:TypeHint,
        return_object:ReturnObject,
        schema_fields:SchemaFields,
        schema_records:SchemaRecords,
        schema_type:RecordType
    }.

normalize_record_separator(line_feed, line_feed) :- !.
normalize_record_separator(nul, nul) :- !.
normalize_record_separator(json, json) :- !.
normalize_record_separator(Atom, Atom).

normalize_field_separator(none, none) :- !.
normalize_field_separator(comma, ',') :- !.
normalize_field_separator(tab, '\t') :- !.
normalize_field_separator(Value, Value).

normalize_record_format(text, text_line) :- !.
normalize_record_format(json, json) :- !.
normalize_record_format(Format, Format).

normalize_input(stdin, stdin) :- !.
normalize_input(file(Path0), file(Path)) :- !,
    expand_file_name(Path0, [Path|_]).
normalize_input(pipe(Command), pipe(Command)) :- !.
normalize_input(Value, Value).

normalize_quote_style(none, none) :- !.
normalize_quote_style(double_quote, double_quote) :- !.
normalize_quote_style(single_quote, single_quote) :- !.
normalize_quote_style(json, json_escape) :- !.
normalize_quote_style(Value, Value).

normalize_type_hint(none, none) :- !.
normalize_type_hint(Type, Normalized) :-
    (   atom(Type)
    ->  Normalized = Type
    ;   string(Type)
    ->  atom_string(Normalized, Type)
    ;   term_to_atom(Type, Normalized)
    ).

normalize_boolean(true, true) :- !.
normalize_boolean(false, false) :- !.
normalize_boolean(1, true) :- !.
normalize_boolean(0, false) :- !.
normalize_boolean(yes, true) :- !.
normalize_boolean(no, false) :- !.
normalize_boolean(on, true) :- !.
normalize_boolean(off, false) :- !.
normalize_boolean(Value, Value).

normalize_columns([], [], []) :- !.
normalize_columns(Columns0, Columns, Selectors) :-
    is_list(Columns0),
    !,
    maplist(normalize_column_selector, Columns0, Columns, Selectors).
normalize_columns(Column, [Name], [Selector]) :-
    normalize_column_selector(Column, Name, Selector).

normalize_column_selector(Value, NameAtom, Selector) :-
    normalize_selector_entry(Value, PathString, Kind),
    atom_string(NameAtom, PathString),
    Selector = column_selector{path:PathString, kind:Kind}.

normalize_selector_entry(jsonpath(PathTerm), PathString, jsonpath) :-
    !,
    normalize_selector_path(PathTerm, PathString).
normalize_selector_entry(Value, PathString, Kind) :-
    normalize_selector_path(Value, PathString),
    (   sub_string(PathString, 0, 1, _, "$")
    ->  Kind = jsonpath
    ;   Kind = column_path
    ).

normalize_selector_path(Value, String) :-
    (   atom(Value)
    ->  atom_string(Value, String)
    ;   string(Value)
    ->  String = Value
    ;   throw(error(domain_error(json_column_entry, Value), _))
    ).

normalize_schema_structure(SchemaRaw, RootType, Fields, [schema_record{type:RootType, fields:Fields}|NestedRecords]) :-
    normalize_schema_fields(SchemaRaw, Fields, [], NestedRecords).

normalize_schema_fields([], [], Acc, Acc) :- !.
normalize_schema_fields([field(Name, Path, Type)|Rest], [FieldDict|Tail], AccIn, AccOut) :-
    normalize_schema_name(Name, NameAtom),
    normalize_schema_path(Path, PathString, Kind),
    normalize_schema_type(Type, NameAtom, FieldMeta, AccIn, AccNext),
    FieldDict = FieldMeta.put(_{
        name:NameAtom,
        path:PathString,
        selector_kind:Kind
    }),
    normalize_schema_fields(Rest, Tail, AccNext, AccOut).
normalize_schema_fields([Invalid|_], _, _, _) :-
    throw(error(domain_error(json_schema_field, Invalid), _)).

normalize_schema_name(Name, Atom) :-
    (   atom(Name)
    ->  Atom = Name
    ;   string(Name)
    ->  atom_string(Atom, Name)
    ).

normalize_schema_path(jsonpath(Value), String, jsonpath) :-
    !,
    normalize_selector_path(Value, String).
normalize_schema_path(Path, String, Kind) :-
    normalize_selector_path(Path, String),
    (   sub_string(String, 0, 1, _, "$")
    ->  Kind = jsonpath
    ;   Kind = column_path
    ).

normalize_schema_type(record(Fields), FieldName, FieldDict, AccIn, AccOut) :-
    default_nested_record_type(FieldName, RecordType),
    normalize_schema_type(record(RecordType, Fields), FieldName, FieldDict, AccIn, AccOut).
normalize_schema_type(record(RecordType, Fields), _FieldName, FieldDict, AccIn, AccOut) :-
    (   is_list(Fields),
        Fields \= []
    ->  normalize_schema_fields(Fields, NestedFields, AccIn, NestedRecords),
        FieldDict = schema_field{
            column_type:json,
            field_kind:record,
            record_type:RecordType,
            nested_fields:NestedFields
        },
        AccOut = [schema_record{type:RecordType, fields:NestedFields}|NestedRecords]
    ;   throw(error(domain_error(json_schema_record_fields, Fields), _))
    ).
normalize_schema_type(Type, _FieldName, FieldDict, Records, Records) :-
    (   atom(Type)
    ->  atom_string(Type, String)
    ;   string(Type)
    ->  String = Type
    ),
    string_lower(String, Lower),
    atom_string(Normalized, Lower),
    FieldDict = schema_field{
        column_type:Normalized,
        field_kind:value,
        record_type:none,
        nested_fields:[]
    }.

schema_record_name(Pred/_, Options, RecordType) :-
    (   option(record_type(Type), Options)
    ->  RecordType = Type
    ;   default_record_type(Pred, RecordType)
    ).

default_record_type(Pred, RecordType) :-
    atom_string(Pred, PredStr),
    split_string(PredStr, '_', '_', Parts),
    maplist(capitalise_string, Parts, Caps),
    atomic_list_concat(Caps, '', Base),
    atom_concat(Base, 'Row', Temp),
    atom_string(RecordType, Temp).

default_nested_record_type(FieldName, RecordType) :-
    atom_string(FieldName, NameStr),
    split_string(NameStr, '_', '_', Parts),
    maplist(capitalise_string, Parts, Caps),
    atomic_list_concat(Caps, '', Base),
    atom_concat(Base, 'Record', Temp),
    atom_string(RecordType, Temp).

capitalise_string(Input, Output) :-
    (   Input = ''
    ->  Output = ''
    ;   string_lower(Input, Lower),
        sub_string(Lower, 0, 1, _, First),
        sub_string(Lower, 1, _, 0, Rest),
        string_upper(First, Upper),
        string_concat(Upper, Rest, Output)
    ).
