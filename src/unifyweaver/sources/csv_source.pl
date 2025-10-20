:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% csv_source.pl - CSV/TSV source plugin for dynamic sources
% Compiles predicates that read data from CSV/TSV files

:- module(csv_source, [
    compile_source/4,          % +Pred/Arity, +Config, +Options, -BashCode
    validate_config/1,         % +Config
    source_info/1              % -Info
]).

:- use_module(library(lists)).
:- use_module('../core/template_system').
:- use_module('../core/dynamic_source_compiler').

%% Register this plugin on load
:- initialization(
    register_source_type(csv, csv_source),
    now
).

%% ============================================
%% PLUGIN INTERFACE
%% ============================================

%% source_info(-Info)
%  Provide information about this source plugin
source_info(info(
    name('CSV/TSV Source'),
    version('1.0.0'),
    description('Read data from CSV and TSV files with header auto-detection'),
    supported_arities([1, 2, 3, 4, 5])
)).

%% validate_config(+Config)
%  Validate configuration for CSV source
validate_config(Config) :-
    % Must have csv_file
    (   member(csv_file(File), Config),
        (   exists_file(File)
        ->  true
        ;   format('Warning: CSV file ~w does not exist~n', [File])
        )
    ->  true
    ;   format('Error: CSV source requires csv_file(File)~n', []),
        fail
    ),
    
    % Validate delimiter if specified
    (   member(delimiter(Delim), Config)
    ->  (   atom_length(Delim, 1)
        ->  true
        ;   format('Error: delimiter must be single character, got ~w~n', [Delim]),
            fail
        )
    ;   true
    ),
    
    % Validate skip_lines if specified
    (   member(skip_lines(N), Config)
    ->  (   integer(N), N >= 0
        ->  true
        ;   format('Error: skip_lines must be non-negative integer, got ~w~n', [N]),
            fail
        )
    ;   true
    ).

%% compile_source(+Pred/Arity, +Config, +Options, -BashCode)
%  Compile CSV source to bash code
compile_source(Pred/Arity, Config, Options, BashCode) :-
    format('  Compiling CSV source: ~w/~w~n', [Pred, Arity]),

    % Validate configuration
    validate_config(Config),

    % Merge config and options
    append(Config, Options, AllOptions),

    % Extract required parameters
    member(csv_file(CsvFile), AllOptions),
    
    % Extract optional parameters with defaults
    (   member(delimiter(Delimiter), AllOptions)
    ->  true
    ;   Delimiter = ','  % Default delimiter
    ),
    (   member(skip_lines(SkipLines), AllOptions)
    ->  true
    ;   SkipLines = 0
    ),
    (   member(quote_char(QuoteChar), AllOptions)
    ->  true
    ;   QuoteChar = '"'
    ),
    (   member(quote_handling(QuoteHandling), AllOptions)
    ->  true
    ;   QuoteHandling = strip
    ),

    % Determine if we have header auto-detection
    (   member(has_header(true), AllOptions)
    ->  HeaderMode = auto,
        detect_csv_headers(CsvFile, Delimiter, DetectedColumns),
        (   length(DetectedColumns, DetectedArity),
            DetectedArity =:= Arity
        ->  Columns = DetectedColumns
        ;   format('Warning: Detected ~w columns but arity is ~w~n', [DetectedColumns, Arity]),
            generate_default_columns(Arity, Columns)
        )
    ;   member(columns(ManualColumns), AllOptions)
    ->  HeaderMode = manual,
        Columns = ManualColumns,
        length(Columns, ManualArity),
        (   ManualArity =:= Arity
        ->  true
        ;   format('Error: columns list length (~w) does not match arity (~w)~n', [ManualArity, Arity]),
            fail
        )
    ;   HeaderMode = positional,
        generate_default_columns(Arity, Columns)
    ),

    % Generate bash code using template
    atom_string(Pred, PredStr),
    generate_csv_bash(PredStr, Arity, CsvFile, Delimiter, SkipLines, 
                     QuoteChar, QuoteHandling, HeaderMode, Columns, BashCode).

%% ============================================
%% HEADER DETECTION
%% ============================================

%% detect_csv_headers(+File, +Delimiter, -Headers)
%  Auto-detect column headers from first line of CSV file
detect_csv_headers(File, Delimiter, Headers) :-
    catch(
        (   open(File, read, Stream),
            read_line_to_string(Stream, FirstLine),
            close(Stream),
            split_string(FirstLine, Delimiter, ' "', HeaderStrings),
            maplist(string_to_atom, HeaderStrings, Headers)
        ),
        Error,
        (   format('Error reading CSV headers from ~w: ~w~n', [File, Error]),
            Headers = []
        )
    ).

%% generate_default_columns(+Arity, -Columns)
%  Generate default column names: col1, col2, etc.
generate_default_columns(Arity, Columns) :-
    numlist(1, Arity, Numbers),
    maplist(default_column_name, Numbers, Columns).

default_column_name(N, ColName) :-
    format(atom(ColName), 'col~w', [N]).

%% ============================================
%% BASH CODE GENERATION
%% ============================================

%% generate_csv_bash(+PredStr, +Arity, +File, +Delimiter, +SkipLines, 
%%                   +QuoteChar, +QuoteHandling, +HeaderMode, +Columns, -BashCode)
%  Generate bash code for CSV source
generate_csv_bash(PredStr, Arity, File, Delimiter, SkipLines, 
                  QuoteChar, QuoteHandling, HeaderMode, Columns, BashCode) :-
    
    % Calculate total lines to skip (header + skip_lines)
    (   HeaderMode = auto
    ->  TotalSkip is SkipLines + 1
    ;   TotalSkip = SkipLines
    ),
    
    % Generate column output format
    generate_output_format(Arity, OutputFormat),
    
    % Generate quote handling code
    generate_quote_handling_code(QuoteChar, QuoteHandling, QuoteCode),
    
    % Escape delimiter for awk
    escape_delimiter(Delimiter, EscapedDelimiter),
    
    % Create column list for comments
    atomic_list_concat(Columns, ', ', ColumnList),
    
    % Render template based on arity
    (   Arity =:= 1 ->
        render_named_template(csv_source_unary,
            [pred=PredStr, file=File, delimiter=EscapedDelimiter, 
             skip_lines=TotalSkip, quote_code=QuoteCode, 
             columns=ColumnList],
            [source_order([file, generated])],
            BashCode)
    ;   render_named_template(csv_source_binary_plus,
            [pred=PredStr, file=File, delimiter=EscapedDelimiter,
             skip_lines=TotalSkip, quote_code=QuoteCode,
             output_format=OutputFormat, columns=ColumnList, arity=Arity],
            [source_order([file, generated])],
            BashCode)
    ).

%% generate_output_format(+Arity, -Format)
%  Generate awk output format for given arity
generate_output_format(Arity, Format) :-
    numlist(1, Arity, Numbers),
    maplist(field_reference, Numbers, Fields),
    atomic_list_concat(Fields, '":"', Format).

field_reference(N, Field) :-
    format(atom(Field), '$~w', [N]).

%% generate_quote_handling_code(+QuoteChar, +Handling, -Code)
%  Generate awk code for quote handling
generate_quote_handling_code(QuoteChar, strip, Code) :-
    format(atom(Code), 'gsub(/~w/, "", $0)', [QuoteChar]).
generate_quote_handling_code(_QuoteChar, preserve, '') :-
    % No quote handling - preserve as-is
    true.
generate_quote_handling_code(QuoteChar, escape, Code) :-
    format(atom(Code), 'gsub(/~w/, "\\~w", $0)', [QuoteChar, QuoteChar]).

%% escape_delimiter(+Delimiter, -Escaped)
%  Escape delimiter for awk field separator
escape_delimiter('\t', '\\t') :- !.  % Tab
escape_delimiter('|', '\\|') :- !.   % Pipe
escape_delimiter('\\', '\\\\') :- !. % Backslash
escape_delimiter(D, D).              % Others pass through

%% ============================================
%% HARDCODED TEMPLATES (fallback)
%% ============================================

:- multifile template_system:template/2.

% Arity 1 template: pred(X) - return all values from first column
template_system:template(csv_source_unary, '#!/bin/bash
# {{pred}} - CSV source (arity 1)
# Columns: {{columns}}

{{pred}}() {
    awk -F"{{delimiter}}" ''
    NR > {{skip_lines}} {
        {{quote_code}}
        if (NF >= 1) print $1
    }
    '' {{file}}
}

{{pred}}_stream() {
    {{pred}}
}

{{pred}}_check() {
    local value="$1"
    [[ -n $({{pred}} | grep -F "$value") ]] && echo "$value exists"
}
').

% Arity 2+ template: pred(Col1, Col2, ...) - return all columns  
template_system:template(csv_source_binary_plus, '#!/bin/bash
# {{pred}} - CSV source (arity {{arity}})
# Columns: {{columns}}

{{pred}}() {
    local target_key="$1"
    
    if [[ -z "$target_key" ]]; then
        # No key provided, stream all rows
        awk -F"{{delimiter}}" ''
        NR > {{skip_lines}} {
            {{quote_code}}
            if (NF >= {{arity}}) print {{output_format}}
        }
        '' {{file}}
    else
        # Lookup mode: find rows where first column matches key
        awk -F"{{delimiter}}" -v key="$target_key" ''
        NR > {{skip_lines}} {
            {{quote_code}}
            if (NF >= {{arity}} && $1 == key) print {{output_format}}
        }
        '' {{file}}
    fi
}

{{pred}}_stream() {
    {{pred}}
}

{{pred}}_all() {
    {{pred}}
}

{{pred}}_check() {
    local key="$1"
    [[ -n $({{pred}} "$key") ]] && echo "$key exists"
}
').
