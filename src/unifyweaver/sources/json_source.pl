:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% json_source.pl - JSON source plugin for dynamic sources
% Compiles predicates that process JSON data using jq

:- module(json_source, [
    compile_source/4,          % +Pred/Arity, +Config, +Options, -BashCode
    validate_config/1,         % +Config
    source_info/1              % -Info
]).

:- use_module(library(lists)).
:- use_module('../core/template_system').
:- use_module('../core/dynamic_source_compiler').

%% Register this plugin on load
:- initialization(
    register_source_type(json, json_source),
    now
).

%% ============================================
%% PLUGIN INTERFACE
%% ============================================

%% source_info(-Info)
%  Provide information about this source plugin
source_info(info(
    name('JSON Source'),
    version('1.0.0'),
    description('Process JSON data using jq with flexible filtering and output formats'),
    supported_arities([1, 2, 3, 4, 5])
)).

%% validate_config(+Config)
%  Validate configuration for JSON source
validate_config(Config) :-
    % Must have either json_file or json_stdin, and jq_filter
    (   member(jq_filter(Filter), Config),
        atom(Filter)
    ->  true
    ;   format('Error: JSON source requires jq_filter(Filter)~n', []),
        fail
    ),
    
    % Must have input source
    (   member(json_file(_), Config)
    ->  true
    ;   member(json_stdin(true), Config)
    ->  true
    ;   format('Error: JSON source requires json_file(File) or json_stdin(true)~n', []),
        fail
    ),
    
    % Validate json_file if present
    (   member(json_file(File), Config)
    ->  (   exists_file(File)
        ->  true
        ;   format('Warning: JSON file ~w does not exist~n', [File])
        )
    ;   true
    ),
    
    % Validate output_format if specified
    (   member(output_format(Format), Config)
    ->  (   member(Format, [tsv, json, raw, csv])
        ->  true
        ;   format('Error: output_format must be tsv/json/raw/csv, got ~w~n', [Format]),
            fail
        )
    ;   true
    ).

%% compile_source(+Pred/Arity, +Config, +Options, -BashCode)
%  Compile JSON source to bash code
compile_source(Pred/Arity, Config, Options, BashCode) :-
    format('  Compiling JSON source: ~w/~w~n', [Pred, Arity]),

    % Validate configuration
    validate_config(Config),

    % Merge config and options
    append(Config, Options, AllOptions),

    % Extract required parameters
    member(jq_filter(Filter), AllOptions),

    % Extract optional parameters with defaults
    (   member(json_file(JsonFile), AllOptions)
    ->  InputMode = file
    ;   member(json_stdin(true), AllOptions)
    ->  InputMode = stdin,
        JsonFile = ''
    ),
    (   member(output_format(OutputFormat), AllOptions)
    ->  true
    ;   OutputFormat = tsv  % Default output format
    ),
    (   member(raw_output(RawOutput), AllOptions)
    ->  true
    ;   RawOutput = true  % Default to raw output
    ),
    (   member(compact_output(CompactOutput), AllOptions)
    ->  true
    ;   CompactOutput = false  % Default to pretty output
    ),
    (   member(null_input(NullInput), AllOptions)
    ->  true
    ;   NullInput = false  % Default to normal input
    ),
    (   member(error_handling(ErrorHandling), AllOptions)
    ->  true
    ;   ErrorHandling = fail  % Default error handling
    ),

    % Generate bash code using template
    atom_string(Pred, PredStr),
    generate_json_bash(PredStr, Arity, Filter, JsonFile, InputMode,
                      OutputFormat, RawOutput, CompactOutput, NullInput, 
                      ErrorHandling, BashCode).

%% ============================================
%% BASH CODE GENERATION
%% ============================================

%% generate_json_bash(+PredStr, +Arity, +Filter, +JsonFile, +InputMode,
%%                    +OutputFormat, +RawOutput, +CompactOutput, +NullInput,
%%                    +ErrorHandling, -BashCode)
%  Generate bash code for JSON source
generate_json_bash(PredStr, Arity, Filter, JsonFile, InputMode,
                  OutputFormat, RawOutput, CompactOutput, NullInput, 
                  ErrorHandling, BashCode) :-
    
    % Generate jq flags
    generate_jq_flags(RawOutput, CompactOutput, NullInput, JqFlags),
    
    % Generate error handling code
    generate_json_error_handling(ErrorHandling, ErrorCode),
    
    % Escape filter for bash (handle single quotes)
    escape_jq_filter(Filter, EscapedFilter),
    
    % Generate output processing based on format
    generate_output_processing(OutputFormat, OutputProcessing),
    
    % Select template based on input mode
    (   InputMode = file ->
        TemplateName = json_file_source
    ;   TemplateName = json_stdin_source
    ),
    
    % Render template
    render_named_template(TemplateName,
        [pred=PredStr, filter=EscapedFilter, json_file=JsonFile,
         jq_flags=JqFlags, error_code=ErrorCode, 
         output_processing=OutputProcessing, output_format=OutputFormat,
         arity=Arity, input_mode=InputMode],
        [source_order([file, generated])],
        BashCode).

%% generate_jq_flags(+RawOutput, +CompactOutput, +NullInput, -Flags)
%  Generate jq command line flags
generate_jq_flags(RawOutput, CompactOutput, NullInput, Flags) :-
    FlagsList = [],
    (   RawOutput = true ->
        append(FlagsList, ['-r'], Flags1)
    ;   Flags1 = FlagsList
    ),
    (   CompactOutput = true ->
        append(Flags1, ['-c'], Flags2)
    ;   Flags2 = Flags1
    ),
    (   NullInput = true ->
        append(Flags2, ['-n'], Flags3)
    ;   Flags3 = Flags2
    ),
    atomic_list_concat(Flags3, ' ', Flags).

%% generate_json_error_handling(+Mode, -Code)
%  Generate error handling code
generate_json_error_handling(fail, 'set -e  # Exit on JSON processing errors') :- !.
generate_json_error_handling(warn, '# JSON warnings enabled') :- !.
generate_json_error_handling(continue, '# Continue on JSON errors') :- !.
generate_json_error_handling(_, '# Default JSON error handling').

%% escape_jq_filter(+Filter, -Escaped)
%  Escape jq filter for bash usage
escape_jq_filter(Filter, Escaped) :-
    % For now, simple pass-through - could add more escaping if needed
    % Main concern is single quotes in bash
    Escaped = Filter.

%% generate_output_processing(+Format, -Processing)
%  Generate output post-processing based on format
generate_output_processing(tsv, '# TSV output - use @tsv in jq filter') :- !.
generate_output_processing(csv, '# CSV output - use @csv in jq filter') :- !.
generate_output_processing(json, '# JSON output') :- !.
generate_output_processing(raw, '# Raw output') :- !.
generate_output_processing(_, '# Default output processing').

%% ============================================
%% HARDCODED TEMPLATES (fallback)
%% ============================================

:- multifile template_system:template/2.

% JSON file template - reads from file
template_system:template(json_file_source, '#!/bin/bash
# {{pred}} - JSON source from file ({{json_file}})

{{pred}}() {
    local json_file="{{json_file}}"
    local additional_filter="$1"
    
    {{error_code}}
    {{output_processing}}
    
    # Check if file exists
    if [[ ! -f "$json_file" ]]; then
        echo "JSON file not found: $json_file" >&2
        return 1
    fi
    
    # Apply jq filter, optionally with additional filter
    if [[ -n "$additional_filter" ]]; then
        # Combine filters with pipe
        jq {{jq_flags}} "{{filter}} | $additional_filter" "$json_file"
    else
        jq {{jq_flags}} "{{filter}}" "$json_file"
    fi
    
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo "jq processing failed with exit code $exit_code" >&2
        return $exit_code
    fi
}

{{pred}}_stream() {
    {{pred}}
}

{{pred}}_raw() {
    # Raw jq output without additional processing
    jq {{jq_flags}} "{{filter}}" "{{json_file}}"
}

{{pred}}_filter() {
    # Apply custom filter
    local custom_filter="$1"
    shift
    jq {{jq_flags}} "$custom_filter" "{{json_file}}" "$@"
}
').

% JSON stdin template - reads from stdin
template_system:template(json_stdin_source, '#!/bin/bash
# {{pred}} - JSON source from stdin

{{pred}}() {
    local additional_filter="$1"
    
    {{error_code}}
    {{output_processing}}
    
    # Apply jq filter to stdin, optionally with additional filter
    if [[ -n "$additional_filter" ]]; then
        # Combine filters with pipe
        jq {{jq_flags}} "{{filter}} | $additional_filter"
    else
        jq {{jq_flags}} "{{filter}}"
    fi
    
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo "jq processing failed with exit code $exit_code" >&2
        return $exit_code
    fi
}

{{pred}}_stream() {
    {{pred}}
}

{{pred}}_raw() {
    # Raw jq output without additional processing
    jq {{jq_flags}} "{{filter}}"
}

{{pred}}_filter() {
    # Apply custom filter
    local custom_filter="$1"
    shift
    jq {{jq_flags}} "$custom_filter" "$@"
}
').
