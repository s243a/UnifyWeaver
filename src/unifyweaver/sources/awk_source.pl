:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% awk_source.pl - AWK source plugin for dynamic sources
% Compiles predicates that read data from AWK commands

:- module(awk_source, [
    compile_source/4,          % +Pred/Arity, +Config, +Options, -BashCode
    validate_config/1,         % +Config
    source_info/1              % -Info
]).

:- use_module(library(lists)).
:- use_module('../core/template_system').
:- use_module('../core/dynamic_source_compiler').

%% Register this plugin on load
:- initialization(
    register_source_type(awk, awk_source),
    now
).

%% ============================================
%% PLUGIN INTERFACE
%% ============================================

%% source_info(-Info)
%  Provide information about this source plugin
source_info(info(
    name('AWK Source'),
    version('1.0.0'),
    description('Execute AWK commands as data sources'),
    supported_arities([1, 2])
)).

%% validate_config(+Config)
%  Validate configuration for AWK source
validate_config(Config) :-
    % Must have either awk_command or awk_file
    (   member(awk_command(_), Config)
    ->  true
    ;   member(awk_file(_), Config)
    ->  true
    ;   format('Error: AWK source requires awk_command or awk_file~n', []),
        fail
    ).

%% compile_source(+Pred/Arity, +Config, +Options, -BashCode)
%  Compile AWK source to bash code
compile_source(Pred/Arity, Config, Options, BashCode) :-
    format('  Compiling AWK source: ~w/~w~n', [Pred, Arity]),

    % Validate configuration
    validate_config(Config),

    % Merge config and options
    append(Config, Options, AllOptions),

    % Extract AWK command/file
    (   member(awk_command(AwkCmd), AllOptions)
    ->  AwkSource = command(AwkCmd)
    ;   member(awk_file(AwkFile), AllOptions),
        AwkSource = file(AwkFile)
    ),

    % Extract optional parameters
    (   member(input_file(InputFile), AllOptions)
    ->  true
    ;   InputFile = none
    ),
    (   member(field_separator(Sep), AllOptions)
    ->  true
    ;   Sep = ':'  % Default separator
    ),

    % Generate bash code using template
    atom_string(Pred, PredStr),
    generate_awk_bash(PredStr, Arity, AwkSource, InputFile, Sep, BashCode).

%% ============================================
%% BASH CODE GENERATION
%% ============================================

%% generate_awk_bash(+PredStr, +Arity, +AwkSource, +InputFile, +Sep, -BashCode)
%  Generate bash code for AWK source
generate_awk_bash(PredStr, Arity, AwkSource, InputFile, Sep, BashCode) :-
    % Prepare AWK command
    (   AwkSource = command(AwkCmd)
    ->  AwkCommandStr = AwkCmd
    ;   AwkSource = file(AwkFile),
        format(string(AwkCommandStr), 'awk -f ~w', [AwkFile])
    ),

    % Prepare input file argument
    (   InputFile = none
    ->  InputFileStr = ''
    ;   format(string(InputFileStr), ' ~w', [InputFile])
    ),

    % Determine function template based on arity
    (   Arity =:= 1 ->
        % Arity 1: pred(X) - return all results
        render_named_template(awk_source_unary,
            [pred=PredStr, awk_cmd=AwkCommandStr, input_file=InputFileStr, sep=Sep],
            [source_order([file, generated])],
            BashCode)
    ;   Arity =:= 2 ->
        % Arity 2: pred(Key, Value) - lookup or stream
        render_named_template(awk_source_binary,
            [pred=PredStr, awk_cmd=AwkCommandStr, input_file=InputFileStr, sep=Sep],
            [source_order([file, generated])],
            BashCode)
    ;   format('Error: AWK source only supports arity 1 or 2, got ~w~n', [Arity]),
        fail
    ).

%% ============================================
%% HARDCODED TEMPLATES (fallback)
%% ============================================

:- multifile template_system:template/2.

% Arity 1 template: pred(X) - return all results
template_system:template(awk_source_unary, '#!/bin/bash
# {{pred}} - AWK source (arity 1)

{{pred}}() {
    if [[ -z "{{input_file}}" ]]; then
        # Read from stdin
        awk -F"{{sep}}" ''{{awk_cmd}}''
    else
        # Read from file
        awk -F"{{sep}}" ''{{awk_cmd}}'' {{input_file}}
    fi
}

{{pred}}_stream() {
    {{pred}}
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    {{pred}} "$@"
fi
').

% Arity 2 template: pred(Key, Value) - lookup or stream
template_system:template(awk_source_binary, '#!/bin/bash
# {{pred}} - AWK source (arity 2)

{{pred}}() {
    local key="$1"
    local value_var="$2"

    if [[ -z "$key" ]]; then
        # No key provided, stream all
        if [[ -z "{{input_file}}" ]]; then
            awk -F"{{sep}}" ''{{awk_cmd}}''
        else
            awk -F"{{sep}}" ''{{awk_cmd}}'' {{input_file}}
        fi
    elif [[ -n "$value_var" ]]; then
        # Lookup mode: find value for key
        if [[ -z "{{input_file}}" ]]; then
            local result=$(awk -F"{{sep}}" -v k="$key" ''{{awk_cmd}} { if ($1 == k) print $2 }'')
        else
            local result=$(awk -F"{{sep}}" -v k="$key" ''{{awk_cmd}} { if ($1 == k) print $2 }'' {{input_file}})
        fi
        if [[ -n "$result" ]]; then
            eval "$value_var=\"$result\""
            echo "$key:$result"
        fi
    else
        # Check mode: does key exist?
        if [[ -z "{{input_file}}" ]]; then
            awk -F"{{sep}}" -v k="$key" ''{{awk_cmd}} { if ($1 == k) { print $0; exit 0 } }''
        else
            awk -F"{{sep}}" -v k="$key" ''{{awk_cmd}} { if ($1 == k) { print $0; exit 0 } }'' {{input_file}}
        fi
    fi
}

{{pred}}_stream() {
    {{pred}}
}

{{pred}}_check() {
    local key="$1"
    [[ -n $({{pred}} "$key") ]] && echo "$key exists"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    {{pred}} "$@"
fi
').
