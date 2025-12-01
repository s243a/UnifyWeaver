:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% sqlite_source.pl - SQLite source plugin for dynamic sources
% Compiles predicates that read data from SQLite databases using sqlite3 CLI

:- module(sqlite_source, []).

:- use_module(library(lists)).
:- use_module('../core/template_system').
:- use_module('../core/dynamic_source_compiler').

%% Register this plugin on load
:- initialization(
    register_source_type(sqlite, sqlite_source),
    now
).

%% ============================================ 
%% PLUGIN INTERFACE
%% ============================================ 

source_info(info(
    name('SQLite Source'),
    version('1.0.0'),
    description('Query SQLite databases using sqlite3 command line tool'),
    supported_arities([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
)).

validate_config(Config) :-
    (   member(sqlite_file(File), Config)
    ->  (
            exists_file(File)
        ->  true
        ;   format('Warning: SQLite file ~w does not exist~n', [File])
        )
    ;   format('Error: SQLite source requires sqlite_file(File)~n', []),
        fail
    ),
    
    (   member(query(_), Config)
    ->  true
    ;   format('Error: SQLite source requires query(SQL)~n', []),
        fail
    ).

compile_source(Pred/Arity, Config, Options, BashCode) :-
    format('  Compiling SQLite source: ~w/~w~n', [Pred, Arity]),
    
    validate_config(Config),
    append(Config, Options, AllOptions),
    
    member(sqlite_file(DbFile), AllOptions),
    member(query(Query), AllOptions),
    
    % Output format
    (   member(output_format(Format), AllOptions)
    ->  true
    ;   Format = tsv
    ),
    
    % Determine delimiter based on format
    (   Format == tsv -> Separator = '\t'
    ;   Format == csv -> Separator = ','
    ;   Format == list -> Separator = '|'
    ;   Separator = ':' % Default
    ),
    
    atom_string(Pred, PredStr),
    
    render_named_template(sqlite_cli_source,
        [
            pred=PredStr,
            db_file=DbFile,
            query=Query,
            separator=Separator
        ],
        [source_order([file, generated])],
        BashCode).

%% ============================================ 
%% TEMPLATES
%% ============================================ 

:- multifile template_system:template/2.

template_system:template(sqlite_cli_source, '#!/bin/bash
# {{pred}} - SQLite source ({{db_file}})

{{pred}}() {
    if ! command -v sqlite3 >/dev/null 2>&1; then
        echo "Error: sqlite3 is required for {{pred}}" >&2
        return 1
    fi
    
    sqlite3 -separator ''{{separator}}'' -noheader "{{db_file}}" "{{query}}"
}

{{pred}}_stream() {
    {{pred}}
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    {{pred}} "$@"
fi
').
