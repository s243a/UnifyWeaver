:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% sqlite_source.pl - SQLite source plugin for dynamic sources
% Compiles predicates that read data from SQLite databases using sqlite3 CLI or Python

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
    version('1.1.0'),
    description('Query SQLite databases using sqlite3 or Python (for safe parameter binding)'),
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
    
    atom_string(Pred, PredStr),
    
    % Check for parameters
    (   member(parameters(Params), AllOptions), Params \= []
    ->  % Use Python engine for safe binding
        generate_sqlite_python_code(DbFile, Query, Params, Format, PythonCode),
        render_named_template(sqlite_python_source,
            [pred=PredStr, python_code=PythonCode],
            [source_order([file, generated])],
            BashCode)
    ;   % Use CLI engine (fast, no params)
        % Determine delimiter based on format
        (   Format == tsv -> Separator = '\t'
        ;   Format == csv -> Separator = ','
        ;   Format == list -> Separator = '|'
        ;   Separator = ':' % Default
        ),
        render_named_template(sqlite_cli_source,
            [
                pred=PredStr,
                db_file=DbFile,
                query=Query,
                separator=Separator
            ],
            [source_order([file, generated])],
            BashCode)
    ).

%% generate_sqlite_python_code(+DbFile, +Query, +Params, +Format, -Code)
generate_sqlite_python_code(DbFile, Query, Params, Format, Code) :-
    % Generate params list for Python: [sys.argv[1], "constant", ...]
    maplist(param_to_python, Params, PyParamsList),
    atomic_list_concat(PyParamsList, ', ', PyParams),
    
    % Generate separator char
    (   Format == tsv -> Sep = '\t'
    ;   Format == csv -> Sep = ','
    ;   Format == list -> Sep = '|'
    ;   Sep = ':'
    ),
    
    format(atom(Code), 'import sqlite3
import sys

db_file = "~w"
query = """~w"""
params = [~w]

try:
    conn = sqlite3.connect(db_file)
    cursor = conn.execute(query, params)
    
    for row in cursor:
        # Convert None to empty string and join
        values = [str(x) if x is not None else "" for x in row]
        print("~w".join(values))
        
    conn.close()
except Exception as e:
    print(f"SQLite error: {e}", file=sys.stderr)
    sys.exit(1)
', [DbFile, Query, PyParams, Sep]).

param_to_python(Param, PyParam) :-
    (   atom(Param), sub_atom(Param, 0, 1, _, '$'), sub_atom(Param, 1, _, 0, Sub), atom_number(Sub, N)
    ->  % $1, $2 ... -> sys.argv[N] (shifting index because sys.argv[0] is script?)
        % If we use `python -c script arg1 arg2`, sys.argv[1] is arg1.
        % Our bash wrapper will pass "$@" to python.
        format(atom(PyParam), 'sys.argv[~w]', [N])
    ;   atom(Param), sub_atom(Param, 0, 1, _, '$')
    ->  % $VAR -> treat as string literal if not number? Or assume user mapped it.
        % For now assume $N only.
        format(atom(PyParam), '"~w"', [Param])
    ;   atom(Param)
    ->  format(atom(PyParam), '"~w"', [Param])
    ;   number(Param)
    ->  format(atom(PyParam), '~w', [Param])
    ;   format(atom(PyParam), '"~w"', [Param])
    ).

%% ============================================ 
%% TEMPLATES
%% ============================================ 

:- multifile template_system:template/2.

template_system:template(sqlite_cli_source, '#!/bin/bash
# {{pred}} - SQLite source ({{db_file}})

{{pred}}() {
    sqlite3 -separator ''{{separator}}'' -noheader "{{db_file}}" "{{query}}"
}

{{pred}}_stream() {
    {{pred}}
}


if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    {{pred}} "$@"
fi
').

template_system:template(sqlite_python_source, '#!/bin/bash
# {{pred}} - SQLite source (Python wrapper)

{{pred}}() {
    python3 - "$@" <<''PYTHON''
{{python_code}}
PYTHON
}

{{pred}}_stream() {
    {{pred}}
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    {{pred}} "$@"
fi
').
