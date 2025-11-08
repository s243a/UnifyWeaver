:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% python_source.pl - Python embedded source plugin for dynamic sources
% Compiles predicates that execute embedded Python code with SQLite support

% Export nothing - all access goes through plugin registry
:- module(python_source, []).

:- use_module(library(lists)).
:- use_module('../core/template_system').
:- use_module('../core/dynamic_source_compiler').

%% Register this plugin on load
:- initialization(
    register_source_type(python, python_source),
    now
).

%% ============================================
%% PLUGIN INTERFACE
%% ============================================

%% source_info(-Info)
%  Provide information about this source plugin
source_info(info(
    name('Python Embedded Source'),
    version('1.0.0'),
    description('Execute embedded Python code with SQLite support using heredoc pattern'),
    supported_arities([1, 2, 3, 4, 5])
)).

%% validate_config(+Config)
%  Validate configuration for Python source
validate_config(Config) :-
    % Must have one of: python_inline, python_file, or sqlite_query
    (   member(python_inline(_), Config)
    ->  true
    ;   member(python_file(File), Config)
    ->  (   exists_file(File)
        ->  true
        ;   format('Error: Python file ~w does not exist~n', [File]),
            fail
        )
    ;   member(sqlite_query(_), Config)
    ->  % SQLite query mode - must also have database
        (   member(database(_), Config)
        ->  true
        ;   format('Error: sqlite_query requires database(File)~n', []),
            fail
        )
    ;   format('Error: Python source requires python_inline, python_file, or sqlite_query~n', []),
        fail
    ),
    
    % Validate timeout if specified
    (   member(timeout(T), Config)
    ->  (   number(T), T > 0
        ->  true
        ;   format('Error: timeout must be positive number, got ~w~n', [T]),
            fail
        )
    ;   true
    ),
    
    % Validate python_interpreter if specified
    (   member(python_interpreter(Interp), Config)
    ->  atom(Interp)
    ;   true
    ).

%% compile_source(+Pred/Arity, +Config, +Options, -BashCode)
%  Compile Python source to bash code
compile_source(Pred/Arity, Config, Options, BashCode) :-
    format('  Compiling Python source: ~w/~w~n', [Pred, Arity]),

    % Validate configuration
    validate_config(Config),

    % Merge config and options
    append(Config, Options, AllOptions),

    % Extract optional parameters with defaults
    (   member(timeout(Timeout), AllOptions)
    ->  true
    ;   Timeout = 30  % Default timeout
    ),
    (   member(python_interpreter(Interpreter), AllOptions)
    ->  true
    ;   Interpreter = python3  % Default interpreter
    ),
    (   member(input_mode(InputMode), AllOptions)
    ->  true
    ;   InputMode = stdin  % Default input mode
    ),
    (   member(output_format(OutputFormat), AllOptions)
    ->  true
    ;   OutputFormat = tsv  % Default output format
    ),
    (   member(error_handling(ErrorHandling), AllOptions)
    ->  true
    ;   ErrorHandling = warn  % Default error handling
    ),

    % Determine Python code source and generate appropriate code
    (   member(sqlite_query(Query), AllOptions),
        member(database(Database), AllOptions)
    ->  generate_sqlite_python_code(Query, Database, PythonCode),
        Mode = sqlite
    ;   member(python_inline(Code), AllOptions)
    ->  PythonCode = Code,
        Mode = inline
    ;   member(python_file(File), AllOptions)
    ->  read_python_file(File, PythonCode),
        Mode = file
    ),

    % Generate bash code using template
    atom_string(Pred, PredStr),
    generate_python_bash(PredStr, Arity, PythonCode, Interpreter, Timeout,
                        InputMode, OutputFormat, ErrorHandling, Mode, BashCode).

%% ============================================
%% PYTHON CODE GENERATION
%% ============================================

%% generate_sqlite_python_code(+Query, +Database, -PythonCode)
%  Generate Python code for SQLite operations
generate_sqlite_python_code(Query, Database, PythonCode) :-
    format(atom(PythonCode), 
'import sqlite3
import sys

try:
    conn = sqlite3.connect(\'~w\')
    cursor = conn.execute(\'\'\'~w\'\'\')
    
    for row in cursor:
        # Convert None to empty string and join with colons
        values = [str(x) if x is not None else \'\' for x in row]
        print(\':\'.join(values))
        
    conn.close()
    
except sqlite3.Error as e:
    print(f"SQLite error: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)', [Database, Query]).

%% read_python_file(+File, -Code)
%  Read Python code from external file
read_python_file(File, Code) :-
    catch(
        (   read_file_to_string(File, Code, [])
        ),
        Error,
        (   format('Error reading Python file ~w: ~w~n', [File, Error]),
            fail
        )
    ).

%% ============================================
%% BASH CODE GENERATION
%% ============================================

%% generate_python_bash(+PredStr, +Arity, +PythonCode, +Interpreter, +Timeout,
%%                      +InputMode, +OutputFormat, +ErrorHandling, +Mode, -BashCode)
%  Generate bash code for Python source using heredoc pattern
generate_python_bash(PredStr, Arity, PythonCode, Interpreter, Timeout,
                     InputMode, OutputFormat, ErrorHandling, Mode, BashCode) :-
    
    % Generate input handling code
    generate_input_handling(InputMode, InputHandling),
    
    % Generate output handling code
    generate_output_handling(OutputFormat, OutputHandling),
    
    % Generate error handling code
    generate_error_handling_code(ErrorHandling, ErrorCode),
    
    % Escape Python code for heredoc (handle single quotes and backslashes)
    escape_python_code(PythonCode, EscapedCode),
    
    % Select template based on mode and complexity
    (   Mode = sqlite ->
        TemplateName = python_sqlite_source
    ;   (   sub_atom(PythonCode, _, _, _, 'sys.stdin')
        ;   InputMode = stdin
        )
    ->  TemplateName = python_stdin_source
    ;   TemplateName = python_basic_source
    ),
    
    % Render template
    render_named_template(TemplateName,
        [pred=PredStr, python_code=EscapedCode, interpreter=Interpreter,
         timeout=Timeout, input_handling=InputHandling, 
         output_handling=OutputHandling, error_code=ErrorCode,
         arity=Arity, mode=Mode],
        [source_order([file, generated])],
        BashCode).

%% generate_input_handling(+Mode, -Code)
%  Generate input handling code for different modes
generate_input_handling(stdin, '# Input from stdin') :- !.
generate_input_handling(args, '# Input from command line arguments') :- !.
generate_input_handling(none, '# No input required') :- !.
generate_input_handling(_, '# Default input handling').

%% generate_output_handling(+Format, -Code)
%  Generate output handling code for different formats
generate_output_handling(tsv, '# Output in TSV format (colon-separated)') :- !.
generate_output_handling(json, '# Output in JSON format') :- !.
generate_output_handling(raw, '# Raw output format') :- !.
generate_output_handling(_, '# Default output format').

%% generate_error_handling_code(+Mode, -Code)
%  Generate error handling code
generate_error_handling_code(fail, 'set -e  # Exit on any error') :- !.
generate_error_handling_code(warn, '# Warnings enabled') :- !.
generate_error_handling_code(silent, '# Silent error handling') :- !.
generate_error_handling_code(_, '# Default error handling').

%% escape_python_code(+Code, -Escaped)
%  Escape Python code for safe heredoc usage
escape_python_code(Code, Escaped) :-
    % For now, simple pass-through - heredoc handles most escaping
    % Could add more sophisticated escaping if needed
    Escaped = Code.

%% ============================================
%% HARDCODED TEMPLATES (fallback)
%% ============================================

:- multifile template_system:template/2.

% Basic Python template - no stdin interaction
template_system:template(python_basic_source, '#!/bin/bash
# {{pred}} - Python source ({{mode}} mode)

{{pred}}() {
    local timeout_val="{{timeout}}"
    {{error_code}}
    {{input_handling}}
    
    timeout "$timeout_val" {{interpreter}} /dev/fd/3 "$@" 3<<''PYTHON''
{{python_code}}
PYTHON
    
    {{output_handling}}
}

{{pred}}_stream() {
    {{pred}}
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    {{pred}} "$@"
fi
').

% Python stdin template - handles stdin input
template_system:template(python_stdin_source, '#!/bin/bash
# {{pred}} - Python source with stdin ({{mode}} mode)

{{pred}}() {
    local timeout_val="{{timeout}}"
    {{error_code}}
    {{input_handling}}
    
    # Use heredoc pattern with stdin passthrough
    timeout "$timeout_val" {{interpreter}} /dev/fd/3 "$@" 3<<''PYTHON''
{{python_code}}
PYTHON
    
    {{output_handling}}
}

{{pred}}_stream() {
    {{pred}}
}

{{pred}}_transform() {
    # Transform stdin through Python code
    {{pred}}
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    {{pred}} "$@"
fi
').

% Specialized SQLite template
template_system:template(python_sqlite_source, '#!/bin/bash
# {{pred}} - SQLite Python source

{{pred}}() {
    local timeout_val="{{timeout}}"
    {{error_code}}
    
    # SQLite query execution with error handling
    timeout "$timeout_val" {{interpreter}} /dev/fd/3 "$@" 3<<''PYTHON''
{{python_code}}
PYTHON
    
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo "SQLite query failed with exit code $exit_code" >&2
        return $exit_code
    fi
    
    {{output_handling}}
}

{{pred}}_stream() {
    {{pred}}
}

{{pred}}_query() {
    # Direct query interface
    {{pred}} "$@"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    {{pred}} "$@"
fi
').
