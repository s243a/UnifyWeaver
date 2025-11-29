:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% yaml_source.pl - YAML source plugin for dynamic sources
% Compiles predicates that process YAML data using Python (PyYAML)

% Export nothing - all access goes through plugin registry
:- module(yaml_source, []).

:- use_module(library(lists)).
:- use_module('../core/template_system').
:- use_module('../core/dynamic_source_compiler').

%% Register this plugin on load
:- initialization(
    register_source_type(yaml, yaml_source),
    now
).

%% ============================================
%% PLUGIN INTERFACE
%% ============================================

%% source_info(-Info)
%  Provide information about this source plugin
source_info(info(
    name('YAML Source'),
    version('1.0.0'),
    description('Process YAML data using Python (PyYAML) with filtering'),
    supported_arities([1, 2, 3, 4, 5])
)).

%% validate_config(+Config)
%  Validate configuration for YAML source
validate_config(Config) :-
    % Must have either yaml_file or yaml_stdin
    (   member(yaml_file(File), Config)
    ->  (   exists_file(File)
        ->  true
        ;   format('Warning: YAML file ~w does not exist~n', [File])
        )
    ;   member(yaml_stdin(true), Config)
    ->  true
    ;   format('Error: YAML source requires yaml_file(File) or yaml_stdin(true)~n', []),
        fail
    ),
    
    % Optional yaml_filter (Python expression)
    (   member(yaml_filter(Filter), Config)
    ->  atom(Filter)
    ;   true
    ).

%% compile_source(+Pred/Arity, +Config, +Options, -BashCode)
%  Compile YAML source to bash code
compile_source(Pred/Arity, Config, Options, BashCode) :-
    format('  Compiling YAML source: ~w/~w~n', [Pred, Arity]),

    % Validate configuration
    validate_config(Config),

    % Merge config and options
    append(Config, Options, AllOptions),

    % Extract parameters
    (   member(yaml_file(YamlFile), AllOptions)
    ->  InputMode = file
    ;   member(yaml_stdin(true), AllOptions)
    ->  InputMode = stdin,
        YamlFile = ''
    ),
    
    (   member(yaml_filter(Filter), AllOptions)
    ->  true
    ;   Filter = "data"  % Default: return whole data (or root list)
    ),

    % Determine Python interpreter
    (   member(python_interpreter(Interpreter), AllOptions)
    ->  true
    ;   Interpreter = python3
    ),

    % Generate the embedded Python script
    generate_yaml_python_driver(Filter, InputMode, YamlFile, Arity, PythonCode),

    % Generate bash code
    atom_string(Pred, PredStr),
    generate_yaml_bash(PredStr, Arity, PythonCode, Interpreter, YamlFile, InputMode, BashCode).


%% ============================================
%% PYTHON DRIVER GENERATION
%% ============================================

generate_yaml_python_driver(Filter, InputMode, YamlFile, Arity, PythonCode) :-
    % We construct a Python script that imports yaml, reads input, applies filter, and prints TSV/JSON
    
    (   InputMode = file
    ->  format(atom(InputSetup), '    with open("~w", "r") as f:
        data = yaml.safe_load(f)', [YamlFile])
    ;   InputSetup = '    data = yaml.safe_load(sys.stdin)'
    ),

    format(atom(PythonCode), 
'import sys
import json

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed (pip install PyYAML)", file=sys.stderr)
    sys.exit(1)

def process():
~w
    
    # Apply user filter (default is "data")
    # We assume "data" is the variable holding the YAML content
    # The filter is a python expression that evaluates to an iterable or dict
    result = ~w
    
    if result is None:
        return

    # Helper to format output based on arity
    def emit(row):
        if isinstance(row, (list, tuple)):
             # Take first N elements for arity, join with colon
             print(":".join(str(x) for x in row[:~w]))
        elif isinstance(row, dict):
             # For dict, we might need specific logic, but here we just dump JSON
             print(json.dumps(row))
        else:
             print(str(row))

    if isinstance(result, list):
        for item in result:
            emit(item)
    elif isinstance(result, dict):
        # If it is a dict, do we emit keys, values, or the dict itself?
        # Default behavior: if arity 1, emit the dict (as json line)
        # If user wants keys, they should use filter "data.keys()"
        emit(result)
    else:
        emit(result)

if __name__ == "__main__":
    try:
        process()
    except Exception as e:
        print(f"YAML processing error: {e}", file=sys.stderr)
        sys.exit(1)
', [InputSetup, Filter, Arity]).


%% ============================================
%% BASH CODE GENERATION
%% ============================================

generate_yaml_bash(PredStr, Arity, PythonCode, Interpreter, YamlFile, InputMode, BashCode) :-
    
    (   InputMode = file
    ->  TemplateName = yaml_file_source
    ;   TemplateName = yaml_stdin_source
    ),
    
    render_named_template(TemplateName,
        [pred=PredStr, python_code=PythonCode, interpreter=Interpreter,
         yaml_file=YamlFile, arity=Arity],
        [source_order([file, generated])],
        BashCode).


%% ============================================
%% BASH TEMPLATES
%% ============================================

:- multifile template_system:template/2.

template_system:template(yaml_file_source, '#!/bin/bash
# {{pred}} - YAML file source ({{yaml_file}})

{{pred}}() {
    # Check dependencies
    if ! {{interpreter}} -c "import yaml" 2>/dev/null; then
        echo "Error: PyYAML is required for {{pred}} (pip install PyYAML)" >&2
        return 1
    fi

    {{interpreter}} /dev/fd/3 3<<''PYTHON''
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

template_system:template(yaml_stdin_source, '#!/bin/bash
# {{pred}} - YAML stdin source

{{pred}}() {
    # Check dependencies
    if ! {{interpreter}} -c "import yaml" 2>/dev/null; then
        echo "Error: PyYAML is required for {{pred}} (pip install PyYAML)" >&2
        return 1
    fi

    # Pass stdin to python
    {{interpreter}} /dev/fd/3 3<<''PYTHON''
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
