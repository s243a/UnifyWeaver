:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_yaml_source.pl - Tests for YAML source plugin

:- module(test_yaml_source, [
    test_yaml_source/0
]).

:- use_module('../../src/unifyweaver/sources/yaml_source').
:- use_module(library(lists)).

%% Main test predicate
test_yaml_source :-
    format('Testing YAML source plugin...~n', []),
    test_yaml_file_compilation,
    test_yaml_stdin_compilation,
    format('✅ All YAML source tests passed~n', []).

test_yaml_file_compilation :-
    format('  Testing YAML file compilation...~n', []),
    
    catch(
        (   yaml_source:compile_source(my_yaml_data/2, [
                yaml_file('config.yaml'),
                yaml_filter('data["users"]')
            ], [], Code),
            
            % Check for python imports
            sub_atom(Code, _, _, _, 'import yaml'),
            sub_atom(Code, _, _, _, 'import json'),
            
            % Check input handling
            sub_atom(Code, _, _, _, 'with open("config.yaml", "r")'),
            
            % Check filter
            sub_atom(Code, _, _, _, 'result = data["users"]'),
            
            format('    ✅ File compilation works~n', [])
        ),
        Error,
        (   format('    ❌ Error: ~w~n', [Error]),
            fail
        )
    ).

test_yaml_stdin_compilation :-
    format('  Testing YAML stdin compilation...~n', []),
    
    catch(
        (   yaml_source:compile_source(stream_data/1, [
                yaml_stdin(true)
            ], [], Code),
            
            % Check input handling
            sub_atom(Code, _, _, _, 'yaml.safe_load(sys.stdin)'),
            
            % Check default filter
            sub_atom(Code, _, _, _, 'result = data'),
            
            format('    ✅ Stdin compilation works~n', [])
        ),
        Error,
        (   format('    ❌ Error: ~w~n', [Error]),
            fail
        )
    ).
