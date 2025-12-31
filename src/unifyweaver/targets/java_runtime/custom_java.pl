% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% custom_java.pl - Custom Java Component Type
%
% Allows injecting raw Java code as a component.
% The code is wrapped in a class with an invoke() method.
%
% Example:
%   declare_component(source, my_transform, custom_java, [
%       code("return input.toUpperCase();"),
%       imports(["java.util.List", "java.util.Map"])
%   ]).

:- module(custom_java, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4
]).

:- use_module('../../core/component_registry').

%% type_info(-Info)
%
%  Metadata about the custom_java component type.
%
type_info(info(
    name('Custom Java Component'),
    version('1.0.0'),
    description('Injects custom Java code and exposes it as a component')
)).

%% validate_config(+Config)
%
%  Validate that the configuration contains required options.
%
validate_config(Config) :-
    (   member(code(Code), Config), string(Code)
    ->  true
    ;   throw(error(missing_or_invalid_code_option))
    ).

%% init_component(+Name, +Config)
%
%  No initialization needed for compile-time component.
%
init_component(_Name, _Config).

%% invoke_component(+Name, +Config, +Input, -Output)
%
%  Runtime invocation not supported in Prolog (compilation only).
%
invoke_component(_Name, _Config, _Input, _Output) :-
    throw(error(runtime_invocation_not_supported(custom_java))).

%% compile_component(+Name, +Config, +Options, -Code)
%
%  Compile the custom Java component to Java code.
%  Generates a class with an invoke() method.
%
compile_component(Name, Config, _Options, Code) :-
    member(code(Body), Config),

    % Collect import statements if specified
    (   member(imports(Imports), Config)
    ->  maplist(format_java_import, Imports, ImportLines),
        atomic_list_concat(ImportLines, '\n', ImportsCode)
    ;   ImportsCode = ''
    ),

    atom_string(Name, NameStr),
    format(string(Code),
"// Custom Component: ~w
~w

/**
 * Custom component: ~w
 */
public class Comp~w<T> {
    /**
     * Process input and return output.
     */
    public T invoke(T input) {
        ~w
    }
}
", [NameStr, ImportsCode, NameStr, NameStr, Body]).

%% format_java_import(+ClassName, -ImportLine)
%
%  Format a Java import statement.
%
format_java_import(ClassName, ImportLine) :-
    format(string(ImportLine), "import ~w;", [ClassName]).

% ============================================================================
% REGISTRATION
% ============================================================================

%% Register this component type with the component registry
%
:- initialization((
    register_component_type(source, custom_java, custom_java, [
        description("Custom Java Code")
    ])
), now).
