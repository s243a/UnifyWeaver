% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% custom_scala.pl - Custom Scala Component Type
%
% Allows injecting raw Scala code as a component.
% The code is wrapped in a class with an invoke() method.
%
% Example:
%   declare_component(source, my_transform, custom_scala, [
%       code("input.toUpperCase"),
%       imports(["scala.collection.mutable._"])
%   ]).

:- module(custom_scala, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4
]).

:- use_module('../../core/component_registry').

%% type_info(-Info)
type_info(info(
    name('Custom Scala Component'),
    version('1.0.0'),
    description('Injects custom Scala code and exposes it as a component')
)).

%% validate_config(+Config)
validate_config(Config) :-
    (   member(code(Code), Config), string(Code)
    ->  true
    ;   throw(error(missing_or_invalid_code_option))
    ).

%% init_component(+Name, +Config)
init_component(_Name, _Config).

%% invoke_component(+Name, +Config, +Input, -Output)
invoke_component(_Name, _Config, _Input, _Output) :-
    throw(error(runtime_invocation_not_supported(custom_scala))).

%% compile_component(+Name, +Config, +Options, -Code)
compile_component(Name, Config, _Options, Code) :-
    member(code(Body), Config),

    (   member(imports(Imports), Config)
    ->  maplist(format_scala_import, Imports, ImportLines),
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
class Comp~w[T] {
  /**
   * Process input and return output.
   */
  def invoke(input: T): T = {
    ~w
  }
}
", [NameStr, ImportsCode, NameStr, NameStr, Body]).

%% format_scala_import(+Package, -ImportLine)
format_scala_import(Package, ImportLine) :-
    format(string(ImportLine), "import ~w", [Package]).

% ============================================================================
% REGISTRATION
% ============================================================================

:- initialization((
    register_component_type(source, custom_scala, custom_scala, [
        description("Custom Scala Code")
    ])
), now).
