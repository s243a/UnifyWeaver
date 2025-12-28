% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% custom_ruby.pl - Custom Ruby Component Type
%
% Allows injecting raw Ruby code as a component.
% The code is wrapped in a class with an invoke() method.
%
% Example:
%   declare_component(source, my_transform, custom_ruby, [
%       code("input.upcase"),
%       requires(["json", "set"])
%   ]).

:- module(custom_ruby, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4
]).

:- use_module('../../core/component_registry').

%% type_info(-Info)
%
%  Metadata about the custom_ruby component type.
%
type_info(info(
    name('Custom Ruby Component'),
    version('1.0.0'),
    description('Injects custom Ruby code and exposes it as a component')
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
    throw(error(runtime_invocation_not_supported(custom_ruby))).

%% compile_component(+Name, +Config, +Options, -Code)
%
%  Compile the custom Ruby component to Ruby code.
%  Generates a class with an invoke() method.
%
compile_component(Name, Config, _Options, Code) :-
    member(code(Body), Config),

    % Collect requires if specified
    (   member(requires(Requires), Config)
    ->  maplist(format_ruby_require, Requires, RequireLines),
        atomic_list_concat(RequireLines, '\n', RequiresCode)
    ;   RequiresCode = ''
    ),

    atom_string(Name, NameStr),
    capitalize_first(NameStr, ClassName),
    format(string(Code),
"# Custom Component: ~w
~w

class Comp~w
  # Custom component: ~w

  def invoke(input)
    # Process input and return output
~w
  end
end
", [NameStr, RequiresCode, ClassName, NameStr, Body]).

%% format_ruby_require(+Module, -RequireLine)
%
%  Format a Ruby require statement.
%
format_ruby_require(Module, RequireLine) :-
    format(string(RequireLine), "require '~w'", [Module]).

%% capitalize_first(+String, -Capitalized)
%
%  Capitalize the first letter of a string.
%
capitalize_first(String, Capitalized) :-
    string_chars(String, [First|Rest]),
    char_code(First, Code),
    (   Code >= 97, Code =< 122  % a-z
    ->  UpperCode is Code - 32,
        char_code(Upper, UpperCode),
        string_chars(Capitalized, [Upper|Rest])
    ;   Capitalized = String
    ).

% ============================================================================
% REGISTRATION
% ============================================================================

%% Register the custom_ruby component type on module load
:- initialization((
    component_registry:register_component_type(source, custom_ruby, custom_ruby, [
        description("Custom Ruby code injection")
    ])
), now).
