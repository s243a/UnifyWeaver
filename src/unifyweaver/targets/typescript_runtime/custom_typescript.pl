% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% custom_typescript.pl - Custom TypeScript Component Type
%
% Allows injecting arbitrary TypeScript code as a component.
% The component generates TypeScript functions from Prolog configuration.
%
% Example:
%   declare_component(source, my_validator, custom_typescript, [
%       code("
%         const data = input as { value: number };
%         if (data.value < 0) throw new Error('Must be positive');
%         return data.value * 2;
%       "),
%       imports(['zod']),
%       input_type('{ value: number }'),
%       output_type('number')
%   ]).
%
%   declare_component(source, rpyc_call, custom_typescript, [
%       code("
%         const { module, func, args } = input as RpycCallInput;
%         return bridge.call(module, func, args);
%       "),
%       imports(['./rpyc_bridge']),
%       async(true)
%   ]).

:- module(custom_typescript, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4
]).

:- use_module('../../core/component_registry').

%% type_info(-Info)
%
%  Returns metadata about this component type.
%
type_info(info(
    name('Custom TypeScript Component'),
    version('1.0.0'),
    description('Injects custom TypeScript code and exposes it as a component')
)).

%% validate_config(+Config)
%
%  Validates the component configuration.
%  Requires a 'code' option with a string value.
%
validate_config(Config) :-
    (   member(code(Code), Config), string(Code)
    ->  true
    ;   throw(error(missing_or_invalid_code_option))
    ).

%% init_component(+Name, +Config)
%
%  Initialize the component (no-op for compile-time components).
%
init_component(_Name, _Config).

%% invoke_component(+Name, +Config, +Input, -Output)
%
%  Runtime invocation not supported (compilation only).
%
invoke_component(_Name, _Config, _Input, _Output) :-
    throw(error(runtime_invocation_not_supported(custom_typescript))).

%% compile_component(+Name, +Config, +Options, -Code)
%
%  Compile the component to TypeScript code.
%
%  Config options:
%    - code(String): The function body code (required)
%    - imports(List): List of imports to add
%    - input_type(String): TypeScript input type (default: 'unknown')
%    - output_type(String): TypeScript output type (default: 'unknown')
%    - async(Bool): Whether the function is async (default: false)
%    - export(Bool): Whether to export the function (default: true)
%    - arrow(Bool): Use arrow function syntax (default: true)
%    - description(String): JSDoc description
%
compile_component(Name, Config, _Options, Code) :-
    member(code(Body), Config),

    % Collect imports if specified
    (   member(imports(Imports), Config)
    ->  forall(member(I, Imports),
            typescript_target:collect_binding_import(I))
    ;   true
    ),

    % Get type annotations
    (member(input_type(InputType), Config) -> true ; InputType = 'unknown'),
    (member(output_type(OutputType), Config) -> true ; OutputType = 'unknown'),

    % Get function modifiers
    (member(async(true), Config) -> Async = 'async ' ; Async = ''),
    (member(export(false), Config) -> Export = '' ; Export = 'export '),

    % Get description for JSDoc
    (member(description(Desc), Config)
    ->  format(string(JSDoc), '/**~n * ~w~n */~n', [Desc])
    ;   JSDoc = ''
    ),

    atom_string(Name, NameStr),

    % Generate function code
    (   member(arrow(false), Config)
    ->  % Traditional function syntax
        format(string(Code),
'~w~w~wfunction ~w(input: ~w): ~w {
~w
}
', [JSDoc, Export, Async, NameStr, InputType, OutputType, Body])
    ;   % Arrow function syntax (default)
        format(string(Code),
'~w~wconst ~w = ~w(input: ~w): ~w => {
~w
};
', [JSDoc, Export, NameStr, Async, InputType, OutputType, Body])
    ).

%% compile_component_class(+Name, +Config, +Options, -Code)
%
%  Alternative: Compile as a class-based component.
%
compile_component_class(Name, Config, _Options, Code) :-
    member(code(Body), Config),

    (member(input_type(InputType), Config) -> true ; InputType = 'unknown'),
    (member(output_type(OutputType), Config) -> true ; OutputType = 'unknown'),
    (member(async(true), Config) -> Async = 'async ' ; Async = ''),

    atom_string(Name, NameStr),

    format(string(Code),
'// Custom Component: ~w
export class ~wComponent {
  ~winvoke(input: ~w): ~w {
~w
  }
}

export const ~w = new ~wComponent();
', [NameStr, NameStr, Async, InputType, OutputType, Body, NameStr, NameStr]).

% ============================================================================
% COMPONENT REGISTRATION
% ============================================================================

% Register this component type when the module is loaded
:- initialization((
    register_component_type(source, custom_typescript, custom_typescript, [
        description("Custom TypeScript Code Injection"),
        target(typescript),
        compile_time(true)
    ])
), now).
