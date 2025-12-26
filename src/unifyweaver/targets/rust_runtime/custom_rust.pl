% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% custom_rust.pl - Custom Rust Component Type
%
% Allows injecting raw Rust code as a component.
% The code is wrapped in a struct with an invoke() method.
%
% Example:
%   declare_component(source, my_transform, custom_rust, [
%       code("input.to_uppercase()"),
%       uses(["std::collections::HashMap", "serde_json"])
%   ]).

:- module(custom_rust, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4
]).

:- use_module('../../core/component_registry').

%% type_info(-Info)
%
%  Metadata about the custom_rust component type.
%
type_info(info(
    name('Custom Rust Component'),
    version('1.0.0'),
    description('Injects custom Rust code and exposes it as a component')
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
    throw(error(runtime_invocation_not_supported(custom_rust))).

%% compile_component(+Name, +Config, +Options, -Code)
%
%  Compile the custom Rust component to Rust code.
%  Generates a struct with an invoke() method.
%
compile_component(Name, Config, _Options, Code) :-
    member(code(Body), Config),

    % Collect use statements if specified
    (   member(uses(Uses), Config)
    ->  maplist(format_rust_use, Uses, UseLines),
        atomic_list_concat(UseLines, '\n', UsesCode)
    ;   UsesCode = ''
    ),

    atom_string(Name, NameStr),
    format(string(Code),
"// Custom Component: ~w
~w

/// Custom component: ~w
pub struct Comp~w;

impl Comp~w {
    /// Process input and return output.
    pub fn invoke<T>(&self, input: T) -> Result<T, Box<dyn std::error::Error>>
    where
        T: Clone,
    {
        Ok(~w)
    }
}
", [NameStr, UsesCode, NameStr, NameStr, NameStr, Body]).

%% format_rust_use(+Module, -UseLine)
%
%  Format a Rust use statement.
%
format_rust_use(Module, UseLine) :-
    format(string(UseLine), "use ~w;", [Module]).

% ============================================================================
% REGISTRATION
% ============================================================================

%% Register this component type with the component registry
%
:- initialization((
    register_component_type(source, custom_rust, custom_rust, [
        description("Custom Rust Code")
    ])
), now).
