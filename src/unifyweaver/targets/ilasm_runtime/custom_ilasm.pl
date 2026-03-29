:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025-2026 John William Creighton (@s243a)
%
% custom_ilasm.pl — Custom ILAsm Component Type
%
% Allows injecting raw CIL assembly code as a component.
% The code is wrapped in a static method.
%
% Example:
%   declare_component(source, my_il_transform, custom_ilasm, [
%       code("ldarg.0\ncall string [mscorlib]System.String::ToUpper()\nret"),
%       method_name("Transform"),
%       return_type("string"),
%       param_types(["string"])
%   ]).

:- module(custom_ilasm, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4
]).

:- use_module('../../core/component_registry').

%% type_info(-Info)
type_info(info(
    name('Custom ILAsm Component'),
    version('1.0.0'),
    description('Injects custom CIL assembly code as a component')
)).

%% validate_config(+Config)
validate_config(Config) :-
    (   member(code(Code), Config), (string(Code) ; atom(Code))
    ->  true
    ;   throw(error(missing_or_invalid_code_option))
    ).

%% init_component(+Name, +Config)
%  No initialization needed for compile-time component.
init_component(_Name, _Config).

%% invoke_component(+Name, +Config, +Input, -Output)
%  Runtime invocation not supported (compilation only).
invoke_component(_Name, _Config, _Input, _Output) :-
    throw(error(runtime_invocation_not_supported(custom_ilasm))).

%% compile_component(+Name, +Config, +Options, -Code)
%  Compile the custom ILAsm component to a CIL static method.
compile_component(Name, Config, _Options, Code) :-
    member(code(Body), Config),

    %% Method name (default: component name)
    (   member(method_name(MethodName), Config)
    ->  true
    ;   atom_string(Name, MethodName)
    ),

    %% Return type (default: int64)
    (   member(return_type(RetType), Config)
    ->  true
    ;   RetType = "int64"
    ),

    %% Parameter types (default: none)
    (   member(param_types(ParamTypes), Config)
    ->  ilasm_format_params(ParamTypes, 0, ParamDecls),
        atomic_list_concat(ParamDecls, ', ', ParamStr)
    ;   ParamStr = ""
    ),

    %% Max stack (default: 8)
    (   member(max_stack(MaxStack), Config)
    ->  true
    ;   MaxStack = 8
    ),

    format(string(Code),
'.method public static ~w ~w(~w) cil managed {
    .maxstack ~w
    // Component: ~w
~w
    ret
}', [RetType, MethodName, ParamStr, MaxStack, Name, Body]).

%% ilasm_format_params(+Types, +Index, -Decls)
ilasm_format_params([], _, []).
ilasm_format_params([Type|Rest], Index, [Decl|RestDecls]) :-
    format(string(Decl), '~w arg~w', [Type, Index]),
    NextIndex is Index + 1,
    ilasm_format_params(Rest, NextIndex, RestDecls).
