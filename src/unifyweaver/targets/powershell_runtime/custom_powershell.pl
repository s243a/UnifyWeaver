% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% custom_powershell.pl - Custom PowerShell Component Type
%
% Allows injecting raw PowerShell code as a component.
% The code is wrapped in a function.
%
% Example:
%   declare_component(source, my_transform, custom_powershell, [
%       code("$input.ToUpper()"),
%       usings(["System.Text", "System.IO"])
%   ]).

:- module(custom_powershell, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4
]).

:- use_module('../../core/component_registry').

%% type_info(-Info)
type_info(info(
    name('Custom PowerShell Component'),
    version('1.0.0'),
    description('Injects custom PowerShell code and exposes it as a component')
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
    throw(error(runtime_invocation_not_supported(custom_powershell))).

%% compile_component(+Name, +Config, +Options, -Code)
compile_component(Name, Config, _Options, Code) :-
    member(code(Body), Config),

    (   member(usings(Usings), Config)
    ->  maplist(format_ps_using, Usings, UsingLines),
        atomic_list_concat(UsingLines, '\n', UsingsCode)
    ;   UsingsCode = ''
    ),

    atom_string(Name, NameStr),
    format(string(Code),
"# Custom Component: ~w
~w

<#
.SYNOPSIS
Custom component: ~w

.DESCRIPTION
Process input and return output.
#>
function Invoke-Comp~w {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true, ValueFromPipeline=$true)]
        $Input
    )
    process {
        ~w
    }
}
", [NameStr, UsingsCode, NameStr, NameStr, Body]).

%% format_ps_using(+Namespace, -UsingLine)
format_ps_using(Namespace, UsingLine) :-
    format(string(UsingLine), "using namespace ~w", [Namespace]).

% ============================================================================
% REGISTRATION
% ============================================================================

:- initialization((
    register_component_type(source, custom_powershell, custom_powershell, [
        description("Custom PowerShell Code")
    ])
), now).
