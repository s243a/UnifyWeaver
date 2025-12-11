% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% binding_codegen.pl - Code generation utilities using bindings
%
% This module provides utilities to generate code from binding declarations.
% It bridges the binding registry with target code generators.
%
% See: docs/proposals/BINDING_PREDICATE_PROPOSAL.md

:- module(binding_codegen, [
    generate_binding_call/4,        % generate_binding_call(Target, Pred, Args, Code)
    generate_function_from_binding/4, % generate_function_from_binding(Target, Pred, Options, Code)
    generate_cmdlet_from_binding/4, % generate_cmdlet_from_binding(powershell, Pred, Options, Code)
    test_binding_codegen/0
]).

:- use_module('../core/binding_registry').
:- use_module('powershell_bindings').
:- use_module('go_bindings').

% ============================================================================
% CODE GENERATION FROM BINDINGS
% ============================================================================

%% generate_binding_call(+Target, +Pred, +Args, -Code)
%
%  Generate a function call based on the binding for Pred in Target.
%
%  @param Target  atom - target language
%  @param Pred    Name/Arity - predicate indicator
%  @param Args    list - actual arguments for the call
%  @param Code    string - generated code
%
generate_binding_call(Target, Pred, Args, Code) :-
    (   binding(Target, Pred, TargetName, _Inputs, _Outputs, _Options)
    ->  generate_call_code(Target, TargetName, Args, Code)
    ;   format(string(Code), "# No binding found for ~w in ~w", [Pred, Target])
    ).

%% generate_call_code(+Target, +TargetName, +Args, -Code)
%
%  Target-specific call generation
%
generate_call_code(powershell, TargetName, Args, Code) :-
    % Handle different PowerShell call patterns
    (   sub_string(TargetName, 0, _, _, "[")
    ->  % .NET static method call: [Class]::Method
        format_dotnet_call(TargetName, Args, Code)
    ;   sub_string(TargetName, 0, _, _, ".")
    ->  % Instance method call: .Method
        format_instance_call(TargetName, Args, Code)
    ;   % Cmdlet call: Verb-Noun -Param Value
        format_cmdlet_call(TargetName, Args, Code)
    ).

generate_call_code(bash, TargetName, Args, Code) :-
    maplist(format_bash_arg, Args, FormattedArgs),
    atomic_list_concat(FormattedArgs, ' ', ArgsStr),
    format(string(Code), "~w ~w", [TargetName, ArgsStr]).

generate_call_code(go, TargetName, Args, Code) :-
    maplist(format_go_arg, Args, FormattedArgs),
    atomic_list_concat(FormattedArgs, ', ', ArgsStr),
    format(string(Code), "~w(~w)", [TargetName, ArgsStr]).

% Helper: Format .NET static method call
format_dotnet_call(TargetName, Args, Code) :-
    maplist(format_ps_arg, Args, FormattedArgs),
    atomic_list_concat(FormattedArgs, ', ', ArgsStr),
    (   ArgsStr = ""
    ->  format(string(Code), "~w", [TargetName])
    ;   format(string(Code), "~w(~w)", [TargetName, ArgsStr])
    ).

% Helper: Format instance method call
format_instance_call(TargetName, [Object|Args], Code) :-
    format_ps_arg(Object, ObjStr),
    maplist(format_ps_arg, Args, FormattedArgs),
    atomic_list_concat(FormattedArgs, ', ', ArgsStr),
    % Check if method name already ends with () - don't add extra parens
    (   sub_string(TargetName, _, 2, 0, "()")
    ->  % Already has () at end, like .Trim()
        format(string(Code), "~w~w", [ObjStr, TargetName])
    ;   ArgsStr = ""
    ->  format(string(Code), "~w~w()", [ObjStr, TargetName])
    ;   format(string(Code), "~w~w(~w)", [ObjStr, TargetName, ArgsStr])
    ).

% Helper: Format cmdlet call
format_cmdlet_call(TargetName, Args, Code) :-
    maplist(format_ps_arg, Args, FormattedArgs),
    atomic_list_concat(FormattedArgs, ' ', ArgsStr),
    (   ArgsStr = ""
    ->  format(string(Code), "~w", [TargetName])
    ;   format(string(Code), "~w ~w", [TargetName, ArgsStr])
    ).

% Argument formatters
format_ps_arg(Arg, Str) :-
    (   atom(Arg)
    ->  format(string(Str), "'~w'", [Arg])
    ;   number(Arg)
    ->  format(string(Str), "~w", [Arg])
    ;   string(Arg)
    ->  format(string(Str), "'~w'", [Arg])
    ;   is_list(Arg)
    ->  maplist(format_ps_arg, Arg, Items),
        atomic_list_concat(Items, ', ', ItemsStr),
        format(string(Str), "@(~w)", [ItemsStr])
    ;   format(string(Str), "~w", [Arg])
    ).

format_bash_arg(Arg, Str) :-
    (   atom(Arg)
    ->  format(string(Str), "'~w'", [Arg])
    ;   format(string(Str), "~w", [Arg])
    ).

format_go_arg(Arg, Str) :-
    (   atom(Arg)
    ->  format(string(Str), "\"~w\"", [Arg])
    ;   format(string(Str), "~w", [Arg])
    ).

% ============================================================================
% FUNCTION GENERATION FROM BINDINGS
% ============================================================================

%% generate_function_from_binding(+Target, +Pred, +Options, -Code)
%
%  Generate a complete function wrapper based on binding information.
%
generate_function_from_binding(powershell, Pred, Options, Code) :-
    generate_cmdlet_from_binding(powershell, Pred, Options, Code).

generate_function_from_binding(Target, Pred, _Options, Code) :-
    Target \= powershell,
    (   binding(Target, Pred, TargetName, Inputs, Outputs, BindOpts)
    ->  format(string(Code),
"# Generated wrapper for ~w
# Binding: ~w -> ~w
# Inputs: ~w
# Outputs: ~w
# Options: ~w
# (Target-specific generation not implemented yet)",
               [Pred, Pred, TargetName, Inputs, Outputs, BindOpts])
    ;   format(string(Code), "# No binding found for ~w in ~w", [Pred, Target])
    ).

% ============================================================================
% POWERSHELL CMDLET GENERATION FROM BINDINGS
% ============================================================================

%% generate_cmdlet_from_binding(+powershell, +Pred, +Options, -Code)
%
%  Generate a PowerShell cmdlet wrapper using binding information.
%  Incorporates features from Chapter 3 (cmdlet generation):
%  - CmdletBinding attribute
%  - Parameter attributes (Mandatory, Position, ValueFromPipeline)
%  - Verbose/Debug output
%  - Error handling
%
generate_cmdlet_from_binding(powershell, Pred, Options, Code) :-
    Pred = Name/Arity,
    (   binding(powershell, Pred, TargetName, Inputs, Outputs, BindOpts)
    ->  % Extract binding properties
        get_time(Timestamp),
        format_time(string(DateStr), '%Y-%m-%d %H:%M:%S', Timestamp),

        % Determine function name (PowerShell Verb-Noun convention)
        generate_cmdlet_name(Name, CmdletName),

        % Generate CmdletBinding attribute
        generate_cmdlet_binding_attr(Options, BindOpts, CmdletBindingAttr),

        % Generate parameters from inputs
        generate_cmdlet_params(Inputs, Options, ParamsCode),

        % Generate verbose output if requested
        (   member(verbose_output(true), Options)
        ->  format(string(VerboseCode), "    Write-Verbose \"[~w] Calling ~w\"", [CmdletName, TargetName])
        ;   VerboseCode = ""
        ),

        % Generate the actual call
        generate_cmdlet_body(TargetName, Inputs, Outputs, BindOpts, BodyCode),

        % Generate output handling
        generate_output_handling(Outputs, BindOpts, OutputCode),

        format(string(Code),
"# Generated PowerShell Cmdlet from Binding
# Predicate: ~w/~w
# Target: ~w
# Generated: ~w
# Generated by UnifyWeaver Binding Codegen

function ~w {
~w
~w

    begin {
~w
    }

    process {
~w
~w
    }

    end {
        Write-Verbose \"[~w] Complete\"
    }
}
", [Name, Arity, TargetName, DateStr, CmdletName, CmdletBindingAttr, ParamsCode, VerboseCode, BodyCode, OutputCode, CmdletName])
    ;   format(string(Code), "# No binding found for ~w in powershell", [Pred])
    ).

%% generate_cmdlet_name(+PredName, -CmdletName)
%
%  Generate PowerShell Verb-Noun name from predicate name.
%
generate_cmdlet_name(Name, CmdletName) :-
    atom_string(Name, NameStr),
    % Map common prefixes to PowerShell verbs
    (   sub_string(NameStr, 0, 4, _, "get_")
    ->  sub_string(NameStr, 4, _, 0, Rest),
        string_to_title(Rest, TitleRest),
        format(string(CmdletName), "Get-~w", [TitleRest])
    ;   sub_string(NameStr, 0, 4, _, "set_")
    ->  sub_string(NameStr, 4, _, 0, Rest),
        string_to_title(Rest, TitleRest),
        format(string(CmdletName), "Set-~w", [TitleRest])
    ;   sub_string(NameStr, 0, 4, _, "new_")
    ->  sub_string(NameStr, 4, _, 0, Rest),
        string_to_title(Rest, TitleRest),
        format(string(CmdletName), "New-~w", [TitleRest])
    ;   sub_string(NameStr, 0, 7, _, "remove_")
    ->  sub_string(NameStr, 7, _, 0, Rest),
        string_to_title(Rest, TitleRest),
        format(string(CmdletName), "Remove-~w", [TitleRest])
    ;   sub_string(NameStr, 0, 5, _, "test_")
    ->  sub_string(NameStr, 5, _, 0, Rest),
        string_to_title(Rest, TitleRest),
        format(string(CmdletName), "Test-~w", [TitleRest])
    ;   sub_string(NameStr, 0, 6, _, "start_")
    ->  sub_string(NameStr, 6, _, 0, Rest),
        string_to_title(Rest, TitleRest),
        format(string(CmdletName), "Start-~w", [TitleRest])
    ;   sub_string(NameStr, 0, 5, _, "stop_")
    ->  sub_string(NameStr, 5, _, 0, Rest),
        string_to_title(Rest, TitleRest),
        format(string(CmdletName), "Stop-~w", [TitleRest])
    ;   % Default: Invoke-Name
        string_to_title(NameStr, TitleName),
        format(string(CmdletName), "Invoke-~w", [TitleName])
    ).

%% string_to_title(+Str, -TitleStr)
%  Convert string to title case (first letter uppercase)
string_to_title(Str, TitleStr) :-
    string_chars(Str, [First|Rest]),
    upcase_atom(First, Upper),
    atom_chars(Upper, [UpperChar]),
    string_chars(TitleStr, [UpperChar|Rest]).

%% generate_cmdlet_binding_attr(+Options, +BindOpts, -Code)
generate_cmdlet_binding_attr(Options, BindOpts, Code) :-
    findall(Attr, (
        (member(supports_should_process(true), Options) -> Attr = "SupportsShouldProcess=$true" ; fail)
    ;   (member(effect(state), BindOpts) -> Attr = "SupportsShouldProcess=$true" ; fail)
    ), Attrs),
    (   Attrs = []
    ->  Code = "    [CmdletBinding()]"
    ;   atomic_list_concat(Attrs, ', ', AttrsStr),
        format(string(Code), "    [CmdletBinding(~w)]", [AttrsStr])
    ).

%% generate_cmdlet_params(+Inputs, +Options, -Code)
generate_cmdlet_params(Inputs, _Options, Code) :-
    length(Inputs, NumInputs),
    (   NumInputs = 0
    ->  Code = "    param()"
    ;   generate_param_list(Inputs, 0, ParamList),
        atomic_list_concat(ParamList, ',\n', ParamsStr),
        format(string(Code), "    param(\n~w\n    )", [ParamsStr])
    ).

generate_param_list([], _, []).
generate_param_list([Input|Rest], Pos, [ParamCode|RestCodes]) :-
    generate_single_param(Input, Pos, ParamCode),
    Pos1 is Pos + 1,
    generate_param_list(Rest, Pos1, RestCodes).

generate_single_param(type(Type), Pos, Code) :-
    format(string(Code),
"        [Parameter(Position=~w)]
        [~w]$Param~w", [Pos, Type, Pos]).
generate_single_param(Input, Pos, Code) :-
    Input \= type(_),
    atom(Input),
    ps_type_from_semantic(Input, PSType),
    atom_string(Input, InputStr),
    string_to_title(InputStr, ParamName),
    (   Pos = 0
    ->  Mandatory = ", Mandatory=$true"
    ;   Mandatory = ""
    ),
    format(string(Code),
"        [Parameter(Position=~w~w)]
        [~w]$~w", [Pos, Mandatory, PSType, ParamName]).

%% ps_type_from_semantic(+Semantic, -PSType)
ps_type_from_semantic(string, "string").
ps_type_from_semantic(int, "int").
ps_type_from_semantic(path, "string").
ps_type_from_semantic(any, "object").
ps_type_from_semantic(list(_), "array").
ps_type_from_semantic(hashtable, "hashtable").
ps_type_from_semantic(object, "PSObject").
ps_type_from_semantic(_, "object").

%% generate_cmdlet_body(+TargetName, +Inputs, +Outputs, +BindOpts, -Code)
generate_cmdlet_body(TargetName, Inputs, _Outputs, BindOpts, Code) :-
    % Generate parameter passing
    length(Inputs, NumInputs),
    (   NumInputs = 0
    ->  ArgsCode = ""
    ;   generate_args_code(Inputs, 0, ArgsList),
        atomic_list_concat(ArgsList, ' ', ArgsCode)
    ),

    % Check for ShouldProcess
    (   member(effect(state), BindOpts)
    ->  format(string(Code),
"        if ($PSCmdlet.ShouldProcess($Param0, '~w')) {
            $result = ~w ~w
        }", [TargetName, TargetName, ArgsCode])
    ;   format(string(Code), "        $result = ~w ~w", [TargetName, ArgsCode])
    ).

generate_args_code([], _, []).
generate_args_code([Input|Rest], Pos, [ArgCode|RestCodes]) :-
    (   Input = type(_)
    ->  format(string(ArgCode), "$Param~w", [Pos])
    ;   atom(Input),
        atom_string(Input, InputStr),
        string_to_title(InputStr, ParamName),
        format(string(ArgCode), "$~w", [ParamName])
    ),
    Pos1 is Pos + 1,
    generate_args_code(Rest, Pos1, RestCodes).

%% generate_output_handling(+Outputs, +BindOpts, -Code)
generate_output_handling([], _, "        # No output").
generate_output_handling([_|_], BindOpts, Code) :-
    (   member(pattern(object_pipeline), BindOpts)
    ->  Code = "        $result"
    ;   member(pattern(cmdlet_output), BindOpts)
    ->  Code = "        $result"
    ;   Code = "        Write-Output $result"
    ).

% ============================================================================
% TESTING
% ============================================================================

test_binding_codegen :-
    format('~n=== Testing Binding Code Generation ===~n~n'),

    % Initialize bindings
    format('[Setup] Initializing PowerShell bindings...~n'),
    init_powershell_bindings,

    % Test 1: Generate call from binding
    format('[Test 1] Generate binding call~n'),
    generate_binding_call(powershell, sqrt/2, [16], Call1),
    format('  sqrt(16) -> ~w~n', [Call1]),
    (   sub_string(Call1, _, _, _, "[Math]::Sqrt")
    ->  format('  [PASS] Correct .NET call generated~n')
    ;   format('  [FAIL] Expected [Math]::Sqrt~n')
    ),

    % Test 2: Generate cmdlet call
    format('[Test 2] Generate cmdlet call~n'),
    generate_binding_call(powershell, get_child_item/2, ['/tmp'], Call2),
    format('  get_child_item("/tmp") -> ~w~n', [Call2]),
    (   sub_string(Call2, _, _, _, "Get-ChildItem")
    ->  format('  [PASS] Cmdlet call generated~n')
    ;   format('  [FAIL] Expected Get-ChildItem~n')
    ),

    % Test 3: Generate full cmdlet from binding
    format('[Test 3] Generate cmdlet from binding~n'),
    generate_cmdlet_from_binding(powershell, test_path/1, [verbose_output(true)], CmdletCode),
    (   sub_string(CmdletCode, _, _, _, "[CmdletBinding()]"),
        sub_string(CmdletCode, _, _, _, "param("),
        sub_string(CmdletCode, _, _, _, "Test-Path")
    ->  format('  [PASS] Full cmdlet structure generated~n')
    ;   format('  [FAIL] Missing cmdlet components~n')
    ),

    % Test 4: Cmdlet name generation
    format('[Test 4] Cmdlet name generation~n'),
    generate_cmdlet_name(get_service, Name1),
    generate_cmdlet_name(test_path, Name2),
    generate_cmdlet_name(custom_pred, Name3),
    format('  get_service -> ~w~n', [Name1]),
    format('  test_path -> ~w~n', [Name2]),
    format('  custom_pred -> ~w~n', [Name3]),
    (   Name1 = "Get-Service", Name2 = "Test-Path"
    ->  format('  [PASS] Verb-Noun names generated correctly~n')
    ;   format('  [FAIL] Name generation issue~n')
    ),

    format('~n=== Binding Code Generation Tests Complete ===~n').
