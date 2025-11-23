:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% bash_pipeline_source.pl - Generic bash pipeline composer
% Allows composing arbitrary bash pipelines as UnifyWeaver sources

:- module(bash_pipeline_source, []).

:- use_module(library(lists)).
:- use_module('../core/template_system').
:- use_module('../core/dynamic_source_compiler').

%% Register this plugin on load
:- initialization(
    register_source_type(bash_pipeline, bash_pipeline_source),
    now
).

%% ============================================
%% PLUGIN INTERFACE
%% ============================================

source_info(info(
    name('Bash Pipeline Source'),
    version('1.0.0'),
    description('Compose multi-stage bash pipelines as UnifyWeaver sources'),
    supported_arities([1, 2, 3, 4, 5])
)).

validate_config(Config) :-
    % Must have stages
    (   member(stages(Stages), Config)
    ->  is_list(Stages),
        Stages \= []
    ;   format('Error: bash_pipeline requires stages/1 with list of stage/3 terms~n', []),
        fail
    ),
    
    % Validate each stage
    forall(
        member(Stage, Stages),
        validate_stage(Stage)
    ).

validate_stage(stage(Tool, Script, Args)) :-
    atom(Tool),
    atom(Script),
    is_list(Args),
    !.
validate_stage(Stage) :-
    format('Error: Invalid stage format ~w. Expected stage(Tool, Script, Args)~n', [Stage]),
    fail.

compile_source(Pred/Arity, Config, Options, BashCode) :-
    format('  Compiling bash pipeline: ~w/~w~n', [Pred, Arity]),
    
    validate_config(Config),
    append(Config, Options, AllOptions),
    
    member(stages(Stages), AllOptions),
    
    % Get input/output
    (   member(input_file(Input), AllOptions)
    ->  InputSpec = Input
    ;   Input = '/dev/stdin',
        InputSpec = ''
    ),
    
    (   member(output_file(Output), AllOptions)
    ->  true
    ;   Output = '/dev/stdout'
    ),
    
    % Build pipeline
    build_pipeline(Stages, Input, Output, PipelineCode),
    
    % Wrap in bash script
    format(atom(BashCode), '#!/bin/bash~n~w~n', [PipelineCode]).

%% ============================================
%% PIPELINE BUILDER
%% ============================================

build_pipeline([], _, _, '') :- !.

build_pipeline([Stage], Input, Output, Code) :-
    !,
    stage_command(Stage, Input, StageCmd),
    format(atom(Code), '~w > ~w', [StageCmd, Output]).

build_pipeline([Stage|Rest], Input, Output, Code) :-
    stage_command(Stage, Input, StageCmd),
    build_pipeline(Rest, '', Output, RestCode),
    format(atom(Code), '~w | \\~n~w', [StageCmd, RestCode]).

%% Convert stage to bash command
stage_command(stage(awk, Script, Args), Input, Command) :-
    build_args(Args, awk, ArgString),
    (   Input = ''
    ->  format(atom(Command), 'awk -f scripts/utils/~w ~w', [Script, ArgString])
    ;   format(atom(Command), 'awk -f scripts/utils/~w ~w ~w', [Script, ArgString, Input])
    ).

stage_command(stage(python, Script, Args), _Input, Command) :-
    build_args(Args, python, ArgString),
    format(atom(Command), 'python3 scripts/utils/~w ~w', [Script, ArgString]).

stage_command(stage(bash, Script, Args), Input, Command) :-
    build_args(Args, bash, ArgString),
    (   Input = ''
    ->  format(atom(Command), 'bash scripts/~w ~w', [Script, ArgString])
    ;   format(atom(Command), 'bash scripts/~w ~w ~w', [Script, ArgString, Input])
    ).

stage_command(stage(custom, Command, _), _, Command).

%% Build argument string for a tool
build_args([], _, '') :- !.
build_args([Arg|Rest], Tool, ArgString) :-
    arg_to_string(Arg, Tool, ArgStr),
    build_args(Rest, Tool, RestStr),
    (   RestStr = ''
    ->  ArgString = ArgStr
    ;   format(atom(ArgString), '~w ~w', [ArgStr, RestStr])
    ).

arg_to_string(Name=Value, awk, ArgStr) :-
    !,
    format(atom(ArgStr), '-v ~w="~w"', [Name, Value]).

arg_to_string(Flag, python, ArgStr) :-
    atom(Flag),
    !,
    format(atom(ArgStr), '--~w', [Flag]).

arg_to_string(Name=Value, python, ArgStr) :-
    !,
    format(atom(ArgStr), '--~w=~w', [Name, Value]).

arg_to_string(Arg, _, Arg).
