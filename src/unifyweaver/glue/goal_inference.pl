/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 John William Creighton (s243a)
 *
 * Goal Inference - Meta-interpreter for deriving pipeline steps from goals
 *
 * This module analyzes high-level Prolog goals and automatically constructs
 * pipeline step/4 terms by consulting target_mapping declarations.
 *
 * Key predicates:
 *   - infer_steps_from_goal/2: Infer steps from a goal's body
 *   - compile_goal_to_pipeline/3: High-level API for goal → script
 *
 * See: docs/proposals/meta_interpreter_inference.md
 */

:- module(goal_inference, [
    % Core inference
    infer_steps_from_goal/2,        % infer_steps_from_goal(+Goal, -Steps)
    infer_steps_from_goal/3,        % infer_steps_from_goal(+Goal, +Options, -Steps)
    
    % Helper predicates (exported for testing)
    body_to_list/2,                 % body_to_list(+Body, -Goals)
    subgoal_to_step/3,              % subgoal_to_step(+Module, +SubGoal, -Step)
    
    % High-level API
    compile_goal_to_pipeline/3,     % compile_goal_to_pipeline(+Goal, +Options, -Script)
    
    % Transport-aware grouping
    group_steps_by_transport/2,     % group_steps_by_transport(+Steps, -Groups)
    step_pair_transport/3,          % step_pair_transport(+Step1, +Step2, -Transport)
    infer_transport_from_targets/3, % infer_transport_from_targets(+Target1, +Target2, -Transport)
    generate_pipeline_for_groups/3  % generate_pipeline_for_groups(+Groups, +Options, -Script)
]).

:- use_module(library(lists)).
:- use_module('../core/target_mapping').
:- use_module('../core/target_registry').

%% ============================================
%% Core Inference Predicates
%% ============================================

%% infer_steps_from_goal(+Goal, -Steps)
%  Infer pipeline step/4 terms from a high-level goal's body.
%
%  The goal must be a defined predicate with target declarations for
%  each subgoal in its body.
%
%  Example:
%    % Given:
%    %   process_data(I,O) :- fetch(I,R), transform(R,P), store(P,O).
%    %   :- declare_target(fetch/2, bash, [file('fetch.sh'), name(fetch_stage)]).
%    %   :- declare_target(transform/2, python, [file('t.py'), name(transform_stage)]).
%    %   :- declare_target(store/2, awk, [file('s.awk'), name(store_stage)]).
%    
%    ?- infer_steps_from_goal(process_data(_, _), Steps).
%    Steps = [step(fetch_stage, bash, 'fetch.sh', []),
%             step(transform_stage, python, 't.py', []),
%             step(store_stage, awk, 's.awk', [])].
%
infer_steps_from_goal(Goal, Steps) :-
    infer_steps_from_goal(Goal, [], Steps).

%% infer_steps_from_goal(+Goal, +Options, -Steps)
%  Infer steps with additional options.
%
%  Options:
%    - module(M): Use module M instead of auto-detecting
%    - skip_missing(true): Skip subgoals without target declarations
%
infer_steps_from_goal(Goal, Options, Steps) :-
    % Determine the module
    goal_module(Goal, Options, Module, CleanGoal),
    
    % Get the clause body
    (   clause(Module:CleanGoal, Body)
    ->  true
    ;   throw(error(no_clause_found(Module:CleanGoal), 
                    context(infer_steps_from_goal/3, 'Goal has no clause')))
    ),
    
    % Convert body to list of subgoals
    body_to_list(Body, SubGoals),
    
    % Map each subgoal to a step
    (   member(skip_missing(true), Options)
    ->  findall(Step, 
            (member(SG, SubGoals), subgoal_to_step(Module, SG, Step)),
            Steps)
    ;   maplist(subgoal_to_step(Module), SubGoals, Steps)
    ).

%% goal_module(+Goal, +Options, -Module, -CleanGoal)
%  Determine the module for a goal.
%
goal_module(Goal, Options, Module, CleanGoal) :-
    % Check if module specified in options
    (   member(module(M), Options)
    ->  Module = M,
        (   Goal = _:G -> CleanGoal = G ; CleanGoal = Goal)
    ;   % Try to extract from goal itself
        (   strip_module(Goal, ExtractedModule, CleanGoal)
        ->  Module = ExtractedModule
        ;   Module = user,
            CleanGoal = Goal
        )
    ).

%% ============================================
%% Body Processing
%% ============================================

%% body_to_list(+Body, -Goals)
%  Convert a comma-separated clause body into a list of goals.
%
%  Example:
%    ?- body_to_list((a, b, c), Goals).
%    Goals = [a, b, c].
%
body_to_list((A, B), [A|Rest]) :-
    !,
    body_to_list(B, Rest).
body_to_list(true, []) :- !.
body_to_list(A, [A]).

%% ============================================
%% Subgoal to Step Conversion
%% ============================================

%% subgoal_to_step(+Module, +SubGoal, -Step)
%  Convert a subgoal to a step/4 term using target declarations.
%
%  Throws an error if no target is declared for the predicate.
%
subgoal_to_step(Module, SubGoal, step(StepName, Target, File, StepOptions)) :-
    % Extract predicate name and arity
    functor(SubGoal, PredName, Arity),
    PredIndicator = PredName/Arity,
    
    % Query the target mapping
    (   target_mapping:predicate_target(PredIndicator, Target)
    ->  true
    ;   % Try module-qualified lookup
        Module:predicate_target(PredIndicator, Target)
    ->  true
    ;   throw(error(no_target_declared(PredIndicator),
                    context(subgoal_to_step/3, 'No declare_target for predicate')))
    ),
    
    % Get options from target mapping
    (   target_mapping:predicate_target_options(PredIndicator, Target, Options)
    ->  true
    ;   Module:predicate_target_options(PredIndicator, Target, Options)
    ->  true
    ;   Options = []
    ),
    
    % Extract file from options
    (   member(file(File), Options)
    ->  true
    ;   throw(error(missing_file_option(PredIndicator),
                    context(subgoal_to_step/3, 'No file() in target options')))
    ),
    
    % Extract or generate step name
    (   member(name(StepName), Options)
    ->  true
    ;   atom_concat(PredName, '_step', StepName)
    ),
    
    % Filter out file and name from StepOptions
    subtract(Options, [file(File), name(StepName)], StepOptions).

%% ============================================
%% High-Level API
%% ============================================

%% compile_goal_to_pipeline(+Goal, +Options, -Script)
%  Compile a high-level goal directly to a pipeline script.
%
%  This is the ultimate convenience predicate that:
%  1. Infers steps from the goal
%  2. Generates the pipeline script
%
%  Options can include both inference options and pipeline options.
%
%  Example:
%    ?- compile_goal_to_pipeline(
%           process_data(_, _),
%           [input('data.csv'), output('result.txt')],
%           Script
%       ).
%
compile_goal_to_pipeline(Goal, Options, Script) :-
    % Separate inference options from pipeline options
    partition(is_inference_option, Options, InferenceOpts, PipelineOpts),
    
    % Infer steps
    infer_steps_from_goal(Goal, InferenceOpts, Steps),
    
    % Generate pipeline (requires shell_glue loaded)
    (   current_predicate(shell_glue:generate_pipeline/3)
    ->  shell_glue:generate_pipeline(Steps, PipelineOpts, Script)
    ;   throw(error(shell_glue_not_loaded,
                    context(compile_goal_to_pipeline/3, 
                            'Load shell_glue module first')))
    ).

%% is_inference_option(+Option)
%  True if Option is an inference-specific option.
%
is_inference_option(module(_)).
is_inference_option(skip_missing(_)).

%% ============================================
%% Transport-Aware Step Grouping
%% ============================================
%%
%% These predicates analyze steps and group them by transport type,
%% enabling optimal glue code generation for each transport.

%% group_steps_by_transport(+Steps, -Groups)
%  Group consecutive steps that share the same transport.
%  Returns list of group(Transport, Steps) terms.
%
%  A group contains steps connected by the same transport.
%  The transport refers to the edge BETWEEN steps.
%
%  Example:
%    Steps = [bash, awk, csharp, powershell, python]
%    Transports = [pipe, pipe, direct, pipe]
%    Groups = [group(pipe, [bash, awk, csharp]), group(direct, [csharp, powershell]), group(pipe, [powershell, python])]
%
%    Note: boundary nodes appear in both groups for proper handoff.
%
group_steps_by_transport([], []).
group_steps_by_transport([Step], [group(single, [Step])]).
group_steps_by_transport(Steps, Groups) :-
    Steps = [Step1, Step2|Rest],
    step_pair_transport(Step1, Step2, FirstTransport),
    group_steps_loop([Step1, Step2|Rest], FirstTransport, [Step1], Groups).

%% group_steps_loop(+RemainingSteps, +CurrentTransport, +AccumulatedSteps, -Groups)
%  Recursively group steps by transport.
%
group_steps_loop([_Last], Transport, Acc, [group(Transport, AccReversed)]) :-
    reverse(Acc, AccReversed).

group_steps_loop([_S1, S2], Transport, Acc, [group(Transport, AllSteps)]) :-
    reverse([S2|Acc], AllSteps).

group_steps_loop([_S1, S2, S3|Rest], CurrentTransport, Acc, Groups) :-
    step_pair_transport(S2, S3, NextTransport),
    (   CurrentTransport == NextTransport
    ->  % Same transport, add S2 to current group
        group_steps_loop([S2, S3|Rest], CurrentTransport, [S2|Acc], Groups)
    ;   % Different transport, finalize current group and start new one
        reverse([S2|Acc], CurrentGroup),
        Groups = [group(CurrentTransport, CurrentGroup)|RestGroups],
        % S2 is the boundary - belongs to next group as its first element
        group_steps_loop([S2, S3|Rest], NextTransport, [S2], RestGroups)
    ).

%% step_pair_transport(+Step1, +Step2, -Transport)
%  Determine the transport between two adjacent steps.
%  Uses target families to determine optimal transport.
%
step_pair_transport(step(_, Target1, _, _), step(_, Target2, _, _), Transport) :-
    % Use target-based transport inference
    infer_transport_from_targets(Target1, Target2, Transport).

%% infer_transport_from_targets(+Target1, +Target2, -Transport)
%  Infer transport based on target families.
%
infer_transport_from_targets(Target1, Target2, Transport) :-
    (   catch(target_registry:targets_same_family(Target1, Target2), _, fail)
    ->  % Same family - check if in-process capable
        catch(target_registry:target_family(Target1, Family), _, fail),
        (   in_process_family(Family)
        ->  Transport = direct
        ;   Transport = pipe
        )
    ;   % Different families - use pipes
        Transport = pipe
    ).

%% in_process_family(+Family)
%  Families that support in-process communication.
%
in_process_family(dotnet).
in_process_family(jvm).

%% ============================================
%% Transport-Aware Pipeline Generation
%% ============================================

%% generate_pipeline_for_groups(+Groups, +Options, -Script)
%  Generate combined pipeline script for transport groups.
%
generate_pipeline_for_groups(Groups, Options, Script) :-
    maplist(generate_group_code(Options), Groups, GroupCodes),
    combine_group_scripts(Groups, GroupCodes, Options, Script).

%% generate_group_code(+Options, +Group, -Code)
%  Generate code for a single transport group.
%
generate_group_code(Options, group(pipe, Steps), Code) :-
    % Use shell_glue for pipe-based groups
    (   current_predicate(shell_glue:generate_pipeline/3)
    ->  shell_glue:generate_pipeline(Steps, Options, Code)
    ;   throw(error(shell_glue_not_loaded, context(generate_group_code/3, 'shell_glue required')))
    ).

generate_group_code(Options, group(direct, Steps), Code) :-
    % Use dotnet_glue for in-process groups
    % Note: dotnet_glue:generate_dotnet_pipeline/3 uses same step/4 format
    (   current_predicate(dotnet_glue:generate_dotnet_pipeline/3)
    ->  dotnet_glue:generate_dotnet_pipeline(Steps, Options, Code)
    ;   % Fallback to pipe if dotnet_glue not available
        generate_group_code(Options, group(pipe, Steps), Code)
    ).

generate_group_code(Options, group(http, Steps), Code) :-
    % Use network_glue for HTTP groups
    (   current_predicate(network_glue:generate_http_pipeline/3)
    ->  network_glue:generate_http_pipeline(Steps, Options, Code)
    ;   % Fallback to pipe
        generate_group_code(Options, group(pipe, Steps), Code)
    ).

generate_group_code(Options, group(_, Steps), Code) :-
    % Unknown transport - fallback to pipe
    generate_group_code(Options, group(pipe, Steps), Code).

%% steps_to_dotnet_steps(+Steps, -DotNetSteps)
%  Convert step/4 terms to dotnet_glue format.
%
steps_to_dotnet_steps([], []).
steps_to_dotnet_steps([step(Name, Target, File, _Opts)|Rest], [step(Target, Name, File)|RestDN]) :-
    steps_to_dotnet_steps(Rest, RestDN).

%% combine_group_scripts(+Groups, +Codes, +Options, -Script)
%  Combine multiple group scripts into a single orchestration script.
%
combine_group_scripts([_SingleGroup], [SingleCode], _Options, SingleCode) :- !.
combine_group_scripts(Groups, Codes, Options, Script) :-
    % Multiple groups require a meta-orchestrator
    generate_multi_transport_orchestrator(Groups, Codes, Options, Script).

%% generate_multi_transport_orchestrator(+Groups, +Codes, +Options, -Script)
%  Generate bash script that orchestrates multiple transport groups.
%
generate_multi_transport_orchestrator(Groups, Codes, Options, Script) :-
    % Output redirection
    (   member(output(OutputFile), Options)
    ->  format(atom(OutputRedir), ' > "~w"', [OutputFile])
    ;   OutputRedir = ''
    ),
    
    % Embed group scripts inline for now (simplified approach)
    maplist(format_group_summary, Groups, GroupSummaries),
    atomic_list_concat(GroupSummaries, '\n# ', GroupComments),
    
    % Just concatenate the codes with pipes between them
    % (This is a simplified version - real impl would handle temp files)
    atomic_list_concat(Codes, '\n\n', CombinedCode),
    
    format(atom(Script),
'#!/bin/bash
# Generated UnifyWeaver Multi-Transport Pipeline
# Groups: ~w
set -euo pipefail

~w~w
', [GroupComments, CombinedCode, OutputRedir]).

format_group_summary(group(Transport, Steps), Summary) :-
    length(Steps, N),
    format(atom(Summary), '~w (~w steps)', [Transport, N]).
