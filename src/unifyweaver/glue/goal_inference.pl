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
    compile_goal_to_pipeline/3      % compile_goal_to_pipeline(+Goal, +Options, -Script)
]).

:- use_module(library(lists)).
:- use_module('../core/target_mapping').

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
