% my_pipeline_example.pl
% This file demonstrates how UnifyWeaver unifies a high-level declarative Prolog
% pipeline definition with a concrete shell pipeline generation, by inferring steps.

% Load necessary modules
:- use_module(library(shell_glue)).
:- use_module(library(target_mapping)).

% 1. High-level declarative Prolog for the data processing logic
%    This defines the *what* and *flow*, without specifying *how* or *where*.
process_data(Input, Output) :-
    fetch(Input, Raw),
    transform(Raw, Processed),
    store(Processed, Output).

% 2. UnifyWeaver Declarations: Map high-level predicates to concrete targets.
%    These tell UnifyWeaver which language/tool implements each logical step.
:- declare_target(fetch/2, bash, [file('fetch.sh'), name('fetch_stage')]).
:- declare_target(transform/2, python, [file('transform.py'), name('transform_stage')]).
:- declare_target(store/2, awk, [file('store.awk'), name('store_stage')]).

% Helper predicate to infer steps from a high-level goal's body
% This simulates UnifyWeaver's internal compiler logic
infer_steps_from_goal(Goal, Steps) :-
    % Get the module of the goal
    (   strip_module(Goal, Module, _CleanGoal)
    ->  true
    ;   Module = user % Default to user if no explicit module
    ),
    % Get the body of the goal (e.g., fetch(I,R),transform(R,P),store(P,O))
    clause(Module:Goal, Body),
    % Convert the comma-separated body into a list of subgoals
    body_to_list(Body, SubGoals),
    % Map each subgoal to a step/4 term using declared targets
    maplist(subgoal_to_step(Module), SubGoals, Steps).

% Convert comma-separated body to list of goals
body_to_list((A,B), [A|Rest]) :-
    !,
    body_to_list(B, Rest).
body_to_list(A, [A]).

% Resolve a subgoal to a step/4 term
subgoal_to_step(Module, SubGoal, step(StepName, Target, File, StepOptions)) :-
    % Extract predicate name and arity
    functor(SubGoal, PredName, Arity),
    % Query the target mapping for this predicate
    (   Module:predicate_target(PredName/Arity, Target)
    ->  true
    ;   throw(error(no_target_declared(PredName/Arity), _))
    ),
    % Get options (including file and step name)
    Module:target_options(PredName/Arity, Options),
    (   member(file(File), Options)
    ->  true
    ;   throw(error(missing_file_option(PredName/Arity), _))
    ),
    (   member(name(StepName), Options)
    ->  true
    ;   % Default step name if not provided
        atom_concat(PredName, '_step', StepName)
    ),
    % Filter out file and name from StepOptions to avoid duplication
    subtract(Options, [file(File), name(StepName)], StepOptions).


% 3. Predicate to generate the pipeline script using inferred steps.
generate_process_data_pipeline_inferred(Script) :-
    % Infer the steps from the high-level process_data/2 goal
    infer_steps_from_goal(my_pipeline_example:process_data(_, _), Steps),
    Options = [
        input('access.log'),
        output('metrics.txt')
    ],
    generate_pipeline(Steps, Options, Script).


% To run this:
% 1. Save the above code as 'my_pipeline_example.pl'.
% 2. In a Prolog interpreter (like SWI-Prolog):
%    ?- consult('my_pipeline_example.pl').
%    ?- generate_process_data_pipeline_inferred(Script), writeln(Script).
%    Script will be unified with the generated Bash script, with Steps inferred.