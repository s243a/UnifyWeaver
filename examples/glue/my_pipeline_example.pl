% my_pipeline_example.pl
% This file demonstrates how UnifyWeaver unifies a high-level declarative Prolog
% pipeline definition with a concrete shell pipeline generation, by inferring steps.
%
% Updated to use the actual UnifyWeaver modules.

% Load necessary modules (using relative paths from examples/glue/)
:- use_module('../../src/unifyweaver/glue/shell_glue').
:- use_module('../../src/unifyweaver/core/target_mapping').
:- use_module('../../src/unifyweaver/core/compiler_driver').

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

% 3. Predicate to generate the pipeline script using inferred steps.
%    Uses infer_steps_from_goal/2 from shell_glue + generate_pipeline/3.
generate_process_data_pipeline_inferred(Script) :-
    % Infer the steps from the high-level process_data/2 goal
    infer_steps_from_goal(my_pipeline_example:process_data(_, _), Steps),
    Options = [
        input('access.log'),
        output('metrics.txt')
    ],
    generate_pipeline(Steps, Options, Script).

% Alternative: Use compile_goal_to_pipeline/3 from compiler_driver for a one-step approach
generate_pipeline_onestep(Script) :-
    compile_goal_to_pipeline(
        my_pipeline_example:process_data(_, _),
        [input('access.log'), output('metrics.txt')],
        Script
    ).

% To run this:
% 1. From the UnifyWeaver root directory:
%    $ swipl -l examples/glue/my_pipeline_example.pl
%
% 2. In the Prolog interpreter:
%    ?- generate_process_data_pipeline_inferred(Script), writeln(Script).
%    OR
%    ?- generate_pipeline_onestep(Script), writeln(Script).
%
%    Script will be unified with the generated Bash script, with Steps inferred.