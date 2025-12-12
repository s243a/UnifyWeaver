# feat(glue): Add meta-interpreter for automatic pipeline step inference

## Summary

Adds a meta-interpreter that automatically derives pipeline `step/4` terms from high-level Prolog goals with `declare_target/3` declarations. This enables users to write declarative logic and have the compiler infer the concrete pipeline orchestration.

## Changes

### New Module
- **`src/unifyweaver/glue/goal_inference.pl`**: Core meta-interpreter
  - `infer_steps_from_goal/2,3` - Analyze goal body to derive steps
  - `body_to_list/2` - Convert comma-body to list
  - `subgoal_to_step/3` - Map subgoal to step/4 via target declarations
  - `compile_goal_to_pipeline/3` - High-level goal → script API

### Modified Modules
- **`src/unifyweaver/core/compiler_driver.pl`**
  - Re-exports `compile_goal_to_pipeline/3`
  - Adds `compile_goal_to_pipeline/4` (also returns inferred steps)

- **`src/unifyweaver/core/target_mapping.pl`**
  - Added `target_options/2` convenience wrapper

- **`src/unifyweaver/glue/shell_glue.pl`**
  - Re-exports `infer_steps_from_goal/2,3`

### Example
- **`examples/glue/my_pipeline_example.pl`** - Updated to use actual modules

## Usage

```prolog
% Define high-level logic
process_data(Input, Output) :-
    fetch(Input, Raw),
    transform(Raw, Processed),
    store(Processed, Output).

% Declare targets for each step
:- declare_target(fetch/2, bash, [file('fetch.sh'), name(fetch_stage)]).
:- declare_target(transform/2, python, [file('transform.py'), name(transform_stage)]).
:- declare_target(store/2, awk, [file('store.awk'), name(store_stage)]).

% Compile to pipeline script
?- compile_goal_to_pipeline(
       process_data(_, _),
       [input('data.csv'), output('result.txt')],
       Script
   ).
```

## Future Work

The compiler currently uses shell pipes for all orchestration. A TODO is documented for future enhancement to use `resolve_transport/3` to choose optimal transport based on target families (in-process for .NET, pipes for shell, sockets for distributed).

## Related Documentation
- `docs/proposals/meta_interpreter_inference.md`
- `education/book-07-cross-target-glue/01_introduction.md` (updated with three-tier approach)
