# fix(glue): Remove unnecessary step format conversion for dotnet_glue

## Summary

Fixes the integration between `goal_inference` and `dotnet_glue` for in-process .NET pipeline generation.

## Problem

The `steps_to_dotnet_steps/2` function was incorrectly converting steps:
- **From:** `step(Name, Target, File, Opts)` (4-arity)
- **To:** `step(Target, Name, File)` (3-arity)

But `dotnet_glue:generate_dotnet_pipeline/3` expects the original 4-arity format.

## Fix

Removed the conversion since both modules already use the same `step/4` format.

## Changes

### [goal_inference.pl](file:///home/s243a/Projects/UnifyWeaver/src/unifyweaver/glue/goal_inference.pl)
- Removed `steps_to_dotnet_steps` call in `generate_group_code/3`
- Direct call to `dotnet_glue:generate_dotnet_pipeline/3` with original steps

## Verification

```prolog
generate_pipeline_for_groups([group(direct, Steps)], [], Code)
```

**Result:** Generates C# code with in-process PowerShell via `PowerShellBridge.InvokeStream`.
