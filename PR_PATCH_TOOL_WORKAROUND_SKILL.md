# PR Title

docs(skills): add patch tool fallback workflow

# PR Description

## Summary

- Adds a local `patch-tool-workaround` skill for cases where `apply_patch` is blocked by approval or path-resolution errors despite targeting a trusted workspace.
- Documents the required order: try `apply_patch` first, verify repo/path state, then use a deterministic repo-local edit script only as a fallback.
- Captures guardrails for immediate diff inspection, `git diff --check`, and avoiding unrelated staged files.

## Validation

- `git diff --check -- skills/patch_tool_workaround/SKILL.md`
