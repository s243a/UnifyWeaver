# GitHub Actions Workflows

## Test Workflow

**File:** `test.yml`

**Purpose:** Runs the complete UnifyWeaver test suite on every push and pull request.

### What it tests:
- ✅ Constraint analyzer (6 tests)
- ✅ Stream compiler
- ✅ Recursive compiler
- ✅ Advanced recursion (24 tests)
- ✅ Constraint integration
- ✅ C# query calibration wrapper contract (unit test + dry-run command shape)
- ✅ Inference test runner generation
- ✅ Generated bash script syntax validation

### Behavior:
- **Non-blocking:** Pushes always succeed, tests run in background
- **Runs on:** Every push to any branch
- **Duration:** ~3-5 minutes
- **Notifications:** Email on failure (can be disabled in GitHub settings)

### How to disable temporarily:
1. Rename `test.yml` to `test.yml.disabled`
2. Commit and push

### How to modify:
Just edit `test.yml` and push. Changes take effect immediately.

### How to add a badge to README:

```markdown
![Tests](https://github.com/s243a/UnifyWeaver/actions/workflows/test.yml/badge.svg)
```

This will show:
- ✅ Green badge when tests pass
- ❌ Red badge when tests fail
- 🟡 Yellow badge when tests are running
