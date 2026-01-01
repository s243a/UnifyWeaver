# Claude Code Project Instructions

## Repository Structure

This project contains **nested independent git repositories**:

| Path | Remote URL | Description |
|------|------------|-------------|
| `/` | `git@github.com:s243a/UnifyWeaver.git` | Main UnifyWeaver codebase |
| `/education/` | `git@github.com:s243a/UnifyWeaver_Education.git` | Educational documentation and books |
| `/education/sandbox/` | `git@github.com:s243a/UnifyWeaver_Education-sandbox.git` | Private version of education repo |
| `/sandbox/` | `git@github.com:s243a/UnifyWeaver-sandbox.git` | Private version of main repo |
| `/training-data/` | `git@github.com:s243a/UnifyWeaver_training-data.git` | Training data files |

**Important:**
- These are NOT submodules. They are independent repos that happen to be nested on disk.
- Each parent repo's `.gitignore` excludes its nested repos.
- These nested repos are **optional** and not guaranteed to be present in every clone. Check if a folder exists before assuming it's available.

To clone all repos:
```bash
git clone git@github.com:s243a/UnifyWeaver.git
cd UnifyWeaver
git clone git@github.com:s243a/UnifyWeaver_Education.git education
git clone git@github.com:s243a/UnifyWeaver-sandbox.git sandbox
git clone git@github.com:s243a/UnifyWeaver_training-data.git training-data
cd education
git clone git@github.com:s243a/UnifyWeaver_Education-sandbox.git sandbox
```

### Working with Nested Repos

When editing files, check which repo you're in:

```bash
# Check current repo
git rev-parse --show-toplevel

# Commit to education repo
cd education && git add . && git commit -m "message"

# Commit to main repo
cd /path/to/UnifyWeaver && git add . && git commit -m "message"
```

If you edit files in `/education/`, you must commit in that repo separately from the main repo.

## Project Overview

UnifyWeaver is a Prolog-based transpiler and glue code generator that compiles declarative specifications to multiple target languages (Bash, Python, Go, Rust, C#, TypeScript, etc.).

### Key Directories

- `src/unifyweaver/` - Core compiler and code generators
- `src/unifyweaver/glue/` - Cross-target glue modules (RPyC, Express, React, visualization)
- `tests/` - Unit and integration tests
- `examples/` - Example projects
- `docs/` - Technical documentation
- `education/` - Educational books and tutorials (separate repo)

### Running Tests

```bash
# Prolog tests
swipl -g "run_tests" -t halt path/to/test_file.pl

# Integration tests for glue modules
swipl -g "run_tests" -t halt tests/integration/glue/test_*.pl
```
