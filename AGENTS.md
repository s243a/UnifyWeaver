# Agent Instructions

Instructions for AI agents working on this project.

## Skill System

This project has a skill routing system for common tasks.

**At startup, read:** `skills/SKILL_ROUTER.md`

The router provides:
- Decision tree for identifying which skill to load (1.0, 2.1, 2.1.1 numbering)
- LOAD/GOTO instructions for conditional navigation
- Footnotes for edge cases

**Skill index:** `skills/SKILL_INDEX.md`

## When to Refresh

If you forget a command, get an error, or feel uncertain → Re-read the skill, or applicable documentation or code, as part of the process to resolve the issue.

## Available Skill Domains

| Domain | Router Section | Primary Skills |
|--------|----------------|----------------|
| Mindmap | 2.0 | linking, MST grouping, cross-links, folder suggestion |
| Bookmark | 3.0 | bookmark filing |
| Compile | 4.0 | playbook compilation, environment |
| Data | 5.0 | JSON sources, record extraction |

## Cursor Cloud specific instructions

UnifyWeaver is a **Prolog transpiler / compile-time CLI tool**, not a long-running
service. "Running the app" means invoking SWI-Prolog to compile Prolog predicates
into a target language (Bash, Go, Rust, C#, Python, SQL, WAM, ...) and then executing
the generated artifact. There is no server, database, or daemon to start for the core
product.

### Toolchain
- **SWI-Prolog** (`swipl`, 9.x) and **Bash 4+** are the only hard dependencies for the
  core transpiler and the bulk of the test suite. The startup/update script installs
  `swipl` if it is missing. Everything else (Go, Rust, .NET, Python libs, Node/`nbb`)
  is only needed to *execute* generated code for a specific target and is optional.
- Always run commands from the repo root so Prolog's `file_search_path` resolves
  `src/unifyweaver/...` module paths.

### Running tests (matches `.github/workflows/test.yml` and `docs/TESTING.md`)
The canonical commands live in `docs/TESTING.md`. Core suite (all pass with plain
`swipl` + `bash`):
```bash
swipl -q -g "use_module('src/unifyweaver/core/constraint_analyzer'), test_constraint_analyzer, halt."
swipl -g "asserta(user:file_search_path(library, 'src/unifyweaver/core')), use_module('src/unifyweaver/core/stream_compiler'), test_stream_compiler, halt."
swipl -g "asserta(user:file_search_path(library, 'src/unifyweaver/core')), use_module('src/unifyweaver/core/recursive_compiler'), test_recursive_compiler, halt."
swipl -g "use_module('src/unifyweaver/core/advanced/test_advanced'), test_all_advanced, halt."
swipl -g "use_module('src/unifyweaver/core/test_constraints'), test_constraints, halt."
```

### Gotchas
- **`output/` and `output/advanced/` must exist** before running tests — the compilers
  write generated scripts there and error with `existence_error(directory,output)` if
  missing. The update script creates them; if you wipe them, recreate with
  `mkdir -p output/advanced`.
- The **inferred test runner** (`output/advanced/inferred_test_runner.sh`) is a
  heuristic generator. Some auto-inferred cases (e.g. `tree_fib`) print `FAIL` because
  the guessed inputs don't match the function's expected argument shape — this is a
  known limitation of the inference tool, not a compiler regression. CI only runs a
  `bash -n` syntax check over the generated scripts, not their inferred assertions.
- Generated bash for recursive predicates depends on the **base relation** being
  sourced too (e.g. `source parent.sh; source ancestor.sh`); compile the base facts
  with `stream_compiler:compile_predicate/3` alongside `recursive_compiler:compile_recursive/3`.
- The `swi-prolog` PPA installer at `scripts/ci/install_swi_prolog.sh` is used by CI,
  but on this Ubuntu (noble) base image `swi-prolog` (9.0.4) is available directly from
  the default `universe` repo, so a plain `apt-get install swi-prolog` is sufficient.
- Optional target toolchains are needed only for their target's tests: `.NET` for C#/F#
  WAM, `node` + `nbb` for the ClojureScript runtime smoke tests, `python3` for the
  C#-query calibration / root-metric wrapper tests, etc. Install these on demand.
