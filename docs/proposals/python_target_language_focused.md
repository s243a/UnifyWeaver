# Python Target: Initial Scope (Generator-Based, StdIO)

**Status:** Draft (focused plan)  
**Audience:** Contributors  
**Goal:** Define the first usable Python target with minimal moving parts.

## What We Will Build First
- **Execution model:** Subprocess-only, streaming via stdin/stdout. No Janus orchestration or in-process switching in this cut.
- **Data format:** JSONL by default (one JSON object per line, UTF-8). Allow NUL-delimited JSON as an option for symmetry with other targets.
- **Codegen style:** Generator-based pipeline (lazy read → transform → write) for memory efficiency and composability.
- **Dependencies:** Standard library only (`json`, `sys`, `typing`). If helpers grow, emit a tiny runtime module alongside the generated script.
- **Outputs:** JSON objects on stdout in the same framing as input (JSONL or NUL-delimited).

## Minimal Contract (v1)
- **Input framing:** `record_format(jsonl)` (default) or `record_format(nul_json)`.
- **Schema/columns:** Accept column selectors to project fields; no type coercion beyond JSON parsing in v1.
- **Recursion:** Support linear and mutual recursion via generated functions/generators; avoid deep stack patterns for now.
- **Errors:** Fail fast on JSON parse errors; non-zero exit code on failure.

## Layout Sketch
- **Prolog compiler module:** `src/unifyweaver/targets/python_target.pl`
  - Translates a plan into a single Python module with:
    - Stream readers/writers (inline helpers)
    - Generated functions for predicates
    - A `main()` that wires read → plan → write
- **Helpers (inline first):**
  - `read_jsonl(stream) -> iterator[dict]`
  - `read_nul_json(stream) -> iterator[dict]`
  - `write_jsonl(records, stream)`
  - `write_nul_json(records, stream)`

## MVP Test Plan
1) **Map/filter over JSONL:** Input a few records, filter on a field, project two fields, assert stdout rows.
2) **Small recursive/join case:** Simple mutual recursion or an aggregate-like fold to ensure generated helpers compose.
- Tests live in `tests/core/test_python_target.pl`; use fixtures in `test_data/`.

## Out of Scope for This Cut
- Janus/in-process execution and orchestration decisions.
- Advanced schema typing or POCO generation.
- External library dependencies (NumPy/Pandas/etc.).
- Network/distributed execution.

## Forward Hooks (kept dormant)
- Placeholder options for `execution_mode(janus|subprocess)` to be wired later.
- Slot for type hints/schemas to grow into typed records once the runtime is stable.
- Possible split of helpers into a reusable runtime module if they expand.
