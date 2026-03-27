# Input Source Design — Unified I/O for Generated Code

## Philosophy

In Unix, everything is a file. stdin is file descriptor 0 — just the
default input file. Generated code should not be hardcoded to one input
method. Instead, it should accept an abstract **input source** that the
compiler resolves to the appropriate target-language construct.

The generated code's job is: "read structured data from a source and
process it." How that source is connected is a deployment concern, not
a compilation concern.

## Problem Statement

Currently, all UnifyWeaver transitive closure templates hardcode stdin
as the input source. This means:

1. Generated code only works in CLI environments with a terminal
2. Browser/WASM runtimes (SciREPL) can't use the generated code
3. Notebooks can't pipe data between cells
4. Testing requires process I/O rather than in-memory data
5. Serverless/cloud deployments need wrapper code

## Design Principles

1. **Input source is a compilation option, not hardcoded**
2. **The default should match the deployment context**
   - CLI target → stdin (backward compatible)
   - WASM/notebook target → embedded or VFS
3. **All targets support the same set of input modes**
4. **The generated code's logic (BFS, recursion, etc.) is independent
   of the input source**
5. **Stdin IS a file — don't treat them as fundamentally different**

## Specification

### Input Source Types

```prolog
%% Option syntax: input(Mode)
%% Mode is one of:

input(stdin)
%   Read from standard input (fd 0)
%   Default for CLI targets
%   Format: "from:to" lines (current behavior)

input(file(Path))
%   Read from a file at the given path
%   Path is a string literal resolved at runtime
%   Example: input(file("facts.txt"))

input(vfs(CellOrPath))
%   Read from the NotebookVFS or SharedVFS
%   CellOrPath: cell name, In[N] reference, or /nb/ path
%   Example: input(vfs("family_tree"))
%   Example: input(vfs("/shared/data/facts.txt"))

input(embedded)
%   Embed the currently asserted facts directly in the generated code
%   No I/O at runtime — all data is compiled in
%   Example: generates add_fact("alice", "bob") statements

input(function)
%   Generate a function/API that accepts data programmatically
%   No I/O — caller provides data via function arguments
%   Example: ancestor_from_list([("alice","bob"), ("bob","charlie")])
```

### Compilation Interface

```prolog
%% Current (backward compatible):
compile_recursive(ancestor/2, [target(lua)], Code).
%% Equivalent to:
compile_recursive(ancestor/2, [target(lua), input(stdin)], Code).

%% New options:
compile_recursive(ancestor/2, [target(lua), input(embedded)], Code).
compile_recursive(ancestor/2, [target(lua), input(file("data.txt"))], Code).
compile_recursive(ancestor/2, [target(lua), input(vfs("family_tree"))], Code).
compile_recursive(ancestor/2, [target(lua), input(function)], Code).
```

### Context-Aware Defaults

```prolog
%% The compiler can infer the best default based on context:
input_default(wasm, embedded).      % browser/WASM → embed facts
input_default(notebook, vfs).       % notebook → read from VFS
input_default(cli, stdin).          % command line → stdin
input_default(test, embedded).      % testing → embed test data
input_default(_, stdin).            % fallback → stdin
```

### Output Structure

For all input modes, the generated code has three sections:

```
1. DEFINITIONS — functions (add_fact, find_all, check)
2. INPUT       — varies by mode (read stdin, read file, embedded, etc.)
3. INTERFACE   — CLI args, function exports, or queries
```

The DEFINITIONS section is identical across all modes. Only INPUT
and INTERFACE change.

## Template Structure

Each target template should be split into composable parts:

```
templates/targets/<lang>/
  transitive_closure.mustache          # full template (backward compat)
  tc_definitions.mustache              # just the functions
  tc_input_stdin.mustache              # stdin reading
  tc_input_file.mustache               # file reading
  tc_input_embedded.mustache           # seed statements
  tc_input_vfs.mustache                # VFS reading (notebook)
  tc_interface_cli.mustache            # CLI arg parsing
  tc_interface_function.mustache       # function/API export
```

Or, a single template with conditionals:

```mustache
{{#definitions}}
local function add_fact(from, to) ... end
local function find_all(start) ... end
{{/definitions}}

{{#input_stdin}}
for line in io.lines() do ... end
{{/input_stdin}}

{{#input_file}}
for line in io.lines("{{input_path}}") do ... end
{{/input_file}}

{{#input_embedded}}
{{seed_code}}
{{/input_embedded}}

{{#input_vfs}}
local code = nb.read("{{vfs_source}}", ".code")
-- parse Prolog facts from code
{{/input_vfs}}
```

## Implementation Plan

### Phase 1: Embedded mode for all targets
- Generalize TypR's `base_seed_code` to all targets
- Add `input(embedded)` option to `compile_recursive`
- Generate `add_fact()` / `add_parent()` calls from asserted facts
- Make `embedded` the default when `context(notebook)` is set

### Phase 2: File mode for all targets
- Add `input(file(Path))` option
- Each target generates its native file-reading code
- Lua: `io.lines(path)`, Python: `open(path)`, R: `readLines(path)`, etc.

### Phase 3: VFS mode for notebook targets
- Add `input(vfs(Cell))` option
- Generate code that reads from NotebookVFS
- Parse Prolog-style facts from the cell content
- Lua: `nb.read(cell, ".code")`, Python: `nb_read(cell, ".code")`, etc.

### Phase 4: Function mode
- Add `input(function)` option
- Generate functions that accept data as arguments
- No I/O — pure functions for embedding in larger programs
- Useful for library generation and testing

## Relationship to Stdin

Stdin is file descriptor 0. In the template, `input(stdin)` and
`input(file("/dev/stdin"))` should generate identical code on Unix.
The distinction exists for:

1. **Portability** — Windows doesn't have /dev/stdin
2. **Clarity** — `input(stdin)` is self-documenting
3. **Default behavior** — stdin mode includes CLI arg parsing;
   file mode does not

But conceptually, they are the same: "read lines from a file handle."
The template should reflect this by using a shared "read from handle"
pattern internally.
