# Proposal: Rust-Based AWK with Native Data Structures

## Motivation

The cross-target effective distance benchmark (PR #1054–#1087) revealed
that AWK excels at streaming text processing but is fundamentally slow
for graph traversal workloads:

| Workload | AWK | Python | Go | Why |
|----------|-----|--------|-----|-----|
| Text streaming | Excellent | Good | Good | AWK's tight C inner loop |
| Graph DFS (300 articles) | 2.46s | 0.73s | 0.43s | Interpreter overhead per operation |
| Visited set check | O(1) via `in` | O(1) via `frozenset` | O(1) via `map` | AWK's `in` is O(1) but per-op cost is high |

AWK's associative arrays are hash tables (O(1) lookup), but they're
**string-keyed and string-valued**. Every operation involves string
hashing and comparison, even for numeric or structured data. The
interpreter overhead per array operation limits performance on workloads
with millions of operations.

### Why Not Perl?

Perl extended AWK with richer data structures and regex, but:
- Lost AWK's simplicity (Perl's syntax is complex)
- Didn't gain speed in AWK's core strength (line-by-line streaming)
- Perl's interpreter is not significantly faster than AWK's for
  structured operations

### Why Rust?

A Rust implementation could keep AWK's streaming model while adding:
- **Native hash maps** with typed keys (not string-only)
- **Zero-cost abstractions** for record processing
- **Compiled performance** for graph/set operations
- **AWK-compatible syntax** for existing scripts

## Vision

A Rust-based AWK variant ("rawk"?) that is:

1. **Fully gawk-compatible**: All GNU AWK extensions (gawk) work
   unchanged. Existing AWK scripts run without modification.

2. **Extended with typed data structures**:
   - `set[key]` — hash set with O(1) membership (not string-based)
   - `map[key] = value` — typed hash map
   - `stack` / `queue` — native stack/queue for DFS/BFS
   - `int_array` — dense integer arrays (not string-keyed)

3. **Compiled hot paths**: Frequently executed blocks (loops, DFS)
   are JIT-compiled or AOT-compiled to native code.

4. **AWK's simplicity preserved**: The core `pattern { action }` model
   stays. Extensions are opt-in.

## Potential Extensions Beyond gawk

| Extension | Purpose | Example |
|-----------|---------|---------|
| Typed sets | O(1) membership without string overhead | `if (node in visited_set)` |
| Typed maps | Integer/float keys without string conversion | `dist[node_id] = 3.14` |
| Native DFS/BFS | Built-in graph traversal primitives | `for (node in dfs(adj, start))` |
| Recursion | Native function recursion (gawk has it) | `function dfs(node, visited)` |
| Binary I/O | Read/write structured binary data | `read_struct(fd, format)` |
| Parallel blocks | Process multiple records concurrently | `@parallel { action }` |

## Relationship to UnifyWeaver

UnifyWeaver's AWK target generates AWK code for text processing
workloads. A faster AWK runtime would benefit all UnifyWeaver-generated
AWK programs. The typed data structure extensions would let the AWK
target generate more efficient code for graph workloads like the
effective distance benchmark.

The AWK target compiler could detect when typed extensions are available
and generate optimized code:

```awk
# Standard AWK (current):
if (index(path, "," node ",") > 0) ...  # O(n) string scan

# Extended AWK (proposed):
if (node in visited_set) ...             # O(1) typed set lookup
```

## Scope and Priority

This is a **long-term, back-burner** proposal. It would be a significant
project (building a language runtime from scratch). Potential approaches:

1. **Fork gawk, rewrite hot paths in Rust** — incremental, keeps compatibility
2. **Build from scratch with AWK grammar + JIT** — clean, fast, significant effort
3. **Transpile AWK to Rust** — UnifyWeaver could generate Rust instead
   of AWK for performance-critical workloads (already possible!)

Option 2 with a **JIT compiler** is the most promising for preserving
AWK's feel (no compile step, instant startup) while approaching compiled
speed. Similar to how CPython 3.13+ is adding JIT tiers:

- **Tier 0**: Interpret AWK directly (fast startup, same as current gawk)
- **Tier 1**: Profile hot blocks (BEGIN loops, pattern actions called 1000+ times)
- **Tier 2**: JIT-compile hot blocks to native code via `cranelift` (Rust JIT
  framework) or LLVM

The JIT would be transparent — AWK scripts run unmodified, hot paths get
compiled automatically. Typed data structures (sets, maps) would use native
Rust `HashMap`/`HashSet` at the JIT tier, avoiding string-key overhead.

The Rust ecosystem has mature JIT infrastructure:
- **cranelift** — fast JIT backend (used by Wasmtime)
- **inkwell** — Rust bindings to LLVM (heavier but more optimized output)
- **dynasm-rs** — lightweight x86/ARM assembler for simple JIT

Option 3 is actually what UnifyWeaver already does — the Rust target
generates the same algorithm as the AWK target but compiled. The question
is whether there's value in a standalone AWK-compatible runtime vs. just
using the Rust target directly.

## Existing Projects to Study

- **goawk** — AWK interpreter in Go (faster than gawk for some workloads)
- **mawk** — Fast AWK implementation in C (often faster than gawk)
- **frawk** — AWK-like tool in Rust (exists! focuses on CSV/TSV)

**frawk** is particularly relevant — it's already a Rust-based AWK variant.
Study it before building something new.
