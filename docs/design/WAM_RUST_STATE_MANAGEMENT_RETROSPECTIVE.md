# WAM Rust State Management: Retrospective and Design Notes

## Context

The `feat/prolog-hybrid-wam-spike` branch implements a WAM (Warren Abstract Machine)
runtime in Rust for benchmarking the effective-distance workload. The WAM compiles
Prolog predicates to instruction arrays and executes them through an interpreter loop.

The central challenge: **efficiently saving and restoring VM state for backtracking.**

SWI-Prolog runs the same workload (300 scale, 386 seed categories, ~60K total paths)
in **338ms**. Our best WAM Rust implementation achieves functional correctness
(270/272 exact line matches) but takes **116s** — a **343x slowdown**.

## What Works: Full State Cloning (`5f4ecce`)

The baseline correct implementation saves the entire VM state in each ChoicePoint:

```rust
pub struct ChoicePoint {
    pub regs: HashMap<String, Value>,     // full clone
    pub stack: Vec<StackEntry>,           // full clone
    pub trail: Vec<TrailEntry>,           // full clone
    pub bindings: HashMap<String, Value>, // full clone
    pub cp: usize,
    pub next_pc: usize,
    pub cut_barrier: usize,
    pub heap_len: usize,
    // ...
}
```

On `backtrack()`:
```rust
self.regs = cp.regs;        // wholesale replace
self.stack = cp.stack;       // wholesale replace
self.trail = cp.trail;       // wholesale replace
self.bindings = cp.bindings; // wholesale replace
```

**Result:** 191s, 270/272 matches. Correct but slow — every `TryMeElse` clones
4 data structures. With thousands of choice points per query, the clone cost dominates.

## Optimization 1: Trail/Heap as Indices (`fcc88aca`)

Instead of cloning `trail` and `heap`, save their lengths and truncate on backtrack:

```rust
pub struct ChoicePoint {
    pub trail_len: usize,  // instead of Vec<TrailEntry>
    pub heap_len: usize,   // instead of Vec<Value>
    // stack and bindings still cloned
}
```

On `backtrack()`:
```rust
self.trail.truncate(cp.trail_len);
self.heap.truncate(cp.heap_len);
```

**Result:** 129s, 270/272 matches. ~32% faster. Trail entries added after the
choice point are simply discarded. Heap terms constructed after the choice point
are truncated. This is safe because:
- Trail entries are append-only (new bindings recorded sequentially)
- Heap is append-only (new terms constructed sequentially)
- Both only need to be "rewound" to the save point

## Optimization 2: Rc<Vec> for Stack (`6815f9a3`)

Wrap the stack in `Rc<Vec<StackEntry>>` for copy-on-write semantics:

```rust
pub stack: Rc<Vec<StackEntry>>,
```

Cloning in TryMeElse is O(1) (Rc reference count increment). Mutations use
`Rc::make_mut(&mut self.stack)` which triggers a deep copy only when the
reference count > 1 (i.e., a choice point holds a reference).

**Result:** 116s, 270/272 matches. ~10% improvement over plain Vec clone.
The benefit is limited because `make_mut` triggers copy-on-write frequently —
every stack mutation (push/pop) after a TryMeElse creates a deep copy if
any choice point references the old stack.

## Failed Approach: Stack Index + Truncation (`006f654`)

Saved `stack_len: usize` instead of cloning the stack. On backtrack,
`stack.truncate(stack_len)`.

**Result:** 65s (fast!) but only 133/272 matches (correctness regression).

**Why it failed:** The stack contains `Env` frames with Y-register HashMaps.
When inner recursive calls modify Y-registers (via `put_reg`), those changes
persist in the Env frame. Truncating the stack removes frames from inner calls
but doesn't restore the Y-register VALUES within surviving frames to their
pre-choice-point state. The full clone approach preserves the exact HashMap
contents; truncation only preserves the frame's existence, not its data.

Example failure:
1. Outer clause `Allocate` → Env at stack[0] with Y1=Cat, Y5=Root
2. `TryMeElse` saves `stack_len=1`
3. Inner recursive call modifies Y1 in stack[0] (same Env frame)
4. Inner call fails → backtrack → `truncate(1)` → stack[0] preserved
5. But Y1 in stack[0] now has the INNER value, not the original Cat

## Failed Approach: Save Only Env Frames (`006f654` variant)

Saved only `Env` frames (position + Y-register HashMap) instead of the full stack:

```rust
pub saved_envs: Vec<(usize, usize, HashMap<String, Value>)>,
```

On backtrack: truncate stack, then patch Env frame contents back.

**Result:** Infinite loop on the first query.

**Why it failed:** The `restore_env_frames` function patches Y-register contents
back into Env frames at their original positions. But `WriteCtx` entries on the
stack between Env frames shift positions. When `PutStructure` or `PutList` pushes
a `WriteCtx` and it's not consumed before backtracking, the position-based patching
writes Y-registers into the wrong stack slot. The interaction between `WriteCtx`
lifecycle and Env frame positioning is complex and fragile.

## Failed Approach: Trail-Only Binding Restoration

Attempted to remove `bindings: HashMap` from ChoicePoint and rely entirely on
trail-based unwinding of the `__binding__` trail entries.

**Result:** Same 133/272 matches as the stack truncation approach.

**Why it failed:** Some code paths modify `self.bindings` without going through
`bind_var` (which records trail entries). Direct `self.bindings.insert()` calls
in builtins or internal helpers bypass the trail, so their changes survive
backtracking when only trail unwinding is used.

## Key Insight: The `unwind_trail` Ordering Bug

A critical bug discovered during the spike: `backtrack()` was doing:

```rust
self.regs = cp.regs;           // 1. Restore registers (correct)
self.unwind_trail(&cp.trail);  // 2. Unwind trail (OVERWRITES registers!)
```

The trail contains entries for register modifications (e.g., `TrailEntry { key: "A1", old_value: ... }`).
After step 1 correctly restored A1 from the choice point snapshot, step 2 iterated
trail entries in reverse and overwrote A1 with a stale value from a nested `\+/1`
sub-invocation.

**Fix:** `unwind_trail_bindings_only()` — only process `__binding__` trail entries,
skip register entries. The choice point's `saved_args`/`regs` is authoritative for
registers; the trail is authoritative only for the bindings table.

## Key Insight: `\+/1` Implementation Cost

The negation-as-failure meta-builtin went through three implementations:

### Version 1: Full State Clone (correct, very slow)
```rust
let saved_regs = self.regs.clone();
let saved_stack = self.stack.clone();
// ... clone everything
// try goal
// restore everything
```
Every `\+(member(X, List))` call cloned the entire VM state. With ~10 calls
per path and ~60K paths, this was millions of deep clones.

### Version 2: Choice Point Based (correct, slow)
Push a NAF choice point. If the goal succeeds, truncate and fail. If the goal
fails, the choice point catches the failure.

Better than full clone but still pushes a full ChoicePoint per `\+` call.

### Version 3: Fast Path for member/2 (correct, fast)
```rust
if functor == "member/2" {
    // Direct list scan — no choice points, no state save
    let found = items.iter().any(|item| deref_var(item) == needle);
    if found { return false; } // \+ fails
    self.pc += 1; return true;  // \+ succeeds
}
```

Zero overhead for the common case. This was the single biggest performance
improvement for the `\+/1` bottleneck.

## Remaining Performance Gap Analysis

| Component | Est. Cost | SWI-Prolog Equivalent |
|-----------|-----------|----------------------|
| `bindings` HashMap clone per CP | ~30% of runtime | Heap-based, no clone needed |
| `regs` save/restore per CP | ~10% | Fixed array, memcpy |
| Instruction dispatch (match loop) | ~20% | Compiled native code |
| HashMap lookups in get_reg/put_reg | ~15% | Array index O(1) |
| Rc make_mut copy-on-write | ~10% | Stack pointer save/restore |
| Binary search in SwitchOnConstant | ~5% | Hash table O(1) |

## Architectural Observations

### Why Rust Fights the WAM Model

The WAM's core requirement — cheap state snapshots for backtracking — conflicts
with Rust's ownership model:

1. **Unique ownership vs shared history:** Rust wants one owner per value.
   WAM needs multiple choice points to reference the same historical state.
   `Rc`/`Arc` works but adds indirection and refcount overhead.

2. **Mutable state vs persistent snapshots:** Rust's `HashMap` is mutable —
   cloning is O(n). Functional languages have persistent maps where "cloning"
   is O(1) by sharing structure.

3. **Stack-based vs heap-based:** Rust's stack frame model doesn't match WAM's
   environment frames that survive across call boundaries and need restoration.

### Why Functional Languages Would Be Better

Languages with immutable/persistent data structures map naturally to WAM backtracking:

- **Haskell:** `Data.Map` (persistent balanced tree) — "clone" is O(log n) by
  sharing subtrees. Trail unwinding = keep the old map reference. Perfect for
  bindings and Y-register storage.

- **Elixir/Erlang:** Immutable maps with structural sharing. Process model maps
  to choice points (each branch is a lightweight process). Pattern matching maps
  to WAM head unification.

- **Clojure:** Persistent vectors and maps with O(1) "clone" via structural sharing.
  `transient` for local mutations, `persistent!` for snapshots.

In any of these, the ChoicePoint would simply hold a reference to the old immutable
state. No cloning needed. Backtracking = swap the reference. O(1).

### Potential Solutions (Not Yet Tried)

#### Rust-Based Approaches

1. **`im` crate (persistent data structures):** `im::HashMap` and `im::Vector`
   provide O(log n) clone with structural sharing. Drop-in replacement for
   `HashMap<String, Value>` and `Vec<StackEntry>`. The ChoicePoint clone
   cost would drop from O(n) to O(log n) with no semantic changes. This is
   probably the **lowest-effort, highest-impact** change to try next.

2. **Arena allocation with generation markers:** Allocate all Values in a
   bump arena (e.g., `bumpalo` crate). ChoicePoints save an arena generation
   marker (just a pointer offset). Backtracking resets the arena pointer. O(1)
   save and restore. The challenge: Values that survive backtracking (results)
   need to be copied out of the arena before reset.

3. **Copy-on-write register array:** Replace `HashMap<String, Value>` for regs
   with a fixed-size `[Option<Value>; 16]` array (A1-A8, X1-X8). Cloning is
   a fixed-cost memcpy (~256 bytes) instead of HashMap's O(n) allocation.
   Y-registers stay in Env frames. This eliminates all HashMap overhead for
   register access and snapshot.

4. **Trail-only state restoration (done correctly):** Our attempt failed because
   some code paths bypass `bind_var`. A more disciplined approach: make ALL
   state mutations go through trail entries — register writes, binding writes,
   stack pushes. Then backtracking is just trail replay. This requires a
   "journaling" discipline across all instructions. The WAM standard already
   specifies this — our implementation just has gaps.

5. **Segmented stack with pointer-based frames:** Instead of a `Vec<StackEntry>`
   that gets cloned, use a linked list of stack segments. Each `Allocate`
   creates a new segment. ChoicePoints save a pointer to the current segment.
   Backtracking follows the pointer — no cloning. Deallocation is O(1)
   (unlink the segment). This is closer to how the real WAM stack works.

6. **Native Rust codegen (no interpreter):** Instead of generating instruction
   arrays fed to a `step()` loop, compile each WAM predicate to a native Rust
   function. `category_ancestor/4` becomes actual Rust code with match
   statements, loops, and direct function calls. Eliminates instruction
   dispatch overhead entirely. This is what the `rust_target` native lowering
   path was designed for — the WAM fallback should be the exception.

7. **Hybrid interpreted + native:** Use WAM instructions for complex predicates
   but generate native Rust for:
   - Fact lookups (HashMap dispatch instead of SwitchOnConstant)
   - Simple predicates (dimension_n/1, max_depth/1 as constants)
   - member/2 and other builtins (already done via fast path)
   This reduces the interpreter's workload to just control flow and unification.

#### Alternative Language Approaches

8. **Haskell WAM with STM:** Implement the WAM in Haskell using Software
   Transactional Memory (STM) for choice points. Each branch is a transaction;
   backtracking aborts and retries. Persistent `Data.Map` for bindings gives
   O(1) snapshots naturally.

9. **Elixir/Erlang WAM with processes:** Each choice point becomes a lightweight
   BEAM process. The parent sends the current state; children explore branches.
   Immutable data sharing between processes is free. This maps the WAM's
   or-parallelism directly to the BEAM VM's process model.

10. **OCaml WAM:** OCaml's algebraic data types and pattern matching map
    naturally to WAM instruction dispatch. Immutable records with functional
    update give cheap snapshots. The GC handles deallocation of abandoned
    choice points. OCaml's native compiler produces fast code.

11. **Rust + WASM persistent map:** Implement the bindings table as a WASM
    module using a persistent trie (like Clojure's HAMT). The WASM module
    provides O(1) snapshot via structural sharing, called from Rust via FFI.
    Unusual architecture but avoids the Rust ownership issues entirely.

#### Algorithmic Approaches (Language-Independent)

12. **Tabling/memoization for category_ancestor:** Instead of exploring all
    paths via backtracking, use tabling (like SWI-Prolog's `:- table` directive)
    to cache `category_ancestor(Cat, Root, Hops, _)` results. The first query
    for each (Cat, Root) pair fills the table; subsequent queries are O(1)
    lookups. This would reduce the 60K-path exploration to ~6K unique (Cat, Root)
    computations. The WAM would need a tabling extension.

13. **Bottom-up evaluation (semi-naive):** Instead of top-down backtracking,
    compute the transitive closure bottom-up. Start with direct
    category_parent facts, then iteratively derive category_ancestor at
    increasing depths. This is what the C# query engine does and is inherently
    more cache-friendly. The WAM is designed for top-down; bottom-up would
    require a different execution model.

14. **Compile \+/1 into WAM instructions at generation time:** Instead of
    implementing `\+/1` as a runtime meta-builtin, have the WAM compiler
    emit the standard NAF instruction sequence:
    ```
    try_me_else L_naf_succeed
    <goal instructions>
    cut
    fail
    L_naf_succeed:
    trust_me
    ```
    This eliminates the meta-builtin entirely and makes NAF a first-class
    WAM construct with zero overhead beyond a single choice point.

## Summary of Spike Results

| Metric | Value |
|--------|-------|
| Functional correctness | 270/272 exact matches with SWI-Prolog |
| tuple_count match | 213/213 (100%) |
| article_count match | 271/272 (99.6%) |
| Best runtime (correct) | 116s (343x slower than SWI-Prolog's 338ms) |
| Best runtime (approx) | 65s with stack truncation (133/272 matches) |
| Key bugs fixed | 12+ WAM runtime issues (backtracking, cut, NAF, indexing, bindings) |
| Lines of code | ~600 Prolog (generator) + ~1300 Rust (template) |

## Recommendations

1. **For the WAM fallback path:** Consider the `im` crate for persistent data
   structures, or investigate arena-based allocation. Either would eliminate
   the clone overhead that dominates runtime.

2. **For the benchmark use case:** The native Rust lowering path (`rust_target`)
   should be the primary target. The WAM fallback is useful for predicates that
   resist native lowering, but performance-critical paths should compile to
   native Rust functions.

3. **For future WAM implementations:** A functional language (Haskell, Elixir,
   or OCaml) would be a better host for the WAM runtime. The persistent data
   structure semantics match WAM backtracking naturally, eliminating the entire
   class of clone/snapshot bugs encountered in this spike.
