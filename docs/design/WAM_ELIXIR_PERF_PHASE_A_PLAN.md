# WAM-Elixir Performance — Phase A Plan

Brings the Elixir WAM target in line with the Phase A–B data-structure
work already landed for Haskell and Rust. See
`docs/design/WAM_PERF_OPTIMIZATION_LOG.md` for the broader context.

This doc covers the mechanical, language-agnostic wins — `heap`, `trail`,
`code` container choices and length caching. FFI-path work, parallel
seeds, and label pre-resolution are explicitly out of scope; see the
"Out of scope" section.

## Problem statement

Audit of `src/unifyweaver/targets/wam_elixir_target.pl` and
`src/unifyweaver/targets/wam_elixir_lowered_emitter.pl` shows four
recurring O(n)-on-hot-path patterns:

| Pattern | Sites | Cost |
|---|---|---|
| `state.heap ++ [val]` | ~15 across target + emitter | O(n) list append per heap write |
| `length(state.heap)` to compute the new addr | co-located with above | O(n), doubled with the append |
| `Enum.at(state.code, state.pc - 1)` | interpreter fetch loop | O(pc) per instruction step |
| `length(state.trail)` / `length(state.choice_points)` | `backtrack`, `unwind_trail` | O(n) per choice-point restore |

Haskell fixed the equivalent issues in commits `8abf0df3`
(Array for instruction fetch), `f757b881` / `bdae58cf` (cached lengths),
and the struct split in `ab829695`. Rust is similarly flat-time on these
operations. Elixir is still O(n).

## Design

### 1. Heap: map keyed by integer addr + cached `heap_len`

BEAM gives us a small set of container options:

| Option | Append | Read | Replace | Notes |
|---|---|---|---|---|
| List + `++ [x]` (today) | O(n) | O(n) via `Enum.at` | O(n) via `List.replace_at` | Everything O(n) |
| Reversed list + prepend | O(1) | O(n) | O(n) | Worse for read-heavy access |
| Tuple + `Tuple.append` | O(n) | O(1) via `elem` | O(1) | Append copies the whole tuple |
| **Map keyed by addr** | **O(1)** | **O(1)** | **O(1)** | Slightly higher constants; easiest to reason about |
| Erlang `:array` | O(log n) | O(log n) | O(log n) | Clean API but slower constants than Map for dense writes |

Choosing **Map**. Addresses are dense integers `0..heap_len-1`, so
indexed iteration (e.g. for `Enum.slice(heap, addr+1, arity)` to read
structure args) becomes a tight comprehension over a known integer
range, with `Map.get/2` at O(1).

**Struct change:**

```elixir
# Before
defstruct pc: 1, cp: :halt, regs: %{}, heap: [], trail: [],
          choice_points: [], stack: [], code: [], labels: %{}

# After
defstruct pc: 1, cp: :halt, regs: %{}, heap: %{}, heap_len: 0,
          trail: [], trail_len: 0,
          choice_points: [], cps_len: 0,
          stack: [], code: {}, labels: %{}
```

**Touch-site pattern:**

```elixir
# Before
addr = length(state.heap)
new_heap = state.heap ++ [val]
%{state | heap: new_heap}

# After
addr = state.heap_len
new_heap = Map.put(state.heap, addr, val)
%{state | heap: new_heap, heap_len: addr + 1}
```

Heap-slice reads (used in `step_get_structure_ref`, `eval_arith`):

```elixir
# Before
args = Enum.slice(state.heap, addr + 1, arity)

# After
args = for i <- (addr + 1)..(addr + arity), do: Map.get(state.heap, i)
```

Trail rewind (`unwind_trail`) and `unify` use `List.replace_at(heap, addr, v)`.
With a Map this becomes `Map.put(heap, addr, v)` — O(1) and simpler.

### 2. Cached counter for trail

`trail` stays a list — prepend is already O(1). Only `length/1` calls
are costly. Add a `trail_len` counter field and maintain it on every
push/pop.

Two code paths actually read the length:

1. `unwind_trail(state, mark)` in the runtime — currently `length(state.trail)`
   on each call. Uses cached `state.trail_len`.
2. `backtrack` — `length(cp.trail)` to establish the mark. Choice
   points snapshot `trail_len` at save time so the restore path reads
   `cp.trail_len` directly.

**Not adding `cps_len`**: no caller ever reads `length(state.choice_points)`
in the current codebase, so there's nothing to optimize. Skip per
"no speculative generality".

### 3. Code: tuple instead of list

`code` is set once at project load and never mutated. Emitting it as a
tuple `{instr1, instr2, ...}` makes `fetch/1` a `elem(state.code, state.pc - 1)`
— O(1). Lowered mode doesn't use `code` at all (each segment is a
`defp`), so this only benefits interpreter mode, but the change is
trivial.

## Correctness risk assessment

The change is purely representational — `heap[addr]` has the same
semantics whether stored as a list or a map. Sites that look vulnerable:

- **Heap slicing for structure args.** Switches from `Enum.slice` to a
  comprehension over an integer range. Only subtle point: `Map.get/2`
  returns `nil` for missing keys; if any code writes a non-contiguous
  heap, we'd read `nil` instead of crashing on out-of-range list access.
  This should never happen in correct WAM code, but a plunit test
  confirms it.
- **Trail rewind.** `List.replace_at(heap, addr, v)` on a list preserves
  the list's length; `Map.put/3` adds a key if `addr ≥ heap_len`, which
  would silently grow the heap. The `heap_len` field guards against
  this: we only ever write to addrs `< heap_len`, and `Map.put` on an
  existing key is update.
- **Choice point save/restore.** Must save `heap_len` alongside `heap`
  so restoration is consistent. Added to the CP struct.

## Plan of commits

1. **Plan doc** (this file).
2. **Heap: Map + `heap_len`.** Covers struct, all target writes/reads,
   all lowered-emitter writes/reads, choice point save/restore, unify,
   unwind_trail, deref_var, eval_arith, step_get_structure_ref.
3. **Cached `trail_len` / `cps_len`.** Covers push/pop sites and the
   `length/1` readers.
4. **Code as tuple.** Predicate-wrapper emitter produces `{...}`;
   `fetch/1` uses `elem/2`; struct default updated.

Each commit should keep the test suite green. Expected affected tests:

- `tests/test_wam_elixir_target.pl` — pattern-match assertions on the
  generated Elixir. Expect updates to the `heap: []` / `code: []`
  defaults and the `state.heap ++ [...]` patterns.
- `tests/test_wam_elixir_utils.pl` — no expected changes (touches only
  label/reg utilities).

## Out of scope (noted here so it's not forgotten)

| Item | Why deferred |
|---|---|
| Pre-resolve string labels → PC ints | Phase C in Haskell; substantial refactor of call/execute/try_me_else/switch. Separate PR. |
| Pre-parse functor arity into instruction tuple | Minor gain; needs generator change. Do alongside Phase C. |
| FFI path (skip-compile fact predicates) | Phase D. No Elixir FFI infrastructure exists. |
| Parallel seeds via `Task.async_stream` / `Flow` | Phase D. Needs benchmark-harness wiring, isolated-state verification. |
| INLINE / UNPACK equivalents | No equivalent in Elixir/BEAM; JIT handles what it handles. |

## Expected impact

Haskell's Phase A–B together closed roughly an order of magnitude on
heap/trail-heavy workloads (the ~2518ms → ~1350ms → further is mixed
Phase A–C). Elixir starts with the same O(n) patterns in roughly the
same places, so a similar 5–10x speedup on the effective-distance
benchmark is plausible once Phase A lands. Actual numbers pending
the benchmark run after implementation.

---
*Co-authored-By: Claude Opus 4.7 (1M context)*
