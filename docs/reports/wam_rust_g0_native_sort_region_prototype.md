<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM: gate G0, native sort and one Stage-2 region on heap cells (D137)

**Date:** 2026-10-11. **Ledger:** D137 (follows D128).
**What:** Gate G0 of the heap-cell rewrite
(`docs/proposals/wam_rust_heap_cell_rewrite_design.md` §12). A throwaway
prototype of the design's `Cell` representation runs `sort/2` (the whole
sort family that a resolve uses) and the largest Stage-2 region,
`group_keyed/2`, on the exact argument terms that a real resolve passes to
them at N=40 and N=5000. It measures their cost against today's runtime on the
same calls, plus boundary copy-in and copy-out and memory. The prototype and the
instrumented crate were never committed. This report, ledger row D137 and the
measured column in design §14 are the committed outputs.

## Headline

All Ir figures are per warm resolve (or per warm call) on origin/main
`4121f9f`. "Proto" is the prototype with comparator fast paths (v2); v1 is the
plain version of the comparator, without the fast paths.

| part | N=40 today | N=40 proto | N=5000 today | N=5000 proto | N=5000 ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 23 × `sort/2` (the only sort-family builtin a resolve calls) | 2.74 M | 0.63 M (v1 0.75 M) | 197.6 M | **32.1 M** (v1 38.6 M) | 6.2× (v1 5.1×) |
| 2 × `group_keyed/2` region | 0.59 M | 0.068 M | 60.2 M | **6.7 M** | 9.0× |
| boundary copy-in of catalog + requests (base segment) | 0 (built outside `call_pred`) | 0.089 M | 0 | **8.9 M** | new cost |
| boundary copy-out of the answer | 0.004 M | 0.013 M | 0.004 M | 0.016 M | negligible |
| wall, sort calls (median) | 0.258 ms | 0.046 ms | 20.3 ms | 2.9 ms | 7.0× |
| wall, `group_keyed` calls (median) | 0.081 ms | 0.005 ms | 8.9 ms | 0.58 ms | 15.4× |
| wall, copy-in | — | 0.007 ms | — | 0.76 ms | |
| catalog in memory: `Value` trees (estimated) vs base segment | 68.7 KB | 20.4 KB | 6.84 MB | **2.04 MB** | 3.4× smaller |

- **Correctness.** For all 25 dumped calls at both sizes (23 `sort/2` and 2
  `group_keyed/2` per resolve), the prototype's result, copied out with
  `to_value`, prints byte-identical to today's output under the runtime's
  `Display`. It is also structurally identical except in one respect: today
  returns an empty list as `Value::List([])`, and the cell model returns the
  atom `[]`. This affects 4 of 25 outputs, inside `deb(...)` version terms. The
  design (§3.1, §9.1) intends this change, the shim's `list_items` accepts
  both forms, and both print `[]`. A separate cross-check of 30 comparator
  and identity quirk cases against today's `term_compare_derefed`,
  `term_compare` and `terms_identical` also matches 30/30.
- **The two measured §14 targets are met.** The sort family comes in at
  32–39 M against a target of 35–65 M. The region's measured share is 6.7 M;
  all regions together project to 16–22 M against 40–70 M. Copy-in is
  8.9 M, inside the ≤10 M allowance, and it replaces today's 19.0 M
  `reset_query`.
- **Revised N=5000 projection: about 71–92 M Ir** if mechanics reach their
  §14 target, and 105 M if mechanics do not improve at all. Today's total is
  392 M. Both figures are at or below the low end of the 100–160 M planning
  range. If wall time scales with Ir, that is ~10–15 ms, or ~0.6–0.9× SWI's
  17.5 ms. **Recommendation: go for R1/R2, with the §14 and R3/R4
  adjustments listed at the end.**

## Method

- `export LANG=C.utf8 LC_ALL=C.utf8`, `mkdir -p output/advanced`. The resolver
  was compiled into a scratch directory with `build.pl` (the same step as
  `examples/pkg_resolver/rust/build.sh`, but writing outside the repo), and
  the shim was copied in. Cases were made with `gen_scale_catalog.mjs` +
  `scale_to_case.mjs` at N=40 and N=5000 (packages 62 / 7,522, depends
  156 / 15,003). The probe is `resolve_layered([p30])`, as in D128.
- **Instrumented scratch crate** (a copy of the generated crate, edited by a
  script):
  - every builtin op name is logged, which gives the census;
  - the `sort/2|msort/2`, `keysort/2`, `sort/4` and `list_to_set/2` arms moved
    into `#[inline(never)]` methods, with an outer wrapper that dumps the
    deref'd input, the output before and after, and `Display`;
  - `region_group_keyed_dispatch` got the same wrapper;
  - the bench shim repeats `call_pred` (`UW_BENCH_REPEAT`) and dumps the
    boundary terms (catalog, requests, answer).

  The dump uses a typed prefix encoding (atom/int/float bits/str/list/var).
  Dumping and timing are switched by environment variables, so the measured
  build is the same binary with them off.
- **Census:** a resolve calls `sort/2` 23 times and no other sort-family
  builtin (no `msort`, `keysort`, `sort/4`, `predsort` or `list_to_set`) at
  either N. Two of the calls are large: the dep rows (156 / 15,003 elements)
  and the pkg rows (62 / 7,522). The other 21 have 1–10 elements. Two
  `group_keyed/2` calls follow, one on each large sorted list.
- **Today's cost** comes from callgrind with `--collect-atstart=no
  --toggle-collect` on the inner arm and region methods and `--dump-after`
  each one, so there is one dump per call. Figures are taken from the
  second (warm) resolve of `UW_BENCH_REPEAT=2`; the cold resolve was 0–8%
  higher. Ir per warm resolve is `(Ir(1+k) − Ir(1))/k` with k=10 / 2, as in
  D128: 23.18 M / 392.24 M (D128: 23.30 M / 388.51 M). Inclusive
  per-function costs were differenced the same way.
- **Prototype cost:** the same toggle-and-dump method, on `#[inline(never)]`
  harness functions:
  - `g0p_run`: one dumped call;
  - `g0p_copy_in`: a warm copy-in, after a cold one has filled the
    per-program functor cache;
  - `g0p_copy_out`.

  Each call's inputs are copied into the heap first and not measured, because
  on the cell machine they already are heap cells. A pitfall to note: release
  builds fold identical functions, so an identical "cold" copy of
  `g0p_copy_in` was at first merged with it and counted twice. The cold copy
  now has a distinct body.
- **Wall time:**
  - today: per-call `Instant` timing in the outer wrappers (`UW_G0_TIME`),
    median over warm resolves 2..R (R=21 at N=40, R=11 at N=5000);
  - prototype: median of 21 repeats per call, truncating the heap back after
    each repeat.

  The box is a shared 4-core container (load ~1.8), so wall numbers are
  indicative; Ir is the primary metric.
- **Memory:** cells in the base segment (×8 bytes) and the heap growth of the
  measured calls. The size of today's catalog as `Value` trees is computed
  from the dumped tree: `size_of::<Value>()` = 24 per element, and
  `Arc<Spine{Vec<Value>, AtomicU8}>` = 48 per non-empty spine, before
  allocator rounding. Process max RSS comes from `getrusage(RUSAGE_CHILDREN)`.
- The same toolchain (rustc 1.94.1) builds both sides. That matters: both
  use `slice::sort_by` and `dedup_by`, so with identical comparator answers
  they make the same comparison sequence and the same stable output, even
  where today's comparator is not transitive (§7.2).

## Prototype design (what was built)

A standalone crate (~1.1 K lines; mimalloc, the same default release
profile as the generated crate).

- **`Cell(u64)`, §3.1 tags:** `REF`=0, `VAR`=1 (`(n<<4)|kind`, the §3.4
  prefix table, kind 15 = interned full name), `ATOM`=2 (the interner id),
  `INT`=3 (61-bit), `STR`=4 → `FUN` header + args, `LIS`=5 (head at the
  address, tail at +1), `FUN`=6 (functor id), `BOX`=7
  (Float/BigInt/Bool). The heap is a `Vec<Cell>` and there are no `Drop`s.
  Functors are interned as `(name atom, arity)`.
- **Native API subset (§7.1):** `deref` (follows `REF` until it reaches a
  non-`REF` cell, and stops at a `VAR`), `view`, `list_cursor` and
  `list_next` (`Elem`, `End` or `Tail`), `mk_list`, `mk_str`, `mk_int`,
  `mk_float`, `fresh_var`, `bind`.
- **`compare_std` (§7.2):** classes var < number < atom/Bool < compound.
  - numbers: `as f64`, NaN = Equal, and on a tie Float sorts before Integer;
  - atoms: text;
  - `STR`/`STR`: arity, then name text, then arguments;
  - `LIS`/`LIS`: head, then tail (iterative);
  - `LIS` vs a `STR` of arity n≠2: `2.cmp(n)`;
  - `LIS` vs arity 2: Equal if the list is proper, otherwise `"[|]"` vs the
    functor name.

  v2 adds same-tag fast paths before the class table: `STR`+`STR`, where the
  same functor skips the name compare and an argument pair that derefs to
  the same cell is skipped without recursing; `ATOM`+`ATOM`; and
  `INT`+`INT`, kept as an `f64` compare for the 2^53 quirk.
- **`identical`:** the same cell after deref; floats by exact `==`; no
  Integer/Float or Bool/atom cross-identity; compound by functor id and
  arguments.
- **`sort/2` and `msort/2`:** the cursor collects deref'd element cells into a
  reused scratch `Vec<Cell>`, then `sort_by(compare_std)`, `dedup_by(== Equal)`
  for `sort/2`, `mk_list`, and binding the output. A partial or non-list
  argument fails, as `deref_list_arg` + `Value::List` does. `keysort/2` is
  implemented too, but no resolve calls it.
- **`group_keyed/2` (region 3b) on cells:**

  ```rust
  // spine check first: today materialises the list and declines on an open tail
  loop { match self.list_next(&mut k) { Step::Elem(_) => {}, Step::End => break, Step::Tail(_) => return None } }
  loop { match self.list_next(&mut k) {
      Step::Elem(e) => {
          let (k2, x2) = match self.key_val(e) { Some(kv) => kv,            // K-_-X: two FUN-id checks
                                                 None => { self.heap.truncate(mark); return None } };
          if !values.is_empty() && self.identical(k2, key) { values.push(x2) }   // ==/2, not compare_std
          else { if !values.is_empty() { let l = self.mk_list(&values, nil);
                                         groups.push(self.mk_str(self.minus2, &[key, l])); values.clear(); }
                 key = k2; values.push(x2); }
      }
      Step::End => break, Step::Tail(_) => unreachable!() } }
  // last group, then bind A2 to mk_list(&groups, nil)
  ```

  The declines and the output are the same as `region_group_keyed_dispatch`.
  `K` and `X` are copied as cells, which are shared references into the
  input, not rebuilt trees.
- **Boundary:**
  - `from_value` imports a `Value` mirror (Arc-shared spines, interned
    names) through one import session. It maps atom symbols straight to
    `ATOM` ids and maps boundary functor symbols such as `"pkg/2"` to
    functor ids through a per-symbol cache (strip `str(...)` and a matching
    `/N`). It reserves each compound's block once, writes atomic arguments
    in place, and puts compound arguments on a reused work stack instead of
    recursing per node.
  - `to_value` prints exactly as `deref_heap` does: bare functor names,
    proper lists as `List`, a partial list as nested `"[|]/2"`, and
    variables by name.
- **Not in the prototype:** trail, obligations and scopes. Binding is
  unconditional, because every variable bound here was created after the last
  rollback point. Unify (the outputs were always unbound on entry), and the
  other regions.

## Correctness

| check | N=40 | N=5000 |
| --- | --- | --- |
| 23 `sort/2`: result, `Display` output | 23/23 byte-identical | 23/23 byte-identical |
| 2 `group_keyed/2`: result, `Display` output | 2/2 byte-identical | 2/2 byte-identical |
| typed structure (dump encoding) | 21/25 strict; 25/25 counting `List([])` and atom `[]` as the same term | same |
| boundary copy-out round trip of the answer | encoding and `Display` identical | identical |
| boundary copy-in → copy-out of catalog and requests | identical up to `"f/N"` → `"f"` normalization (what `deref_heap` prints) | identical |
| quirk cross-check (30 cases vs today's `term_compare_derefed`, `term_compare`, `terms_identical`) | 30/30 | |

The quirk cases are: `[a]` vs `f(b,c)` Equal; `[a|T]` vs `f(b,c)` Less and
vs `'Z'(b,c)` Greater; lists vs arity 1 and 3; 2^53 vs 2^53+1 and 2^62 vs
2^62+1 Equal; `1.0` vs `1` Less; NaN vs 1.0 Equal; `0.0` vs `-0.0` identical;
`Bool(true)` vs `true` Equal and not identical; `List([])` vs `[]` identical;
`_V10` vs `_V9` Less; `f/2(a,b)` vs `f(a,b)` identical; and atom, number and
compound class order.

**Coverage on real data.** The real calls exercise only the common branches.
At N=5000 `compare_std` ran 426,636 loop iterations:

- 254,278 `STR`/`STR` with the same functor;
- 85,913 identical-cell shortcuts;
- 61,174 atom text compares;
- 25,271 numeric compares;
- 0 class mismatches, variables, `LIS`/`LIS` steps or list-vs-structure
  quirks.

`identical` ran 57,527 times. The quirk branches are checked only by the
synthetic cross-check, not by resolver traffic.

## Measurements

### Per call (Ir, warm)

| # | call | elements | N=40 today | N=40 proto | N=5000 today | N=5000 proto | N=5000 ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `sort/2` dep rows `(N-V)-I-Req` | 156 / 15,003 | 2,289,792 | 525,190 | 160,340,056 | 24,871,478 | 6.4× |
| 1 | `group_keyed/2` dep rows | 156 / 15,003 | 464,828 | 54,469 | 45,460,355 | 5,151,389 | 8.8× |
| 2 | `sort/2` pkg rows `N-I-V` | 62 / 7,522 | 365,136 | 93,897 | 37,181,133 | 7,220,312 | 5.1× |
| 3 | `group_keyed/2` pkg rows | 62 / 7,522 | 125,686 | 13,585 | 14,719,917 | 1,538,905 | 9.6× |
| 4–24 | 21 small `sort/2` (1, 2 or 10 elements) | | 90,011 | 12,048 | 88,571 | 12,048 | 7.4× |
| | **sort total** | | **2,744,939** | **631,135** | **197,609,760** | **32,103,838** | **6.2×** |
| | **`group_keyed` total** | | **590,514** | **68,054** | **60,180,272** | **6,690,294** | **9.0×** |

A small `sort/2` of one element costs 1.0–4.3 K Ir today: argument decode,
`deref_list_arg`, a `Vec` and a spine. On cells it costs 206 Ir. v1 (without
the comparator fast paths) costs 753,231 / 38,627,272 for the sort total; the
fast paths remove 17% at N=5000.

### Boundary and memory

| | N=40 | N=5000 |
| --- | ---: | ---: |
| base segment after copy-in | 2,555 cells = 20.4 KB, 0 boxes | 254,996 cells = 2.04 MB, 0 boxes |
| copy-in Ir (warm; iterative, block-reserving import) | 89,309 | 8,924,863 (35 Ir/cell) |
| copy-in wall | 0.007 ms | 0.76 ms |
| copy-out Ir, prototype `to_value` / today's `deref_heap` in `call_pred` | 13,104 / 3,888 | 15,736 / 3,888 |
| the same catalog as `Value` trees (estimated) | 529 spines, 1,803 Values ≈ 68.7 KB | 52,471 spines, 179,995 Values ≈ 6.84 MB |
| heap growth of the 25 measured calls (their outputs; no GC) | 1,383 cells = 11 KB | 140,211 cells = 1.12 MB |
| today: process max RSS (one resolve) | 24.3 MB | 73.5 MB |
| today: `reset_query` per warm resolve (freed by `H = base` instead) | 0.31 M Ir | 19.0 M Ir |

- A first recursive `from_value`, with one call per node and `Vec` pushes,
  cost about 1.4× the final import (~13 M at N=5000). That figure is half of
  a measurement that counted the cold and warm passes together.
- Copy-in is new per-call work: today the shim builds the catalog `Value`
  once, outside `call_pred`. It is less than the `reset_query` cost it
  replaces (−19.0 M, +8.9 M).
- With the base segment reused across resolves (an R7 option), copy-in
  disappears for an unchanged catalog.

### Today's N=5000 resolve on current main (inclusive, per warm resolve)

| part | Ir | share |
| --- | ---: | ---: |
| `sort/2` × 23 | 197.6 M | 50.4% |
| other builtins | 17.6 M | 4.5% |
| regions (`lowered_call`) | 140.4 M | 35.8% |
| — `group_keyed` (`terms_identical` 28.8 M, `region_key_val` 17.1 M) | 60.2 M | 15.3% |
| — `key_dep_rows` | 51.3 M | 13.1% |
| — `build_tree` | 14.8 M | 3.8% |
| — `key_pkg_rows` | 13.9 M | 3.5% |
| — others (`tree_lookup`, `filter_satisfies`) | 0.3 M | 0.1% |
| `reset_query` | 19.0 M | 4.8% |
| mechanics and remainder | 17.7 M | 4.5% |
| **total** | **392.2 M** | |

`group_keyed` is still the largest region, which is why it was the one
ported. Today's `group_keyed` cost is materialization: `deref_heap` inside
`terms_identical` rebuilds both keys for each comparison, and
`region_key_val` clones `Arc`s. On cells a key is a `STR` cell, `identical`
walks it in place, and the output shares `K` and `X` with the input.

## Revised N=5000 projection (§14)

| part | today (main) | §14 target | G0 | status |
| --- | ---: | ---: | ---: | --- |
| sort family | 197.6 M | 35–65 M | **32.1 M** (v1 38.6 M) | measured |
| other builtins | 17.6 M | 10–17.5 M | 10–17.6 M | unchanged, unvalidated |
| regions | 140.4 M | 40–70 M | **15.6–22.4 M** | `group_keyed` 6.7 M measured; the other 80.2 M projected at the measured 9.0× (`group_keyed`) to 5.1× (sort v1) → 8.9–15.7 M |
| `reset_query` | 19.0 M | ~0 | ~0 | follows from `H = base` |
| mechanics | 17.7 M | ~4 M | 4 M (target), 17.7 M if unchanged | unvalidated |
| boundary copy-in / copy-out | (outside the resolve) | ≤10 M | **8.9 M / 0.02 M** | measured; copy-in is 0 with base-segment reuse |
| **total** | **392.2 M** | **100–160 M** | **70.6–91.5 M**; 105.2 M with mechanics unchanged | |

- **Is the large-catalog target still achievable? Yes.** The two parts G0
  measured beat their targets: the sort family by a small margin, and the
  region by about 6×. Even the pessimistic assembly, with mechanics not
  improved at all and the three unported regions gaining only the sort's
  5.1×, is 105 M, inside the 100–160 M range. If wall time scales with Ir
  from today's 57.9 ms warm resolve on this box, 71–105 M is ~10–15 ms
  (~0.6–0.9× SWI's 17.5 ms reference). For the ported operations, the
  measured wall ratios were larger than the Ir ratios (sort 7.0× vs 6.2×,
  `group_keyed` 15.4× vs 9.0×), so that assumption looks conservative here.
- **What dominates the residual.** The sort comparisons are 32 M, or 31–45%
  of the projected total. Call 0 alone is 24.9 M. A stable sort of 15,003
  `(N-V)-I-Req` rows makes about n·log₂n ≈ 2×10⁵ comparisons at ~120 Ir
  each, and each comparison walks three nested `-/2` levels down to the
  package-name atom and its text compare. This is inherent to the
  resolver's sort; a cell representation cannot remove it. After that come:
  - other builtins: 10–18 M;
  - the three unported regions: 9–16 M; `key_dep_rows` is the largest
    unmeasured piece, and it is construction-heavy;
  - copy-in: 8.9 M;
  - mechanics: 4–18 M.

## Risks and surprises

- **Most of the projected total is still unmeasured.** Mechanics (4 M is a
  target; today 17.7 M), the other builtins, and the three regions G0 did not
  port (80 M today, projected by ratio) together make up 32–48% of the
  projected total. The pessimistic case still lands in range, but R5 should
  measure `key_dep_rows` early, because it builds new terms per row rather
  than sharing them.
- **Copy-in is a real new cost.** A naive recursive import costs ~1.4× the
  prototyped one. R3 should use the iterative block-reserving import, keep
  the functor-symbol cache per program and the work stack per machine, and
  treat base-segment reuse (R7) as worth ~9 M at N=5000.
- **`[]` changes structure but not text.** Copy-out returns `Atom("[]")`
  where today returns `List([])`. Text and JSON are unaffected (the shim's
  `list_items` handles both), but any Rust consumer that pattern-matches an
  empty `Value::List` would change behavior. R3 should grep for such
  consumers.
- **Real data does not exercise the quirks.** No resolver call reaches a
  list-vs-structure, variable or class-mismatch comparison, so those branches
  rest on the 30-case cross-check and on §11.2's reference-model tests, not on
  traffic.
- **Same comparison sequence needs the same `sort_by`.** Byte identity for
  Equal-but-distinct elements (today's comparator is not transitive) relies
  on both sides using std's stable sort with identical comparator answers.
  A comparator "optimization" that changes any answer, or a hand-written
  sort, would break that.
- **Layout and trailing.** Inputs were copied fresh and compact before each
  call. In the machine they come from program construction (sort output →
  `group_keyed` input), which may cost some wall time but not Ir. The
  prototype has no conditional-trail check on bind: one compare per output
  bind, negligible here.
- **Memory.** There is no GC: the outputs of the 25 calls are 1.12 MB of heap
  per resolve at N=5000, reclaimed by `H = base`. The base segment is 3.4×
  smaller than the `Value` trees it replaces.
- **Function folding.** Release builds merge identical functions, which
  silently doubled one callgrind toggle in this gate. Any future
  per-function gate should give each measured wrapper a distinct body.

## Recommendation: go for R1/R2, with adjustments

1. **§14.** Replace the sort-family target 35–65 M with the measured
   32–39 M. Replace the region target 40–70 M with 16–22 M (`group_keyed`
   6.7 M measured, the rest projected). Record the boundary as measured
   8.9 M copy-in per call and 0.02 M copy-out. Done in this change, marked
   measured or projected.
2. **R3.** Use an iterative, block-reserving `from_value` with a
   per-program functor-symbol cache, which the prototype shows at ~35 Ir per
   cell. Move base-segment reuse from "later option" to an early R7 item.
3. **R4.** The native sort is the simple design: a cursor into a scratch
   `Vec<Cell>`, then `sort_by(compare_std)`, `dedup_by` and `mk_list`. Include
   the same-tag fast paths in `compare_std` (−17% sort Ir).
4. **R5.** Port `key_dep_rows` (51 M today, the largest region left) and
   measure it before the other regions, as a small follow-up gate.
5. Nothing found argues for a representation change. The 8-byte cell, the
   `FUN` header with interned functor ids, and the `LIS` pair work as
   designed for both the comparator and the region.

## Files

Committed: this report, ledger row D137 in
`docs/proposals/JS_TARGETS_PARITY_PUNCHLIST.md`, and the measured column in
§14 of `docs/proposals/wam_rust_heap_cell_rewrite_design.md`. The prototype
crate, the instrumented scratch crate, the dumps and the callgrind outputs
stayed in scratch space and are not committed. `resolver.pl` and
`resolver_store.pl` are unmodified. The generated crate was not regenerated
in the repo.
