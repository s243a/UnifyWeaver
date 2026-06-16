# Cross-target audit — `switch_on_constant` key lookup

**Question:** the Rust WAM target had a `switch_on_constant` (first-argument
indexing) bug — the runtime binary-searched the index table assuming it was
sorted, but generation emitted it in clause order (`k0,k1,..,k9,k10,..`, not
lexically sorted), so binary search dropped keys. Do the other WAM targets share
this defect? (Fixed for Rust in the `switch_on_constant` linear-scan change +
`tests/test_wam_rust_switch_on_constant_keydrop_exec.pl`.)

**Answer: no.** The defect was unique to Rust. Every other target that uses an
order-dependent lookup (binary search or an ordered/tree map) **sorts the table
at generation with the same comparator the lookup uses**, or uses a lookup that
is inherently order-independent (linear scan / hash map / intern-id map). Rust
was the only one that binary-searched *without* a matching generation-side sort.

## Per-target findings

| Target | `switch_on_constant` lookup | Order-safe because | Verdict |
|--------|------------------------------|--------------------|---------|
| **rust** | linear scan by `==` (was binary search) | order-independent (after fix) | fixed |
| **go** | label: linear scan; `…Pc`: `sort.Search` | `…Pc` cases are `sort.Slice`-sorted at gen with the **same** `compareValues` used by `sort.Search` (`wam_go_target.pl` ~2520 / lookup ~2257) | correct |
| **fsharp** | label: `Map.tryFind`; `…Pc`: `binarySearchStr` | label map is `Map.ofList` (self-sorting); `…Pc` table is `List.sortBy fst` at gen and `binarySearchStr` uses `String.Compare(…, Ordinal)` — F# `sortBy` on strings is ordinal too, so they agree (`wam_fsharp_target.pl` ~3405; `fsharp_wam_bindings.pl:879`) | correct |
| **haskell** | `…Pc`: `IM.lookup` (IntMap); label: `Map.lookup` (Data.Map) | IntMap is keyed by the **interned atom id / integer** (no string ordering at all); Data.Map is `Ord`-organized — both order-independent (`wam_haskell_target.pl` ~2271 / ~2480) | correct |
| **cpp** | `std::map::find` | tree map, self-organizing | correct |
| **python** | `dict` membership | hash map | correct |
| **clojure** | hash map `contains?` | hash map | correct |
| **elixir** | `Map.get` | hash map | correct |
| **kotlin** | map indexing | hash map | correct |
| **c** | linear scan (`val_equal` loop) | order-independent | correct |
| **lua** | linear scan (`ipairs` + `==`) | order-independent | correct |
| **r** | linear scan (`identical`) | order-independent | correct |
| **scala** | linear scan (`cases.filter(_.value == key)`) | order-independent | correct |
| **wat** | linear scan loop | order-independent | correct |
| **llvm** | delegates to the C runtime (`@wam_switch_on_constant`) | C uses linear scan | correct |
| **jvm** | `stepSwitchOnConstant` (deferred method) | not order-dependent | correct |
| **ilasm** | not implemented (no first-arg indexing) | — | N/A |

## The invariant (and why Rust broke it)

An order-dependent constant-index lookup (binary search / ordered map) is correct
**iff** the index table is built sorted by exactly the comparator the lookup
uses. Go and F# satisfy this by sorting at generation with the matching
comparator; Haskell sidesteps it with intern-id `IntMap` / `Ord` `Data.Map`.
Rust's handler binary-searched on the atom *string* but generation emitted clause
order — numeric clause order (`k0..k9,k10,..`) is **not** lexical order
(`"k10" < "k2"`), so the search mis-navigated. Rust is now a linear scan, which
removes the dependency on table order entirely.

**Regression risk for the order-dependent targets (go/fsharp):** their
correctness depends on the gen-side sort staying in lock-step with the lookup
comparator. A future change that drops the sort, or changes one comparator but
not the other, would reintroduce this class of bug. The order-independent
targets (rust/c/lua/r/scala/wat + the hash/tree-map targets) have no such
coupling.
