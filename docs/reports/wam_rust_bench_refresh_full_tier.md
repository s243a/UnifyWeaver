<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# uw-resolve benchmark refresh — full Rust lowered tier (regions 1–5 + general recognizer)

Re-measure of the whole cross-target `uw-resolve` benchmark on one box so the
table in [`examples/pkg_resolver/BENCHMARKS.md`](../../examples/pkg_resolver/BENCHMARKS.md)
reflects the **full current Rust lowered tier**: the five Stage-2 fused regions
(D78–D81 regions 1/2/3a/3b/4) **plus** region 5 (`dep_breaks/5`, D88,
[`wam_rust_stage2_region5.md`](wam_rust_stage2_region5.md)) **plus** the general
deterministic-recursion recognizer (D89, sibling-gap family
`filter_satisfies`/`key_pkg_rows`/`tree_lookup`,
[`wam_rust_genrec_sibling_gap.md`](wam_rust_genrec_sibling_gap.md)) — all
default-ON and gated. The previous table (D82) predated region 5 and the general
recognizer.

**Box.** Coordinator container, `nproc`=4, Linux, `LC_ALL=C.UTF-8`. Absolute
times vary by machine; the ratios are the result. Every leg re-run on this one
box for internal comparability. HEAD `1be7be268` (ledger D89). MEASUREMENT ONLY:
`resolver.pl`, `resolver_store.pl`, and all target/spec source unmodified.

## Gates (all legs, this box)

- term corpus (B1): **51/51** matched SWI — SWI, Go, Rust, wamjs, ClojureScript.
- term differential (B2): **2600 cases / 0 divergences / 0 crashes** — Go, Rust,
  wamjs, ClojureScript.
- store differential: **503 cases / 0 divergences** — wamjs store, Go store,
  Rust store.

## Rust term binaries — verified relink

Full lowered tier ON (default) vs all-lowering OFF (`UW_REGION{1,2,3A,3B,4,5}_OFF=1`
and `UW_GENREC_OFF=1`; F11 is opt-in and stays off). Distinct sha256, and the ON
crate's generated `lib.rs` carries **9** `region_*_dispatch` arms (regions 1–5 +
the three genrec shapes) vs the OFF crate's **0**:

```
OFF sha256 ca3f5c51f6a47e4b8243492cc3cfd406ab19ca8df3af5ef96bcc0d1a0581d550  (0 dispatch arms)
ON  sha256 90f3b42400f49408fe814bc29924a687df1ac4b3073dced48fb1d1bff966133d  (9 dispatch arms)
```

ON arms: `region_matching_deps_dispatch` (1), `region_matching_versions_dispatch`
(2), `region_key_dep_rows_dispatch` (3a), `region_group_keyed_dispatch` (3b),
`region_build_tree_dispatch` (4), `region_dep_breaks_dispatch` (5),
`region_filter_satisfies_dispatch`, `region_key_pkg_rows_dispatch`,
`region_tree_lookup_dispatch` (genrec). The full 5,000-package B3 term-lane
output is **byte-identical** OFF vs ON (`cmp`), `selection_size` 10 both.

## B3 `resolve_layered` (5k catalog) — ON vs OFF, interleaved, this box

Interleaved (OFF then ON each round) on the same 5,000-package `case_5000.json`,
timing `resolve_ms` (`load_ms` — JSON→term, not the index path — is ~23 ms
either way):

| round | OFF (ms) | ON (ms) | delta (ms) | delta (%) |
|---:|---:|---:|---:|---:|
| 1 | 2090.9 | 375.5 | −1715.4 | −82.04 |
| 2 | 2029.7 | 385.3 | −1644.4 | −81.02 |
| 3 | 1903.4 | 372.6 | −1530.8 | −80.42 |
| 4 | 1918.6 | 371.9 | −1546.7 | −80.62 |
| 5 | 1927.0 | 371.9 | −1555.1 | −80.70 |
| 6 | 1913.4 | 366.9 | −1546.5 | −80.82 |
| 7 | 1927.1 | 370.6 | −1556.6 | −80.77 |
| 8 | 2000.0 | 372.6 | −1627.4 | −81.37 |

**OFF median ≈1927 ms; ON median ≈372 ms; delta median ≈−80.7%.** All eight
rounds negative, ranges non-overlapping (OFF 1903–2091 ms; ON 367–385 ms). This
is the **cumulative** delta pristine-interpreter → full lowered tier — deeper
than the D82 table's −74% (regions 1/2/3/4 only), the extra ~7 points coming from
region 5 and, on this indexed resolve path, mostly the general recognizer's
`key_pkg_rows`/`tree_lookup`/`filter_satisfies` (D89's isolated scale-5000 A/B was
−21% to −24% on top of regions 1–5).

## Measured cells (this box)

| Leg | B1 corpus | B2 differential | B3 5k resolve (load / resolve) |
|---|---:|---:|---|
| SWI (oracle) | 0.113 s | 2.6 s (2600) | 0.027 s / 0.020 s (carried over) |
| Go term | 0.070 s | 29.5 s (2600) | 0.098 s / 15.12 s |
| Rust term (full tier ON) | 0.048 s | 19.1 s (2600) | 0.024 s / 0.385 s |
| wamjs term | 0.209 s | 16.5 s (2600) | — (store row) |
| ClojureScript | 2.02 s | 107.7 s (2600) | 0.228 s / 28.24 s |
| wamjs store | 0.209 s | 9.23 s (503) | seek / 0.163 s (0.63%, 820 reads) |
| Go store | 51/51 | 34.8 s (503) | seek / 0.56 s (0.97%, 880 reads) |
| Rust store | 51/51 | 3.74 s (503) | seek / 0.040 s (0.90%, 820 reads) |

Store bytes-read/read-count are byte-identical to D82 (Rust/wamjs store 10,305
bytes / 820 reads; Go store 11,025 / 880), confirming the store data path is
unchanged by the term-lane regions. The SWI B3 cell is carried over: the frozen
SWI reference loader (`rust/swi_scale_ref.pl`, `store/scale_demo.pl`) still cannot
construct the current 5k catalog (verified: it emits no `swi_term_*` lines and
exits non-zero), and SWI + `resolver.pl` are frozen, so SWI's native ~20 ms
resolve is unchanged.

## Is Rust faster than SWI? (per axis, this box)

- **Startup (B1):** Rust **beats** SWI — 0.048 s vs 0.113 s.
- **Throughput (B2):** SWI still **~7.2× faster** — Rust 19.1 s vs SWI 2.64 s
  (was ~8.8× at the first table, ~8.2× before region 5).
- **Term resolve (B3):** SWI **~19.5× faster** — Rust 0.385 s vs SWI 0.0197 s —
  but the gap collapsed from **~98×** (pristine interpreter) → 26× (D82) → ~19.5×
  now.
- **Store resolve (B3):** SWI **~2.0× faster** — Rust store 0.040 s vs SWI
  0.0197 s — within a factor of two, touching 0.90% of the store.

The Rust-vs-Rust improvement percentages above are Rust-lowered-tier vs
Rust-pristine-interpreter — they do **not** mean Rust overtook SWI on those axes.
Rust beats SWI only on startup.
