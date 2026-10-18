<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# uw-resolve cross-target benchmark refresh — 2026-09-07 (Stage-2 Rust lowered tier)

Single-box re-measurement backing the refreshed
[`examples/pkg_resolver/BENCHMARKS.md`](../../examples/pkg_resolver/BENCHMARKS.md).
Box: 4 cores (`nproc`=4), Linux, `LC_ALL=C.UTF-8`. HEAD `ddd627a6a` (ledger
D81 — the Rust deterministic lowered tier complete: regions 1, 2, 3a, 3b, 4
default-ON, F11 OFF). No target source, template, or spec was modified — this
is a measurement-only pass. Tool versions: SWI 9.0.4, Go 1.24.7, Node 22.22.2,
cargo 1.94.1, nbb 1.5.212.

## Gates (all green on this box)

| leg | corpus | differential |
|---|---|---|
| SWI oracle | 51/51 (reference) | 2600 (reference) |
| Go WAM term | 51/51 | 2600 / 0 |
| Rust WAM term (regions ON) | 51/51 | 2600 / 0 |
| Rust WAM term (all regions OFF) | 51/51 | 2600 / 0 (per D78–D81; B3 A/B here) |
| wamjs term | 51/51 | 2600 / 0 |
| ClojureScript (nbb) | 51/51 | 2600 / 0 |
| wamjs store | 51/51 | 503 / 0 |
| Go store | 51/51 | 503 / 0 |
| Rust store | 51/51 | 503 / 0 |

## Term legs (B1 / B2 / B3)

- SWI: B1 0.107 s (median of 3); B2 same-run ~2.5 s; B3 carried over (frozen
  loader can't parse the current 5k generator's version/alternatives rows —
  confirmed across `swi_scale_ref.pl`, `store/scale_demo.pl`, `cljs/bench_scale.sh`).
- Go term: B1 0.077 s; B2 27.0 s vs SWI 2.64 s (2600/0); B3 load 0.084 s /
  resolve 14.78 s, selection 10.
- Rust term: B1 0.072 s; B2 20.7 s (median of 20.64/20.71/21.14) vs SWI 2.52 s
  (2600/0) → 8.2×; B3 regions ON load 0.024 s / resolve 0.517 s, selection 10.
- wamjs term: B1 0.209 s; B2 13.57 s vs SWI 2.49 s (2600/0) → 5.5×.
- ClojureScript: B1 2.038 s; B2 100.95 s vs SWI 2.51 s (2600/0) → 40×; B3 load
  0.225 s / resolve 25.35 s, selection 10.

## Rust B3 regions ON vs OFF (the Stage-2 headline)

Two release binaries from the same codegen. OFF = every
`UW_REGION{1,2,3A,3B,4}_OFF=1` (pristine interpreter); ON = default (all five
regions). Genuine relink verified: OFF `lib.rs` has 0 `region_*_dispatch` arms,
ON has 5; distinct sha256:

```
OFF cc1a6d2a44881b3532cf7dd98331258fea324aebfcc9a4c0538576e9ba86f544
ON  2d4474d99988524adc83c0ea8bb4eceed11d9976405ed840c70bcb51231da6c4
```

Interleaved OFF-then-ON, 6 rounds, `case_5000.json`, `resolve_ms`:

| round | OFF (ms) | ON (ms) | delta (%) |
|---:|---:|---:|---:|
| 1 | 1924.8 | 504.7 | −73.8 |
| 2 | 1935.3 | 492.2 | −74.6 |
| 3 | 1911.3 | 510.0 | −73.3 |
| 4 | 1995.6 | 504.8 | −74.7 |
| 5 | 2019.5 | 524.4 | −74.0 |
| 6 | 1940.1 | 501.0 | −74.2 |

OFF median 1937.7 ms; ON median 504.8 ms; delta −74.0% (all six negative).
Output byte-identical OFF vs ON, `selection_size` 10 both. `load_ms` ~23 ms
either way. This reproduces the cumulative ledger claim (D81: ≈2000 → ≈530 ms,
≈−73%).

## Store legs (indexed backend, cached materialisation) — unchanged from D72

| leg | store diff (target/SWI, 503/0) | B3 resolve | bytes_read / total | reads | sel |
|---|---|---:|---|---:|---:|
| wamjs store | 8.45 s / 0.69 s | 0.159 s | 10,305 / 1,646,323 (0.63%) | 820 | 10 |
| Go store | 29.33 s / 0.69 s | 0.564 s | 11,025 / 1,142,225 (0.97%) | 880 | 10 |
| Rust store | 3.99 s / 0.67 s | 0.047 s | 10,305 / 1,142,225 (0.90%) | 820 | 10 |

`bytes_read` and `n_reads` are byte-identical to D72 → the store data path is
unaffected by the Stage-2 regions (the index builders are never invoked on the
`resolve_layered_store` path).

## References

- Region reports: `wam_rust_stage2_region{1,2,3,4}.md` (same directory).
- Ledger rows D78–D81 in `docs/proposals/JS_TARGETS_PARITY_PUNCHLIST.md`.
- Plan: `docs/proposals/WAM_RUST_LOWERED_TIER_THROUGHPUT_PLAN.md`.
