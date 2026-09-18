<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM mimalloc global allocator — A/B (D110)

**Date:** 2026-09-18. **Ledger:** D110. **Author:** Sonnet (implementer).
**What:** a drop-in `mimalloc` `#[global_allocator]` for the generated Rust
WAM binaries (`bench`, `uw_resolve`, `uw_resolve_store`), feature-gated
(`mimalloc`, default ON). OFF = the system (glibc) allocator, for a clean A/B.
This is byte-identical **by construction** — a global allocator changes *how*
memory is allocated, not *what* is computed — so the correctness risk is near
zero; the interesting questions are build/link (mimalloc vendors C) and
whether the wall-clock win D109 predicted actually shows up.

**Motivation (D109):** the B3 re-profile after the D101–D104 term levers
found the term path allocator-bound — glibc malloc internals (`_int_malloc`/
`_int_free`/`malloc`/`free`/`malloc_consolidate`/`unlink_chunk`) were **37.1%**
of B3's callgrind Ir, and recommended a `#[global_allocator]` swap as the
next, lowest-risk lever.

## The change

- **Dependency:** `mimalloc = { version = "0.1", optional = true }` in every
  generated crate's `[dependencies]` (vendors + `cc`-compiles the mimalloc C
  source; no system lib needed).
- **Feature:** `mimalloc = ["dep:mimalloc"]`, declared in `[features]` exactly
  like `store_lmdb`/`parallel` (Cargo's `dep:` syntax activates the optional
  dependency only when the feature is on), and added to `default` alongside
  `decorate_sort`/`intern`/`deref_memo`/`trail_enum`/`store_cache`. An OFF
  build (`--no-default-features --features "decorate_sort intern deref_memo
  trail_enum store_cache"`) never fetches or compiles the mimalloc crate at
  all — confirmed: the OFF `cargo build` pulled zero new crates.
- **Placement:** the `#[global_allocator]` static lives in each **binary**
  crate root, never in `lib.rs` — three places, all gated behind
  `#[cfg(feature = "mimalloc")]`:
  - `templates/targets/rust_wam/main.rs.mustache` → every generated crate's
    `src/main.rs` (the `bench` bin).
  - `examples/pkg_resolver/rust/shim/main.rs` → copied by `build.sh` to
    `src/bin/uw_resolve/main.rs` — the binary `run_corpus_rust.sh` /
    `run_differential_rust.sh` / `run_scale_rust.sh` actually exercise.
  - `examples/pkg_resolver/rust_store/shim/main.rs` → copied by `build.sh` to
    `src/bin/uw_resolve_store/main.rs` — same role for the store lane.

  Each binary is compiled as its own separate crate graph (cargo builds
  `bench`, `uw_resolve`, `uw_resolve_store` as independent executables that
  each depend on the shared library), so declaring the same
  `#[global_allocator]` item in each of the three binary roots does not
  collide — "only one global allocator per program" is satisfied per binary,
  and it is never declared in `lib.rs` (which would work too, since only one
  binary links each generated lib in practice, but the brief asked for the
  attribute to live in the binary roots, where it visibly governs the
  executable that gets benchmarked).

Wired into the generator two places:
- `src/unifyweaver/targets/wam_rust_target.pl` — the `[features]` block
  `format/2` string (default list + the new `mimalloc = ["dep:mimalloc"]`
  feature declaration, documented like the D106/D108 entries next to it).
- `src/unifyweaver/core/template_system.pl` — the **inline**
  `template(rust_wam_cargo, ...)` fact, which the module's own comment marks
  authoritative (`templates/targets/rust_wam/Cargo.toml.mustache` is a kept
  "companion mirror"); both were updated so they stay in sync, and both were
  exercised (the mirror is not yet wired into the live generation path, so
  this is `Cargo.toml`-content parity for a future migration, not a
  functional no-op skipped).

## Gate matrix (LC_ALL=C.UTF-8), ON (default) vs OFF

`OFF` = `--no-default-features --features "decorate_sort intern deref_memo trail_enum store_cache"`.

| Gate | ON | OFF |
| --- | --- | --- |
| Term corpus (`run_corpus_rust.sh`) | 51/51 matched SWI | 51/51 matched SWI |
| Term differential (`run_differential_rust.sh`, 2600 cases) | 0 divergences, 0 crashes | 0 divergences, 0 crashes |
| Store corpus (`run_corpus_rust_store.sh`) | 51/51, identical to term corpus | 51/51, identical to term corpus |
| Store differential (`run_differential_rust_store.sh`, 503 cases) | 0 divergences, 0 crashes | 0 divergences, 0 crashes |
| `cargo test --lib` (term crate) | 232/232 | 232/232 |
| `cargo test --lib` (store crate) | 232/232 | 232/232 |
| Transactional alias test (`test_wam_rust_foreign_tuple_aliases.pl`) | exit 0 | (test always builds with default features; see below — I additionally confirmed a manual `--no-default-features` build of the same generated project adds no warnings either) |

**Byte-identity, `cmp`-clean, ON vs OFF:**
- term differential output (2600 lines): `cmp`-clean.
- term corpus output (51 lines): `cmp`-clean.
- store differential output (503 lines, rebuilt against the SAME baked
  `STORE_DIR` for both configs — see harness note below): `cmp`-clean.
- store corpus output (51 lines, same baked-`STORE_DIR` discipline): `cmp`-clean.
- 5000-package B3 selection (`--bench` `selection_size`/`selection` fields):
  identical between ON and OFF (both configs pass the same `resolve_layered`
  through the same term crate; the differential/corpus checks above already
  cover byte-identity of every resolver output field at both scales).

**Harness note (same class of gotcha D106/D108 recorded):** the store crate
bakes its absolute `STORE_DIR` into `setup_foreign_predicates` at generation
time. `run_corpus_rust_store.sh` and `run_differential_rust_store.sh` each
call `build.sh` with their OWN `STORE_DIR` internally (`store/.out/corpus` vs
`store/.out/scale`), and `build.sh`'s own `cargo build` step always uses
**default** features — there is no feature-flag passthrough. To get a valid
OFF measurement without silently comparing an OFF binary baked to the wrong
store, I ran `build.sh` myself (regenerating + baking to the target store,
default/ON), copied that binary aside, then re-ran `cargo build --release
--bin uw_resolve_store --no-default-features --features "..."` **in the same
already-generated project** (so the baked path stays correct) and drove the
resulting OFF binary directly against that store's `cases.jsonl`, comparing
with the already-captured SWI oracle. Hit this exactly once during the work
(a first attempt that let `run_corpus_rust_store.sh`'s internal `build.sh`
silently re-bake the project to the corpus store while measuring against the
scale store produced 236/503 "divergences" — a harness artifact, not a
regression; re-run correctly, it is 0/503).

## Transactional alias test — exit 0, warnings unrelated to this change

`test_wam_rust_foreign_tuple_aliases.pl` generates an empty-predicate-list
project (`write_wam_rust_project([], ...)`) and runs `cargo test --test
foreign_tuple_aliases --quiet`; the plunit test passed (`.`, exit 0) both
through the harness and reproduced manually.

Honesty check on "zero warnings": this rustc (1.94.1, 2026-03-25) emits 4–5
pre-existing warnings from `state.rs` on **every** generated crate — an
`unreachable_patterns` warning (`_ => false` collectively unreachable against
~51 other arms), an unused `sum_parent_deg` variable/assignment, and an
unused `addr` binding in a `Value::Ref` match arm. These are **not**
introduced by this change: they reproduce identically (same line numbers)
building the committed HEAD `state.rs` untouched by any of this work — I
verified by reverting `state.rs` (and every other regenerated file except
`Cargo.toml`/`main.rs`) to the exact committed content and rebuilding; the
same 4 warnings appear. `mimalloc`/`global_allocator`/`unexpected_cfgs`
contribute **zero** additional warnings in any configuration (ON, OFF, or the
empty-predicate alias-test project) — the only thing this report can honestly
claim zero of. The pre-existing `state.rs` warnings are out of scope for
D110 (untouched by the generator changes here) and are not resolver.pl-facing.

## A/B measurement — wall-clock (NOT callgrind)

**Why not callgrind:** `valgrind --tool=callgrind` runs the whole program
under valgrind's own allocator emulation, which intercepts/replaces the
target's `malloc`/`free` calls — a global-allocator swap is invisible to it
(both ON and OFF report equivalent Ir once translated through valgrind's own
allocator). Wall-clock is the only instrument that can see this change.

**Method:** `uw_resolve --bench < .scale/case_5000.json` (B3: one
`resolve_layered` on the 5000-package scale catalog, same shape D94/D100/D109
used), term lane, interleaved ON/OFF reps (ON, OFF, ON, OFF, …) after one
warm-up rep each, to spread out any host-load drift evenly across both
configs rather than let it bias whichever config ran first/last. n=21 reps
each. Container is a shared/noisy host (visible below as a bimodal ON
distribution), so **median** (not mean) is the headline number, with the
full spread reported honestly.

| | resolve_ms | | load_ms | |
| --- | ---: | --- | ---: | --- |
| | ON (mimalloc) | OFF (system) | ON (mimalloc) | OFF (system) |
| median | **65.60** | 75.62 | **19.49** | 25.60 |
| p25 | 62.62 | 73.28 | 18.80 | 25.13 |
| p75 | 73.93 | 78.67 | 20.91 | 26.92 |
| min | 60.87 | 71.96 | 17.89 | 24.55 |
| max | 139.32 | 82.22 | 97.42 | 29.58 |
| n | 21 | 21 | 21 | 21 |

**resolve_ms median: -10.02 ms, -13.2%.** **load_ms median: -6.11 ms,
-23.9%** (load includes JSON parse + index build, also allocation-heavy).

**Honest spread caveat:** ON's distribution is bimodal — 17/21 reps cluster
at 60.9–79.6 ms (non-spike median 63.96 ms, i.e. an even larger -15.4% vs
OFF), but 4/21 reps (≈19%) spike to 127–139 ms. OFF's spread is tight
(72.0–82.2 ms, no spikes) by comparison. I did not track down the spike's
exact cause (candidates: this shared/noisy container's own scheduler jitter,
or mimalloc's background purge/reclaim thread — mimalloc runs a low-priority
thread that periodically returns freed pages to the OS, which could
occasionally coincide with and stall a `resolve_layered` call); either way,
it is real variance, not cherry-picked away, and it is reported rather than
hidden. Even the pessimistic reading (median across ALL 21 reps, spikes
included) still shows mimalloc winning by 13%.

**B2 (differential wall), quick check — same direction:** the
`run_differential_rust.sh` timing line from each of the ON and OFF term-lane
gate runs above (2600 cases, single process, includes process startup +
2600×`resolve` calls):
- ON: `wam_rust 8.915s`
- OFF: `wam_rust 10.769s`
- **-1.854s, -17.2%.**

Consistent in direction and rough magnitude with the B3 median (-13.2%),
from an entirely independent workload shape (many small resolves vs one
5000-package resolve) — this is not a single noisy data point.

## Verdict

D109 predicted the biggest remaining lever was the allocator itself (37.1%
of B3 Ir was glibc malloc internals) and recommended mimalloc first as the
lowest-risk, highest-leverage next step. The wall-clock A/B confirms a real,
reproducible win in both B3 (median -13.2%, non-spike median -15.4%) and B2
(-17.2%) — smaller than the raw 37% Ir share would suggest (expected: not
every allocator instruction converts to time 1:1, and the callgrind Ir count
under D109 was measured under valgrind's own allocator substitution and was
never going to be directly comparable to a wall-clock delta anyway), but
unambiguously in the predicted direction and large enough to matter at this
scale. Build/link risk was zero in practice: mimalloc's vendored C source
built cleanly via `cc` with no additional toolchain setup beyond what was
already installed, and crates.io was directly reachable from this container
(`index.crates.io` is on this environment's `no_proxy` allowlist).

Output is byte-identical ON vs OFF on every gate (as guaranteed by
construction — a global allocator cannot change term computation), so this
lever carries none of the correctness risk of a site-specific optimization.
`resolver.pl`/`resolver_store.pl` are unmodified.

## Addendum (D110 follow-up): mimalloc is opt-in in the generator, not global default (offline-CI fix)

**Date:** 2026-09-18 (same day). **Why:** the original change above put
`"mimalloc"` in the `default` feature list emitted for **every** generated
`wam_rust` crate. That turned the **WAM Conformance Smoke (rust)** CI job red
(run 35356783599, PR #4277). That job generates fresh crates and builds them
with `cargo build --offline`, and a declared optional dependency is resolved
into `Cargo.lock` **even when its feature is off** — so a fresh crate on a
runner with no cached crates.io index failed at resolve time:

```
error: no matching package named `mimalloc` found
location searched: crates.io index
required by package `uw_rust_ct v0.1.0`
```

Local D110 verification passed only because this box's cargo cache already
held `mimalloc-0.1.52`; the CI runner's offline cache did not. The
conformance harness documents the broken invariant explicitly
(`tests/test_wam_cross_target_conformance.pl`: "the generated crate has no
external deps, so `--offline` needs no network").

**Fix** (mirrors the `store_lmdb` / lmdb-zero pattern exactly): mimalloc is
now **OFF by default in the generator**, gated on a `mimalloc(true)` build
option. BOTH the optional-dependency line AND the feature's `dep:mimalloc`
activation are gated — when off, the dependency line is omitted and the
feature is emitted **empty** (`mimalloc = []`, kept declared so
`#[cfg(feature="mimalloc")]` never trips the unexpected-cfgs lint), so cargo
never resolves mimalloc from the registry index and a plain generated crate
has zero external deps. The `pkg_resolver` term and store benches pass
`mimalloc(true)` (`examples/pkg_resolver/rust/build.pl`,
`examples/pkg_resolver/rust_store/build.pl`), so their crates keep mimalloc
default-ON — the checked-in `rust/uw_resolve_wam/Cargo.toml` is unchanged and
byte-for-byte reproduced by the `mimalloc(true)` path. All A/B numbers above
still stand (they were measured on the mimalloc-ON bench crates, which are
unaffected).

**Verified:** the exact failing conformance job
(`CONFORMANCE_TARGETS=rust CONFORMANCE_PROGRAMS=member,builtins
CONFORMANCE_SAMPLE=2 CONFORMANCE_SEED=15551`, `LANG=C.utf8`) now passes rc=0;
a mimalloc-off generated crate emits empty `[dependencies]` + `mimalloc = []`;
a mimalloc-on crate is structurally identical to the checked-in bench
`Cargo.toml`.
