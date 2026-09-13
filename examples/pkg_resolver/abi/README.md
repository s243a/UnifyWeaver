<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (@s243a) -->

# Symbol-level ABI-compatibility lane (`pkg_resolver/abi`)

This lane promotes the coarse package resolver's `provides` (a package provides
a virtual name / satisfies `libfoo (>= 2.0)`) to **fine-grained ABI
compatibility at the *symbol* level**. It computes, for a binary, the **MIN**
and **MAX (newest)** compatible library version via symbol-set containment —
the real `[min, max]` range that `ldd` will not tell you.

It reuses the SAME interval-store idea one level down: instead of a package
having a version tenure, each exported **symbol** has a validity interval
`[intro, inf)` *within a soname*. A library version `V` provides a symbol iff
`intro(sym) =< V` (until a soname bump / removal). The frozen `resolver.pl` /
`resolver_store.pl` are **not** edited — version comparison is delegated to
`resolver:version_lt/2`.

## The model (one generalization of `provides`)

- A **library version PROVIDES** a set of exported versioned symbols plus a
  `soname` (its ABI generation, e.g. `libc.so.6`).
- A **binary REQUIRES** a set of referenced versioned symbols plus the sonames
  it links (`DT_NEEDED`).
- `compatible(Bin, LibVer)` ⟺ `soname matches` AND `requires ⊆ provides`.

## Files

| File | Role |
|------|------|
| `ingest_symbols.mjs` | Ingest symbol metadata → P/2 JSONL interval store (3 tiers). |
| `abi_resolve.pl` | Prolog resolver: intervals, containment, min/max. New module; reuses `resolver:version_lt/2`. |
| `abi_cli.pl` | Callable driver: `min` / `newest` / `range` / `compat`. |
| `test_abi.pl` | Real-data assertions (the PoC's numbers). |
| `run_abi_verify.sh` | End-to-end: build the store from this machine's data + verify. |

Generated stores land in `./.out` (gitignored) — rebuild with
`./run_abi_verify.sh`.

## The three ingest tiers (cheapest first)

The full per-symbol interval data is *curated metadata, not the binary* — so the
common path needs **zero package download**.

1. **`Packages` `Depends:` — coarse min for free.** Debian/Ubuntu's package
   index already carries `libc6 (>= 2.34)`, computed *from the symbols* by
   `dpkg-shlibdeps` at build time. The coarse floor needs no download at all.
   (Consumed by the coarse resolver; this lane refines it.)
2. **`.symbols` control member — curated intervals (`ingest_symbols.mjs
   symbols-file`).** Library packages ship a `.symbols` file whose lines are
   literally `symbol@Base minimum-version` — *exactly our `[sym, intro]`
   interval store, precomputed by the maintainer*. It lives in the package's
   small `control.tar.*` member (a few KB), not the multi-MB data archive. On an
   installed system it is already on disk under `/var/lib/dpkg/info/*.symbols`.
3. **`readelf` fallback — any ELF (`ingest_symbols.mjs elf` / `requires`).** If
   no `.symbols` is published, `readelf -W --dyn-syms` gives a `.so`'s exported
   versioned symbols (`intro` = the symbol's own version tag), and a binary's
   `readelf -V` verneed table attributes each *required* symbol to its soname.
   Applies to **any ELF**, so the lane works for Debian, Ubuntu, and arbitrary
   binaries with no repo metadata at all.

Applies to **Debian AND Ubuntu** unchanged (same `dpkg` / `.symbols` format),
plus the `readelf` path for any ELF on any distro.

### Store shape (P/2 JSONL, `[key, value]`, `load_p2_jsonl`-compatible)

```
symprov.jsonl : ["<soname>|<sym>", "<intro>#inf"]   # interval; to=inf within soname
symreq.jsonl  : ["<binary>|<sym>", "<soname>#<ver>"] # verneed floor, per soname
needed.jsonl  : ["<binary>", "<soname>"]             # DT_NEEDED
```

## The resolver (`abi_resolve.pl`)

- `provides_at(SoName, Sym, V)` — true iff `intro(SoName|Sym) =< V` (interval
  lookup; `to=inf` within a soname).
- `abi_min(Binary, SoName, Min)` — the **verneed floor**: the highest required
  version among the binary's required symbols of that soname. Hard, derived,
  exact.
- `abi_compatible(Binary, SoName, V[, Drop])` — containment: every required
  symbol of that soname is provided at `V`. `Drop = drop(Sym, At)` simulates a
  soname-era ABI break (symbol removed at `At`) for testing the max.
- `newest_abi_compatible(Binary, SoName, Candidates[, Drop], Result)` — newest
  `V` in `Candidates` (descending) with `abi_compatible` → `compatible(V)`, else
  `no_candidate`. `Candidates` = `soname_candidates/2` (the distinct intro
  versions in the store — the soname's version axis).
- `abi_range(Binary, SoName, Candidates[, Drop], Result)` — combines the derived
  MIN with the newest-compatible MAX → `range(Min, Max)`, or `no_candidate(...)`
  when `Min > Max` (a removed symbol squeezed the range empty).

Note the division of labor (matching the PoC): `abi_compatible` is *name-intro
containment* and drives the **MAX** search; the **MIN** is the separate verneed
floor from `abi_min`. So `abi_compatible(2.33)` can be true by name even though
the binary references the `GLIBC_2.34` node — `abi_range` applies the floor and
reports `2.34` as the effective lower bound.

## Min vs max — the epistemics

- **min** = the verneed floor (highest required symver). **Hard, derived,
  exact.**
- **max** = a **defeasible default** = "the soname generation" (no upper bound
  for backward-compatible libs), refined *downward* only by hard evidence: a
  needed symbol removed, or a soname bump. Symbol analysis can only ever NARROW
  the default.
- **Symbol ABSENCE / soname mismatch = a hard veto** (proof of "no") — it can
  override an optimistic or wrong declared dependency.
- **Symbol PRESENCE = a defeasible "might work"** — necessary but not sufficient
  (semantics unverified); it never outranks a declared/tested dep, it only
  widens the candidate set with a low-confidence maybe.

Confidence order: `tested > declared repo dep > ELF-hard-veto / ELF-maybe >
soname default`. See `SYMBOL_ABI_HOWTO.md` for the full design and worked
examples.

## Verify on this machine

```
./run_abi_verify.sh          # builds the store from /var/lib/dpkg + /bin/ls, runs all checks
```

Real-data results (Ubuntu 22.04, libc6, `/bin/ls`):

- `libc6` `.symbols` → **4827** symprov interval rows across 20 sonames
  (**3006** for `libc.so.6`); `/bin/ls` → **112** symreq rows + `NEEDED =
  [libselinux.so.1, libc.so.6]`.
- Tier-2 cross-check: `readelf(libc.so.6)` vs curated `.symbols` agree on the
  intro axis for **2241 / 2443 (91.7%)** shared symbols. Every disagreement is
  the **glibc 2.34 pthread/rt-into-libc merge** — `.symbols` conservatively
  dates those symbols to 2.34 (when they entered `libc.so.6`'s stable ABI) while
  the raw ELF keeps the historical libpthread version node.
- `abi_min(/bin/ls, libc.so.6)` = **2.34** (set by
  `__libc_start_main@GLIBC_2.34`).
- `abi_compatible(/bin/ls, libc.so.6, 2.35)` = **TRUE** (0 missing);
  `newest_abi_compatible` = `compatible(2.35)`; `abi_range` = **range(2.34,
  2.35)**.
- Simulated removal of `getenv` at 2.30 (a symbol `/bin/ls` needs): the newest
  version still exporting `getenv` caps at **2.29**; combined with min **2.34**
  → **min > max squeeze** → **`no_candidate`**.
