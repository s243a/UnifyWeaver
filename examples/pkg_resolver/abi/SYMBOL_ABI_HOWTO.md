<!-- How symbol-level ABI resolution works — with worked examples on real libs -->

# Symbol-level ABI resolution — how it works, with examples

This extends the package resolver from coarse version constraints
(`libfoo (>= 2.0)`) to **fine-grained ABI compatibility** at the *symbol*
level, so we can compute the real *[min, max]* compatible library-version range
for a binary — the thing `ldd` won't tell you.

## The model (one generalization of "provides")

- A **library version PROVIDES** a set of exported versioned symbols
  `{sym@ver}` plus a `soname` (its ABI generation, e.g. `libc.so.6`).
- A **binary/package REQUIRES** a set of referenced versioned symbols `{sym@ver}`
  plus the `sonames` it links (`NEEDED`).
- **Compatible(bin, libver)** ⟺ `soname matches` AND `requires ⊆ provides`.

Everything below is just this predicate plus how the versions are stored.

## The tools (PoC)

- `elf_symbols.mjs extract|rows|compat <file...>` — read provides/requires/soname
  from any ELF via `readelf`; `compat` checks `requires ⊆ provides`.
- `sym_abi.mjs intervals|newest <file...>` — treat each symbol's version tag as
  the version it was *introduced* in, giving each symbol a validity interval
  `[intro, inf)` within the soname; `newest` computes the min/max compatible
  version.

## Worked examples (real output, this machine)

### 1. A library's provides, a binary's requires
```
$ elf_symbols.mjs extract libc.so.6   -> n_provides: 2970, soname: libc.so.6
$ elf_symbols.mjs extract /bin/ls     -> n_requires: 112,  needed: [libselinux.so.1, libc.so.6],
                                         min_versions: { GLIBC: 2.34, LIBSELINUX: 1.0 }
```
The **minimum** version is *derived*, not declared: it's the highest version
tag among the required symbols (`/bin/ls` references a `@GLIBC_2.34` symbol, so
it needs glibc ≥ 2.34). `ldd` never shows this; `readelf` does.

### 2. Compatibility = set containment
```
$ elf_symbols.mjs compat /bin/ls libc.so.6 libselinux.so.1
   binary_requires: 112, libs_provide: 3207, compatible: true, missing_count: 0
```
All 112 needs are covered → compatible. If any were missing → `missing > 0` →
a **hard incompatibility** (proof of "no").

### 3. Symbols as validity intervals (the real ABI-growth curve)
```
$ sym_abi.mjs intervals libc.so.6
   2443 symbols across 35 versions [2.2.5 .. 2.35]
   introduced per version:  ... 2.33: 12   2.34: 212   2.35: 4
   row example: ["pthread_setname_np","2.12","inf"]
```
Each symbol is stored ONCE as `[sym, intro, inf]` — a validity interval within
the soname. (glibc 2.34 adding 212 symbols is the real pthread/rt-into-libc
merge.) "Does version V provide sym?" = `intro(sym) ≤ V`. This is the same
interval store we built for snapshot membership, one level down — and it's the
memory-pressure dataset (thousands of symbols × versions × snapshots).

### 4. Newest-compatible, no ABI break
```
$ sym_abi.mjs newest /bin/ls libc.so.6
   min_compatible_version: 2.34
   newest_compatible_version: 2.35        (max unbounded within soname libc.so.6)
```
Newest version whose provides still cover ls's needs. With a backward-compatible
library there's no upper bound inside the soname — the default max is "the
soname generation."

### 5. Where a real MAX comes from — a symbol removal
```
$ sym_abi.mjs newest /bin/ls libc.so.6 --drop=getenv@2.30   # simulate ABI break
   min = 2.34,  binary needs getenv: true,  effective MAX = 2.29
   => min (2.34) > max (2.29)  => NO compatible version  (correct no_candidate)
```
If a newer version *removes* a symbol the binary needs (or bumps the soname),
that caps the max. Here ls needs both 2.34-era symbols *and* `getenv`, so
removing `getenv` at 2.30 makes the range empty — the resolver correctly returns
no candidate.

## Min vs max — the epistemics (defeasible defaults + hard vetoes)

- **min** = the verneed floor (highest required symver). Hard, derived, exact.
- **max** = a *defeasible default* = "the soname generation" (no upper bound for
  backward-compatible libs), *refined downward* only by hard evidence: a needed
  symbol removed, or a soname bump. Symbol analysis can only ever NARROW the
  default.
- **Symbol ABSENCE / soname mismatch = a hard veto** (proof of "no") — this can
  override an optimistic or wrong declared dependency.
- **Symbol PRESENCE = a defeasible "might work"** — necessary but not sufficient
  (semantics unverified). It never outranks a declared/tested dependency; it only
  widens the candidate set with a low-confidence maybe.

Confidence order: `tested > declared repo dep > ELF-hard-veto / ELF-maybe >
soname default`.

## "Is the symbol info in the repo, so we don't download the whole package?"

Largely yes — in three tiers, cheapest first:

1. **The min-version dependency is already in the repo index.** Debian's
   `Packages` file (fetched by `apt update`) carries each package's `Depends:`,
   e.g. `libc6 (>= 2.34)` — and that bound was computed *from the symbols* by
   `dpkg-shlibdeps` at build time. So the coarse min needs **zero** package
   download.
2. **The full per-symbol interval data is curated metadata, not the binary.**
   Debian library packages ship a `.symbols` file whose lines are literally
   `symbol@Base minimum-version` — i.e. *exactly our `[sym, intro]` interval
   store, precomputed by the maintainer*. It lives in the package's small
   **control member** (`control.tar.*`), so you fetch that member (a few KB),
   not the multi-MB data. (A mirror could also publish a symbols index directly.)
3. **Only if neither is published** do you fetch the binary (or HTTP
   range-fetch just its `.dynsym`/`.gnu.version_*` sections) and run `readelf` —
   which is what this PoC does — then cache the result.

Either way it's a **one-time ingest** per `(pkg,ver)`, deduped across snapshots
by the interval encoding — never a per-resolution download. And on a real
machine the "locked set" libraries are already installed, so their provides are
free; only the *candidates* you're weighing need any fetch, lazily, newest-first.

## How it lands in the store / resolver (no new engine)

- Ingest → `symprov(soname|sym -> intro#inf)` and `symreq(bin|sym -> ver)` rows,
  interval-encoded and deduped exactly like the package pool + membership.
- `Compatible` is the existing `provides ⊇ requires` containment; the
  already-built `newest_compatible_snap` generalizes from coarse version
  constraints to symbol-set containment — "newest ABI-compatible library version
  across snapshots" becomes a real query on real data.
- Confidence tag per edge (`declared | elf_hard | elf_maybe | soname_default |
  tested`) so a package manager prunes on hard + prefers declared, and a coding
  agent can turn an `elf_maybe` into `tested` by actually building — the Level-3
  oracle a package manager lacks.
