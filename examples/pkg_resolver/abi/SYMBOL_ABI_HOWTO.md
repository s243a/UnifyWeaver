<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (@s243a) -->

# Symbol-level ABI resolution — how it works, with real examples

This extends the package resolver from coarse version constraints
(`libfoo (>= 2.0)`) to ABI compatibility at the *exact versioned symbol* level,
so we can compute the real `[min, max]` compatible release range for a binary.

## The model

- A **library release PROVIDES** a set of exported `sym@node` identities under
  a `soname` (its ABI generation, e.g. `libc.so.6`).
- A **binary REQUIRES** a set of `sym@node` identities, each attributed to the
  soname its ELF verneed table names, plus the sonames it links (`DT_NEEDED`),
  plus any unversioned references.
- **Compatible(bin, release)** ⟺ every NEEDED soname is offered under its exact
  name AND every required `sym@node` is exported by that soname at that
  release (unversioned references: exported as `Base` or as the **default**
  version — `@@`, or the oldest version node — by some NEEDED object; a
  hidden `@` export does not count, exactly as in the loader).

Two axes, never mixed:

1. **ELF version nodes** are labels. `GLIBC_2.34` is not the number 2.34 —
   it is the name of a node in `.gnu.version_d`. A requirement `foo@LIB_1` is
   satisfied by `foo@LIB_1` and by nothing else; in particular not by
   `foo@LIB_2` on the same soname (the loader fails with
   `undefined symbol: foo, version LIB_1`). Non-numeric nodes (`PUBLIC`,
   `GLIBC_PRIVATE`) and unversioned symbols (`Base`) are first-class identities.
2. **Debian package versions** order the release candidates and the `.symbols`
   minimum-version field, through the frozen resolver's `deb/3` comparison
   (`debian/deb_parse.pl` + `resolver:version_lt/2`): epochs (`1:2.3-1`),
   tildes (`3.1~ < 3.1`), revisions (`2.35 < 2.35-0ubuntu3`), any number of
   components.

## Where the evidence comes from (three tiers, cheapest first)

1. **`Packages` `Depends:`** — `libc6 (>= 2.34)` was computed by
   `dpkg-shlibdeps` from the symbols at build time. Zero download; the coarse
   resolver consumes it. `abi_floor/3` recomputes it from tier 2 and, on this
   machine, reproduces coreutils' `libc6 (>= 2.34), libselinux1 (>= 3.1~)`.
2. **`.symbols` control member** — one row per `sym@node` with a curated
   minimum package version:
   ```
   libc.so.6 libc6 #MINVER#
   | libc6 (>> 2.35), libc6 (<< 2.36)
    getenv@GLIBC_2.2.5 2.2.5
    pthread_setname_np@GLIBC_2.12 2.12
    pthread_setname_np@GLIBC_2.34 2.34
    __libc_enable_secure@GLIBC_PRIVATE 0 1
   ```
   The minimum version is a **lower bound** a dependent must declare, *not* a
   ground-truth introduction date: Debian policy lets a maintainer raise it
   after a compatible behaviour change. The lane therefore stores it as
   `since(Min, ..., R0, Bind)` tied to the evidence release `R0` the file was
   taken at, and reports a release below it as `below_floor` (the same
   conservative floor `dpkg-shlibdeps` emits), never as proof the symbol was
   absent — and never when another evidence row (e.g. readelf at that
   release) shows the symbol present. A `.symbols` file says nothing about
   `@` vs `@@`, so its rows are `unproven` for unversioned references unless
   the file is cross-checked against the library with `--elf lib.so`
   (`run_abi_verify.sh` does this for libc6: 3006 rows, 0 disagreements).
   Source-template tags are whitelisted (`(arch…)` with `--arch`,
   `(ignore-blacklist)`, `(optional)` only with `--elf`); anything else, or a
   block whose evidence release is unknown, rejects the whole file (exit 3),
   also in `symbols-dir` batch mode.
3. **`readelf`** on the ELF itself — exact for that file's release
   (`at(R0, Bind)`, with the default-version binding from `.gnu.version`),
   unknown for older releases, extrapolated for newer ones.

## Worked examples (real output, Ubuntu 22.04.5, after `./run_abi_verify.sh`)

### 1. Requirements attributed through the version index

```
$ node ingest_symbols.mjs requires /bin/ls --stdout | grep -E 'start_main|freecon|gmon'
["/bin/ls|__libc_start_main@GLIBC_2.34",["libc.so.6","GLOBAL"]]
["/bin/ls|freecon@LIBSELINUX_1.0",["libselinux.so.1","GLOBAL"]]
["/bin/ls|__gmon_start__",["","WEAK"]]
```
Each undefined symbol's `.gnu.version` index is looked up in `.gnu.version_r`,
giving *(file, node)*. Two libraries that both define a node called `COMMON_1`
therefore cannot be confused (fixture `usecommon`: `alpha_fn@COMMON_1 ->
libalpha.so.1`, `beta_fn@COMMON_1 -> libbeta.so.1`). Unversioned and weak
references are kept, flagged, and never silently dropped.

### 2. The curated floor equals the declared dependency

```
$ swipl -q -g main -t halt abi_cli.pl -- .out/store floor /bin/ls libc.so.6
floor /bin/ls libc.so.6: 2.34
$ swipl -q -g main -t halt abi_cli.pl -- .out/store floor /bin/ls libselinux.so.1
floor /bin/ls libselinux.so.1: 3.1~
$ dpkg-query -W -f '${Pre-Depends}\n' coreutils
libacl1 (>= 2.2.23), libattr1 (>= 1:2.4.44), libc6 (>= 2.34), libgmp10 (>= 2:6.2.1+dfsg), libselinux1 (>= 3.1~)
```

### 3. Verdicts on the actual release axis

```
$ ... axis libc.so.6
axis libc.so.6: [2.35-0ubuntu3,2.35-0ubuntu3.15]          # apt-cache madison + dpkg
$ ... range /bin/ls libc.so.6
range /bin/ls libc.so.6: [2.35-0ubuntu3, 2.35-0ubuntu3.15]
  2.35-0ubuntu3: compatible(curated)
  2.35-0ubuntu3.15: compatible(curated)
$ ... verdict /bin/ls libc.so.6 2.31-0ubuntu9.9
verdict /bin/ls libc.so.6 2.31-0ubuntu9.9: incompatible([below_floor('__libc_start_main'@'GLIBC_2.34','2.34'),below_floor(lstat@'GLIBC_2.33','2.33')])
```
The candidates are releases, not symbol-introduction points; `range/3` reports
a `[min, max]` whose ends both carry `compatible` verdicts. With an axis
extended below the floor (`2.31-0ubuntu9.9, 2.34-0ubuntu3, ...`) the min is
`2.34-0ubuntu3` — the first release that actually satisfies the requirements.
The basis is `curated` because the only evidence is the `.symbols` file;
`exact` is reserved for readelf observations at that very release. Bounds
from several evidence rows are aggregated, never taken first-match: if
readelf evidence for `2.31-0ubuntu9.9` were ingested and showed every
required identity, that release would be `compatible(exact)` despite the
curated floor of 2.34 (test A25 does exactly that).

### 4. Exact node identity (the case the old model got wrong)

Fixture: `libfoo.so.1` v1 exports `foo@LIB_1`; v2 keeps a `LIB_1` node but
exports `foo` only as `foo@LIB_2`. `usefoo` is linked against v1.
```
$ LD_LIBRARY_PATH=.out/fx/v2 .out/fx/usefoo
usefoo: symbol lookup error: usefoo: undefined symbol: foo, version LIB_1
$ ... .out/fx/store_foo_v2 verdict .out/fx/usefoo libfoo.so.1 2.0-1
verdict ... libfoo.so.1 2.0-1: incompatible([missing(foo@'LIB_1')])
```
A name-level model with a numeric intro would have said "compatible" here.

### 5. Where a real MAX comes from — a removal (hypothetical)

```
$ ... range /bin/ls libc.so.6 getenv GLIBC_2.2.5 2.35-0ubuntu3.15
range /bin/ls libc.so.6: [2.35-0ubuntu3, 2.35-0ubuntu3]
  2.35-0ubuntu3: compatible(exact)
  2.35-0ubuntu3.15: incompatible([missing(getenv@'GLIBC_2.2.5',hypothetical_drop)])
$ ... range /bin/ls libc.so.6 __libc_start_main GLIBC_2.34 2.35-0ubuntu3
range /bin/ls libc.so.6: no_candidate
```

### 6. Incomplete evidence is "unknown", not "compatible"

```
$ node ingest_symbols.mjs requires /nonexistent/bin --out .out/x ; echo exit=$?
ingest_symbols: requires /nonexistent/bin: missing_file; recorded INCOMPLETE evidence, no requirement rows
exit=3
$ ... verdict /nonexistent/bin libc.so.6 2.35-0ubuntu3
verdict /nonexistent/bin libc.so.6 2.35-0ubuntu3: unknown([no_requires_evidence('/nonexistent/bin')])
```
Likewise a NEEDED soname with no provider evidence yields
`unknown([no_provider_evidence(So)])`, and readelf-only evidence yields
`unknown` for releases older than the file's own.

### 7. Cross-check: the two provider tiers agree exactly

```
  exact sym@node identity: .symbols=3006 readelf=3006 shared=3006 only-readelf=0 only-.symbols=0
  identity agreement: 3006/3006 (100.0%)
  per-name node-set agreement: 2763/2763 (100.0%)
```
The earlier "91.7% agreement, explained by the glibc 2.34 pthread/rt merge"
was an aggregation bug: the last curated row of a symbol was compared with the
earliest ELF node of that symbol, so any symbol with two nodes
(`pthread_setname_np@GLIBC_2.12` + `@GLIBC_2.34`) "disagreed". Compared
consistently, everything agrees; the merge explanation was false. The
per-name figure now compares the *set* of nodes per name — nothing is parsed
out of a node name and nothing is ordered — and `fixtures/crosscheck/` pins
the two-node case (plus a `GLIBC_PRIVATE` node and a `Base` node) as a static
regression. Two empty inputs fail the check instead of passing on `NaN`.

### 8. An unversioned reference vs a non-default-only export

Fixture `libhid.so.1`: `hid_fn` is exported only as hidden `hid_fn@HID_1`
(`.symver` with a single `@`); `usehid` was linked against an unversioned
build and so references `hid_fn` without a version.
```
$ LD_LIBRARY_PATH=.out/fx/hid_idx3 .out/fx/usehid
usehid: symbol lookup error: usehid: undefined symbol: hid_fn
$ grep hid_fn .out/fx/store_hid_idx3/symprov.jsonl
["libhid.so.1|hid_fn@HID_1",["at","1.0-1","nondefault"]]
$ ... .out/fx/store_hid_idx3 status .out/fx/usehid libhid.so.1 1.0-1
  missing(hid_fn,no_default_export('libhid.so.1','HID_1'))
```
The same hidden export at verdef index 2 (the oldest node, `hid_idx2`) is
bound by the loader for legacy unversioned references, runs, and is recorded
`default`. A `.symbols`-only store cannot tell `@` from `@@`, so there the
answer is `unknown([unknown(hid_fn, default_binding_unproven(...))])`, never
`compatible`.

## Epistemics

- **Absence / soname mismatch = hard veto** (`incompatible`) — valid only
  when requirement evidence is complete and provider evidence for that soname
  is complete and attributed by index. Otherwise the answer is `unknown`.
- **Presence = defeasible "structurally possible"** (`compatible(exact |
  curated | extrapolated)`) — necessary, not sufficient; semantics are
  unverified. It never outranks a declared or tested dependency; it widens
  the candidate set with a low-confidence maybe.
- **`curated`** means the presence rests on `.symbols` metadata (a curated
  export list), not on a direct readelf observation at that release; it is
  reported distinctly from `exact`.
- **`below_floor`** is a curated floor, not proof of absence, and is reported
  distinctly so a caller can choose to treat it as declared-dependency
  strength rather than ELF-hard strength. It never overrides another evidence
  row that shows the identity present at the release.
- **`extrapolated`** relies on the in-soname monotone-export assumption and is
  reported distinctly from `exact`. Absence propagates the other way: absent
  from complete evidence at `R1` means absent at every `Rel <= R1`
  (`missing(..., observed_absent(Src, R1))`), but says nothing about later
  releases (`unknown(..., absent_at(Src, R1))`). Present at `R0` and absent
  at a later `R1` is an observed drop: the releases in between are
  `unknown(..., dropped_between(R0, Src, R1))`.

Confidence order: `tested > declared repo dep > ELF hard veto / ELF maybe >
soname default`.

## Limits

- The unversioned-reference rule follows `ld.so` for `DT_NEEDED`-driven
  binding: `Base`, the `@@` default, or the oldest version node (verdef
  index 2, accepted by glibc for legacy binaries even when hidden). `dlsym()`
  lookups use a different rule and are not modelled.
- `.symbols` source-template constructs that need the binary to expand
  (`(symver)`, `(regex)`, quoted C++ patterns) and any unknown tag are
  rejected, not interpreted; `(optional)` rows need `--elf`.
- `soname_mismatch` needs a declared `replaces(New, Old)` row; the lane does
  not guess that `libfoo.so.2` succeeds `libfoo.so.1` from the name.
- The release axis is an input (from `apt-cache madison`, dpkg, a snapshot
  store); the lane never invents candidates from symbol data.
