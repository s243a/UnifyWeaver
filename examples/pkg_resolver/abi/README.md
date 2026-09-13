<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (@s243a) -->

# Symbol-level ABI-compatibility lane (`pkg_resolver/abi`)

This lane refines the coarse package resolver's `provides` (`libc6 (>= 2.34)`)
to **ABI compatibility at the exact versioned-symbol level**: for a binary and
a shared library it answers *compatible / incompatible / unknown* per candidate
release, and computes the `[min, max]` compatible release range that `ldd`
cannot tell you. It is a driver **above** the frozen resolver: `resolver.pl`
and `resolver_store.pl` are not edited; the Debian-version comparison is
delegated to `resolver:version_lt/2` on `deb/3` terms from
`debian/deb_parse.pl`.

The lane was redesigned after the PR #4262 review; `REVIEW_NOTES.md` maps each
review point to the code and the fixture that proves it.

## The model

Identity is the exact **(soname, symbol, version-node)** triple. Two axes are
kept strictly apart:

| Axis | Values | Ordering | Used for |
|------|--------|----------|----------|
| ELF version node | `GLIBC_2.34`, `LIBSELINUX_1.0`, `COMMON_1`, `PUBLIC`, `Base` | **none** — opaque labels matched by string equality | *Does release R export `Sym@Node`?* A requirement `foo@LIB_1` is satisfied only by a provider row `foo@LIB_1` on the same soname. `foo@LIB_2` is not a match (the loader agrees: `undefined symbol: foo, version LIB_1`). `Base` is dpkg's spelling of "unversioned". |
| Debian package version | `2.34`, `3.1~`, `1:2.3-1`, `2.35-0ubuntu3.15` | `resolver:version_lt/2` over `deb(Epoch, Upstream, Revision)` (epoch, `~`, revision, any number of components) | the `.symbols` minimum-version field (a curated **lower bound**, not an introduction date) and the **release candidates**. |

Evidence is explicit and tri-state:

- `prov_evidence(So, symbols|elf, R0, complete)` — the soname's export set is
  completely known at evidence release `R0`.
- `req_evidence(Bin, readelf, complete | missing_file | readelf_failed | inconsistent, Detail)`.
- Provider bounds: `since(Min)` from `.symbols` (exported at every release
  `>= Min`, exact up to `R0`, *extrapolated* beyond it); `at(R0)` from readelf
  (exact at `R0`, extrapolated beyond, **unknown** before).

Verdict for `(Bin, So, Rel)`:

```
compatible(exact | extrapolated)          presence — defeasible "structurally possible"
incompatible([missing(Sym@Node) | below_floor(Sym@Node, Min) | soname_mismatch(...)])
                                          HARD veto — only reachable with complete, attributed evidence
unknown([no_requires_evidence | requires_evidence(Status,_) | no_provider_evidence(So)
         | unknown(Sym@Node, evidence_release(R0)) | ...])
not_needed(So)
```

Documented defeasible assumption: within a soname, exports do not disappear
(removing one is an ABI break that requires a soname bump), so presence at
`R0` extrapolates to later releases and absence from a complete export set is
a veto for the soname. `drop(Sym, Node, At)` models a violation of that
assumption to exercise the upper bound.

## Files

| File | Role |
|------|------|
| `ingest_symbols.mjs` | Ingest: `.symbols` (since bounds), `readelf` provides (`at` bounds), `readelf` requires (attributed via the ELF version **index**), NEEDED, evidence, release axis. Loud failures (exit 3), never an empty success. |
| `abi_resolve.pl` | Resolver: store loading, the two axes, per-requirement status, verdicts, floor, range over the real release axis. |
| `abi_cli.pl` | Driver: `verdict` / `status` / `floor` / `axis` / `range`. |
| `crosscheck.mjs` | readelf-vs-`.symbols` cross-check on exact identity (+ the corrected legacy per-name figure). |
| `test_abi.pl` | Assertions: real data (A), version axes (B), model fixtures (C), gcc-built ELF fixtures + `.symbols` template fixtures (D). |
| `run_abi_verify.sh` | End-to-end: build the real store, cross-check, build/ingest fixtures, run the tests. |
| `fixtures/` | C sources + version scripts for the ELF fixtures; `.symbols` fixtures (simple cases, template rejects, arch selectors). |

Generated stores land in `./.out` (gitignored).

## Store shape (P/2 JSONL, `[key, value]`)

```
symprov.jsonl  ["<soname>|<sym>@<node>", ["since", "<debver>"]]      # .symbols tier
               ["<soname>|<sym>@<node>", ["at", "<release-id>"]]     # readelf tier
symreq.jsonl   ["<binary>|<sym>@<node>", ["<soname>", "GLOBAL"|"WEAK"]]
               ["<binary>|<sym>",        ["", "GLOBAL"|"WEAK"]]      # unversioned reference
needed.jsonl   ["<binary>", "<soname>"]
evidence.jsonl ["provides|<soname>", ["symbols"|"elf", "<release-id>", "complete", "<source>"]]
               ["requires|<binary>", ["readelf", "complete"|"missing_file"|"readelf_failed"|"inconsistent", "<detail>"]]
releases.jsonl ["<soname>", "<debver>"]                               # the candidate axis (actual releases)
```

## The three provider tiers (cheapest first)

1. **`Packages` `Depends:`** — the coarse floor (`libc6 (>= 2.34)`), consumed
   by the coarse resolver. This lane's `abi_floor/3` recomputes it from the
   symbols and, on this machine, matches coreutils' declared
   `libc6 (>= 2.34), libselinux1 (>= 3.1~)` exactly.
2. **`.symbols` control member** (`ingest_symbols.mjs symbols-file`) — every
   exported `sym@node` with a curated minimum package version. Already on disk
   under `/var/lib/dpkg/info/*.symbols`; a few KB per package, no binary
   download. Binary-form files are fully supported; source-template constructs
   are processed where their semantics are known (`(optional)`, `(arch=...)`
   with `--arch`) and **rejected loudly** otherwise (`(symver)`, `(regex)`,
   quoted C++ patterns, `#include`).
3. **`readelf`** (`ingest_symbols.mjs elf` / `requires`) — any ELF. Provides
   are exact for the file's own release; requires are joined to their soname
   through `.gnu.version` -> `.gnu.version_r` by index, so two libraries using
   the same node name never collide.

## Verify on this machine

```
./run_abi_verify.sh
```

Real-data results (Ubuntu 22.04.5, libc6 2.35-0ubuntu3.15, libselinux1 3.3-1build2, `/bin/ls`):

- Ingest: libc6 `.symbols` -> **4827** `symprov` rows over 20 sonames
  (**3006** for `libc.so.6`); libselinux1 -> 238; `/bin/ls` -> **112 versioned
  + 3 unversioned (weak)** requirements, `NEEDED = [libselinux.so.1, libc.so.6]`.
- Cross-check `readelf(libc.so.6)` vs `.symbols`: **3006/3006 exact
  `sym@node` identities agree (100%)**. The corrected legacy per-name
  comparison (earliest row on *both* sides) is **2478/2478 (100%)**; the
  earlier 91.7% was an aggregation bug (last curated row vs earliest ELF row),
  not a glibc 2.34 merge effect.
- `floor /bin/ls libc.so.6` = **2.34**; `floor /bin/ls libselinux.so.1` =
  **3.1~** — equal to coreutils' declared Pre-Depends.
- Release axis (from `apt-cache madison` + dpkg): `[2.35-0ubuntu3,
  2.35-0ubuntu3.15]`; `range /bin/ls libc.so.6` = **[2.35-0ubuntu3,
  2.35-0ubuntu3.15]**, both ends `compatible(exact)`.
- Hypothetical older release `2.31-0ubuntu9.9` ->
  `incompatible([below_floor(__libc_start_main@GLIBC_2.34, 2.34), below_floor(lstat@GLIBC_2.33, 2.33)])`;
  an axis extended below the floor yields min `2.34-0ubuntu3`, never the
  lowest release.
- Hypothetical removal of `getenv@GLIBC_2.2.5` at `2.35-0ubuntu3.15` caps the
  range at `[2.35-0ubuntu3, 2.35-0ubuntu3]`; removing
  `__libc_start_main@GLIBC_2.34` at the oldest release -> `no_candidate`.
- Offering `libc.so.7` -> `incompatible([soname_mismatch(...)])`; a binary with
  no requirement evidence -> `unknown([no_requires_evidence(...)])`.
- ELF fixtures (gcc): `foo@LIB_1` required vs `foo@LIB_2` exported under the
  same soname -> `incompatible([missing(foo@LIB_1)])`, and the loader agrees
  (`undefined symbol: foo, version LIB_1`); `COMMON_1` in two libraries
  attributed correctly; `pub_fn@PUBLIC` + unversioned `plain_fn` preserved;
  missing ELF -> `unknown([requires_evidence(missing_file, _)])`.

`test_abi.pl`: 63 checks, all passing.
