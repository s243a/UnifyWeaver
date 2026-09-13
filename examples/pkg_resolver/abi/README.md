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

The lane was redesigned after the PR #4262 review and revised after Sol's
re-review; `REVIEW_NOTES.md` maps each review point to the code and the
fixture that proves it.

## The model

Identity is the exact **(soname, symbol, version-node)** triple. Two axes are
kept strictly apart:

| Axis | Values | Ordering | Used for |
|------|--------|----------|----------|
| ELF version node | `GLIBC_2.34`, `LIBSELINUX_1.0`, `COMMON_1`, `PUBLIC`, `Base` | **none** — opaque labels matched by string equality | *Does release R export `Sym@Node`?* A requirement `foo@LIB_1` is satisfied only by a provider row `foo@LIB_1` on the same soname. `foo@LIB_2` is not a match (the loader agrees: `undefined symbol: foo, version LIB_1`). `Base` is dpkg's spelling of "unversioned". An **unversioned requirement** binds, as the loader does, only to a `Base` export or a **default** (`@@`) export — never to a hidden (`@`) one. |
| Debian package version | `2.34`, `3.1~`, `1:2.3-1`, `2.35-0ubuntu3.15` | `resolver:version_lt/2` over `deb(Epoch, Upstream, Revision)` (epoch, `~`, revision, any number of components) | the `.symbols` minimum-version field (a curated **lower bound**, not an introduction date) and the **release candidates**. |

Evidence is explicit, and every provider bound is tied to the evidence row it
came from:

- `prov_evidence(So, symbols|elf, R0, complete)` — the soname's export set is
  completely known at evidence release `R0`.
- `req_evidence(Bin, readelf, complete | missing_file | readelf_failed | inconsistent, Detail)`.
- Provider bounds: `since(Min, MinAtom, R0, Bind)` from the `.symbols` evidence
  taken at `R0` (exported at `R0` and, by the curated bound, at every release
  `>= Min`); `at(R0, Bind)` from readelf at `R0`. `Bind` is the
  default-version binding: `default` (an unversioned reference binds to it:
  `@@`, `Base`, or the oldest version node which the loader also accepts),
  `nondefault` (hidden `@`), or `unproven` (a `.symbols` row not cross-checked
  against the ELF with `--elf`).

Per identity and release, **all** evidence rows are aggregated: evidence
taken *at* the release decides (readelf before `.symbols`); otherwise the
nearest evidence below (presence extrapolates upward) and the nearest
evidence above (absence propagates downward; a curated floor covers
`Rel >= Min`) are combined. A release satisfied by any evidence row is never
vetoed by another row's floor.

Verdict for `(Bin, So, Rel)`:

```
compatible(exact | curated | extrapolated) presence — defeasible "structurally possible"
    exact        every requirement observed by readelf at exactly Rel
    curated      some requirement rests on .symbols metadata only
    extrapolated some requirement rests on the monotone-export assumption
incompatible([missing(Sym@Node) | missing(Sym@Node, observed_absent(Src, R1))
              | missing(Sym, no_default_export(So, Node)) | below_floor(Sym@Node, Min)
              | soname_mismatch(offered(So), needed(N))])
                                          HARD veto — only reachable with complete, attributed evidence
unknown([no_requires_evidence | requires_evidence(Status,_) | no_provider_evidence(So)
         | unknown(Sym@Node, evidence_release(R0) | absent_at(Src, R1) | dropped_between(R0, Src, R1))
         | unknown(Sym, default_binding_unproven(So, Node)) | ...])
not_needed(So)                            not in DT_NEEDED and not a declared replacement
```

Documented defeasible assumption: within a soname, exports do not disappear
(removing one is an ABI break that requires a soname bump), so presence at
`R0` extrapolates to later releases and absence from a complete export set at
`R1` is a veto for every release `<= R1` (later releases may add symbols:
unknown). `drop(Sym, Node, At)` models a violation of that assumption to
exercise the upper bound; an *observed* violation (present at `R0`, absent
at a later `R1`) makes the releases in between `unknown(dropped_between(...))`.
`soname_mismatch` needs a declared `replaces(New, Old)` relation
(`replaces.jsonl`); there is no name-stem heuristic.

## Files

| File | Role |
|------|------|
| `ingest_symbols.mjs` | Ingest: `.symbols` (since bounds, optionally cross-checked against the ELF with `--elf`), `readelf` provides (`at` bounds with default-version binding), `readelf` requires (attributed via the ELF version **index**), NEEDED, evidence, release axis, declared soname succession (`replaces`). Loud, atomic failures (exit 3), never an empty or partial success. |
| `abi_resolve.pl` | Resolver: store loading, the two axes, per-identity evidence aggregation, per-requirement status, verdicts, floor, range over the real release axis. |
| `abi_cli.pl` | Driver: `verdict` / `status` / `floor` / `axis` / `range`. |
| `crosscheck.mjs` | readelf-vs-`.symbols` cross-check on exact identity + per-name node sets (no node-name parsing or ordering); fails on empty input. |
| `test_abi.pl` | Assertions: real data (A), version axes (B), model fixtures (C), gcc-built ELF fixtures + `.symbols` template fixtures (D). |
| `run_abi_verify.sh` | End-to-end: build the real store, cross-check (+ negatives), build/ingest fixtures with the loader as ground truth, run the tests. |
| `fixtures/` | C sources + version scripts for the ELF fixtures (incl. the hidden-version `libhid`); `.symbols` fixtures (simple cases, template rejects, arch selectors, `(optional)`, unknown tag, batch atomicity); `crosscheck/` static regression pair. |

Generated stores land in `./.out` (gitignored).

## Store shape (P/2 JSONL, `[key, value]`)

```
symprov.jsonl  ["<soname>|<sym>@<node>", ["since", "<debver>", "<evidence-release>", <binding>]]  # .symbols tier
               ["<soname>|<sym>@<node>", ["at", "<evidence-release>", <binding>]]                # readelf tier
               <binding> = "default" | "nondefault" | "unproven"
symreq.jsonl   ["<binary>|<sym>@<node>", ["<soname>", "GLOBAL"|"WEAK"]]
               ["<binary>|<sym>",        ["", "GLOBAL"|"WEAK"]]      # unversioned reference
needed.jsonl   ["<binary>", "<soname>"]
evidence.jsonl ["provides|<soname>", ["symbols"|"elf", "<release-id>", "complete", "<source>"]]
               ["requires|<binary>", ["readelf", "complete"|"missing_file"|"readelf_failed"|"inconsistent", "<detail>"]]
releases.jsonl ["<soname>", "<debver>"]                               # the candidate axis (actual releases)
replaces.jsonl ["<new-soname>", "<old-soname>"]                       # declared soname succession
```

## The three provider tiers (cheapest first)

1. **`Packages` `Depends:`** — the coarse floor (`libc6 (>= 2.34)`), consumed
   by the coarse resolver. This lane's `abi_floor/3` recomputes it from the
   symbols and, on this machine, matches coreutils' declared
   `libc6 (>= 2.34), libselinux1 (>= 3.1~)` exactly.
2. **`.symbols` control member** (`ingest_symbols.mjs symbols-file`) — every
   exported `sym@node` with a curated minimum package version. Already on disk
   under `/var/lib/dpkg/info/*.symbols`; a few KB per package, no binary
   download. Binary-form files are fully supported. Source-template tags are
   **whitelisted**: `(arch=...)` / `(arch-bits=...)` / `(arch-endian=...)`
   with `--arch`, `(ignore-blacklist)`, and `(optional)` **only with
   `--elf <lib>`** (dpkg lets an optional symbol stay in the template after
   it left the binary, so the row counts only if the ELF exports it);
   every other tag, `(symver)`, `(regex)`, quoted C++ patterns, `#include`,
   an unknown evidence release or a non-Debian `--release` **rejects the whole
   file** (exit 3, nothing partial, also in batch mode). With `--elf` the
   file is cross-checked against the ELF and each row's default-version
   binding is taken from it; without it rows are `unproven` for unversioned
   references.
3. **`readelf`** (`ingest_symbols.mjs elf` / `requires`) — any ELF. Provides
   are exact for the file's own release and carry the default-version binding
   (`.gnu.version` hidden bit, cross-checked against the `@`/`@@` spelling);
   requires are joined to their soname through `.gnu.version` ->
   `.gnu.version_r` by index, so two libraries using the same node name never
   collide.

## Verify on this machine

```
./run_abi_verify.sh
```

Real-data results (Ubuntu 22.04.5, libc6 2.35-0ubuntu3.15, libselinux1 3.3-1build2, `/bin/ls`):

- Ingest: libc6 `.symbols` cross-checked against `libc.so.6` with `--elf`
  -> **4827** `symprov` rows over 20 sonames (**3006** for `libc.so.6`, 0
  optional rows dropped, 0 disagreements, **115** hidden `@` exports recorded
  `nondefault`, `memcpy@GLIBC_2.2.5` at verdef index 2 recorded `default`);
  libselinux1 -> 238; `/bin/ls` -> **112 versioned + 3 unversioned (weak)**
  requirements, `NEEDED = [libselinux.so.1, libc.so.6]`.
- Cross-check `readelf(libc.so.6)` vs `.symbols`: **3006/3006 exact
  `sym@node` identities agree (100%)**; per-name node sets **2763/2763
  (100%)**. The earlier 91.7% was an aggregation bug (last curated row vs
  earliest ELF row), not a glibc 2.34 merge effect; `fixtures/crosscheck/`
  pins that case. Empty inputs fail the cross-check.
- `floor /bin/ls libc.so.6` = **2.34**; `floor /bin/ls libselinux.so.1` =
  **3.1~** — equal to coreutils' declared Pre-Depends.
- Release axis (from `apt-cache madison` + dpkg): `[2.35-0ubuntu3,
  2.35-0ubuntu3.15]`; `range /bin/ls libc.so.6` = **[2.35-0ubuntu3,
  2.35-0ubuntu3.15]**, both ends `compatible(curated)` (the evidence is
  `.symbols` metadata; `exact` needs readelf at that release).
- Hypothetical older release `2.31-0ubuntu9.9` ->
  `incompatible([below_floor(__libc_start_main@GLIBC_2.34, 2.34), below_floor(lstat@GLIBC_2.33, 2.33)])`;
  an axis extended below the floor yields min `2.34-0ubuntu3`, never the
  lowest release. Adding readelf evidence for that release turns the verdict
  into `compatible(exact)` — a curated floor never vetoes a release another
  evidence row satisfies.
- Hypothetical removal of `getenv@GLIBC_2.2.5` at `2.35-0ubuntu3.15` caps the
  range at `[2.35-0ubuntu3, 2.35-0ubuntu3]`; removing
  `__libc_start_main@GLIBC_2.34` at the oldest release -> `no_candidate`.
- Offering `libc.so.7` -> `incompatible([soname_mismatch(...)])` because the
  store declares `replaces(libc.so.7, libc.so.6)`; `libselinux.so.10` (same
  stem, no declaration) -> `not_needed`; a binary with no requirement
  evidence -> `unknown([no_requires_evidence(...)])`.
- ELF fixtures (gcc): `foo@LIB_1` required vs `foo@LIB_2` exported under the
  same soname -> `incompatible([missing(foo@LIB_1)])`, and the loader agrees
  (`undefined symbol: foo, version LIB_1`); `hid_fn` exported only at a
  hidden `@HID_1` (verdef index 3) does **not** satisfy an unversioned
  requirement -> `incompatible([missing(hid_fn, no_default_export(...))])`,
  and the loader agrees (`undefined symbol: hid_fn`), while the same hidden
  export at verdef index 2 binds (loader and lane agree); `COMMON_1` in two
  libraries attributed correctly; `pub_fn@PUBLIC` + unversioned `plain_fn`
  preserved; missing ELF -> `unknown([requires_evidence(missing_file, _)])`;
  `(optional)` row absent from the ELF dropped, present rows kept.

`test_abi.pl`: 92 checks, all passing; `run_abi_verify.sh` additionally
asserts the exit-3 rejections, batch atomicity and the cross-check negatives.
