<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (@s243a) -->

# Review notes — redesign of the symbol-level ABI lane (PR #4262 follow-up)

Astra's REQUEST-CHANGES review found the first implementation too lossy: it
collapsed `sym@node` to bare names with a numeric "intro", dropped non-numeric
and unversioned obligations, attributed requirements by version-*name*, mixed
the Debian package-version axis with the ELF node axis, computed an
unsatisfiable lower bound, mis-aggregated the cross-check (91.7%), and
silently mis-ingested `.symbols` templates. The model and ingest were
redesigned; the direction (three evidence tiers, a driver above the frozen
resolver) is unchanged. Sol's re-review of that redesign (REQUEST-CHANGES,
eight points) is addressed in the second section below. Frozen `resolver.pl`
/ `resolver_store.pl` / `debian/` are untouched (`git diff origin/main --
examples/pkg_resolver/{resolver.pl,resolver_store.pl,debian}` is empty).

Run `./run_abi_verify.sh` (needs node, readelf, swipl, gcc). Expected tail:
`== 92 passed, 0 failed, 0 skipped ==`. Check names in `test_abi.pl` carry the
review point they prove: `(#n)` for Astra's points, `(sol-...)` for Sol's.
The shell script's own assertions (exit codes, files not written, expected
cross-check failures) abort the run with `FAIL:` before the Prolog section.

## Sol re-review — fixes

Each row: Sol's point -> what changed -> the fixture that fails if the fix is
reverted. Sol judged the extrapolated/monotone-export disclosure adequate; it
is unchanged.

| Sol point | Fix (location) | Proving fixture(s) |
|-----------|----------------|--------------------|
| **P1a** `->` committed to the first `symprov/4`; `since(2.0)` + `at(1.0)` vetoed release 1.0 as `below_floor` | `abi_resolve.pl`: every bound now carries its evidence release (`since(Min, MinAtom, R0, Bind)`, `at(R0, Bind)`) and is usable only through its own complete `prov_evidence` row (`bound_evidence/4`). `ident_status/5` collects what EVERY evidence row says about the identity (`ev_says/5`) and combines them: evidence AT the release decides (readelf before `.symbols`); otherwise the nearest evidence BELOW (presence extrapolates upward) and ABOVE (absence propagates downward under monotone exports; a curated floor covers `Rel >= Min`) are combined in `combine/4`. A release satisfied by any row is never vetoed by another row's floor. Ingest (`ingest_symbols.mjs`) writes the evidence release into every `since` row. | **C7** (`since(2.0)` asserted first, `at(1.0)` second; release 1.0 -> `compatible(exact)`), C7b (1.5 extrapolated, 2.5/3.0 curated, 4.0 extrapolated), C7c (0.5 -> unknown, not a veto), C7d (range min is the observed release), C7e (a bound without its evidence row does not count), C7f (absence propagates down, not up), C7g (present-then-absent -> `unknown(dropped_between)`), **A25** (real `/bin/ls`: ELF evidence at 2.31-0ubuntu9.9 turns the below_floor veto into `compatible(exact)`), D7 (rows tied to the evidence release). |
| **P1b** `@` (non-default) vs `@@` (default) discarded; a non-default-only export satisfied an unversioned reference | Ingest `elfProvides()` records a binding per row from the `.gnu.version` hidden bit, cross-checked against the `@`/`@@` spelling (`inconsistent` -> exit 3): `default` = non-hidden **or verdef index 2** (the loader binds legacy unversioned references to the oldest node even when hidden — verified with the loader, see D10c), `nondefault` = hidden at index >= 3. `.symbols` rows are `unproven` unless cross-checked with `--elf` (then the ELF's binding is copied); `Base` always binds. Store: `["at", rel, binding]`, `["since", minver, rel, binding]`. Resolver `unversioned_status/7`: satisfied only by a `Base` or `default` export (`unversioned_in/5`); an `unproven` row yields `unknown(Sym, default_binding_unproven(So, Node))`; a `nondefault`-only export yields the hard `missing(Sym, no_default_export(So, Node))`. | **D10** (gcc `libhid.so.1` exporting `hid_fn` only at hidden `@HID_1`, verdef index 3: unversioned require -> `incompatible([missing(hid_fn, no_default_export(...))])`; `run_abi_verify.sh` runs the loader: `undefined symbol: hid_fn`), D10b (`@Base` -> compatible), **D10c** (hidden at index 2 -> recorded `default`, compatible; loader binds), **C8** (model: nondefault-only -> hard veto), C8b (adding a default node -> compatible), **C8c** (`.symbols`-only -> `unknown(default_binding_unproven)`), C8d (`Base` row -> compatible(curated)), C8e/C8f (versioned requirements unaffected), A7b/A7c (real libc6: every `libc.so.6` row ELF-proven; `memcpy@GLIBC_2.2.5` index 2 -> default, 115 hidden rows -> nondefault), D4b. |
| **P1c** `(optional)` template rows emitted as complete exports (D8 asserted the unsafe behaviour) | Ingest: an `(optional)` row is not an export fact by itself. With `--elf <lib>` the file is cross-checked against the ELF: an optional row the ELF does not export is **dropped** (stderr notice), a non-optional row the ELF does not export or an ELF export absent from the file **rejects** the file; without `--elf` any `(optional)` row rejects the file (exit 3). Binary-form `.symbols` under `/var/lib/dpkg/info` carry no tags, so the real path is unaffected; `run_abi_verify.sh` now ingests libc6's `.symbols` with `--elf libc.so.6` (0 dropped, 0 disagreements). | **D8 inverted** (`tmpl_optional.symbols` + `--elf libtmpl.so.1` which lacks `maybe_fn`: only `plain_fn` stored, evidence complete, `\+ symprov(maybe_fn)`), **D8b** (no `--elf` -> file rejected, no store), D8d (non-optional row absent from the ELF -> rejected), `run_abi_verify.sh` asserts the exit-3 and the dropped-row notice. `tmpl_arch.symbols` no longer carries the optional row (D8c keeps the arch checks: 3 rows). |
| **P2a** batch mode skipped a block with an unknown evidence release but counted the file as success and exited 0 | `cmdSymbolsFile()` collects every problem of every block and returns `null` (nothing added to the sink) if any exists — single-file and batch alike; `symbols-dir` counts it as rejected and exits 3. | **D11** + `run_abi_verify.sh` (`fixtures/batch/mixed.symbols`: block 1 resolvable via installed `libc6`, block 2 `#PACKAGE#`: exit 3, stderr `files=0 rejected=1 symprov=0`, `symprov.jsonl` and `evidence.jsonl` empty — no `good_fn` row). |
| **P2b** only a fixed tag list rejected; unknown tags silently stripped; `--release` unvalidated | `SUPPORTED_TAGS` whitelist (`optional`, `arch`, `arch-bits`, `arch-endian`, `ignore-blacklist`); every other tag rejects the file. `validRelease()` is the single Debian-version gate: `--release` is validated up front for every command, dpkg-guessed releases and the `releases` axis pass through the same regex. | **D9 tmpl_unknown_tag** (`(frobnicate)` -> exit 3, no store), **D9 bad_release** (`--release definitely-not-a-debian-version` -> exit 3, no store; also asserted for the `elf` path), `run_abi_verify.sh` `expect_reject` for both. |
| **P2c** `crosscheck.mjs` passed `/dev/null` vs `/dev/null` (`NaN < 100` is false) | Both symbol sets and the per-name denominator must be nonzero; `fail()` exits 1 otherwise; the pass conditions are `=== 100`, never `< 100`. | `run_abi_verify.sh` (sol-P2c): `/dev/null` vs `/dev/null` and `sym` vs `/dev/null` must fail with `FAIL: empty symbol set`. |
| **P2d** `soname_mismatch` from a stem heuristic (`libfoo.so.2` vs unrelated NEEDED `libfoo.so.1-extra`) | `so_stem/2` removed. `soname_offer/3` reports `mismatch(N)` only under a declared `replaces(Offered, N)` with `needed(Bin, N)`; anything else not NEEDED is `not_needed`. New store file `replaces.jsonl` (`ingest_symbols.mjs replaces <new> <old>...`); `run_abi_verify.sh` declares `libc.so.7 replaces libc.so.6` for the real store. | **C9** (offer `libfoo.so.2`, NEEDED `libfoo.so.1-extra` -> `not_needed`), C9b (still not_needed with `libfoo.so.1` NEEDED but no relation), C9c (declared relation -> `soname_mismatch`), **A19b** (real: `libselinux.so.10` -> `not_needed`), A19 (real: `libc.so.7` mismatch only because declared). |
| **P3** local dotted-version comparator over node names in the legacy per-name figure | `nodeNum()`/`cmpDotted()`/`earliest()` removed. The per-name figure is now "per-name node-SET agreement" (sets compared, nothing parsed or ordered). Real data: 3006/3006 exact, 2763/2763 per-name. | `fixtures/crosscheck/` (sol-P3): a two-node symbol, a `GLIBC_PRIVATE` node, a `Base` node, another soname to be ignored -> 5/5 and 4/4; `elf_dropnode` must fail on both checks; asserted in `run_abi_verify.sh`. |
| **Residual** `.symbols`-derived verdicts labelled `compatible(exact)` | New basis `curated`: presence resting on `.symbols` metadata (at or below its evidence release, `Min =< Rel`) is `provided(_, curated)`; `exact` is reserved for readelf at exactly the release. The verdict basis is the weakest among the requirements (`exact < curated < extrapolated`). | A12 (real: `compatible(curated)` at the `.symbols` evidence release), C1b, C7b, C8d; `exact` still on ELF fixtures (D2, D3b, D4b, D10b/c, C7). |

Changes that fell out of the P1a restructuring (documented, not requested):

- **Absence propagates down, not up.** Before, a missing provider row was a
  veto for *every* release of the soname; now absence from complete evidence
  at `R1` is `missing` for `Rel =< R1` (reported as `missing(Sym@Node)` at
  `R1` itself and `missing(Sym@Node, observed_absent(Src, R1))` when
  inferred) and `unknown(absent_at(Src, R1))` for `Rel > R1` — later releases
  add symbols. C7f, A17.
- **Observed drop.** Present at `R0`, absent at a later `R1`: a release in
  between is `unknown(dropped_between(R0, Src, R1))`, neither extrapolated
  nor vetoed. C7g.
- `abi_floor/3` takes the highest minimum over all `since` rows of an
  identity (several `.symbols` evidence releases may coexist).

## Sol's checklist (first review)

### (a) Exact version-node identity end-to-end — Astra #1

- **Ingest** (`ingest_symbols.mjs`): `.symbols` rows are split at the last
  `@` into `(sym, node)` and stored as `"<so>|<sym>@<node>"` with the minimum
  version *verbatim* (`["since", "<debver>", "<evidence-release>", <binding>]`).
  readelf provides carry the node from `.gnu.version_d` by index
  (`["at", "<release>", <binding>]`). Nothing is parsed out of a node name.
- **Store** (`abi_resolve.pl`): `symprov(So, Sym, Node, Bound)`,
  `symreq(Bin, Sym, Node, So, Bind)`. Loading rejects an empty node.
- **Match** (`versioned_status/7` -> `ident_status/5`): same soname, same
  symbol, same node, by unification. No fallback to `symprov(So, Sym, _, _)`
  for versioned requirements.
- **Fixtures**: C1/C1b, **D1** (gcc-built `libfoo.so.1` v1/v2; the loader
  must print `undefined symbol: foo, version LIB_1`), A17/A18 on real libc,
  A7, B8.

### (b) verneed attribution via version INDEX — Astra #3

- `elfTables()` parses `readelf -W -V`: `.gnu.version` (per-dynsym index,
  hidden bit `h`), `.gnu.version_d` (index -> name) and `.gnu.version_r`
  (index -> `{file, name, weak}`). `elfRequires()` joins each UND symbol by
  its **index** to `(file, node)`; the `name@VER` text is only a consistency
  assertion (mismatch -> `inconsistent` evidence, exit 3).
- **Fixture D3**: `libalpha.so.1` / `libbeta.so.1` both define `COMMON_1`;
  attribution asserted both ways. A4/A5 on `/bin/ls`.

### (c) Two axes separate; deb parsing reused — Astra #4

- Package versions are parsed once at load by `debian/deb_parse:parse_deb_version/2`
  and ordered by the frozen `resolver:version_lt/2` on `deb/3`. No local
  version arithmetic exists in the lane (the last one, in `crosscheck.mjs`,
  went with Sol P3).
- **Fixtures**: B1–B8, D7, **A8–A10** (computed floors `2.34` / `3.1~` equal
  coreutils' declared dependency read from dpkg).

### (d) Incomplete evidence never becomes a false veto or false compat — Astra #2

- Every successful ingest emits an `evidence` row; a missing/unreadable ELF
  makes `requires` exit 3 **and** record an incomplete evidence row.
- `abi_verdict/5` short-circuits to `unknown(...)` without complete
  requirement evidence or complete provider evidence; `at(R0)` evidence
  yields `unknown(Sym@Node, evidence_release(R0))` for older releases; an
  unversioned obligation is `unknown` while any NEEDED object lacks evidence
  or while its only providers are `unproven`.
- **Fixtures**: C3/C3b, C4–C4e, C5/C5b, C6, C8c, **D4–D6**, A1–A3, A6,
  A20–A22.

### (e) `[min, max]` satisfiable at both ends — Astra #5

- `abi_range/5` evaluates **every** release of the actual axis and takes
  min/max from the compatible ones (`range_min_max/3`).
- **Fixtures**: **C2** (`range(5.0, 5.0)`, not `range(1,5)`), C2c/C2d, C3b,
  C7d, A13, **A15**, A23/A24, D2.

### (f) Cross-check aggregation fixed; false 2.34-merge claim removed — Astra #6

- `crosscheck.mjs`: exact `sym@node` sets 3006/3006 and per-name node sets
  2763/2763 on Ubuntu 22.04 libc6; `fixtures/crosscheck/` pins the two-node
  case that produced the old 91.7%. Docs no longer claim a glibc 2.34 merge
  divergence.

### (g) Frozen files untouched

`git diff origin/main -- examples/pkg_resolver/resolver.pl
examples/pkg_resolver/resolver_store.pl examples/pkg_resolver/debian/` is
empty. The lane only *imports* `resolver:version_lt/2` and
`deb_parse:parse_deb_version/2`.

### Astra #7 — `.symbols` templates handled or rejected loudly

- Whitelisted tags are processed (`(arch=…)`, `(arch-bits=…)`,
  `(arch-endian=…)` with `--arch`; `(ignore-blacklist)`; `(optional)` only
  with `--elf`), everything else rejects the whole file with line numbers;
  nothing partial is written, also in batch mode. `#PACKAGE#` headers make
  `--release` mandatory.
- **Fixtures**: D7, D8/D8b/D8c/D8d, D9 x5, D11.

## Things worth a second look

- `default` for a hidden export at verdef index 2 encodes glibc's legacy
  rule for unversioned references (`dl-lookup.c`: index < 3 is accepted
  before the hidden test; D10c and `run_abi_verify.sh` verify it with the
  loader). `dlsym()` lookups (`DL_LOOKUP_RETURN_NEWEST`) do not use that rule;
  the lane models link-time/`DT_NEEDED` references only.
- With `--elf`, the ELF is used to prove bindings and to cross-check the
  `.symbols` rows, but no `at()` rows are emitted from it, so the verdict
  basis stays `curated`; ingest the ELF with `elf` (as `run_abi_verify.sh`
  does for the fixture stores) to get `exact`.
- `unknown(dropped_between(R0, Src, R1))` and `unknown(absent_at(Src, R1))`
  are new reasons callers may want to distinguish from the evidence-gap
  reasons.
- `abi_floor/3` fails (rather than guessing) when a requirement has no
  `since()` row — readelf-only provider evidence has no package floor.
