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

## Sol re-review pass 2 — fixes

Sol's re-review CLOSED five of the eight original findings (P1b, P1c, P2a, P2c,
P2d) and returned REQUEST-CHANGES on four more, all rooted in one principle: a
`.symbols` file is a curated LOWER-BOUND list, not a complete export set, so its
ABSENCE proves nothing. Each fix is proven by a fixture whose check name carries
its point and that fails if the fix is reverted. `run_abi_verify.sh`:
`== 98 passed, 0 failed, 0 skipped ==`.

| Sol finding | Fix (file) | Proving fixture(s) |
|---|---|---|
| P1 `abi_resolve.pl:289` curated absence = false hard veto | Ingest tags a plain `.symbols` (no `--elf`) as `curated`; only an `--elf` cross-check is `complete` (`ingest_symbols.mjs` `cmdSymbolsFile`). The resolver's `prov_usable/4` + `ev_says/5` let curated evidence establish PRESENCE but never ABSENCE (a curated omission yields no `Says` row → `unknown(absent_from_incomplete_evidence)`, never `missing`); `complete` absence still vetoes. **Unversioned path** (`unversioned_status/7`): the `missing` veto is gated on `\+ prov_evidence(S,_,_,complete)` for the NEEDED object, so a curated-only NEEDED library also yields `unknown` — a Fable re-verify caught an earlier over-widening to `prov_usable` here | **C10/C10b/C10c** (versioned: curated omission → unknown; present-in-curated → `provided(curated)`; the SAME store tagged `complete` DOES veto), **C12/C12b** (unversioned: curated-only omission → unknown; complete omission → missing), **A26** (real libselinux1, ingested without `--elf`, is curated: an absent symbol → unknown), **D7** (simple.symbols without `--elf` → `curated`) |
| P1 `abi_resolve.pl:311` contradictory curated floor overrides presence | A curated minimum ABOVE its evidence release is contradictory: rejected at ingest (`debLe`, exit 3) and at load (`assert_symprov` `rel_le(Deb,R0)` → the store fails to load). The existing evidence-tied aggregation (C7) already lets direct presence beat a floor | **C11/C11b** (`assert_symprov` rejects min 2.0 > release 1.0; accepts 1.0 ⩽ 1.0), **contradictory.symbols** (ingest exit 3, nothing written) |
| P2 `ingest_symbols.mjs:114` `DEB_VERSION_RE` not a real validator | `validDebVersion()` parses `[epoch:]upstream[-revision]` the way `dpkg --validate-version` does; `1:`, `1-`, `1::2` are rejected; one gate for every version/`--release` | **run_abi_verify** `bad_release_{1:,1-,1::2}` (exit 3), D9 `bad_release` |
| P3 `crosscheck.mjs` fixture cannot detect comparator restoration | Added `numnode@LIBX_2.1` / `numnode@LIBX_2.10` (opaque labels that a numeric parser would collapse) to the crosscheck pair, plus a negative `elf_numcollapse` fixture that FAILS under the correct opaque comparison but would PASS only if a numeric node comparator were reintroduced (2.10 == 2.1) | **run_abi_verify** `(sol2-P3)` numeric-collapse pair must FAIL; equal pair now 7/7 identity, 5/5 per-name |

### Deliberate model points (documented, not requested)

- After the P1(289) fix, `unknown(absent_at)` / `unknown(dropped_between)` treat
  only DIRECT (`complete`) absence as a real absent endpoint; a curated omission
  is `unknown`, never an absent observation.
- The verdict basis is the WEAKEST contributing basis; `compatible(curated)` (any
  `.symbols`-only requirement) is kept distinct from `compatible(exact)` (readelf
  at that release), and curated evidence can never be laundered into `exact`.
- `below_floor` remains a hard veto only as a DECLARED minimum (dpkg-shlibdeps'
  floor), never from mere absence, and never overriding a direct presence
  observation.
- Pre-existing, noted for awareness (unchanged this pass): if a COMPLETE ELF row
  observes a symbol ABSENT at R and a curated row claims `since` ≤ R, `combine/4`
  extrapolates the curated presence over the contradicting complete absence
  between the two evidence releases. It needs genuinely contradictory tiers and
  is not a contract violation (`Min` is documented as defeasible), but a future
  pass could treat a complete absence as authoritative over a curated floor.
- Robustness: `debLe` `die()`s with a clear message if `dpkg` is not on PATH
  (rather than silently rejecting every row); `load_abi_store/1` clears the
  partial store if a row throws; the now-unused `bound_evidence/4` was removed.

## Astra review — fixes

Astra (the original reviewer) re-reviewed and confirmed original findings
#1/#3/#5/#6 CLOSED but returned REQUEST-CHANGES with new issues, the most
important found by ingesting live `libstdc++.so.6`. All real findings are fixed
and fixture-proven; `run_abi_verify.sh`: `== 108 passed, 0 failed, 0 skipped ==`.

| Astra finding | Fix | Proving fixture(s) |
|---|---|---|
| P1 `ingest:378` STB_GNU_UNIQUE exports dropped (elfProvides kept only GLOBAL/WEAK) → false `missing` veto; live libstdc++ dropped 106 | `elfProvides` now keeps `UNIQUE` too | **run_abi_verify (astra)** ingests libstdc++.so.6 and asserts a real UNIQUE export (`_ZNSt10moneypunctIcLb0EE4intlE@GLIBCXX_3.4`) is stored |
| P1 `abi_resolve:461` unversioned historical-completeness false veto (complete evidence only BELOW Rel vetoed) | `unversioned_status/7` now vetoes `missing` only when absence is ESTABLISHED at Rel (`absence_established/2`: complete evidence at a release ≥ Rel); otherwise `unknown(absence_unestablished(...))` | **C14/C14b** (complete only at 1.0, query 2.0 → unknown; query 1.0 → missing) |
| P1 `abi_resolve:474` default binding detached from evidence (any/orphaned row's binding used) → false `compatible(exact)` for unversioned | `binding_at/6` ties the default-version binding to the evidence APPLICABLE at Rel; `unversioned_in`/`unproven`/`nondefault` use it | **C15/C15b/C15c** (default@1.0 + nondefault@2.0: query 2.0 → no_default_export; query 1.0 → compatible; orphaned default ignored) |
| P2 `ingest:201` arch selectors ≠ dpkg-architecture (`linux-any`/`any-arm` mis-matched) | `archSelects` supports only exact names (± `!`); wildcard patterns are REJECTED (exit 3), not mis-selected | **tmpl_arch_wild.symbols** rejected; **D9** asserts no store written |
| P2 `ingest:128` `validDebVersion` epoch unbounded (`2147483648:1` accepted) | epoch capped at 2147483647 (dpkg's limit) | **run_abi_verify** `bad_release_form` includes `2147483648:1` (exit 3) |
| P2 `abi_resolve:448` hypothetical unversioned drop ignored node + release | `unversioned_in` checks `\+ hyp_dropped(Hyp, Sym, Node, Rel)` per candidate export (node + release aware), like the versioned path | **C13/C13b** (drop of a different node/later release leaves the export; dropping the exact node at the release removes it) |

### Deliberate dispositions (Astra P3, not changed)

- `crosscheck.mjs` per-name mutation sensitivity: the negative `elf_numcollapse`
  fixture guards the EXACT-IDENTITY path against a restored numeric node
  comparator. The per-name figure is mathematically redundant with exact identity
  when the `sym@node` key sets match (identical keys ⇒ identical per-name node
  sets), so a per-name-only numeric mutation cannot be isolated by a fixture with
  matching identity; the identity guard is the one the model relies on.
- `debLe` uses `dpkg --compare-versions` (the reference implementation, as the
  ingest already does for `dpkg-query`/`dpkg -S`). The FROZEN-predicate boundary
  governs the RESOLVER, which orders exclusively via `resolver:version_lt/2`; the
  authoritative contradictory-floor rejection is the Prolog `assert_symprov`
  `rel_le/2` check (frozen path), with the JS `debLe` only a loud early guard.

### Fable re-verify of the Astra fixes — two regressions caught and fixed

A Fable re-verification of the Astra-fix commit caught two regressions the
`binding_at` rework introduced (the harness missed them because every unversioned
`.symbols` test queries AT the evidence release):

- **R1** (false `missing` veto): `binding_at` took the binding only from the row
  AT Rel or BELOW, but `ident_status` can credit a curated `.symbols` floor row
  ABOVE Rel (`combine/4`, `Min =< Rel`). Fixed: `binding_at` now also selects that
  above curated row (mirroring `combine`). Fixture **C16/C16b** (unversioned ref
  BELOW a curated floor, query 1.5/1.0 → `compatible(curated)`, not `missing`).
- **R2** (cross-axis false unknown): the missing-veto coverage gate compared So's
  query release with a sibling's evidence release. Fixed: the gate applies to the
  queried `So` only; siblings are evaluated at their own release (a sibling
  lacking complete evidence is already handled earlier). Fixture **C17**.

Also cleaned a stray NUL byte in the `debLe` cache-key string (now ` ` as
source text). `run_abi_verify.sh`: `== 111 passed, 0 failed, 0 skipped ==`.

## Astra re-review 2 — unversioned-path unification + arch hardening

Astra re-reviewed again and found more unversioned-path issues, all rooted in
`binding_at` being a SEPARATE re-derivation from `ident_status` and the veto
branches not being uniformly release-gated. Rather than patch case-by-case, the
binding was UNIFIED into the aggregation. `run_abi_verify.sh`:
`== 118 passed, 0 failed, 0 skipped ==`.

- **Unification (fixes P1 522):** `ident_status` now returns `provided(Basis,
  Binding)`; `says_status`/`combine` carry the binding from the SAME row that
  establishes presence, so binding and presence can never diverge. When a
  present-below row and a covering curated-above row disagree on the binding it is
  `ambiguous`. `binding_at` is deleted; `unversioned_in`/`unproven`/`nondefault`
  and a new `unversioned_ambiguous` read the binding straight from `ident_status`.
  An `ambiguous` binding → `unknown(default_binding_conflict(...))`, never a veto
  or a confident compatible. Fixture **C19**.
- **P1 470 (no_default_export release-gated):** the coverage gate
  (`\+ absence_established(So, Rel)` → unknown) now precedes BOTH veto branches
  (`no_default_export` and `missing`), so a nondefault export seen only in
  complete evidence BELOW the query release yields `unknown`, not a veto (a
  default export could be added later). Fixtures **C18/C18b**.
- **P2 arch (208):** `archSelects` validates EVERY term first (an early match or
  negation can no longer skip a later unsupported term), and rejects any term
  containing `any`, a tuple/GNU form (`gnu-any-amd64`), a comma-list
  (`amd64,arm64`), or anything not a clean exact arch name. Fixtures
  **tmpl_arch_tuple / tmpl_arch_list** (D9).
- **P2 arch attributes (285):** `arch-bits`/`arch-endian` use explicit
  `ARCH_BITS`/`ARCH_ENDIAN` tables (verified vs dpkg-architecture, incl.
  kfreebsd-amd64=64, mips64=big); a non-tabulated `--arch` REJECTS the row rather
  than guessing. Fixture **arch_bits_unknown**.
- **P2 (test isolation):** the epoch cap is now proven through the `releases`
  command (**epoch_releases**), which has no `debLe` floor check to mask it; the
  hypothetical-drop node- and release-matching are isolated by **C13c** (same
  release, different node) and **C13d** (same node, future release).
