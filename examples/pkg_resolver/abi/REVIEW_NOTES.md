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
resolver) is unchanged. Frozen `resolver.pl` / `resolver_store.pl` /
`debian/` are untouched (`git diff origin/main -- examples/pkg_resolver/{resolver.pl,resolver_store.pl,debian}` is empty).

Run `./run_abi_verify.sh` (needs node, readelf, swipl, gcc). Expected tail:
`== 63 passed, 0 failed, 0 skipped ==`. Check names in `test_abi.pl` carry the
review point they prove, e.g. `(#3)`.

## Sol's checklist

### (a) Exact version-node identity end-to-end — Astra #1

- **Ingest** (`ingest_symbols.mjs`): `.symbols` rows are split at the last
  `@` into `(sym, node)` and stored as `"<so>|<sym>@<node>"` with the minimum
  version *verbatim* (`["since", "<debver>"]`). readelf provides carry the
  node from `.gnu.version_d` by index (`["at", "<release>"]`). There is no
  `verNum()` anymore; nothing is parsed out of a node name.
- **Store** (`abi_resolve.pl`): `symprov(So, Sym, Node, Bound)`,
  `symreq(Bin, Sym, Node, So, Bind)`. Loading rejects an empty node.
- **Match** (`versioned_status/7`): `symprov(So, Sym, Node, _)` — same
  soname, same symbol, same node, by unification. No fallback to `symprov(So,
  Sym, _, _)` for versioned requirements.
- **Fixtures**: C1/C1b (in-memory: `foo@LIB_1` vs `foo@LIB_2` ->
  `incompatible([missing(foo@LIB_1)])`, then exact row -> `compatible(exact)`);
  **D1** (gcc-built `libfoo.so.1` v1/v2, real ingest; the loader is run as
  ground truth and must print `undefined symbol: foo, version LIB_1`);
  A17/A18 on real libc (`getenv@GLIBC_2.99` -> missing;
  `pthread_setname_np@GLIBC_2.12` (real non-default node) -> provided,
  `@GLIBC_2.13` -> missing); A7 (no node-less provider rows); B8 (node
  labels are never ordered numerically).

### (b) verneed attribution via version INDEX — Astra #3

- `elfTables()` parses `readelf -W -V`: the `.gnu.version` array (per-dynsym
  index, hidden bit `h`), `.gnu.version_d` (index -> name) and
  `.gnu.version_r` (index -> `{file, name, weak}`). `elfRequires()` joins each
  UND symbol by its **index** to `(file, node)`; the `name@VER` text is only
  used as a consistency assertion (mismatch -> `inconsistent` evidence, exit 3).
- Also enforced: the verneed file must be in DT_NEEDED; a symbol whose index
  has no verneed entry is an ingest error, not a `?` soname.
- **Fixture D3**: `libalpha.so.1` and `libbeta.so.1` both define `COMMON_1`;
  `usecommon` needs one symbol from each. Asserted: `alpha_fn@COMMON_1 ->
  libalpha.so.1`, `beta_fn@COMMON_1 -> libbeta.so.1`, and the negatives. D3b:
  both verdicts `compatible(exact)`. A4/A5 on `/bin/ls`: `__libc_start_main@GLIBC_2.34
  -> libc.so.6`, `freecon@LIBSELINUX_1.0 -> libselinux.so.1`.

### (c) Two axes separate; deb parsing reused — Astra #4

- Package versions are parsed once at load by `debian/deb_parse:parse_deb_version/2`
  (`rel_term/2`, `assert_symprov/2`) and ordered by the frozen
  `resolver:version_lt/2` on `deb/3` (`rel_lt/2`, `rel_le/2`). No local
  version arithmetic exists in the lane. `since(Deb, Atom)` keeps the original
  atom so `3.1~` is reported as `3.1~`, not re-formatted.
- ELF nodes never enter `rel_term/2` as versions: a node string that is not a
  Debian version becomes `label(Atom)` and only matches itself (B7/B8).
- **Fixtures**: B1–B6 (`3.1~ < 3.1`, `3.1~ < 3.1~rc1`, `1:2.3-1 > 3.1`,
  `2.35 < 2.35-0ubuntu3 < 2.35-0ubuntu3.15`, `2.2.5 < 2.2.5.1 < 2.2.6`, `0`
  lowest); D7 (`.symbols` fixture with `1:0.9-2` and `1.2~rc1` minimums loaded
  as `deb(1, ...)` / preserved atoms); **A8–A10** on real data: the computed
  floors `2.34` and `3.1~` equal coreutils' declared
  `libc6 (>= 2.34), libselinux1 (>= 3.1~)` read from dpkg.
- Documented (`README.md`, `SYMBOL_ABI_HOWTO.md`, header comment of
  `abi_resolve.pl`): `.symbols` minimum versions are curated lower bounds,
  raisable by policy, not introduction dates; a release below one is reported
  as `below_floor`, distinct from `missing`.

### (d) Incomplete evidence never becomes a false veto or false compat — Astra #2

- `evidence.jsonl` rows are emitted by every successful ingest; a missing or
  unreadable ELF makes `requires` exit 3 **and** record
  `["requires|<bin>", ["readelf", "missing_file", ...]]` (no empty success);
  `elf` on a missing file exits 3 with nothing written.
- `abi_verdict/5` short-circuits to `unknown(...)` when: no `req_evidence`,
  `req_evidence` not `complete`, or no complete `prov_evidence` for the
  soname; `at(R0)` evidence yields `unknown(Sym@Node, evidence_release(R0))`
  for older releases; an unversioned obligation is `unknown` while any NEEDED
  object lacks evidence. `needed/2` is checked (`not_needed`, and
  `soname_mismatch` is a hard veto when the offered soname differs from the
  NEEDED one with the same stem).
- Non-numeric nodes and unversioned references are kept (`pub_fn@PUBLIC`,
  `plain_fn`), weak references are flagged `WEAK` and classified
  `weak_unresolved` (never a veto).
- **Fixtures**: C3/C3b (at-evidence: older -> unknown), C4–C4e (no provider
  evidence -> unknown; `missing_file` -> unknown; unversioned with one lib
  unevidenced -> unknown, with all evidenced and nobody exporting ->
  `missing`, exported by another NEEDED lib -> compatible), C5/C5b, C6;
  **D4–D6** through the real pipeline (`usepub` / `libpub` / `libplain`;
  `store_pub_nolibpub`, `store_pub_noplain`, `store_missing`); A1–A3, A6,
  A19–A22 on real data.

### (e) `[min, max]` satisfiable at both ends — Astra #5

- The candidate axis is `releases.jsonl` (actual releases; on this machine
  from `apt-cache madison libc6` + dpkg), never symbol-intro points.
  `abi_range/5` evaluates **every** release and takes min/max from the
  releases whose verdict is `compatible(_)` (`range_min_max/3`), so both ends
  are compatible by construction; releases that add no symbols are still
  evaluated (C2d).
- **Fixtures**: **C2** — releases 1..5, `a since 1`, `b since 5` ->
  `range(5.0, 5.0)`, verdict at 1.0 `incompatible([below_floor(b@L, 5.0)])`
  (the old code returned `range(1,5)`); C2c (6.0 -> `compatible(extrapolated)`,
  `range(5.0, 6.0)`); C3b (unknown releases are reported, not counted); A13
  (real axis, both ends compatible); **A15** (axis extended below the floor:
  min is `2.34-0ubuntu3`, not `2.31-0ubuntu9.9`); A23/A24 (hypothetical
  removal caps the max / squeezes to `no_candidate`); D2 (`range(1.0-1, 1.0-1)`).

### (f) Cross-check aggregation fixed; false 2.34-merge claim removed — Astra #6

- `crosscheck.mjs` compares the two tiers on **exact `sym@node` sets**:
  `.symbols=3006 readelf=3006 shared=3006`, 100%. It also reproduces the
  legacy per-name figure with earliest-row aggregation on *both* sides:
  2478/2478 (100%) (the review's 2443/2443 used the old script's symbol
  filter; either way it is 100%). Both must be 100% or the script fails.
- `README.md` and `SYMBOL_ABI_HOWTO.md` no longer claim a glibc 2.34
  pthread/rt-merge divergence; they state the 91.7% was the last-row-vs-
  earliest-row bug.

### (g) Frozen files untouched

`git diff origin/main -- examples/pkg_resolver/resolver.pl
examples/pkg_resolver/resolver_store.pl examples/pkg_resolver/debian/` is
empty. The lane only *imports* `resolver:version_lt/2` and
`deb_parse:parse_deb_version/2`.

### Astra #7 — `.symbols` templates handled or rejected loudly

- `parseSymbolsFile()` processes `(optional)` (kept), `(ignore-blacklist)`
  (kept), `(arch=…)`, `(arch-bits=…)`, `(arch-endian=…)` (row kept iff it
  selects `--arch`; **rejected** without `--arch`), and rejects `(symver)`,
  `(regex)`, `(c++…)`, quoted pattern rows, `#include`, malformed rows and
  non-Debian minimum versions — the whole file is refused (exit 3) with line
  numbers; nothing partial is written. `#PACKAGE#` headers are accepted but
  then `--release` is mandatory (the evidence release cannot be looked up).
- The simple cases keep working: multiple sonames, `|` alt-dep lines, `*`
  meta fields, `#` comments, `#MINVER#`, private minimum `0`, dep-id column.
- **Fixtures**: D7 (`fixtures/simple.symbols`), D8
  (`tmpl_arch.symbols` with `--arch amd64`: `!amd64` row dropped, `(optional)`
  kept), D9 x3 (`tmpl_symver`, `tmpl_cxx`, `tmpl_arch` without `--arch` ->
  no store written; `run_abi_verify.sh` also asserts the non-zero exit).

## Things worth a second look

- `unversioned_status/7` treats an unversioned reference as satisfied by *any*
  exported node of the symbol in *any* NEEDED object (the loader's default-
  version rule is not modelled). Documented in the HOWTO "Limits".
- `bound_holds(since(Min,_), R0, Rel, extrapolated)` for `Rel > R0` and
  `at(R0)` for `Rel > R0` both rely on the in-soname monotone-export
  assumption; the `Basis` value makes that visible to callers.
- `abi_floor/3` fails (rather than guessing) when a requirement has no
  `since()` row — readelf-only provider evidence has no package floor.
