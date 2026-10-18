<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Grok prompt — uw-resolve P3: Debian semantics + real-index ingestion

> Branch from the coordinator tip as `grok/pkg-resolver-debian`. This is the
> first round since P0.5 allowed to touch `resolver.pl` — ADDITIVELY, with
> every existing gate as the proof. Concurrent-lane dirs stay frozen
> (`cli/`, `cljs/`, `rust/`, `go/`, `BENCHMARKS.md`); the coordinator
> rebuilds those lanes' builds after merge, so do not rebuild them yourself.

---

## Copy from here ↓↓↓

You are contributing to **UnifyWeaver** (`github.com/s243a/unifyweaver`).
Branch from **`claude/peerhailer-exploratory-docs-aodas5`** as
`grok/pkg-resolver-debian`. SWI-Prolog (`swipl`); Node v18+.

### The mission
uw-resolve's whole point is reasoning about a frozen Debian-derived base
(TrixiePup), but it still speaks toy versions (`v(M,I,P)`) and toy
dependencies. P3 makes it speak Debian: real version semantics, virtual
packages, alternatives — and ingest a REAL slice of a Debian Packages index
into the D48/D56 store pipeline. Read first:
`docs/proposals/PACKAGE_MANAGER_LOGIC_PROPOSAL.md` §§2d–2h,
`examples/pkg_resolver/README.md`, and the D48 JSONL schema section.

### 1. Debian version semantics
- New version term alongside `v/3` (which stays fully supported —
  back-compat is a GATE): `deb(Epoch, Upstream, Revision)` where Upstream
  and Revision are PRE-SEGMENTED at parse time into alternating
  non-digit/digit runs (so the resolver's comparison is pure list walking —
  no string parsing in the hot path; the parser lives at the ingestion/API
  edge). Comparison per Debian Policy §5.6.12: epoch first (numeric,
  default 0); then upstream, then revision, each segment-wise — non-digit
  runs compare with `~` sorting BEFORE EVERYTHING including the empty
  string, letters before non-letters; digit runs compare numerically;
  a missing part compares as empty (which `~` still precedes).
- Constraint forms: keep `any/eq/gte/lt/range` working over deb versions;
  add the Debian relation spellings at the ingestion edge (`>=`, `<=`,
  `>>`, `<<`, `=` map onto the existing forms — `>>`/`<<` are strict; say
  in the README how each maps).
- **Oracle discipline**: SWI is the spec. Additionally, when
  `dpkg --compare-versions` exists in your environment, cross-check:
  ≥2,000 seeded random version pairs PLUS a curated table of the classics
  (`1.0~rc1 < 1.0`, `1.0~~ < 1.0~`, epoch beats everything,
  `1.0-1 < 1.0-2`, letter/digit alternation, `+dfsg` suffixes). Gate the
  dpkg arm on availability like D43's lmdb arm; the curated table itself is
  an UNGATED plunit test either way. State whether the dpkg arm ran.

### 2. Provides / virtual packages + alternatives
- Catalog extension (additive): `provides(Name, Ver, VirtualName)` rows —
  optionally versioned (`provides(..., VirtualName, VirtualVer)`) if the
  slice you ingest needs it; say which you shipped. A dependency on X is
  satisfied by a real package X or by any selected package providing X.
  A virtual name with no real package is never itself selected — a
  provider is. Determinism policy documented: real package preferred over
  providers; providers in a documented stable order.
- Alternatives (`a | b | c` in Depends): an ordered candidate group —
  first satisfiable alternative under the current partial selection, with
  backtracking into later alternatives when a choice dead-ends downstream
  (this is genuine new backtracking surface — expect it to stress the
  targets exactly like D44 did). Layered resolution: an alternative
  already satisfied by a loaded layer wins without re-selection.
- `explain_blocked` stays honest for both: a blocked virtual dependency
  names the providers and their ceilings; a blocked alternatives group
  reports the per-alternative reason list.

### 3. Real-index ingestion
- A parser for the Debian `Packages` control-stanza format (Prolog or
  .mjs — your choice, stated) covering: Package, Version, Depends,
  Pre-Depends (treat as Depends, note it), Provides, Conflicts, Breaks
  (treat as Conflicts, note it), Essential (map to a `base`-candidate
  marker the README explains). Unknown fields skipped loudly-once.
- Emit the D48 JSONL dump schema, extended: `deb` version encoding,
  `provides` rows, alternative groups in `depends` rows (document the
  extended schema in the README next to the existing one).
- **Commit a real sample slice**: 300–1,000 real stanzas from a Debian
  (or Devuan/Ubuntu) Packages file (it is metadata; keep it small), as
  `examples/pkg_resolver/debian/sample_packages` + its generated JSONL +
  stores. If you have network access, state the exact source URL +
  snapshot date; if not, a slice assembled from real stanzas you have is
  acceptable — provenance stated either way.
- **Demonstration queries on the real slice** (paste output): resolve a
  real package with a non-trivial closure; a `resolve_layered` against a
  frozen base drawn from the slice (pick Essential+libc-ish holds) with
  one deliberately-too-old hold so `explain_blocked` names a REAL ceiling
  with REAL version strings; `freeze_audit` on that base.

### 4. Corpus + differentials (the gates)
- Every existing scenario UNCHANGED and green (38 plunit / 39 dump — the
  v/3 world must not shift by a byte).
- ~12–18 new scenarios: the curated version-ordering table; provides
  satisfaction (real preferred; provider chosen when no real; versioned
  provides if shipped); virtual-with-multiple-providers determinism;
  alternatives first-preference; alternatives backtrack-to-second on a
  downstream conflict; layered + provides; layered + alternatives; a hold
  that is a PROVIDER (freeze reasoning through provides); epoch/tilde
  ceilings in `explain_blocked`.
- `gen_catalogs.mjs` extended (new seed fine): emit deb versions,
  provides, alternatives. Term differential **≥2,400 / 0 divergences**
  across all queries; store-backed differential **≥500 / 0** (rebuild
  stores through the extended schema).
- wamjs corpus extended and green (rebuild `wamjs/` and `wamjs_store/`
  via their build.sh — deb-version strings will stress the D34/D37 string
  machinery; a runtime bug forced = probe-pinned fix, D44 precedent).
- Shared-lane sanity re-run: 4 JS suites, `CONFORMANCE_TARGETS=javascript`,
  cli_args 17/17 + 5067/0/0, cache-equiv + D48 bytes-read proof.

### Guardrails
- `resolver.pl` edits ADDITIVE ONLY: new clauses/predicates for deb
  versions, provides, alternatives; existing v/3 behavior byte-identical
  (the unchanged corpus + differential are the proof). `test_resolver.pl`
  append-only. New files under `examples/pkg_resolver/debian/` preferred
  for the parser + slice.
- May edit: `examples/pkg_resolver/` EXCEPT `cli/`, `cljs/`, `rust/`,
  `go/`, `BENCHMARKS.md` (frozen — the coordinator rebuilds those lanes
  against the new resolver.pl at merge); `scripts/js_wam/*` additive;
  `javascript_wam` runtime/templates only if a real bug forces it
  (probe-pinned). Everything else in the repo: frozen, as always
  (`examples/cli_args/`, `wam_target.pl`, `wam_text_parser.pl`, all other
  targets, harness).
- No repo package.json deps; dpkg arm gated like lmdb.
- Wrong output is worse than refusal; the differentials are the gate.

### Acceptance (must pass before handoff)
1. Existing corpus + differentials unchanged and green (state counts).
2. New scenarios green (list them); curated version table green.
3. dpkg cross-check: ran/didn't, and its numbers if it ran.
4. Real-slice ingestion: provenance, sizes, the three demonstration
   queries with real output.
5. Term differential ≥2,400/0 and store differential ≥500/0 on the
   extended generator; wamjs corpus green; shared-lane list green.

### Handoff format
Return: the deb-version encoding + comparison design, the provides and
alternatives model + determinism policy, the extended JSONL schema, the
ingestion parser choice + field coverage + slice provenance, corpus and
differential numbers, the real-slice demonstration output, any runtime
fixes (probe-pinned), whether the dpkg arm ran, and residuals (Breaks
vs Conflicts fidelity, Pre-Depends ordering, Recommends/Suggests,
multiarch, epoch display formatting, write paths).

## ↑↑↑ Copy to here
